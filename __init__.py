import lichtfeld as lf

from .pruner import install_pruner, uninstall_pruner, set_runtime_mode
from .settings import GuardSettings, initialize_runtime_settings, save_persistent_settings

_PLUGIN_NAME = "Lumi_GS_pruning"
_REGISTERED_CLASSES = []
_PRUNER_INSTALLED = False


def _is_headless_ui_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "retained panel features require the retained ui manager" in msg
        or "unavailable in this runtime" in msg
    )


def _get_panel_classes():
    try:
        from .main_panel import ObjectConstraintPanel
    except Exception as exc:
        lf.log.warn(f"[{_PLUGIN_NAME}] Panel import failed; continuing without UI: {exc}")
        return []
    return [ObjectConstraintPanel]


def _register_optional_ui():
    global _REGISTERED_CLASSES
    for cls in _get_panel_classes():
        try:
            lf.register_class(cls)
            _REGISTERED_CLASSES.append(cls)
        except Exception as exc:
            if _is_headless_ui_error(exc):
                lf.log.info(f"[{_PLUGIN_NAME}] UI panel registration skipped in headless/non-retained runtime.")
                continue
            raise


def on_load():
    global _PRUNER_INSTALLED
    initialize_runtime_settings(GuardSettings.get_instance())

    _register_optional_ui()
    mode = "ui" if _REGISTERED_CLASSES else "headless/no-ui"
    set_runtime_mode(mode)

    install_pruner(runtime_mode=mode)
    _PRUNER_INSTALLED = True
    lf.log.info(f"[{_PLUGIN_NAME}] Training hooks installed")
    lf.log.info(f"{_PLUGIN_NAME} loaded ({mode})")


def on_unload():
    global _PRUNER_INSTALLED, _REGISTERED_CLASSES

    try:
        save_persistent_settings(GuardSettings.get_instance())
    except Exception as exc:
        lf.log.warn(f"{_PLUGIN_NAME}: failed to save settings on unload: {exc}")

    if _PRUNER_INSTALLED:
        try:
            uninstall_pruner()
        except Exception as exc:
            lf.log.warn(f"{_PLUGIN_NAME}: failed to uninstall pruner: {exc}")
        _PRUNER_INSTALLED = False

    for cls in reversed(_REGISTERED_CLASSES):
        try:
            lf.unregister_class(cls)
        except Exception as exc:
            if not _is_headless_ui_error(exc):
                lf.log.warn(f"{_PLUGIN_NAME}: failed to unregister {cls}: {exc}")
    _REGISTERED_CLASSES = []

    lf.log.info(f"{_PLUGIN_NAME} unloaded")
