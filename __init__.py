import lichtfeld as lf

from .main_panel import ObjectConstraintPanel
from .pruner import install_pruner, uninstall_pruner
from .settings import GuardSettings, initialize_runtime_settings

_classes = [ObjectConstraintPanel]


def on_load():
    initialize_runtime_settings(GuardSettings.get_instance())
    for cls in _classes:
        lf.register_class(cls)
    install_pruner()
    lf.log.info("Lumi_GS_pruning loaded")


def on_unload():
    uninstall_pruner()
    for cls in reversed(_classes):
        try:
            lf.unregister_class(cls)
        except Exception as exc:
            lf.log.warn(f"Lumi_GS_pruning: failed to unregister {cls}: {exc}")
    lf.log.info("Lumi_GS_pruning unloaded")
