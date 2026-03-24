import lichtfeld as lf
from lfs_plugins.ui.state import AppState

from ..pruner import get_pruner
from ..settings import (
    ACTION_ITEMS,
    CENTER_MODE_ITEMS,
    GuardSettings,
    SCALE_SCOPE_ITEMS,
    load_persistent_settings,
    save_persistent_settings,
)


class ObjectConstraintPanel(lf.ui.Panel):
    id = "Lumi_GS_pruning.panel"
    label = "Lumi GS pruning"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 220
    update_interval_ms = 250
    poll_dependencies = {
        lf.ui.PollDependency.SCENE,
        lf.ui.PollDependency.TRAINING,
    }

    @classmethod
    def poll(cls, context) -> bool:
        del context
        return True

    def _redraw(self):
        try:
            lf.ui.request_redraw()
        except Exception:
            pass

    def on_update(self):
        self._redraw()

    def _save(self, settings):
        save_persistent_settings(settings)
        self._redraw()

    def _draw_bool(self, layout, settings, attr: str, label: str):
        changed, value = layout.checkbox(label, bool(getattr(settings, attr)))
        if changed:
            setattr(settings, attr, bool(value))
            self._save(settings)

    def _draw_input_int(self, layout, settings, attr: str, label: str, step: int = 1, step_fast: int = 100):
        changed, value = layout.input_int(label, int(getattr(settings, attr)), step=step, step_fast=step_fast)
        if changed:
            setattr(settings, attr, int(value))
            self._save(settings)

    def _draw_input_float(self, layout, settings, attr: str, label: str, step: float = 0.0, step_fast: float = 0.0, fmt: str = "%.4f"):
        changed, value = layout.input_float(label, float(getattr(settings, attr)), step=step, step_fast=step_fast, format=fmt)
        if changed:
            setattr(settings, attr, float(value))
            self._save(settings)

    def _draw_combo(self, layout, settings, attr: str, label: str, items):
        ids = [item[0] for item in items]
        labels = [item[1] for item in items]
        current_value = str(getattr(settings, attr))
        try:
            current_idx = ids.index(current_value)
        except ValueError:
            current_idx = 0
        changed, new_idx = layout.combo(label, current_idx, labels)
        if changed:
            setattr(settings, attr, ids[int(new_idx)])
            self._save(settings)

    def _draw_center_inputs(self, layout, settings):
        layout.label("Center XYZ")
        center = list(settings.center)
        changed_any = False
        for idx, axis in enumerate(("X", "Y", "Z")):
            changed, value = layout.input_float(f"{axis}##center_{axis.lower()}", float(center[idx]), step=0.0, step_fast=0.0, format="%.6f")
            if changed:
                center[idx] = float(value)
                changed_any = True
        if changed_any:
            settings.center = tuple(center)
            self._save(settings)

    def _draw_rule_block(self, layout, settings, title: str, enable_attr: str, start_attr: str, end_attr: str, action_attr: str, scope_attr: str, start_label: str, end_label: str, fmt: str):
        layout.separator()
        self._draw_bool(layout, settings, enable_attr, title)
        self._draw_combo(layout, settings, action_attr, f"Action##{action_attr}", ACTION_ITEMS)
        self._draw_combo(layout, settings, scope_attr, f"Scale scope##{scope_attr}", SCALE_SCOPE_ITEMS)
        self._draw_input_float(layout, settings, start_attr, start_label, step=0.0, step_fast=0.0, fmt=fmt)
        self._draw_input_float(layout, settings, end_attr, end_label, step=0.0, step_fast=0.0, fmt=fmt)

    def draw(self, layout):
        settings = GuardSettings.get_instance()
        guard = get_pruner()
        scale = layout.get_dpi_scale()

        self._draw_bool(layout, settings, "enabled", "Enabled")
        self._draw_bool(layout, settings, "log_each_hit", "Log updates")
        self._draw_input_int(layout, settings, "warmup_iters", "Warmup iterations", step=1, step_fast=100)
        self._draw_input_int(layout, settings, "apply_every", "Apply every N steps", step=1, step_fast=10)

        layout.separator()
        self._draw_combo(layout, settings, "center_mode", "Center mode", CENTER_MODE_ITEMS)
        self._draw_center_inputs(layout, settings)

        self._draw_rule_block(
            layout,
            settings,
            "Match outside radius",
            "enable_radius",
            "radius_start",
            "radius_end",
            "radius_action",
            "radius_scale_scope",
            "Radius at start",
            "Radius at end",
            "%.6f",
        )

        self._draw_rule_block(
            layout,
            settings,
            "Match oversized Gaussians",
            "enable_max_axis",
            "max_axis_start",
            "max_axis_end",
            "max_axis_action",
            "max_axis_scale_scope",
            "Max axis at start",
            "Max axis at end",
            "%.6f",
        )

        self._draw_rule_block(
            layout,
            settings,
            "Match stretched Gaussians",
            "enable_aspect",
            "max_aspect_start",
            "max_aspect_end",
            "aspect_action",
            "aspect_scale_scope",
            "Max aspect at start",
            "Max aspect at end",
            "%.6f",
        )

        layout.separator()
        layout.label("Shared action parameters")
        self._draw_input_float(layout, settings, "opacity_target", "Opacity target", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, "scale_multiplier", "Scale multiplier", step=0.0, step_fast=0.0, fmt="%.6f")

        layout.separator()
        if guard is None:
            layout.text_colored("Pruner not initialized.", (1.0, 0.5, 0.5, 1.0))
            return

        thresholds = guard.current_thresholds()
        layout.label(f"Last affected: {guard.last_removed:,}")
        layout.label(f"Total affected: {guard.total_removed:,}")
        layout.label(f"Last seen iteration: {guard.last_seen_iteration}")
        layout.label(f"Last applied iteration: {guard.last_pruned_iteration}")
        layout.label(f"Captured center: {guard._fmt_center(guard.center_xyz)}")
        layout.text_colored(
            f"Current thresholds -> radius={thresholds['radius']:.6f}, max_axis={thresholds['max_axis']:.6f}, aspect={thresholds['aspect']:.6f}",
            (0.8, 0.9, 0.8, 1.0),
        )
        layout.text_colored(
            f"Last counts -> radius={guard.last_counts['radius']}, max_axis={guard.last_counts['max_axis']}, aspect={guard.last_counts['aspect']}",
            (0.8, 0.8, 0.8, 1.0),
        )
        actions_text = ", ".join(guard.last_actions) if guard.last_actions else "<none>"
        layout.text_colored(f"Last actions: {actions_text}", (0.8, 0.9, 0.8, 1.0))
        layout.text_colored(f"Status: {guard.status_message}", (0.7, 0.9, 1.0, 1.0))

        layout.separator()
        if layout.button("Capture center now", (-1, 30 * scale)):
            guard.capture_center_from_scene(force=True)
            self._redraw()

        if layout.button("Run suppression once now", (-1, 30 * scale)):
            guard.request_manual_prune()
            self._redraw()

        if layout.button("Clear soft-delete mask", (-1, 30 * scale)):
            guard.clear_old_mask_now()
            self._redraw()

        layout.separator()
        if layout.button("Reload saved settings", (-1, 28 * scale)):
            load_persistent_settings(settings)
            self._redraw()

        if layout.button("Save settings now", (-1, 28 * scale)):
            save_persistent_settings(settings)
            self._redraw()
