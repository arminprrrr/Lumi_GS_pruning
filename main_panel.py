import lichtfeld as lf

from .pruner import get_pruner
from .settings import ACTION_ITEMS, CENTER_MODE_ITEMS, GuardSettings, SCALE_SCOPE_ITEMS, load_persistent_settings, save_persistent_settings


class ObjectConstraintPanel(lf.ui.Panel):
    id = "Lumi_GS_pruning.panel"
    label = "Lumi GS pruning"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 220
    update_interval_ms = 250
    poll_dependencies = {lf.ui.PollDependency.SCENE, lf.ui.PollDependency.TRAINING}

    @classmethod
    def poll(cls, context) -> bool:
        del context
        return True

    def _redraw(self):
        try:
            lf.ui.request_redraw()
        except Exception:
            pass

    def on_update(self, *args):
        del args
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

    def _draw_input_text(self, layout, settings, attr: str, label: str):
        changed, value = layout.input_text(label, str(getattr(settings, attr)))
        if changed:
            setattr(settings, attr, str(value))
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

    def _draw_rule_block(self, layout, settings, rule_prefix: str, title: str, threshold_start_attr: str, threshold_end_attr: str, start_label: str, end_label: str, fmt: str):
        layout.separator()
        layout.label(title)
        self._draw_bool(layout, settings, f"enable_{rule_prefix}", f"Enable {title}")
        self._draw_combo(layout, settings, f"{rule_prefix}_action", f"Action##{rule_prefix}", ACTION_ITEMS)
        self._draw_combo(layout, settings, f"{rule_prefix}_scale_scope", f"Scale scope##{rule_prefix}", SCALE_SCOPE_ITEMS)
        self._draw_input_float(layout, settings, threshold_start_attr, start_label, step=0.0, step_fast=0.0, fmt=fmt)
        self._draw_input_float(layout, settings, threshold_end_attr, end_label, step=0.0, step_fast=0.0, fmt=fmt)
        self._draw_input_float(layout, settings, f"{rule_prefix}_opacity_start", f"Opacity multiplier start##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_opacity_end", f"Opacity multiplier end##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_scale_multiplier_start", f"Scale multiplier start##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_scale_multiplier_end", f"Scale multiplier end##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")

    def _draw_runtime(self, layout, guard):
        thresholds = guard.current_thresholds()
        layout.separator()
        layout.label("Runtime and log")
        layout.label(f"Active profile: {guard.active_profile_label}")
        layout.label(f"Last affected: {guard.last_removed:,}")
        layout.label(f"Total affected: {guard.total_removed:,}")
        layout.label(f"Last seen iteration: {guard.last_seen_iteration}")
        layout.label(f"Last applied iteration: {guard.last_pruned_iteration}")
        layout.label(f"Captured center: {guard._fmt_center(guard.center_xyz)}")
        layout.text_colored(f"Current thresholds -> radius={thresholds['radius']:.6f}, max_axis={thresholds['max_axis']:.6f}, aspect={thresholds['aspect']:.6f}", (0.8, 0.9, 0.8, 1.0))
        layout.text_colored(f"Current params -> radius[action={guard.last_rule_values['radius']['action']}, opacity_mult={guard.last_rule_values['radius']['opacity_multiplier']:.4f}, scale_mult={guard.last_rule_values['radius']['scale_multiplier']:.4f}]", (0.8, 0.8, 0.95, 1.0))
        layout.text_colored(f"Current params -> max_axis[action={guard.last_rule_values['max_axis']['action']}, opacity_mult={guard.last_rule_values['max_axis']['opacity_multiplier']:.4f}, scale_mult={guard.last_rule_values['max_axis']['scale_multiplier']:.4f}]", (0.8, 0.8, 0.95, 1.0))
        layout.text_colored(f"Current params -> aspect[action={guard.last_rule_values['aspect']['action']}, opacity_mult={guard.last_rule_values['aspect']['opacity_multiplier']:.4f}, scale_mult={guard.last_rule_values['aspect']['scale_multiplier']:.4f}]", (0.8, 0.8, 0.95, 1.0))
        layout.text_colored(f"Last counts -> radius={guard.last_counts['radius']}, max_axis={guard.last_counts['max_axis']}, aspect={guard.last_counts['aspect']}", (0.8, 0.8, 0.8, 1.0))
        actions_text = ", ".join(guard.last_actions) if guard.last_actions else "<none>"
        layout.text_colored(f"Last actions: {actions_text}", (0.8, 0.9, 0.8, 1.0))
        layout.text_colored(f"Status: {guard.status_message}", (0.7, 0.9, 1.0, 1.0))
        layout.label("Recent runtime log")
        for line in guard.get_recent_status_lines(limit=10):
            layout.text_colored(line, (0.85, 0.85, 0.85, 1.0))
        if layout.button("Clear runtime log", (-1, 24)):
            guard.clear_status_log()
            self._redraw()

    def _draw_manual_controls(self, layout, guard):
        layout.separator()
        layout.label("Manual controls")
        scale = layout.get_dpi_scale()
        if layout.button("Capture center now", (-1, 30 * scale)):
            guard.capture_center_from_scene(force=True)
            self._redraw()
        if layout.button("Run suppression once now", (-1, 30 * scale)):
            guard.request_manual_prune()
            self._redraw()
        if layout.button("Clear soft-delete mask", (-1, 30 * scale)):
            guard.clear_old_mask_now()
            self._redraw()

    def _draw_settings_io(self, layout, settings):
        layout.separator()
        layout.label("Persistence")
        scale = layout.get_dpi_scale()
        if layout.button("Reload saved settings", (-1, 28 * scale)):
            load_persistent_settings(settings)
            self._redraw()
        if layout.button("Save settings now", (-1, 28 * scale)):
            save_persistent_settings(settings)
            self._redraw()

    def draw(self, layout):
        settings = GuardSettings.get_instance()
        guard = get_pruner()
        layout.label("General")
        self._draw_bool(layout, settings, "enabled", "Enabled")
        self._draw_bool(layout, settings, "log_each_hit", "Log updates")
        self._draw_input_text(layout, settings, "warmup_iters", "Warmup iterations or %")
        self._draw_input_text(layout, settings, "stop_iters", "Stop iterations or %")
        self._draw_input_int(layout, settings, "apply_every", "Apply every N steps", step=1, step_fast=10)
        layout.separator()
        layout.label("Center controls")
        self._draw_combo(layout, settings, "center_mode", "Center mode", CENTER_MODE_ITEMS)
        self._draw_center_inputs(layout, settings)
        layout.separator()
        layout.label("Rule settings")
        self._draw_rule_block(layout, settings, "radius", "Match outside radius", "radius_start", "radius_end", "Radius start", "Radius end", "%.6f")
        self._draw_rule_block(layout, settings, "max_axis", "Match oversized Gaussians", "max_axis_start", "max_axis_end", "Max axis start", "Max axis end", "%.6f")
        self._draw_rule_block(layout, settings, "aspect", "Match stretched Gaussians", "max_aspect_start", "max_aspect_end", "Max aspect start", "Max aspect end", "%.6f")
        if guard is None:
            layout.separator()
            layout.text_colored("Pruner not initialized.", (1.0, 0.5, 0.5, 1.0))
            return
        self._draw_runtime(layout, guard)
        self._draw_manual_controls(layout, guard)
        self._draw_settings_io(layout, settings)
