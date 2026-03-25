import lichtfeld as lf

from .pruner import get_pruner
from .settings import (
    ACTION_ITEMS,
    CENTER_MODE_ITEMS,
    GuardSettings,
    SCALE_SCOPE_ITEMS,
    add_phase,
    build_three_phase_30k_scaffold,
    clear_phases,
    copy_global_rules_to_phase,
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

    def _draw_rule_block(self, layout, settings, rule_prefix: str, title: str, threshold_start_attr: str, threshold_end_attr: str, start_label: str, end_label: str, fmt: str, default_open: bool = False):
        if not layout.collapsing_header(title, default_open=default_open):
            return
        self._draw_bool(layout, settings, f"enable_{rule_prefix}", f"Enable {title}")
        self._draw_combo(layout, settings, f"{rule_prefix}_action", f"Action##{rule_prefix}", ACTION_ITEMS)
        self._draw_combo(layout, settings, f"{rule_prefix}_scale_scope", f"Scale scope##{rule_prefix}", SCALE_SCOPE_ITEMS)
        self._draw_input_float(layout, settings, threshold_start_attr, start_label, step=0.0, step_fast=0.0, fmt=fmt)
        self._draw_input_float(layout, settings, threshold_end_attr, end_label, step=0.0, step_fast=0.0, fmt=fmt)
        self._draw_input_float(layout, settings, f"{rule_prefix}_opacity_start", f"Opacity start##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_opacity_end", f"Opacity end##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_scale_multiplier_start", f"Scale multiplier start##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_scale_multiplier_end", f"Scale multiplier end##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_clamp_start", f"Clamp scale start##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")
        self._draw_input_float(layout, settings, f"{rule_prefix}_clamp_end", f"Clamp scale end##{rule_prefix}", step=0.0, step_fast=0.0, fmt="%.6f")

    def _draw_global_rules(self, layout, settings):
        if not layout.collapsing_header("Global rule settings", default_open=True):
            return
        self._draw_rule_block(
            layout,
            settings,
            "radius",
            "Match outside radius",
            "radius_start",
            "radius_end",
            "Radius at start",
            "Radius at end",
            "%.6f",
            default_open=False,
        )
        self._draw_rule_block(
            layout,
            settings,
            "max_axis",
            "Match oversized Gaussians",
            "max_axis_start",
            "max_axis_end",
            "Max axis at start",
            "Max axis at end",
            "%.6f",
            default_open=False,
        )
        self._draw_rule_block(
            layout,
            settings,
            "aspect",
            "Match stretched Gaussians",
            "max_aspect_start",
            "max_aspect_end",
            "Max aspect at start",
            "Max aspect at end",
            "%.6f",
            default_open=True,
        )

    def _draw_phase_controls(self, layout, settings):
        if not layout.collapsing_header("Phase workflow", default_open=False):
            return

        self._draw_bool(layout, settings, "use_phases", "Use phases")

        if layout.button("Add phase from current globals", (-1, 28)):
            start_iter = 0
            end_iter = 5000
            if len(settings.phases) > 0:
                last_phase = settings.phases[len(settings.phases) - 1]
                start_iter = int(last_phase.end_iter) + 1
                end_iter = start_iter + 5000
            add_phase(settings, start_iter=start_iter, end_iter=end_iter)
            self._save(settings)

        if layout.button("Build 30k three-phase scaffold", (-1, 28)):
            build_three_phase_30k_scaffold(settings)
            self._save(settings)

        if layout.button("Clear all phases", (-1, 28)):
            clear_phases(settings)
            self._save(settings)

        phase_count = len(settings.phases)
        layout.label(f"Configured phases: {phase_count}")

        if phase_count == 0:
            layout.text_colored("No phases configured. Global settings stay active until you add a phase.", (0.9, 0.85, 0.65, 1.0))
            return

        if not bool(settings.use_phases):
            layout.text_colored("Phases are configured but currently bypassed because Use phases is off.", (0.9, 0.85, 0.65, 1.0))

        for idx in range(phase_count):
            phase = settings.phases[idx]
            phase_name = str(phase.name).strip() or f"Phase {idx + 1}"
            header = f"Phase {idx + 1}: {phase_name} [{int(phase.start_iter)} - {int(phase.end_iter)}]"
            if not layout.collapsing_header(header, default_open=False):
                continue

            self._draw_bool(layout, phase, "enabled", f"Enabled##phase_enabled_{idx}")
            self._draw_input_text(layout, phase, "name", f"Name##phase_name_{idx}")
            self._draw_input_int(layout, phase, "start_iter", f"Start iter##phase_start_{idx}", step=1, step_fast=100)
            self._draw_input_int(layout, phase, "end_iter", f"End iter##phase_end_{idx}", step=1, step_fast=100)

            if layout.button(f"Copy current globals into this phase##copy_phase_{idx}", (-1, 26)):
                copy_global_rules_to_phase(settings, phase)
                self._save(settings)

            if idx > 0 and layout.button(f"Move phase up##phase_up_{idx}", (-1, 24)):
                settings.phases.move(idx, idx - 1)
                self._save(settings)
            if idx < len(settings.phases) - 1 and layout.button(f"Move phase down##phase_down_{idx}", (-1, 24)):
                settings.phases.move(idx, idx + 1)
                self._save(settings)
            if layout.button(f"Remove phase##phase_remove_{idx}", (-1, 24)):
                settings.phases.remove(idx)
                self._save(settings)
                break

            self._draw_rule_block(
                layout,
                phase,
                "radius",
                "Radius rule",
                "radius_start",
                "radius_end",
                "Radius at phase start",
                "Radius at phase end",
                "%.6f",
                default_open=False,
            )
            self._draw_rule_block(
                layout,
                phase,
                "max_axis",
                "Oversized rule",
                "max_axis_start",
                "max_axis_end",
                "Max axis at phase start",
                "Max axis at phase end",
                "%.6f",
                default_open=False,
            )
            self._draw_rule_block(
                layout,
                phase,
                "aspect",
                "Stretched rule",
                "max_aspect_start",
                "max_aspect_end",
                "Max aspect at phase start",
                "Max aspect at phase end",
                "%.6f",
                default_open=False,
            )

    def _draw_runtime(self, layout, guard):
        if not layout.collapsing_header("Runtime and log", default_open=True):
            return
        thresholds = guard.current_thresholds()
        layout.label(f"Active profile: {guard.active_profile_label}")
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
            f"Current params -> radius[action={guard.last_rule_values['radius']['action']}, opacity={guard.last_rule_values['radius']['opacity']:.4f}, mult={guard.last_rule_values['radius']['scale_multiplier']:.4f}, clamp={guard.last_rule_values['radius']['clamp_value']:.4f}]",
            (0.8, 0.8, 0.95, 1.0),
        )
        layout.text_colored(
            f"Current params -> max_axis[action={guard.last_rule_values['max_axis']['action']}, opacity={guard.last_rule_values['max_axis']['opacity']:.4f}, mult={guard.last_rule_values['max_axis']['scale_multiplier']:.4f}, clamp={guard.last_rule_values['max_axis']['clamp_value']:.4f}]",
            (0.8, 0.8, 0.95, 1.0),
        )
        layout.text_colored(
            f"Current params -> aspect[action={guard.last_rule_values['aspect']['action']}, opacity={guard.last_rule_values['aspect']['opacity']:.4f}, mult={guard.last_rule_values['aspect']['scale_multiplier']:.4f}, clamp={guard.last_rule_values['aspect']['clamp_value']:.4f}]",
            (0.8, 0.8, 0.95, 1.0),
        )
        layout.text_colored(
            f"Last counts -> radius={guard.last_counts['radius']}, max_axis={guard.last_counts['max_axis']}, aspect={guard.last_counts['aspect']}",
            (0.8, 0.8, 0.8, 1.0),
        )
        actions_text = ", ".join(guard.last_actions) if guard.last_actions else "<none>"
        layout.text_colored(f"Last actions: {actions_text}", (0.8, 0.9, 0.8, 1.0))
        layout.text_colored(f"Status: {guard.status_message}", (0.7, 0.9, 1.0, 1.0))

        if layout.collapsing_header("Recent runtime log", default_open=True):
            for line in guard.get_recent_status_lines(limit=10):
                layout.text_colored(line, (0.85, 0.85, 0.85, 1.0))
            if layout.button("Clear runtime log", (-1, 24)):
                guard.clear_status_log()
                self._redraw()

    def _draw_manual_controls(self, layout, guard):
        if not layout.collapsing_header("Manual controls", default_open=True):
            return
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
        if not layout.collapsing_header("Persistence", default_open=False):
            return
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

        self._draw_bool(layout, settings, "enabled", "Enabled")
        self._draw_bool(layout, settings, "log_each_hit", "Log updates")
        self._draw_input_int(layout, settings, "warmup_iters", "Warmup iterations", step=1, step_fast=100)
        self._draw_input_int(layout, settings, "apply_every", "Apply every N steps", step=1, step_fast=10)

        if layout.collapsing_header("Center controls", default_open=False):
            self._draw_combo(layout, settings, "center_mode", "Center mode", CENTER_MODE_ITEMS)
            self._draw_center_inputs(layout, settings)

        self._draw_global_rules(layout, settings)
        self._draw_phase_controls(layout, settings)

        layout.separator()
        if guard is None:
            layout.text_colored("Pruner not initialized.", (1.0, 0.5, 0.5, 1.0))
            return

        self._draw_runtime(layout, guard)
        self._draw_manual_controls(layout, guard)
        self._draw_settings_io(layout, settings)
