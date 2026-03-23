import lichtfeld as lf
from lfs_plugins.ui.state import AppState

from ..settings import GuardSettings
from ..pruner import get_pruner


class ObjectConstraintPanel(lf.ui.Panel):
    id = "object_constraint_guard.panel"
    label = "Object Constraint Guard"
    parent = "lfs.training"
    order = 220
    update_interval_ms = 250
    poll_dependencies = {
        lf.ui.PollDependency.SCENE,
        lf.ui.PollDependency.TRAINING,
    }

    def __init__(self):
        self._warmup_step = 10
        self._radius_step = 0.1
        self._max_axis_step = 0.01
        self._max_aspect_step = 1.0
        self._center_step = 0.05
        self._opacity_step = 0.01
        self._scale_mult_step = 0.05

    @classmethod
    def poll(cls, context) -> bool:
        del context
        return AppState.has_scene.value or AppState.has_trainer.value

    def _redraw(self):
        try:
            lf.ui.request_redraw()
        except Exception:
            pass

    def on_update(self):
        self._redraw()

    def _clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def _nudge_int(self, settings, attr, delta, lo, hi):
        current = int(getattr(settings, attr))
        setattr(settings, attr, int(self._clamp(current + delta, lo, hi)))
        self._redraw()

    def _nudge_float(self, settings, attr, delta, lo, hi):
        current = float(getattr(settings, attr))
        setattr(settings, attr, float(self._clamp(current + delta, lo, hi)))
        self._redraw()

    def _nudge_center(self, settings, axis_index, delta):
        center = list(settings.center)
        center[axis_index] = float(self._clamp(center[axis_index] + delta, -100.0, 100.0))
        settings.center = tuple(center)
        self._redraw()

    def _draw_bool(self, layout, label, value):
        _, new_value = layout.checkbox(label, bool(value))
        return bool(new_value)

    def _draw_stepper_int(self, layout, label, value, minus_id, plus_id, step):
        scale = layout.get_dpi_scale()
        layout.label(f"{label}: {int(value)}")
        if layout.button(f"-##{minus_id}", (26 * scale, 0)):
            return -step
        layout.same_line()
        if layout.button(f"+##{plus_id}", (26 * scale, 0)):
            return step
        return 0

    def _draw_stepper_float(self, layout, label, value, minus_id, plus_id, step, digits=4):
        scale = layout.get_dpi_scale()
        layout.label(f"{label}: {value:.{digits}f}")
        if layout.button(f"-##{minus_id}", (26 * scale, 0)):
            return -step
        layout.same_line()
        if layout.button(f"+##{plus_id}", (26 * scale, 0)):
            return step
        return 0.0

    def draw(self, layout):
        settings = GuardSettings.get_instance()
        guard = get_pruner()
        scale = layout.get_dpi_scale()

        layout.label("Training-time gaussian suppression")
        layout.label(f"Trainer state: {AppState.trainer_state.value}")
        layout.label(f"AppState iteration: {int(AppState.iteration.value)}")
        layout.label(f"AppState is_training: {bool(AppState.is_training.value)}")

        if not AppState.has_scene.value:
            layout.text_colored("No scene loaded yet.", (1.0, 0.75, 0.4, 1.0))
        if not AppState.has_trainer.value:
            layout.text_colored(
                "No trainer active. Manual actions still need a scene/model.",
                (0.7, 0.7, 0.7, 1.0),
            )

        layout.text_colored(
            "This build uses the same AppState iteration signal path as your working pruning version.",
            (0.75, 0.90, 1.0, 1.0),
        )

        layout.separator()

        settings.enabled = self._draw_bool(layout, "Enabled", settings.enabled)
        settings.enable_radius = self._draw_bool(layout, "Match outside radius", settings.enable_radius)
        settings.enable_max_axis = self._draw_bool(layout, "Match oversized Gaussians", settings.enable_max_axis)
        settings.enable_aspect = self._draw_bool(layout, "Match stretched Gaussians", settings.enable_aspect)
        settings.fade_matched = self._draw_bool(layout, "Fade matched", settings.fade_matched)
        settings.shrink_matched = self._draw_bool(layout, "Shrink matched", settings.shrink_matched)
        settings.log_each_hit = self._draw_bool(layout, "Log updates", settings.log_each_hit)

        layout.separator()

        delta = self._draw_stepper_int(
            layout,
            "Warmup iterations",
            settings.warmup_iters,
            "warmup_minus",
            "warmup_plus",
            self._warmup_step,
        )
        if delta != 0:
            self._nudge_int(settings, "warmup_iters", delta, 0, 10000)

        delta = self._draw_stepper_int(
            layout,
            "Apply every N steps",
            settings.apply_every,
            "apply_minus",
            "apply_plus",
            1,
        )
        if delta != 0:
            self._nudge_int(settings, "apply_every", delta, 1, 100)

        layout.separator()

        layout.label(f"Center mode: {'Manual' if settings.center_mode == 'manual' else 'Auto once'}")

        if layout.button("Use auto center##center_auto", (-1, 28 * scale)):
            settings.center_mode = "auto_once"
            self._redraw()

        if layout.button("Use manual center##center_manual", (-1, 28 * scale)):
            settings.center_mode = "manual"
            self._redraw()

        if settings.center_mode == "manual":
            layout.separator()
            layout.label("Manual center XYZ")

            for axis_index, axis_name in enumerate(("X", "Y", "Z")):
                center_val = float(settings.center[axis_index])
                layout.label(f"{axis_name}: {center_val:.4f}")

                if layout.button(f"-##{axis_name}_minus", (26 * scale, 0)):
                    self._nudge_center(settings, axis_index, -self._center_step)
                layout.same_line()
                if layout.button(f"+##{axis_name}_plus", (26 * scale, 0)):
                    self._nudge_center(settings, axis_index, self._center_step)

        layout.separator()

        if settings.enable_radius:
            delta = self._draw_stepper_float(
                layout,
                "Radius",
                float(settings.radius),
                "radius_minus",
                "radius_plus",
                self._radius_step,
                digits=3,
            )
            if delta != 0.0:
                self._nudge_float(settings, "radius", delta, 0.0, 50.0)

        if settings.enable_max_axis:
            delta = self._draw_stepper_float(
                layout,
                "Max axis scale",
                float(settings.max_axis),
                "axis_minus",
                "axis_plus",
                self._max_axis_step,
                digits=4,
            )
            if delta != 0.0:
                self._nudge_float(settings, "max_axis", delta, 0.0, 10.0)

        if settings.enable_aspect:
            delta = self._draw_stepper_float(
                layout,
                "Max aspect ratio",
                float(settings.max_aspect),
                "aspect_minus",
                "aspect_plus",
                self._max_aspect_step,
                digits=2,
            )
            if delta != 0.0:
                self._nudge_float(settings, "max_aspect", delta, 1.0, 1000.0)

        layout.separator()

        if settings.fade_matched:
            delta = self._draw_stepper_float(
                layout,
                "Opacity target",
                float(settings.opacity_target),
                "opacity_minus",
                "opacity_plus",
                self._opacity_step,
                digits=3,
            )
            if delta != 0.0:
                self._nudge_float(settings, "opacity_target", delta, 0.001, 1.0)

        if settings.shrink_matched:
            delta = self._draw_stepper_float(
                layout,
                "Scale multiplier",
                float(settings.scale_multiplier),
                "scale_mult_minus",
                "scale_mult_plus",
                self._scale_mult_step,
                digits=3,
            )
            if delta != 0.0:
                self._nudge_float(settings, "scale_multiplier", delta, 0.01, 1.0)

        layout.separator()

        if guard is None:
            layout.text_colored("Pruner not initialized.", (1.0, 0.5, 0.5, 1.0))
            return

        layout.label(f"Last affected: {guard.last_removed:,}")
        layout.label(f"Total affected: {guard.total_removed:,}")
        layout.label(f"Last seen iteration signal: {guard.last_seen_iteration}")
        layout.label(f"Last applied iteration: {guard.last_pruned_iteration}")
        layout.label(f"Captured center: {guard._fmt_center(guard.center_xyz)}")
        layout.text_colored(
            f"Last counts: radius={guard.last_counts['radius']}, max_axis={guard.last_counts['max_axis']}, aspect={guard.last_counts['aspect']}",
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

        if layout.button("Clear old soft-delete mask", (-1, 30 * scale)):
            guard.clear_old_mask_now()
            self._redraw()

        layout.separator()

        if layout.collapsing_header("Quick presets", default_open=False):
            if layout.button("Preset: mild fade", (-1, 28 * scale)):
                settings.warmup_iters = 0
                settings.apply_every = 1
                settings.radius = 0.5
                settings.max_axis = 0.03
                settings.max_aspect = 10.0
                settings.enable_radius = True
                settings.enable_max_axis = True
                settings.enable_aspect = True
                settings.fade_matched = True
                settings.opacity_target = 0.10
                settings.shrink_matched = False
                self._redraw()

            if layout.button("Preset: fade + shrink", (-1, 28 * scale)):
                settings.warmup_iters = 0
                settings.apply_every = 1
                settings.radius = 0.5
                settings.max_axis = 0.03
                settings.max_aspect = 10.0
                settings.enable_radius = True
                settings.enable_max_axis = True
                settings.enable_aspect = True
                settings.fade_matched = True
                settings.opacity_target = 0.05
                settings.shrink_matched = True
                settings.scale_multiplier = 0.50
                self._redraw()
