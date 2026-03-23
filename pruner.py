import math

import lichtfeld as lf
from lfs_plugins.ui.state import AppState

from .settings import GuardSettings

_EPS = 1e-8
_PLUGIN_OWNER = "object_constraint_guard"
_PRUNER = None


def get_pruner():
    return _PRUNER


def install_pruner():
    global _PRUNER
    _PRUNER = ObjectConstraintPruner()

    lf.on_training_start(_PRUNER.on_training_start)
    lf.on_training_end(_PRUNER.on_training_end)



def uninstall_pruner():
    global _PRUNER
    _PRUNER = None


class ObjectConstraintPruner:
    def __init__(self):
        self.settings = GuardSettings.get_instance()
        self.center_xyz = None
        self.last_removed = 0
        self.total_removed = 0
        self.status_message = "Idle."
        self.pending_manual_prune = False
        self.last_seen_iteration = -1
        self.last_pruned_iteration = -1
        self.last_counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        self.last_actions = []

        AppState.iteration.subscribe_as(_PLUGIN_OWNER, self._on_iteration_signal)
        AppState.trainer_state.subscribe_as(_PLUGIN_OWNER, self._on_trainer_state_signal)
        AppState.has_trainer.subscribe_as(_PLUGIN_OWNER, self._on_has_trainer_signal)

    def _request_redraw(self):
        try:
            lf.ui.request_redraw()
        except Exception:
            pass

    def _set_status(self, message: str, level: str = "info"):
        self.status_message = message

        if bool(self.settings.log_each_hit) or level in ("warn", "error"):
            if level == "warn":
                lf.log.warn(f"[{_PLUGIN_OWNER}] {message}")
            elif level == "error":
                lf.log.error(f"[{_PLUGIN_OWNER}] {message}")
            else:
                lf.log.info(f"[{_PLUGIN_OWNER}] {message}")

    def _fmt_center(self, center_xyz):
        if center_xyz is None:
            return "<unset>"
        x, y, z = center_xyz
        return f"({x:.4f}, {y:.4f}, {z:.4f})"

    def _reduce_values(self, reduced):
        if hasattr(reduced, "values"):
            return reduced.values
        if isinstance(reduced, tuple):
            return reduced[0]
        return reduced

    def _assign_tensor(self, tensor_view, updated):
        if tensor_view.ndim == 1:
            tensor_view[:] = updated
        elif tensor_view.ndim == 2:
            tensor_view[:, :] = updated
        else:
            tensor_view[...] = updated
        return True

    def _clear_deleted_mask(self, model):
        if not hasattr(model, "clear_deleted"):
            return True
        try:
            model.clear_deleted()
            return True
        except Exception:
            return False

    def _to_logit(self, p: float) -> float:
        p = max(1e-6, min(1.0 - 1e-6, float(p)))
        return math.log(p / (1.0 - p))

    def _set_opacity_target(self, model, kill_mask, target_opacity: float):
        raw_target = self._to_logit(target_opacity)

        try:
            raw = model.opacity_raw
            mask = kill_mask.unsqueeze(1)
            updated = raw.clone().masked_fill(mask, raw_target)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Opacity update failed: {exc}", level="warn")
            return False

    def _multiply_scale_on_mask(self, model, kill_mask, multiplier: float):
        multiplier = max(1e-6, min(1.0, float(multiplier)))
        delta_raw = math.log(multiplier)

        try:
            raw = model.scaling_raw
            mask_f = kill_mask.unsqueeze(1).to(raw.dtype)
            updated = raw + (mask_f * delta_raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Scale update failed: {exc}", level="warn")
            return False

    def _on_has_trainer_signal(self, has_trainer: bool):
        if not has_trainer:
            self._set_status("No trainer available.", level="warn")
            self._request_redraw()

    def _on_trainer_state_signal(self, state: str):
        self._set_status(f"Trainer state: {state}")
        self._request_redraw()

    def _on_iteration_signal(self, iteration: int):
        try:
            iteration = int(iteration)
            self.last_seen_iteration = iteration

            if not AppState.is_training.value:
                self._set_status(f"Trainer state: {AppState.trainer_state.value}")
                self._request_redraw()
                return

            if self.pending_manual_prune:
                self.pending_manual_prune = False
                self.prune_once(manual_trigger=True, forced_iteration=iteration)
                self._request_redraw()
                return

            self.prune_once(manual_trigger=False, forced_iteration=iteration)
            self._request_redraw()

        except Exception as exc:
            self._set_status(f"Iteration signal failed: {exc}", level="error")
            self._request_redraw()

    def on_training_start(self):
        self.last_removed = 0
        self.total_removed = 0
        self.last_seen_iteration = -1
        self.last_pruned_iteration = -1
        self.last_counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        self.last_actions = []
        self.pending_manual_prune = False

        scene = lf.get_scene()
        model = scene.combined_model() if scene is not None else None
        if model is not None:
            self._clear_deleted_mask(model)
            if scene is not None:
                try:
                    scene.notify_changed()
                except Exception:
                    pass

        if self.settings.center_mode == "manual":
            self.center_xyz = tuple(float(v) for v in self.settings.center)
            self._set_status(
                f"Training started. Using manual center {self._fmt_center(self.center_xyz)}."
            )
        else:
            self.center_xyz = None
            captured = self.capture_center_from_scene(force=True)
            if captured is None:
                self._set_status(
                    "Training started. Waiting for Gaussians before center capture.",
                    level="warn",
                )

        self._request_redraw()

    def on_training_end(self):
        self.pending_manual_prune = False
        self._set_status("Training ended.")
        self._request_redraw()

    def capture_center_from_scene(self, force: bool = False):
        if self.settings.center_mode == "manual" and not force:
            self.center_xyz = tuple(float(v) for v in self.settings.center)
            self._set_status(f"Using manual center {self._fmt_center(self.center_xyz)}.")
            return self.center_xyz

        scene = lf.get_scene()
        if scene is None:
            self._set_status("No scene loaded. Open a scene first.", level="warn")
            return None

        model = scene.combined_model()
        if model is None:
            self._set_status("No Gaussian model available yet.", level="warn")
            return None

        if int(model.num_points) == 0:
            self._set_status("Gaussian model has zero points.", level="warn")
            return None

        means = model.means_raw
        center = means.mean(dim=0)
        self.center_xyz = (
            float(center[0].item()),
            float(center[1].item()),
            float(center[2].item()),
        )

        self._set_status(f"Captured center {self._fmt_center(self.center_xyz)}.")
        return self.center_xyz

    def request_manual_prune(self):
        if AppState.is_training.value:
            self.pending_manual_prune = True
            self._set_status("Manual suppression queued for next iteration.")
            self._request_redraw()
            return 0

        out = self.prune_once(
            manual_trigger=True,
            forced_iteration=int(AppState.iteration.value),
        )
        self._request_redraw()
        return out

    def clear_old_mask_now(self):
        scene = lf.get_scene()
        if scene is None:
            self._set_status("No scene loaded. Open a scene first.", level="warn")
            self._request_redraw()
            return False

        model = scene.combined_model()
        if model is None:
            self._set_status("No Gaussian model available yet.", level="warn")
            self._request_redraw()
            return False

        ok = self._clear_deleted_mask(model)
        if ok:
            try:
                scene.notify_changed()
            except Exception:
                pass
            self._set_status("Cleared old deleted mask on combined_model.")
        else:
            self._set_status("Failed to clear old deleted mask on combined_model.", level="warn")

        self._request_redraw()
        return ok

    def prune_once(self, manual_trigger: bool = False, forced_iteration: int | None = None):
        s = self.settings
        self.last_removed = 0
        self.last_actions = []

        iteration = int(forced_iteration if forced_iteration is not None else AppState.iteration.value)

        if not bool(s.enabled) and not manual_trigger:
            self._set_status("Plugin is disabled.")
            return 0

        if not manual_trigger:
            if not AppState.is_training.value:
                self._set_status(f"Trainer state: {AppState.trainer_state.value}")
                return 0

            if iteration < int(s.warmup_iters):
                self._set_status(
                    f"Warmup active: iteration {iteration} < {int(s.warmup_iters)}."
                )
                return 0

            if iteration == self.last_pruned_iteration:
                return 0

            if iteration % max(1, int(s.apply_every)) != 0:
                self._set_status(
                    f"Skipping iteration {iteration} (apply_every={int(s.apply_every)})."
                )
                return 0

        scene = lf.get_scene()
        if scene is None:
            self._set_status("No scene loaded. Open a scene first.", level="warn")
            return 0

        model = scene.combined_model()
        if model is None:
            self._set_status("No Gaussian model available yet.", level="warn")
            return 0

        if int(model.num_points) == 0:
            self._set_status("Gaussian model has zero points.", level="warn")
            return 0

        if s.center_mode == "manual":
            self.center_xyz = tuple(float(v) for v in s.center)
        elif self.center_xyz is None:
            self.capture_center_from_scene(force=True)

        kill_mask, counts = self._build_kill_mask(model)
        self.last_counts = counts

        if kill_mask is None:
            self._set_status("No match criteria are enabled.")
            return 0

        affected = int(kill_mask.sum().item())
        if affected == 0:
            self.last_pruned_iteration = iteration
            self._set_status(
                f"iter={iteration}: matched 0 "
                f"(radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']})"
            )
            return 0

        actions = []
        try:
            if bool(s.fade_matched):
                if self._set_opacity_target(model, kill_mask, float(s.opacity_target)):
                    actions.append(f"opacity->{float(s.opacity_target):.3f}")

            if bool(s.shrink_matched):
                if self._multiply_scale_on_mask(model, kill_mask, float(s.scale_multiplier)):
                    actions.append(f"scale*={float(s.scale_multiplier):.3f}")

            try:
                scene.notify_changed()
            except Exception:
                pass

            try:
                ctx = lf.context()
                ctx.refresh()
            except Exception:
                pass

        except Exception as exc:
            self._set_status(f"Suppression failed: {exc}", level="error")
            return 0

        self.last_removed = affected
        self.total_removed += affected
        self.last_pruned_iteration = iteration
        self.last_actions = actions

        prefix = "manual" if manual_trigger else f"iter={iteration}"
        if not actions:
            self._set_status(
                f"{prefix}: matched={affected}, but no suppression action is enabled."
            )
            return affected

        self._set_status(
            f"{prefix}: affected={affected} "
            f"(radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']}); "
            f"actions={', '.join(actions)}"
        )
        return affected

    def _build_kill_mask(self, model):
        n = int(model.num_points)
        if n <= 0:
            return None, {"radius": 0, "max_axis": 0, "aspect": 0}

        any_rule = False
        kill = lf.Tensor.zeros([n], dtype="bool", device="cuda")
        counts = {"radius": 0, "max_axis": 0, "aspect": 0}

        if bool(self.settings.enable_max_axis) or bool(self.settings.enable_aspect):
            any_rule = True
            scaling = model.get_scaling()
            max_axis = self._reduce_values(scaling.max(dim=1))
            min_axis = self._reduce_values(scaling.min(dim=1))

            if bool(self.settings.enable_max_axis):
                big_mask = max_axis > float(self.settings.max_axis)
                counts["max_axis"] = int(big_mask.sum().item())
                kill = kill | big_mask

            if bool(self.settings.enable_aspect):
                aspect = max_axis / (min_axis + _EPS)
                stretch_mask = aspect > float(self.settings.max_aspect)
                counts["aspect"] = int(stretch_mask.sum().item())
                kill = kill | stretch_mask

        if bool(self.settings.enable_radius):
            any_rule = True
            if self.center_xyz is None:
                self._set_status(
                    "Radius matching is enabled, but center is not set yet.",
                    level="warn",
                )
            else:
                cx, cy, cz = self.center_xyz
                means = model.means_raw

                dx = means[:, 0] - cx
                dy = means[:, 1] - cy
                dz = means[:, 2] - cz

                dist2 = dx * dx + dy * dy + dz * dz
                radius2 = float(self.settings.radius * self.settings.radius)

                far_mask = dist2 > radius2
                counts["radius"] = int(far_mask.sum().item())
                kill = kill | far_mask

        if not any_rule:
            return None, counts

        return kill, counts
