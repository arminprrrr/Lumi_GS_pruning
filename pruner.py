import math
from typing import Any

import lichtfeld as lf
from lfs_plugins.ui.state import AppState

from .settings import GuardSettings, save_persistent_settings

_EPS = 1e-8
_PLUGIN_OWNER = "Lumi_GS_pruning"
_PRUNER = None


def get_pruner():
    return _PRUNER


def install_pruner():
    global _PRUNER
    _PRUNER = ObjectConstraintPruner()

    lf.on_training_start(_PRUNER.on_training_start)
    lf.on_post_step(_PRUNER.on_post_step)
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
        self.last_actions: list[str] = []
        self.last_thresholds = {"radius": 0.0, "max_axis": 0.0, "aspect": 0.0}

        AppState.iteration.subscribe_as(_PLUGIN_OWNER, self._on_iteration_signal)
        AppState.trainer_state.subscribe_as(_PLUGIN_OWNER, self._on_trainer_state_signal)
        AppState.has_trainer.subscribe_as(_PLUGIN_OWNER, self._on_has_trainer_signal)

    def save_settings(self):
        save_persistent_settings(self.settings)

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

    def _current_total_iterations(self) -> int:
        try:
            total = int(lf.trainer_total_iterations())
            if total > 0:
                return total
        except Exception:
            pass
        return max(1, int(self.last_seen_iteration) + 1)

    def _current_iteration(self) -> int:
        try:
            return int(lf.trainer_current_iteration())
        except Exception:
            return int(AppState.iteration.value)

    def _lerp(self, start: float, end: float, t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return float(start) + (float(end) - float(start)) * t

    def _progress_ratio(self, iteration: int) -> float:
        total = self._current_total_iterations()
        if total <= 1:
            return 1.0
        return max(0.0, min(1.0, float(iteration) / float(total - 1)))

    def current_thresholds(self, iteration: int | None = None) -> dict[str, float]:
        iteration = self._current_iteration() if iteration is None else int(iteration)
        progress = self._progress_ratio(iteration)
        thresholds = {
            "radius": self._lerp(self.settings.radius_start, self.settings.radius_end, progress),
            "max_axis": self._lerp(self.settings.max_axis_start, self.settings.max_axis_end, progress),
            "aspect": self._lerp(self.settings.max_aspect_start, self.settings.max_aspect_end, progress),
        }
        thresholds["radius"] = max(0.0, thresholds["radius"])
        thresholds["max_axis"] = max(0.0, thresholds["max_axis"])
        thresholds["aspect"] = max(1.0, thresholds["aspect"])
        self.last_thresholds = thresholds
        return thresholds

    def _set_opacity_target(self, model, mask, target_opacity: float):
        raw_target = self._to_logit(target_opacity)

        try:
            raw = model.opacity_raw
            cond = mask.unsqueeze(1)
            updated = lf.Tensor.where(cond, raw_target, raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Opacity update failed: {exc}", level="warn")
            return False

    def _largest_axis_columns(self, raw_scaling):
        raw_max = self._reduce_values(raw_scaling.max(dim=1))
        return raw_scaling >= (raw_max.unsqueeze(1) - 1e-8)

    def _multiply_scale_on_mask(self, model, mask, multiplier: float, scope: str):
        multiplier = max(1e-6, min(1.0, float(multiplier)))
        delta_raw = math.log(multiplier)

        try:
            raw = model.scaling_raw
            if scope == "all_axes":
                cond = mask.unsqueeze(1)
            else:
                largest_cols = self._largest_axis_columns(raw)
                cond = mask.unsqueeze(1) & largest_cols
            updated = lf.Tensor.where(cond, raw + delta_raw, raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Scale update failed: {exc}", level="warn")
            return False

    def _shrink_aspect_to_threshold(self, model, mask, max_aspect: float, scope: str, multiplier: float):
        max_aspect = max(1.0, float(max_aspect))
        multiplier = max(1e-6, min(1.0, float(multiplier)))
        target_delta = math.log(max_aspect)
        try:
            raw = model.scaling_raw
            updated = raw.clone()
            if scope == "all_axes":
                updated = updated + (mask.unsqueeze(1).to(updated.dtype) * math.log(multiplier))
            raw_max = self._reduce_values(updated.max(dim=1))
            raw_min = self._reduce_values(updated.min(dim=1))
            target_raw_max = raw_min + target_delta
            largest_cols = updated >= (raw_max.unsqueeze(1) - 1e-8)
            corrected = lf.Tensor.where(
                largest_cols,
                lf.Tensor.minimum(updated, target_raw_max.unsqueeze(1)),
                updated,
            )
            final = lf.Tensor.where(mask.unsqueeze(1), corrected, raw)
            return self._assign_tensor(raw, final)
        except Exception as exc:
            self._set_status(f"Aspect shrink failed: {exc}", level="warn")
            return False

    def _soft_delete(self, model, mask):
        try:
            model.soft_delete(mask)
            return True
        except Exception as exc:
            self._set_status(f"Delete action failed: {exc}", level="warn")
            return False

    def _move_toward_center(self, model, mask, keep_ratio: float = 0.05):
        if self.center_xyz is None:
            self.capture_center_from_scene(force=True)
        if self.center_xyz is None:
            self._set_status("Move action needs a valid center.", level="warn")
            return False
        keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
        try:
            cx, cy, cz = self.center_xyz
            means = model.means_raw
            center_tensor = lf.Tensor.full([int(model.num_points), 3], 0.0, device=means.device, dtype=means.dtype)
            center_tensor[:, 0] = cx
            center_tensor[:, 1] = cy
            center_tensor[:, 2] = cz
            near_center = center_tensor + ((means - center_tensor) * keep_ratio)
            cond = mask.unsqueeze(1)
            updated = lf.Tensor.where(cond, near_center, means)
            return self._assign_tensor(means, updated)
        except Exception as exc:
            self._set_status(f"Move action failed: {exc}", level="warn")
            return False

    def _alive_mask(self, model):
        try:
            if hasattr(model, "has_deleted_mask") and model.has_deleted_mask():
                return model.deleted == False
        except Exception:
            pass
        return None

    def _masked_count(self, mask) -> int:
        if mask is None:
            return 0
        return int(mask.sum().item())

    def _on_has_trainer_signal(self, has_trainer: bool):
        if not has_trainer:
            self._set_status("No trainer available.", level="warn")
            self._request_redraw()

    def _on_trainer_state_signal(self, state: str):
        self._set_status(f"Trainer state: {state}")
        self._request_redraw()

    def _on_iteration_signal(self, iteration: int):
        try:
            self.last_seen_iteration = int(iteration)
            self.current_thresholds(self.last_seen_iteration)
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
        self.current_thresholds(0)

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

    def on_post_step(self, *args, **kwargs):
        del args, kwargs
        try:
            if not AppState.is_training.value:
                return
            iteration = self._current_iteration()
            self.last_seen_iteration = iteration

            if self.pending_manual_prune:
                self.pending_manual_prune = False
                self.prune_once(manual_trigger=True, forced_iteration=iteration)
            else:
                self.prune_once(manual_trigger=False, forced_iteration=iteration)

            self._request_redraw()
        except Exception as exc:
            self._set_status(f"Post-step callback failed: {exc}", level="error")
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
            self._set_status("Manual suppression queued for next training step.")
            self._request_redraw()
            return 0

        out = self.prune_once(
            manual_trigger=True,
            forced_iteration=self._current_iteration(),
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
            self._set_status("Cleared soft-delete mask on combined_model.")
        else:
            self._set_status("Failed to clear soft-delete mask on combined_model.", level="warn")

        self._request_redraw()
        return ok

    def prune_once(self, manual_trigger: bool = False, forced_iteration: int | None = None):
        s = self.settings
        self.last_removed = 0
        self.last_actions = []

        iteration = int(forced_iteration if forced_iteration is not None else self._current_iteration())
        thresholds = self.current_thresholds(iteration)

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

        matches, counts = self._build_condition_masks(model, thresholds)
        self.last_counts = counts

        if matches is None:
            self._set_status("No match criteria are enabled.")
            return 0

        total_matched = counts["radius"] + counts["max_axis"] + counts["aspect"]
        if total_matched == 0:
            self.last_pruned_iteration = iteration
            self._set_status(
                f"iter={iteration}: matched 0 (radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']})"
            )
            return 0

        actions_taken: list[str] = []
        affected_union = None
        try:
            for rule_name, action_name in (
                ("radius", s.radius_action),
                ("max_axis", s.max_axis_action),
                ("aspect", s.aspect_action),
            ):
                rule_mask = matches[rule_name]
                if self._masked_count(rule_mask) == 0 or action_name != "delete":
                    continue
                if self._soft_delete(model, rule_mask):
                    actions_taken.append(f"{rule_name}:delete")
                    affected_union = rule_mask if affected_union is None else (affected_union | rule_mask)

            for rule_name, action_name in (
                ("radius", s.radius_action),
                ("max_axis", s.max_axis_action),
                ("aspect", s.aspect_action),
            ):
                rule_mask = matches[rule_name]
                if self._masked_count(rule_mask) == 0 or action_name == "delete":
                    continue

                ok = False
                if action_name == "fade":
                    ok = self._set_opacity_target(model, rule_mask, float(s.opacity_target))
                elif action_name == "move":
                    ok = self._move_toward_center(model, rule_mask)
                elif action_name == "shrink":
                    if rule_name == "aspect":
                        ok = self._shrink_aspect_to_threshold(
                            model,
                            rule_mask,
                            thresholds["aspect"],
                            str(s.aspect_scale_scope),
                            float(s.scale_multiplier),
                        )
                    elif rule_name == "max_axis":
                        ok = self._multiply_scale_on_mask(
                            model,
                            rule_mask,
                            float(s.scale_multiplier),
                            str(s.max_axis_scale_scope),
                        )
                    else:
                        ok = self._multiply_scale_on_mask(
                            model,
                            rule_mask,
                            float(s.scale_multiplier),
                            str(s.radius_scale_scope),
                        )

                if ok:
                    actions_taken.append(f"{rule_name}:{action_name}")
                    affected_union = rule_mask if affected_union is None else (affected_union | rule_mask)

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

        affected = self._masked_count(affected_union)
        self.last_removed = affected
        self.total_removed += affected
        self.last_pruned_iteration = iteration
        self.last_actions = actions_taken

        prefix = "manual" if manual_trigger else f"iter={iteration}"
        if not actions_taken:
            self._set_status(
                f"{prefix}: matched radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']}, but no actions were applied."
            )
            return affected

        self._set_status(
            f"{prefix}: affected={affected} "
            f"(radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']}; "
            f"thr_radius={thresholds['radius']:.4f}, thr_max_axis={thresholds['max_axis']:.6f}, thr_aspect={thresholds['aspect']:.4f}); "
            f"actions={', '.join(actions_taken)}"
        )
        return affected

    def _build_condition_masks(self, model, thresholds: dict[str, float]):
        n = int(model.num_points)
        if n <= 0:
            return None, {"radius": 0, "max_axis": 0, "aspect": 0}

        any_rule = False
        counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        alive = self._alive_mask(model)
        false_mask = lf.Tensor.zeros([n], dtype="bool", device="cuda")
        matches: dict[str, Any] = {
            "radius": false_mask.clone(),
            "max_axis": false_mask.clone(),
            "aspect": false_mask.clone(),
        }

        if bool(self.settings.enable_max_axis) or bool(self.settings.enable_aspect):
            scaling = model.get_scaling()
            max_axis = self._reduce_values(scaling.max(dim=1))
            min_axis = self._reduce_values(scaling.min(dim=1))

            if bool(self.settings.enable_max_axis):
                any_rule = True
                big_mask = max_axis > float(thresholds["max_axis"])
                if alive is not None:
                    big_mask = big_mask & alive
                matches["max_axis"] = big_mask
                counts["max_axis"] = self._masked_count(big_mask)

            if bool(self.settings.enable_aspect):
                any_rule = True
                aspect = max_axis / (min_axis + _EPS)
                stretch_mask = aspect > float(thresholds["aspect"])
                if alive is not None:
                    stretch_mask = stretch_mask & alive
                matches["aspect"] = stretch_mask
                counts["aspect"] = self._masked_count(stretch_mask)

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
                radius2 = float(thresholds["radius"] * thresholds["radius"])
                far_mask = dist2 > radius2
                if alive is not None:
                    far_mask = far_mask & alive
                matches["radius"] = far_mask
                counts["radius"] = self._masked_count(far_mask)

        if not any_rule:
            return None, counts

        return matches, counts
