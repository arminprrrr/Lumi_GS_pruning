import math
from collections import deque
from typing import Any

import lichtfeld as lf
from lfs_plugins.ui.state import AppState

from .settings import GuardSettings, save_persistent_settings

_EPS = 1e-8
_PLUGIN_OWNER = "Lumi_GS_pruning"
_PRUNER = None
_RULES = ("radius", "max_axis", "aspect")
_THRESHOLD_ATTRS = {
    "radius": ("radius_start", "radius_end"),
    "max_axis": ("max_axis_start", "max_axis_end"),
    "aspect": ("max_aspect_start", "max_aspect_end"),
}


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
        self.status_log = deque(maxlen=80)
        self.pending_manual_prune = False
        self.last_seen_iteration = -1
        self.last_pruned_iteration = -1
        self.last_counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        self.last_actions: list[str] = []
        self.last_thresholds = {"radius": 0.0, "max_axis": 0.0, "aspect": 0.0}
        self.last_rule_values = {
            rule: {
                "opacity_multiplier": 1.0,
                "scale_multiplier": 1.0,
                "clamp_value": 1.0,
                "action": "none",
            }
            for rule in _RULES
        }
        self.active_profile_label = "Global"
        self._append_status_log("Idle.", level="info")

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

    def _append_status_log(self, message: str, level: str = "info"):
        line = f"[{str(level).upper()}] {message}"
        if len(self.status_log) == 0 or self.status_log[-1] != line:
            self.status_log.append(line)

    def get_recent_status_lines(self, limit: int = 12) -> list[str]:
        limit = max(1, int(limit))
        if not self.status_log:
            return []
        return list(self.status_log)[-limit:]

    def clear_status_log(self):
        self.status_log.clear()
        self.status_message = "Runtime log cleared."
        self._append_status_log(self.status_message, level="info")
        self._request_redraw()

    def _set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self._append_status_log(message, level=level)

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

    def _phase_progress_ratio(self, iteration: int, start_iter: int, end_iter: int) -> float:
        span = max(1, int(end_iter) - int(start_iter))
        return max(0.0, min(1.0, float(iteration - int(start_iter)) / float(span)))

    def _find_active_phase(self, iteration: int):
        if not bool(self.settings.use_phases) or len(self.settings.phases) <= 0:
            return None, None
        for phase in self.settings.phases:
            if not bool(phase.enabled):
                continue
            if int(phase.start_iter) <= iteration <= int(phase.end_iter):
                label = str(phase.name).strip() or "Phase"
                return phase, label
        return None, None

    def _build_rule_config(self, source, rule: str, progress: float):
        threshold_start_attr, threshold_end_attr = _THRESHOLD_ATTRS[rule]
        threshold = self._lerp(
            getattr(source, threshold_start_attr),
            getattr(source, threshold_end_attr),
            progress,
        )
        if rule == "aspect":
            threshold = max(1.0, float(threshold))
        else:
            threshold = max(0.0, float(threshold))

        return {
            "enabled": bool(getattr(source, f"enable_{rule}")),
            "threshold": threshold,
            "action": str(getattr(source, f"{rule}_action")),
            "scale_scope": str(getattr(source, f"{rule}_scale_scope")),
            "opacity_multiplier": max(
                1e-6,
                min(
                    1.0,
                    self._lerp(
                        getattr(source, f"{rule}_opacity_start"),
                        getattr(source, f"{rule}_opacity_end"),
                        progress,
                    ),
                ),
            ),
            "scale_multiplier": max(
                1e-6,
                self._lerp(
                    getattr(source, f"{rule}_scale_multiplier_start"),
                    getattr(source, f"{rule}_scale_multiplier_end"),
                    progress,
                ),
            ),
            "clamp_value": max(
                1e-6,
                self._lerp(
                    getattr(source, f"{rule}_clamp_start"),
                    getattr(source, f"{rule}_clamp_end"),
                    progress,
                ),
            ),
        }

    def _current_profile(self, iteration: int | None = None):
        iteration = self._current_iteration() if iteration is None else int(iteration)
        phase, phase_label = self._find_active_phase(iteration)

        if phase is not None:
            progress = self._phase_progress_ratio(iteration, int(phase.start_iter), int(phase.end_iter))
            source = phase
            label = phase_label or "Phase"
        elif bool(self.settings.use_phases) and len(self.settings.phases) > 0:
            self.active_profile_label = "No active phase"
            return None
        else:
            progress = self._progress_ratio(iteration)
            source = self.settings
            label = "Global"

        rules = {rule: self._build_rule_config(source, rule, progress) for rule in _RULES}
        self.last_thresholds = {rule: rules[rule]["threshold"] for rule in _RULES}
        self.last_rule_values = {
            rule: {
                "opacity_multiplier": rules[rule]["opacity_multiplier"],
                "scale_multiplier": rules[rule]["scale_multiplier"],
                "clamp_value": rules[rule]["clamp_value"],
                "action": rules[rule]["action"],
            }
            for rule in _RULES
        }
        self.active_profile_label = label
        return {"label": label, "progress": progress, "rules": rules}

    def current_thresholds(self, iteration: int | None = None) -> dict[str, float]:
        profile = self._current_profile(iteration)
        if profile is None:
            self.last_thresholds = {"radius": 0.0, "max_axis": 0.0, "aspect": 0.0}
            return dict(self.last_thresholds)
        return dict(self.last_thresholds)

    def _tensor_log(self, tensor):
        if hasattr(tensor, "log"):
            return tensor.log()
        if hasattr(lf.Tensor, "log"):
            return lf.Tensor.log(tensor)
        cpu_np = tensor.cpu().numpy(copy=True)
        import numpy as np
        return lf.Tensor.from_numpy(np.log(cpu_np), copy=True).to(tensor.dtype).cuda() if tensor.is_cuda else lf.Tensor.from_numpy(np.log(cpu_np), copy=True).to(tensor.dtype)

    def _tensor_logit(self, tensor):
        return self._tensor_log(tensor / (1.0 - tensor))

    def _apply_opacity_multiplier(self, model, mask, multiplier: float):
        multiplier = max(1e-6, min(1.0, float(multiplier)))
        if abs(multiplier - 1.0) <= 1e-12:
            return False

        try:
            current = model.get_opacity()
            target = current * multiplier
            if hasattr(target, "clamp"):
                target = target.clamp(1e-6, 1.0 - 1e-6)
            else:
                target = lf.Tensor.minimum(target, lf.Tensor.full(list(target.shape), 1.0 - 1e-6, device=target.device, dtype=target.dtype))
                floor = lf.Tensor.full(list(target.shape), 1e-6, device=target.device, dtype=target.dtype)
                target = lf.Tensor.where(target < floor, floor, target)

            raw_target = self._tensor_logit(target).unsqueeze(1)
            raw = model.opacity_raw
            cond = mask.unsqueeze(1)
            updated = lf.Tensor.where(cond, raw_target, raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Opacity multiply failed: {exc}", level="warn")
            return False

    def _largest_axis_columns(self, raw_scaling):
        raw_max = self._reduce_values(raw_scaling.max(dim=1))
        return raw_scaling >= (raw_max.unsqueeze(1) - 1e-8)

    def _scale_condition(self, raw_scaling, mask, scope: str):
        if scope == "all_axes":
            return mask.unsqueeze(1)
        largest_cols = self._largest_axis_columns(raw_scaling)
        return mask.unsqueeze(1) & largest_cols

    def _apply_scale_multiplier(self, model, mask, multiplier: float, scope: str, mode: str):
        try:
            raw = model.scaling_raw
            if mode == "shrink":
                multiplier = min(1.0, max(1e-6, float(multiplier)))
            elif mode == "expand":
                multiplier = max(1.0, float(multiplier))
            else:
                raise ValueError(f"Unknown scale mode: {mode}")

            if abs(multiplier - 1.0) <= 1e-12:
                return False

            delta_raw = math.log(multiplier)
            cond = self._scale_condition(raw, mask, scope)
            updated = lf.Tensor.where(cond, raw + delta_raw, raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Scale {mode} failed: {exc}", level="warn")
            return False

    def _apply_scale_clamp(self, model, mask, clamp_value: float, scope: str):
        clamp_value = max(1e-6, float(clamp_value))
        try:
            raw = model.scaling_raw
            clamp_raw = math.log(clamp_value)
            cond = self._scale_condition(raw, mask, scope)
            if int((cond & (raw > clamp_raw)).any().item()) == 0:
                return False
            clamp_tensor = lf.Tensor.full(list(raw.shape), clamp_raw, device=raw.device, dtype=raw.dtype)
            capped = lf.Tensor.minimum(raw, clamp_tensor)
            updated = lf.Tensor.where(cond, capped, raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Scale clamp failed: {exc}", level="warn")
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

        if not bool(self.settings.enabled):
            self._set_status("Training started. Plugin is disabled, so it will not modify the model.")
            self._request_redraw()
            return

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

            if not bool(self.settings.enabled):
                self.pending_manual_prune = False
                self.current_thresholds(iteration)
            elif self.pending_manual_prune:
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
        if not bool(self.settings.enabled):
            self.pending_manual_prune = False
            self._set_status("Plugin is disabled. Manual suppression was ignored.")
            self._request_redraw()
            return 0

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

    def _apply_action(self, model, rule_mask, rule_cfg: dict[str, Any]):
        action_name = str(rule_cfg["action"])
        if action_name in {"none", "delete"}:
            return False

        changed = False
        if action_name in {"fade", "fade_shrink", "fade_expand", "fade_clamp"}:
            changed = self._apply_opacity_multiplier(model, rule_mask, float(rule_cfg["opacity_multiplier"])) or changed

        if action_name in {"shrink", "fade_shrink"}:
            changed = self._apply_scale_multiplier(
                model,
                rule_mask,
                float(rule_cfg["scale_multiplier"]),
                str(rule_cfg["scale_scope"]),
                mode="shrink",
            ) or changed
        elif action_name in {"expand", "fade_expand"}:
            changed = self._apply_scale_multiplier(
                model,
                rule_mask,
                float(rule_cfg["scale_multiplier"]),
                str(rule_cfg["scale_scope"]),
                mode="expand",
            ) or changed
        elif action_name in {"clamp", "fade_clamp"}:
            changed = self._apply_scale_clamp(
                model,
                rule_mask,
                float(rule_cfg["clamp_value"]),
                str(rule_cfg["scale_scope"]),
            ) or changed
        elif action_name == "move":
            changed = self._move_toward_center(model, rule_mask) or changed

        return bool(changed)

    def prune_once(self, manual_trigger: bool = False, forced_iteration: int | None = None):
        s = self.settings
        self.last_removed = 0
        self.last_actions = []

        iteration = int(forced_iteration if forced_iteration is not None else self._current_iteration())
        profile = self._current_profile(iteration)

        if not bool(s.enabled):
            self.pending_manual_prune = False
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

        if profile is None:
            self._set_status(f"iter={iteration}: no active phase matched this iteration.")
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

        matches, counts = self._build_condition_masks(model, profile["rules"])
        self.last_counts = counts

        if matches is None:
            self._set_status("No match criteria are enabled.")
            return 0

        total_matched = counts["radius"] + counts["max_axis"] + counts["aspect"]
        if total_matched == 0:
            self.last_pruned_iteration = iteration
            self._set_status(
                f"iter={iteration}: matched 0 in {profile['label']} "
                f"(radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']})"
            )
            return 0

        actions_taken: list[str] = []
        affected_union = None
        try:
            for rule_name in _RULES:
                rule_cfg = profile["rules"][rule_name]
                action_name = str(rule_cfg["action"])
                rule_mask = matches[rule_name]
                if self._masked_count(rule_mask) == 0 or action_name != "delete":
                    continue
                if self._soft_delete(model, rule_mask):
                    actions_taken.append(f"{rule_name}:delete")
                    affected_union = rule_mask if affected_union is None else (affected_union | rule_mask)

            for rule_name in _RULES:
                rule_cfg = profile["rules"][rule_name]
                action_name = str(rule_cfg["action"])
                rule_mask = matches[rule_name]
                if self._masked_count(rule_mask) == 0 or action_name == "delete":
                    continue
                if self._apply_action(model, rule_mask, rule_cfg):
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
                f"{prefix}: matched radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']} in {profile['label']}, but no actions were applied."
            )
            return affected

        self._set_status(
            f"{prefix}: profile={profile['label']} affected={affected} "
            f"(radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']}; "
            f"thr_radius={self.last_thresholds['radius']:.4f}, thr_max_axis={self.last_thresholds['max_axis']:.6f}, thr_aspect={self.last_thresholds['aspect']:.4f}); "
            f"actions={', '.join(actions_taken)}"
        )
        return affected

    def _build_condition_masks(self, model, rule_configs: dict[str, dict[str, Any]]):
        n = int(model.num_points)
        if n <= 0:
            return None, {"radius": 0, "max_axis": 0, "aspect": 0}

        any_rule = False
        counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        alive = self._alive_mask(model)
        try:
            device = model.means_raw.device
        except Exception:
            device = "cuda"
        false_mask = lf.Tensor.zeros([n], dtype="bool", device=device)
        matches: dict[str, Any] = {
            "radius": false_mask.clone(),
            "max_axis": false_mask.clone(),
            "aspect": false_mask.clone(),
        }

        if bool(rule_configs["max_axis"]["enabled"]) or bool(rule_configs["aspect"]["enabled"]):
            scaling = model.get_scaling()
            max_axis = self._reduce_values(scaling.max(dim=1))
            min_axis = self._reduce_values(scaling.min(dim=1))

            if bool(rule_configs["max_axis"]["enabled"]):
                any_rule = True
                big_mask = max_axis > float(rule_configs["max_axis"]["threshold"])
                if alive is not None:
                    big_mask = big_mask & alive
                matches["max_axis"] = big_mask
                counts["max_axis"] = self._masked_count(big_mask)

            if bool(rule_configs["aspect"]["enabled"]):
                any_rule = True
                aspect = max_axis / (min_axis + _EPS)
                stretch_mask = aspect > float(rule_configs["aspect"]["threshold"])
                if alive is not None:
                    stretch_mask = stretch_mask & alive
                matches["aspect"] = stretch_mask
                counts["aspect"] = self._masked_count(stretch_mask)

        if bool(rule_configs["radius"]["enabled"]):
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
                radius2 = float(rule_configs["radius"]["threshold"] * rule_configs["radius"]["threshold"])
                far_mask = dist2 > radius2
                if alive is not None:
                    far_mask = far_mask & alive
                matches["radius"] = far_mask
                counts["radius"] = self._masked_count(far_mask)

        if not any_rule:
            return None, counts

        return matches, counts
