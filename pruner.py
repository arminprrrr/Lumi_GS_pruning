import math
from collections import deque
from typing import Any

import lichtfeld as lf

try:
    from lfs_plugins.ui.state import AppState
except Exception:
    AppState = None

from .settings import GuardSettings, save_persistent_settings

_EPS = 1e-8
_PLUGIN_OWNER = "Lumi_GS_pruning"
_PRUNER = None
_CALLBACK_HANDLER = None
_RUNTIME_MODE = "auto"
_HOOK_REGISTRATION = "unknown"
_RULES = ("radius", "max_axis", "aspect")
_THRESHOLD_ATTRS = {
    "radius": ("radius_start", "radius_end"),
    "max_axis": ("max_axis_start", "max_axis_end"),
    "aspect": ("max_aspect_start", "max_aspect_end"),
}
_GREYSCALE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def get_pruner():
    return _PRUNER


def set_runtime_mode(mode: str | None):
    global _RUNTIME_MODE
    text = str(mode or "auto").strip().lower()
    if text in {"headless/no-ui", "headless_no_ui", "headless-no-ui", "headless"}:
        _RUNTIME_MODE = "headless"
    elif text in {"ui", "retained-ui", "retained_ui", "retained"}:
        _RUNTIME_MODE = "ui"
    else:
        _RUNTIME_MODE = "auto"


def get_runtime_mode() -> str:
    return str(_RUNTIME_MODE)



def _dispatch_training_start(*args, **kwargs):
    if _PRUNER is not None:
        return _PRUNER.on_training_start(*args, **kwargs)
    return None



def _dispatch_iteration_start(*args, **kwargs):
    if _PRUNER is not None:
        return _PRUNER.on_iteration_start(*args, **kwargs)
    return None



def _dispatch_post_step(*args, **kwargs):
    if _PRUNER is not None:
        return _PRUNER.on_post_step(*args, **kwargs)
    return None



def _dispatch_training_end(*args, **kwargs):
    if _PRUNER is not None:
        return _PRUNER.on_training_end(*args, **kwargs)
    return None



def _app_state_signal_value(name: str, default=None):
    if AppState is None:
        return default
    try:
        signal = getattr(AppState, name)
    except Exception:
        return default
    try:
        return signal.value
    except Exception:
        return default



def _is_headless_runtime() -> bool:
    if _RUNTIME_MODE == "headless":
        return True
    if _RUNTIME_MODE == "ui":
        return False
    value = _app_state_signal_value("is_headless", None)
    if value is not None:
        try:
            return bool(value)
        except Exception:
            pass
    return False



def _register_global_hooks():
    global _HOOK_REGISTRATION
    lf.on_training_start(_dispatch_training_start)
    lf.on_iteration_start(_dispatch_iteration_start)
    lf.on_post_step(_dispatch_post_step)
    lf.on_training_end(_dispatch_training_end)
    _HOOK_REGISTRATION = "global"



def _install_callback_handler():
    global _CALLBACK_HANDLER, _HOOK_REGISTRATION
    if _is_headless_runtime():
        _CALLBACK_HANDLER = None
        _register_global_hooks()
        return

    handler_cls = getattr(lf, "ScopedHandler", None) or getattr(lf, "ControlSession", None)
    if handler_cls is None:
        _CALLBACK_HANDLER = None
        _register_global_hooks()
        return
    try:
        handler = handler_cls()
        handler.on_training_start(_dispatch_training_start)
        handler.on_iteration_start(_dispatch_iteration_start)
        handler.on_post_step(_dispatch_post_step)
        handler.on_training_end(_dispatch_training_end)
        _CALLBACK_HANDLER = handler
        _HOOK_REGISTRATION = handler_cls.__name__
    except Exception as exc:
        _CALLBACK_HANDLER = None
        lf.log.warn(f"[{_PLUGIN_OWNER}] Scoped callback registration failed; falling back to global hooks: {exc}")
        _register_global_hooks()



def install_pruner(runtime_mode: str | None = None):
    global _PRUNER
    if runtime_mode is not None:
        set_runtime_mode(runtime_mode)
    uninstall_pruner()
    _PRUNER = ObjectConstraintPruner()
    _install_callback_handler()
    try:
        lf.log.info(f"[{_PLUGIN_OWNER}] Runtime mode={get_runtime_mode()} hook_registration={_HOOK_REGISTRATION}")
    except Exception:
        pass



def uninstall_pruner():
    global _PRUNER, _CALLBACK_HANDLER, _HOOK_REGISTRATION
    if _CALLBACK_HANDLER is not None:
        try:
            _CALLBACK_HANDLER.clear()
        except Exception as exc:
            lf.log.warn(f"[{_PLUGIN_OWNER}] Failed to clear callback handler: {exc}")
        _CALLBACK_HANDLER = None
    _HOOK_REGISTRATION = "unknown"
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
        self.pending_manual_greyscale = False
        self.last_seen_iteration = -1
        self.last_pruned_iteration = -1
        self.last_processed_iteration = -1
        self.post_step_calls = 0
        self.stale_iteration_hits = 0
        self.iteration_source = "startup"
        self.iteration_hint = None
        self.last_counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        self.last_actions: list[str] = []
        self.last_thresholds = {"radius": 0.0, "max_axis": 0.0, "aspect": 0.0}
        self.last_rule_values = {rule: {"opacity_multiplier": 1.0, "scale_multiplier": 1.0, "action": "none"} for rule in _RULES}
        self.active_profile_label = "Global"
        self.last_greyscale_iteration = -1
        self.last_greyscale_affected = 0
        self.total_greyscale_affected = 0
        self.last_greyscale_fields: list[str] = []
        self._last_scene = None
        self._last_scene_source = "none"
        self._scene_wait_hits = 0
        self._model_wait_hits = 0
        self._append_status_log("Idle.", level="info")
        self._subscribe_app_state()

    def _subscribe_app_state(self):
        if AppState is None:
            return
        try:
            AppState.iteration.subscribe_as(_PLUGIN_OWNER, self._on_iteration_signal)
        except Exception:
            pass
        try:
            AppState.trainer_state.subscribe_as(_PLUGIN_OWNER, self._on_trainer_state_signal)
        except Exception:
            pass
        try:
            AppState.has_trainer.subscribe_as(_PLUGIN_OWNER, self._on_has_trainer_signal)
        except Exception:
            pass

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
        return list(self.status_log)[-max(1, int(limit)):]

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

    def _set_status_quiet(self, message: str):
        self.status_message = message

    def _set_wait_status(self, reason: str, *, manual_trigger: bool = False, level: str = "warn"):
        if manual_trigger:
            self._set_status(reason, level=level)
            return
        self._set_status_quiet(reason)
        hits_attr = "_scene_wait_hits" if "scene" in reason.lower() else "_model_wait_hits"
        hits = int(getattr(self, hits_attr, 0)) + 1
        setattr(self, hits_attr, hits)
        if hits in {3, 10} or (hits % 250 == 0):
            self._set_status(reason, level="warn")

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

    def _shape_dim(self, tensor, dim: int, default: int = 0) -> int:
        try:
            return int(tensor.shape[dim])
        except Exception:
            return int(default)

    def _build_greyscale_rgb(self, colors):
        n = self._shape_dim(colors, 0, 0)
        if n <= 0:
            return None
        gray = (colors[:, 0] * _GREYSCALE_WEIGHTS[0]) + (colors[:, 1] * _GREYSCALE_WEIGHTS[1]) + (colors[:, 2] * _GREYSCALE_WEIGHTS[2])
        gray_col = gray.unsqueeze(1)
        out = lf.Tensor.zeros([n, 3], device=colors.device, dtype=colors.dtype)
        out[:, 0] = gray_col
        out[:, 1] = gray_col
        out[:, 2] = gray_col
        return out

    def _greyscale_higher_sh(self, coeffs) -> bool:
        if coeffs is None or int(getattr(coeffs, "ndim", 0)) != 3:
            return False
        if self._shape_dim(coeffs, 2, 0) != 3:
            return False
        if self._shape_dim(coeffs, 1, 0) <= 0:
            return False
        gray = (coeffs[:, :, 0] * _GREYSCALE_WEIGHTS[0]) + (coeffs[:, :, 1] * _GREYSCALE_WEIGHTS[1]) + (coeffs[:, :, 2] * _GREYSCALE_WEIGHTS[2])
        gray_ch = gray.unsqueeze(2)
        coeffs[:, :, 0] = gray_ch
        coeffs[:, :, 1] = gray_ch
        coeffs[:, :, 2] = gray_ch
        return True

    def _candidate_scenes(self):
        out: list[tuple[str, Any]] = []
        seen: set[int] = set()
        for label, getter in (
            ("scene", getattr(lf, "get_scene", None)),
            ("render_scene", getattr(lf, "get_render_scene", None)),
        ):
            if getter is None:
                continue
            try:
                scene = getter()
            except Exception:
                scene = None
            if scene is None:
                continue
            key = id(scene)
            if key in seen:
                continue
            seen.add(key)
            out.append((label, scene))
        if self._last_scene is not None:
            key = id(self._last_scene)
            if key not in seen:
                out.append((f"cached:{self._last_scene_source}", self._last_scene))
        return out

    def _resolve_scene(self):
        candidates = self._candidate_scenes()
        if not candidates:
            return None, "none"
        label, scene = candidates[0]
        self._last_scene = scene
        self._last_scene_source = label
        self._scene_wait_hits = 0
        return scene, label

    def _iter_node_splat_models(self, scene):
        try:
            nodes = scene.get_nodes()
        except Exception:
            nodes = []
        for node in nodes:
            try:
                splat_data = node.splat_data()
            except Exception:
                splat_data = None
            if splat_data is not None:
                yield f"node:{getattr(node, 'name', '<unnamed>')}", splat_data

    def _iter_greyscale_targets(self, scene, prefer_training: bool = False):
        if prefer_training:
            try:
                training_model = scene.training_model()
            except Exception:
                training_model = None
            if training_model is not None:
                yield "training_model", training_model
            try:
                combined_model = scene.combined_model()
            except Exception:
                combined_model = None
            if combined_model is not None and combined_model is not training_model:
                yield "combined_model", combined_model
            for label, splat_data in self._iter_node_splat_models(scene):
                yield label, splat_data
            return
        found_any = False
        for label, splat_data in self._iter_node_splat_models(scene):
            found_any = True
            yield label, splat_data
        if found_any:
            return
        try:
            training_model = scene.training_model()
        except Exception:
            training_model = None
        if training_model is not None:
            yield "training_model", training_model
        try:
            combined_model = scene.combined_model()
        except Exception:
            combined_model = None
        if combined_model is not None and combined_model is not training_model:
            yield "combined_model", combined_model

    def _resolve_scene_and_model(
        self,
        *,
        prefer_training: bool | None = None,
        allow_combined_fallback: bool = True,
        allow_node_fallback: bool = True,
    ):
        if prefer_training is None:
            prefer_training = self._is_training_active()
        scene, scene_source = self._resolve_scene()
        if scene is None:
            return None, None, "no_scene"

        model = None
        model_source = "none"
        if prefer_training:
            try:
                model = scene.training_model()
            except Exception:
                model = None
            if model is not None:
                self._model_wait_hits = 0
                return scene, model, f"{scene_source}:training_model"
        if allow_combined_fallback:
            try:
                model = scene.combined_model()
            except Exception:
                model = None
            if model is not None:
                self._model_wait_hits = 0
                return scene, model, f"{scene_source}:combined_model"
        if allow_node_fallback:
            for label, splat_data in self._iter_node_splat_models(scene):
                self._model_wait_hits = 0
                return scene, splat_data, f"{scene_source}:{label}"
        return scene, None, f"{scene_source}:no_model"

    def apply_greyscale_once(self, manual_trigger: bool = False, forced_iteration: int | None = None):
        iteration = int(forced_iteration if forced_iteration is not None else self._current_iteration())
        training_active = self._is_training_active()
        scene, _model, source = self._resolve_scene_and_model(prefer_training=training_active, allow_combined_fallback=True, allow_node_fallback=True)
        if scene is None:
            self._set_wait_status(
                f"Scene not ready yet for greyscale at iter={iteration}; skipping until LichtFeld exposes a scene object.",
                manual_trigger=manual_trigger,
            )
            return 0
        changed_targets: list[str] = []
        failures: list[str] = []
        affected = 0
        for label, splat_data in self._iter_greyscale_targets(scene, prefer_training=training_active):
            try:
                point_count = int(getattr(splat_data, "num_points", 0))
                if point_count <= 0:
                    continue
                gray_rgb = self._build_greyscale_rgb(splat_data.get_colors_rgb())
                if gray_rgb is None:
                    continue
                splat_data.set_colors_rgb(gray_rgb)
                self._greyscale_higher_sh(getattr(splat_data, "shN_raw", None))
                changed_targets.append(label)
                affected += point_count
            except Exception as exc:
                failures.append(f"{label}: {exc}")
        self.last_greyscale_iteration = int(iteration)
        self.last_greyscale_affected = int(affected)
        self.total_greyscale_affected += int(affected)
        self.last_greyscale_fields = list(changed_targets)
        if not changed_targets:
            details = f" Failures: {'; '.join(failures[:3])}" if failures else ""
            self._set_wait_status(
                f"No writable greyscale targets were updated from {source}.{details}",
                manual_trigger=manual_trigger,
                level="warn",
            )
            return 0
        self._sync_scene_after_edit(scene, training_active=training_active)
        if manual_trigger:
            message = f"Manual greyscale applied to {affected:,} gaussians across {len(changed_targets)} target(s): {', '.join(changed_targets)}"
            if failures:
                message += f" Failures: {'; '.join(failures[:3])}"
            self._set_status(message, level="warn" if failures else "info")
        else:
            message = f"Greyscale enforced on {affected:,} gaussians across {len(changed_targets)} target(s) via {source}."
            if failures:
                message += f" Failures: {'; '.join(failures[:3])}"
                self._set_status(message, level="warn")
            else:
                self._set_status_quiet(message)
        return affected

    def _clear_deleted_mask(self, model):
        if not hasattr(model, "clear_deleted"):
            return True
        try:
            model.clear_deleted()
            return True
        except Exception:
            return False

    def _app_state_value(self, name: str, default=None):
        if AppState is None:
            return default
        try:
            signal = getattr(AppState, name)
        except Exception:
            return default
        try:
            return signal.value
        except Exception:
            return default

    def _trainer_context(self, refresh: bool = False):
        try:
            ctx = lf.context()
        except Exception:
            return None
        if refresh and hasattr(ctx, "refresh"):
            try:
                ctx.refresh()
            except Exception:
                pass
        return ctx

    def _sync_scene_after_edit(self, scene, training_active: bool | None = None):
        if scene is None:
            return False
        if training_active is None:
            training_active = self._is_training_active()
        if training_active:
            return False
        try:
            scene.notify_changed()
        except Exception:
            pass
        ctx = self._trainer_context(refresh=False)
        if ctx is not None and hasattr(ctx, "refresh"):
            try:
                ctx.refresh()
            except Exception:
                pass
        return True

    def _trainer_state_text(self) -> str:
        try:
            state = lf.trainer_state()
            if state is not None:
                text = str(state).strip()
                if text:
                    return text
        except Exception:
            pass
        state = self._app_state_value("trainer_state", "")
        return str(state).strip()

    def _trainer_phase_text(self) -> str:
        ctx = self._trainer_context(refresh=True)
        if ctx is not None:
            try:
                phase = getattr(ctx, "phase", None)
                if callable(phase):
                    text = str(phase()).strip()
                    if text:
                        return text
            except Exception:
                pass
            try:
                phase_value = getattr(ctx, "phase", None)
                if phase_value is not None and not callable(phase_value):
                    text = str(phase_value).strip()
                    if text:
                        return text
            except Exception:
                pass
        phase = self._app_state_value("training_phase", "")
        return str(phase).strip()

    def _in_safe_edit_window(self) -> bool:
        if not self._is_training_active():
            return True
        phase = self._trainer_phase_text().lower()
        return phase in {"safe_control", "idle", "training_start", "training_end"}

    def _is_training_active(self) -> bool:
        votes: list[bool] = []
        ctx = self._trainer_context(refresh=True)
        if ctx is not None:
            try:
                is_training = getattr(ctx, "is_training", None)
                if is_training is not None:
                    votes.append(bool(is_training))
            except Exception:
                pass
        state = self._trainer_state_text().lower()
        if state:
            votes.append(state in {"running", "training", "refining"})
        app_state_value = self._app_state_value("is_training", None)
        if app_state_value is not None:
            votes.append(bool(app_state_value))
        return any(votes) if votes else False

    def _current_total_iterations(self) -> int:
        ctx = self._trainer_context(refresh=True)
        if ctx is not None:
            try:
                total = int(getattr(ctx, "max_iterations", 0))
                if total > 0:
                    return total
            except Exception:
                pass
        try:
            total = int(lf.trainer_total_iterations())
            if total > 0:
                return total
        except Exception:
            pass
        last_seen = max(int(self.last_seen_iteration), int(self._app_state_value("iteration", -1)))
        return max(1, last_seen + 1)

    def _iteration_candidates(self, refresh: bool = True) -> list[tuple[str, int]]:
        candidates: list[tuple[str, int]] = []
        ctx = self._trainer_context(refresh=refresh)
        if ctx is not None:
            try:
                value = int(getattr(ctx, "iteration"))
                if value >= 0:
                    candidates.append(("context", value))
            except Exception:
                pass
        try:
            value = int(lf.trainer_current_iteration())
            if value >= 0:
                candidates.append(("trainer_current_iteration", value))
        except Exception:
            pass
        app_value = self._app_state_value("iteration", None)
        if app_value is not None:
            try:
                value = int(app_value)
                if value >= 0:
                    candidates.append(("app_state", value))
            except Exception:
                pass
        return candidates

    def _resolve_callback_iteration(self) -> int | None:
        candidates: list[tuple[str, int]] = []
        if self.iteration_hint is not None:
            try:
                hint_value = int(self.iteration_hint)
                if hint_value >= 0:
                    candidates.append(("iteration_start_hint", hint_value))
            except Exception:
                pass
        candidates.extend(self._iteration_candidates(refresh=True))
        target_floor = max(int(self.last_pruned_iteration), int(getattr(self, "last_processed_iteration", -1)))
        advanced = [(name, value) for name, value in candidates if value > target_floor]
        if advanced:
            name, value = max(advanced, key=lambda item: item[1])
            self.iteration_source = name
            self.stale_iteration_hits = 0
            self.iteration_hint = None
            return int(value)
        self.stale_iteration_hits += 1
        if candidates:
            name, value = max(candidates, key=lambda item: item[1])
            self.iteration_source = f"{name}->stale"
            if self.stale_iteration_hits in {3, 10} or (self.stale_iteration_hits % 1000 == 0):
                self._set_status(
                    f"Iteration source '{name}' stayed at {value}; skipping this callback to avoid duplicate edits.",
                    level="warn",
                )
        else:
            self.iteration_source = "no_iteration_source"
            if self.stale_iteration_hits in {3, 10} or (self.stale_iteration_hits % 1000 == 0):
                self._set_status(
                    "Training callback has no usable iteration source; skipping this callback.",
                    level="warn",
                )
        self.iteration_hint = None
        return None

    def _current_iteration(self) -> int:
        candidates = self._iteration_candidates(refresh=True)
        if candidates:
            name, value = max(candidates, key=lambda item: item[1])
            self.iteration_source = name
            return int(value)
        return int(self.last_seen_iteration if self.last_seen_iteration >= 0 else 0)

    def _lerp(self, start: float, end: float, t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return float(start) + (float(end) - float(start)) * t

    def _progress_ratio(self, iteration: int) -> float:
        total = self._current_total_iterations()
        return 1.0 if total <= 1 else max(0.0, min(1.0, float(iteration) / float(total - 1)))

    def resolve_iter_spec(self, value: Any):
        text = str(value).strip()
        if not text:
            return None
        try:
            if text.endswith("%"):
                pct = max(0.0, min(100.0, float(text[:-1].strip())))
                return max(0, int(round((pct / 100.0) * self._current_total_iterations())))
            return max(0, int(float(text)))
        except Exception:
            self._set_status(f"Invalid iteration spec ignored: {text}", level="warn")
            return None

    def _build_rule_config(self, rule: str, progress: float):
        start_attr, end_attr = _THRESHOLD_ATTRS[rule]
        threshold = self._lerp(getattr(self.settings, start_attr), getattr(self.settings, end_attr), progress)
        threshold = max(1.0, float(threshold)) if rule == "aspect" else max(0.0, float(threshold))
        opacity_multiplier = max(1e-6, min(1.0, self._lerp(getattr(self.settings, f"{rule}_opacity_start"), getattr(self.settings, f"{rule}_opacity_end"), progress)))
        scale_multiplier = max(1e-6, self._lerp(getattr(self.settings, f"{rule}_scale_multiplier_start"), getattr(self.settings, f"{rule}_scale_multiplier_end"), progress))
        return {
            "enabled": bool(getattr(self.settings, f"enable_{rule}")),
            "threshold": threshold,
            "action": str(getattr(self.settings, f"{rule}_action")),
            "scale_scope": str(getattr(self.settings, f"{rule}_scale_scope")),
            "opacity_multiplier": opacity_multiplier,
            "scale_multiplier": scale_multiplier,
        }

    def _current_profile(self, iteration: int | None = None):
        iteration = self._current_iteration() if iteration is None else int(iteration)
        progress = self._progress_ratio(iteration)
        rules = {rule: self._build_rule_config(rule, progress) for rule in _RULES}
        self.last_thresholds = {rule: rules[rule]["threshold"] for rule in _RULES}
        self.last_rule_values = {rule: {"opacity_multiplier": rules[rule]["opacity_multiplier"], "scale_multiplier": rules[rule]["scale_multiplier"], "action": rules[rule]["action"]} for rule in _RULES}
        self.active_profile_label = "Global"
        return {"label": "Global", "rules": rules}

    def current_thresholds(self, iteration: int | None = None) -> dict[str, float]:
        self._current_profile(iteration)
        return dict(self.last_thresholds)

    def _tensor_log(self, tensor):
        if hasattr(tensor, "log"):
            return tensor.log()
        if hasattr(lf.Tensor, "log"):
            return lf.Tensor.log(tensor)
        import numpy as np
        cpu_np = tensor.cpu().numpy(copy=True)
        out = lf.Tensor.from_numpy(np.log(cpu_np), copy=True).to(tensor.dtype)
        return out.cuda() if getattr(tensor, "is_cuda", False) else out

    def _tensor_logit(self, tensor):
        return self._tensor_log(tensor / (1.0 - tensor))

    def _apply_opacity_multiplier(self, model, mask, multiplier: float):
        multiplier = max(1e-6, min(1.0, float(multiplier)))
        if abs(multiplier - 1.0) <= 1e-12:
            return False
        try:
            target = (model.get_opacity() * multiplier).clamp(1e-6, 1.0 - 1e-6)
            raw_target = self._tensor_logit(target).unsqueeze(1)
            raw = model.opacity_raw
            updated = lf.Tensor.where(mask.unsqueeze(1), raw_target, raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Opacity multiply failed: {exc}", level="warn")
            return False

    def _largest_axis_columns(self, raw_scaling):
        raw_max = self._reduce_values(raw_scaling.max(dim=1))
        return raw_scaling >= (raw_max.unsqueeze(1) - 1e-8)

    def _scale_condition(self, raw_scaling, mask, scope: str):
        return mask.unsqueeze(1) if scope == "all_axes" else (mask.unsqueeze(1) & self._largest_axis_columns(raw_scaling))

    def _apply_scale_multiplier(self, model, mask, multiplier: float, scope: str, mode: str):
        try:
            raw = model.scaling_raw
            multiplier = min(1.0, max(1e-6, float(multiplier))) if mode == "shrink" else max(1.0, float(multiplier))
            if abs(multiplier - 1.0) <= 1e-12:
                return False
            updated = lf.Tensor.where(self._scale_condition(raw, mask, scope), raw + math.log(multiplier), raw)
            return self._assign_tensor(raw, updated)
        except Exception as exc:
            self._set_status(f"Scale {mode} failed: {exc}", level="warn")
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
            updated = lf.Tensor.where(mask.unsqueeze(1), near_center, means)
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
        return 0 if mask is None else int(mask.sum().item())

    def _on_has_trainer_signal(self, has_trainer: bool):
        if not has_trainer:
            self._set_status("No trainer available.", level="warn")
            self._request_redraw()

    def _on_trainer_state_signal(self, state: str):
        self._set_status(f"Trainer state: {state}")
        self._request_redraw()

    def _on_iteration_signal(self, iteration: int):
        try:
            value = int(iteration)
            self.iteration_hint = value
            self.iteration_source = "app_state->post_step"
            self.current_thresholds(value)
            self._request_redraw()
        except Exception as exc:
            self._set_status(f"Iteration signal failed: {exc}", level="error")
            self._request_redraw()

    def on_training_start(self, *args, **kwargs):
        del args, kwargs
        self.last_removed = 0
        self.total_removed = 0
        self.last_seen_iteration = -1
        self.last_pruned_iteration = -1
        self.last_processed_iteration = -1
        self.post_step_calls = 0
        self.stale_iteration_hits = 0
        self.iteration_source = "training_start"
        self.iteration_hint = None
        self.last_counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        self.last_actions = []
        self.pending_manual_prune = False
        self.pending_manual_greyscale = False
        self.last_greyscale_iteration = -1
        self.last_greyscale_affected = 0
        self.last_greyscale_fields = []
        self._scene_wait_hits = 0
        self._model_wait_hits = 0
        self.current_thresholds(0)
        pruning_enabled = bool(self.settings.enabled)
        greyscale_enabled = bool(getattr(self.settings, "enforce_greyscale", False))
        if not pruning_enabled and not greyscale_enabled:
            self._set_status("Training started. Plugin is disabled, so it will not modify the model.")
            self._request_redraw()
            return
        scene, model, source = self._resolve_scene_and_model(prefer_training=True, allow_combined_fallback=True, allow_node_fallback=True)
        if model is not None and pruning_enabled:
            self._clear_deleted_mask(model)
        elif pruning_enabled:
            self._set_wait_status(
                f"Training started, but the scene/model is not ready yet ({source}). Waiting for Gaussians before applying pruning.",
                manual_trigger=False,
            )
        if pruning_enabled:
            if self.settings.center_mode == "manual":
                self.center_xyz = tuple(float(v) for v in self.settings.center)
                self._set_status(f"Training started. Using manual center {self._fmt_center(self.center_xyz)}.")
            else:
                self.center_xyz = None
                if self.capture_center_from_scene(force=True) is None:
                    self._set_wait_status("Training started. Waiting for Gaussians before center capture.", manual_trigger=False)
        if greyscale_enabled:
            self.apply_greyscale_once(manual_trigger=False, forced_iteration=0)
        self._request_redraw()

    def on_iteration_start(self, *args, **kwargs):
        del args, kwargs
        try:
            candidates = self._iteration_candidates(refresh=True)
            if candidates:
                name, value = max(candidates, key=lambda item: item[1])
                self.iteration_hint = int(value)
                self.iteration_source = f"{name}->post_step"
                self.current_thresholds(self.iteration_hint)
        except Exception as exc:
            self._set_status(f"Iteration-start callback failed: {exc}", level="error")

    def on_post_step(self, *args, **kwargs):
        del args, kwargs
        try:
            self.post_step_calls += 1
            iteration = self._resolve_callback_iteration()
            if iteration is None:
                self._request_redraw()
                return
            self.last_seen_iteration = max(self.last_seen_iteration, int(iteration))
            self.last_processed_iteration = max(self.last_processed_iteration, int(iteration))
            self.current_thresholds(self.last_seen_iteration)
            if not bool(self.settings.enabled):
                self.pending_manual_prune = False
            elif self.pending_manual_prune:
                self.pending_manual_prune = False
                self.prune_once(manual_trigger=True, forced_iteration=self.last_seen_iteration)
            else:
                self.prune_once(manual_trigger=False, forced_iteration=self.last_seen_iteration)
            greyscale_manual = bool(self.pending_manual_greyscale)
            greyscale_enabled = bool(getattr(self.settings, "enforce_greyscale", False))
            self.pending_manual_greyscale = False
            if greyscale_manual or greyscale_enabled:
                self.apply_greyscale_once(manual_trigger=greyscale_manual, forced_iteration=self.last_seen_iteration)
            self._request_redraw()
        except Exception as exc:
            self._set_status(f"Post-step callback failed: {exc}", level="error")
            self._request_redraw()

    def on_training_end(self, *args, **kwargs):
        del args, kwargs
        self.pending_manual_prune = False
        self.pending_manual_greyscale = False
        if self._is_training_active():
            self._set_status_quiet("Ignored stale training_end callback while a new training run is already active.")
            self._request_redraw()
            return
        self._set_status("Training ended.")
        self._request_redraw()

    def capture_center_from_scene(self, force: bool = False):
        if self.settings.center_mode == "manual" and not force:
            self.center_xyz = tuple(float(v) for v in self.settings.center)
            self._set_status(f"Using manual center {self._fmt_center(self.center_xyz)}.")
            return self.center_xyz
        scene, model, source = self._resolve_scene_and_model(
            prefer_training=self._is_training_active(),
            allow_combined_fallback=True,
            allow_node_fallback=True,
        )
        if scene is None:
            self._set_wait_status("Scene not ready yet for center capture; skipping until a scene becomes available.")
            return None
        if model is None:
            self._set_wait_status(f"No Gaussian model available yet for center capture ({source}).")
            return None
        if int(getattr(model, "num_points", 0)) == 0:
            self._set_wait_status("Gaussian model has zero points.")
            return None
        center = model.means_raw.mean(dim=0)
        self.center_xyz = (float(center[0].item()), float(center[1].item()), float(center[2].item()))
        self._set_status(f"Captured center {self._fmt_center(self.center_xyz)} from {source}.")
        return self.center_xyz

    def request_manual_prune(self):
        if not bool(self.settings.enabled):
            self.pending_manual_prune = False
            self._set_status("Plugin is disabled. Manual suppression was ignored.")
            self._request_redraw()
            return 0
        if self._is_training_active():
            self.pending_manual_prune = True
            self._set_status("Manual suppression queued for next training step.")
            self._request_redraw()
            return 0
        out = self.prune_once(manual_trigger=True, forced_iteration=self._current_iteration())
        self._request_redraw()
        return out

    def request_manual_greyscale(self):
        if self._is_training_active():
            self.pending_manual_greyscale = True
            self._set_status("Manual greyscale queued for next training step.")
            self._request_redraw()
            return 0
        out = self.apply_greyscale_once(manual_trigger=True, forced_iteration=self._current_iteration())
        self._request_redraw()
        return out

    def clear_old_mask_now(self):
        scene, model, source = self._resolve_scene_and_model(prefer_training=False, allow_combined_fallback=True, allow_node_fallback=True)
        if scene is None:
            self._set_status("No scene is available yet. Open or prepare a scene first.", level="warn")
            self._request_redraw()
            return False
        if model is None:
            self._set_status(f"No Gaussian model available yet ({source}).", level="warn")
            self._request_redraw()
            return False
        ok = self._clear_deleted_mask(model)
        if ok:
            try:
                scene.notify_changed()
            except Exception:
                pass
            self._set_status(f"Cleared soft-delete mask on {source}.")
        else:
            self._set_status(f"Failed to clear soft-delete mask on {source}.", level="warn")
        self._request_redraw()
        return ok

    def _apply_action(self, model, rule_mask, rule_cfg: dict[str, Any]):
        action_name = str(rule_cfg["action"])
        if action_name in {"none", "delete"}:
            return False
        changed = False
        if action_name in {"fade", "fade_shrink", "fade_expand"}:
            changed = self._apply_opacity_multiplier(model, rule_mask, float(rule_cfg["opacity_multiplier"])) or changed
        if action_name in {"shrink", "fade_shrink"}:
            changed = self._apply_scale_multiplier(model, rule_mask, float(rule_cfg["scale_multiplier"]), str(rule_cfg["scale_scope"]), mode="shrink") or changed
        elif action_name in {"expand", "fade_expand"}:
            changed = self._apply_scale_multiplier(model, rule_mask, float(rule_cfg["scale_multiplier"]), str(rule_cfg["scale_scope"]), mode="expand") or changed
        elif action_name == "move":
            changed = self._move_toward_center(model, rule_mask) or changed
        return bool(changed)

    def prune_once(self, manual_trigger: bool = False, forced_iteration: int | None = None):
        s = self.settings
        self.last_removed = 0
        self.last_actions = []
        iteration = int(forced_iteration if forced_iteration is not None else self._current_iteration())
        profile = self._current_profile(iteration)
        warmup_iter = self.resolve_iter_spec(getattr(s, "warmup_iters", "0")) or 0
        stop_iter = self.resolve_iter_spec(getattr(s, "stop_iters", ""))
        if not bool(s.enabled):
            self.pending_manual_prune = False
            self._set_status("Plugin is disabled.")
            return 0
        if not manual_trigger:
            if not self._is_training_active():
                self._set_status(f"Trainer state: {self._trainer_state_text() or 'idle'}")
                return 0
            if iteration < warmup_iter:
                self._set_status(f"Warmup active: iteration {iteration} < {warmup_iter}.")
                return 0
            if stop_iter is not None and iteration > stop_iter:
                self._set_status(f"Stop reached: iteration {iteration} > {stop_iter}.")
                return 0
            if iteration == self.last_pruned_iteration:
                return 0
            if iteration % max(1, int(s.apply_every)) != 0:
                self._set_status(f"Skipping iteration {iteration} (apply_every={int(s.apply_every)}).")
                return 0
        training_active = self._is_training_active()
        scene, model, source = self._resolve_scene_and_model(
            prefer_training=training_active,
            allow_combined_fallback=True,
            allow_node_fallback=True,
        )
        if scene is None:
            self._set_wait_status(
                f"Scene not ready yet for pruning at iter={iteration}; skipping until LichtFeld exposes a scene object.",
                manual_trigger=manual_trigger,
            )
            return 0
        if model is None:
            self._set_wait_status(
                f"No Gaussian model available yet for pruning at iter={iteration} ({source}).",
                manual_trigger=manual_trigger,
            )
            return 0
        if int(getattr(model, "num_points", 0)) == 0:
            self._set_wait_status(
                f"Gaussian model has zero points at iter={iteration} ({source}).",
                manual_trigger=manual_trigger,
            )
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
        if counts["radius"] + counts["max_axis"] + counts["aspect"] == 0:
            self.last_pruned_iteration = iteration
            self._set_status(f"iter={iteration}: matched 0 in {profile['label']} via {source} (radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']})")
            return 0
        actions_taken = []
        affected_union = None
        try:
            for rule_name in _RULES:
                rule_cfg = profile["rules"][rule_name]
                rule_mask = matches[rule_name]
                if self._masked_count(rule_mask) == 0 or str(rule_cfg["action"]) != "delete":
                    continue
                if self._soft_delete(model, rule_mask):
                    actions_taken.append(f"{rule_name}:delete")
                    affected_union = rule_mask if affected_union is None else (affected_union | rule_mask)
            for rule_name in _RULES:
                rule_cfg = profile["rules"][rule_name]
                rule_mask = matches[rule_name]
                if self._masked_count(rule_mask) == 0 or str(rule_cfg["action"]) == "delete":
                    continue
                if self._apply_action(model, rule_mask, rule_cfg):
                    actions_taken.append(f"{rule_name}:{rule_cfg['action']}")
                    affected_union = rule_mask if affected_union is None else (affected_union | rule_mask)
            self._sync_scene_after_edit(scene, training_active=training_active)
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
            self._set_status(f"{prefix}: matched radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']} in {profile['label']} via {source}, but no actions were applied.")
            return affected
        self._set_status(f"{prefix}: profile={profile['label']} affected={affected} via {source} (radius={counts['radius']}, max_axis={counts['max_axis']}, aspect={counts['aspect']}; thr_radius={self.last_thresholds['radius']:.4f}, thr_max_axis={self.last_thresholds['max_axis']:.6f}, thr_aspect={self.last_thresholds['aspect']:.4f}; source={self.iteration_source}); actions={', '.join(actions_taken)}")
        return affected

    def _build_condition_masks(self, model, rule_configs: dict[str, dict[str, Any]]):
        n = int(model.num_points)
        if n <= 0:
            return None, {"radius": 0, "max_axis": 0, "aspect": 0}
        any_rule = False
        counts = {"radius": 0, "max_axis": 0, "aspect": 0}
        alive = self._alive_mask(model)
        false_mask = lf.Tensor.zeros([n], dtype="bool", device=model.means_raw.device)
        matches = {"radius": false_mask.clone(), "max_axis": false_mask.clone(), "aspect": false_mask.clone()}
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
                stretch_mask = (max_axis / (min_axis + _EPS)) > float(rule_configs["aspect"]["threshold"])
                if alive is not None:
                    stretch_mask = stretch_mask & alive
                matches["aspect"] = stretch_mask
                counts["aspect"] = self._masked_count(stretch_mask)
        if bool(rule_configs["radius"]["enabled"]):
            any_rule = True
            if self.center_xyz is None:
                self._set_status("Radius matching is enabled, but center is not set yet.", level="warn")
            else:
                cx, cy, cz = self.center_xyz
                means = model.means_raw
                dist2 = (means[:, 0] - cx) ** 2 + (means[:, 1] - cy) ** 2 + (means[:, 2] - cz) ** 2
                far_mask = dist2 > float(rule_configs["radius"]["threshold"] ** 2)
                if alive is not None:
                    far_mask = far_mask & alive
                matches["radius"] = far_mask
                counts["radius"] = self._masked_count(far_mask)
        return (matches, counts) if any_rule else (None, counts)
