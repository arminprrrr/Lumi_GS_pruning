import json
import os
import sys
from typing import Any

import lichtfeld as lf
from lfs_plugins.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PropertyGroup,
    StringProperty,
)

PLUGIN_NAME = "Lumi_GS_pruning"
PERSIST_KEY = "guard_settings_v4_json"
LEGACY_PERSIST_KEYS = ("guard_settings_v4_json", "guard_settings_v3_json", "guard_settings_v2_json")

ACTION_ITEMS = [
    ("none", "None", "Do not apply any action to matched Gaussians"),
    ("fade", "Fade", "Multiply matched Gaussian opacity by the interpolated opacity multiplier"),
    ("shrink", "Shrink", "Multiply matched Gaussian scale by the interpolated shrink multiplier"),
    ("expand", "Expand", "Multiply matched Gaussian scale by the interpolated expand multiplier"),
    ("fade_shrink", "Fade + Shrink", "Multiply matched Gaussian opacity and shrink matched Gaussian scale"),
    ("fade_expand", "Fade + Expand", "Multiply matched Gaussian opacity and expand matched Gaussian scale"),
    ("delete", "Delete", "Soft-delete matched Gaussians"),
    ("move", "Move", "Move matched Gaussians toward the center"),
]

CENTER_MODE_ITEMS = [
    ("auto_once", "Auto once", "Capture center once from current Gaussians"),
    ("manual", "Manual", "Use the manual center below"),
]

SCALE_SCOPE_ITEMS = [
    ("largest_axis", "Largest axis", "Apply scale edits only to the largest axis"),
    ("all_axes", "All axes", "Apply scale edits uniformly to all axes"),
]


class GuardSettings(PropertyGroup):
    enabled = BoolProperty(default=True, name="Enabled", description="Enable automatic Gaussian suppression during training")
    warmup_iters = StringProperty(default="0", maxlen=64, name="Warmup iterations", description="Accepts 5000 or 10%")
    stop_iters = StringProperty(default="", maxlen=64, name="Stop iterations", description="Accepts 25000 or 80%")
    apply_every = IntProperty(default=1, min=1, max=100000, step=1, name="Apply every N steps", description="1 = run every iteration")
    center_mode = EnumProperty(items=CENTER_MODE_ITEMS, default="auto_once", name="Center mode", description="How the center is chosen")
    center = FloatVectorProperty(default=(0.0, 0.0, 0.0), size=3, min=-100000.0, max=100000.0, subtype="XYZ", name="Center XYZ", description="Manual center in world space")

    enable_radius = BoolProperty(default=False, name="Match outside radius", description="Match Gaussians farther than the current radius from the center")
    radius_start = FloatProperty(default=1.0, min=0.0, max=100000.0, step=0.1, precision=6, name="Radius start")
    radius_end = FloatProperty(default=1.0, min=0.0, max=100000.0, step=0.1, precision=6, name="Radius end")
    radius_action = EnumProperty(items=ACTION_ITEMS, default="fade", name="Radius action")
    radius_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Radius scale scope")
    radius_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Radius opacity multiplier start")
    radius_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Radius opacity multiplier end")
    radius_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius scale multiplier start")
    radius_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius scale multiplier end")

    enable_max_axis = BoolProperty(default=True, name="Match oversized Gaussians", description="Match when the largest axis exceeds the current max-axis threshold")
    max_axis_start = FloatProperty(default=0.17, min=0.0, max=100000.0, step=0.01, precision=6, name="Max axis start")
    max_axis_end = FloatProperty(default=0.17, min=0.0, max=100000.0, step=0.01, precision=6, name="Max axis end")
    max_axis_action = EnumProperty(items=ACTION_ITEMS, default="shrink", name="Oversized action")
    max_axis_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Oversized scale scope")
    max_axis_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Oversized opacity multiplier start")
    max_axis_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Oversized opacity multiplier end")
    max_axis_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized scale multiplier start")
    max_axis_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized scale multiplier end")

    enable_aspect = BoolProperty(default=False, name="Match stretched Gaussians", description="Match when aspect ratio exceeds the current aspect threshold")
    max_aspect_start = FloatProperty(default=10.0, min=1.0, max=100000.0, step=0.1, precision=6, name="Max aspect start")
    max_aspect_end = FloatProperty(default=10.0, min=1.0, max=100000.0, step=0.1, precision=6, name="Max aspect end")
    aspect_action = EnumProperty(items=ACTION_ITEMS, default="shrink", name="Stretch action")
    aspect_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Stretch scale scope")
    aspect_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Stretch opacity multiplier start")
    aspect_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Stretch opacity multiplier end")
    aspect_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch scale multiplier start")
    aspect_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch scale multiplier end")

    log_each_hit = BoolProperty(default=True, name="Log updates", description="Write activity to the LichtFeld log")


FIELD_SPECS = {
    "enabled": {"type": "bool"},
    "warmup_iters": {"type": "string"},
    "stop_iters": {"type": "string"},
    "apply_every": {"type": "int", "min": 1, "max": 100000},
    "center_mode": {"type": "enum", "items": {k for k, _, _ in CENTER_MODE_ITEMS}},
    "center": {"type": "float3", "min": -100000.0, "max": 100000.0},
    "log_each_hit": {"type": "bool"},
    "enable_radius": {"type": "bool"},
    "radius_start": {"type": "float", "min": 0.0, "max": 100000.0},
    "radius_end": {"type": "float", "min": 0.0, "max": 100000.0},
    "radius_action": {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}},
    "radius_scale_scope": {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}},
    "radius_opacity_start": {"type": "float", "min": 0.000001, "max": 1.0},
    "radius_opacity_end": {"type": "float", "min": 0.000001, "max": 1.0},
    "radius_scale_multiplier_start": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "radius_scale_multiplier_end": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "enable_max_axis": {"type": "bool"},
    "max_axis_start": {"type": "float", "min": 0.0, "max": 100000.0},
    "max_axis_end": {"type": "float", "min": 0.0, "max": 100000.0},
    "max_axis_action": {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}},
    "max_axis_scale_scope": {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}},
    "max_axis_opacity_start": {"type": "float", "min": 0.000001, "max": 1.0},
    "max_axis_opacity_end": {"type": "float", "min": 0.000001, "max": 1.0},
    "max_axis_scale_multiplier_start": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "max_axis_scale_multiplier_end": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "enable_aspect": {"type": "bool"},
    "max_aspect_start": {"type": "float", "min": 1.0, "max": 100000.0},
    "max_aspect_end": {"type": "float", "min": 1.0, "max": 100000.0},
    "aspect_action": {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}},
    "aspect_scale_scope": {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}},
    "aspect_opacity_start": {"type": "float", "min": 0.000001, "max": 1.0},
    "aspect_opacity_end": {"type": "float", "min": 0.000001, "max": 1.0},
    "aspect_scale_multiplier_start": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "aspect_scale_multiplier_end": {"type": "float", "min": 0.000001, "max": 1000000.0},
}

CLI_ALIASES = {
    "radius": "radius_start",
    "max_axis": "max_axis_start",
    "max_aspect": "max_aspect_start",
    "action_radius": "radius_action",
    "action_max_axis": "max_axis_action",
    "action_aspect": "aspect_action",
    "radius_scope": "radius_scale_scope",
    "max_axis_scope": "max_axis_scale_scope",
    "aspect_scope": "aspect_scale_scope",
}

LEGACY_ACTION_MAP = {"clamp": "shrink", "fade_clamp": "fade_shrink"}


def _clamp(value: float, lo: float | None = None, hi: float | None = None):
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _coerce_field(name: str, value: Any):
    spec = FIELD_SPECS[name]
    field_type = spec["type"]
    if field_type == "bool":
        return _parse_bool(value)
    if field_type == "int":
        return int(_clamp(int(value), spec.get("min"), spec.get("max")))
    if field_type == "float":
        return float(_clamp(float(value), spec.get("min"), spec.get("max")))
    if field_type == "enum":
        out = LEGACY_ACTION_MAP.get(str(value).strip(), str(value).strip()) if name.endswith("_action") else str(value).strip()
        if out not in spec["items"]:
            raise ValueError(f"Invalid value for {name}: {value}")
        return out
    if field_type == "float3":
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()] if isinstance(value, str) else list(value)
        if len(parts) != 3:
            raise ValueError(f"{name} requires 3 values")
        lo = spec.get("min")
        hi = spec.get("max")
        return tuple(float(_clamp(float(v), lo, hi)) for v in parts)
    if field_type == "string":
        return str(value).strip()
    raise ValueError(f"Unsupported field type for {name}: {field_type}")


def settings_to_dict(settings: GuardSettings) -> dict[str, Any]:
    return {name: getattr(settings, name) for name in FIELD_SPECS}


def apply_dict(settings: GuardSettings, data: dict[str, Any], log_prefix: str = "settings"):
    changed = False
    for raw_name, raw_value in data.items():
        name = CLI_ALIASES.get(raw_name, raw_name)
        if name not in FIELD_SPECS:
            if raw_name in {"phases", "use_phases"} or raw_name.endswith("_clamp_start") or raw_name.endswith("_clamp_end"):
                continue
            lf.log.warn(f"[{PLUGIN_NAME}] Unknown {log_prefix} field ignored: {raw_name}")
            continue
        try:
            value = _coerce_field(name, raw_value)
            if getattr(settings, name) != value:
                setattr(settings, name, value)
                changed = True
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to apply {log_prefix} field {raw_name}: {exc}")
    return changed


def _get_plugin_store():
    try:
        return lf.plugins.settings(PLUGIN_NAME)
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Persistent settings unavailable: {exc}")
        return None


def _store_get(store, key: str, default=None):
    if store is None:
        return default
    try:
        getter = getattr(store, "get", None)
        if callable(getter):
            return getter(key, default)
    except Exception:
        pass
    try:
        return store[key]
    except Exception:
        pass
    try:
        return getattr(store, key)
    except Exception:
        return default


def _store_set(store, key: str, value):
    if store is None:
        return False
    try:
        setter = getattr(store, "set", None)
        if callable(setter):
            setter(key, value)
            return True
    except Exception:
        pass
    try:
        store[key] = value
        return True
    except Exception:
        pass
    try:
        setattr(store, key, value)
        return True
    except Exception:
        return False


def load_persistent_settings(settings: GuardSettings):
    store = _get_plugin_store()
    payload = ""
    for key in LEGACY_PERSIST_KEYS:
        payload = _store_get(store, key, "") or ""
        if payload:
            break
    if not payload:
        return False
    try:
        data = json.loads(payload)
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse saved settings: {exc}")
        return False
    return apply_dict(settings, data, log_prefix="saved")


def save_persistent_settings(settings: GuardSettings):
    store = _get_plugin_store()
    if store is None:
        return False
    try:
        payload = json.dumps(settings_to_dict(settings))
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to serialize settings: {exc}")
        return False
    ok = _store_set(store, PERSIST_KEY, payload)
    if not ok:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to save persistent settings")
    return ok


def _normalize_cli_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _extract_prefixed_arg(arg: str, prefixes: tuple[str, ...]):
    for prefix in prefixes:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def parse_cli_overrides(argv: list[str] | None = None) -> dict[str, Any]:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides: dict[str, Any] = {}
    prefixes = ("--lumi-gs-pruning-", "--lumi-guard-", "--lumi-")
    for arg in argv:
        extracted = _extract_prefixed_arg(arg, prefixes)
        if extracted is None:
            continue
        if extracted.startswith("no-"):
            key = _normalize_cli_name(extracted[3:])
            if key in FIELD_SPECS and FIELD_SPECS[key]["type"] == "bool":
                overrides[key] = False
            continue
        if "=" not in extracted:
            key = _normalize_cli_name(extracted)
            if key in FIELD_SPECS and FIELD_SPECS[key]["type"] == "bool":
                overrides[key] = True
            continue
        raw_key, raw_value = extracted.split("=", 1)
        key = _normalize_cli_name(raw_key)
        if key in {"json", "config_json"}:
            try:
                blob = json.loads(raw_value)
                if isinstance(blob, dict):
                    overrides.update(blob)
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse CLI JSON overrides: {exc}")
            continue
        if key in {"file", "config_file"}:
            try:
                with open(raw_value, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                if isinstance(blob, dict):
                    overrides.update(blob)
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to read CLI config file {raw_value}: {exc}")
            continue
        overrides[key] = raw_value
    env_json = os.environ.get("LUMI_GS_PRUNING_JSON") or os.environ.get("LUMI_GUARD_JSON")
    if env_json:
        try:
            blob = json.loads(env_json)
            if isinstance(blob, dict):
                overrides.update(blob)
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse env JSON overrides: {exc}")
    env_file = os.environ.get("LUMI_GS_PRUNING_FILE") or os.environ.get("LUMI_GUARD_FILE")
    if env_file:
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if isinstance(blob, dict):
                overrides.update(blob)
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to read env config file {env_file}: {exc}")
    return overrides


def initialize_runtime_settings(settings: GuardSettings | None = None):
    settings = settings or GuardSettings.get_instance()
    load_persistent_settings(settings)
    cli = parse_cli_overrides()
    if cli:
        apply_dict(settings, cli, log_prefix="CLI")
        save_persistent_settings(settings)
    return settings
