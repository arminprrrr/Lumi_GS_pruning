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
)

PLUGIN_NAME = "Lumi_GS_pruning"
PERSIST_KEY = "guard_settings_v2_json"

ACTION_ITEMS = [
    ("shrink", "Shrink", "Shrink matched Gaussians"),
    ("fade", "Fade", "Fade matched Gaussians"),
    ("delete", "Delete", "Soft-delete matched Gaussians"),
    ("move", "Move", "Move matched Gaussians toward the center"),
]

CENTER_MODE_ITEMS = [
    ("auto_once", "Auto once", "Capture center once from current Gaussians"),
    ("manual", "Manual", "Use the manual center below"),
]

SCALE_SCOPE_ITEMS = [
    ("largest_axis", "Largest axis", "Apply shrink only to the largest axis"),
    ("all_axes", "All axes", "Apply shrink uniformly to all axes"),
]


class GuardSettings(PropertyGroup):
    enabled = BoolProperty(
        default=True,
        name="Enabled",
        description="Enable automatic Gaussian suppression during training",
    )

    warmup_iters = IntProperty(
        default=0,
        min=0,
        max=10000000,
        step=10,
        name="Warmup iterations",
        description="Do not apply any action before this iteration",
    )

    apply_every = IntProperty(
        default=1,
        min=1,
        max=100000,
        step=1,
        name="Apply every N steps",
        description="1 = run every iteration",
    )

    center_mode = EnumProperty(
        items=CENTER_MODE_ITEMS,
        default="auto_once",
        name="Center mode",
        description="How the center is chosen",
    )

    center = FloatVectorProperty(
        default=(0.0, 0.0, 0.0),
        size=3,
        min=-100000.0,
        max=100000.0,
        subtype="XYZ",
        name="Center XYZ",
        description="Manual center in world space",
    )

    enable_radius = BoolProperty(
        default=False,
        name="Match outside radius",
        description="Match Gaussians farther than the current radius from the center",
    )

    radius_start = FloatProperty(
        default=1.0,
        min=0.0,
        max=100000.0,
        step=0.1,
        precision=4,
        name="Radius start",
        description="Radius threshold at the beginning of training",
    )

    radius_end = FloatProperty(
        default=1.0,
        min=0.0,
        max=100000.0,
        step=0.1,
        precision=4,
        name="Radius end",
        description="Radius threshold at the final iteration of training",
    )

    radius_action = EnumProperty(
        items=ACTION_ITEMS,
        default="fade",
        name="Radius action",
        description="Action for Gaussians matched by the radius condition",
    )

    enable_max_axis = BoolProperty(
        default=True,
        name="Match oversized Gaussians",
        description="Match when the largest axis exceeds the current max-axis threshold",
    )

    max_axis_start = FloatProperty(
        default=0.17,
        min=0.0,
        max=100000.0,
        step=0.01,
        precision=6,
        name="Max axis start",
        description="Max-axis threshold at the beginning of training",
    )

    max_axis_end = FloatProperty(
        default=0.17,
        min=0.0,
        max=100000.0,
        step=0.01,
        precision=6,
        name="Max axis end",
        description="Max-axis threshold at the final iteration of training",
    )

    max_axis_action = EnumProperty(
        items=ACTION_ITEMS,
        default="shrink",
        name="Oversized action",
        description="Action for Gaussians matched by the max-axis condition",
    )

    enable_aspect = BoolProperty(
        default=False,
        name="Match stretched Gaussians",
        description="Match when aspect ratio exceeds the current aspect threshold",
    )

    max_aspect_start = FloatProperty(
        default=10.0,
        min=1.0,
        max=100000.0,
        step=0.1,
        precision=4,
        name="Max aspect start",
        description="Aspect threshold at the beginning of training",
    )

    max_aspect_end = FloatProperty(
        default=10.0,
        min=1.0,
        max=100000.0,
        step=0.1,
        precision=4,
        name="Max aspect end",
        description="Aspect threshold at the final iteration of training",
    )

    aspect_action = EnumProperty(
        items=ACTION_ITEMS,
        default="shrink",
        name="Stretch action",
        description="Action for Gaussians matched by the aspect condition",
    )

    opacity_target = FloatProperty(
        default=0.10,
        min=0.000001,
        max=1.0,
        step=0.01,
        precision=4,
        name="Opacity target",
        description="Opacity target for fade actions",
    )

    scale_multiplier = FloatProperty(
        default=0.50,
        min=0.000001,
        max=1.0,
        step=0.01,
        precision=4,
        name="Scale multiplier",
        description="Multiplier used by generic shrink actions",
    )

    radius_scale_scope = EnumProperty(
        items=SCALE_SCOPE_ITEMS,
        default="largest_axis",
        name="Radius scale scope",
        description="Shrink scope for the radius condition",
    )

    max_axis_scale_scope = EnumProperty(
        items=SCALE_SCOPE_ITEMS,
        default="largest_axis",
        name="Oversized scale scope",
        description="Shrink scope for the oversized condition",
    )

    aspect_scale_scope = EnumProperty(
        items=SCALE_SCOPE_ITEMS,
        default="largest_axis",
        name="Stretch scale scope",
        description="Shrink scope for the stretched condition",
    )

    log_each_hit = BoolProperty(
        default=True,
        name="Log updates",
        description="Write activity to the LichtFeld log",
    )


FIELD_SPECS: dict[str, dict[str, Any]] = {
    "enabled": {"type": "bool"},
    "warmup_iters": {"type": "int", "min": 0, "max": 10000000},
    "apply_every": {"type": "int", "min": 1, "max": 100000},
    "center_mode": {"type": "enum", "items": {k for k, _, _ in CENTER_MODE_ITEMS}},
    "center": {"type": "float3", "min": -100000.0, "max": 100000.0},
    "enable_radius": {"type": "bool"},
    "radius_start": {"type": "float", "min": 0.0, "max": 100000.0},
    "radius_end": {"type": "float", "min": 0.0, "max": 100000.0},
    "radius_action": {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}},
    "enable_max_axis": {"type": "bool"},
    "max_axis_start": {"type": "float", "min": 0.0, "max": 100000.0},
    "max_axis_end": {"type": "float", "min": 0.0, "max": 100000.0},
    "max_axis_action": {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}},
    "enable_aspect": {"type": "bool"},
    "max_aspect_start": {"type": "float", "min": 1.0, "max": 100000.0},
    "max_aspect_end": {"type": "float", "min": 1.0, "max": 100000.0},
    "aspect_action": {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}},
    "opacity_target": {"type": "float", "min": 0.000001, "max": 1.0},
    "scale_multiplier": {"type": "float", "min": 0.000001, "max": 1.0},
    "radius_scale_scope": {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}},
    "max_axis_scale_scope": {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}},
    "aspect_scale_scope": {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}},
    "log_each_hit": {"type": "bool"},
}

CLI_ALIASES: dict[str, str] = {
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
        out = int(value)
        return int(_clamp(out, spec.get("min"), spec.get("max")))

    if field_type == "float":
        out = float(value)
        return float(_clamp(out, spec.get("min"), spec.get("max")))

    if field_type == "enum":
        out = str(value)
        if out not in spec["items"]:
            raise ValueError(f"Invalid value for {name}: {value}")
        return out

    if field_type == "float3":
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        else:
            parts = list(value)
        if len(parts) != 3:
            raise ValueError(f"{name} requires 3 values")
        lo = spec.get("min")
        hi = spec.get("max")
        return tuple(float(_clamp(float(v), lo, hi)) for v in parts)

    raise ValueError(f"Unsupported field type for {name}: {field_type}")


def settings_to_dict(settings: GuardSettings) -> dict[str, Any]:
    return {name: getattr(settings, name) for name in FIELD_SPECS}


def apply_dict(settings: GuardSettings, data: dict[str, Any], log_prefix: str = "settings"):
    changed = False
    for raw_name, raw_value in data.items():
        alias_name = CLI_ALIASES.get(raw_name, raw_name)
        if alias_name == "scale_scope":
            try:
                value = _coerce_field("radius_scale_scope", raw_value)
                for target_name in ("radius_scale_scope", "max_axis_scale_scope", "aspect_scale_scope"):
                    if getattr(settings, target_name) != value:
                        setattr(settings, target_name, value)
                        changed = True
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to apply {log_prefix} field {raw_name}: {exc}")
            continue

        name = alias_name
        if name not in FIELD_SPECS:
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
    payload = _store_get(store, PERSIST_KEY, "")
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
            return arg[len(prefix) :]
    return None


def parse_cli_overrides(argv: list[str] | None = None) -> dict[str, Any]:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides: dict[str, Any] = {}

    prefixes = (
        "--lumi-gs-pruning-",
        "--lumi-guard-",
        "--lumi-",
    )

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
