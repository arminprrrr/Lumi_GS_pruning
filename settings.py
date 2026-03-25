import json
import os
import sys
from typing import Any

import lichtfeld as lf
from lfs_plugins.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PropertyGroup,
    StringProperty,
)

PLUGIN_NAME = "Lumi_GS_pruning"
PERSIST_KEY = "guard_settings_v3_json"

RULE_KEYS = ("radius", "max_axis", "aspect")

ACTION_ITEMS = [
    ("none", "None", "Do not apply any action to matched Gaussians"),
    ("fade", "Fade", "Set matched Gaussian opacity to the interpolated opacity target"),
    ("shrink", "Shrink", "Multiply matched Gaussian scale by the interpolated shrink multiplier"),
    ("expand", "Expand", "Multiply matched Gaussian scale by the interpolated expand multiplier"),
    ("clamp", "Clamp", "Clamp matched Gaussian scale down to the interpolated clamp target"),
    ("fade_shrink", "Fade + Shrink", "Fade opacity and shrink matched Gaussian scale"),
    ("fade_expand", "Fade + Expand", "Fade opacity and expand matched Gaussian scale"),
    ("fade_clamp", "Fade + Clamp", "Fade opacity and clamp matched Gaussian scale"),
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


class PhaseSettings(PropertyGroup):
    name = StringProperty(
        default="Phase",
        maxlen=128,
        name="Phase name",
        description="Friendly label for this phase",
    )

    enabled = BoolProperty(
        default=True,
        name="Enabled",
        description="Whether this phase participates in phase-based control",
    )

    start_iter = IntProperty(
        default=0,
        min=0,
        max=100000000,
        step=100,
        name="Start iteration",
        description="Inclusive start iteration for this phase",
    )

    end_iter = IntProperty(
        default=5000,
        min=0,
        max=100000000,
        step=100,
        name="End iteration",
        description="Inclusive end iteration for this phase",
    )

    enable_radius = BoolProperty(default=False, name="Match outside radius")
    radius_start = FloatProperty(default=1.0, min=0.0, max=100000.0, step=0.1, precision=6, name="Radius start")
    radius_end = FloatProperty(default=1.0, min=0.0, max=100000.0, step=0.1, precision=6, name="Radius end")
    radius_action = EnumProperty(items=ACTION_ITEMS, default="fade", name="Radius action")
    radius_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Radius scale scope")
    radius_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Radius opacity start")
    radius_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Radius opacity end")
    radius_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius scale multiplier start")
    radius_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius scale multiplier end")
    radius_clamp_start = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius clamp start")
    radius_clamp_end = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius clamp end")

    enable_max_axis = BoolProperty(default=True, name="Match oversized Gaussians")
    max_axis_start = FloatProperty(default=0.17, min=0.0, max=100000.0, step=0.01, precision=6, name="Max axis start")
    max_axis_end = FloatProperty(default=0.17, min=0.0, max=100000.0, step=0.01, precision=6, name="Max axis end")
    max_axis_action = EnumProperty(items=ACTION_ITEMS, default="shrink", name="Oversized action")
    max_axis_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Oversized scale scope")
    max_axis_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Oversized opacity start")
    max_axis_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Oversized opacity end")
    max_axis_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized scale multiplier start")
    max_axis_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized scale multiplier end")
    max_axis_clamp_start = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized clamp start")
    max_axis_clamp_end = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized clamp end")

    enable_aspect = BoolProperty(default=False, name="Match stretched Gaussians")
    max_aspect_start = FloatProperty(default=10.0, min=1.0, max=100000.0, step=0.1, precision=6, name="Max aspect start")
    max_aspect_end = FloatProperty(default=10.0, min=1.0, max=100000.0, step=0.1, precision=6, name="Max aspect end")
    aspect_action = EnumProperty(items=ACTION_ITEMS, default="shrink", name="Stretch action")
    aspect_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Stretch scale scope")
    aspect_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Stretch opacity start")
    aspect_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Stretch opacity end")
    aspect_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch scale multiplier start")
    aspect_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch scale multiplier end")
    aspect_clamp_start = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch clamp start")
    aspect_clamp_end = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch clamp end")


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
    radius_start = FloatProperty(default=1.0, min=0.0, max=100000.0, step=0.1, precision=6, name="Radius start")
    radius_end = FloatProperty(default=1.0, min=0.0, max=100000.0, step=0.1, precision=6, name="Radius end")
    radius_action = EnumProperty(items=ACTION_ITEMS, default="fade", name="Radius action")
    radius_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Radius scale scope")
    radius_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Radius opacity start")
    radius_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Radius opacity end")
    radius_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius scale multiplier start")
    radius_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius scale multiplier end")
    radius_clamp_start = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius clamp start")
    radius_clamp_end = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Radius clamp end")

    enable_max_axis = BoolProperty(
        default=True,
        name="Match oversized Gaussians",
        description="Match when the largest axis exceeds the current max-axis threshold",
    )
    max_axis_start = FloatProperty(default=0.17, min=0.0, max=100000.0, step=0.01, precision=6, name="Max axis start")
    max_axis_end = FloatProperty(default=0.17, min=0.0, max=100000.0, step=0.01, precision=6, name="Max axis end")
    max_axis_action = EnumProperty(items=ACTION_ITEMS, default="shrink", name="Oversized action")
    max_axis_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Oversized scale scope")
    max_axis_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Oversized opacity start")
    max_axis_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Oversized opacity end")
    max_axis_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized scale multiplier start")
    max_axis_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized scale multiplier end")
    max_axis_clamp_start = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized clamp start")
    max_axis_clamp_end = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Oversized clamp end")

    enable_aspect = BoolProperty(
        default=False,
        name="Match stretched Gaussians",
        description="Match when aspect ratio exceeds the current aspect threshold",
    )
    max_aspect_start = FloatProperty(default=10.0, min=1.0, max=100000.0, step=0.1, precision=6, name="Max aspect start")
    max_aspect_end = FloatProperty(default=10.0, min=1.0, max=100000.0, step=0.1, precision=6, name="Max aspect end")
    aspect_action = EnumProperty(items=ACTION_ITEMS, default="shrink", name="Stretch action")
    aspect_scale_scope = EnumProperty(items=SCALE_SCOPE_ITEMS, default="largest_axis", name="Stretch scale scope")
    aspect_opacity_start = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Stretch opacity start")
    aspect_opacity_end = FloatProperty(default=0.10, min=0.000001, max=1.0, step=0.01, precision=6, name="Stretch opacity end")
    aspect_scale_multiplier_start = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch scale multiplier start")
    aspect_scale_multiplier_end = FloatProperty(default=0.50, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch scale multiplier end")
    aspect_clamp_start = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch clamp start")
    aspect_clamp_end = FloatProperty(default=1.0, min=0.000001, max=1000000.0, step=0.01, precision=6, name="Stretch clamp end")

    use_phases = BoolProperty(
        default=False,
        name="Use phases",
        description="Use the per-phase parameter list instead of the global rule settings",
    )

    phases = CollectionProperty(
        type=PhaseSettings,
        name="Phases",
        description="Phase-specific rule settings",
    )

    log_each_hit = BoolProperty(
        default=True,
        name="Log updates",
        description="Write activity to the LichtFeld log",
    )


ROOT_FIELD_SPECS: dict[str, dict[str, Any]] = {
    "enabled": {"type": "bool"},
    "warmup_iters": {"type": "int", "min": 0, "max": 10000000},
    "apply_every": {"type": "int", "min": 1, "max": 100000},
    "center_mode": {"type": "enum", "items": {k for k, _, _ in CENTER_MODE_ITEMS}},
    "center": {"type": "float3", "min": -100000.0, "max": 100000.0},
    "log_each_hit": {"type": "bool"},
    "use_phases": {"type": "bool"},
}

_PHASE_FIELD_BASE: dict[str, dict[str, Any]] = {
    "name": {"type": "string"},
    "enabled": {"type": "bool"},
    "start_iter": {"type": "int", "min": 0, "max": 100000000},
    "end_iter": {"type": "int", "min": 0, "max": 100000000},
}

_RULE_FIELD_SPECS: dict[str, dict[str, Any]] = {
    "enable_{rule}": {"type": "bool"},
    "{rule}_start": {"type": "float", "min": 0.0, "max": 100000.0},
    "{rule}_end": {"type": "float", "min": 0.0, "max": 100000.0},
    "{rule}_action": {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}},
    "{rule}_scale_scope": {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}},
    "{rule}_opacity_start": {"type": "float", "min": 0.000001, "max": 1.0},
    "{rule}_opacity_end": {"type": "float", "min": 0.000001, "max": 1.0},
    "{rule}_scale_multiplier_start": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "{rule}_scale_multiplier_end": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "{rule}_clamp_start": {"type": "float", "min": 0.000001, "max": 1000000.0},
    "{rule}_clamp_end": {"type": "float", "min": 0.000001, "max": 1000000.0},
}

THRESHOLD_ATTRS = {
    "radius": ("radius_start", "radius_end"),
    "max_axis": ("max_axis_start", "max_axis_end"),
    "aspect": ("max_aspect_start", "max_aspect_end"),
}

ACTION_ATTRS = {
    "radius": ("radius_action", "radius_scale_scope", "radius_opacity_start", "radius_opacity_end", "radius_scale_multiplier_start", "radius_scale_multiplier_end", "radius_clamp_start", "radius_clamp_end"),
    "max_axis": ("max_axis_action", "max_axis_scale_scope", "max_axis_opacity_start", "max_axis_opacity_end", "max_axis_scale_multiplier_start", "max_axis_scale_multiplier_end", "max_axis_clamp_start", "max_axis_clamp_end"),
    "aspect": ("aspect_action", "aspect_scale_scope", "aspect_opacity_start", "aspect_opacity_end", "aspect_scale_multiplier_start", "aspect_scale_multiplier_end", "aspect_clamp_start", "aspect_clamp_end"),
}

PHASE_FIELD_SPECS: dict[str, dict[str, Any]] = dict(_PHASE_FIELD_BASE)
for _rule, (_start_name, _end_name) in THRESHOLD_ATTRS.items():
    min_threshold = 1.0 if _rule == "aspect" else 0.0
    PHASE_FIELD_SPECS[f"enable_{_rule}"] = {"type": "bool"}
    PHASE_FIELD_SPECS[_start_name] = {"type": "float", "min": min_threshold, "max": 100000.0}
    PHASE_FIELD_SPECS[_end_name] = {"type": "float", "min": min_threshold, "max": 100000.0}
for _rule, names in ACTION_ATTRS.items():
    action_name, scope_name, opacity_start_name, opacity_end_name, mult_start_name, mult_end_name, clamp_start_name, clamp_end_name = names
    PHASE_FIELD_SPECS[action_name] = {"type": "enum", "items": {k for k, _, _ in ACTION_ITEMS}}
    PHASE_FIELD_SPECS[scope_name] = {"type": "enum", "items": {k for k, _, _ in SCALE_SCOPE_ITEMS}}
    PHASE_FIELD_SPECS[opacity_start_name] = {"type": "float", "min": 0.000001, "max": 1.0}
    PHASE_FIELD_SPECS[opacity_end_name] = {"type": "float", "min": 0.000001, "max": 1.0}
    PHASE_FIELD_SPECS[mult_start_name] = {"type": "float", "min": 0.000001, "max": 1000000.0}
    PHASE_FIELD_SPECS[mult_end_name] = {"type": "float", "min": 0.000001, "max": 1000000.0}
    PHASE_FIELD_SPECS[clamp_start_name] = {"type": "float", "min": 0.000001, "max": 1000000.0}
    PHASE_FIELD_SPECS[clamp_end_name] = {"type": "float", "min": 0.000001, "max": 1000000.0}

FIELD_SPECS: dict[str, dict[str, Any]] = dict(ROOT_FIELD_SPECS)
for _name, _spec in PHASE_FIELD_SPECS.items():
    if _name in {"name", "start_iter", "end_iter", "enabled"}:
        continue
    FIELD_SPECS[_name] = _spec
FIELD_SPECS["phases"] = {"type": "phase_list"}

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


def _coerce_field(name: str, value: Any, specs: dict[str, dict[str, Any]] | None = None):
    specs = FIELD_SPECS if specs is None else specs
    spec = specs[name]
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

    if field_type == "string":
        return str(value)

    raise ValueError(f"Unsupported field type for {name}: {field_type}")


def _serialize_group(group: PropertyGroup, specs: dict[str, dict[str, Any]]):
    return {name: getattr(group, name) for name in specs}


def _apply_spec_dict(group: PropertyGroup, data: dict[str, Any], specs: dict[str, dict[str, Any]], log_prefix: str):
    changed = False
    for raw_name, raw_value in data.items():
        alias_name = CLI_ALIASES.get(raw_name, raw_name)
        name = alias_name
        if name not in specs:
            lf.log.warn(f"[{PLUGIN_NAME}] Unknown {log_prefix} field ignored: {raw_name}")
            continue
        try:
            value = _coerce_field(name, raw_value, specs)
            if getattr(group, name) != value:
                setattr(group, name, value)
                changed = True
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to apply {log_prefix} field {raw_name}: {exc}")
    return changed


def phase_to_dict(phase: PhaseSettings) -> dict[str, Any]:
    return _serialize_group(phase, PHASE_FIELD_SPECS)


def settings_to_dict(settings: GuardSettings) -> dict[str, Any]:
    out = _serialize_group(settings, {k: v for k, v in FIELD_SPECS.items() if k != "phases"})
    out["phases"] = [phase_to_dict(phase) for phase in settings.phases]
    return out


def apply_phase_dict(phase: PhaseSettings, data: dict[str, Any], log_prefix: str = "phase"):
    return _apply_spec_dict(phase, data, PHASE_FIELD_SPECS, log_prefix)


def clear_phases(settings: GuardSettings):
    settings.phases.clear()


def add_phase(settings: GuardSettings, name: str | None = None, start_iter: int = 0, end_iter: int = 0):
    phase = settings.phases.add()
    phase.name = str(name or f"Phase {len(settings.phases)}")
    phase.start_iter = int(max(0, start_iter))
    phase.end_iter = int(max(phase.start_iter, end_iter))
    copy_global_rules_to_phase(settings, phase)
    settings.use_phases = True
    return phase


def replace_phases(settings: GuardSettings, phase_data: list[dict[str, Any]], log_prefix: str = "phases"):
    settings.phases.clear()
    changed = False
    for idx, entry in enumerate(phase_data):
        if not isinstance(entry, dict):
            lf.log.warn(f"[{PLUGIN_NAME}] Ignoring non-object {log_prefix}[{idx}]")
            continue
        phase = settings.phases.add()
        apply_phase_dict(phase, entry, log_prefix=f"{log_prefix}[{idx}]")
        changed = True
    return changed


def copy_global_rules_to_phase(settings: GuardSettings, phase: PhaseSettings):
    for name in PHASE_FIELD_SPECS:
        if hasattr(settings, name):
            setattr(phase, name, getattr(settings, name))
    return phase


def build_three_phase_30k_scaffold(settings: GuardSettings):
    settings.phases.clear()

    p1 = settings.phases.add()
    p1.name = "Phase 1"
    p1.start_iter = 0
    p1.end_iter = 5000
    copy_global_rules_to_phase(settings, p1)

    p2 = settings.phases.add()
    p2.name = "Phase 2"
    p2.start_iter = 5001
    p2.end_iter = 20000
    copy_global_rules_to_phase(settings, p2)

    p3 = settings.phases.add()
    p3.name = "Phase 3"
    p3.start_iter = 20001
    p3.end_iter = 30000
    copy_global_rules_to_phase(settings, p3)

    settings.use_phases = True
    return settings.phases


def apply_dict(settings: GuardSettings, data: dict[str, Any], log_prefix: str = "settings"):
    changed = _apply_spec_dict(
        settings,
        {k: v for k, v in data.items() if k != "phases"},
        {k: v for k, v in FIELD_SPECS.items() if k != "phases"},
        log_prefix,
    )

    if "phases" in data:
        raw_phases = data.get("phases")
        if isinstance(raw_phases, list):
            changed = replace_phases(settings, raw_phases, log_prefix=f"{log_prefix}.phases") or changed
        else:
            lf.log.warn(f"[{PLUGIN_NAME}] Ignoring non-list {log_prefix}.phases")

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
        legacy_payload = _store_get(store, "guard_settings_v2_json", "")
        payload = legacy_payload or ""
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
