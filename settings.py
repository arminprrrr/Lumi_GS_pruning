import json
import os
import sys
from pathlib import Path
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
PERSIST_KEY = "guard_settings_v5_json"
LEGACY_PERSIST_KEYS = ("guard_settings_v4_json", "guard_settings_v3_json", "guard_settings_v2_json")
LOCAL_PERSIST_PATH = Path(__file__).with_name("guard_settings.json")

PLUGIN_CONFIG_SECTION_KEYS = ("lumi_gs_pruning", "lumi_guard", "lumi_gs_pruning_settings", "lumi_guard_settings")
PLUGIN_CONFIG_CONTAINER_KEYS = ("plugins", "plugin_configs", "plugin_settings")
HOST_CONFIG_SIDECAR_SUFFIXES = (".lumi_gs_pruning.json", ".lumi_guard.json", ".plugin.json")
_LAST_RUNTIME_INIT_REPORT: dict[str, Any] = {}

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
    enforce_greyscale = BoolProperty(default=False, name="Enforce greyscale", description="Continuously convert Gaussian colors to greyscale during training and when requested manually")
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
    "enforce_greyscale": {"type": "bool"},
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
    "enforce_grayscale": "enforce_greyscale",
    "grayscale": "enforce_greyscale",
    "greyscale": "enforce_greyscale",
}

LEGACY_ACTION_MAP = {"clamp": "shrink", "fade_clamp": "fade_shrink"}
LEGACY_SCALE_SCOPE_MAP = {"all_axis": "all_axes", "allaxis": "all_axes", "largest": "largest_axis"}


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
        out = str(value).strip()
        if name.endswith("_action"):
            out = LEGACY_ACTION_MAP.get(out, out)
        elif name.endswith("_scale_scope"):
            out = LEGACY_SCALE_SCOPE_MAP.get(out, out)
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



def _json_key_lookup(data: dict[str, Any]) -> dict[str, str]:
    return {str(key).strip().lower(): key for key in data.keys() if isinstance(key, str)}


def _normalize_config_blob(data: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(data, dict):
        raise ValueError("Plugin config payload must be a JSON object")
    key_lookup = _json_key_lookup(data)
    for logical_key in PLUGIN_CONFIG_SECTION_KEYS:
        actual_key = key_lookup.get(logical_key)
        if actual_key is None:
            continue
        section = data.get(actual_key)
        if isinstance(section, dict):
            return section, actual_key
    for logical_key in PLUGIN_CONFIG_CONTAINER_KEYS:
        actual_key = key_lookup.get(logical_key)
        if actual_key is None:
            continue
        container = data.get(actual_key)
        if not isinstance(container, dict):
            continue
        nested_lookup = _json_key_lookup(container)
        for nested_key in PLUGIN_CONFIG_SECTION_KEYS + (PLUGIN_NAME.lower(),):
            actual_nested = nested_lookup.get(nested_key)
            if actual_nested is None:
                continue
            section = container.get(actual_nested)
            if isinstance(section, dict):
                return section, f"{actual_key}.{actual_nested}"
    plugin_key = key_lookup.get("plugin")
    if plugin_key is not None and isinstance(data.get(plugin_key), dict):
        plugin_blob = data.get(plugin_key)
        plugin_name = str(plugin_blob.get("name", "")).strip().lower()
        plugin_settings = plugin_blob.get("settings")
        if isinstance(plugin_settings, dict) and plugin_name in {"", PLUGIN_NAME.lower(), "lumi_guard"}:
            return plugin_settings, f"{plugin_key}.settings"
        if any(CLI_ALIASES.get(str(k), str(k)) in FIELD_SPECS for k in plugin_blob.keys()):
            return plugin_blob, plugin_key
    if any(CLI_ALIASES.get(str(k), str(k)) in FIELD_SPECS for k in data.keys()):
        return data, "top_level"
    return {}, "no_plugin_fields"


def _read_json_file(path_like: os.PathLike[str] | str, source_label: str) -> tuple[dict[str, Any], str | None]:
    path = Path(path_like).expanduser()
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    payload, section = _normalize_config_blob(blob)
    if not isinstance(payload, dict):
        raise ValueError(f"{source_label} did not contain a JSON object")
    return payload, section


def _extract_host_config_path(argv: list[str]) -> str | None:
    for idx, arg in enumerate(argv):
        if not isinstance(arg, str):
            continue
        raw = arg.strip().strip('"')
        if raw.startswith("--config="):
            return raw.split("=", 1)[1].strip().strip('"')
        if raw in {"--config", "-c"} and idx + 1 < len(argv):
            return str(argv[idx + 1]).strip().strip('"')
    return None


def _sidecar_candidates_for_host_config(host_config_path: str | None) -> list[Path]:
    if not host_config_path:
        return []
    host_path = Path(host_config_path).expanduser()
    if not host_path.name:
        return []
    out: list[Path] = []
    for suffix in HOST_CONFIG_SIDECAR_SUFFIXES:
        out.append(host_path.with_name(f"{host_path.stem}{suffix}"))
    out.extend([
        host_path.with_name("Lumi_GS_pruning.json"),
        host_path.with_name("lumi_gs_pruning.json"),
        host_path.with_name("lumi_guard.json"),
    ])
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in out:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _json_dumps_compact(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)


def get_last_runtime_init_report() -> dict[str, Any]:
    return dict(_LAST_RUNTIME_INIT_REPORT)


def log_runtime_settings_summary(settings: GuardSettings, report: dict[str, Any] | None = None):
    report = dict(_LAST_RUNTIME_INIT_REPORT if report is None else report)
    source_bits = []
    if report.get("persistent_loaded"):
        source_bits.append("persistent_settings")
    source_bits.extend(str(item) for item in report.get("sources", []) if item)
    source_text = ", ".join(source_bits) if source_bits else "defaults_only"
    override_keys = report.get("override_keys") or []
    lf.log.info(f"[{PLUGIN_NAME}] Runtime config sources: {source_text}")
    if override_keys:
        lf.log.info(f"[{PLUGIN_NAME}] Runtime override keys: {', '.join(str(k) for k in override_keys)}")
    lf.log.info(f"[{PLUGIN_NAME}] Effective settings: {_json_dumps_compact(settings_to_dict(settings))}")


def collect_runtime_overrides(argv: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides: dict[str, Any] = {}
    sources: list[str] = []
    override_keys: list[str] = []

    def merge_blob(blob: dict[str, Any], source: str):
        nonlocal overrides, sources, override_keys
        if not isinstance(blob, dict) or not blob:
            return
        overrides.update(blob)
        sources.append(source)
        for key in blob.keys():
            normalized = CLI_ALIASES.get(str(key), str(key))
            if normalized not in override_keys:
                override_keys.append(normalized)

    host_config_path = _extract_host_config_path(argv)
    if host_config_path:
        try:
            blob, section = _read_json_file(host_config_path, "host config")
            if blob:
                section_label = f"#{section}" if section and section != "top_level" else ""
                merge_blob(blob, f"host_config:{host_config_path}{section_label}")
        except FileNotFoundError:
            lf.log.warn(f"[{PLUGIN_NAME}] Host config file not found: {host_config_path}")
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to inspect host config {host_config_path}: {exc}")
        for candidate in _sidecar_candidates_for_host_config(host_config_path):
            if not candidate.exists():
                continue
            try:
                blob, section = _read_json_file(candidate, "host config sidecar")
                section_label = f"#{section}" if section and section != "top_level" else ""
                merge_blob(blob, f"host_config_sidecar:{candidate}{section_label}")
                break
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to read sidecar config {candidate}: {exc}")

    prefixes = ("--lumi-gs-pruning-", "--lumi-guard-", "--lumi-")
    for arg in argv:
        extracted = _extract_prefixed_arg(arg, prefixes)
        if extracted is None:
            continue
        if extracted.startswith("no-"):
            key = _normalize_cli_name(extracted[3:])
            if key in FIELD_SPECS and FIELD_SPECS[key]["type"] == "bool":
                merge_blob({key: False}, f"cli_flag:{key}=false")
            continue
        if "=" not in extracted:
            key = _normalize_cli_name(extracted)
            if key in FIELD_SPECS and FIELD_SPECS[key]["type"] == "bool":
                merge_blob({key: True}, f"cli_flag:{key}=true")
            continue
        raw_key, raw_value = extracted.split("=", 1)
        key = _normalize_cli_name(raw_key)
        if key in {"json", "config_json"}:
            try:
                blob, section = _normalize_config_blob(json.loads(raw_value))
                section_label = f"#{section}" if section and section != "top_level" else ""
                merge_blob(blob, f"cli_json{section_label}")
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse CLI JSON overrides: {exc}")
            continue
        if key in {"file", "config_file"}:
            try:
                blob, section = _read_json_file(raw_value, "CLI config file")
                section_label = f"#{section}" if section and section != "top_level" else ""
                merge_blob(blob, f"cli_file:{raw_value}{section_label}")
            except Exception as exc:
                lf.log.warn(f"[{PLUGIN_NAME}] Failed to read CLI config file {raw_value}: {exc}")
            continue
        merge_blob({key: raw_value}, f"cli_value:{key}")

    env_json = os.environ.get("LUMI_GS_PRUNING_JSON") or os.environ.get("LUMI_GUARD_JSON")
    if env_json:
        try:
            blob, section = _normalize_config_blob(json.loads(env_json))
            section_label = f"#{section}" if section and section != "top_level" else ""
            merge_blob(blob, f"env_json{section_label}")
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse env JSON overrides: {exc}")
    env_file = os.environ.get("LUMI_GS_PRUNING_FILE") or os.environ.get("LUMI_GUARD_FILE")
    if env_file:
        try:
            blob, section = _read_json_file(env_file, "env config file")
            section_label = f"#{section}" if section and section != "top_level" else ""
            merge_blob(blob, f"env_file:{env_file}{section_label}")
        except Exception as exc:
            lf.log.warn(f"[{PLUGIN_NAME}] Failed to read env config file {env_file}: {exc}")

    return overrides, {"sources": sources, "override_keys": override_keys, "host_config_path": host_config_path}



def _load_local_payload() -> str:
    try:
        if LOCAL_PERSIST_PATH.exists():
            return LOCAL_PERSIST_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to read local settings file {LOCAL_PERSIST_PATH}: {exc}")
    return ""


def _save_local_payload(payload: str) -> bool:
    try:
        LOCAL_PERSIST_PATH.write_text(str(payload), encoding="utf-8")
        return True
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to write local settings file {LOCAL_PERSIST_PATH}: {exc}")
        return False

def load_persistent_settings(settings: GuardSettings):
    store = _get_plugin_store()
    payload = ""
    loaded_from = None
    for key in (PERSIST_KEY, *LEGACY_PERSIST_KEYS):
        payload = _store_get(store, key, "") or ""
        if payload:
            loaded_from = key
            break
    if not payload:
        payload = _load_local_payload()
        if payload:
            loaded_from = "local_file"
    if not payload:
        return False
    try:
        data = json.loads(payload)
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to parse saved settings: {exc}")
        return False
    if not isinstance(data, dict):
        lf.log.warn(f"[{PLUGIN_NAME}] Saved settings payload must be a dict")
        return False
    changed = apply_dict(settings, data, log_prefix="saved")
    if loaded_from != PERSIST_KEY:
        save_persistent_settings(settings)
    return bool(changed or data)


def save_persistent_settings(settings: GuardSettings):
    try:
        payload = json.dumps(settings_to_dict(settings), sort_keys=True)
    except Exception as exc:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to serialize settings: {exc}")
        return False
    store = _get_plugin_store()
    store_ok = _store_set(store, PERSIST_KEY, payload) if store is not None else False
    file_ok = _save_local_payload(payload)
    if store is not None and not store_ok:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to save persistent settings to plugin store")
    if not store_ok and not file_ok:
        lf.log.warn(f"[{PLUGIN_NAME}] Failed to save persistent settings")
    return bool(store_ok or file_ok)


def _normalize_cli_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _extract_prefixed_arg(arg: str, prefixes: tuple[str, ...]):
    for prefix in prefixes:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def parse_cli_overrides(argv: list[str] | None = None) -> dict[str, Any]:
    overrides, _meta = collect_runtime_overrides(argv)
    return overrides




def initialize_runtime_settings(settings: GuardSettings | None = None):
    global _LAST_RUNTIME_INIT_REPORT
    settings = settings or GuardSettings.get_instance()
    persistent_loaded = load_persistent_settings(settings)
    cli, meta = collect_runtime_overrides()
    if cli:
        apply_dict(settings, cli, log_prefix="runtime")
        save_persistent_settings(settings)
    _LAST_RUNTIME_INIT_REPORT = {
        "persistent_loaded": bool(persistent_loaded),
        "sources": list(meta.get("sources", [])),
        "override_keys": list(meta.get("override_keys", [])),
        "host_config_path": meta.get("host_config_path"),
        "effective_settings": settings_to_dict(settings),
    }
    log_runtime_settings_summary(settings, _LAST_RUNTIME_INIT_REPORT)
    return settings
