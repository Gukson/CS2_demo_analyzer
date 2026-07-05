from __future__ import annotations

import hashlib
import inspect
import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TICK_PROPS = [
    "X",
    "Y",
    "Z",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "pitch",
    "yaw",
    "team_num",
    "active_weapon",
    "weapon_class",
    "health",
    "armor_value",
    "is_alive",
    "life_state",
    "is_defusing",
    "death_time",
    "balance",
    "start_balance",
    "cash_spent_this_round",
    "total_cash_spent",
    "round_start_equip_value",
    "current_equip_value",
    "inventory",
    "inventory_as_ids",
    "has_helmet",
    "has_defuser",
    "has_rifle",
    "has_awp_or_scout",
    "has_armor",
    "has_armor_helmet",
]

MINIMAL_EVENTS = [
    "round_start",
    "round_end",
    "round_end_verbose",
    "round_officially_ended",
    "player_death",
    "bomb_planted",
    "bomb_defused",
    "bomb_exploded",
]

EXTRA_EVENTS = [
    "player_hurt",
    "weapon_fire",
    "item_pickup",
    "grenade_thrown",
    "hegrenade_detonate",
    "flashbang_detonate",
    "smokegrenade_detonate",
    "molotov_detonate",
    "inferno_startburn",
    "inferno_expire",
]


def _load_demoparser() -> Any:
    try:
        from demoparser2 import DemoParser
    except ImportError as exc:
        raise RuntimeError(
            "demoparser2 is required for parsing. Install with `pip install '.[parser]'` "
            "or `pip install demoparser2`."
        ) from exc
    return DemoParser


def _sanitize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        try:
            return _sanitize_scalar(value.item())
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_scalar(v) for k, v in value.items()}
    return str(value)


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _sanitize_scalar(v) for k, v in record.items()}


def _to_records(obj: Any) -> list[dict[str, Any]]:
    if obj is None:
        return []
    if hasattr(obj, "to_dict"):
        try:
            return [sanitize_record(r) for r in obj.to_dict("records")]
        except TypeError:
            pass
    if isinstance(obj, list):
        return [sanitize_record(dict(r)) for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if not keys:
            return []
        values = list(obj.values())
        if all(isinstance(v, list) for v in values):
            rows = []
            for idx in range(max(len(v) for v in values)):
                rows.append({k: values[pos][idx] if idx < len(values[pos]) else None for pos, k in enumerate(keys)})
            return [sanitize_record(r) for r in rows]
        return [sanitize_record(obj)]
    return []


def _call_compatible(obj: Any, method_names: list[str], *args: Any, **kwargs: Any) -> Any:
    for name in method_names:
        method = getattr(obj, name, None)
        if method is None:
            continue
        try:
            return method(*args, **kwargs)
        except TypeError:
            try:
                sig = inspect.signature(method)
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return method(*args, **filtered)
            except Exception:
                continue
    raise AttributeError(f"None of these parser methods exist or accepted the call: {method_names}")


def parse_demo_header(demo_path: str | Path) -> dict[str, Any]:
    DemoParser = _load_demoparser()
    parser = DemoParser(str(demo_path))
    try:
        header = _call_compatible(parser, ["parse_header", "parse_demo_header", "get_header"])
    except Exception as exc:
        log.warning("Could not parse demo header for %s: %s", demo_path, exc)
        header = {}
    if not isinstance(header, dict):
        header = getattr(header, "__dict__", {"raw_header": str(header)})
    header = sanitize_record(header)
    header.setdefault("demo_path", str(demo_path))
    header.setdefault("file_sha1_1mb", _file_prefix_sha1(demo_path))
    return header


def _file_prefix_sha1(path: str | Path, size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as fh:
        digest.update(fh.read(size))
    return digest.hexdigest()


def _resolve_tick_props(tick_props: Any) -> list[str]:
    if tick_props == "all" or tick_props is None:
        return DEFAULT_TICK_PROPS
    return list(tick_props)


def _discover_event_names(parser: Any, cfg: dict[str, Any]) -> list[str]:
    requested = cfg.get("events", "all")
    skip = set(cfg.get("skip_events") or [])
    if requested != "all":
        return [name for name in requested if name not in skip]
    for method_name in ("list_game_events", "list_events", "get_event_names"):
        method = getattr(parser, method_name, None)
        if method is None:
            continue
        try:
            names = method()
            return [str(name) for name in names if str(name) not in skip]
        except Exception as exc:
            log.debug("Could not discover events with %s: %s", method_name, exc)
    names = MINIMAL_EVENTS + ([] if not cfg.get("include_grenades", True) else EXTRA_EVENTS)
    return [name for name in names if name not in skip]


def _parse_ticks(parser: Any, props: list[str]) -> list[dict[str, Any]]:
    try:
        parsed = _call_compatible(parser, ["parse_ticks", "parse_tick_data"], props)
    except Exception:
        parsed = _call_compatible(parser, ["parse_ticks", "parse_tick_data"], wanted_props=props)
    records = _to_records(parsed)
    for rec in records:
        rec.setdefault("tick", rec.get("game_tick", rec.get("tick_id")))
    return records


def _parse_one_event(parser: Any, event_name: str) -> list[dict[str, Any]]:
    parsed = _call_compatible(
        parser,
        ["parse_event", "parse_events", "parse_game_event"],
        event_name,
    )
    records = _to_records(parsed)
    for rec in records:
        rec["event_name"] = rec.get("event_name") or event_name
        rec.setdefault("tick", rec.get("game_tick", rec.get("tick_id")))
    return records


def _parse_events(parser: Any, cfg: dict[str, Any], event_names: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    failed = []
    for name in event_names:
        try:
            events.extend(_parse_one_event(parser, name))
        except Exception as exc:
            failed.append(name)
            log.debug("Skipping event %s after parser error: %s", name, exc)
    if failed and cfg.get("event_fallback_minimal_on_error", True):
        for name in MINIMAL_EVENTS:
            if name in event_names:
                continue
            try:
                events.extend(_parse_one_event(parser, name))
            except Exception:
                pass
    if not events and cfg.get("event_fallback_minimal_on_empty", True):
        for name in MINIMAL_EVENTS:
            try:
                events.extend(_parse_one_event(parser, name))
            except Exception:
                pass
    return events


def _attach_events(frames: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    by_tick: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        tick = event.get("tick")
        if tick is None:
            continue
        by_tick.setdefault(int(tick), []).append(event)
    for frame in frames:
        tick = frame.get("tick")
        frame["events"] = by_tick.get(int(tick), []) if tick is not None else []


def _extract_players(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    id_fields = ("steamid", "steam_id", "user_steamid", "player_steamid", "entity_id", "name")
    for row in frames:
        player_id = next((row.get(field) for field in id_fields if row.get(field) is not None), None)
        if player_id is None:
            continue
        key = str(player_id)
        if key not in seen:
            seen[key] = {
                "player_id": key,
                "name": row.get("name") or row.get("player_name"),
                "steamid": row.get("steamid") or row.get("steam_id") or row.get("user_steamid"),
                "team_num": row.get("team_num") or row.get("team"),
            }
    return list(seen.values())


def parse_demo_to_frames(demo_path: str | Path, parsing_cfg: dict[str, Any]) -> dict[str, Any]:
    DemoParser = _load_demoparser()
    parser = DemoParser(str(demo_path))
    header = parse_demo_header(demo_path)
    props = _resolve_tick_props(parsing_cfg.get("tick_props"))
    frames = _parse_ticks(parser, props)
    event_names = _discover_event_names(parser, parsing_cfg)
    events = _parse_events(parser, parsing_cfg, event_names)
    _attach_events(frames, events)
    return {
        "frames": frames,
        "header": header,
        "players": _extract_players(frames),
        "events": events,
        "event_names": sorted({event.get("event_name") for event in events if event.get("event_name")}),
    }
