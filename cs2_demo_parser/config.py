from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "mode": "local",
        "local_demo_dir": "demos/extracted",
        "local_recursive": True,
        "local_match_id_strategy": "header",
        "local_delete_processed_demo": True,
        "local_rejected_demo_dir": "demos/rejected",
    },
    "parsing": {
        "sample_every_ticks": 16,
        "max_workers": 1,
        "context": {"window_size_ticks": 128, "window_count": 8},
        "tick_props": "all",
        "events": "all",
        "skip_events": [],
        "event_fallback_minimal_on_error": True,
        "event_fallback_minimal_on_empty": True,
        "lazy_sanitize": True,
        "include_grenades": True,
    },
    "output": {
        "mode": "mongo_ml",
        "mongo_ml": {
            "uri": "mongodb://localhost:27017",
            "db": "cs2_demo_ml",
            "matches_collection": "ml_matches",
            "samples_collection": "ml_samples",
            "meta_collection": "ml_meta",
            "ensure_indexes": True,
            "insert_batch_size": 200,
            "server_selection_timeout_ms": 5000,
            "connect_timeout_ms": 5000,
            "socket_timeout_ms": 120000,
            "retry_writes": True,
            "allow_zero_samples": False,
            "show_progress": True,
            "min_rounds_per_match": 5,
        },
    },
    "dataset": {
        "history_steps": 64,
        "future_steps": 32,
        "sample_stride_steps": 4,
        "stride_steps": 1,
        "max_players": 10,
        "predict_future": True,
        "features": [],
        "target_features": ["X", "Y", "Z"],
        "categorical_features": ["active_weapon", "weapon_class"],
        "normalization": "zscore",
        "zscore_clip_abs": 20.0,
        "global": {"include": True},
    },
    "state": {"processed_matches_file": "state/processed_matches.json"},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return deep_merge(DEFAULT_CONFIG, raw)


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.input_mode:
        cfg["input"]["mode"] = args.input_mode
    if args.max_workers is not None:
        cfg["parsing"]["max_workers"] = args.max_workers
    if args.sample_every_ticks is not None:
        cfg["parsing"]["sample_every_ticks"] = args.sample_every_ticks
    if args.context_window_size is not None:
        cfg["parsing"].setdefault("context", {})["window_size_ticks"] = args.context_window_size
    if args.context_window_count is not None:
        cfg["parsing"].setdefault("context", {})["window_count"] = args.context_window_count
    if args.last_n is not None:
        cfg["input"]["last_n"] = args.last_n
    if args.date_from:
        cfg["input"]["date_from"] = args.date_from
    if args.date_to:
        cfg["input"]["date_to"] = args.date_to
    if getattr(args, "quiet_progress", False):
        cfg.setdefault("output", {}).setdefault("mongo_ml", {})["show_progress"] = False
    return cfg
