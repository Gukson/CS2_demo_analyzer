from __future__ import annotations

import datetime as dt
import logging
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Any, Iterable

from tqdm import tqdm

log = logging.getLogger(__name__)


class DemoDataQualityError(RuntimeError):
    """Raised when a parsed demo is not usable for ML sample generation."""


def _load_pymongo() -> Any:
    try:
        import pymongo
    except ImportError as exc:
        raise RuntimeError(
            "pymongo is required for output.mode=mongo_ml. Install with `pip install '.[mongo]'` "
            "or `pip install pymongo`."
        ) from exc
    return pymongo


class MongoMLWriter:
    def __init__(self, output_cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> None:
        pymongo = _load_pymongo()
        mongo_cfg = output_cfg["mongo_ml"]
        self.client = pymongo.MongoClient(
            mongo_cfg["uri"],
            serverSelectionTimeoutMS=int(mongo_cfg.get("server_selection_timeout_ms", 5000)),
            connectTimeoutMS=int(mongo_cfg.get("connect_timeout_ms", 5000)),
            socketTimeoutMS=int(mongo_cfg.get("socket_timeout_ms", 120000)),
            retryWrites=bool(mongo_cfg.get("retry_writes", True)),
        )
        self.client.admin.command("ping")
        self.db = self.client[mongo_cfg["db"]]
        self.matches = self.db[mongo_cfg.get("matches_collection", "ml_matches")]
        self.samples = self.db[mongo_cfg.get("samples_collection", "ml_samples")]
        self.meta = self.db[mongo_cfg.get("meta_collection", "ml_meta")]
        self.batch_size = int(mongo_cfg.get("insert_batch_size", 200))
        self.allow_zero_samples = bool(mongo_cfg.get("allow_zero_samples", False))
        self.show_progress = bool(mongo_cfg.get("show_progress", True))
        self.min_rounds_per_match = int(mongo_cfg.get("min_rounds_per_match", 1))
        self.dataset_cfg = dataset_cfg
        self.feature_names = list(dataset_cfg.get("features") or [])
        self.target_features = list(dataset_cfg.get("target_features") or [])
        self.categorical = set(dataset_cfg.get("categorical_features") or [])
        self.max_players = int(dataset_cfg.get("max_players", 10))
        self.history_steps = int(dataset_cfg.get("history_steps", 32))
        self.future_steps = int(dataset_cfg.get("future_steps", 8))
        self.temporal_stride_steps = int(dataset_cfg.get("stride_steps", 1))
        self.sample_stride_steps = int(dataset_cfg.get("sample_stride_steps", self.temporal_stride_steps))
        if mongo_cfg.get("ensure_indexes", True):
            self.ensure_indexes()

    def ensure_indexes(self) -> None:
        self.matches.create_index("match_id", unique=True)
        self.samples.create_index([("match_id", 1), ("round_number", 1), ("tick_anchor", 1)])
        self.samples.create_index([("split", 1), ("match_id", 1)])
        self.samples.create_index([("match_id", 1), ("round_number", 1), ("tick_anchor", 1), ("_id", 1)])
        self.meta.create_index("dataset_id", unique=True)

    def match_exists(self, match_id: str) -> bool:
        return self.matches.count_documents(
            {
                "match_id": match_id,
                "parse_status": "complete",
                "sample_count": {"$gt": 0},
                "round_count": {"$gte": self.min_rounds_per_match},
            },
            limit=1,
        ) > 0

    def match_summary(self, match_id: str) -> dict[str, Any] | None:
        return self.matches.find_one(
            {"match_id": match_id, "parse_status": "complete"},
            projection={"_id": 0, "source": 1, "sample_count": 1, "round_count": 1},
        )

    def write_match(
        self,
        *,
        match_id: str,
        source: str,
        match_info: dict[str, Any],
        demo_meta: dict[str, Any],
        frames: list[dict[str, Any]],
        rounds: list[dict[str, Any]],
        players: list[dict[str, Any]],
        parsing_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        map_name = _pick_first(demo_meta, ["map_name", "map", "network_protocol_map"])
        if len(rounds) < self.min_rounds_per_match:
            raise DemoDataQualityError(
                f"Suspiciously few rounds detected for match_id={match_id}: "
                f"rounds={len(rounds)} min_rounds_per_match={self.min_rounds_per_match}. "
                "Refusing to write incomplete round segmentation to Mongo."
            )
        samples, stats, vocab = self.build_samples(match_id, frames, map_name=map_name)
        if not samples and not self.allow_zero_samples:
            diagnostics = build_sample_diagnostics(frames, rounds, self.dataset_cfg)
            raise DemoDataQualityError(
                f"No ML samples built for match_id={match_id}. Refusing to mark demo as processed. "
                f"Diagnostics: {diagnostics}"
            )
        now = dt.datetime.now(dt.timezone.utc)
        match_doc = {
            "match_id": match_id,
            "source": source,
            "map": map_name,
            "match_info": match_info,
            "demo_meta": demo_meta,
            "players": players,
            "rounds": rounds,
            "round_count": len(rounds),
            "sample_count": len(samples),
            "updated_at": now,
        }
        try:
            log.info(
                "Writing match_id=%s to Mongo: samples=%s rounds=%s batches=%s",
                match_id,
                len(samples),
                len(rounds),
                math.ceil(len(samples) / self.batch_size) if samples else 0,
            )
            self.samples.delete_many({"match_id": match_id})
            inserted = 0
            batches = list(_chunks(samples, self.batch_size))
            batch_iter = tqdm(
                batches,
                desc=f"Mongo insert {match_id[:14]}",
                leave=False,
                disable=not self.show_progress or len(batches) <= 1,
            )
            for batch in batch_iter:
                if batch:
                    result = self.samples.insert_many(batch, ordered=True)
                    inserted += len(result.inserted_ids)
            if inserted != len(samples):
                raise RuntimeError(f"Inserted {inserted} samples, expected {len(samples)}")
            stored_samples = self.samples.count_documents({"match_id": match_id})
            if stored_samples != len(samples):
                raise RuntimeError(f"Mongo has {stored_samples} samples, expected {len(samples)}")
            self._upsert_meta(parsing_cfg, stats, vocab)
            match_doc["parse_status"] = "complete"
            self.matches.replace_one({"match_id": match_id}, match_doc, upsert=True)
            self._verify_committed_match(match_id, len(samples))
        except Exception:
            log.exception("Mongo write failed for match_id=%s; aborting before marking processed", match_id)
            raise
        return {"sample_count": len(samples), "round_count": len(rounds)}

    def _verify_committed_match(self, match_id: str, expected_samples: int) -> None:
        match = self.matches.find_one(
            {"match_id": match_id, "parse_status": "complete"},
            projection={"_id": 1, "sample_count": 1},
        )
        if not match:
            raise RuntimeError(f"Mongo commit verification failed for {match_id}: match document missing")
        if int(match.get("sample_count", -1)) != expected_samples:
            raise RuntimeError(
                f"Mongo commit verification failed for {match_id}: "
                f"match sample_count={match.get('sample_count')} expected={expected_samples}"
            )
        stored_samples = self.samples.count_documents({"match_id": match_id})
        if stored_samples != expected_samples:
            raise RuntimeError(
                f"Mongo commit verification failed for {match_id}: samples={stored_samples} expected={expected_samples}"
            )

    def build_samples(
        self,
        match_id: str,
        frames: list[dict[str, Any]],
        *,
        map_name: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        rows_by_tick = _group_rows_by_tick(frames)
        ticks = sorted(rows_by_tick)
        sampled_ticks = _sample_ticks(ticks, int(self.dataset_cfg.get("sample_every_ticks", 1)))
        global_features = _GlobalFeatureBuilder(rows_by_tick, ticks, self.dataset_cfg)
        vocab = self._build_vocab(frames)
        stats_builder = _StatsBuilder(self.feature_names, self.categorical)
        samples: list[dict[str, Any]] = []
        skipped_no_round = 0
        skipped_cross_round = 0
        first_anchor = (self.history_steps - 1) * self.temporal_stride_steps
        last_anchor = len(sampled_ticks) - 1
        if self.dataset_cfg.get("predict_future", True):
            last_anchor -= self.future_steps * self.temporal_stride_steps
        anchor_range = range(first_anchor, last_anchor + 1, max(1, self.sample_stride_steps))
        log.info(
            "Building samples for %s: unique_ticks=%s sampled_ticks=%s candidate_anchors=%s",
            match_id,
            len(ticks),
            len(sampled_ticks),
            len(anchor_range),
        )
        anchor_iter = tqdm(
            anchor_range,
            desc=f"Build samples {match_id[:14]}",
            leave=False,
            disable=not self.show_progress,
        )
        for idx in anchor_iter:
            tick = sampled_ticks[idx]
            round_number = _round_number_for_rows(rows_by_tick[tick])
            if round_number is None:
                skipped_no_round += 1
                continue
            history_ticks = [
                sampled_ticks[idx - step * self.temporal_stride_steps] for step in reversed(range(self.history_steps))
            ]
            future_ticks = [
                sampled_ticks[idx + (step + 1) * self.temporal_stride_steps] for step in range(self.future_steps)
            ]
            if not _same_round_window(rows_by_tick, history_ticks + future_ticks, round_number):
                skipped_cross_round += 1
                continue
            history, history_mask, player_ids = self._tensor_for_ticks(history_ticks, rows_by_tick, vocab, self.feature_names)
            future, future_mask, _ = self._tensor_for_ticks(future_ticks, rows_by_tick, vocab, self.target_features, player_ids)
            future_alive, _, _ = self._tensor_for_ticks(future_ticks, rows_by_tick, vocab, ["is_alive"], player_ids)
            for timestep in history:
                for player in timestep:
                    stats_builder.update(player)
            labels = self._labels(rows_by_tick, tick, future_ticks, round_number)
            samples.append(
                {
                    "match_id": match_id,
                    "schema_version": 2,
                    "round_number": round_number,
                    "map": map_name,
                    "tick_start": history_ticks[0],
                    "tick_anchor": tick,
                    "tick_future_end": future_ticks[-1] if future_ticks else None,
                    "history_ticks": history_ticks,
                    "future_ticks": future_ticks,
                    "split": "train",
                    "player_ids": player_ids,
                    "history": history,
                    "history_mask": history_mask,
                    "future": future,
                    "future_mask": future_mask,
                    "future_alive": future_alive,
                    "global_features": global_features.features(tick, round_number),
                    "labels": labels,
                    "round_winner": labels["round_winner"],
                }
            )
        log.info(
            "Built samples for %s: samples=%s skipped_no_round=%s skipped_cross_round=%s",
            match_id,
            len(samples),
            skipped_no_round,
            skipped_cross_round,
        )
        return samples, stats_builder.finish(), vocab

    def _tensor_for_ticks(
        self,
        ticks: list[int],
        rows_by_tick: dict[int, list[dict[str, Any]]],
        vocab: dict[str, dict[str, int]],
        feature_names: list[str],
        fixed_player_ids: list[str] | None = None,
    ) -> tuple[list[list[list[float]]], list[list[int]], list[str]]:
        if fixed_player_ids is None:
            players = _ordered_players([row for tick in ticks for row in rows_by_tick.get(tick, [])], self.max_players)
        else:
            players = fixed_player_ids[: self.max_players]
        players = players + [""] * (self.max_players - len(players))
        tensor: list[list[list[float]]] = []
        mask: list[list[int]] = []
        for tick in ticks:
            by_player = {_player_id(row): row for row in rows_by_tick.get(tick, [])}
            tick_values: list[list[float]] = []
            tick_mask: list[int] = []
            for player_id in players:
                row = by_player.get(player_id) if player_id else None
                tick_mask.append(1 if row else 0)
                tick_values.append([self._feature_value(row, name, vocab) for name in feature_names])
            tensor.append(tick_values)
            mask.append(tick_mask)
        return tensor, mask, players

    def _feature_value(self, row: dict[str, Any] | None, name: str, vocab: dict[str, dict[str, int]]) -> float:
        if row is None:
            return 0.0
        value = row.get(name)
        if name in self.categorical:
            return float(vocab.setdefault(name, {}).setdefault(str(value), len(vocab.setdefault(name, {})) + 1)) if value is not None else 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isnan(float(value)) or math.isinf(float(value)):
                return 0.0
            return float(value)
        if isinstance(value, (list, dict)):
            return float(len(value))
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _build_vocab(self, frames: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        vocab: dict[str, dict[str, int]] = {name: {} for name in self.categorical}
        for row in frames:
            for name in self.categorical:
                value = row.get(name)
                if value is not None and str(value) not in vocab[name]:
                    vocab[name][str(value)] = len(vocab[name]) + 1
        return vocab

    def _global_features(self, rows_by_tick: dict[int, list[dict[str, Any]]], tick: int, round_number: int | None) -> dict[str, Any]:
        if not self.dataset_cfg.get("global", {}).get("include", True):
            return {}
        rows = rows_by_tick.get(tick, [])
        result: dict[str, Any] = {
            "round_number": round_number,
            "tick": tick,
            "tick_in_round": _tick_in_round(rows_by_tick, tick, round_number),
            "alive_t": 0,
            "alive_ct": 0,
            "health_t": 0.0,
            "health_ct": 0.0,
            "armor_t": 0.0,
            "armor_ct": 0.0,
            "equip_value_t": 0.0,
            "equip_value_ct": 0.0,
            "balance_t": 0.0,
            "balance_ct": 0.0,
            "bomb_planted": 0,
            "bomb_carried": 0,
            "bomb_dropped": 0,
            "event_counts": {},
            "recent_event_counts": {},
        }
        for row in rows:
            team = int(row.get("team_num") or row.get("team") or 0)
            alive = _is_alive(row)
            suffix = "t" if team == 2 else "ct" if team == 3 else None
            if suffix:
                result[f"alive_{suffix}"] += 1 if alive else 0
                result[f"health_{suffix}"] += _num(row.get("health"))
                result[f"armor_{suffix}"] += _num(row.get("armor_value"))
                result[f"equip_value_{suffix}"] += _num(row.get("current_equip_value"))
                result[f"balance_{suffix}"] += _num(row.get("balance"))
            _update_bomb_state(result, row)
            for event in row.get("events") or []:
                name = event.get("event_name")
                if name:
                    result["event_counts"][name] = result["event_counts"].get(name, 0) + 1
        recent_window = int(self.dataset_cfg.get("global", {}).get("team_recent_window_steps", 32))
        for recent_tick in sorted(rows_by_tick):
            if recent_tick > tick:
                break
            if recent_tick < tick - recent_window * int(self.dataset_cfg.get("sample_every_ticks", 1)):
                continue
            for row in rows_by_tick[recent_tick]:
                for event in row.get("events") or []:
                    name = event.get("event_name")
                    if name:
                        result["recent_event_counts"][name] = result["recent_event_counts"].get(name, 0) + 1
        return result

    def _labels(
        self,
        rows_by_tick: dict[int, list[dict[str, Any]]],
        tick: int,
        future_ticks: list[int],
        round_number: int | None,
    ) -> dict[str, Any]:
        future_events: dict[str, int] = {}
        for future_tick in future_ticks:
            if _round_number_for_rows(rows_by_tick.get(future_tick, [])) != round_number:
                continue
            for row in rows_by_tick.get(future_tick, []):
                for event in row.get("events") or []:
                    name = event.get("event_name")
                    if name:
                        future_events[name] = future_events.get(name, 0) + 1
        return {
            "round_winner": _winner_for_rows(rows_by_tick.get(tick, [])),
            "future_event_counts": future_events,
            "future_has_kill": int(future_events.get("player_death", 0) > 0),
            "future_has_bomb_planted": int(future_events.get("bomb_planted", 0) > 0),
            "future_has_bomb_defused": int(future_events.get("bomb_defused", 0) > 0),
            "future_has_bomb_exploded": int(future_events.get("bomb_exploded", 0) > 0),
        }

    def _upsert_meta(self, parsing_cfg: dict[str, Any], stats: dict[str, Any], vocab: dict[str, Any]) -> None:
        dataset_id = "default"
        self.meta.replace_one(
            {"dataset_id": dataset_id},
            {
                "dataset_id": dataset_id,
                "updated_at": dt.datetime.now(dt.timezone.utc),
                "dataset_config": self.dataset_cfg,
                "parsing_config": parsing_cfg,
                "features": self.feature_names,
                "target_features": self.target_features,
                "global_features": [
                    "round_number",
                    "tick",
                    "tick_in_round",
                    "alive_t",
                    "alive_ct",
                    "health_t",
                    "health_ct",
                    "armor_t",
                    "armor_ct",
                    "equip_value_t",
                    "equip_value_ct",
                    "balance_t",
                    "balance_ct",
                    "bomb_planted",
                    "bomb_carried",
                    "bomb_dropped",
                    "event_counts",
                    "recent_event_counts",
                ],
                "sample_schema": {
                    "schema_version": 2,
                    "history_shape": [self.history_steps, self.max_players, len(self.feature_names)],
                    "future_shape": [self.future_steps, self.max_players, len(self.target_features)],
                    "future_alive_shape": [self.future_steps, self.max_players, 1],
                    "sample_every_ticks": self.dataset_cfg.get("sample_every_ticks"),
                    "sample_stride_steps": self.sample_stride_steps,
                    "temporal_stride_steps": self.temporal_stride_steps,
                },
                "normalization": stats,
                "categorical_vocabulary": vocab,
            },
            upsert=True,
        )


class _StatsBuilder:
    def __init__(self, feature_names: list[str], categorical: set[str]) -> None:
        self.feature_names = feature_names
        self.categorical = categorical
        self.count = defaultdict(int)
        self.sum = defaultdict(float)
        self.sum_sq = defaultdict(float)

    def update(self, values: list[float]) -> None:
        for name, value in zip(self.feature_names, values):
            if name in self.categorical:
                continue
            self.count[name] += 1
            self.sum[name] += value
            self.sum_sq[name] += value * value

    def finish(self) -> dict[str, Any]:
        stats = {}
        for name in self.feature_names:
            if name in self.categorical:
                continue
            count = self.count[name]
            if not count:
                stats[name] = {"mean": 0.0, "std": 1.0, "count": 0}
                continue
            mean = self.sum[name] / count
            variance = max((self.sum_sq[name] / count) - mean * mean, 0.0)
            stats[name] = {"mean": mean, "std": math.sqrt(variance) or 1.0, "count": count}
        return {"type": "zscore", "features": stats}


class _GlobalFeatureBuilder:
    def __init__(
        self,
        rows_by_tick: dict[int, list[dict[str, Any]]],
        ticks: list[int],
        dataset_cfg: dict[str, Any],
    ) -> None:
        self.rows_by_tick = rows_by_tick
        self.include = bool(dataset_cfg.get("global", {}).get("include", True))
        self.recent_window_ticks = int(dataset_cfg.get("global", {}).get("team_recent_window_steps", 32)) * int(
            dataset_cfg.get("sample_every_ticks", 1)
        )
        self.round_start_by_number = self._build_round_starts(ticks)
        self.events_by_tick = self._build_events_by_tick(ticks)
        self.event_ticks = sorted(self.events_by_tick)

    def features(self, tick: int, round_number: int | None) -> dict[str, Any]:
        if not self.include:
            return {}
        rows = self.rows_by_tick.get(tick, [])
        result: dict[str, Any] = {
            "round_number": round_number,
            "tick": tick,
            "tick_in_round": self._tick_in_round(tick, round_number),
            "alive_t": 0,
            "alive_ct": 0,
            "health_t": 0.0,
            "health_ct": 0.0,
            "armor_t": 0.0,
            "armor_ct": 0.0,
            "equip_value_t": 0.0,
            "equip_value_ct": 0.0,
            "balance_t": 0.0,
            "balance_ct": 0.0,
            "bomb_planted": 0,
            "bomb_carried": 0,
            "bomb_dropped": 0,
            "event_counts": {},
            "recent_event_counts": self._recent_event_counts(tick),
        }
        for row in rows:
            team = int(row.get("team_num") or row.get("team") or 0)
            alive = _is_alive(row)
            suffix = "t" if team == 2 else "ct" if team == 3 else None
            if suffix:
                result[f"alive_{suffix}"] += 1 if alive else 0
                result[f"health_{suffix}"] += _num(row.get("health"))
                result[f"armor_{suffix}"] += _num(row.get("armor_value"))
                result[f"equip_value_{suffix}"] += _num(row.get("current_equip_value"))
                result[f"balance_{suffix}"] += _num(row.get("balance"))
            _update_bomb_state(result, row)
            for name in _event_names_from_row(row):
                result["event_counts"][name] = result["event_counts"].get(name, 0) + 1
        return result

    def _tick_in_round(self, tick: int, round_number: int | None) -> int | None:
        if round_number is None:
            return None
        round_start = self.round_start_by_number.get(round_number)
        if round_start is None:
            return None
        return tick - round_start

    def _recent_event_counts(self, tick: int) -> dict[str, int]:
        if not self.event_ticks:
            return {}
        start_tick = tick - self.recent_window_ticks
        start_idx = bisect_left(self.event_ticks, start_tick)
        end_idx = bisect_right(self.event_ticks, tick)
        counts: dict[str, int] = {}
        for event_tick in self.event_ticks[start_idx:end_idx]:
            for name in self.events_by_tick[event_tick]:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _build_round_starts(self, ticks: list[int]) -> dict[int, int]:
        starts: dict[int, int] = {}
        for tick in ticks:
            round_number = _round_number_for_rows(self.rows_by_tick.get(tick, []))
            if round_number is not None and round_number not in starts:
                starts[round_number] = tick
        return starts

    def _build_events_by_tick(self, ticks: list[int]) -> dict[int, list[str]]:
        events_by_tick: dict[int, list[str]] = {}
        for tick in ticks:
            names = [
                name
                for row in self.rows_by_tick.get(tick, [])
                for name in _event_names_from_row(row)
            ]
            if names:
                events_by_tick[tick] = names
        return events_by_tick


def _group_rows_by_tick(frames: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    rows_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        if row.get("tick") is not None:
            rows_by_tick[int(row["tick"])].append(row)
    return rows_by_tick


def _sample_ticks(ticks: list[int], every: int) -> list[int]:
    if not ticks:
        return []
    every = max(1, every)
    sampled = [ticks[0]]
    last = ticks[0]
    for tick in ticks[1:]:
        if tick - last >= every:
            sampled.append(tick)
            last = tick
    return sampled


def _ordered_players(rows: list[dict[str, Any]], max_players: int) -> list[str]:
    first_seen = {}
    for row in rows:
        player_id = _player_id(row)
        if player_id and player_id not in first_seen:
            first_seen[player_id] = (int(row.get("team_num") or row.get("team") or 99), len(first_seen))
    return [pid for pid, _ in sorted(first_seen.items(), key=lambda item: item[1])[:max_players]]


def _player_id(row: dict[str, Any]) -> str:
    for key in ("steamid", "steam_id", "user_steamid", "player_steamid", "entity_id", "name"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def _round_number_for_rows(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row.get("round_number") is not None:
            return int(row["round_number"])
    return None


def _winner_for_rows(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row.get("round_winner_team") is not None:
            return int(row["round_winner_team"])
    return None


def _same_round_window(rows_by_tick: dict[int, list[dict[str, Any]]], ticks: list[int], round_number: int) -> bool:
    for tick in ticks:
        rows = rows_by_tick.get(tick, [])
        if not rows:
            return False
        row_round = _round_number_for_rows(rows)
        if row_round != round_number:
            return False
    return True


def _tick_in_round(rows_by_tick: dict[int, list[dict[str, Any]]], tick: int, round_number: int | None) -> int | None:
    if round_number is None:
        return None
    round_ticks = [
        candidate
        for candidate, rows in rows_by_tick.items()
        if candidate <= tick and _round_number_for_rows(rows) == round_number
    ]
    if not round_ticks:
        return None
    return tick - min(round_ticks)


def _event_names_from_row(row: dict[str, Any]) -> list[str]:
    return [
        event["event_name"]
        for event in row.get("events") or []
        if event.get("event_name")
    ]


def _update_bomb_state(result: dict[str, Any], row: dict[str, Any]) -> None:
    for key in ("bomb_planted", "is_bomb_planted", "bomb_is_planted"):
        if _truthy(row.get(key)):
            result["bomb_planted"] = 1
    for key in ("has_bomb", "is_bomb_carrier", "bomb_carrier"):
        if _truthy(row.get(key)):
            result["bomb_carried"] = 1
    for key in ("bomb_dropped", "is_bomb_dropped"):
        if _truthy(row.get(key)):
            result["bomb_dropped"] = 1
    for event in row.get("events") or []:
        name = event.get("event_name")
        if name == "bomb_planted":
            result["bomb_planted"] = 1
            result["bomb_carried"] = 0
        elif name in {"bomb_dropped", "bomb_pickup"}:
            result["bomb_dropped"] = int(name == "bomb_dropped")
            result["bomb_carried"] = int(name == "bomb_pickup")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def _is_alive(row: dict[str, Any]) -> bool:
    if row.get("is_alive") is not None:
        return bool(row["is_alive"])
    if row.get("life_state") is not None:
        return int(row["life_state"]) == 0
    return _num(row.get("health")) > 0


def _num(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pick_first(source: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if source.get(key) is not None:
            return source[key]
    return None


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, size)
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def build_sample_diagnostics(
    frames: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
) -> dict[str, Any]:
    rows_by_tick = _group_rows_by_tick(frames)
    ticks = sorted(rows_by_tick)
    sampled_ticks = _sample_ticks(ticks, int(dataset_cfg.get("sample_every_ticks", 1)))
    round_numbers = sorted(
        {
            int(row["round_number"])
            for rows in rows_by_tick.values()
            for row in rows
            if row.get("round_number") is not None
        }
    )
    history_steps = int(dataset_cfg.get("history_steps", 0))
    future_steps = int(dataset_cfg.get("future_steps", 0))
    temporal_stride = int(dataset_cfg.get("stride_steps", 1))
    required_ticks = ((history_steps - 1) + future_steps) * max(1, temporal_stride) + 1
    candidate_anchor_count = max(0, len(sampled_ticks) - required_ticks + 1)
    ticks_per_round: dict[int, int] = {}
    for tick, rows in rows_by_tick.items():
        round_number = _round_number_for_rows(rows)
        if round_number is not None:
            ticks_per_round[round_number] = ticks_per_round.get(round_number, 0) + 1
    sampled_per_round: dict[int, int] = {}
    for tick in sampled_ticks:
        round_number = _round_number_for_rows(rows_by_tick.get(tick, []))
        if round_number is not None:
            sampled_per_round[round_number] = sampled_per_round.get(round_number, 0) + 1
    return {
        "frame_rows": len(frames),
        "unique_ticks": len(ticks),
        "sampled_ticks": len(sampled_ticks),
        "detected_rounds": len(rounds),
        "round_numbers_on_frames": round_numbers[:40],
        "candidate_anchor_count_before_round_window_filter": candidate_anchor_count,
        "required_sampled_ticks_per_window": required_ticks,
        "history_steps": history_steps,
        "future_steps": future_steps,
        "sample_every_ticks": dataset_cfg.get("sample_every_ticks"),
        "ticks_per_round_first_10": dict(list(sorted(ticks_per_round.items()))[:10]),
        "sampled_ticks_per_round_first_10": dict(list(sorted(sampled_per_round.items()))[:10]),
    }
