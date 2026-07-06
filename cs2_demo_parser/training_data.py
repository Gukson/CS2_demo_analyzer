from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .model_samples import MongoSequenceSampleStore


DEFAULT_GLOBAL_KEYS = [
    "round_number",
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
]

DEFAULT_EVENT_KEYS = [
    "player_death",
    "bomb_planted",
    "bomb_defused",
    "bomb_exploded",
    "player_hurt",
    "weapon_fire",
    "grenade_thrown",
    "flashbang_detonate",
    "smokegrenade_detonate",
    "hegrenade_detonate",
    "molotov_detonate",
]

FEATURE_SCALES = {
    "X": 4096.0,
    "Y": 4096.0,
    "Z": 2048.0,
    "velocity_X": 1000.0,
    "velocity_Y": 1000.0,
    "velocity_Z": 1000.0,
    "pitch": 90.0,
    "yaw": 180.0,
    "health": 100.0,
    "armor_value": 100.0,
    "life_state": 2.0,
    "death_time": 2000.0,
    "balance": 16000.0,
    "start_balance": 16000.0,
    "cash_spent_this_round": 20000.0,
    "total_cash_spent": 60000.0,
    "round_start_equip_value": 40000.0,
    "current_equip_value": 40000.0,
    "inventory": 10.0,
    "inventory_as_ids": 10.0,
    "active_weapon": 128.0,
    "weapon_class": 64.0,
}


@dataclass
class SampleSchema:
    player_feature_dim: int
    target_feature_dim: int
    global_feature_dim: int
    history_steps: int
    future_steps: int
    max_players: int
    event_label_dim: int
    feature_names: list[str]
    target_feature_names: list[str]
    global_feature_names: list[str]
    event_label_names: list[str]


class GlobalFeatureVectorizer:
    def __init__(self, numeric_keys: list[str] | None = None, event_keys: list[str] | None = None) -> None:
        self.numeric_keys = numeric_keys or DEFAULT_GLOBAL_KEYS
        self.event_keys = event_keys or DEFAULT_EVENT_KEYS
        self.names = (
            self.numeric_keys
            + [f"event_counts.{name}" for name in self.event_keys]
            + [f"recent_event_counts.{name}" for name in self.event_keys]
        )

    def transform(self, global_features: dict[str, Any]) -> list[float]:
        values = [_stable_num(global_features.get(key)) for key in self.numeric_keys]
        event_counts = global_features.get("event_counts") or {}
        recent_event_counts = global_features.get("recent_event_counts") or {}
        values.extend(_stable_num(event_counts.get(key)) for key in self.event_keys)
        values.extend(_stable_num(recent_event_counts.get(key)) for key in self.event_keys)
        return values


class MongoWorldModelDataset(IterableDataset):
    def __init__(
        self,
        mongo_cfg: dict[str, Any],
        *,
        split: str,
        fold: str,
        val_fraction: float,
        event_label_names: list[str],
        vectorizer: GlobalFeatureVectorizer,
        mongo_batch_size: int = 512,
    ) -> None:
        super().__init__()
        self.mongo_cfg = mongo_cfg
        self.split = split
        self.fold = fold
        self.val_fraction = val_fraction
        self.event_label_names = event_label_names
        self.vectorizer = vectorizer
        self.mongo_batch_size = mongo_batch_size
        meta = MongoSequenceSampleStore(self.mongo_cfg).get_meta() or {}
        self.feature_names = list(meta.get("features") or [])
        self.target_feature_names = list(meta.get("target_features") or [])

    def __iter__(self):
        worker = get_worker_info()
        store = MongoSequenceSampleStore(self.mongo_cfg)
        for sample in store.iter_samples(split=self.split, batch_size=self.mongo_batch_size):
            if not _belongs_to_fold(sample, self.fold, self.val_fraction):
                continue
            if worker is not None and _sample_bucket(sample, worker.num_workers) != worker.id:
                continue
            yield encode_sample(
                sample,
                self.vectorizer,
                self.event_label_names,
                feature_names=self.feature_names,
                target_feature_names=self.target_feature_names,
            )


class ShardedWorldModelDataset(IterableDataset):
    def __init__(
        self,
        shard_dir: str | Path,
        *,
        fold: str,
        shuffle_shards: bool = True,
        shuffle_samples: bool = True,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.fold = fold
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.seed = seed
        self.files = sorted((self.shard_dir / fold).glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No {fold} shard files found in {self.shard_dir / fold}")

    def __iter__(self):
        worker = get_worker_info()
        files = list(self.files)
        rng = random.Random(self.seed + (worker.id if worker is not None else 0))
        if self.shuffle_shards:
            rng.shuffle(files)
        if worker is not None:
            files = [path for idx, path in enumerate(files) if idx % worker.num_workers == worker.id]
        for path in files:
            payload = _torch_load_cpu(path)
            tensors = payload.get("tensors") or payload
            count = int(payload.get("count") or _infer_tensor_count(tensors))
            indices = list(range(count))
            if self.shuffle_samples:
                rng.shuffle(indices)
            for idx in indices:
                yield {key: value[idx] for key, value in tensors.items()}


def infer_schema(meta: dict[str, Any], event_label_names: list[str], vectorizer: GlobalFeatureVectorizer) -> SampleSchema:
    sample_schema = meta.get("sample_schema") or {}
    feature_names = list(meta.get("features") or [])
    target_feature_names = list(meta.get("target_features") or [])
    history_shape = sample_schema.get("history_shape") or [0, 10, len(feature_names)]
    future_shape = sample_schema.get("future_shape") or [0, 10, len(target_feature_names)]
    return SampleSchema(
        player_feature_dim=int(history_shape[2]),
        target_feature_dim=int(future_shape[2]),
        global_feature_dim=len(vectorizer.names),
        history_steps=int(history_shape[0]),
        future_steps=int(future_shape[0]),
        max_players=int(history_shape[1]),
        event_label_dim=len(event_label_names),
        feature_names=feature_names,
        target_feature_names=target_feature_names,
        global_feature_names=vectorizer.names,
        event_label_names=event_label_names,
    )


def encode_sample(
    sample: dict[str, Any],
    vectorizer: GlobalFeatureVectorizer,
    event_label_names: list[str],
    feature_names: list[str] | None = None,
    target_feature_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    labels = sample.get("labels") or {}
    winner = labels.get("round_winner", sample.get("round_winner"))
    winner_class = 0 if winner is None else 1 if int(winner) == 2 else 2 if int(winner) == 3 else 0
    feature_names = feature_names or []
    target_feature_names = target_feature_names or []
    return {
        "history": torch.tensor(_normalize_sequence(sample["history"], feature_names), dtype=torch.float32),
        "history_mask": torch.tensor(sample["history_mask"], dtype=torch.float32),
        "future": torch.tensor(_normalize_sequence(sample["future"], target_feature_names), dtype=torch.float32),
        "future_mask": torch.tensor(sample["future_mask"], dtype=torch.float32),
        "future_alive": torch.tensor(sample.get("future_alive") or _alive_from_mask(sample["future_mask"]), dtype=torch.float32),
        "global_features": torch.tensor(vectorizer.transform(sample.get("global_features") or {}), dtype=torch.float32),
        "event_labels": torch.tensor([float(labels.get(name, 0.0)) for name in event_label_names], dtype=torch.float32),
        "winner_label": torch.tensor(winner_class, dtype=torch.long),
        "winner_known": torch.tensor(0.0 if winner_class == 0 else 1.0, dtype=torch.float32),
    }


def collate_world_model_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items], dim=0) for key in keys}


def stack_encoded_samples(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return collate_world_model_batch(items)


def load_shard_manifest(shard_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(shard_dir) / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Shard manifest not found: {manifest_path}")
    import json

    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _torch_load_cpu(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_tensor_count(tensors: dict[str, torch.Tensor]) -> int:
    first = next(iter(tensors.values()), None)
    return 0 if first is None else int(first.shape[0])


def _normalize_sequence(sequence: list[list[list[Any]]], names: list[str]) -> list[list[list[float]]]:
    if not names:
        return sequence
    return [[[_normalize_feature(name, value) for name, value in zip(names, player)] for player in tick] for tick in sequence]


def _normalize_feature(name: str, value: Any) -> float:
    numeric = _raw_num(value)
    if name == "team_num":
        if int(numeric) == 2:
            return -1.0
        if int(numeric) == 3:
            return 1.0
        return 0.0
    scale = FEATURE_SCALES.get(name)
    if scale:
        return max(-20.0, min(20.0, numeric / scale))
    return max(-20.0, min(20.0, numeric))


def _raw_num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return numeric


def _alive_from_mask(mask: list[list[int]]) -> list[list[list[float]]]:
    return [[[float(value)] for value in row] for row in mask]


def _belongs_to_fold(sample: dict[str, Any], fold: str, val_fraction: float) -> bool:
    if val_fraction <= 0.0:
        return fold == "train"
    key = f"{sample.get('match_id')}:{sample.get('round_number')}:{sample.get('tick_anchor')}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    is_val = bucket < val_fraction
    return is_val if fold == "val" else not is_val


def _sample_bucket(sample: dict[str, Any], bucket_count: int) -> int:
    key = f"{sample.get('match_id')}:{sample.get('round_number')}:{sample.get('tick_anchor')}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, bucket_count)


def _stable_num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return math.copysign(math.log1p(abs(numeric)), numeric)
