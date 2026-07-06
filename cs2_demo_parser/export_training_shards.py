from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .model_samples import MongoSequenceSampleStore
from .training_data import (
    GlobalFeatureVectorizer,
    _belongs_to_fold,
    encode_sample,
    stack_encoded_samples,
)

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Mongo ML samples into local PyTorch training shards.")
    parser.add_argument("--config", default="config.train.json")
    parser.add_argument("--output-dir", default="data/world_model_shards")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--mongo-batch-size", type=int)
    parser.add_argument("--limit", type=int, help="Export at most N source samples, useful for smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _load_json(args.config)
    export_shards(config, args)
    return 0


def export_shards(config: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output dir already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)

    dataset_cfg = config["dataset"]
    split = dataset_cfg.get("split", "train")
    val_fraction = float(dataset_cfg.get("val_fraction", 0.1))
    event_label_names = list(dataset_cfg.get("event_labels") or [])
    mongo_batch_size = int(args.mongo_batch_size or dataset_cfg.get("mongo_batch_size", 512))
    shard_size = max(1, int(args.shard_size))

    store = MongoSequenceSampleStore(config["mongo"])
    meta = store.get_meta() or {}
    if not meta:
        raise RuntimeError("No ml_meta document found in Mongo. Run the demo parser first.")
    feature_names = list(meta.get("features") or [])
    target_feature_names = list(meta.get("target_features") or [])
    vectorizer = GlobalFeatureVectorizer()

    counts = {"train": 0, "val": 0}
    shard_counts = {"train": 0, "val": 0}
    buffers: dict[str, list[dict[str, torch.Tensor]]] = {"train": [], "val": []}
    total = args.limit or store.count_samples(split=split)

    iterator = store.iter_samples(split=split, batch_size=mongo_batch_size)
    for source_idx, sample in enumerate(tqdm(iterator, total=total, desc="Export shards"), start=1):
        fold = "val" if _belongs_to_fold(sample, "val", val_fraction) else "train"
        encoded = encode_sample(
            sample,
            vectorizer,
            event_label_names,
            feature_names=feature_names,
            target_feature_names=target_feature_names,
        )
        buffers[fold].append(encoded)
        counts[fold] += 1
        if len(buffers[fold]) >= shard_size:
            shard_counts[fold] += 1
            _write_shard(output_dir, fold, shard_counts[fold], buffers[fold])
            buffers[fold] = []
        if args.limit and source_idx >= args.limit:
            break

    for fold in ("train", "val"):
        if buffers[fold]:
            shard_counts[fold] += 1
            _write_shard(output_dir, fold, shard_counts[fold], buffers[fold])
            buffers[fold] = []

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "format": "cs2_world_model_shards_v1",
        "source": {
            "mongo_db": config["mongo"].get("db"),
            "samples_collection": config["mongo"].get("samples_collection", "ml_samples"),
            "split": split,
        },
        "dataset": {
            "val_fraction": val_fraction,
            "event_labels": event_label_names,
            "shard_size": shard_size,
            "feature_names": feature_names,
            "target_feature_names": target_feature_names,
            "global_feature_names": vectorizer.names,
            "meta": meta,
        },
        "counts": counts,
        "shards": shard_counts,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info(
        "Exported shards to %s: train=%s samples/%s shards val=%s samples/%s shards",
        output_dir,
        counts["train"],
        shard_counts["train"],
        counts["val"],
        shard_counts["val"],
    )


def _write_shard(output_dir: Path, fold: str, shard_number: int, items: list[dict[str, torch.Tensor]]) -> None:
    path = output_dir / fold / f"shard_{shard_number:06d}.pt"
    tensors = stack_encoded_samples(items)
    torch.save({"count": len(items), "tensors": tensors}, path)
    log.info("Wrote %s samples to %s", len(items), path)


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    raise SystemExit(main())
