from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .model_samples import MongoSequenceSampleStore
from .training_data import GlobalFeatureVectorizer, encode_sample

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Mongo sample read throughput.")
    parser.add_argument("--config", default="config.train.json")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--mongo-batch-size", type=int, default=None)
    parser.add_argument("--encode", action="store_true", help="Also encode samples into training tensors.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _load_json(args.config)
    benchmark(config, args)
    return 0


def benchmark(config: dict[str, Any], args: argparse.Namespace) -> None:
    store = MongoSequenceSampleStore(config["mongo"])
    meta = store.get_meta() or {}
    feature_names = list(meta.get("features") or [])
    target_feature_names = list(meta.get("target_features") or [])
    event_label_names = list(config["dataset"].get("event_labels") or [])
    vectorizer = GlobalFeatureVectorizer()
    mongo_batch_size = int(args.mongo_batch_size or config["dataset"].get("mongo_batch_size", 512))

    try:
        from bson import BSON
    except ImportError:
        BSON = None

    started = time.perf_counter()
    docs = 0
    bytes_read = 0
    encode_seconds = 0.0
    iterator = store.iter_samples(split=config["dataset"].get("split", "train"), batch_size=mongo_batch_size)
    for sample in tqdm(iterator, total=args.limit, desc="Mongo read"):
        docs += 1
        if BSON is not None:
            bytes_read += len(BSON.encode(sample))
        if args.encode:
            encode_started = time.perf_counter()
            encode_sample(
                sample,
                vectorizer,
                event_label_names,
                feature_names=feature_names,
                target_feature_names=target_feature_names,
            )
            encode_seconds += time.perf_counter() - encode_started
        if docs >= args.limit:
            break

    elapsed = max(time.perf_counter() - started, 1e-9)
    mb_read = bytes_read / (1024 * 1024)
    log.info(
        "Mongo read benchmark: docs=%s elapsed=%.2fs docs/s=%.2f approx_mb=%.2f approx_mb/s=%.2f encode=%.2fs mongo_batch_size=%s",
        docs,
        elapsed,
        docs / elapsed,
        mb_read,
        mb_read / elapsed if bytes_read else 0.0,
        encode_seconds,
        mongo_batch_size,
    )


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    raise SystemExit(main())
