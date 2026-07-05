from __future__ import annotations

import argparse
import logging
from typing import Any

from tqdm import tqdm

log = logging.getLogger(__name__)


def _load_pymongo() -> Any:
    try:
        import pymongo
    except ImportError as exc:
        raise RuntimeError("pymongo is required. Install with `pip install pymongo`.") from exc
    return pymongo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync or move parsed CS2 ML data between MongoDB instances.")
    parser.add_argument("--source-uri", default="mongodb://localhost:27017")
    parser.add_argument("--target-uri", default="mongodb://192.168.1.106:27017")
    parser.add_argument("--db", default="cs2_demo_ml")
    parser.add_argument("--matches-collection", default="ml_matches")
    parser.add_argument("--samples-collection", default="ml_samples")
    parser.add_argument("--meta-collection", default="ml_meta")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--match-id", action="append", help="Sync only this match_id. Can be passed multiple times.")
    parser.add_argument("--limit", type=int, help="Sync at most N matches.")
    parser.add_argument("--force", action="store_true", help="Rewrite target matches even if they are already complete.")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Delete source samples and match documents after they are verified on the target.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be copied.")
    parser.add_argument("--skip-meta", action="store_true", help="Do not copy ml_meta documents.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stats = sync_mongo(args)
    log.info(
        "Sync finished: copied=%s skipped=%s moved=%s samples=%s dry_run=%s",
        stats["matches_copied"],
        stats["matches_skipped"],
        stats["matches_moved"],
        stats["samples_copied"],
        args.dry_run,
    )
    return 0


def sync_mongo(args: argparse.Namespace) -> dict[str, int]:
    if args.source_uri == args.target_uri:
        raise ValueError("source-uri and target-uri are identical; refusing to sync a MongoDB into itself.")

    pymongo = _load_pymongo()
    source_client = pymongo.MongoClient(args.source_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    target_client = pymongo.MongoClient(
        args.target_uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=120000,
        retryWrites=True,
    )
    source_client.admin.command("ping")
    target_client.admin.command("ping")

    source_db = source_client[args.db]
    target_db = target_client[args.db]
    source_matches = source_db[args.matches_collection]
    source_samples = source_db[args.samples_collection]
    target_matches = target_db[args.matches_collection]
    target_samples = target_db[args.samples_collection]

    if not args.dry_run:
        _ensure_target_indexes(target_matches, target_samples, target_db[args.meta_collection])

    query: dict[str, Any] = {"parse_status": "complete", "sample_count": {"$gt": 0}}
    if args.match_id:
        query["match_id"] = {"$in": args.match_id}

    total = source_matches.count_documents(query)
    if args.limit is not None:
        total = min(total, args.limit)
    cursor = source_matches.find(query).sort("match_id", 1)
    if args.limit is not None:
        cursor = cursor.limit(args.limit)

    copied_matches = 0
    skipped_matches = 0
    moved_matches = 0
    copied_samples = 0
    for match_doc in tqdm(cursor, total=total, desc="Sync matches"):
        match_id = match_doc["match_id"]
        sample_count = int(match_doc.get("sample_count") or 0)
        target_doc = target_matches.find_one(
            {"match_id": match_id, "parse_status": "complete"},
            projection={"_id": 1, "sample_count": 1, "round_count": 1},
        )
        if target_doc and not args.force:
            target_sample_count = target_samples.count_documents({"match_id": match_id})
            if int(target_doc.get("sample_count") or 0) == sample_count and target_sample_count == sample_count:
                skipped_matches += 1
                if args.move and not args.dry_run:
                    _delete_source_match(source_matches, source_samples, match_id)
                    moved_matches += 1
                continue

        log.info("%s %s: samples=%s", "Moving" if args.move else "Copying", match_id, sample_count)
        if args.dry_run:
            copied_matches += 1
            copied_samples += sample_count
            continue

        target_samples.delete_many({"match_id": match_id})
        inserted = _copy_samples(source_samples, target_samples, match_id, args.batch_size)
        if inserted != sample_count:
            raise RuntimeError(f"Copied {inserted} samples for {match_id}, expected {sample_count}")

        replacement = _without_id(match_doc)
        target_matches.replace_one({"match_id": match_id}, replacement, upsert=True)
        _verify_match(target_matches, target_samples, match_id, sample_count)
        if args.move:
            _delete_source_match(source_matches, source_samples, match_id)
            moved_matches += 1
        copied_matches += 1
        copied_samples += inserted

    if not args.skip_meta and not args.dry_run:
        _copy_meta(source_db[args.meta_collection], target_db[args.meta_collection])

    return {
        "matches_copied": copied_matches,
        "matches_skipped": skipped_matches,
        "matches_moved": moved_matches,
        "samples_copied": copied_samples,
    }


def _copy_samples(source_samples: Any, target_samples: Any, match_id: str, batch_size: int) -> int:
    inserted = 0
    batch: list[dict[str, Any]] = []
    cursor = source_samples.find({"match_id": match_id}).batch_size(max(1, batch_size))
    for sample_doc in cursor:
        batch.append(_without_id(sample_doc))
        if len(batch) >= batch_size:
            inserted += _insert_batch(target_samples, batch)
            batch = []
    if batch:
        inserted += _insert_batch(target_samples, batch)
    return inserted


def _insert_batch(collection: Any, batch: list[dict[str, Any]]) -> int:
    result = collection.insert_many(batch, ordered=True)
    return len(result.inserted_ids)


def _copy_meta(source_meta: Any, target_meta: Any) -> None:
    for meta_doc in source_meta.find({}):
        replacement = _without_id(meta_doc)
        dataset_id = replacement.get("dataset_id")
        if dataset_id is None:
            continue
        target_meta.replace_one({"dataset_id": dataset_id}, replacement, upsert=True)


def _delete_source_match(source_matches: Any, source_samples: Any, match_id: str) -> None:
    source_samples.delete_many({"match_id": match_id})
    source_matches.delete_one({"match_id": match_id})


def _verify_match(target_matches: Any, target_samples: Any, match_id: str, expected_samples: int) -> None:
    match_doc = target_matches.find_one(
        {"match_id": match_id, "parse_status": "complete"},
        projection={"_id": 1, "sample_count": 1},
    )
    if not match_doc:
        raise RuntimeError(f"Target verification failed for {match_id}: match document missing")
    if int(match_doc.get("sample_count") or -1) != expected_samples:
        raise RuntimeError(
            f"Target verification failed for {match_id}: "
            f"match sample_count={match_doc.get('sample_count')} expected={expected_samples}"
        )
    stored_samples = target_samples.count_documents({"match_id": match_id})
    if stored_samples != expected_samples:
        raise RuntimeError(
            f"Target verification failed for {match_id}: samples={stored_samples} expected={expected_samples}"
        )


def _ensure_target_indexes(matches: Any, samples: Any, meta: Any) -> None:
    matches.create_index("match_id", unique=True)
    samples.create_index([("match_id", 1), ("round_number", 1), ("tick_anchor", 1)])
    samples.create_index([("split", 1), ("match_id", 1)])
    samples.create_index([("match_id", 1), ("round_number", 1), ("tick_anchor", 1), ("_id", 1)])
    meta.create_index("dataset_id", unique=True)


def _without_id(doc: dict[str, Any]) -> dict[str, Any]:
    clean = dict(doc)
    clean.pop("_id", None)
    return clean


if __name__ == "__main__":
    raise SystemExit(main())
