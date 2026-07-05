from __future__ import annotations

import logging
import time
from typing import Any, Iterator

log = logging.getLogger(__name__)


def _load_pymongo() -> Any:
    try:
        import pymongo
    except ImportError as exc:
        raise RuntimeError("pymongo is required to load ML samples from MongoDB.") from exc
    return pymongo


class MongoSequenceSampleStore:
    """Small training-facing reader for samples produced by MongoMLWriter.

    It intentionally returns plain dictionaries/lists so PyTorch, JAX or NumPy code can decide
    its own tensor conversion and batching strategy.
    """

    def __init__(self, mongo_cfg: dict[str, Any]) -> None:
        pymongo = _load_pymongo()
        self.client = pymongo.MongoClient(
            mongo_cfg["uri"],
            serverSelectionTimeoutMS=int(mongo_cfg.get("server_selection_timeout_ms", 5000)),
            connectTimeoutMS=int(mongo_cfg.get("connect_timeout_ms", 5000)),
            socketTimeoutMS=int(mongo_cfg.get("socket_timeout_ms", 120000)),
            retryReads=bool(mongo_cfg.get("retry_reads", True)),
        )
        self.db = self.client[mongo_cfg["db"]]
        self.samples = self.db[mongo_cfg.get("samples_collection", "ml_samples")]
        self.meta = self.db[mongo_cfg.get("meta_collection", "ml_meta")]
        self.cursor_retries = int(mongo_cfg.get("cursor_retries", 10))
        self.cursor_retry_sleep_seconds = float(mongo_cfg.get("cursor_retry_sleep_seconds", 5.0))

    def iter_samples(
        self,
        *,
        split: str = "train",
        match_id: str | None = None,
        round_number: int | None = None,
        batch_size: int = 512,
        sort_samples: bool = False,
    ) -> Iterator[dict[str, Any]]:
        query: dict[str, Any] = {"split": split}
        if match_id is not None:
            query["match_id"] = match_id
        if round_number is not None:
            query["round_number"] = round_number
        projection = {
            "_id": 1,
            "match_id": 1,
            "map": 1,
            "round_number": 1,
            "tick_anchor": 1,
            "player_ids": 1,
            "history": 1,
            "history_mask": 1,
            "future": 1,
            "future_mask": 1,
            "future_alive": 1,
            "global_features": 1,
            "labels": 1,
        }
        if sort_samples:
            projection["_id"] = 0
            cursor = self.samples.find(query, projection=projection, batch_size=batch_size)
            cursor = cursor.sort([("match_id", 1), ("round_number", 1), ("tick_anchor", 1)])
            yield from cursor
            return
        yield from self._iter_samples_resumable(query, projection, batch_size)

    def _iter_samples_resumable(
        self,
        query: dict[str, Any],
        projection: dict[str, int],
        batch_size: int,
    ) -> Iterator[dict[str, Any]]:
        pymongo = _load_pymongo()
        recoverable_errors = (
            pymongo.errors.AutoReconnect,
            pymongo.errors.ConnectionFailure,
            pymongo.errors.NetworkTimeout,
            pymongo.errors.OperationFailure,
            pymongo.errors.PyMongoError,
        )
        last_id = None
        retries = 0
        while True:
            page_query = dict(query)
            if last_id is not None:
                page_query["_id"] = {"$gt": last_id}
            try:
                cursor = self.samples.find(page_query, projection=projection, batch_size=batch_size).sort("_id", 1)
                yielded = 0
                for doc in cursor:
                    last_id = doc.get("_id")
                    doc.pop("_id", None)
                    retries = 0
                    yielded += 1
                    yield doc
                if yielded == 0:
                    return
            except recoverable_errors as exc:
                retries += 1
                if retries > self.cursor_retries:
                    raise
                log.warning(
                    "Mongo cursor interrupted after _id=%s; retrying %s/%s in %.1fs: %s",
                    last_id,
                    retries,
                    self.cursor_retries,
                    self.cursor_retry_sleep_seconds,
                    exc,
                )
                time.sleep(self.cursor_retry_sleep_seconds)

    def get_meta(self, dataset_id: str = "default") -> dict[str, Any] | None:
        return self.meta.find_one({"dataset_id": dataset_id}, projection={"_id": 0})

    def count_samples(self, *, split: str = "train") -> int:
        return int(self.samples.count_documents({"split": split}))
