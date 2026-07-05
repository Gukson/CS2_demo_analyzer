from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .faceit import collect_faceit_tasks
from .mongo_ml_writer import DemoDataQualityError, MongoMLWriter
from .parser_backend import parse_demo_header, parse_demo_to_frames
from .rounds import assign_round_numbers, detect_rounds
from .state import ProcessedState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseTask:
    demo_path: str
    match_id: str
    source: str
    match_info: dict[str, Any]
    delete_after_success: bool


def run_pipeline(config: dict[str, Any]) -> None:
    if config.get("output", {}).get("mode") != "mongo_ml":
        raise ValueError("Only output.mode='mongo_ml' is implemented.")
    parsing_cfg = dict(config["parsing"])
    dataset_cfg = dict(config["dataset"])
    dataset_cfg["sample_every_ticks"] = int(parsing_cfg.get("sample_every_ticks", 16))
    writer = MongoMLWriter(config["output"], dataset_cfg)
    state = ProcessedState(config.get("state", {}).get("processed_matches_file", "state/processed_matches.json"))
    tasks = collect_tasks(config)
    runnable = []
    for task in tasks:
        if writer.match_exists(task.match_id):
            log.info("Skipping already processed match %s (%s)", task.match_id, task.demo_path)
            summary = writer.match_summary(task.match_id) or {}
            _finalize_success(task, state, summary)
            continue
        if state.is_processed(task.match_id):
            log.warning(
                "State file says match %s was processed, but Mongo has no complete match document. Reprocessing %s.",
                task.match_id,
                task.demo_path,
            )
        runnable.append(task)
    if not runnable:
        log.info("No demos to process.")
        return
    max_workers = max(1, int(parsing_cfg.get("max_workers", 1)))
    if max_workers == 1:
        for task in tqdm(runnable, total=len(runnable), desc="Parsing demos"):
            try:
                result = _parse_task(task, parsing_cfg)
                stats = _write_result_to_mongo(result, writer, parsing_cfg)
                _finalize_success(result["task"], state, stats)
            except DemoDataQualityError as exc:
                _finalize_demo_failure(task, state, config, exc)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_and_write_task, task, config) for task in runnable]
            try:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parsing demos"):
                    result = future.result()
                    task = ParseTask(**result["task"])
                    if result.get("status") == "skipped":
                        _finalize_demo_failure(task, state, config, result["error"])
                    else:
                        _finalize_success(task, state, result["stats"])
            except Exception:
                for pending in futures:
                    pending.cancel()
                log.exception("Pipeline aborted. Pending parse jobs were cancelled where possible.")
                raise


def collect_tasks(config: dict[str, Any]) -> list[ParseTask]:
    input_cfg = config["input"]
    mode = input_cfg.get("mode", "local")
    if mode == "local":
        return collect_local_tasks(input_cfg)
    if mode == "faceit":
        return [
            ParseTask(
                demo_path=str(task.demo_path),
                match_id=task.match_id,
                source=task.source,
                match_info=task.match_info,
                delete_after_success=bool(input_cfg.get("local_delete_processed_demo", False)),
            )
            for task in collect_faceit_tasks(input_cfg)
        ]
    raise ValueError(f"Unknown input mode: {mode}")


def collect_local_tasks(input_cfg: dict[str, Any]) -> list[ParseTask]:
    demo_dir = Path(input_cfg.get("local_demo_dir", "demos/extracted"))
    pattern = "**/*.dem" if input_cfg.get("local_recursive", True) else "*.dem"
    paths = sorted(demo_dir.glob(pattern))
    paths = _filter_paths_by_date(paths, input_cfg.get("date_from"), input_cfg.get("date_to"))
    last_n = input_cfg.get("last_n")
    if last_n:
        paths = paths[-int(last_n) :]
    tasks: list[ParseTask] = []
    for demo_path in paths:
        header = parse_demo_header(demo_path)
        match_id = local_match_id(demo_path, header, input_cfg.get("local_match_id_strategy", "header"))
        tasks.append(
            ParseTask(
                demo_path=str(demo_path),
                match_id=match_id,
                source="local",
                match_info={"demo_path": str(demo_path), "file_name": demo_path.name},
                delete_after_success=bool(input_cfg.get("local_delete_processed_demo", True)),
            )
        )
    return tasks


def _filter_paths_by_date(paths: list[Path], date_from: str | None, date_to: str | None) -> list[Path]:
    if not date_from and not date_to:
        return paths
    start = _parse_date(date_from, end_of_day=False) if date_from else None
    end = _parse_date(date_to, end_of_day=True) if date_to else None
    filtered = []
    for path in paths:
        modified = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        if start and modified < start:
            continue
        if end and modified > end:
            continue
        filtered.append(path)
    return filtered


def _parse_date(value: str, *, end_of_day: bool) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    if len(value) == 10:
        time = dt.time.max if end_of_day else dt.time.min
        parsed = dt.datetime.combine(parsed.date(), time, tzinfo=parsed.tzinfo)
    return parsed


def local_match_id(demo_path: str | Path, header: dict[str, Any], strategy: str = "header") -> str:
    path = Path(demo_path)
    if strategy == "filename":
        raw = {"file_name": path.name}
    elif strategy == "sha1":
        raw = {"file_sha1_1mb": header.get("file_sha1_1mb"), "file_name": path.name}
    else:
        raw = {
            "file_name": path.name,
            "map": header.get("map_name") or header.get("map") or header.get("network_protocol_map"),
            "demo_file_stamp": header.get("demo_file_stamp"),
            "network_protocol": header.get("network_protocol"),
            "server_name": header.get("server_name"),
            "client_name": header.get("client_name"),
            "file_sha1_1mb": header.get("file_sha1_1mb"),
        }
    encoded = json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
    return "local_" + hashlib.sha1(encoded).hexdigest()


def _parse_task(task: ParseTask, parsing_cfg: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    log.info("Parsing %s", task.demo_path)
    parsed = parse_demo_to_frames(task.demo_path, parsing_cfg)
    log.info(
        "Parsed raw demo %s: frame_rows=%s events=%s event_types=%s players=%s in %.1fs",
        task.demo_path,
        len(parsed["frames"]),
        len(parsed.get("events") or []),
        len(parsed.get("event_names") or []),
        len(parsed.get("players") or []),
        time.perf_counter() - started,
    )
    rounds = detect_rounds(parsed.get("events") or [], parsed["frames"])
    frames = assign_round_numbers(parsed["frames"], rounds)
    assigned = sum(1 for frame in frames if frame.get("round_number") is not None)
    log.info(
        "Detected rounds for %s: rounds=%s frame_rows_with_round=%s/%s",
        task.demo_path,
        len(rounds),
        assigned,
        len(frames),
    )
    return {
        "task": task,
        "frames": frames,
        "header": parsed["header"],
        "players": parsed["players"],
        "event_names": parsed["event_names"],
        "rounds": rounds,
    }


def _process_and_write_task(task: ParseTask, config: dict[str, Any]) -> dict[str, Any]:
    parsing_cfg = dict(config["parsing"])
    dataset_cfg = dict(config["dataset"])
    dataset_cfg["sample_every_ticks"] = int(parsing_cfg.get("sample_every_ticks", 16))
    writer = MongoMLWriter(config["output"], dataset_cfg)
    try:
        result = _parse_task(task, parsing_cfg)
        stats = _write_result_to_mongo(result, writer, parsing_cfg)
    except DemoDataQualityError as exc:
        return {"status": "skipped", "task": asdict(task), "error": str(exc)}
    return {"status": "complete", "task": asdict(task), "stats": stats}


def _write_result_to_mongo(
    result: dict[str, Any],
    writer: MongoMLWriter,
    parsing_cfg: dict[str, Any],
) -> dict[str, Any]:
    task: ParseTask = result["task"]
    return writer.write_match(
        match_id=task.match_id,
        source=task.source,
        match_info=task.match_info,
        demo_meta={
            **result["header"],
            "event_names": result["event_names"],
            "parsed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        frames=result["frames"],
        rounds=result["rounds"],
        players=result["players"],
        parsing_cfg=parsing_cfg,
    )


def _finalize_success(task: ParseTask, state: ProcessedState, stats: dict[str, Any]) -> None:
    state.mark_processed(
        task.match_id,
        {
            "source": task.source,
            "demo_path": task.demo_path,
            "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **stats,
        },
    )
    if task.delete_after_success:
        Path(task.demo_path).unlink(missing_ok=True)
    log.info("Processed %s: %s samples", task.match_id, stats.get("sample_count"))


def _finalize_demo_failure(
    task: ParseTask,
    state: ProcessedState,
    config: dict[str, Any],
    error: Exception | str,
) -> None:
    error_text = str(error)
    rejected_path = _move_rejected_demo(task, config)
    state.mark_failed(
        task.match_id,
        {
            "source": task.source,
            "demo_path": task.demo_path,
            "rejected_path": str(rejected_path) if rejected_path else None,
            "failed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": error_text,
        },
    )
    log.warning("Rejected %s (%s): %s", task.match_id, rejected_path or task.demo_path, error_text)


def _move_rejected_demo(task: ParseTask, config: dict[str, Any]) -> Path | None:
    if task.source != "local":
        return None
    source = Path(task.demo_path)
    if not source.exists():
        return None
    rejected_dir = Path(config.get("input", {}).get("local_rejected_demo_dir", "demos/rejected"))
    rejected_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(rejected_dir / source.name)
    source.replace(destination)
    return destination


def _unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
