from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class FaceitTask:
    demo_path: Path
    match_id: str
    source: str
    match_info: dict[str, Any]


def collect_faceit_tasks(input_cfg: dict[str, Any]) -> list[FaceitTask]:
    """Best-effort FACEIT downloader hook.

    FACEIT setups differ by endpoint, token scope and archive format, so the local parser remains
    the stable default. This hook supports a pragmatic config shape:
    input.faceit.matches = [{"match_id": "...", "demo_url": "..."}]
    input.faceit.download_dir = "demos/faceit"
    input.faceit.api_key_env = "FACEIT_API_KEY"
    """
    faceit_cfg = input_cfg.get("faceit") or {}
    matches = faceit_cfg.get("matches") or []
    if not matches:
        log.warning("input-mode=faceit requested but no input.faceit.matches were configured")
        return []
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for FACEIT downloads. Install with `pip install '.[faceit]'`.") from exc
    download_dir = Path(faceit_cfg.get("download_dir", "demos/faceit"))
    download_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[FaceitTask] = []
    for item in matches:
        match_id = str(item["match_id"])
        demo_url = item.get("demo_url")
        if not demo_url:
            log.warning("Skipping FACEIT match %s without demo_url", match_id)
            continue
        target = download_dir / f"{match_id}.dem"
        if not target.exists():
            response = requests.get(demo_url, timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        tasks.append(FaceitTask(demo_path=target, match_id=match_id, source="faceit", match_info=item))
    return tasks
