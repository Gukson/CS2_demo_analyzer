from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProcessedState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {"processed": {}, "failed": {}}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            self.data = loaded
            self.data.setdefault("processed", {})
            self.data.setdefault("failed", {})

    def is_processed(self, match_id: str) -> bool:
        return match_id in self.data.get("processed", {})

    def mark_processed(self, match_id: str, payload: dict[str, Any]) -> None:
        self.data.setdefault("processed", {})[match_id] = payload
        self.data.setdefault("failed", {}).pop(match_id, None)
        self._save()

    def mark_failed(self, match_id: str, payload: dict[str, Any]) -> None:
        self.data.setdefault("failed", {})[match_id] = payload
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=True)
        tmp_path.replace(self.path)
