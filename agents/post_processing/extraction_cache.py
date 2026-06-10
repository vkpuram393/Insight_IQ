"""Persistent JSON file cache for LLM-extracted column mappings.

Cache key:  "{tool_name}:{intent}"   e.g. "claims_api:claim_list"
Cache file: agents/post_processing/structure_cache.json

The cache survives process restarts and is pre-seeded with mappings for all
known (tool_name, intent) pairs so that the LLM is never called in production
for established API+intent combinations.

All file I/O exceptions are caught and logged — never propagated to callers.
A load failure returns an empty dict (LLM re-extracts and writes a fresh entry).
A save failure is logged as a warning but does not affect the current request.
"""

import json
import os
import threading
from typing import Optional

from agents.post_processing.column_mapping import ColumnDef, ColumnMapping
from core.logger import get_logger

logger = get_logger(__name__)

# Default path relative to this file so it works regardless of CWD.
_DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "structure_cache.json")


class ExtractionCache:
    """JSON file cache with a per-instance write lock.

    The lock prevents concurrent threads from interleaving read-modify-write
    cycles and corrupting the JSON file. Cross-process safety (multiple gunicorn
    workers) requires an external lock (e.g. Redis) — out of scope here since the
    pre-seeded cache covers all known combinations and new LLM writes are rare.
    """

    def __init__(self, cache_path: str = _DEFAULT_CACHE_PATH) -> None:
        self._path = cache_path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get(self, tool_name: str, intent: str) -> Optional[ColumnMapping]:
        """Return cached ColumnMapping or None on miss / corrupt entry."""
        entry = self._load().get(self._key(tool_name, intent))
        if not entry:
            return None
        try:
            return ColumnMapping(
                data_path=entry["data_path"],
                columns=[
                    ColumnDef(
                        header=c["header"],
                        path=c["path"],
                        format=c["format"],
                    )
                    for c in entry["columns"]
                ],
                tool_name=entry.get("tool_name", tool_name),
                intent=entry.get("intent", intent),
                created_at=entry.get("created_at", ""),
            )
        except (KeyError, TypeError) as exc:
            logger.warning(
                "extraction_cache: corrupt entry key=%s — %s",
                self._key(tool_name, intent),
                exc,
            )
            return None

    def set(self, mapping: ColumnMapping) -> None:
        """Write mapping to cache. Thread-safe via lock. Silently logs on save failure."""
        with self._lock:
            data = self._load()
            data[self._key(mapping.tool_name, mapping.intent)] = {
                "data_path": mapping.data_path,
                "columns": [
                    {"header": c.header, "path": c.path, "format": c.format}
                    for c in mapping.columns
                ],
                "tool_name": mapping.tool_name,
                "intent": mapping.intent,
                "created_at": mapping.created_at,
            }
            self._save(data)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _key(tool_name: str, intent: str) -> str:
        return f"{tool_name}:{intent}"

    def _load(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("extraction_cache: load failed path=%s — %s", self._path, exc)
            return {}

    def _save(self, data: dict) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("extraction_cache: save failed path=%s — %s", self._path, exc)
