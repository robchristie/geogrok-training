from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import pair_key

DEFAULT_ANNOTATION_DB = Path("artifacts/observability/annotations/review.sqlite")
ANNOTATION_STATUS_OPTIONS = (
    "all",
    "unreviewed",
    "reviewed",
    "confirmed",
    "incorrect_label",
    "interesting",
    "needs_followup",
)


@dataclass(frozen=True)
class PairAnnotation:
    pair_key: str
    query_chip_id: str
    candidate_chip_id: str
    status: str
    bookmarked: bool
    tags: tuple[str, ...]
    note: str
    created_at: str
    updated_at: str


class AnnotationStore:
    def __init__(self, path: Path = DEFAULT_ANNOTATION_DB) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_annotations (
                    pair_key TEXT PRIMARY KEY,
                    query_chip_id TEXT NOT NULL,
                    candidate_chip_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    bookmarked INTEGER NOT NULL DEFAULT 0,
                    tags_json TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(pair_annotations)")
            }
            if "bookmarked" not in columns:
                connection.execute(
                    "ALTER TABLE pair_annotations ADD COLUMN bookmarked INTEGER NOT NULL DEFAULT 0"
                )
            connection.commit()

    def get_pair_annotation(self, pair_id: str) -> PairAnnotation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    pair_key,
                    query_chip_id,
                    candidate_chip_id,
                    status,
                    bookmarked,
                    tags_json,
                    note,
                    created_at,
                    updated_at
                FROM pair_annotations
                WHERE pair_key = ?
                """,
                (pair_id,),
            ).fetchone()
        return row_to_pair_annotation(row) if row is not None else None

    def upsert_pair_annotation(
        self,
        *,
        query_chip_id: str,
        candidate_chip_id: str,
        status: str,
        bookmarked: bool | None = None,
        note: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
    ) -> PairAnnotation:
        pair_id = pair_key(query_chip_id, candidate_chip_id)
        normalized_tags = tuple(sorted({tag.strip() for tag in (tags or []) if tag.strip()}))
        timestamp = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at, bookmarked FROM pair_annotations WHERE pair_key = ?",
                (pair_id,),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing is not None else timestamp
            bookmarked_value = (
                bool(existing["bookmarked"])
                if bookmarked is None and existing is not None
                else bool(bookmarked)
            )
            connection.execute(
                """
                INSERT INTO pair_annotations (
                    pair_key, query_chip_id, candidate_chip_id, status, bookmarked, tags_json, note,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair_key) DO UPDATE SET
                    status = excluded.status,
                    bookmarked = excluded.bookmarked,
                    tags_json = excluded.tags_json,
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (
                    pair_id,
                    query_chip_id,
                    candidate_chip_id,
                    status.strip() or "unreviewed",
                    int(bookmarked_value),
                    json.dumps(list(normalized_tags)),
                    note.strip(),
                    created_at,
                    timestamp,
                ),
            )
            connection.commit()
        annotation = self.get_pair_annotation(pair_id)
        if annotation is None:
            raise RuntimeError(f"Failed to upsert annotation for {pair_id}")
        return annotation

    def pair_annotation_map(self, pair_ids: list[str]) -> dict[str, PairAnnotation]:
        normalized = sorted({value for value in pair_ids if value})
        if not normalized:
            return {}
        placeholders = ", ".join("?" for _ in normalized)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    pair_key,
                    query_chip_id,
                    candidate_chip_id,
                    status,
                    bookmarked,
                    tags_json,
                    note,
                    created_at,
                    updated_at
                FROM pair_annotations
                WHERE pair_key IN ({placeholders})
                """,
                tuple(normalized),
            ).fetchall()
        annotations = [row_to_pair_annotation(row) for row in rows]
        return {annotation.pair_key: annotation for annotation in annotations}


def row_to_pair_annotation(row: sqlite3.Row) -> PairAnnotation:
    tags_raw = row["tags_json"]
    tags = tuple(str(value) for value in json.loads(tags_raw))
    return PairAnnotation(
        pair_key=str(row["pair_key"]),
        query_chip_id=str(row["query_chip_id"]),
        candidate_chip_id=str(row["candidate_chip_id"]),
        status=str(row["status"]),
        bookmarked=bool(row["bookmarked"]),
        tags=tags,
        note=str(row["note"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def attach_pair_annotations(
    records: list[dict[str, Any]],
    store: AnnotationStore,
) -> list[dict[str, Any]]:
    pair_ids = [
        str(record.get("pair_key"))
        for record in records
        if record.get("pair_key") is not None
    ]
    annotations = store.pair_annotation_map(pair_ids)
    enriched: list[dict[str, Any]] = []
    for record in records:
        pair_id = str(record.get("pair_key", ""))
        annotation = annotations.get(pair_id)
        payload = dict(record)
        payload["annotation"] = asdict(annotation) if annotation is not None else None
        enriched.append(payload)
    return enriched


def filter_pair_records_by_annotation_status(
    records: list[dict[str, Any]],
    *,
    annotation_status: str | None,
    bookmarked_only: bool = False,
) -> list[dict[str, Any]]:
    normalized = (annotation_status or "all").strip().lower()
    filtered: list[dict[str, Any]] = []
    for record in records:
        annotation = record.get("annotation")
        annotation_value = (
            str(annotation.get("status", "unreviewed")).strip().lower()
            if isinstance(annotation, dict)
            else "unreviewed"
        )
        bookmarked = bool(annotation.get("bookmarked")) if isinstance(annotation, dict) else False
        if bookmarked_only and not bookmarked:
            continue
        if normalized in ("", "all"):
            filtered.append(record)
            continue
        if normalized == "unreviewed":
            if annotation is None or annotation_value == "unreviewed":
                filtered.append(record)
        elif normalized == "reviewed":
            if annotation is not None and annotation_value != "unreviewed":
                filtered.append(record)
        elif annotation_value == normalized:
            filtered.append(record)
    return sort_pair_records_for_review(filtered)


def sort_pair_records_for_review(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple[int, float, str]:
        annotation = record.get("annotation")
        bookmarked = bool(annotation.get("bookmarked")) if isinstance(annotation, dict) else False
        updated_at = str(annotation.get("updated_at", "")) if isinstance(annotation, dict) else ""
        timestamp = 0.0
        if updated_at:
            try:
                timestamp = datetime.fromisoformat(updated_at).timestamp()
            except ValueError:
                timestamp = 0.0
        pair_id = str(record.get("pair_key", ""))
        return (0 if bookmarked else 1, -timestamp, pair_id)

    return sorted(records, key=key)
