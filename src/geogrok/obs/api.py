from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .annotations import (
    ANNOTATION_STATUS_OPTIONS,
    AnnotationStore,
    attach_pair_annotations,
    filter_pair_records_by_annotation_status,
)
from .data import (
    chip_frame_with_strings,
    default_data_paths,
    list_chip_records,
    list_pair_records,
    load_chips,
    load_pairs,
    pair_frame_with_keys,
    split_pair_key,
)
from .quicklook import chip_quicklook_from_frame, pair_quicklook_png_bytes
from .review_artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    chip_review_artifact_payload,
    load_chip_artifact_content,
    pair_review_artifact_payload,
    runtime_capabilities,
)
from .review_tables import describe_run, disagreement_response, failure_response
from .run_index import collect_run_summary_dicts, find_run_summary


def require_fastapi() -> tuple[Any, Any, Any, Any, Any]:
    try:
        fastapi = importlib.import_module("fastapi")
        fastapi_responses = importlib.import_module("fastapi.responses")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "FastAPI is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra obs`."
        ) from exc
    return (
        fastapi.FastAPI,
        fastapi.HTTPException,
        fastapi.Response,
        fastapi_responses.JSONResponse,
        fastapi.Body,
    )


def create_app(
    *,
    run_root: Path = Path("artifacts/runs"),
    gdal_prefix: Path | None = None,
    annotation_db: Path = Path("artifacts/observability/annotations/review.sqlite"),
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Any:
    FastAPI, HTTPException, Response, JSONResponse, Body = require_fastapi()
    app = FastAPI(title="GeoGrok Observability")
    data_paths = default_data_paths()
    chips = chip_frame_with_strings(load_chips(data_paths))
    pairs = pair_frame_with_keys(load_pairs(data_paths))
    annotations = AnnotationStore(annotation_db)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def api_runs() -> Any:
        return JSONResponse(collect_run_summary_dicts(run_root))

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> Any:
        run = find_run_summary(run_id, run_root)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return JSONResponse(describe_run(run))

    @app.get("/api/runs/{run_id}/failures")
    def api_run_failures(
        run_id: str,
        selection: str | None = None,
        top_k: int = 10,
        limit: int = 24,
        annotation_status: str | None = None,
        bookmarked_only: bool = False,
    ) -> Any:
        run = find_run_summary(run_id, run_root)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        try:
            payload = failure_response(run, selection_id=selection, top_k=top_k, limit=limit)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload["queue_totals"] = dict(payload["queue_counts"])
        payload["false_negatives"] = filter_pair_records_by_annotation_status(
            attach_pair_annotations(payload["false_negatives"], annotations),
            annotation_status=annotation_status,
            bookmarked_only=bookmarked_only,
        )
        payload["false_positives"] = filter_pair_records_by_annotation_status(
            attach_pair_annotations(payload["false_positives"], annotations),
            annotation_status=annotation_status,
            bookmarked_only=bookmarked_only,
        )
        payload["queue_counts"] = {
            "false_negatives": int(len(payload["false_negatives"])),
            "false_positives": int(len(payload["false_positives"])),
        }
        payload["annotation_status"] = annotation_status or "all"
        payload["bookmarked_only"] = bookmarked_only
        return JSONResponse(payload)

    @app.get("/api/runs/{run_id}/disagreements")
    def api_run_disagreements(
        run_id: str,
        limit: int = 24,
        annotation_status: str | None = None,
        bookmarked_only: bool = False,
    ) -> Any:
        run = find_run_summary(run_id, run_root)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        try:
            payload = disagreement_response(run, limit=limit)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload["queue_totals"] = dict(payload["queue_counts"])
        for key in (
            "teacher_ahead_positives",
            "student_ahead_positives",
            "student_confused_negatives",
            "teacher_confused_negatives",
        ):
            payload[key] = filter_pair_records_by_annotation_status(
                attach_pair_annotations(payload[key], annotations),
                annotation_status=annotation_status,
                bookmarked_only=bookmarked_only,
            )
        payload["queue_counts"] = {
            "teacher_ahead_positives": int(len(payload["teacher_ahead_positives"])),
            "student_ahead_positives": int(len(payload["student_ahead_positives"])),
            "student_confused_negatives": int(len(payload["student_confused_negatives"])),
            "teacher_confused_negatives": int(len(payload["teacher_confused_negatives"])),
        }
        payload["annotation_status"] = annotation_status or "all"
        payload["bookmarked_only"] = bookmarked_only
        return JSONResponse(payload)

    @app.get("/api/chips")
    def api_chips(
        city: str | None = None,
        split: str | None = None,
        modality: str | None = None,
        sensor: str | None = None,
        limit: int = 60,
    ) -> Any:
        records = list_chip_records(
            chips,
            city=city,
            split=split,
            modality=modality,
            sensor=sensor,
            limit=limit,
        )
        return JSONResponse(records)

    @app.get("/api/chip-facets")
    def api_chip_facets() -> Any:
        return JSONResponse(
            {
                "cities": sorted(chips["city"].dropna().astype(str).unique().tolist()),
                "splits": sorted(chips["split"].dropna().astype(str).unique().tolist()),
                "modalities": sorted(chips["modality"].dropna().astype(str).unique().tolist()),
                "sensors": sorted(
                    [
                        value
                        for value in chips["sensor"].dropna().astype(str).unique().tolist()
                        if value
                    ]
                ),
            }
        )

    @app.get("/api/chips/{chip_id}")
    def api_chip(chip_id: str) -> Any:
        frame = chips[chips["chip_id"] == chip_id].reset_index(drop=True)
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"Unknown chip_id: {chip_id}")
        return JSONResponse(frame.iloc[0].to_dict())

    @app.get("/api/chips/{chip_id}/image")
    def api_chip_image(
        chip_id: str,
        size: int = 256,
        pmin: float = 2.0,
        pmax: float = 98.0,
    ) -> Any:
        try:
            payload = chip_quicklook_from_frame(
                chips,
                chip_id=chip_id,
                size=size,
                gdal_prefix=gdal_prefix,
                pmin=pmin,
                pmax=pmax,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown chip_id: {chip_id}") from exc
        return Response(content=payload, media_type="image/png")

    @app.get("/api/review-artifacts/runtime")
    def api_review_artifact_runtime() -> Any:
        return JSONResponse(runtime_capabilities())

    @app.get("/api/chips/{chip_id}/review-artifact")
    def api_chip_review_artifact(chip_id: str) -> Any:
        try:
            payload = chip_review_artifact_payload(
                chips,
                chip_id=chip_id,
                artifact_root=artifact_root,
                gdal_prefix=gdal_prefix,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown chip_id: {chip_id}") from exc
        return JSONResponse(payload)

    @app.get("/api/chips/{chip_id}/review-artifact/content")
    def api_chip_review_artifact_content(chip_id: str) -> Any:
        try:
            payload, record = load_chip_artifact_content(
                chips,
                chip_id=chip_id,
                artifact_root=artifact_root,
                gdal_prefix=gdal_prefix,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown chip_id: {chip_id}") from exc
        return Response(content=payload, media_type=record.media_type)

    @app.get("/api/pairs")
    def api_pairs(
        pair_label: str | None = None,
        city: str | None = None,
        split: str | None = None,
        annotation_status: str | None = None,
        bookmarked_only: bool = False,
        limit: int = 60,
    ) -> Any:
        records = list_pair_records(
            pairs,
            pair_label=pair_label,
            city=city,
            split=split,
            limit=limit,
        )
        records = filter_pair_records_by_annotation_status(
            attach_pair_annotations(records, annotations),
            annotation_status=annotation_status,
            bookmarked_only=bookmarked_only,
        )
        return JSONResponse(records)

    @app.get("/api/pair-facets")
    def api_pair_facets() -> Any:
        return JSONResponse(
            {
                "pair_labels": sorted(pairs["pair_label"].dropna().astype(str).unique().tolist()),
                "cities": sorted(pairs["city"].dropna().astype(str).unique().tolist()),
                "splits": sorted(
                    {
                        *pairs["query_split"].dropna().astype(str).unique().tolist(),
                        *pairs["candidate_split"].dropna().astype(str).unique().tolist(),
                    }
                ),
                "annotation_statuses": list(ANNOTATION_STATUS_OPTIONS[1:]),
            }
        )

    @app.get("/api/pairs/{pair_key}")
    def api_pair(pair_key: str) -> Any:
        query_chip_id, candidate_chip_id = split_pair_key(pair_key)
        frame = pairs[
            (pairs["query_chip_id"] == query_chip_id)
            & (pairs["candidate_chip_id"] == candidate_chip_id)
        ].reset_index(drop=True)
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"Unknown pair_key: {pair_key}")
        record = frame.iloc[0].to_dict()
        annotated = attach_pair_annotations([record], annotations)[0]
        return JSONResponse(annotated)

    @app.get("/api/pairs/{pair_key}/image")
    def api_pair_image(
        pair_key: str,
        size: int = 256,
        gap: int = 12,
        pmin: float = 2.0,
        pmax: float = 98.0,
    ) -> Any:
        try:
            query_chip_id, candidate_chip_id = split_pair_key(pair_key)
            payload = pair_quicklook_png_bytes(
                chips,
                query_chip_id=query_chip_id,
                candidate_chip_id=candidate_chip_id,
                size=size,
                gap=gap,
                gdal_prefix=gdal_prefix,
                pmin=pmin,
                pmax=pmax,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown pair_key: {pair_key}") from exc
        return Response(content=payload, media_type="image/png")

    @app.get("/api/pairs/{pair_key}/review-artifact")
    def api_pair_review_artifact(pair_key: str) -> Any:
        try:
            payload = pair_review_artifact_payload(
                chips,
                pair_key=pair_key,
                artifact_root=artifact_root,
                gdal_prefix=gdal_prefix,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown pair_key: {pair_key}") from exc
        return JSONResponse(payload)

    @app.get("/api/annotations/pairs/{pair_key}")
    def api_pair_annotation(pair_key: str) -> Any:
        annotation = annotations.get_pair_annotation(pair_key)
        return JSONResponse(annotation.__dict__ if annotation is not None else None)

    required_body = Body(...)

    @app.post("/api/annotations/pairs/{pair_key}")
    def api_upsert_pair_annotation(
        pair_key: str,
        payload: dict[str, Any] = required_body,
    ) -> Any:
        query_chip_id, candidate_chip_id = split_pair_key(pair_key)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Annotation payload must be a JSON object.")
        status = str(payload.get("status", "unreviewed"))
        note = str(payload.get("note", ""))
        raw_tags = payload.get("tags", [])
        tags: list[str]
        if isinstance(raw_tags, list):
            tags = [str(value) for value in raw_tags]
        else:
            tags = []
        annotation = annotations.upsert_pair_annotation(
            query_chip_id=query_chip_id,
            candidate_chip_id=candidate_chip_id,
            status=status,
            bookmarked=(
                bool(payload["bookmarked"])
                if "bookmarked" in payload
                else None
            ),
            note=note,
            tags=tags,
        )
        return JSONResponse(annotation.__dict__)

    @app.get("/api/plan")
    def api_plan() -> dict[str, str]:
        return {
            "plan_path": str(Path("docs/observability-plan.md").resolve()),
            "mode": "scaffold",
        }

    return app


def main() -> int:
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "uvicorn is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra obs`."
        ) from exc
    uvicorn.run(create_app(), host="127.0.0.1", port=8787)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
