from __future__ import annotations

from pathlib import Path

from geogrok.obs.annotations import (
    AnnotationStore,
    attach_pair_annotations,
    filter_pair_records_by_annotation_status,
)


def test_annotation_store_upserts_and_reads_pair_annotations(tmp_path: Path):
    store = AnnotationStore(tmp_path / "review.sqlite")

    annotation = store.upsert_pair_annotation(
        query_chip_id="chip_a",
        candidate_chip_id="chip_b",
        status="confirmed",
        note="Looks correct.",
        tags=["geometry", "clear"],
    )

    fetched = store.get_pair_annotation("chip_a__chip_b")

    assert fetched is not None
    assert fetched.pair_key == "chip_a__chip_b"
    assert fetched.status == "confirmed"
    assert fetched.bookmarked is False
    assert fetched.note == "Looks correct."
    assert fetched.tags == ("clear", "geometry")
    assert annotation.pair_key == fetched.pair_key


def test_attach_pair_annotations_enriches_pair_records(tmp_path: Path):
    store = AnnotationStore(tmp_path / "review.sqlite")
    store.upsert_pair_annotation(
        query_chip_id="chip_a",
        candidate_chip_id="chip_b",
        status="interesting",
        note="Worth reviewing again.",
    )
    records = [
        {
            "pair_key": "chip_a__chip_b",
            "query_chip_id": "chip_a",
            "candidate_chip_id": "chip_b",
        },
        {
            "pair_key": "chip_c__chip_d",
            "query_chip_id": "chip_c",
            "candidate_chip_id": "chip_d",
        },
    ]

    enriched = attach_pair_annotations(records, store)

    assert enriched[0]["annotation"] is not None
    assert enriched[0]["annotation"]["status"] == "interesting"
    assert enriched[1]["annotation"] is None


def test_filter_pair_records_by_annotation_status_handles_unreviewed_and_reviewed():
    records = [
        {
            "pair_key": "a__b",
            "annotation": {"status": "confirmed"},
        },
        {
            "pair_key": "c__d",
            "annotation": None,
        },
        {
            "pair_key": "e__f",
            "annotation": {"status": "interesting"},
        },
    ]

    unreviewed = filter_pair_records_by_annotation_status(records, annotation_status="unreviewed")
    reviewed = filter_pair_records_by_annotation_status(records, annotation_status="reviewed")
    interesting = filter_pair_records_by_annotation_status(records, annotation_status="interesting")

    assert [record["pair_key"] for record in unreviewed] == ["c__d"]
    assert [record["pair_key"] for record in reviewed] == ["a__b", "e__f"]
    assert [record["pair_key"] for record in interesting] == ["e__f"]


def test_bookmarked_annotations_round_trip_and_filter(tmp_path: Path):
    store = AnnotationStore(tmp_path / "review.sqlite")
    annotation = store.upsert_pair_annotation(
        query_chip_id="bookmark_a",
        candidate_chip_id="bookmark_b",
        status="interesting",
        bookmarked=True,
        note="Saved for later.",
    )

    assert annotation.bookmarked is True

    records = [
        {"pair_key": "bookmark_a__bookmark_b", "annotation": annotation.__dict__},
        {"pair_key": "plain_a__plain_b", "annotation": None},
    ]
    bookmarked = filter_pair_records_by_annotation_status(
        records,
        annotation_status="all",
        bookmarked_only=True,
    )
    assert [record["pair_key"] for record in bookmarked] == ["bookmark_a__bookmark_b"]
