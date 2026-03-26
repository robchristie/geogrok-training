from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ObsDataPaths:
    chips_path: Path
    pairs_path: Path


def default_data_paths() -> ObsDataPaths:
    chip_candidates = [
        Path("datasets/manifests/spacenet/chips.parquet"),
        Path("datasets/manifests/spacenet-pan-adapt-smoke/chips.parquet"),
        Path("/mnt/media/datasets/manifests/spacenet/chips.parquet"),
        Path("/mnt/media/datasets/manifests/spacenet-pan-adapt-smoke/chips.parquet"),
    ]
    pair_candidates = [
        Path("datasets/pairs/spacenet/pairs.parquet"),
        Path("datasets/pairs/spacenet-pan-adapt-smoke/pairs.parquet"),
        Path("/mnt/media/datasets/pairs/spacenet/pairs.parquet"),
        Path("/mnt/media/datasets/pairs/spacenet-pan-adapt-smoke/pairs.parquet"),
    ]
    chips_path = resolve_existing_path(chip_candidates)
    pairs_path = resolve_existing_path(pair_candidates)
    return ObsDataPaths(chips_path=chips_path, pairs_path=pairs_path)


def resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "None of the expected observability data paths exist: "
        + ", ".join(str(path) for path in candidates)
    )


def load_chips(paths: ObsDataPaths | None = None) -> pd.DataFrame:
    resolved = paths if paths is not None else default_data_paths()
    return pd.read_parquet(resolved.chips_path)


def load_pairs(paths: ObsDataPaths | None = None) -> pd.DataFrame:
    resolved = paths if paths is not None else default_data_paths()
    return pd.read_parquet(resolved.pairs_path)


def chip_frame_with_strings(frame: pd.DataFrame) -> pd.DataFrame:
    chips = frame.copy()
    for column in ("chip_id", "asset_id", "scene_id", "city", "split", "modality"):
        if column in chips.columns:
            chips[column] = chips[column].astype(str)
    if "sensor" in chips.columns:
        chips["sensor"] = chips["sensor"].fillna("").astype(str)
    if "capture_id" in chips.columns:
        chips["capture_id"] = chips["capture_id"].fillna("").astype(str)
    if "local_path" in chips.columns:
        chips["local_path"] = chips["local_path"].astype(str)
    if "acq_time" in chips.columns:
        chips["acq_time"] = chips["acq_time"].astype(str)
    return chips


def pair_frame_with_keys(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = frame.copy()
    for column in (
        "query_chip_id",
        "candidate_chip_id",
        "pair_label",
        "pair_group",
        "query_split",
        "candidate_split",
        "city",
        "modality",
    ):
        if column in pairs.columns:
            pairs[column] = pairs[column].astype(str)
    for column in ("query_acq_time", "candidate_acq_time"):
        if column in pairs.columns:
            pairs[column] = pairs[column].astype(str)
    pairs["pair_key"] = [
        pair_key(str(query_chip_id), str(candidate_chip_id))
        for query_chip_id, candidate_chip_id in zip(
            pairs["query_chip_id"],
            pairs["candidate_chip_id"],
            strict=True,
        )
    ]
    return pairs


def pair_key(query_chip_id: str, candidate_chip_id: str) -> str:
    return f"{query_chip_id}__{candidate_chip_id}"


def split_pair_key(value: str) -> tuple[str, str]:
    left, right = value.split("__", maxsplit=1)
    return left, right


def chip_record(chips: pd.DataFrame, chip_id: str) -> dict[str, Any]:
    chips_by_id = chips.set_index("chip_id", drop=False)
    if chip_id not in chips_by_id.index:
        raise KeyError(chip_id)
    record = chips_by_id.loc[chip_id]
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())
    raise KeyError(chip_id)


def list_chip_records(
    chips: pd.DataFrame,
    *,
    city: str | None = None,
    split: str | None = None,
    modality: str | None = None,
    sensor: str | None = None,
    limit: int = 60,
) -> list[dict[str, Any]]:
    frame = chips.copy()
    if city:
        frame = frame[frame["city"] == city]
    if split:
        frame = frame[frame["split"] == split]
    if modality:
        frame = frame[frame["modality"] == modality]
    if sensor:
        frame = frame[frame["sensor"] == sensor]
    frame = frame.head(limit).reset_index(drop=True)
    return frame.to_dict(orient="records")


def list_pair_records(
    pairs: pd.DataFrame,
    *,
    pair_label: str | None = None,
    city: str | None = None,
    split: str | None = None,
    limit: int = 60,
) -> list[dict[str, Any]]:
    frame = pairs.copy()
    if pair_label:
        frame = frame[frame["pair_label"] == pair_label]
    if city:
        frame = frame[frame["city"] == city]
    if split:
        frame = frame[(frame["query_split"] == split) | (frame["candidate_split"] == split)]
    frame = frame.head(limit).reset_index(drop=True)
    return frame.to_dict(orient="records")
