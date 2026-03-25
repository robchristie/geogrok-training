from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from geogrok.data.chips import load_chip_manifest, select_chip_rows
from geogrok.io.raster import RasterArrayChip, read_chip_array


@dataclass(frozen=True)
class ChipRecord:
    chip_id: str
    asset_id: str
    capture_id: str | None
    scene_id: str
    split: str
    city: str
    modality: str
    sensor: str | None
    local_path: Path
    x0: int
    y0: int
    width: int
    height: int


@dataclass(frozen=True)
class ChipSample:
    record: ChipRecord
    chip: RasterArrayChip


class OnDemandChipDataset:
    def __init__(
        self,
        chips: pd.DataFrame,
        *,
        gdal_prefix: str | Path | None = None,
    ) -> None:
        self._chips = chips.reset_index(drop=True)
        self._gdal_prefix = Path(gdal_prefix).resolve() if gdal_prefix is not None else None

    @classmethod
    def from_manifest(
        cls,
        chips_path: str | Path,
        *,
        splits: tuple[str, ...] | None = None,
        modalities: tuple[str, ...] | None = None,
        limit: int | None = None,
        gdal_prefix: str | Path | None = None,
    ) -> OnDemandChipDataset:
        chips = load_chip_manifest(Path(chips_path))
        selected = select_chip_rows(
            chips,
            splits=splits,
            modalities=modalities,
            limit=limit,
        )
        return cls(selected, gdal_prefix=gdal_prefix)

    def __len__(self) -> int:
        return len(self._chips)

    def record(self, index: int) -> ChipRecord:
        row = self._chips.iloc[index].to_dict()
        return chip_record_from_mapping(row)

    def sample(self, index: int) -> ChipSample:
        record = self.record(index)
        chip = read_chip_array(
            record.local_path,
            x0=record.x0,
            y0=record.y0,
            width=record.width,
            height=record.height,
            prefix=self._gdal_prefix,
        )
        return ChipSample(record=record, chip=chip)

    def records_frame(self) -> pd.DataFrame:
        return self._chips.copy()


def chip_record_from_mapping(record: dict[str, Any]) -> ChipRecord:
    return ChipRecord(
        chip_id=str(record["chip_id"]),
        asset_id=str(record["asset_id"]),
        capture_id=_optional_string(record.get("capture_id")),
        scene_id=str(record["scene_id"]),
        split=str(record["split"]),
        city=str(record["city"]),
        modality=str(record["modality"]),
        sensor=_optional_string(record.get("sensor")),
        local_path=Path(str(record["local_path"])).expanduser().resolve(),
        x0=int(record["x0"]),
        y0=int(record["y0"]),
        width=int(record["width"]),
        height=int(record["height"]),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return str(value)
