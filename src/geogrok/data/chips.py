from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from geogrok.io.raster import RasterMetadata, extract_chip_to_geotiff, inspect_raster

DEFAULT_CHIPS_MANIFEST = Path("datasets/manifests/spacenet/chips.parquet")
DEFAULT_OUTPUT_ROOT = Path("datasets/chips/spacenet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raster chips from the chip manifest.")
    parser.add_argument("--chips-path", type=Path, default=DEFAULT_CHIPS_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gdal-prefix",
        type=Path,
        help="Override the local GDAL/Kakadu install prefix.",
    )
    parser.add_argument(
        "--split",
        action="append",
        help="Restrict extraction to this split. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        help="Restrict extraction to this modality. Repeat to add more modalities.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of extracted chips after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite chip files that already exist.",
    )
    return parser.parse_args(argv)


def load_chip_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_parquet(path)
    required_columns = {
        "chip_id",
        "asset_id",
        "scene_id",
        "split",
        "city",
        "modality",
        "local_path",
        "local_exists",
        "x0",
        "y0",
        "width",
        "height",
    }
    missing = required_columns.difference(table.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Chip manifest is missing required columns: {missing_columns}")
    return table


def select_chip_rows(
    chips: pd.DataFrame,
    *,
    splits: Sequence[str] | None,
    modalities: Sequence[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    selected = chips.copy()
    selected = selected[selected["local_exists"].fillna(False)]

    if splits:
        normalized = {token.lower() for token in splits}
        selected = selected[selected["split"].astype(str).str.lower().isin(normalized)]

    if modalities:
        normalized = {token.upper() for token in modalities}
        selected = selected[selected["modality"].astype(str).str.upper().isin(normalized)]

    selected = selected.sort_values(["split", "city", "scene_id", "asset_id", "y0", "x0"])
    if limit is not None:
        selected = selected.head(limit)
    return selected.reset_index(drop=True)


def build_chip_output_path(record: Mapping[str, object], output_root: Path) -> Path:
    split = slugify_path_token(record["split"])
    modality = slugify_path_token(record["modality"])
    city = slugify_path_token(record["city"])
    chip_id = str(record["chip_id"])
    return output_root / split / modality / city / f"{chip_id}.tif"


def extract_chip_dataset(
    chips: pd.DataFrame,
    *,
    output_root: Path,
    gdal_prefix: Path | None,
    overwrite: bool,
) -> pd.DataFrame:
    metadata_cache: dict[Path, RasterMetadata] = {}
    output_rows: list[dict[str, object]] = []

    for record in chips.to_dict("records"):
        source_path = Path(str(record["local_path"])).expanduser().resolve()
        output_path = build_chip_output_path(record, output_root).resolve()
        status = "existing"

        if overwrite or not output_path.exists():
            extract_chip_to_geotiff(
                source_path,
                output_path=output_path,
                x0=int(record["x0"]),
                y0=int(record["y0"]),
                width=int(record["width"]),
                height=int(record["height"]),
                prefix=gdal_prefix,
            )
            status = "written"

        source_metadata = metadata_cache.get(source_path)
        if source_metadata is None:
            source_metadata = inspect_raster(source_path, prefix=gdal_prefix)
            metadata_cache[source_path] = source_metadata

        output_rows.append(
            {
                **record,
                "output_path": str(output_path),
                "output_exists": output_path.exists(),
                "write_status": status,
                "source_driver": source_metadata.driver,
                "source_raster_x": source_metadata.raster_x,
                "source_raster_y": source_metadata.raster_y,
                "source_band_count": source_metadata.band_count,
                "source_band_dtypes": list(source_metadata.band_dtypes),
                "source_has_rpc_metadata": source_metadata.has_rpc_metadata,
            }
        )

    return pd.DataFrame(output_rows)


def write_chip_outputs(extracted: pd.DataFrame, *, output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.parquet"
    summary_path = output_root / "summary.json"

    extracted.to_parquet(index_path, index=False, compression="zstd")
    summary = {
        "rows": int(len(extracted)),
        "write_status": _counts(extracted, "write_status"),
        "split": _counts(extracted, "split"),
        "modality": _counts(extracted, "modality"),
        "city": _counts(extracted, "city"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"index": index_path, "summary": summary_path}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chips_path = args.chips_path.resolve()
    output_root = args.output_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None

    chips = load_chip_manifest(chips_path)
    selected = select_chip_rows(
        chips,
        splits=args.split,
        modalities=args.modality,
        limit=args.limit,
    )
    if selected.empty:
        raise SystemExit("No local chip rows matched the requested filters.")

    extracted = extract_chip_dataset(
        selected,
        output_root=output_root,
        gdal_prefix=gdal_prefix,
        overwrite=args.overwrite,
    )
    paths = write_chip_outputs(extracted, output_root=output_root)

    print(f"chips: {chips_path}")
    print(f"selected: {len(selected):,}")
    print(f"index: {paths['index']}")
    print(f"summary: {paths['summary']}")
    return 0


def slugify_path_token(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame or frame.empty:
        return {}
    counts = frame[column].fillna("null").astype(str).value_counts().sort_index()
    return {key: int(value) for key, value in counts.items()}


if __name__ == "__main__":
    raise SystemExit(main())
