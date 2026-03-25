from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

import pandas as pd

DEFAULT_METADATA_CANDIDATES = (
    Path("tools/spacenet/artifacts/s3-spacenet-dataset-images-ds"),
    Path("tools/spacenet/artifacts/s3-spacenet-dataset-images.parquet"),
)
DEFAULT_OUTPUT_ROOT = Path("datasets/manifests/spacenet")
DEFAULT_DOWNLOAD_ROOT = Path("datasets/spacenet.ai")
DEFAULT_VAL_CITIES = ("Omaha",)
DEFAULT_TEST_CITIES = ("UCSD",)
DEFAULT_CHIP_MODALITIES = ("PAN",)
DEFAULT_CHIP_SIZE = 1024
DEFAULT_CHIP_STRIDE = 1024
BUCKET = "spacenet-dataset"

AOI_RE = re.compile(r"(AOI_\d+_[^/]+)")
PRODUCT_RE = re.compile(r"-(P[0-9][A-Z0-9]+)-", re.IGNORECASE)
SATELLITE_CITY_RE = re.compile(r"Satellite-Images/([^/]+)/", re.IGNORECASE)
SCENE_SUFFIX_RE = re.compile(r"(_lv1)?\.(ntf|tif)$", re.IGNORECASE)
SENSOR_RE = re.compile(r"(WV\d)", re.IGNORECASE)

KNOWN_MODALITIES = (
    "PS-RGBNIR",
    "SAR-MAG-POL",
    "PS-RGB",
    "PS-MS",
    "RGBNIR",
    "PAN",
    "MS",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized dataset manifests.")
    parser.add_argument(
        "--metadata-path", type=Path, help="Parquet file or parquet dataset directory."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
        help="Local root that mirrors the spacenet.ai bucket key structure.",
    )
    parser.add_argument(
        "--chip-modality",
        action="append",
        default=list(DEFAULT_CHIP_MODALITIES),
        help="Create chips for this modality. Repeat to add more modalities.",
    )
    parser.add_argument("--chip-size", type=int, default=DEFAULT_CHIP_SIZE)
    parser.add_argument("--chip-stride", type=int, default=DEFAULT_CHIP_STRIDE)
    parser.add_argument(
        "--val-city",
        action="append",
        default=list(DEFAULT_VAL_CITIES),
        help="Assign this city to the validation split. Repeat to add more cities.",
    )
    parser.add_argument(
        "--test-city",
        action="append",
        default=list(DEFAULT_TEST_CITIES),
        help="Assign this city to the test split. Repeat to add more cities.",
    )
    parser.add_argument(
        "--limit-assets",
        type=int,
        help="Limit the normalized asset rows before scene and chip generation.",
    )
    return parser.parse_args(argv)


def resolve_metadata_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DEFAULT_METADATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No metadata parquet found under tools/spacenet/artifacts/")


def load_metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_asset_manifest(
    metadata: pd.DataFrame,
    *,
    download_root: Path,
    val_cities: Sequence[str],
    test_cities: Sequence[str],
    source_metadata_path: Path,
) -> pd.DataFrame:
    table = metadata.copy()
    if "key" not in table:
        raise ValueError("Metadata table must contain a 'key' column.")

    table["key"] = table["key"].astype(str)
    for column in (
        "scene_id",
        "acq_time",
        "acq_time_start",
        "acq_time_end",
        "driver",
        "sensor_hint",
        "geom_source",
        "geom",
        "size",
        "last_modified",
        "etag",
        "raster_x",
        "raster_y",
    ):
        if column not in table:
            table[column] = None

    normalized_val_cities = {normalize_token(city) for city in val_cities}
    normalized_test_cities = {normalize_token(city) for city in test_cities}

    asset_rows: list[dict[str, object]] = []
    for record in table.to_dict("records"):
        key = str(record["key"])
        scene_id = normalize_scene_id(record.get("scene_id"), key)
        city = extract_city(key)
        local_path = (download_root / key).resolve()
        local_exists = local_path.exists()
        sensor = normalize_sensor(record.get("sensor_hint"), key)
        modality = extract_modality(key)

        asset_rows.append(
            {
                "asset_id": stable_id("asset", key),
                "capture_id": stable_id("capture", infer_capture_id(key)),
                "scene_id": scene_id,
                "key": key,
                "remote_uri": f"s3://{BUCKET}/{key}",
                "local_path": str(local_path),
                "local_exists": local_exists,
                "city": city,
                "area_name": extract_area_name(key),
                "split": assign_split(city, normalized_val_cities, normalized_test_cities),
                "modality": modality,
                "sensor": sensor,
                "product_code": extract_product_code(key),
                "file_extension": Path(key).suffix.lower(),
                "asset_preference_rank": asset_preference_rank(key),
                "size": record.get("size"),
                "last_modified": record.get("last_modified"),
                "etag": record.get("etag"),
                "acq_time": record.get("acq_time"),
                "acq_time_start": record.get("acq_time_start"),
                "acq_time_end": record.get("acq_time_end"),
                "driver": record.get("driver"),
                "geom_source": record.get("geom_source"),
                "geom": record.get("geom"),
                "raster_x": record.get("raster_x"),
                "raster_y": record.get("raster_y"),
                "source_metadata_path": str(source_metadata_path.resolve()),
            }
        )

    assets = pd.DataFrame(asset_rows)
    ordered_columns = [
        "asset_id",
        "capture_id",
        "scene_id",
        "key",
        "remote_uri",
        "local_path",
        "local_exists",
        "city",
        "area_name",
        "split",
        "modality",
        "sensor",
        "product_code",
        "file_extension",
        "asset_preference_rank",
        "size",
        "last_modified",
        "etag",
        "acq_time",
        "acq_time_start",
        "acq_time_end",
        "driver",
        "geom_source",
        "geom",
        "raster_x",
        "raster_y",
        "source_metadata_path",
    ]
    return assets[ordered_columns]


def build_scene_manifest(assets: pd.DataFrame) -> pd.DataFrame:
    ranked = assets.copy()
    ranked["local_preference_rank"] = ranked["local_exists"].map(lambda value: 0 if value else 1)
    group_columns = ["scene_id", "modality"]
    ranked = ranked.sort_values(
        ["scene_id", "modality", "local_preference_rank", "asset_preference_rank", "key"]
    )

    grouped_counts = ranked.groupby(group_columns, dropna=False).size()
    variant_counts = pd.DataFrame(
        {
            "scene_id": grouped_counts.index.get_level_values("scene_id"),
            "modality": grouped_counts.index.get_level_values("modality"),
            "asset_variant_count": grouped_counts.to_numpy(),
        }
    )
    scene_manifest = ranked.drop_duplicates(group_columns, keep="first").copy()
    scene_manifest = scene_manifest.merge(
        variant_counts, on=group_columns, how="left", validate="one_to_one"
    )
    return scene_manifest.drop(columns=["local_preference_rank"])


def build_chip_manifest(
    scenes: pd.DataFrame,
    *,
    chip_modalities: Sequence[str],
    chip_size: int,
    chip_stride: int,
) -> pd.DataFrame:
    if chip_size <= 0:
        raise ValueError("--chip-size must be positive.")
    if chip_stride <= 0:
        raise ValueError("--chip-stride must be positive.")

    modalities = {normalize_token(modality) for modality in chip_modalities}
    chip_rows: list[dict[str, object]] = []

    for record in scenes.to_dict("records"):
        modality = str(record["modality"])
        if normalize_token(modality) not in modalities:
            continue

        raster_x = to_int(record.get("raster_x"))
        raster_y = to_int(record.get("raster_y"))
        if raster_x is None or raster_y is None or raster_x <= 0 or raster_y <= 0:
            continue

        for x0 in tile_positions(raster_x, chip_size, chip_stride):
            for y0 in tile_positions(raster_y, chip_size, chip_stride):
                width = min(chip_size, raster_x - x0)
                height = min(chip_size, raster_y - y0)
                chip_key = f"{record['asset_id']}:{x0}:{y0}:{width}:{height}"
                chip_rows.append(
                    {
                        "chip_id": stable_id("chip", chip_key),
                        "asset_id": record["asset_id"],
                        "capture_id": record["capture_id"],
                        "scene_id": record["scene_id"],
                        "split": record["split"],
                        "city": record["city"],
                        "modality": modality,
                        "sensor": record["sensor"],
                        "local_path": record["local_path"],
                        "local_exists": record["local_exists"],
                        "remote_uri": record["remote_uri"],
                        "acq_time": record["acq_time"],
                        "x0": x0,
                        "y0": y0,
                        "x1": x0 + width,
                        "y1": y0 + height,
                        "width": width,
                        "height": height,
                        "chip_size": chip_size,
                        "chip_stride": chip_stride,
                    }
                )

    return pd.DataFrame(chip_rows)


def write_manifests(
    *,
    assets: pd.DataFrame,
    scenes: pd.DataFrame,
    chips: pd.DataFrame,
    output_root: Path,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    assets_path = output_root / "assets.parquet"
    scenes_path = output_root / "scenes.parquet"
    chips_path = output_root / "chips.parquet"
    summary_path = output_root / "summary.json"

    assets.to_parquet(assets_path, index=False, compression="zstd")
    scenes.to_parquet(scenes_path, index=False, compression="zstd")
    chips.to_parquet(chips_path, index=False, compression="zstd")
    summary = build_summary(assets=assets, scenes=scenes, chips=chips)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "assets": assets_path,
        "scenes": scenes_path,
        "chips": chips_path,
        "summary": summary_path,
    }


def build_summary(
    *,
    assets: pd.DataFrame,
    scenes: pd.DataFrame,
    chips: pd.DataFrame,
) -> dict[str, object]:
    return {
        "assets": summarize_frame(assets, count_columns=("split", "city", "modality")),
        "scenes": summarize_frame(scenes, count_columns=("split", "city", "modality")),
        "chips": summarize_frame(chips, count_columns=("split", "city", "modality")),
    }


def summarize_frame(frame: pd.DataFrame, *, count_columns: Iterable[str]) -> dict[str, object]:
    summary: dict[str, object] = {"rows": int(len(frame))}
    for column in count_columns:
        if column in frame and not frame.empty:
            counts = frame[column].fillna("null").astype(str).value_counts().sort_index()
            summary[f"by_{column}"] = {key: int(value) for key, value in counts.items()}
    return summary


def extract_city(key: str) -> str:
    satellite_city = SATELLITE_CITY_RE.search(key)
    if satellite_city:
        return satellite_city.group(1)

    aoi = AOI_RE.search(key)
    if aoi:
        token = aoi.group(1)
        parts = token.split("_", 2)
        if len(parts) == 3:
            return parts[2]
        return token

    return "unknown"


def extract_area_name(key: str) -> str:
    aoi = AOI_RE.search(key)
    if aoi:
        return aoi.group(1)

    satellite_city = SATELLITE_CITY_RE.search(key)
    if satellite_city:
        return satellite_city.group(1)

    return "unknown"


def extract_modality(key: str) -> str:
    path_parts = [part.upper() for part in PurePosixPath(key).parts]
    for modality in KNOWN_MODALITIES:
        if modality.upper() in path_parts:
            return modality.upper()

    upper_key = key.upper()
    for modality in KNOWN_MODALITIES:
        if re.search(rf"(?<![A-Z0-9]){re.escape(modality)}(?![A-Z0-9])", upper_key):
            return modality.upper()

    return "UNKNOWN"


def extract_product_code(key: str) -> str | None:
    match = PRODUCT_RE.search(key)
    if match:
        return match.group(1).upper()
    return None


def infer_capture_id(key: str) -> str:
    canonical = key
    for modality in KNOWN_MODALITIES:
        canonical = re.sub(
            rf"/{re.escape(modality)}/", "/<MODALITY>/", canonical, flags=re.IGNORECASE
        )
        canonical = re.sub(
            rf"(?<![A-Z0-9]){re.escape(modality)}(?![A-Z0-9])",
            "<MODALITY>",
            canonical,
            flags=re.IGNORECASE,
        )
    return SCENE_SUFFIX_RE.sub("", canonical)


def normalize_scene_id(scene_id: object, key: str) -> str:
    if scene_id is not None and not pd.isna(scene_id):
        return str(scene_id)
    return SCENE_SUFFIX_RE.sub("", key)


def normalize_sensor(sensor_hint: object, key: str) -> str | None:
    if sensor_hint is not None and not pd.isna(sensor_hint):
        return str(sensor_hint).upper()
    match = SENSOR_RE.search(key)
    if match:
        return match.group(1).upper()
    return None


def asset_preference_rank(key: str) -> int:
    upper_key = key.upper()
    if upper_key.endswith(".NTF"):
        return 0
    if upper_key.endswith("_LV1.TIF"):
        return 2
    if upper_key.endswith(".TIF"):
        return 1
    return 9


def assign_split(city: str, val_cities: set[str], test_cities: set[str]) -> str:
    normalized_city = normalize_token(city)
    if normalized_city in test_cities:
        return "test"
    if normalized_city in val_cities:
        return "val"
    return "train"


def normalize_token(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    return normalized or "UNKNOWN"


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{prefix}_{digest[:16]}"


def tile_positions(length: int, chip_size: int, stride: int) -> list[int]:
    if length <= chip_size:
        return [0]

    positions = list(range(0, length - chip_size + 1, stride))
    last = length - chip_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def to_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_path = resolve_metadata_path(args.metadata_path)
    output_root = args.output_root.resolve()
    download_root = args.download_root.resolve()

    metadata = load_metadata(metadata_path)
    assets = build_asset_manifest(
        metadata,
        download_root=download_root,
        val_cities=args.val_city,
        test_cities=args.test_city,
        source_metadata_path=metadata_path,
    )

    if args.limit_assets is not None:
        assets = assets.sort_values(["key"]).head(args.limit_assets).reset_index(drop=True)

    scenes = build_scene_manifest(assets)
    chips = build_chip_manifest(
        scenes,
        chip_modalities=args.chip_modality,
        chip_size=args.chip_size,
        chip_stride=args.chip_stride,
    )
    paths = write_manifests(assets=assets, scenes=scenes, chips=chips, output_root=output_root)

    print(f"metadata: {metadata_path}")
    print(f"assets: {len(assets):,} -> {paths['assets']}")
    print(f"scenes: {len(scenes):,} -> {paths['scenes']}")
    print(f"chips: {len(chips):,} -> {paths['chips']}")
    print(f"summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
