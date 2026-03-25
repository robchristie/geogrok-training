#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from osgeo import gdal, osr
from shapely.geometry import Polygon

gdal.UseExceptions()
osr.UseExceptions()

BUCKET = "spacenet-dataset"
DEFAULT_INDEX_PATH = Path("artifacts/s3-spacenet-dataset.parquet")
DEFAULT_OUTPUT_PATH = Path("artifacts/s3-spacenet-dataset-images-ds")
TIMESTAMP_14_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")
SENSOR_RE = re.compile(r"WV\d", re.IGNORECASE)
SCENE_SUFFIX_RE = re.compile(r"(_lv1)?\.(ntf|tif)$", re.IGNORECASE)


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("key", pa.string()),
        pa.field("size", pa.int64()),
        pa.field("last_modified", pa.timestamp("us", tz="UTC")),
        pa.field("etag", pa.string()),
        pa.field("scene_id", pa.string()),
        pa.field("acq_time", pa.timestamp("us")),
        pa.field("acq_time_start", pa.timestamp("us")),
        pa.field("acq_time_end", pa.timestamp("us")),
        pa.field("driver", pa.string()),
        pa.field("sensor_hint", pa.string()),
        pa.field("geom_source", pa.string()),
        pa.field("geom", pa.binary()),
        pa.field("raster_x", pa.int32()),
        pa.field("raster_y", pa.int32()),
    ]
)


@dataclass
class Stats:
    scanned: int = 0
    matched: int = 0
    skipped_existing: int = 0
    extracted: int = 0
    no_geom: int = 0
    errors: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read remote raster metadata from the S3 index into a parquet dataset."
    )
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--include",
        action="append",
        default=["PAN"],
        help="Require each substring to be present in the key. Repeat to add more filters.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip keys containing this substring. Repeat to add more filters.",
    )
    parser.add_argument(
        "--key-regex",
        help="Optional case-insensitive regex applied after include/exclude filters.",
    )
    parser.add_argument("--path-prefix", help="Optional key prefix filter.")
    parser.add_argument(
        "--ext",
        action="append",
        default=[".NTF", ".TIF"],
        help="File extension to scan. Repeat to add more.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4) * 2),
        help="Metadata fetch worker threads.",
    )
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=20_000,
        help="Rows to stream from the parquet index at a time.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=500,
        help="Records to buffer before flushing to parquet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after extracting this many matching records.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip keys already written to the output dataset.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete the output dataset directory before starting.",
    )
    return parser.parse_args()


def configure_gdal() -> None:
    configure_proj()
    gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO")
    gdal.SetConfigOption("VSI_CACHE", "TRUE")
    gdal.SetConfigOption("VSI_CACHE_SIZE", str(16 * 1024 * 1024))
    # We only need NITF header metadata and corner coordinates, not the imagery payload.
    gdal.SetConfigOption("NITF_OPEN_UNDERLYING_DS", "NO")


def configure_proj() -> None:
    candidate_dirs = [
        Path(os.environ["PROJ_DATA"]) if "PROJ_DATA" in os.environ else None,
        Path(os.environ["PROJ_LIB"]) if "PROJ_LIB" in os.environ else None,
        Path.home() / ".pixi/envs/default/share/proj",
        Path.home() / ".cache/rattler/cache/pkgs/proj-9.8.0-he0df7b0_0/share/proj",
        Path(sys.prefix) / "share/proj",
        Path(sys.prefix) / "lib/python3.13/site-packages/pyproj/proj_dir/share/proj",
        Path(__file__).resolve().parent.parent
        / ".venv/lib/python3.13/site-packages/pyproj/proj_dir/share/proj",
        Path.home() / ".cache/uv/archive-v0/-TOf8JufZTKJrlkQ4IxK7/pyproj/proj_dir/share/proj",
    ]
    for candidate in candidate_dirs:
        if candidate and candidate.exists():
            os.environ.setdefault("PROJ_DATA", str(candidate))
            os.environ.setdefault("PROJ_LIB", str(candidate))
            break


def wgs84_spatial_ref() -> osr.SpatialReference:
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)
    return spatial_ref


def normalize_extensions(exts: list[str]) -> tuple[str, ...]:
    normalized = []
    for ext in exts:
        ext = ext.upper()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(dict.fromkeys(normalized))


def build_key_matcher(args: argparse.Namespace):
    extensions = normalize_extensions(args.ext)
    includes = [token.upper() for token in args.include if token]
    excludes = [token.upper() for token in args.exclude if token]
    key_regex = re.compile(args.key_regex, re.IGNORECASE) if args.key_regex else None
    path_prefix = args.path_prefix

    def matches(key: str) -> bool:
        upper = key.upper()
        if not upper.endswith(extensions):
            return False
        if path_prefix and not key.startswith(path_prefix):
            return False
        if any(token not in upper for token in includes):
            return False
        if any(token in upper for token in excludes):
            return False
        if key_regex and not key_regex.search(key):
            return False
        return True

    return matches


def vsis3_path(key: str) -> str:
    return f"/vsis3/{BUCKET}/{key}"


def parse_datetime_14(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if len(value) >= 14 and value[:14].isdigit():
        return datetime.strptime(value[:14], "%Y%m%d%H%M%S")
    return None


def parse_tiff_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def parse_key_timestamps(key: str) -> tuple[datetime | None, datetime | None]:
    matches = TIMESTAMP_14_RE.findall(key)
    if not matches:
        return None, None
    start = parse_datetime_14(matches[0])
    end = parse_datetime_14(matches[1]) if len(matches) > 1 else None
    return start, end


def parse_acquisition_times(md: dict[str, str], key: str) -> tuple[datetime | None, datetime | None, datetime | None]:
    acq_time = parse_datetime_14(md.get("NITF_IDATIM")) or parse_datetime_14(md.get("IDATIM"))
    if acq_time is None:
        acq_time = parse_tiff_datetime(md.get("TIFFTAG_DATETIME"))

    key_start, key_end = parse_key_timestamps(key)
    if acq_time is None:
        acq_time = key_start
    return acq_time, key_start, key_end


def scene_id_from_key(key: str) -> str:
    return SCENE_SUFFIX_RE.sub("", key)


def parse_igeolo_dms(value: str | None) -> Polygon | None:
    if not value:
        return None
    value = value.strip()
    if len(value) < 60:
        return None

    def parse_lat(token: str) -> float:
        deg = int(token[0:2])
        minute = int(token[2:4])
        second = int(token[4:6])
        hemi = token[6].upper()
        decimal = deg + minute / 60.0 + second / 3600.0
        return -decimal if hemi == "S" else decimal

    def parse_lon(token: str) -> float:
        deg = int(token[0:3])
        minute = int(token[3:5])
        second = int(token[5:7])
        hemi = token[7].upper()
        decimal = deg + minute / 60.0 + second / 3600.0
        return -decimal if hemi == "W" else decimal

    corners = []
    for idx in range(0, 60, 15):
        chunk = value[idx : idx + 15]
        if len(chunk) != 15:
            return None
        corners.append((parse_lon(chunk[7:15]), parse_lat(chunk[0:7])))

    return Polygon(corners + [corners[0]])


def polygon_from_geotransform(ds) -> Polygon | None:
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        return None

    width = ds.RasterXSize
    height = ds.RasterYSize

    def xy(col: float, row: float) -> tuple[float, float]:
        x = gt[0] + col * gt[1] + row * gt[2]
        y = gt[3] + col * gt[4] + row * gt[5]
        return x, y

    ring = [xy(0, 0), xy(width, 0), xy(width, height), xy(0, height), xy(0, 0)]
    spatial_ref = ds.GetSpatialRef()
    if spatial_ref is None:
        return Polygon(ring)

    try:
        wgs84 = wgs84_spatial_ref()
        if spatial_ref.IsSame(wgs84):
            return Polygon(ring)
        transform = osr.CoordinateTransformation(spatial_ref, wgs84)
        transformed = []
        for x, y in ring:
            lon, lat, _ = transform.TransformPoint(x, y)
            transformed.append((lon, lat))
        return Polygon(transformed)
    except RuntimeError:
        return None


def footprint_from_dataset(ds, md: dict[str, str]) -> tuple[Polygon | None, str | None]:
    polygon = polygon_from_geotransform(ds)
    if polygon is not None:
        return polygon, "geotransform"

    if md.get("NITF_ICORDS") == "G":
        polygon = parse_igeolo_dms(md.get("NITF_IGEOLO"))
        if polygon is not None:
            return polygon, "nitf_igeolo"

    return None, None


def extract_record(record: dict) -> dict | None:
    key = record["key"]
    path = vsis3_path(key)

    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
    except RuntimeError:
        return None

    if ds is None:
        return None

    md = ds.GetMetadata()
    geom, geom_source = footprint_from_dataset(ds, md)
    if geom is None or geom.is_empty:
        return None

    acq_time, acq_time_start, acq_time_end = parse_acquisition_times(md, key)
    sensor_match = SENSOR_RE.search(key)

    return {
        "key": key,
        "size": record["size"],
        "last_modified": record["last_modified"],
        "etag": record["etag"],
        "scene_id": scene_id_from_key(key),
        "acq_time": acq_time,
        "acq_time_start": acq_time_start,
        "acq_time_end": acq_time_end,
        "driver": ds.GetDriver().ShortName if ds.GetDriver() else None,
        "sensor_hint": sensor_match.group(0).upper() if sensor_match else None,
        "geom_source": geom_source,
        "geom": geom.wkb,
        "raster_x": ds.RasterXSize,
        "raster_y": ds.RasterYSize,
    }


def ensure_output_path(output_path: Path, reset_output: bool) -> None:
    if output_path.exists() and reset_output:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.mkdir(parents=True, exist_ok=True)


def load_existing_keys(output_path: Path) -> set[str]:
    keys: set[str] = set()
    part_files = sorted(output_path.glob("*.parquet"))
    for part_file in part_files:
        table = pq.read_table(part_file, columns=["key"])
        keys.update(table.column("key").to_pylist())
    return keys


def iter_index_records(index_path: Path, batch_size: int):
    parquet_file = pq.ParquetFile(index_path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["key", "size", "last_modified", "etag"],
    ):
        yield batch.to_pylist()


def write_rows(output_path: Path, rows: list[dict], part_id: int) -> int:
    if not rows:
        return part_id
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    part_name = output_path / f"part-{part_id:06d}.parquet"
    pq.write_table(table, part_name, compression="zstd")
    return part_id + 1


def progress_line(stats: Stats, output_path: Path) -> str:
    return (
        f"\rscanned: {stats.scanned:,} matched: {stats.matched:,} "
        f"existing: {stats.skipped_existing:,} extracted: {stats.extracted:,} "
        f"no_geom: {stats.no_geom:,} errors: {stats.errors:,} -> {output_path}"
    )


def main() -> int:
    args = parse_args()
    configure_gdal()
    ensure_output_path(args.output_path, args.reset_output)

    matcher = build_key_matcher(args)
    existing_keys = load_existing_keys(args.output_path) if args.resume else set()
    part_id = 0
    existing_parts = sorted(args.output_path.glob("part-*.parquet"))
    if existing_parts:
        last_part = existing_parts[-1].stem.split("-")[-1]
        part_id = int(last_part) + 1

    stats = Stats()
    write_buffer: list[dict] = []
    stop = False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for records in iter_index_records(args.index_path, args.index_batch_size):
            candidates = []
            for record in records:
                stats.scanned += 1
                key = record["key"]
                if not matcher(key):
                    continue
                stats.matched += 1
                if key in existing_keys:
                    stats.skipped_existing += 1
                    continue
                candidates.append(record)

            if candidates:
                for result in executor.map(extract_record, candidates):
                    if result is None:
                        stats.no_geom += 1
                        continue
                    write_buffer.append(result)
                    existing_keys.add(result["key"])
                    stats.extracted += 1

                    if args.limit and stats.extracted >= args.limit:
                        stop = True
                        break

                    if len(write_buffer) >= args.write_batch_size:
                        part_id = write_rows(args.output_path, write_buffer, part_id)
                        write_buffer.clear()

            sys.stdout.write(progress_line(stats, args.output_path))
            sys.stdout.flush()

            if stop:
                break

    if write_buffer:
        part_id = write_rows(args.output_path, write_buffer, part_id)
        write_buffer.clear()

    print()
    print(
        "finished",
        f"output={args.output_path}",
        f"parts={part_id}",
        f"extracted={stats.extracted:,}",
        f"no_geom={stats.no_geom:,}",
        f"existing={stats.skipped_existing:,}",
        f"matched={stats.matched:,}",
        f"scanned={stats.scanned:,}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
