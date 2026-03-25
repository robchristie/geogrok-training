#!/usr/bin/env python3

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from shapely import from_wkb

from find_overlap import overlap_components, resolve_metadata_path

BUCKET = "spacenet-dataset"
DEFAULT_DOWNLOAD_ROOT = Path("spacenet.ai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all images that belong to overlap groups from the SpaceNet bucket."
    )
    parser.add_argument("--metadata-path", type=Path, help="Parquet file or parquet dataset directory.")
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
        help="Directory under which S3 keys are mirrored.",
    )
    parser.add_argument("--predicate", default="overlaps")
    parser.add_argument(
        "--contains",
        action="append",
        default=["WV", "P1BS"],
        help="Require each substring to be present in the key. Repeat to add more filters.",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[".NTF", ".TIF"],
        help="File extension to include. Repeat to add more filters.",
    )
    parser.add_argument(
        "--dedupe-scene",
        action="store_true",
        help="Keep one preferred record per scene_id before overlap grouping.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Download at most this many overlapping files after filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be downloaded without fetching them.",
    )
    return parser.parse_args()


def s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def normalize_extensions(exts: list[str]) -> tuple[str, ...]:
    normalized = []
    for ext in exts:
        ext = ext.upper()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(dict.fromkeys(normalized))


def filter_metadata(df: pd.DataFrame, contains: list[str], exts: list[str], dedupe_scene: bool) -> pd.DataFrame:
    for token in contains:
        df = df[df["key"].str.contains(token, case=False, na=False)]

    extensions = normalize_extensions(exts)
    df = df[df["key"].str.upper().str.endswith(extensions)]

    if dedupe_scene and "scene_id" in df:
        key_upper = df["key"].str.upper()
        df = (
            df.assign(
                _scene_rank=np.select(
                    [
                        key_upper.str.endswith(".NTF"),
                        key_upper.str.endswith(".TIF") & ~key_upper.str.contains("_LV1.TIF", regex=False),
                        key_upper.str.contains("_LV1.TIF", regex=False),
                    ],
                    [0, 1, 2],
                    default=3,
                )
            )
            .sort_values(["scene_id", "_scene_rank", "key"])
            .drop_duplicates("scene_id", keep="first")
            .drop(columns="_scene_rank")
        )

    return df.reset_index(drop=True)


def overlapping_keys(df: pd.DataFrame, predicate: str) -> list[str]:
    if df.empty:
        return []

    df = df.copy()
    df["geometry"] = from_wkb(df["geom"].values)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    _, components, _ = overlap_components(gdf, predicate=predicate)

    selected = []
    for idxs in components:
        if len(idxs) > 1:
            selected.extend(gdf.loc[idxs, "key"].tolist())

    return sorted(dict.fromkeys(selected))


def local_path_for_key(download_root: Path, key: str) -> Path:
    return download_root / key


def download_one(client, bucket: str, key: str, download_root: Path) -> tuple[str, str]:
    target_path = local_path_for_key(download_root, key)
    if target_path.exists():
        return "skipped", key

    target_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target_path))
    return "downloaded", key


def main() -> int:
    args = parse_args()
    metadata_path = resolve_metadata_path(args.metadata_path)
    download_root = args.download_root

    df = pd.read_parquet(metadata_path)
    df = filter_metadata(df, args.contains, args.ext, args.dedupe_scene)
    keys = overlapping_keys(df, args.predicate)

    if args.limit is not None:
        keys = keys[: args.limit]

    if not keys:
        print("No overlapping keys matched the requested filters.")
        return 0

    print(f"metadata: {metadata_path}")
    print(f"download_root: {download_root}")
    print(f"overlapping files: {len(keys):,}")

    if args.dry_run:
        for key in keys:
            print(key)
        return 0

    client = s3_client()
    downloaded = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, client, BUCKET, key, download_root) for key in keys]
        for idx, future in enumerate(futures, start=1):
            try:
                status, key = future.result()
            except Exception as exc:
                errors += 1
                sys.stdout.write(
                    f"\rprocessed: {idx:,}/{len(keys):,} downloaded: {downloaded:,} "
                    f"skipped: {skipped:,} errors: {errors:,}"
                )
                sys.stdout.flush()
                print(f"\nERROR {exc}")
                continue

            if status == "downloaded":
                downloaded += 1
            else:
                skipped += 1

            sys.stdout.write(
                f"\rprocessed: {idx:,}/{len(keys):,} downloaded: {downloaded:,} "
                f"skipped: {skipped:,} errors: {errors:,}"
            )
            sys.stdout.flush()

    print()
    print(
        "finished",
        f"downloaded={downloaded:,}",
        f"skipped={skipped:,}",
        f"errors={errors:,}",
        f"root={download_root}",
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
