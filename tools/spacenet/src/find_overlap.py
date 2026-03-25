#!/usr/bin/env python3

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkb


DEFAULT_METADATA_CANDIDATES = [
    Path("artifacts/s3-spacenet-dataset-images-ds"),
    Path("artifacts/s3-spacenet-dataset-images.parquet"),
]


def overlap_components(
    gdf: gpd.GeoDataFrame,
    geom_col: str = "geometry",
    predicate: str = "intersects",
):
    gdf = gdf.reset_index(drop=True)
    n = len(gdf)

    sindex = gdf.sindex
    pairs = sindex.query(gdf[geom_col], predicate=predicate)
    i = pairs[0].astype(int)
    j = pairs[1].astype(int)

    mask = i < j
    i, j = i[mask], j[mask]
    edges = pd.DataFrame({"i": i, "j": j})

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in zip(i, j):
        union(a, b)

    roots = np.array([find(k) for k in range(n)])
    unique_roots, comp_id = np.unique(roots, return_inverse=True)

    components = [[] for _ in range(len(unique_roots))]
    for idx, cid in enumerate(comp_id):
        components[cid].append(idx)

    return comp_id, components, edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find overlapping raster footprints.")
    parser.add_argument("--metadata-path", type=Path, help="Parquet file or parquet dataset directory.")
    parser.add_argument("--predicate", default="overlaps")
    parser.add_argument("--contains", action="append", default=["PAN"])
    parser.add_argument("--ext", action="append", default=[".NTF"])
    parser.add_argument(
        "--dedupe-scene",
        action="store_true",
        help="Keep one preferred record per scene_id to drop NTF/TIF/lv1 duplicates.",
    )
    return parser.parse_args()


def resolve_metadata_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DEFAULT_METADATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No metadata parquet file or dataset found under artifacts/")


def main() -> int:
    args = parse_args()
    metadata_path = resolve_metadata_path(args.metadata_path)
    df = pd.read_parquet(metadata_path)

    for token in args.contains:
        df = df[df["key"].str.contains(token, case=False, na=False)]

    extensions = tuple(ext.upper() if ext.startswith(".") else f".{ext.upper()}" for ext in args.ext)
    df = df[df["key"].str.upper().str.endswith(extensions)]

    if args.dedupe_scene and "scene_id" in df:
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

    df = df.reset_index(drop=True)
    df["geometry"] = from_wkb(df["geom"].values)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    comp_id, components, edges = overlap_components(gdf, predicate=args.predicate)

    groups = [gdf.loc[idxs, "key"].tolist() for idxs in components]
    for group in groups:
        if len(group) > 1:
            print(group[0], ":", len(group))
            print("====")

    print(f"loaded {len(gdf):,} geometries from {metadata_path}")
    print(f"connected components: {len(components):,}")
    print(f"overlap edges: {len(edges):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
