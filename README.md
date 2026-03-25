# geogrok-training

Training and evaluation scaffolding for a PAN-first geospatial image
intelligence stack.

The current repo focus is phase 0:

- build a reliable GDAL + Kakadu runtime for NITF / JP2K access
- normalize SpaceNet metadata into stable manifests
- prepare the data layer for retrieval and dense-task baselines

`uv` manages the Python environment in this repo.

## Current pieces

- [docs/implementation-plan.md](/nvme/development/geogrok-training/docs/implementation-plan.md):
  staged implementation plan
- [scripts/build_gdal_kakadu.sh](/nvme/development/geogrok-training/scripts/build_gdal_kakadu.sh):
  repo-local GDAL + Kakadu build
- [scripts/check_gdal_kakadu.sh](/nvme/development/geogrok-training/scripts/check_gdal_kakadu.sh):
  validation for an existing GDAL + Kakadu build
- [scripts/smoke_gdal_kakadu_rasters.sh](/nvme/development/geogrok-training/scripts/smoke_gdal_kakadu_rasters.sh):
  real raster open/read smoke test
- [src/geogrok/data/manifests.py](/nvme/development/geogrok-training/src/geogrok/data/manifests.py):
  manifest generation CLI
- [src/geogrok/io/gdal_env.py](/nvme/development/geogrok-training/src/geogrok/io/gdal_env.py):
  Python-side GDAL runtime activation helper

## Python setup

Install the root environment and dev tools:

```bash
uv sync --extra dev
```

Repo checks:

```bash
uv run ruff check .
uv run ty check
uv run --extra dev pytest -q
```

## GDAL + Kakadu

The repo includes vendored source trees under:

- `third_party/gdal/`
- `third_party/kakadu/`

The goal is to produce a repo-local GDAL build with:

- Kakadu-backed `JP2KAK`
- `NITF`
- Python bindings importable as `from osgeo import gdal`

### Build

Run:

```bash
./scripts/build_gdal_kakadu.sh
```

Validate an existing build without rebuilding:

```bash
./scripts/check_gdal_kakadu.sh
```

Run a real raster smoke test through the built Python bindings:

```bash
./scripts/smoke_gdal_kakadu_rasters.sh
```

Optional overrides:

- `GEOTIFF_PATH=/abs/path/to/file.tif`
- `NITF_PATH=/abs/path/to/file.ntf`
- `RASTER_SEARCH_ROOTS=/extra/search/root1:/extra/search/root2`

This installs into:

- prefix: `.local/gdal-kakadu/`
- build dir: `.build/gdal-kakadu/`
- Python build env: `.build/gdal-kakadu-python/`

The script handles:

- Kakadu auxiliary library generation when missing
- SWIG discovery from the local environment
- GDAL Python binding build
- runtime patching so the installed Python extensions load the local
  Kakadu-enabled `libgdal.so`
- generation of `.local/gdal-kakadu/env.sh`

### Activate

Source the generated environment before using the repo-local GDAL runtime:

```bash
source .local/gdal-kakadu/env.sh
```

This sets:

- `PATH`
- `GDAL_DATA`
- `PROJ_DATA`
- `PROJ_LIB`
- `LD_LIBRARY_PATH`
- `PYTHONPATH`

### Verified working state

The current validated commands are:

```bash
source .local/gdal-kakadu/env.sh
.local/gdal-kakadu/bin/gdalinfo --formats | rg 'JP2KAK|NITF'
.build/gdal-kakadu-python/bin/python - <<'PY'
from osgeo import gdal
print(gdal.VersionInfo("--version"))
print("JP2KAK", bool(gdal.GetDriverByName("JP2KAK")))
print("NITF", bool(gdal.GetDriverByName("NITF")))
PY
```

The same validation is wrapped in:

```bash
./scripts/check_gdal_kakadu.sh
```

The end-to-end raster open/read smoke test is:

```bash
./scripts/smoke_gdal_kakadu_rasters.sh
```

Expected output includes:

- `JP2KAK -raster,vector-`
- `NITF -raster-`
- `JP2KAK True`
- `NITF True`

### Notes

- The Python bindings are installed under
  `.local/gdal-kakadu/lib/python3.13/site-packages/`.
- The script currently falls back to the host Pixi `proj.db` when the local
  install prefix does not contain a populated PROJ database.
- The build uses `patchelf` so the installed `osgeo` extension modules resolve
  the local `libgdal.so` rather than any unrelated GDAL already present on the
  host.
- The raster smoke test prefers real files under `datasets/spacenet.ai/` when
  present. It searches the bucket-mirror tree directly, including uppercase
  `.NTF` and `.TIF` names. If the local dataset mirror is empty, it falls back to small
  GeoTIFF and NITF fixtures from `third_party/gdal/autotest/`. The default NITF
  fixture is JP2-compressed so the smoke read exercises the Kakadu-backed path.

### Common failure modes

`swig` is not on `PATH`

- Symptom: CMake fails with a Python binding error complaining that SWIG is
  missing.
- Fix: provide a usable SWIG install. The build script can discover a local
  SWIG binary and its library directory, but if that fails you can override it:

```bash
SWIG_EXECUTABLE=/path/to/swig SWIG_DIR=/path/to/share/swig/4.x.y \
  ./scripts/build_gdal_kakadu.sh
```

SWIG runs but cannot find `swig.swg` or `python.swg`

- Symptom: wrapper generation fails with messages like `Unable to find
  'swig.swg'`.
- Cause: the SWIG binary is present, but its library tree is not being found.
- Fix: ensure `SWIG_DIR` points at the directory containing `swig.swg`. The
  script exports `SWIG_LIB` for this reason.

Python imports `osgeo.gdal` but `JP2KAK` is missing

- Symptom: `gdalinfo --formats` shows `JP2KAK`, but Python reports
  `gdal.GetDriverByName("JP2KAK") is None`.
- Cause: the Python extension modules are loading a different `libgdal.so`
  already present on the host.
- Fix: source `.local/gdal-kakadu/env.sh` before using the bindings. The build
  script also rewrites the Python extension `RPATH` to prefer the local GDAL
  install.

`gdalinfo` or Python fails with missing shared libraries

- Symptom: runtime errors mentioning libraries like `libdeflate.so`,
  `libtiff.so`, or similar.
- Cause: the local GDAL build depends on host libraries that are not on the
  runtime loader path.
- Fix: source `.local/gdal-kakadu/env.sh` so `LD_LIBRARY_PATH` includes both
  the local GDAL install and the host Python environment library directory.

PROJ errors such as `proj_create: no database context specified`

- Symptom: `gdalinfo` emits PROJ errors about missing CRS parsing support.
- Cause: `PROJ_DATA` points at a directory without a valid `proj.db`.
- Fix: source `.local/gdal-kakadu/env.sh`. The script currently points
  `PROJ_DATA` and `PROJ_LIB` at the host Pixi `share/proj` when the local
  prefix does not contain a populated PROJ database.

Kakadu core exists but auxiliary library is missing

- Symptom: GDAL configure or link fails even though `libkdu_vs85R.so` is
  present.
- Cause: GDAL needs both Kakadu core and auxiliary libraries.
- Fix: rerun `./scripts/build_gdal_kakadu.sh`. It will build the Kakadu managed
  auxiliary library if `libkdu_a*.so` is missing.

## Manifest generation

Generate bootstrap manifests from the current SpaceNet metadata:

```bash
uv run geogrok-make-manifests
```

Smoke test on a small slice:

```bash
uv run geogrok-make-manifests --limit-assets 8
```

Default outputs go under:

- `datasets/manifests/spacenet/`

Current artifacts:

- `assets.parquet`
- `scenes.parquet`
- `chips.parquet`
- `summary.json`

## Next steps

The next intended repo work after phase 0 is:

1. mirror selected imagery into `datasets/spacenet.ai/`
2. use the GDAL/Kakadu runtime in actual raster readers and chip extraction
3. extend metadata and manifests with view-angle and off-nadir details
4. add first retrieval and dense-task baselines
