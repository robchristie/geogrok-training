25-03-2026 2:28 PM

Your direction is basically right. The main change I’d make is to treat this as a staged PAN deployment program, not a single “foundation model selection” problem.

Right now the repo is still in phase 0. You already have useful exploration utilities in
tools/spacenet/src/read_metadata.py, tools/spacenet/src/find_overlap.py, and tools/spacenet/src/download_overlap_images.py. The local SpaceNet bucket index already covers about 4.55M S3 objects, and your filtered PAN metadata slice has 494 WV/P1BS candidates, but datasets/ itself is not populated yet. Also, tools/spacenet/pyproject.toml currently pins stock gdal==3.12.2; that will not give you the Kakadu-enabled NITF/JP2K path you want.

Recommended Plan

1. Phase 0: get I/O and manifests stable.
 Build a repo-local GDAL with Kakadu and Python bindings, install it into an isolated
 prefix, and make all raster access go through that build. Do not rely on the PyPI GDAL
 wheel for production I/O. The important repo fact here is that GDAL’s Kakadu support is
 wired through KDU_ROOT, and the Python bindings can be built from source as part of the
 same install. Verify two things before doing any ML work: gdalinfo --formats shows JP2KAK, and from osgeo import gdal can open both the SpaceNet GeoTIFFs and the NITF/JP2K products you care about.

2. Phase 0: materialize the dataset as manifests first, imagery second.
 Keep raw files under datasets/spacenet.ai/raw/, and create parquet manifests under
 something like datasets/manifests/. Your first manifest should be scene-level and should normalize: scene_id, modality, sensor, acquisition time, off-nadir / view metadata when available, local path, raster size, approximate footprint, and split assignment. The second manifest should be chip-level and store matched PAN/RGB/MS chips from the same scene footprint. For this project, RGB/MS should be privileged teacher data, not runtime data.

3. Phase 0: define the evaluation protocol before training.
 This is the biggest missing piece in your writeup. You need fixed splits by city/region, view angle, and time so you do not leak near-duplicates across train and test. For this problem I would make the primary test sets:
 PAN in-domain seen-city,
 PAN unseen-city,
 PAN off-nadir hard split,
 PAN low-overlap / low-context object split.
 If you skip this, every later model comparison will be noisy.

4. Phase 1: establish cheap baselines.
 Start with two baseline tracks only:
 Retrieval: RemoteCLIP teacher -> simple PAN student baseline.
 Dense perception: one strong dense encoder with a UNet-style decoder and PEFT.
 Do not start with Panopticon + DOFA-CLIP + LAE-DINO + Falcon all at once. That is too
 many moving parts, and you will not know what helped. On your 3090, the first dense
 baseline should be chip-based building / road / scene classification with PEFT, not full fine-tuning.

5. Phase 1: retrieval benchmark first, because it gives you leverage.
 Build two indices from day one:
 scene index from whole PAN tiles or large chips,
 region index from detector proposals or segmentation regions.
 Evaluate with chip-to-chip retrieval, text-to-chip retrieval, and region retrieval. The
 metrics should be R@1/5/10, mAP, and a small analyst-style benchmark of hard text
 queries. This is where caption generation, hard-negative mining, and taxonomy expansion
 from frontier models are actually useful.

6. Phase 2: dense task baseline second.
 For dense tasks, keep the deployment backbone PAN-only. Train with SpaceNet geometry as
 the high-trust core, and use OSM/SRTM only as weak priors or text/context features. For
 buildings in off-nadir imagery, do not treat OSM footprints as literal image-space truth. Either move to roof labels, or train an explicit roof-to-footprint offset head. If you keep only footprint supervision, at least report metrics stratified by off-nadir angle so you can see the failure mode instead of averaging it away.

7. Phase 3: assemble the framework.
 At this point the framework should have four modules:
 io: GDAL/Kakadu-backed raster access, manifests, chipping, caching.
 retrieval: teacher inference, PAN student embeddings, scene and region indices.
 dense: classification / detection / segmentation inference and training.
 orchestration: query planner / analyst-facing layer.
 I would make the agent layer last. Falcon-like orchestration is useful only after
 retrieval and dense outputs are already reliable.

8. Phase 4: training program.
 Your training sequence should be:
 zero-shot teacher baselines,
 PEFT dense fine-tunes on SpaceNet labels,
 PAN student distillation from RGB/MS teachers,
 pseudo-label loop with human or LLM triage,
 long-text alignment / analyst-query tuning.
 That ordering matters. Distillation and pseudo-labeling are much higher leverage after
 you have a trustworthy manifest, split protocol, and baseline metrics.

What I would implement in this repo next
Use uv throughout and add a real package layout, for example:

src/geogrok/io/
src/geogrok/data/
src/geogrok/models/
src/geogrok/eval/
src/geogrok/index/
configs/
scripts/
experiments/

Then add these concrete milestones:
script/build_gdal_kakadu.sh: builds and verifies the local GDAL/Kakadu Python install.
src/geogrok/io/gdal_env.py: forces the right osgeo runtime and config.
src/geogrok/data/catalog.py: reads and writes scene and chip manifests.
src/geogrok/data/chip_pan_teacher_pairs.py: builds matched PAN/RGB/MS crops.
src/geogrok/eval/splits.py: city / angle / time split logic.
src/geogrok/eval/retrieval_eval.py: scene and region retrieval metrics.
src/geogrok/eval/dense_eval.py: AP / IoU / F1 with off-nadir stratification.

What’s missing or still unclear
You have the model thesis, but these project decisions are still underspecified:
The initial task set. Are the first production tasks building retrieval, building
segmentation, road extraction, vehicle detection, or broad scene understanding?
The annotation geometry. Roofs, footprints, centers, or offset-aware paired labels?
The operating constraints. Batch offline indexing is one thing; interactive analyst
retrieval with latency targets is another.

The data budget. Full raw WV2/WV3 NITF storage and local caching strategy will affect the loader design.

The licensing boundary. Kakadu, model weights, and use of commercial frontier models for
training data all need an explicit policy.

My strong recommendation is: do not begin with the full stack. Start with GDAL/Kakadu I/O -> manifests -> splits -> retrieval baseline -> dense baseline. Once those are stable, your proposed teacher-student PAN stack becomes much easier to execute cleanly.

Relevant sources I checked:

- GDAL build docs: https://gdal.org/en/stable/development/building_from_source.html
- GDAL JP2KAK driver docs: https://gdal.org/en/stable/drivers/raster/jp2kak.html
- SpaceNet MVOI off-nadir benchmark:

https://openaccess.thecvf.com/content_ICCV_2019/html/Weir_SpaceNet_MVOI_A_Multi-View_Overhead_Imagery_Dataset_ICCV_2019_paper.html
- Maxar base product notes on Basic 1B / Standard 2A:
https://csda-maxar-pdfs.s3.amazonaws.com/DigitalGlobe-Base-Product.pdf

If you want, I can take the next step and turn this into concrete repo work: a docs/
implementation-plan.md, the GDAL/Kakadu build script, and the first dataset manifest
pipeline.

---

The next step should be wiring this runtime into the manifest/chipping path so actual raster reads go through src/geogrok/io/gdal_env.py and then adding the downloader + first chip extractor.

The next sensible step is to enrich manifests and extracted-chip metadata with image-native acquisition fields, especially off-nadir / view geometry from the NITF side, so the first retrieval and dense baselines can be stratified by view difficulty instead of treating all PAN chips as equivalent.

---

• I would not use individual HTJ2K chip files as the main training format.

  HTJ2K is attractive for storage, but for training it usually loses on the thing that matters
  most: decode throughput and operational simplicity. You already need Kakadu for raw-source
  access; forcing every training sample through JP2/HTJ2K decode adds CPU pressure, loader
  complexity, and another failure mode. That is usually the wrong trade for chip-scale
  training data.

  My recommendation is a 3-tier setup:

  1. Raw truth stays in the original source products.
     Keep the original NITF / GeoTIFF assets under datasets/spacenet.ai/ and treat the
     manifest windows as the canonical chip definition.
  2. Training should default to on-the-fly reads plus an NVMe cache.
     For early experiments, read windows directly from the source rasters using the chip
     manifest and cache decoded chips locally. This avoids materializing a huge chip corpus
     too early and keeps the dataset definition flexible.
  3. If you need a materialized chip corpus, use lossless uint16 GeoTIFF with ZSTD, not LZW,
     and not HTJ2K.
     This is the best practical middle ground: preserves 11/12-bit PAN in uint16, compresses
     well because the high bits are sparse and the signal is spatially correlated, and decodes
     much faster and more simply than JP2/HTJ2K.

  If I had to pick one concrete default for materialized chips, it would be:

  - COMPRESS=ZSTD
  - PREDICTOR=2
  - ZSTD_LEVEL=6 or 9
  - TILED=YES
  - BLOCKXSIZE=512
  - BLOCKYSIZE=512
  - NUM_THREADS=ALL_CPUS

  That is a better default than the current LZW setting in src/geogrok/io/raster.py.

  What I would not do:

  - Not millions of tiny uncompressed or LZW TIFFs as the main corpus.
  - Not HTJ2K chips as the main trainer input format.
  - Not converting to float32 on disk.
  - Not throwing away bit depth early unless you have a measured reason.

  For larger-scale training, the better long-term move is probably not “better per-chip TIFF
  compression”, but “fewer filesystem objects”. The two serious options are:

  - Keep chips virtual: manifest + source rasters + local cache.
    This is the best starting point and probably enough for your 3090 experiments.
  - Pack chips into shard containers for large runs.
    If you get into large-scale repeated training, move to sharded storage instead of one-
    file-per-chip. I would look at either:
      - WebDataset tar shards if you want simple PyTorch streaming
      - Zarr if you want chunked random access and compressed array storage

  Between those two, I’d lean:

  - WebDataset for large distributed training runs
  - Zarr for analysis-heavy workflows and random chip access

  So the practical answer is:

  - Best immediate approach: manifest-defined chips, on-the-fly reads, optional local cache.
  - Best materialized fallback: lossless uint16 GeoTIFF with ZSTD.
  - Best archival compression: HTJ2K, but not as the main training-chip format.

---

  The next sensible step is to build the first actual trainer-facing dataset wrapper on top of src/geogrok/data/runtime.py: batching, normalization, optional augmentations, and split-aware sampling for a first PAN classification or retrieval baseline.

