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

  The next step I’d take is to stay on this torch path and make the retrieval benchmark harder rather than jumping models again: larger eval slices, stronger positive definitions than same-scene alone, and view/off-nadir stratification once we enrich the manifests.


  The better retrieval targets are:

  1. Same-location across acquisitions or views.
     This is much stronger than same-raster retrieval. A positive should be a chip covering
     the same or nearby ground area, but from a different image, time, or view angle.
  2. Same semantic class or composition.
     For example: “dense residential blocks”, “container port”, “runway”, “mountain ridge with
     sparse roads”, “coastal urban edge”. These can come from different scenes entirely.
  3. Region/object retrieval, not just scene-chip retrieval.
     If a tile is mixed ocean/urban/mountains, whole-chip similarity gets muddy. Region-level
     retrieval is closer to analyst intent.
  4. Hard negatives from the same large scene.
     This is especially important for your case. Two chips from the same image but different
     semantics should often be explicit negatives, not automatic positives.

---


• Yes. I think this is the right time to do it, and I would design it as an observability and
  review system, not just a gallery.

  The repo now has enough structure that a UI can be genuinely useful:

  - stable chip manifests
  - explicit pair labels
  - benchmark outputs
  - model artifacts
  - performance metrics
  - a real notion of “interesting failures”

  Because of that, I’d optimize for three things from day one:

  1. fast visual inspection of chips and pairs
  2. direct linkage to benchmark results and failure modes
  3. a path to human review / annotation later without redesigning the stack

  ## Recommended Architecture

  I would use SvelteKit for the UI, but I would not make SvelteKit responsible for raster
  access or analytics logic.

  The clean split is:

  - Python backend
      - owns GDAL/Kakadu
      - reads manifests, pairs, benchmark artifacts
      - renders quicklook images from source chips
      - computes review tables and outlier candidates
  - SvelteKit frontend
      - owns browsing, filtering, pair inspection, run comparison, annotation UX
      - talks to the Python backend over JSON + image endpoints

  Reason: all the difficult project-specific logic already lives in Python:

  - manifest semantics
  - pair protocol
  - benchmark outputs
  - GDAL-backed chip extraction
  - teacher/student metrics

  Trying to recreate that in Node would be the wrong split.

  ## Suggested High-Level Layout

  I’d add something like:

  src/geogrok/obs/
    api.py
    quicklook.py
    review_tables.py
    run_index.py
    annotations.py

  web/
    package.json
    svelte.config.js
    vite.config.ts
    src/routes/...

  Or, if you want to keep the frontend clearly separated:

  apps/review-ui/
  apps/review-api/

  But given the repo is Python-first, I’d probably keep:

  - Python API inside src/geogrok/obs/
  - SvelteKit app under web/

  ## Backend Design

  I’d use FastAPI for the review API. It’s the pragmatic choice here.

  ### Why FastAPI

  - simple JSON/image endpoints
  - easy local dev
  - easy future auth if needed
  - easy reuse of existing Python code
  - easy to mount DuckDB / SQLite / parquet logic

  ### Backend Responsibilities

  1. Quicklook rendering
     Endpoints like:
      - GET /api/chips/:chip_id/image
      - GET /api/pairs/:pair_id/image-strip
      - GET /api/assets/:asset_id/overview

     These should:
      - read the source chip on demand via existing raster code
      - normalize for display
      - return PNG or JPEG quicklooks
      - support display params like:
          - stretch=percentile
          - clip_min
          - clip_max
          - invert=false
          - size=256
  2. Manifest / pair browsing
     Endpoints like:
      - GET /api/chips
      - GET /api/pairs
      - GET /api/chips/:chip_id
      - GET /api/pairs/:pair_id

     These should expose:
      - chip metadata
      - pair label
      - city
      - split
      - sensor
      - acquisition time
      - overlap metrics
      - center distance
      - local path provenance
  3. Run and benchmark indexing
     Endpoints like:
      - GET /api/runs
      - GET /api/runs/:run_id
      - GET /api/runs/:run_id/failures
      - GET /api/runs/:run_id/query/:chip_id

     These should expose:
      - retrieval summaries
      - training metrics
      - teacher/student comparisons
      - top-k retrieved candidates for a query
      - failure classification
  4. Review / annotation persistence
     Endpoints like:
      - POST /api/reviews/pair
      - POST /api/reviews/chip
      - GET /api/reviews/queue
      - GET /api/reviews/export

     This is where human labels eventually live.

  ## Storage Design

  This is the part that matters most if you want it to scale cleanly.

  I would use three storage layers.

  ### 1. Immutable analytical tables: Parquet

  Keep these as parquet:

  - manifests
  - pairs
  - benchmark outputs
  - review candidate tables
  - nearest-neighbor result tables

  Reason:

  - already aligned with repo
  - easy to generate from Python
  - efficient for batch filtering / materialization

  ### 2. Mutable review state: SQLite

  Use SQLite for:

  - user annotations
  - review status
  - tags
  - issue flags
  - queue state
  - free-text notes

  Reason:

  - mutable
  - local-first
  - easy for SvelteKit + FastAPI
  - much better fit than constantly rewriting parquet

  I would not use parquet for mutable human annotations.

  ### 3. Cached quicklooks: filesystem cache

  Store rendered display images under something like:

  artifacts/observability/quicklooks/
    chips/<chip_id>.png
    pairs/<pair_key>.jpg
    assets/<asset_id>.jpg

  Reason:

  - avoid re-rendering every request
  - make the UI feel immediate
  - keep source raster reads available when needed, but not mandatory for every click

  ## Data Model I’d Use

  ### Chip record

  Fields:

  - chip_id
  - asset_id
  - scene_id
  - capture_id
  - city
  - split
  - sensor
  - modality
  - chip_x
  - chip_y
  - chip_width
  - chip_height
  - acq_time
  - local_path

  ### Pair record

  Fields:

  - pair_id or deterministic composite key
  - query_chip_id
  - candidate_chip_id
  - pair_label
  - pair_group
  - overlap_fraction
  - overlap_iou
  - center_distance_m
  - time_delta_seconds
  - query_split
  - candidate_split
  - city
  - query_sensor
  - candidate_sensor

  ### Run record

  Fields:

  - run_id
  - run_type
      - pretrained_benchmark
      - pan_adapt_benchmark
      - torch_embedding
  - timestamp
  - config_path
  - summary_path
  - teacher_model
  - student_model
  - metrics

  ### Review annotation

  Fields:

  - review_id
  - target_type
      - chip
      - pair
      - query_result
  - target_id
  - status
      - unreviewed
      - confirmed
      - incorrect_label
      - interesting
      - needs_followup
  - tags
  - note
  - reviewer
  - created_at
  - updated_at

  ## UI Design

  I would build four main views first.

  ### 1. Dataset Browser

  Purpose:

  - inspect chips by city, split, sensor, modality, asset, acquisition time

  UI:

  - left filter panel
  - chip grid in main panel
  - detail drawer on click

  Good filters:

  - city
  - split
  - sensor
  - modality
  - asset
  - acquisition date range
  - pair participation
  - annotation state

  This gives you the “show me example chips from category X” workflow.

  ### 2. Pair Inspector

  Purpose:

  - inspect positive and negative pairs directly

  UI:

  - side-by-side chips
  - metadata strip in the middle or top
  - label summary:
      - positive_exact
      - positive_weak
      - negative_hard
  - overlap / distance / time delta shown prominently
  - buttons:
      - confirm label
      - mark suspicious
      - tag cloud/occlusion/artifact
      - add note

  This is probably the highest-value first page.

  ### 3. Run Explorer

  Purpose:

  - inspect benchmark outputs and model behavior

  UI:

  - choose run
  - choose query chip
  - show:
      - query chip
      - top-k retrieved chips
      - whether each is true positive / hard negative / unlabeled
      - similarity scores
      - rank positions
  - toggle between teacher and student results

  This is where you answer:

  - “why did this fail?”
  - “what kinds of errors is the student making?”
  - “what is the teacher seeing that the student is not?”

  ### 4. Outlier Review Queue

  Purpose:

  - surface the most informative examples automatically

  Queues I would generate:

  - false negatives
      - positive_exact or positive_weak pairs with low model similarity
  - false positives
      - negative_hard pairs with high model similarity
  - teacher-student disagreement
      - teacher high, student low
      - student high, teacher low
  - label suspicion
      - supposed positives that look obviously wrong
      - supposed negatives that look obviously overlapping
  - data quality
      - clouds
      - severe occlusion
      - striping/artifacts
      - extreme exposure

  This is the part that naturally turns into human annotation later.

  ## What “Observability” Should Mean Here

  I think you want observability at three levels.

  ### A. Data observability

  Questions:

  - what chips are actually in train / val / test?
  - what do positive_exact and negative_hard really look like?
  - what are the weird edge cases?

  ### B. Model observability

  Questions:

  - which positives are missed?
  - which hard negatives are being confused?
  - how do teacher and student differ?
  - are errors clustered by city / sensor / time delta / overlap band?

  ### C. Pipeline observability

  Questions:

  - what was the run config?
  - what metrics changed?
  - what was the throughput?
  - what artifacts belong to this run?

  The UI should support all three, but I would build them in that order:

  1. data observability
  2. model observability
  3. annotation workflow

  ## Review Table Materialization

  A key design decision: do not compute everything on every page load.

  I would add Python jobs that materialize review tables under something like:

  artifacts/observability/
    run_index.parquet
    review_candidates/
      run_<id>_false_negatives.parquet
      run_<id>_false_positives.parquet
      run_<id>_teacher_student_disagreement.parquet

  Then the UI can stay fast and simple.

  Each row in a review candidate table should include:

  - query_chip_id
  - candidate_chip_id
  - pair_label
  - rank
  - similarity
  - teacher_similarity
  - student_similarity
  - error_type
  - city
  - split
  - time_delta_seconds
  - overlap_fraction

  That gives you sortable, filterable queues without recomputing from raw run outputs every
  time.

  ## Quicklook Rendering Strategy

  For PAN imagery, display matters a lot. If you just dump raw 11/12-bit grayscale to PNG, the
  UI will often look poor.

  I’d support:

  - percentile stretch, default
  - optional histogram equalization later
  - invert toggle
  - consistent display normalization for side-by-side comparison
  - cached renderings

  And I’d keep the rendering logic in Python, not JS.

  A simple endpoint contract:

  - GET /api/chips/:chip_id/image?mode=percentile&pmin=2&pmax=98&size=256

  For pairs:

  - GET /api/pairs/:pair_key/composite?size=256
    returns a stitched strip for quick scanning in tables.

  ## Annotation Design

  You said you eventually want human annotation. I would design for that now, even if the
  first version is light.

  Minimum annotation features:

  - mark pair label as:
      - correct
      - incorrect
      - uncertain
  - add issue tags:
      - cloud
      - shadow
      - water
      - seasonal change
      - geometry issue
      - low quality
  - add free-text note

  Later, this can expand to:

  - corrected relevance
  - task-specific tags
  - building presence / road presence / port / runway etc.
  - adjudication workflow

  The key thing is: make annotation append-only and auditable.

  ## Recommended First Implementation Scope

  I would not build the full review system in one pass.

  I’d do this in three phases.

  ### Phase 1: Read-only observability

  Build:

  - SvelteKit shell
  - Python review API
  - chip browser
  - pair inspector
  - run explorer for one benchmark type
  - on-demand quicklook rendering + cache

  This gets immediate value.

  ### Phase 2: Outlier queues

  Build:

  - review candidate materialization
  - false positive / false negative queues
  - teacher-student disagreement views
  - filtering and sorting

  This makes the UI useful for model iteration.

  ### Phase 3: Human review

  Build:

  - SQLite-backed annotations
  - review actions in UI
  - queue state
  - export reviewed examples

  This makes it part of the training loop.

  ## My Concrete Recommendation

  If I were implementing this repo next, I’d build:

  Backend

  - src/geogrok/obs/api.py
  - src/geogrok/obs/quicklook.py
  - src/geogrok/obs/review_tables.py
  - src/geogrok/obs/run_index.py
  - src/geogrok/obs/annotations.py

  Frontend

  - web/src/routes/
      - /chips
      - /pairs
      - /runs
      - /runs/[runId]
      - /review

  Initial API endpoints

  - GET /api/chips
  - GET /api/chips/:chip_id
  - GET /api/chips/:chip_id/image
  - GET /api/pairs
  - GET /api/pairs/:pair_key
  - GET /api/pairs/:pair_key/images
  - GET /api/runs
  - GET /api/runs/:run_id
  - GET /api/runs/:run_id/failures

  ## The Most Important Design Choice

  The most important design choice is this:

  make the UI driven by explicit review tables and run artifacts, not by ad hoc filesystem
  browsing.

  That gives you:

  - reproducibility
  - explainability
  - easy queue generation
  - a clean path to annotation
  - low-friction comparison between runs

  Without that, the UI tends to turn into a pretty image browser with weak analytical value.

  ## My Recommendation In One Sentence

  Use SvelteKit as a thin, fast review UI over a Python observability API that serves
  quicklooks, run artifacts, pair tables, and annotation state, with parquet for immutable
  analytics, SQLite for mutable review data, and review queues built from benchmark failures
  as the core product.

  If you want, I can take the next step and scaffold this concretely:

  1. docs/observability-plan.md
  2. Python review API skeleton
  3. SvelteKit app skeleton
  4. first read-only pages: /chips, /pairs, /runs



  The next step I’d take is to make the mined negatives part of evaluation as well: add a small adversarial-negative retrieval report so we can measure whether training is actually improving rejection of these confusable pairs instead of only watching aggregate R@K drift.
