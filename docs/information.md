Yes. The current testing is measuring a fairly specific thing: can a PAN-only encoder
  produce embeddings that retrieve other chips from the same scene under a nontrivial held-out
  split, and how fast can it do that end to end?

  What We Are Measuring

  There are three layers of measurement in the current setup.

  1. Retrieval quality
     Implemented in src/geogrok/retrieval/baseline.py and used by the torch runner in src/
     geogrok/retrieval/torch_encoder.py.

     For each evaluated chip, we embed it, search nearest neighbors in the gallery, and ask
     whether chips from the same positive group are retrieved near the top.

     Right now the positive group is:
      - positive_key = scene_id

     So a retrieval is counted as correct if the returned chip comes from the same scene_id as
     the query, subject to extra filtering:
      - query splits: val, test
      - gallery splits: val, test
      - positives must be at least 1024 pixels apart by chip center
      - overlapping positives are excluded by default
      - each scene contributes a bounded number of chips:
          - min_chips_per_scene = 2
          - max_chips_per_scene = 4

     The metrics are:
      - R@1: fraction of queries where a valid positive is the top result
      - R@5: fraction of queries where a valid positive appears in the top 5
      - R@10: fraction of queries where a valid positive appears in the top 10
      - MRR: mean reciprocal rank, which rewards getting the first positive earlier in the
        ranking

     So this is not “classification accuracy” and not “semantic understanding” in the broad
     sense. It is instance / scene-coherence retrieval under held-out splits.
  2. Data path performance
     Implemented across src/geogrok/data/runtime.py, src/geogrok/data/training.py, and the
     extraction helpers reused in src/geogrok/retrieval/torch_encoder.py.

     This measures how fast the repo can:
      - resolve a chip row from chips.parquet
      - open the underlying real raster via GDAL/Kakadu
      - read the chip window from the source NITF or GeoTIFF
      - normalize the pixel values
      - resize to the model input size

     The key numbers reported are:
      - samples/s during preprocessing
      - read_latency_ms_p95
      - resize_latency_ms_p95

     In other words, this tells you how expensive the real raster I/O path is before the model
     even runs.
  3. Model-side training and inference throughput
     Implemented in src/geogrok/retrieval/torch_encoder.py.

     For the torch baseline we also measure:
      - images/s during contrastive training
      - batch latency during training
      - eval embedding throughput
      - GPU memory usage

     That tells you whether you are bottlenecked by:
      - disk / decode / resize
      - training compute
      - eval embedding throughput
      - GPU memory

  What The Current Torch Test Actually Does

  The current torch smoke run is roughly:

  - source real PAN chips from the manifest backed by real mirrored WV3/WV2 imagery
  - preprocess each chip as:
      - PAN-only
      - clipped to 0..2047
      - scaled to 0..1
      - mean-downsampled to 128 x 128
  - train a small CNN on the train split using same-scene positive pairs
  - evaluate retrieval on val and test

  The current smoke settings in scripts/smoke_torch_embedding.sh are:

  - train_limit = 256
  - eval_limit = 96
  - image_size = 128
  - base_channels = 48
  - embedding_dim = 128
  - epochs = 24
  - steps_per_epoch = 48
  - pairs_per_batch = 32
  - amp = on
  - device = auto, which used the RTX 3090 on this node

  So the current result means:

  “Using a small PAN-only CNN trained contrastively on same-scene pairs, with real source-
  raster reads and held-out val/test retrieval, how often do we recover another non-
  overlapping chip from the same scene?”

  What It Does Not Measure Yet

  This is the important part.

  It does not yet measure:

  - text-to-image retrieval
  - object-level retrieval
  - cross-city semantic retrieval beyond same-scene coherence
  - dense perception quality like segmentation or detection
  - off-nadir robustness explicitly as a stratified metric
  - retrieval by building type, road structure, or analyst intent
  - geographic generalization in a rigorous benchmark sense
  - production latency of a full ANN index

  It also does not yet use:

  - captions
  - RemoteCLIP / DOFA teachers
  - RGB/MS privileged supervision
  - object proposals
  - region indexing

  So this is still a bootstrap retrieval benchmark, not the final system benchmark.

  How To Interpret The Current Scores

  The current torch numbers:

  - R@1 = 0.135
  - R@5 = 0.229
  - R@10 = 0.302
  - MRR = 0.195

  mean:

  - top-1 exact retrieval is still weak
  - top-5 and top-10 are somewhat useful
  - the learned model is now better than the deterministic handcrafted baseline on early-rank
    retrieval
  - but this is still far from “analyst-grade semantic retrieval”

  That is a good milestone, because it proves:

  - the on-demand raster path works
  - the harder protocol is not trivial
  - a learned PAN encoder can already beat the handcrafted control
  - the GPU path is fast enough to iterate cheaply

  What I Think We Should Measure Next

  If we want the benchmark to become more representative of your actual target system, the
  next additions should be:

  - Harder positive definitions
    Not just same scene_id, but matched chips across acquisitions, views, or nearby
    footprints.
  - Stratified retrieval
    Report metrics by city, sensor, and off-nadir bucket once manifests carry that metadata.
  - Region/object retrieval
    Scene-level chip retrieval is useful, but not enough for “find me scenes containing X”.
  - Teacher-aligned retrieval
    Evaluate whether PAN embeddings recover semantics defined by RGB/MS teacher embeddings or
    captions, not just scene co-membership.
  - Index-time and search-time metrics
    Once we move beyond brute-force nearest neighbors, we should measure embedding throughput,
    index build time, and ANN query latency separately.

  If useful, I can turn this into a short docs/evaluation-protocol.md so the benchmark
  contract is explicit and stable before we scale the retrieval experiments further.


• Yes. In the current repo, “same scene” means chips that inherit the same normalized scene_id
  in the manifest, and in practice that is usually different chips cut from the same
  underlying raster scene/product, not just the same city or nearby geography.

  The exact logic is in src/geogrok/data/manifests.py:398. When building assets.parquet, the
  code sets scene_id like this:

  - if the source metadata already has a scene_id, use that
  - otherwise, fall back to the object key with only the filename suffix removed, via
    normalize_scene_id(...)

  That fallback strips things like .NTF, .TIF, and _lv1, but it does not attempt to merge
  different acquisitions of the same place. So it is a fairly literal product-level
  identifier.

  Then src/geogrok/data/manifests.py:191 groups by scene_id + modality and picks one preferred
  asset for that scene, and src/geogrok/data/manifests.py:218 copies that scene_id onto every
  chip cut from that raster. That means that when retrieval uses positive_key=scene_id, it is
  effectively asking:

  “Given this PAN chip, can I retrieve another spatially distinct PAN chip from the same
  source scene image?”

  Two important clarifications:

  - It does not mean “same AOI” or “same city”.
    Two chips from Jacksonville are not positives unless they share the same scene_id.
  - It does not mean “same ground footprint across different acquisitions”.
    If the same location was collected on two different dates or views and got different
    scene_ids, those are currently treated as different scenes.

  There is one more wrinkle: retrieval also filters positives in src/geogrok/retrieval/
  baseline.py:280. A candidate positive must:

  - share the same scene_id
  - not be the exact same chip
  - be at least min_positive_center_distance apart
  - be non-overlapping unless overlap is explicitly allowed

  So in plain English, the current benchmark is not “find the same place again across time/
  view changes.” It is narrower:

  “Find another non-overlapping chip from the same original raster scene.”

  That is useful as a bootstrap test because it measures whether the embedding preserves
  within-scene visual/structural coherence. But it is weaker than the benchmark you probably
  want long term.

  For contrast, the repo also has capture_id, created in src/geogrok/data/manifests.py:383.
  That normalizes modality tokens out of the key, so it is closer to “same acquisition family
  across modality variants.” If we switched retrieval to positive_key=capture_id, we would be
  testing a slightly broader notion than the current scene_id setup.

  So the short version is:

  - same scene now = same normalized source scene/product ID
  - current positives = different chips from the same raster scene
  - it is not yet cross-time, cross-view, or cross-acquisition location matching

  If you want, I can make the repo terminology more explicit by adding a short Benchmark
  Semantics section to the README and renaming this internally to something like “same-raster retrieval” so it is harder to misread.

