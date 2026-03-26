from __future__ import annotations

import json

import numpy as np

from geogrok.obs.review_artifacts import (
    ReviewArtifactRecord,
    _load_existing_record,
    _write_png_artifact,
    _write_record_json,
    infer_bit_depth,
)


def test_infer_bit_depth_for_common_pan_ranges():
    assert infer_bit_depth(np.zeros((4, 4), dtype=np.uint16)) == 1
    assert infer_bit_depth(np.full((4, 4), 2047, dtype=np.uint16)) == 11
    assert infer_bit_depth(np.full((4, 4), 4095, dtype=np.uint16)) == 12
    assert infer_bit_depth(np.full((4, 4), 255, dtype=np.uint8)) == 8


def test_write_png_artifact_and_reload_record(tmp_path):
    chip = {
        "chip_id": "chip_demo",
        "local_path": "/tmp/source.ntf",
        "x0": 10,
        "y0": 20,
        "width": 64,
        "height": 64,
    }
    array = np.arange(64 * 64, dtype=np.uint16).reshape(1, 64, 64)

    record = _write_png_artifact(
        chip=chip,
        array=array,
        artifact_dir=tmp_path,
        codec_profile="review_visually_lossless",
    )

    assert record.artifact_kind == "png"
    assert record.media_type == "image/png"
    assert record.file_size_bytes > 0
    assert record.width == 64
    assert record.height == 64

    _write_record_json(record, tmp_path)
    reloaded = _load_existing_record("chip_demo", tmp_path)
    assert isinstance(reloaded, ReviewArtifactRecord)
    assert reloaded == record
    payload = json.loads((tmp_path / "chip_demo.json").read_text())
    assert payload["chip_id"] == "chip_demo"
    assert (tmp_path / "chip_demo.png").exists()
