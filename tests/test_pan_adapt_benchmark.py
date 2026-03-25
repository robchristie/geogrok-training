from __future__ import annotations

import pandas as pd
import pytest

from geogrok.retrieval.pan_adapt_benchmark import assert_record_alignment, parse_args


def test_assert_record_alignment_accepts_matching_chip_ids():
    assert_record_alignment(
        expected_records=[{"chip_id": "a"}, {"chip_id": "b"}],
        actual_metadata=pd.DataFrame({"chip_id": ["a", "b"]}),
        field_name="test",
    )


def test_assert_record_alignment_rejects_mismatched_order():
    with pytest.raises(RuntimeError, match="teacher_eval"):
        assert_record_alignment(
            expected_records=[{"chip_id": "a"}, {"chip_id": "b"}],
            actual_metadata=pd.DataFrame({"chip_id": ["b", "a"]}),
            field_name="teacher_eval",
        )


def test_parse_args_exposes_new_adaptation_weights():
    args = parse_args(["--pairs-path", "pairs.parquet"])
    assert args.structure_weight == pytest.approx(0.5)
    assert args.view_consistency_weight == pytest.approx(0.25)
    assert args.positive_pair_weight == pytest.approx(0.5)
    assert args.hard_negative_weight == pytest.approx(0.25)
    assert args.positive_exact_weight == pytest.approx(2.0)
    assert args.positive_weak_weight == pytest.approx(1.0)
    assert args.hard_negative_max_similarity == pytest.approx(0.2)
    assert args.hard_negative_gap_scale == pytest.approx(0.5)
    assert args.hard_negative_min_similarity == pytest.approx(-0.25)
    assert args.adversarial_negative_top_fraction == pytest.approx(0.25)
    assert args.adversarial_negative_max_pairs == 512
    assert args.adversarial_negative_min_teacher_similarity == pytest.approx(0.0)
    assert args.augmentation_min_crop_scale == pytest.approx(0.7)
    assert args.augmentation_noise_std == pytest.approx(0.02)
    assert args.augmentation_gamma_jitter == pytest.approx(0.15)
    assert args.augmentation_blur_probability == pytest.approx(0.2)


def test_parse_args_accepts_residual_student_configuration():
    args = parse_args(
        [
            "--pairs-path",
            "pairs.parquet",
            "--teacher-model",
            "dinov3_vitb16",
            "--student-arch",
            "residual_cnn",
            "--structure-weight",
            "0.8",
            "--view-consistency-weight",
            "0.3",
            "--positive-pair-weight",
            "0.6",
            "--hard-negative-weight",
            "0.4",
            "--hard-negative-gap-scale",
            "0.75",
            "--hard-negative-min-similarity",
            "-0.1",
            "--adversarial-negative-top-fraction",
            "0.4",
            "--adversarial-negative-max-pairs",
            "128",
            "--adversarial-negative-min-teacher-similarity",
            "0.2",
        ]
    )
    assert args.teacher_model == "dinov3_vitb16"
    assert args.student_arch == "residual_cnn"
    assert args.structure_weight == pytest.approx(0.8)
    assert args.view_consistency_weight == pytest.approx(0.3)
    assert args.positive_pair_weight == pytest.approx(0.6)
    assert args.hard_negative_weight == pytest.approx(0.4)
    assert args.hard_negative_gap_scale == pytest.approx(0.75)
    assert args.hard_negative_min_similarity == pytest.approx(-0.1)
    assert args.adversarial_negative_top_fraction == pytest.approx(0.4)
    assert args.adversarial_negative_max_pairs == 128
    assert args.adversarial_negative_min_teacher_similarity == pytest.approx(0.2)
