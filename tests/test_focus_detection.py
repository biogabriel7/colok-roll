"""Unit tests for the focus detection module."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from colokroll.imaging_preprocessing import (
    aggregate_focus_scores,
    compute_focus_scores,
    detect_oof_slices,
    find_oof_slices,
)


BLUR_SLICES = (3, 4)


def _synthetic_stack(z: int = 8, y: int = 64, x: int = 64, c: int = 3) -> np.ndarray:
    rng = np.random.default_rng(1234)
    stack = rng.standard_normal((z, y, x, c)).astype(np.float32)

    grid_y = np.linspace(-1.0, 1.0, y, dtype=np.float32)[:, None]
    grid_x = np.linspace(-1.0, 1.0, x, dtype=np.float32)[None, :]
    texture = np.sin(6.0 * np.pi * grid_y) + np.cos(6.0 * np.pi * grid_x)

    for ch in range(c):
        stack[:, :, :, ch] += (ch + 1) * texture

    stack -= stack.min()
    stack /= stack.max()

    for idx in BLUR_SLICES:
        for ch in range(c):
            stack[idx, :, :, ch] = ndimage.gaussian_filter(stack[idx, :, :, ch], sigma=3.0)

    return stack


def test_find_oof_slices_flags_blurred_planes() -> None:
    stack = _synthetic_stack()
    result = find_oof_slices(
        stack,
        strategy="percentile",
        threshold=40,
        smooth=1,
    )

    removed = np.where(result.mask_oof)[0]
    for idx in BLUR_SLICES:
        assert idx in removed

    assert result.scores_zc.shape == (stack.shape[0], stack.shape[3])
    assert result.mask_oof_zc.shape == result.scores_zc.shape
    assert result.indices_in_focus.shape[0] < stack.shape[0]


def test_aggregate_focus_scores_variants() -> None:
    stack = _synthetic_stack()
    scores_zc, _ = compute_focus_scores(stack, method="tenengrad")

    mean_scores = aggregate_focus_scores(scores_zc, aggregation="mean")
    weighted_scores = aggregate_focus_scores(
        scores_zc,
        aggregation="weighted",
        weights=[0.2, 0.3, 0.5],
    )

    expected_weighted = (scores_zc * np.array([0.2, 0.3, 0.5], dtype=np.float32)).sum(axis=1)
    expected_weighted /= 1.0

    assert mean_scores.shape == (stack.shape[0],)
    assert weighted_scores.shape == (stack.shape[0],)
    assert np.allclose(weighted_scores, expected_weighted, rtol=1e-5, atol=1e-6)


def test_axes_argument_matches_default_path() -> None:
    stack = _synthetic_stack()
    permuted = np.transpose(stack, (3, 0, 1, 2))  # C, Z, Y, X

    baseline = find_oof_slices(
        stack,
        strategy="percentile",
        threshold=40,
        smooth=1,
    )

    permuted_result = find_oof_slices(
        permuted,
        axes="CZYX",
        strategy="percentile",
        threshold=40,
        smooth=1,
    )

    assert np.allclose(permuted_result.scores_agg, baseline.scores_agg)
    assert np.array_equal(permuted_result.mask_oof, baseline.mask_oof)
    assert np.array_equal(permuted_result.indices_in_focus, baseline.indices_in_focus)


def test_detect_oof_topk_keeps_requested_planes() -> None:
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.8], dtype=np.float32)
    result = detect_oof_slices(scores, strategy="topk", keep_top=2)

    kept = result["indices_in_focus"]
    assert np.array_equal(kept, np.array([3, 4]))
    assert result["mask_oof"].sum() == scores.size - kept.size


