"""Shared frame calibration for local bimanual Trossen deployments."""

from __future__ import annotations

import torch


# Store the homogeneous transform from left-arm base coordinates
# into right-arm base coordinates.
# Calibration assumption:
# - Left base origin is +0.92 m along right-base X.
# - Left base is rotated 180 degrees around Z relative to right base.
RIGHT_FROM_LEFT_BASE: tuple[tuple[float, float, float, float], ...] | None = (
    (-1.0, 0.0, 0.0, 0.92),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _validate_homogeneous_transform(
    matrix: tuple[tuple[float, float, float, float], ...],
) -> None:
    """Validate that a matrix is a 4x4 homogeneous rigid transform.

    Args:
        matrix: Candidate homogeneous transform matrix.

    Raises:
        ValueError: If the matrix shape or last row is invalid.
    """
    # Materialize the candidate as a tensor for shape and value checks.
    transform = torch.tensor(matrix, dtype=torch.float64)  # (4, 4)

    # Enforce a strict 4x4 shape for homogeneous coordinates.
    if transform.shape != (4, 4):
        raise ValueError(
            f"RIGHT_FROM_LEFT_BASE must be shape (4, 4), got {tuple(transform.shape)}."
        )

    # Enforce canonical homogeneous last-row values.
    expected_last_row = transform.new_tensor([0.0, 0.0, 0.0, 1.0])  # (4,)
    if not torch.allclose(transform[3], expected_last_row, atol=1e-6, rtol=0.0):
        raise ValueError(
            "RIGHT_FROM_LEFT_BASE must end with homogeneous row [0, 0, 0, 1]."
        )


def right_from_left_base() -> tuple[tuple[float, float, float, float], ...]:
    """Return the calibrated transform from left base to right base.

    Returns:
        tuple[tuple[float, float, float, float], ...]: Homogeneous 4x4 matrix.

    Raises:
        ValueError: If calibration values are unset or invalid.
    """
    # Fail hard when calibration constants have not been configured yet.
    if RIGHT_FROM_LEFT_BASE is None:
        raise ValueError(
            "RIGHT_FROM_LEFT_BASE is unset. Configure "
            "nt/_client/imitation_mirror/utils/trossen_bimanual_frames.py "
            "with the measured left->right base transform."
        )

    # Validate matrix structure once before returning it.
    _validate_homogeneous_transform(RIGHT_FROM_LEFT_BASE)
    return RIGHT_FROM_LEFT_BASE


def left_from_right_base() -> tuple[tuple[float, float, float, float], ...]:
    """Return the inverse transform from right base to left base.

    Returns:
        tuple[tuple[float, float, float, float], ...]: Homogeneous 4x4 matrix.
    """
    # Load and validate the forward calibration matrix first.
    right_from_left = torch.tensor(
        right_from_left_base(), dtype=torch.float64
    )  # (4, 4)

    # Invert to obtain the right->left mapping used by robot egress remap.
    left_from_right = torch.linalg.inv(right_from_left)  # (4, 4)

    # Convert to a plain nested tuple so callers can serialize it easily.
    return tuple(tuple(float(v) for v in row.tolist()) for row in left_from_right)
