"""Integer-only causal AR(1)+bias predictor primitives for CPR1 coefficients.

This module deliberately contains no VQ, codebook, or learned side payload.
It only supplies a deterministic, exact residual transform that a later
carrier serializer may choose to store with Rice or ANS residual coding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q8_MIN = -512
Q8_MAX = 512
BIAS_MIN = -16
BIAS_MAX = 16


class CoefficientPredictorError(ValueError):
    """An integer coefficient predictor input is invalid."""


def signed_mod(values: np.ndarray) -> np.ndarray:
    """Reduce signed values into CPR1's exact signed-int12 domain."""
    return ((np.asarray(values, dtype=np.int64) + 2048) & 0xFFF).astype(np.int32) - 2048


def round_q8(values: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Round signed Q8 products to nearest integer, ties away from zero."""
    products = np.asarray(values, dtype=np.int64) * np.asarray(factors, dtype=np.int64)
    return np.where(
        products >= 0, (products + 128) // 256, -((-products + 128) // 256)
    ).astype(np.int32)


@dataclass(frozen=True)
class Ar1BiasModel:
    """Schema-ready fixed-point AR(1) parameters, one per coefficient dimension."""

    factors_q8: np.ndarray
    biases: np.ndarray

    def __post_init__(self) -> None:
        factors = np.asarray(self.factors_q8)
        biases = np.asarray(self.biases)
        if (
            factors.ndim != 1
            or biases.ndim != 1
            or factors.shape != biases.shape
            or not factors.size
        ):
            raise CoefficientPredictorError(
                "AR1 parameters must be equal nonempty vectors"
            )
        if not np.issubdtype(factors.dtype, np.integer) or not np.issubdtype(
            biases.dtype, np.integer
        ):
            raise CoefficientPredictorError("AR1 parameters must be integers")
        if np.any(factors < Q8_MIN) or np.any(factors > Q8_MAX):
            raise CoefficientPredictorError("AR1 Q8 factor outside fixed schema range")
        if np.any(biases < BIAS_MIN) or np.any(biases > BIAS_MAX):
            raise CoefficientPredictorError("AR1 bias outside fixed schema range")

    @property
    def dimensions(self) -> int:
        return int(np.asarray(self.factors_q8).size)

    @property
    def metadata_bytes(self) -> int:
        # int16 Q8 factor plus int8 signed bias for every known schema dimension.
        return self.dimensions * 3


def pack_ar1_bias_metadata(model: Ar1BiasModel) -> bytes:
    """Serialize schema-charged AR parameters in canonical little-endian form."""
    return (
        np.asarray(model.factors_q8, dtype="<i2").tobytes()
        + np.asarray(model.biases, dtype="i1").tobytes()
    )


def unpack_ar1_bias_metadata(payload: bytes, dimensions: int) -> Ar1BiasModel:
    """Strictly decode the fixed-size AR parameter field with no trailing data."""
    if dimensions <= 0 or len(payload) != dimensions * 3:
        raise CoefficientPredictorError("invalid AR1 metadata length")
    factors = np.frombuffer(payload[: 2 * dimensions], dtype="<i2").copy()
    biases = np.frombuffer(payload[2 * dimensions :], dtype="i1").copy()
    return Ar1BiasModel(factors, biases)


def _require_coefficients(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise CoefficientPredictorError(
            "coefficients must be a two-dimensional integer matrix with two frames"
        )
    values = values.astype(np.int32, copy=False)
    if np.any(values < -2048) or np.any(values > 2047):
        raise CoefficientPredictorError("coefficients exceed CPR1 signed-int12 range")
    return values


def ar1_bias_residuals(coefficients: np.ndarray, model: Ar1BiasModel) -> np.ndarray:
    """Return exact signed-int12 residuals under the causal fixed-point model."""
    values = _require_coefficients(coefficients)
    if values.shape[1] != model.dimensions:
        raise CoefficientPredictorError("coefficient dimensions do not match AR1 model")
    output = np.empty_like(values)
    output[0] = values[0]
    factors = np.asarray(model.factors_q8, dtype=np.int16)
    biases = np.asarray(model.biases, dtype=np.int16)
    for frame in range(1, values.shape[0]):
        prediction = signed_mod(round_q8(values[frame - 1], factors) + biases)
        output[frame] = signed_mod(values[frame] - prediction)
    return output


def restore_ar1_bias(residuals: np.ndarray, model: Ar1BiasModel) -> np.ndarray:
    """Strict inverse of :func:`ar1_bias_residuals` in the int12 domain."""
    values = _require_coefficients(residuals)
    if values.shape[1] != model.dimensions:
        raise CoefficientPredictorError("residual dimensions do not match AR1 model")
    output = np.empty_like(values)
    output[0] = values[0]
    factors = np.asarray(model.factors_q8, dtype=np.int16)
    biases = np.asarray(model.biases, dtype=np.int16)
    for frame in range(1, values.shape[0]):
        prediction = signed_mod(round_q8(output[frame - 1], factors) + biases)
        output[frame] = signed_mod(prediction + values[frame])
    return output


def zigzag_int12(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64)
    if np.any(source < -2048) or np.any(source > 2047):
        raise CoefficientPredictorError("residual exceeds signed-int12 range")
    return ((source << 1) ^ (source >> 63)).astype(np.int32) & 0xFFF


def rice_bit_cost(residuals: np.ndarray) -> tuple[int, np.ndarray]:
    """Return the exact CPS3 order-0 Rice bit cost and per-dimension k values."""
    encoded = zigzag_int12(residuals)
    if encoded.ndim != 2:
        raise CoefficientPredictorError(
            "Rice residuals must be a two-dimensional matrix"
        )
    ks = np.empty(encoded.shape[1], dtype=np.uint8)
    total = 0
    for dimension in range(encoded.shape[1]):
        column = encoded[:, dimension].astype(np.uint64, copy=False)
        bits, k = min(
            (
                int((column >> candidate).sum()) + column.size * (candidate + 1),
                candidate,
            )
            for candidate in range(12)
        )
        total += bits
        ks[dimension] = k
    return total, ks


def _fit_dimension(column: np.ndarray) -> tuple[int, int]:
    previous = column[:-1].astype(np.int64)
    target = column[1:].astype(np.int64)
    denominator = int(np.square(previous).sum())
    numerator = 256 * int((previous * target).sum())
    estimated = (
        0
        if denominator == 0
        else (1 if numerator >= 0 else -1)
        * ((abs(numerator) + denominator // 2) // denominator)
    )
    centre = min(Q8_MAX, max(Q8_MIN, estimated))
    candidates = range(max(Q8_MIN, centre - 96), min(Q8_MAX, centre + 96) + 1)
    best: tuple[int, int, int, int] | None = None
    for factor in candidates:
        baseline = round_q8(
            previous, np.full(previous.shape, factor, dtype=np.int16)
        ).astype(np.int64)
        # The median makes a compact integer starting point.  A small fixed
        # neighbourhood avoids floating point and deterministically includes
        # a no-bias model.
        central_bias = min(BIAS_MAX, max(BIAS_MIN, int(np.median(target - baseline))))
        for bias in range(
            max(BIAS_MIN, central_bias - 4), min(BIAS_MAX, central_bias + 4) + 1
        ):
            residual = signed_mod(target - signed_mod(baseline + bias))
            bits, _ = rice_bit_cost(
                np.concatenate(
                    (np.zeros((1, 1), dtype=np.int32), residual[:, None]), axis=0
                )
            )
            candidate = (bits, abs(factor - 256), factor, bias)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[2], best[3]


def fit_ar1_bias(coefficients: np.ndarray) -> Ar1BiasModel:
    """Fit a deterministic compact AR(1)+bias model using exact Rice cost."""
    values = _require_coefficients(coefficients)
    pairs = [
        _fit_dimension(values[:, dimension]) for dimension in range(values.shape[1])
    ]
    return Ar1BiasModel(
        np.asarray([pair[0] for pair in pairs], dtype=np.int16),
        np.asarray([pair[1] for pair in pairs], dtype=np.int8),
    )
