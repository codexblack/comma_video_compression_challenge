"""Exact non-VQ AR(1)+bias/Rice carrier codec for controlled experiments."""

from __future__ import annotations

import struct

import numpy as np

from ..carrier_repack import (
    CPR1_MAGIC,
    _coefficients_from_cpr1,
    _parse_cpr1,
    _predict_residuals,
    _read_u24,
    _rice_decode,
    _rice_encode,
    _u24,
    _unzigzag,
    _zigzag,
)
from .coefficient_predictor import (
    CoefficientPredictorError,
    ar1_bias_residuals,
    fit_ar1_bias,
    pack_ar1_bias_metadata,
    restore_ar1_bias,
    unpack_ar1_bias_metadata,
)

CAP1_MAGIC = b"CAP1"
CAP1_VERSION = 1
_HEADER_BYTES = 14


class CoefficientAr1CodecError(ValueError):
    """A CAP1 exact carrier payload is malformed or noncanonical."""


def encode_cap1(
    raw_cpr1: bytes, *, frames: int, dimensions: int
) -> tuple[bytes, dict[str, object]]:
    """Encode a canonical CPR1 carrier with a fixed-point AR(1)+bias model."""
    try:
        basis_bits, _, fields = _parse_cpr1(raw_cpr1, dimensions=dimensions)
        _, coefficients = _coefficients_from_cpr1(
            raw_cpr1, frames=frames, dimensions=dimensions
        )
        model = fit_ar1_bias(coefficients)
        residuals = ar1_bias_residuals(coefficients, model)
        ks, payload, residual_bits = _rice_encode(_zigzag(residuals), 1)
    except (CoefficientPredictorError, ValueError) as error:
        raise CoefficientAr1CodecError(
            f"cannot encode CAP1 carrier: {error}"
        ) from error
    header = (
        CAP1_MAGIC
        + bytes((CAP1_VERSION, 0, 0, 0))
        + _u24(basis_bits)
        + _u24(residual_bits)
    )
    blob = (
        header
        + pack_ar1_bias_metadata(model)
        + fields["scales"]
        + fields["lengths"]
        + ks.reshape(-1).tobytes()
        + fields["basis"]
        + payload
    )
    restored = decode_cap1(blob, frames=frames, dimensions=dimensions)
    if restored != raw_cpr1:
        raise CoefficientAr1CodecError(
            "CAP1 encoder did not restore canonical CPR1 bytes"
        )
    return blob, inspect_cap1(blob, frames=frames, dimensions=dimensions)


def inspect_cap1(blob: bytes, *, frames: int, dimensions: int) -> dict[str, object]:
    """Validate CAP1 framing and expose every charged field size."""
    if len(blob) < _HEADER_BYTES or blob[:4] != CAP1_MAGIC:
        raise CoefficientAr1CodecError("invalid CAP1 magic or truncated header")
    version, reserved_a, reserved_b, reserved_c = blob[4:8]
    if version != CAP1_VERSION or reserved_a or reserved_b or reserved_c:
        raise CoefficientAr1CodecError("unsupported or noncanonical CAP1 header")
    basis_bits, residual_bits = _read_u24(blob[8:11]), _read_u24(blob[11:14])
    if not basis_bits or not residual_bits:
        raise CoefficientAr1CodecError("CAP1 bit counts must be nonzero")
    metadata_bytes = dimensions * 3
    basis_bytes, residual_bytes = (basis_bits + 7) // 8, (residual_bits + 7) // 8
    fixed_after_header = metadata_bytes + 8 * dimensions + 32 + dimensions
    expected = _HEADER_BYTES + fixed_after_header + basis_bytes + residual_bytes
    if len(blob) != expected:
        raise CoefficientAr1CodecError("invalid CAP1 field lengths or trailing bytes")
    try:
        model = unpack_ar1_bias_metadata(
            blob[_HEADER_BYTES : _HEADER_BYTES + metadata_bytes], dimensions
        )
    except CoefficientPredictorError as error:
        raise CoefficientAr1CodecError(f"invalid CAP1 AR metadata: {error}") from error
    offset = _HEADER_BYTES + metadata_bytes
    scales = blob[offset : offset + 8 * dimensions]
    offset += 8 * dimensions
    lengths = blob[offset : offset + 32]
    offset += 32
    ks = np.frombuffer(blob[offset : offset + dimensions], dtype=np.uint8).copy()
    offset += dimensions
    basis = blob[offset : offset + basis_bytes]
    offset += basis_bytes
    rice = blob[offset:]
    try:
        _rice_decode(ks.reshape(dimensions, 1), rice, residual_bits, frames, dimensions)
    except ValueError as error:
        raise CoefficientAr1CodecError(f"invalid CAP1 Rice stream: {error}") from error
    return {
        "carrier_bytes": len(blob),
        "header_bytes": _HEADER_BYTES,
        "predictor_metadata_bytes": metadata_bytes,
        "scales_bytes": len(scales),
        "basis_lengths_bytes": len(lengths),
        "rice_parameter_bytes": len(ks),
        "basis_bytes": len(basis),
        "rice_payload_bytes": len(rice),
        "basis_bits": basis_bits,
        "rice_payload_bits": residual_bits,
        "factors_q8": np.asarray(model.factors_q8, dtype=np.int16).tolist(),
        "biases": np.asarray(model.biases, dtype=np.int8).tolist(),
        "rice_ks": ks.tolist(),
    }


def decode_cap1(blob: bytes, *, frames: int, dimensions: int) -> bytes:
    """Strictly reconstruct the canonical CPR1 carrier bytes from CAP1."""
    info = inspect_cap1(blob, frames=frames, dimensions=dimensions)
    metadata_bytes = dimensions * 3
    basis_bits = int(info["basis_bits"])
    residual_bits = int(info["rice_payload_bits"])
    offset = _HEADER_BYTES
    model = unpack_ar1_bias_metadata(blob[offset : offset + metadata_bytes], dimensions)
    offset += metadata_bytes
    scales = blob[offset : offset + 8 * dimensions]
    offset += 8 * dimensions
    lengths = blob[offset : offset + 32]
    offset += 32
    ks = (
        np.frombuffer(blob[offset : offset + dimensions], dtype=np.uint8)
        .reshape(dimensions, 1)
        .copy()
    )
    offset += dimensions
    basis_bytes = (basis_bits + 7) // 8
    basis = blob[offset : offset + basis_bytes]
    offset += basis_bytes
    residuals = _unzigzag(
        _rice_decode(ks, blob[offset:], residual_bits, frames, dimensions)
    )
    try:
        coefficients = restore_ar1_bias(residuals, model)
        original = _predict_residuals(
            coefficients, np.zeros(dimensions, dtype=np.uint8), 0
        )
        original_ks, original_payload, original_bits = _rice_encode(
            _zigzag(original), 1
        )
    except (CoefficientPredictorError, ValueError) as error:
        raise CoefficientAr1CodecError(
            f"invalid CAP1 reconstructed coefficients: {error}"
        ) from error
    return (
        CPR1_MAGIC
        + struct.pack("<II", basis_bits, original_bits)
        + scales
        + lengths
        + original_ks.reshape(-1).tobytes()
        + basis
        + original_payload
    )
