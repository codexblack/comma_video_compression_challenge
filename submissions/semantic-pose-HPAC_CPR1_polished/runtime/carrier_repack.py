"""Exact, versioned CPR1 carrier section ordering for Gate B."""

from __future__ import annotations

import struct

import numpy as np

from .frame0_selector import decode_selector

CPR1_MAGIC = b"CPR1"
CPS2_MAGIC = b"CPS2"
CPS3_MAGIC = b"CPS3"
CVQ1_MAGIC = b"CVQ1"
CAP1_MAGIC = b"CAP1"
FRAME0_SELECTOR_CARRIER_MAGIC = b"F0C1"
_CPS3_VERSION = 1
_CPS3_FIRST = 0
_CPS3_SECOND = 1
_CPS3_PER_DIM = 2
_CPS3_PIECEWISE = 3
_ORDERS = {
    0: ("scales", "lengths", "ks", "basis", "coefficients"),
    1: ("lengths", "ks", "basis", "coefficients", "scales"),
    2: ("coefficients", "basis", "scales", "lengths", "ks"),
    3: ("scales", "basis", "coefficients", "lengths", "ks"),
}


FIXED_CPS3_PREFIX_BYTES = 4


class CarrierRepackError(ValueError):
    """A CPR1/CPS2 carrier representation is malformed."""


def _u24(value: int) -> bytes:
    if not 0 < value < 1 << 24:
        raise CarrierRepackError("carrier bit count does not fit u24")
    return value.to_bytes(3, "little")


def _read_u24(raw: bytes) -> int:
    if len(raw) != 3:
        raise CarrierRepackError("truncated u24 carrier bit count")
    return int.from_bytes(raw, "little")


def _parse_cpr1(raw: bytes, *, dimensions: int) -> tuple[int, int, dict[str, bytes]]:
    prefix = 12 + 8 * dimensions + 32 + dimensions
    if len(raw) < prefix or raw[:4] != CPR1_MAGIC:
        raise CarrierRepackError("invalid CPR1 carrier magic or length")
    basis_bits, coefficient_bits = struct.unpack_from("<II", raw, 4)
    basis_bytes, coefficient_bytes = (basis_bits + 7) // 8, (coefficient_bits + 7) // 8
    if (
        not basis_bits
        or not coefficient_bits
        or len(raw) != prefix + basis_bytes + coefficient_bytes
    ):
        raise CarrierRepackError("invalid CPR1 carrier bit counts")
    offset = 12
    fields = {
        "scales": raw[offset : offset + 8 * dimensions],
        "lengths": raw[offset + 8 * dimensions : offset + 8 * dimensions + 32],
        "ks": raw[offset + 8 * dimensions + 32 : prefix],
        "basis": raw[prefix : prefix + basis_bytes],
        "coefficients": raw[prefix + basis_bytes :],
    }
    return basis_bits, coefficient_bits, fields


def encode_cps2(raw_cpr1: bytes, *, dimensions: int, order: int) -> bytes:
    """Reorder exact CPR1 sections; all bit counts remain charged as u24."""
    if order not in _ORDERS:
        raise CarrierRepackError("unsupported CPS2 section order")
    basis_bits, coefficient_bits, fields = _parse_cpr1(raw_cpr1, dimensions=dimensions)
    output = bytearray(
        CPS2_MAGIC + bytes((order,)) + _u24(basis_bits) + _u24(coefficient_bits)
    )
    for name in _ORDERS[order]:
        output.extend(fields[name])
    return bytes(output)


def decode_cps2(blob: bytes, *, dimensions: int) -> bytes:
    if len(blob) < 11 or blob[:4] != CPS2_MAGIC:
        raise CarrierRepackError("invalid CPS2 carrier magic or length")
    order = blob[4]
    if order not in _ORDERS:
        raise CarrierRepackError("reserved CPS2 section order")
    basis_bits, coefficient_bits = _read_u24(blob[5:8]), _read_u24(blob[8:11])
    basis_bytes, coefficient_bytes = (basis_bits + 7) // 8, (coefficient_bits + 7) // 8
    sizes = {
        "scales": 8 * dimensions,
        "lengths": 32,
        "ks": dimensions,
        "basis": basis_bytes,
        "coefficients": coefficient_bytes,
    }
    offset = 11
    fields: dict[str, bytes] = {}
    for name in _ORDERS[order]:
        size = sizes[name]
        if len(blob) < offset + size:
            raise CarrierRepackError(f"truncated CPS2 {name} section")
        fields[name] = blob[offset : offset + size]
        offset += size
    if offset != len(blob):
        raise CarrierRepackError("CPS2 trailing data")
    raw = (
        CPR1_MAGIC
        + struct.pack("<II", basis_bits, coefficient_bits)
        + fields["scales"]
        + fields["lengths"]
        + fields["ks"]
        + fields["basis"]
        + fields["coefficients"]
    )
    # Re-parse to reject zero padding/length inconsistencies before handoff.
    parsed_bits, parsed_coeff_bits, parsed_fields = _parse_cpr1(
        raw, dimensions=dimensions
    )
    if (
        parsed_bits != basis_bits
        or parsed_coeff_bits != coefficient_bits
        or any(parsed_fields[name] != fields[name] for name in fields)
    ):
        raise CarrierRepackError("non-canonical CPS2 carrier")
    return raw


def materialize_cpr1(blob: bytes, runtime) -> bytes:
    blob, _ = split_frame0_selector_carrier(blob)
    if blob.startswith(CPR1_MAGIC):
        return blob
    if blob.startswith(CPS2_MAGIC):
        return decode_cps2(blob, dimensions=runtime.CARRIER_DIM)
    if blob.startswith(CPS3_MAGIC):
        return decode_cps3(blob, frames=runtime.N, dimensions=runtime.CARRIER_DIM)
    if blob.startswith(CVQ1_MAGIC):
        return decode_cvq1(blob, frames=runtime.N, dimensions=runtime.CARRIER_DIM)
    if blob.startswith(CAP1_MAGIC):
        from .entropy.coefficient_ar1_codec import decode_cap1

        return decode_cap1(blob, frames=runtime.N, dimensions=runtime.CARRIER_DIM)
    if blob.startswith(CVH1_MAGIC):
        return decode_cvh1(blob, frames=runtime.N, dimensions=runtime.CARRIER_DIM)
    raise CarrierRepackError("unknown carrier representation magic")


def pack_frame0_selector_carrier(carrier_blob: bytes, selector_blob: bytes) -> bytes:
    """Attach a fully charged selector payload to one carrier representation."""
    if not carrier_blob or len(carrier_blob) >= 1 << 16:
        raise CarrierRepackError("selector carrier base length must fit u16")
    try:
        decode_selector(selector_blob)
    except ValueError as error:
        raise CarrierRepackError(
            f"invalid frame-0 selector payload: {error}"
        ) from error
    return (
        FRAME0_SELECTOR_CARRIER_MAGIC
        + struct.pack("<H", len(carrier_blob))
        + carrier_blob
        + selector_blob
    )


def split_frame0_selector_carrier(blob: bytes) -> tuple[bytes, bytes | None]:
    """Return the base carrier and optional strict selector payload."""
    if not blob.startswith(FRAME0_SELECTOR_CARRIER_MAGIC):
        return blob, None
    if len(blob) < 7:
        raise CarrierRepackError("truncated frame-0 selector carrier")
    carrier_bytes = struct.unpack_from("<H", blob, 4)[0]
    offset = 6
    if not carrier_bytes or offset + carrier_bytes >= len(blob):
        raise CarrierRepackError("invalid frame-0 selector carrier length")
    carrier, selector = (
        blob[offset : offset + carrier_bytes],
        blob[offset + carrier_bytes :],
    )
    try:
        decode_selector(selector)
    except ValueError as error:
        raise CarrierRepackError(
            f"invalid frame-0 selector payload: {error}"
        ) from error
    return carrier, selector


def _pack_bits(bits) -> tuple[bytes, int]:
    values = list(bits)
    if not values or any(bit not in (0, 1) for bit in values):
        raise CarrierRepackError("invalid empty or non-binary carrier bitstream")
    return np.packbits(
        np.asarray(values, dtype=np.uint8), bitorder="big"
    ).tobytes(), len(values)


def _zigzag(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < -2048) or np.any(values > 2047):
        raise CarrierRepackError("predictive residual exceeds signed int12")
    return ((values << 1) ^ (values >> 63)).astype(np.int32) & 0xFFF


def _unzigzag(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= 4096):
        raise CarrierRepackError("Rice value exceeds unsigned int12")
    return ((values >> 1) ^ -(values & 1)).astype(np.int32)


def _rice_bits(values: np.ndarray, k: int) -> int:
    quotient_sum = int((np.asarray(values, dtype=np.uint64) >> k).sum())
    return quotient_sum + int(values.size) * (k + 1)


def _rice_encode(values: np.ndarray, segments: int) -> tuple[np.ndarray, bytes, int]:
    values = np.asarray(values, dtype=np.int64)
    if (
        values.ndim != 2
        or not values.size
        or np.any(values < 0)
        or np.any(values >= 4096)
        or segments not in (1, 2, 4, 8)
    ):
        raise CarrierRepackError("invalid predictive Rice input")
    frames, dimensions = values.shape
    if frames % segments:
        raise CarrierRepackError("Rice segment count must divide frame count")
    per_segment = frames // segments
    ks = np.empty((dimensions, segments), dtype=np.uint8)
    bits = []
    for dimension in range(dimensions):
        for segment in range(segments):
            column = values[
                segment * per_segment : (segment + 1) * per_segment, dimension
            ]
            _, k = min(
                (_rice_bits(column, candidate), candidate) for candidate in range(12)
            )
            ks[dimension, segment] = k
            for item in column:
                item = int(item)
                bits.extend((0,) * (item >> k))
                bits.append(1)
                bits.extend((item >> shift) & 1 for shift in range(k - 1, -1, -1))
    payload, count = _pack_bits(bits)
    return ks, payload, count


def _rice_decode(
    ks: np.ndarray, payload: bytes, count: int, frames: int, dimensions: int
) -> np.ndarray:
    ks = np.asarray(ks, dtype=np.int64)
    if (
        ks.ndim != 2
        or ks.shape[0] != dimensions
        or not ks.shape[1]
        or frames % ks.shape[1]
        or count <= 0
        or len(payload) != (count + 7) // 8
    ):
        raise CarrierRepackError("invalid predictive Rice payload")
    if (
        np.any(ks < 0)
        or np.any(ks >= 12)
        or (count % 8 and payload[-1] & ((1 << (8 - count % 8)) - 1))
    ):
        raise CarrierRepackError("invalid predictive Rice parameters or padding")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[:count]
    result = np.empty((frames, dimensions), dtype=np.int32)
    cursor = 0
    per_segment = frames // ks.shape[1]
    for dimension in range(dimensions):
        for segment in range(ks.shape[1]):
            k = int(ks[dimension, segment])
            for frame in range(segment * per_segment, (segment + 1) * per_segment):
                quotient = 0
                while True:
                    if cursor >= count:
                        raise CarrierRepackError("truncated predictive Rice unary code")
                    bit = int(bits[cursor])
                    cursor += 1
                    if bit:
                        break
                    quotient += 1
                    if quotient > 4095 >> k:
                        raise CarrierRepackError("predictive Rice value exceeds int12")
                if cursor + k > count:
                    raise CarrierRepackError("truncated predictive Rice remainder")
                remainder = 0
                for _ in range(k):
                    remainder = (remainder << 1) | int(bits[cursor])
                    cursor += 1
                result[frame, dimension] = (quotient << k) | remainder
    if cursor != count:
        raise CarrierRepackError("predictive Rice trailing bits")
    return result


def _signed_mod(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.int64) + 2048) & 0xFFF).astype(np.int32) - 2048


def _predictor_ids(coefficients: np.ndarray, mode: int) -> tuple[np.ndarray, int]:
    dimensions = coefficients.shape[1]
    if mode == _CPS3_FIRST:
        return np.zeros(dimensions, dtype=np.uint8), 0
    if mode == _CPS3_SECOND:
        return np.ones(dimensions, dtype=np.uint8), 0
    if mode != _CPS3_PER_DIM:
        raise CarrierRepackError("unsupported per-dimension predictor mode")
    candidates = []
    for predictor in (0, 1):
        residuals = _predict_residuals(
            coefficients, np.full(dimensions, predictor, dtype=np.uint8), 0
        )
        candidates.append(
            sum(
                min(_rice_bits(residuals[:, dim], k) for k in range(12))
                for dim in range(dimensions)
            )
        )
    # Select each dimension independently rather than choosing one global mode.
    choices = np.empty(dimensions, dtype=np.uint8)
    for dim in range(dimensions):
        choices[dim] = min(
            (
                min(
                    _rice_bits(
                        _predict_residuals(
                            coefficients, np.full(dimensions, p, dtype=np.uint8), 0
                        )[:, dim],
                        k,
                    )
                    for k in range(12)
                ),
                p,
            )
            for p in (0, 1)
        )[1]
    return choices, 0


def _predict_residuals(
    coefficients: np.ndarray, ids: np.ndarray, threshold: int
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.int32)
    frames, dimensions = values.shape
    result = np.empty_like(values)
    for frame in range(frames):
        if frame == 0:
            prediction = np.zeros(dimensions, dtype=np.int32)
        elif frame == 1:
            prediction = values[frame - 1]
        else:
            previous_delta = _signed_mod(values[frame - 1] - values[frame - 2])
            second = _signed_mod(values[frame - 1] + previous_delta)
            if threshold:
                use_second = np.abs(previous_delta) <= threshold
                prediction = np.where(use_second, second, values[frame - 1])
            else:
                prediction = np.where(ids.astype(bool), second, values[frame - 1])
        result[frame] = _signed_mod(values[frame] - prediction)
    return result


def _restore_coefficients(
    residuals: np.ndarray, ids: np.ndarray, threshold: int
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.int32)
    frames, dimensions = residuals.shape
    values = np.empty_like(residuals)
    for frame in range(frames):
        if frame == 0:
            prediction = np.zeros(dimensions, dtype=np.int32)
        elif frame == 1:
            prediction = values[frame - 1]
        else:
            previous_delta = _signed_mod(values[frame - 1] - values[frame - 2])
            second = _signed_mod(values[frame - 1] + previous_delta)
            prediction = (
                np.where(np.abs(previous_delta) <= threshold, second, values[frame - 1])
                if threshold
                else np.where(ids.astype(bool), second, values[frame - 1])
            )
        values[frame] = _signed_mod(prediction + residuals[frame])
    return values


def _coefficients_from_cpr1(
    raw: bytes, *, frames: int, dimensions: int
) -> tuple[dict[str, bytes], np.ndarray]:
    _basis_bits, coefficient_bits, fields = _parse_cpr1(raw, dimensions=dimensions)
    # CPR1's original table is one Rice segment per dimension.
    encoded = _rice_decode(
        np.frombuffer(fields["ks"], dtype=np.uint8).reshape(dimensions, 1),
        fields["coefficients"],
        coefficient_bits,
        frames,
        dimensions,
    )
    return fields, _restore_coefficients(
        _unzigzag(encoded), np.zeros(dimensions, dtype=np.uint8), 0
    )


def decode_cpr1_coefficients(
    raw_cpr1: bytes, *, frames: int, dimensions: int
) -> np.ndarray:
    """Return the signed int12 carrier codes in frame-major order."""
    _, coefficients = _coefficients_from_cpr1(
        raw_cpr1, frames=frames, dimensions=dimensions
    )
    return coefficients.copy()


def cpr1_coefficient_scales(raw_cpr1: bytes, *, dimensions: int) -> np.ndarray:
    """Return the exact float32 per-dimension coefficient scales."""
    _, _, fields = _parse_cpr1(raw_cpr1, dimensions=dimensions)
    return np.frombuffer(fields["scales"][4 * dimensions :], dtype="<f4").copy()


def _encode_cps3(
    fields: dict[str, bytes],
    coefficients: np.ndarray,
    *,
    frames: int,
    dimensions: int,
    mode: int,
    segments: int,
    threshold: int,
) -> bytes:
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (frames, dimensions) or not np.issubdtype(
        coefficients.dtype, np.integer
    ):
        raise CarrierRepackError(
            "carrier coefficients must be a frame-major integer matrix"
        )
    coefficients = coefficients.astype(np.int32, copy=False)
    if np.any(coefficients < -2048) or np.any(coefficients > 2047):
        raise CarrierRepackError("carrier coefficients exceed signed int12")
    if mode == _CPS3_PIECEWISE:
        if not 1 <= threshold <= 127:
            raise CarrierRepackError(
                "piecewise predictor needs a charged threshold in 1..127"
            )
        ids = np.zeros(dimensions, dtype=np.uint8)
    else:
        if threshold:
            raise CarrierRepackError(
                "only the piecewise predictor may carry a threshold"
            )
        ids, _ = _predictor_ids(coefficients, mode)
    residuals = _predict_residuals(coefficients, ids, threshold)
    ks, payload, bit_count = _rice_encode(_zigzag(residuals), segments)
    basis_bits = struct.unpack_from("<I", fields["header"], 0)[0]
    metadata = (
        b"" if mode != _CPS3_PER_DIM else np.packbits(ids, bitorder="little").tobytes()
    )
    header = (
        CPS3_MAGIC
        + bytes((_CPS3_VERSION, mode, segments, threshold))
        + _u24(basis_bits)
        + _u24(bit_count)
    )
    return (
        header
        + metadata
        + fields["scales"]
        + fields["lengths"]
        + ks.tobytes()
        + fields["basis"]
        + payload
    )


def encode_cps3_coefficients(
    raw_cpr1: bytes,
    coefficients: np.ndarray,
    *,
    frames: int,
    dimensions: int,
    mode: int,
    segments: int = 1,
    threshold: int = 0,
) -> bytes:
    """Encode supplied signed int12 coefficients with frozen CPR1 basis fields."""
    basis_bits, _, fields = _parse_cpr1(raw_cpr1, dimensions=dimensions)
    fields = dict(fields)
    fields["header"] = struct.pack("<I", basis_bits)
    return _encode_cps3(
        fields,
        coefficients,
        frames=frames,
        dimensions=dimensions,
        mode=mode,
        segments=segments,
        threshold=threshold,
    )


def encode_cps3(
    raw_cpr1: bytes,
    *,
    frames: int,
    dimensions: int,
    mode: int,
    segments: int = 1,
    threshold: int = 0,
) -> bytes:
    """Exact predictive Rice carrier; basis fields are retained byte-for-byte."""
    return encode_cps3_coefficients(
        raw_cpr1,
        decode_cpr1_coefficients(raw_cpr1, frames=frames, dimensions=dimensions),
        frames=frames,
        dimensions=dimensions,
        mode=mode,
        segments=segments,
        threshold=threshold,
    )


def decode_cps3(blob: bytes, *, frames: int, dimensions: int) -> bytes:
    if len(blob) < 14 or blob[:4] != CPS3_MAGIC:
        raise CarrierRepackError("invalid CPS3 carrier magic or length")
    version, mode, segments, threshold = blob[4:8]
    if (
        version != _CPS3_VERSION
        or mode not in (_CPS3_FIRST, _CPS3_SECOND, _CPS3_PER_DIM, _CPS3_PIECEWISE)
        or segments not in (1, 2, 4, 8)
        or (mode == _CPS3_PIECEWISE) != bool(threshold)
    ):
        raise CarrierRepackError("invalid CPS3 header")
    basis_bits, coefficient_bits = _read_u24(blob[8:11]), _read_u24(blob[11:14])
    offset = 14
    meta_bytes = (dimensions + 7) // 8 if mode == _CPS3_PER_DIM else 0
    if len(blob) < offset + meta_bytes + 8 * dimensions + 32 + dimensions * segments:
        raise CarrierRepackError("truncated CPS3 metadata")
    ids = (
        np.unpackbits(
            np.frombuffer(blob[offset : offset + meta_bytes], dtype=np.uint8),
            bitorder="little",
        )[:dimensions]
        if meta_bytes
        else np.full(dimensions, mode == _CPS3_SECOND, dtype=np.uint8)
    )
    offset += meta_bytes
    scales = blob[offset : offset + 8 * dimensions]
    offset += 8 * dimensions
    lengths = blob[offset : offset + 32]
    offset += 32
    ks = (
        np.frombuffer(blob[offset : offset + dimensions * segments], dtype=np.uint8)
        .reshape(dimensions, segments)
        .copy()
    )
    offset += dimensions * segments
    basis_bytes, coeff_bytes = (basis_bits + 7) // 8, (coefficient_bits + 7) // 8
    if (
        not basis_bits
        or not coefficient_bits
        or len(blob) != offset + basis_bytes + coeff_bytes
    ):
        raise CarrierRepackError("invalid CPS3 payload lengths")
    basis, payload = blob[offset : offset + basis_bytes], blob[offset + basis_bytes :]
    coefficients = _restore_coefficients(
        _unzigzag(_rice_decode(ks, payload, coefficient_bits, frames, dimensions)),
        ids,
        threshold,
    )
    original = _zigzag(
        _predict_residuals(coefficients, np.zeros(dimensions, dtype=np.uint8), 0)
    )
    original_ks, original_payload, original_bits = _rice_encode(original, 1)
    return (
        CPR1_MAGIC
        + struct.pack("<II", basis_bits, original_bits)
        + scales
        + lengths
        + original_ks.reshape(-1).tobytes()
        + basis
        + original_payload
    )


_CVQ1_VERSION = 1
_CVQ1_ABSOLUTE = 0
_CVQ1_AR1 = 1
_CVQ1_AR1_PARENT = 2
_CVQ1_HEADER = struct.Struct("<4sBBBBH3s3sB")


def _cvq_round_q8(values: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Signed, platform-stable nearest-integer Q8 multiplication."""
    products = np.asarray(values, dtype=np.int64) * np.asarray(factors, dtype=np.int64)
    return np.where(
        products >= 0, (products + 128) // 256, -((-products + 128) // 256)
    ).astype(np.int32)


def _cvq_ar_factors(coefficients: np.ndarray) -> np.ndarray:
    """Fit one bounded, signed Q8 AR(1) factor per coefficient dimension."""
    previous = np.asarray(coefficients[:-1], dtype=np.int64)
    current = np.asarray(coefficients[1:], dtype=np.int64)
    denominator = (previous * previous).sum(axis=0)
    numerator = (previous * current).sum(axis=0)
    factors = np.zeros(coefficients.shape[1], dtype=np.int16)
    active = denominator != 0
    # rint is deterministic for these small integer values; the stored Q8
    # coefficient is the decoder contract, never the transient float value.
    factors[active] = np.clip(
        np.rint(256.0 * numerator[active] / denominator[active]), -512, 511
    ).astype(np.int16)
    return factors


def _cvq_parent_factors(
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one signed-Q8 AR factor and at most one causal current-frame parent."""
    values = np.asarray(coefficients, dtype=np.int32)
    frames, dimensions = values.shape
    ar = _cvq_ar_factors(values)
    parents = np.full(dimensions, 255, dtype=np.uint8)
    cross = np.zeros(dimensions, dtype=np.int16)
    if frames < 2:
        return ar, parents, cross
    for dimension in range(1, dimensions):
        previous = values[:-1, dimension].astype(np.int64)
        target = values[1:, dimension].astype(np.int64)
        baseline_prediction = _cvq_round_q8(
            previous, np.full(previous.shape, ar[dimension], dtype=np.int16)
        )
        best = (
            int(np.square(_signed_mod(target - baseline_prediction)).sum()),
            255,
            int(ar[dimension]),
            0,
        )
        for parent in range(dimension):
            current_parent = values[1:, parent].astype(np.int64)
            xx = int((previous * previous).sum())
            zz = int((current_parent * current_parent).sum())
            xz = int((previous * current_parent).sum())
            xy = int((previous * target).sum())
            zy = int((current_parent * target).sum())
            determinant = xx * zz - xz * xz
            if determinant == 0:
                continue
            a = int(
                np.clip(np.rint(256.0 * (xy * zz - zy * xz) / determinant), -512, 511)
            )
            b = int(
                np.clip(np.rint(256.0 * (zy * xx - xy * xz) / determinant), -512, 511)
            )
            products = previous * a + current_parent * b
            prediction = np.where(
                products >= 0, (products + 128) // 256, -((-products + 128) // 256)
            ).astype(np.int32)
            candidate = (
                int(np.square(_signed_mod(target - prediction)).sum()),
                parent,
                a,
                b,
            )
            best = min(best, candidate)
        _, parent, a, b = best
        parents[dimension] = parent
        ar[dimension] = a
        cross[dimension] = b
    return ar, parents, cross


def _cvq_predictor_bytes(predictor: int, dimensions: int) -> int:
    if predictor == _CVQ1_ABSOLUTE:
        return 0
    if predictor == _CVQ1_AR1:
        return 2 * dimensions
    if predictor == _CVQ1_AR1_PARENT:
        return 5 * dimensions
    raise CarrierRepackError("unsupported CVQ predictor")


def _cvq_predictor_pack(predictor: int, factors) -> bytes:
    if predictor == _CVQ1_ABSOLUTE:
        return b""
    if predictor == _CVQ1_AR1:
        return np.asarray(factors, dtype="<i2").tobytes()
    ar, parents, cross = factors
    return (
        np.asarray(ar, dtype="<i2").tobytes()
        + np.asarray(parents, dtype=np.uint8).tobytes()
        + np.asarray(cross, dtype="<i2").tobytes()
    )


def _cvq_predictor_unpack(blob: bytes, offset: int, predictor: int, dimensions: int):
    size = _cvq_predictor_bytes(predictor, dimensions)
    if len(blob) < offset + size:
        raise CarrierRepackError("truncated CVQ predictor metadata")
    if predictor == _CVQ1_ABSOLUTE:
        return None, offset
    ar = np.frombuffer(blob[offset : offset + 2 * dimensions], dtype="<i2").astype(
        np.int16
    )
    if predictor == _CVQ1_AR1:
        return ar, offset + size
    parents = np.frombuffer(
        blob[offset + 2 * dimensions : offset + 3 * dimensions], dtype=np.uint8
    ).copy()
    cross = np.frombuffer(
        blob[offset + 3 * dimensions : offset + size], dtype="<i2"
    ).astype(np.int16)
    for dimension, parent in enumerate(parents):
        if (parent != 255 and parent >= dimension) or (
            parent == 255 and cross[dimension] != 0
        ):
            raise CarrierRepackError("invalid CVQ causal parent metadata")
    return (ar, parents, cross), offset + size


def _cvq_residuals(
    coefficients: np.ndarray, predictor: int, factors: np.ndarray | None
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.int32)
    result = np.empty_like(values)
    for frame in range(values.shape[0]):
        if frame == 0 or predictor == _CVQ1_ABSOLUTE:
            prediction = np.zeros(values.shape[1], dtype=np.int32)
        elif predictor == _CVQ1_AR1:
            if factors is None:
                raise CarrierRepackError("AR1 CVQ predictor requires stored factors")
            prediction = _cvq_round_q8(values[frame - 1], factors)
        elif predictor == _CVQ1_AR1_PARENT:
            if factors is None:
                raise CarrierRepackError(
                    "AR1-parent CVQ predictor requires stored factors"
                )
            ar, parents, cross = factors
            prediction = _cvq_round_q8(values[frame - 1], ar)
            for dimension, parent in enumerate(parents):
                if parent != 255:
                    products = np.int64(values[frame - 1, dimension]) * int(
                        ar[dimension]
                    ) + np.int64(values[frame, parent]) * int(cross[dimension])
                    prediction[dimension] = int(
                        (products + 128) // 256
                        if products >= 0
                        else -((-products + 128) // 256)
                    )
        else:  # pragma: no cover - callers validate the closed predictor set.
            raise CarrierRepackError("unsupported CVQ predictor")
        result[frame] = _signed_mod(values[frame] - prediction)
    return result


def _cvq_restore(
    residuals: np.ndarray, predictor: int, factors: np.ndarray | None
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.int32)
    result = np.empty_like(residuals)
    for frame in range(result.shape[0]):
        if frame == 0 or predictor == _CVQ1_ABSOLUTE:
            prediction = np.zeros(result.shape[1], dtype=np.int32)
        elif predictor == _CVQ1_AR1:
            if factors is None:
                raise CarrierRepackError("AR1 CVQ predictor requires stored factors")
            prediction = _cvq_round_q8(result[frame - 1], factors)
        elif predictor == _CVQ1_AR1_PARENT:
            if factors is None:
                raise CarrierRepackError(
                    "AR1-parent CVQ predictor requires stored factors"
                )
            ar, parents, cross = factors
            restored = np.empty(result.shape[1], dtype=np.int32)
            for dimension, parent in enumerate(parents):
                products = np.int64(result[frame - 1, dimension]) * int(ar[dimension])
                if parent != 255:
                    products += np.int64(restored[parent]) * int(cross[dimension])
                prediction = int(
                    (products + 128) // 256
                    if products >= 0
                    else -((-products + 128) // 256)
                )
                restored[dimension] = _signed_mod(
                    np.asarray(
                        [prediction + residuals[frame, dimension]], dtype=np.int64
                    )
                )[0]
            result[frame] = restored
            continue
        else:  # pragma: no cover - callers validate the closed predictor set.
            raise CarrierRepackError("unsupported CVQ predictor")
        result[frame] = _signed_mod(prediction + residuals[frame])
    return result


def _cvq_codebook(
    values: np.ndarray, k: int, iterations: int, *, implicit_zero: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded deterministic integer Lloyd clustering for one vector per frame."""
    points = np.asarray(values, dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < k or iterations < 1:
        raise CarrierRepackError("invalid CVQ clustering input")
    # Quantile seeds are deterministic and retain the observed negative as
    # well as positive temporal residual modes.  Ties resolve by frame order.
    ordering = np.lexsort(
        tuple(points[:, dimension] for dimension in range(points.shape[1] - 1, -1, -1))
    )
    selected = np.linspace(0, points.shape[0] - 1, k, dtype=np.int64)
    centroids = points[ordering[selected]].copy()
    if implicit_zero:
        centroids[0] = 0
    assignment = np.zeros(points.shape[0], dtype=np.uint8)
    for _ in range(iterations):
        differences = points[:, None, :].astype(np.int64) - centroids[
            None, :, :
        ].astype(np.int64)
        assignment = np.argmin((differences * differences).sum(axis=2), axis=1).astype(
            np.uint8
        )
        updated = centroids.copy()
        for index in range(k):
            if implicit_zero and index == 0:
                continue
            members = points[assignment == index]
            if members.size:
                updated[index] = np.clip(
                    np.rint(members.mean(axis=0)), -2048, 2047
                ).astype(np.int32)
        if np.array_equal(updated, centroids):
            break
        centroids = updated
    differences = points[:, None, :].astype(np.int64) - centroids[None, :, :].astype(
        np.int64
    )
    assignment = np.argmin((differences * differences).sum(axis=2), axis=1).astype(
        np.uint8
    )
    return centroids, assignment


def _cvq_pack_fixed(values: np.ndarray, width: int) -> bytes:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if not 1 <= width <= 8 or np.any(values < 0) or np.any(values >= 1 << width):
        raise CarrierRepackError("invalid fixed-width CVQ values")
    bits: list[int] = []
    for value in values:
        bits.extend((int(value) >> shift) & 1 for shift in range(width - 1, -1, -1))
    return _pack_bits(bits)[0]


def _cvq_unpack_fixed(payload: bytes, count: int, width: int) -> np.ndarray:
    expected = (count * width + 7) // 8
    if len(payload) != expected or not 1 <= width <= 8:
        raise CarrierRepackError("invalid fixed-width CVQ payload length")
    bit_count = count * width
    if bit_count % 8 and payload[-1] & ((1 << (8 - bit_count % 8)) - 1):
        raise CarrierRepackError("nonzero fixed-width CVQ padding")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[
        :bit_count
    ]
    output = np.zeros(count, dtype=np.int32)
    for bit in range(width):
        output = (output << 1) | bits[bit::width]
    return output


def _cvq_zigzag(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < -4095) or np.any(values > 4095):
        raise CarrierRepackError("CVQ correction exceeds signed int13")
    return ((values << 1) ^ (values >> 63)).astype(np.int32)


def _cvq_unzigzag(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < 0) or np.any(values > 8190):
        raise CarrierRepackError("CVQ correction exceeds unsigned int13")
    return ((values >> 1) ^ -(values & 1)).astype(np.int32)


def _cvq_rice_encode(values: np.ndarray) -> tuple[int, bytes, int]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if not values.size:
        return 0, b"", 0
    if np.any(values < 0) or np.any(values > 8190):
        raise CarrierRepackError("invalid CVQ Rice correction values")
    k = min((_rice_bits(values, candidate), candidate) for candidate in range(14))[1]
    bits: list[int] = []
    for value in values:
        integer = int(value)
        bits.extend((0,) * (integer >> k))
        bits.append(1)
        bits.extend((integer >> shift) & 1 for shift in range(k - 1, -1, -1))
    payload, bit_count = _pack_bits(bits)
    return k, payload, bit_count


def _cvq_rice_decode(k: int, payload: bytes, bit_count: int, count: int) -> np.ndarray:
    if (
        not 0 <= k < 14
        or count < 0
        or bit_count < 0
        or len(payload) != (bit_count + 7) // 8
    ):
        raise CarrierRepackError("invalid CVQ Rice metadata")
    if count == 0:
        if payload or bit_count or k:
            raise CarrierRepackError("noncanonical empty CVQ correction stream")
        return np.empty(0, dtype=np.int32)
    if (
        not bit_count
        or bit_count > len(payload) * 8
        or (bit_count % 8 and payload[-1] & ((1 << (8 - bit_count % 8)) - 1))
    ):
        raise CarrierRepackError("invalid CVQ Rice padding")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[
        :bit_count
    ]
    output = np.empty(count, dtype=np.int32)
    cursor = 0
    for item in range(count):
        quotient = 0
        while True:
            if cursor >= bit_count:
                raise CarrierRepackError("truncated CVQ Rice unary code")
            bit = int(bits[cursor])
            cursor += 1
            if bit:
                break
            quotient += 1
            if quotient > 8190 >> k:
                raise CarrierRepackError("CVQ Rice value exceeds int13")
        if cursor + k > bit_count:
            raise CarrierRepackError("truncated CVQ Rice remainder")
        remainder = 0
        for _ in range(k):
            remainder = (remainder << 1) | int(bits[cursor])
            cursor += 1
        output[item] = (quotient << k) | remainder
    if cursor != bit_count:
        raise CarrierRepackError("CVQ Rice trailing bits")
    return output


def _cvq_header(
    blob: bytes, *, frames: int, dimensions: int
) -> tuple[int, int, int, int, int, int]:
    if len(blob) < _CVQ1_HEADER.size:
        raise CarrierRepackError("truncated CVQ1 header")
    (
        magic,
        version,
        predictor,
        k,
        stored_dimensions,
        stored_frames,
        basis_raw,
        correction_raw,
        code_width,
    ) = _CVQ1_HEADER.unpack_from(blob)
    basis_bits, correction_bits = (
        _read_u24(basis_raw),
        int.from_bytes(correction_raw, "little"),
    )
    if (
        magic != CVQ1_MAGIC
        or version != _CVQ1_VERSION
        or predictor not in (_CVQ1_ABSOLUTE, _CVQ1_AR1, _CVQ1_AR1_PARENT)
        or k not in (8, 16)
        or stored_dimensions != dimensions
        or stored_frames != frames
        or not basis_bits
        or code_width not in (1, 2)
    ):
        raise CarrierRepackError("invalid CVQ1 header")
    return predictor, k, basis_bits, correction_bits, code_width, _CVQ1_HEADER.size


def inspect_cvq1(blob: bytes, *, frames: int, dimensions: int) -> dict[str, int | str]:
    """Return fully charged CVQ1 component sizes after strict structural checks."""
    predictor, k, basis_bits, correction_bits, code_width, offset = _cvq_header(
        blob, frames=frames, dimensions=dimensions
    )
    ar_bytes = _cvq_predictor_bytes(predictor, dimensions)
    basis_bytes = (basis_bits + 7) // 8
    index_width = 3 if k == 8 else 4
    index_bytes = (frames * index_width + 7) // 8
    codebook_bytes = (k - 1) * dimensions * code_width
    mask_capacity = (frames * dimensions + 7) // 8
    prefix = (
        offset
        + ar_bytes
        + 8 * dimensions
        + 32
        + basis_bytes
        + codebook_bytes
        + index_bytes
    )
    if len(blob) <= prefix:
        raise CarrierRepackError("truncated CVQ1 correction mode")
    correction_mode = int(blob[prefix])
    if correction_mode == 0:  # dense values; no bitmap is needed.
        correction_mask_bytes, correction_rice_header_bytes = 0, 1
    elif correction_mode == 1:  # sparse values gated by a charged bitmap.
        correction_mask_bytes, correction_rice_header_bytes = mask_capacity, 1
    elif correction_mode == 2:  # every codeword vector is already exact.
        if correction_bits:
            raise CarrierRepackError("nonempty CVQ1 exact-codeword correction stream")
        correction_mask_bytes, correction_rice_header_bytes = 0, 0
    else:
        raise CarrierRepackError("invalid CVQ1 correction mode")
    correction_value_bytes = (correction_bits + 7) // 8
    expected = (
        prefix
        + 1
        + correction_mask_bytes
        + correction_rice_header_bytes
        + correction_value_bytes
    )
    if len(blob) != expected:
        raise CarrierRepackError("truncated or trailing CVQ1 payload")
    return {
        "predictor": "absolute"
        if predictor == _CVQ1_ABSOLUTE
        else "ar1_q8"
        if predictor == _CVQ1_AR1
        else "ar1_parent_q8",
        "k": k,
        "header_bytes": _CVQ1_HEADER.size,
        "predictor_bytes": ar_bytes,
        "frozen_metadata_bytes": 8 * dimensions + 32 + basis_bytes,
        "codebook_bytes": codebook_bytes,
        "index_bytes": index_bytes,
        "correction_format": (
            "dense"
            if correction_mode == 0
            else "sparse"
            if correction_mode == 1
            else "none"
        ),
        "correction_mode_bytes": 1,
        "correction_mask_bytes": correction_mask_bytes,
        "correction_rice_header_bytes": correction_rice_header_bytes,
        "correction_value_bytes": correction_value_bytes,
        "correction_bytes": 1
        + correction_mask_bytes
        + correction_rice_header_bytes
        + correction_value_bytes,
        "raw_carrier_bytes": len(blob),
    }


def encode_cvq1(
    raw_cpr1: bytes,
    *,
    frames: int,
    dimensions: int,
    k: int,
    predictor: str = "ar1",
    iterations: int = 16,
) -> bytes:
    """Exact CVQ carrier replacing CPR1's coefficient Rice stream.

    CVQ stores one K-vector codebook, packed per-frame indices, and a sparse
    bitmap/Rice correction stream.  The corrections make reconstruction exact;
    no rendered frame or pose coefficient is approximated by this format.
    """
    if (
        k not in (8, 16)
        or predictor not in {"absolute", "ar1", "ar1_parent"}
        or not 1 <= iterations <= 64
    ):
        raise CarrierRepackError("unsupported CVQ1 encoder configuration")
    basis_bits, _, fields = _parse_cpr1(raw_cpr1, dimensions=dimensions)
    coefficients = decode_cpr1_coefficients(
        raw_cpr1, frames=frames, dimensions=dimensions
    )
    predictor_id = (
        _CVQ1_ABSOLUTE
        if predictor == "absolute"
        else _CVQ1_AR1
        if predictor == "ar1"
        else _CVQ1_AR1_PARENT
    )
    factors = (
        None
        if predictor_id == _CVQ1_ABSOLUTE
        else _cvq_ar_factors(coefficients)
        if predictor_id == _CVQ1_AR1
        else _cvq_parent_factors(coefficients)
    )
    residuals = _cvq_residuals(coefficients, predictor_id, factors)
    codebook, indices = _cvq_codebook(residuals, k, iterations, implicit_zero=True)
    corrections = residuals - codebook[indices]
    flat_corrections = corrections.reshape(-1)
    correction_mask = flat_corrections != 0
    dense_values = _cvq_zigzag(flat_corrections)
    dense_k, dense_payload, dense_bits = _cvq_rice_encode(dense_values)
    sparse_values = _cvq_zigzag(flat_corrections[correction_mask])
    sparse_k, sparse_payload, sparse_bits = _cvq_rice_encode(sparse_values)
    mask_payload = np.packbits(
        correction_mask.astype(np.uint8), bitorder="big"
    ).tobytes()
    # Exact vector corrections are often dense; forcing an incompressible
    # bitmap in that case loses more than the codebook can save.
    options = [(1 + 1 + len(dense_payload), 0, b"", dense_k, dense_payload, dense_bits)]
    options.append(
        (
            1 + len(mask_payload) + 1 + len(sparse_payload),
            1,
            mask_payload,
            sparse_k,
            sparse_payload,
            sparse_bits,
        )
    )
    if not correction_mask.any():
        options.append((1, 2, b"", 0, b"", 0))
    _, correction_mode, mask_payload, rice_k, rice_payload, correction_bits = min(
        options, key=lambda item: (item[0], item[1])
    )
    code_width = 1 if int(codebook.min()) >= -128 and int(codebook.max()) <= 127 else 2
    codebook_payload = (
        codebook[1:].astype("i1" if code_width == 1 else "<i2", copy=False).tobytes()
    )
    index_payload = _cvq_pack_fixed(indices, 3 if k == 8 else 4)
    header = _CVQ1_HEADER.pack(
        CVQ1_MAGIC,
        _CVQ1_VERSION,
        predictor_id,
        k,
        dimensions,
        frames,
        _u24(basis_bits),
        correction_bits.to_bytes(3, "little"),
        code_width,
    )
    output = (
        header
        + _cvq_predictor_pack(predictor_id, factors)
        + fields["scales"]
        + fields["lengths"]
        + fields["basis"]
        + codebook_payload
        + index_payload
        + bytes((correction_mode,))
        + mask_payload
        + (b"" if correction_mode == 2 else bytes((rice_k,)) + rice_payload)
    )
    if decode_cvq1(output, frames=frames, dimensions=dimensions) != raw_cpr1:
        raise CarrierRepackError("CVQ1 encoder failed exact CPR1 reconstruction")
    return output


def decode_cvq1(blob: bytes, *, frames: int, dimensions: int) -> bytes:
    """Strictly restore the canonical CPR1 carrier from a CVQ1 payload."""
    predictor, k, basis_bits, correction_bits, code_width, offset = _cvq_header(
        blob, frames=frames, dimensions=dimensions
    )
    _cvq_predictor_bytes(predictor, dimensions)
    basis_bytes = (basis_bits + 7) // 8
    index_width = 3 if k == 8 else 4
    index_bytes = (frames * index_width + 7) // 8
    codebook_bytes = (k - 1) * dimensions * code_width
    mask_capacity = (frames * dimensions + 7) // 8
    factors, offset = _cvq_predictor_unpack(blob, offset, predictor, dimensions)
    scales = blob[offset : offset + 8 * dimensions]
    offset += 8 * dimensions
    lengths = blob[offset : offset + 32]
    offset += 32
    basis = blob[offset : offset + basis_bytes]
    offset += basis_bytes
    stored_codebook = (
        np.frombuffer(
            blob[offset : offset + codebook_bytes],
            dtype="i1" if code_width == 1 else "<i2",
        )
        .astype(np.int32)
        .reshape(k - 1, dimensions)
    )
    offset += codebook_bytes
    codebook = np.vstack((np.zeros((1, dimensions), dtype=np.int32), stored_codebook))
    indices = _cvq_unpack_fixed(
        blob[offset : offset + index_bytes], frames, index_width
    )
    offset += index_bytes
    if np.any(indices >= k):
        raise CarrierRepackError("CVQ1 index exceeds codebook")
    if len(blob) <= offset:
        raise CarrierRepackError("truncated CVQ1 correction mode")
    correction_mode = int(blob[offset])
    offset += 1
    if correction_mode == 0:
        correction_mask = np.ones(frames * dimensions, dtype=bool)
    elif correction_mode == 1:
        if len(blob) < offset + mask_capacity:
            raise CarrierRepackError("truncated CVQ1 correction mask")
        mask_blob = blob[offset : offset + mask_capacity]
        offset += mask_capacity
        if frames * dimensions % 8 and mask_blob[-1] & (
            (1 << (8 - frames * dimensions % 8)) - 1
        ):
            raise CarrierRepackError("nonzero CVQ1 correction-mask padding")
        correction_mask = np.unpackbits(
            np.frombuffer(mask_blob, dtype=np.uint8), bitorder="big"
        )[: frames * dimensions].astype(bool)
    elif correction_mode == 2:
        if correction_bits or offset != len(blob):
            raise CarrierRepackError(
                "noncanonical CVQ1 exact-codeword correction stream"
            )
        correction_mask = np.zeros(frames * dimensions, dtype=bool)
    else:
        raise CarrierRepackError("invalid CVQ1 correction mode")
    if correction_mode == 2:
        correction_values = np.empty(0, dtype=np.int32)
    else:
        if len(blob) <= offset:
            raise CarrierRepackError("truncated CVQ1 Rice parameter")
        rice_k = int(blob[offset])
        offset += 1
        correction_values = _cvq_rice_decode(
            rice_k, blob[offset:], correction_bits, int(correction_mask.sum())
        )
    residuals = codebook[indices]
    residuals.reshape(-1)[correction_mask] += _cvq_unzigzag(correction_values)
    if np.any(residuals < -2048) or np.any(residuals > 2047):
        raise CarrierRepackError("CVQ1 residual is outside signed int12")
    coefficients = _cvq_restore(residuals, predictor, factors)
    original = _zigzag(
        _predict_residuals(coefficients, np.zeros(dimensions, dtype=np.uint8), 0)
    )
    original_ks, original_payload, original_bits = _rice_encode(original, 1)
    return (
        CPR1_MAGIC
        + struct.pack("<II", basis_bits, original_bits)
        + scales
        + lengths
        + original_ks.reshape(-1).tobytes()
        + basis
        + original_payload
    )


CVH1_MAGIC = b"CVH1"
_CVH1_VERSION = 1
_CVH1_HEADER = struct.Struct("<4sBBBBH3sB")


def _cvh_rice_payload(residuals: np.ndarray) -> bytes:
    ks, payload, bits = _rice_encode(_zigzag(residuals), 1)
    if bits >= 1 << 16:
        raise CarrierRepackError("CVH Rice block bit count exceeds u16")
    return ks.tobytes() + struct.pack("<H", bits) + payload


def _cvh_rice_decode(payload: bytes, frames: int, dimensions: int) -> np.ndarray:
    if len(payload) < dimensions + 2:
        raise CarrierRepackError("truncated CVH Rice block")
    ks = (
        np.frombuffer(payload[:dimensions], dtype=np.uint8)
        .reshape(dimensions, 1)
        .copy()
    )
    bits = struct.unpack_from("<H", payload, dimensions)[0]
    values = payload[dimensions + 2 :]
    if len(values) != (bits + 7) // 8:
        raise CarrierRepackError("invalid CVH Rice block length")
    return _unzigzag(_rice_decode(ks, values, bits, frames, dimensions))


def _cvh_vq_payload(
    residuals: np.ndarray, codebook: np.ndarray, indices: np.ndarray, *, k: int
) -> bytes:
    _frames, _dimensions = residuals.shape
    width = 3 if k == 8 else 4
    corrections = residuals - codebook[indices]
    flat_corrections = corrections.reshape(-1)
    mask = flat_corrections != 0
    dense_k, dense_payload, dense_bits = _cvq_rice_encode(_cvq_zigzag(flat_corrections))
    sparse_k, sparse_payload, sparse_bits = _cvq_rice_encode(
        _cvq_zigzag(flat_corrections[mask])
    )
    mask_payload = np.packbits(mask.astype(np.uint8), bitorder="big").tobytes()
    options = [(1 + 1 + len(dense_payload), 0, b"", dense_k, dense_payload, dense_bits)]
    options.append(
        (
            1 + len(mask_payload) + 1 + len(sparse_payload),
            1,
            mask_payload,
            sparse_k,
            sparse_payload,
            sparse_bits,
        )
    )
    if not mask.any():
        options.append((1, 2, b"", 0, b"", 0))
    _, correction_mode, mask_payload, rice_k, rice_payload, rice_bits = min(
        options, key=lambda item: (item[0], item[1])
    )
    if rice_bits >= 1 << 16:
        raise CarrierRepackError("CVH VQ correction bit count exceeds u16")
    return (
        struct.pack("<H", rice_bits)
        + _cvq_pack_fixed(indices, width)
        + bytes((correction_mode,))
        + mask_payload
        + (b"" if correction_mode == 2 else bytes((rice_k,)) + rice_payload)
    )


def _cvh_vq_decode(
    payload: bytes, frames: int, dimensions: int, codebook: np.ndarray, *, k: int
) -> np.ndarray:
    width = 3 if k == 8 else 4
    index_bytes = (frames * width + 7) // 8
    mask_capacity = (frames * dimensions + 7) // 8
    minimum = 2 + index_bytes + 1
    if len(payload) < minimum:
        raise CarrierRepackError("truncated CVH VQ block")
    bits = struct.unpack_from("<H", payload)[0]
    indices = _cvq_unpack_fixed(payload[2 : 2 + index_bytes], frames, width)
    if np.any(indices >= k):
        raise CarrierRepackError("CVH VQ index exceeds codebook")
    offset = 2 + index_bytes
    correction_mode = int(payload[offset])
    offset += 1
    if correction_mode == 0:
        mask = np.ones(frames * dimensions, dtype=bool)
    elif correction_mode == 1:
        if len(payload) < offset + mask_capacity:
            raise CarrierRepackError("truncated CVH correction mask")
        mask_blob = payload[offset : offset + mask_capacity]
        offset += mask_capacity
        if frames * dimensions % 8 and mask_blob[-1] & (
            (1 << (8 - frames * dimensions % 8)) - 1
        ):
            raise CarrierRepackError("nonzero CVH correction-mask padding")
        mask = np.unpackbits(np.frombuffer(mask_blob, dtype=np.uint8), bitorder="big")[
            : frames * dimensions
        ].astype(bool)
    elif correction_mode == 2:
        if bits or offset != len(payload):
            raise CarrierRepackError(
                "noncanonical CVH exact-codeword correction stream"
            )
        mask = np.zeros(frames * dimensions, dtype=bool)
    else:
        raise CarrierRepackError("invalid CVH correction mode")
    if correction_mode == 2:
        values = np.empty(0, dtype=np.int32)
    else:
        if len(payload) <= offset:
            raise CarrierRepackError("truncated CVH Rice parameter")
        rice_k = int(payload[offset])
        offset += 1
        values = _cvq_rice_decode(rice_k, payload[offset:], bits, int(mask.sum()))
    residuals = codebook[indices]
    residuals.reshape(-1)[mask] += _cvq_unzigzag(values)
    if np.any(residuals < -2048) or np.any(residuals > 2047):
        raise CarrierRepackError("CVH VQ residual is outside signed int12")
    return residuals


def _cvh_restore_piecewise(block: np.ndarray, output: np.ndarray, start: int) -> None:
    for local in range(block.shape[0]):
        frame = start + local
        if frame == 0:
            prediction = np.zeros(output.shape[1], dtype=np.int32)
        elif frame == 1:
            prediction = output[frame - 1]
        else:
            delta = _signed_mod(output[frame - 1] - output[frame - 2])
            second = _signed_mod(output[frame - 1] + delta)
            prediction = np.where(np.abs(delta) <= 16, second, output[frame - 1])
        output[frame] = _signed_mod(prediction + block[local])


def _cvh_restore_ar1(
    block: np.ndarray, output: np.ndarray, start: int, factors: np.ndarray
) -> None:
    for local in range(block.shape[0]):
        frame = start + local
        prediction = (
            np.zeros(output.shape[1], dtype=np.int32)
            if frame == 0
            else _cvq_round_q8(output[frame - 1], factors)
        )
        output[frame] = _signed_mod(prediction + block[local])


def _cvh_header(
    blob: bytes, *, frames: int, dimensions: int
) -> tuple[int, int, int, int, int]:
    if len(blob) < _CVH1_HEADER.size:
        raise CarrierRepackError("truncated CVH1 header")
    (
        magic,
        version,
        k,
        stored_dimensions,
        block_frames,
        stored_frames,
        basis_raw,
        code_width,
    ) = _CVH1_HEADER.unpack_from(blob)
    basis_bits = _read_u24(basis_raw)
    if (
        magic != CVH1_MAGIC
        or version != _CVH1_VERSION
        or k not in (8, 16)
        or stored_dimensions != dimensions
        or stored_frames != frames
        or not block_frames
        or frames % block_frames
        or not 1 <= frames // block_frames <= 16
        or not basis_bits
        or code_width not in (1, 2)
    ):
        raise CarrierRepackError("invalid CVH1 header")
    return k, block_frames, basis_bits, code_width, _CVH1_HEADER.size


def inspect_cvh1(
    blob: bytes, *, frames: int, dimensions: int
) -> dict[str, int | list[int]]:
    """Inspect strict, fully charged block-hybrid CVH1 framing."""
    k, block_frames, basis_bits, code_width, offset = _cvh_header(
        blob, frames=frames, dimensions=dimensions
    )
    block_count = frames // block_frames
    basis_bytes = (basis_bits + 7) // 8
    codebook_bytes = (k - 1) * dimensions * code_width
    mask_bytes = (block_count + 7) // 8
    minimum = (
        offset
        + 2 * dimensions
        + 8 * dimensions
        + 32
        + basis_bytes
        + codebook_bytes
        + mask_bytes
        + 2 * block_count
    )
    if len(blob) < minimum:
        raise CarrierRepackError("truncated CVH1 metadata")
    cursor = (
        offset + 2 * dimensions + 8 * dimensions + 32 + basis_bytes + codebook_bytes
    )
    mode_mask = blob[cursor : cursor + mask_bytes]
    cursor += mask_bytes
    if block_count % 8 and mode_mask[-1] & ((1 << (8 - block_count % 8)) - 1):
        raise CarrierRepackError("nonzero CVH block-mode padding")
    lengths = np.frombuffer(
        blob[cursor : cursor + 2 * block_count], dtype="<u2"
    ).astype(np.int32)
    cursor += 2 * block_count
    if np.any(lengths == 0) or len(blob) != cursor + int(lengths.sum()):
        raise CarrierRepackError("truncated or trailing CVH block payloads")
    selected = [
        index
        for index in range(block_count)
        if mode_mask[index // 8] & (1 << (index % 8))
    ]
    return {
        "k": k,
        "block_frames": block_frames,
        "block_count": block_count,
        "vq_blocks": selected,
        "vq_block_count": len(selected),
        "header_bytes": _CVH1_HEADER.size,
        "predictor_bytes": 2 * dimensions,
        "frozen_metadata_bytes": 8 * dimensions + 32 + basis_bytes,
        "codebook_bytes": codebook_bytes,
        "mode_mask_bytes": mask_bytes,
        "block_length_table_bytes": 2 * block_count,
        "block_payload_bytes": int(lengths.sum()),
        "raw_carrier_bytes": len(blob),
    }


def probe_cvh1(
    raw_cpr1: bytes,
    *,
    frames: int,
    dimensions: int,
    k: int,
    block_frames: int = 75,
    iterations: int = 16,
) -> dict[str, int | list[int]]:
    """Measure one fixed-block Rice/AR+VQ decision without emitting a carrier."""
    if (
        k not in (8, 16)
        or not 1 <= block_frames <= 255
        or frames % block_frames
        or not 1 <= frames // block_frames <= 16
        or not 1 <= iterations <= 64
    ):
        raise CarrierRepackError("unsupported CVH1 probe configuration")
    basis_bits, _, _fields = _parse_cpr1(raw_cpr1, dimensions=dimensions)
    coefficients = decode_cpr1_coefficients(
        raw_cpr1, frames=frames, dimensions=dimensions
    )
    rice_residuals = _predict_residuals(
        coefficients, np.zeros(dimensions, dtype=np.uint8), 16
    )
    factors = _cvq_ar_factors(coefficients)
    vq_residuals = _cvq_residuals(coefficients, _CVQ1_AR1, factors)
    codebook, assignments = _cvq_codebook(
        vq_residuals, k, iterations, implicit_zero=True
    )
    block_count = frames // block_frames
    rice_sizes: list[int] = []
    vq_sizes: list[int] = []
    for block in range(block_count):
        start, end = block * block_frames, (block + 1) * block_frames
        rice_sizes.append(len(_cvh_rice_payload(rice_residuals[start:end])))
        vq_sizes.append(
            len(
                _cvh_vq_payload(
                    vq_residuals[start:end], codebook, assignments[start:end], k=k
                )
            )
        )
    selected = [
        index
        for index, (rice, vq) in enumerate(zip(rice_sizes, vq_sizes, strict=True))
        if vq < rice
    ]
    code_width = (
        1 if int(codebook[1:].min()) >= -128 and int(codebook[1:].max()) <= 127 else 2
    )
    common = (
        _CVH1_HEADER.size + 2 * dimensions + 8 * dimensions + 32 + (basis_bits + 7) // 8
    )
    table = (block_count + 7) // 8 + 2 * block_count
    codebook_bytes = (k - 1) * dimensions * code_width
    return {
        "k": k,
        "block_frames": block_frames,
        "block_count": block_count,
        "rice_block_bytes": rice_sizes,
        "vq_block_bytes": vq_sizes,
        "profitable_vq_blocks": selected,
        "positive_raw_block_savings_bytes": sum(
            max(rice - vq, 0) for rice, vq in zip(rice_sizes, vq_sizes, strict=True)
        ),
        "shared_vq_metadata_bytes": 2 * dimensions + codebook_bytes,
        "all_rice_hybrid_bytes": common + table + codebook_bytes + sum(rice_sizes),
        "selected_hybrid_bytes": common
        + table
        + codebook_bytes
        + sum(
            vq if index in selected else rice
            for index, (rice, vq) in enumerate(zip(rice_sizes, vq_sizes, strict=True))
        ),
    }


def encode_cvh1(
    raw_cpr1: bytes,
    *,
    frames: int,
    dimensions: int,
    k: int,
    block_frames: int = 75,
    iterations: int = 16,
) -> bytes:
    """Exact hybrid: each fixed block selects CPS3-style Rice or AR1+VQ.

    The Rice path uses piecewise-linear threshold-16 residuals, while the VQ
    path uses signed Q8 AR(1) residuals.  Both reconstruct the same int12
    coefficient matrix exactly, so a selected VQ block replaces rather than
    accompanies its CPS3/Rice representation.
    """
    if (
        k not in (8, 16)
        or not 1 <= block_frames <= 255
        or frames % block_frames
        or not 1 <= frames // block_frames <= 16
        or not 1 <= iterations <= 64
    ):
        raise CarrierRepackError("unsupported CVH1 encoder configuration")
    basis_bits, _, fields = _parse_cpr1(raw_cpr1, dimensions=dimensions)
    coefficients = decode_cpr1_coefficients(
        raw_cpr1, frames=frames, dimensions=dimensions
    )
    rice_residuals = _predict_residuals(
        coefficients, np.zeros(dimensions, dtype=np.uint8), 16
    )
    factors = _cvq_ar_factors(coefficients)
    vq_residuals = _cvq_residuals(coefficients, _CVQ1_AR1, factors)
    block_count = frames // block_frames
    codebook, assignments = _cvq_codebook(
        vq_residuals, k, iterations, implicit_zero=True
    )

    def choose(
        current_codebook: np.ndarray, current_assignments: np.ndarray
    ) -> tuple[list[bool], list[bytes], list[int]]:
        modes: list[bool] = []
        blocks: list[bytes] = []
        deltas: list[int] = []
        for block in range(block_count):
            start, end = block * block_frames, (block + 1) * block_frames
            rice = _cvh_rice_payload(rice_residuals[start:end])
            vq = _cvh_vq_payload(
                vq_residuals[start:end],
                current_codebook,
                current_assignments[start:end],
                k=k,
            )
            use_vq = len(vq) < len(rice)
            modes.append(use_vq)
            blocks.append(vq if use_vq else rice)
            deltas.append(len(rice) - len(vq))
        return modes, blocks, deltas

    modes, blocks, _deltas = choose(codebook, assignments)
    # One bounded refit focuses the shared codebook on blocks that actually
    # selected VQ; this is deliberately not a convergence loop.
    if any(modes):
        selected = np.concatenate(
            [
                vq_residuals[index * block_frames : (index + 1) * block_frames]
                for index, enabled in enumerate(modes)
                if enabled
            ]
        )
        codebook, _ = _cvq_codebook(selected, k, iterations, implicit_zero=True)
        differences = vq_residuals[:, None, :].astype(np.int64) - codebook[
            None, :, :
        ].astype(np.int64)
        assignments = np.argmin((differences * differences).sum(axis=2), axis=1).astype(
            np.uint8
        )
        modes, blocks, _deltas = choose(codebook, assignments)
    if not any(modes):
        raise CarrierRepackError("CVH1 selected no VQ block; retain CPS3 instead")
    code_width = (
        1 if int(codebook[1:].min()) >= -128 and int(codebook[1:].max()) <= 127 else 2
    )
    header = _CVH1_HEADER.pack(
        CVH1_MAGIC,
        _CVH1_VERSION,
        k,
        dimensions,
        block_frames,
        frames,
        _u24(basis_bits),
        code_width,
    )
    mode_mask = bytearray((block_count + 7) // 8)
    for index, enabled in enumerate(modes):
        if enabled:
            mode_mask[index // 8] |= 1 << (index % 8)
    if any(len(block) >= 1 << 16 for block in blocks):
        raise CarrierRepackError("CVH block payload exceeds u16")
    output = (
        header
        + factors.astype("<i2", copy=False).tobytes()
        + fields["scales"]
        + fields["lengths"]
        + fields["basis"]
        + codebook[1:].astype("i1" if code_width == 1 else "<i2", copy=False).tobytes()
        + bytes(mode_mask)
        + np.asarray([len(block) for block in blocks], dtype="<u2").tobytes()
        + b"".join(blocks)
    )
    if decode_cvh1(output, frames=frames, dimensions=dimensions) != raw_cpr1:
        raise CarrierRepackError("CVH1 encoder failed exact CPR1 reconstruction")
    return output


def decode_cvh1(blob: bytes, *, frames: int, dimensions: int) -> bytes:
    """Strictly restore canonical CPR1 from an exact block-hybrid CVH1 stream."""
    k, block_frames, basis_bits, code_width, offset = _cvh_header(
        blob, frames=frames, dimensions=dimensions
    )
    block_count = frames // block_frames
    basis_bytes = (basis_bits + 7) // 8
    codebook_bytes = (k - 1) * dimensions * code_width
    mask_bytes = (block_count + 7) // 8
    minimum = (
        offset
        + 2 * dimensions
        + 8 * dimensions
        + 32
        + basis_bytes
        + codebook_bytes
        + mask_bytes
        + 2 * block_count
    )
    if len(blob) < minimum:
        raise CarrierRepackError("truncated CVH1 metadata")
    factors = np.frombuffer(blob[offset : offset + 2 * dimensions], dtype="<i2").astype(
        np.int16
    )
    offset += 2 * dimensions
    scales = blob[offset : offset + 8 * dimensions]
    offset += 8 * dimensions
    lengths = blob[offset : offset + 32]
    offset += 32
    basis = blob[offset : offset + basis_bytes]
    offset += basis_bytes
    stored = (
        np.frombuffer(
            blob[offset : offset + codebook_bytes],
            dtype="i1" if code_width == 1 else "<i2",
        )
        .astype(np.int32)
        .reshape(k - 1, dimensions)
    )
    offset += codebook_bytes
    codebook = np.vstack((np.zeros((1, dimensions), dtype=np.int32), stored))
    mode_mask = blob[offset : offset + mask_bytes]
    offset += mask_bytes
    if block_count % 8 and mode_mask[-1] & ((1 << (8 - block_count % 8)) - 1):
        raise CarrierRepackError("nonzero CVH block-mode padding")
    lengths_table = np.frombuffer(
        blob[offset : offset + 2 * block_count], dtype="<u2"
    ).astype(np.int32)
    offset += 2 * block_count
    if np.any(lengths_table == 0) or len(blob) != offset + int(lengths_table.sum()):
        raise CarrierRepackError("truncated or trailing CVH block payloads")
    coefficients = np.empty((frames, dimensions), dtype=np.int32)
    for block, size in enumerate(lengths_table):
        payload = blob[offset : offset + int(size)]
        offset += int(size)
        start = block * block_frames
        if mode_mask[block // 8] & (1 << (block % 8)):
            _cvh_restore_ar1(
                _cvh_vq_decode(payload, block_frames, dimensions, codebook, k=k),
                coefficients,
                start,
                factors,
            )
        else:
            _cvh_restore_piecewise(
                _cvh_rice_decode(payload, block_frames, dimensions), coefficients, start
            )
    original = _zigzag(
        _predict_residuals(coefficients, np.zeros(dimensions, dtype=np.uint8), 0)
    )
    original_ks, original_payload, original_bits = _rice_encode(original, 1)
    return (
        CPR1_MAGIC
        + struct.pack("<II", basis_bits, original_bits)
        + scales
        + lengths
        + original_ks.reshape(-1).tobytes()
        + basis
        + original_payload
    )


def strip_fixed_cps3(blob: bytes, *, frames: int, dimensions: int) -> bytes:
    """Remove only schema-derived fields from a fixed CPS3 carrier payload.

    The production schema pins version 1, the piecewise predictor, one Rice
    segment, and threshold 16. The learned/data-dependent basis bit count and
    last-byte valid-bit count remain stored and charged.
    """
    if len(blob) < 14 or blob[:4] != CPS3_MAGIC:
        raise CarrierRepackError("invalid CPS3 carrier magic or length")
    version, mode, segments, threshold = blob[4:8]
    if (version, mode, segments, threshold) != (_CPS3_VERSION, _CPS3_PIECEWISE, 1, 16):
        raise CarrierRepackError(
            "fixed CPS3 schema requires version-1 piecewise threshold-16 carrier"
        )
    basis_bits, coefficient_bits = _read_u24(blob[8:11]), _read_u24(blob[11:14])
    if not basis_bits or not coefficient_bits:
        raise CarrierRepackError("fixed CPS3 schema requires nonempty bitstreams")
    # Reject malformed Rice payloads and non-canonical padding before packing.
    decode_cps3(blob, frames=frames, dimensions=dimensions)
    return blob[8:11] + bytes((coefficient_bits % 8 or 8,)) + blob[14:]


def restore_fixed_cps3(blob: bytes, *, frames: int, dimensions: int) -> bytes:
    """Restore the exact CPS3 payload from the compact fixed-schema form."""
    minimum = FIXED_CPS3_PREFIX_BYTES + 8 * dimensions + 32 + dimensions
    if len(blob) < minimum:
        raise CarrierRepackError("truncated fixed CPS3 payload")
    basis_bits = _read_u24(blob[:3])
    valid_bits = int(blob[3])
    if not basis_bits or not 1 <= valid_bits <= 8:
        raise CarrierRepackError("invalid fixed CPS3 bit metadata")
    basis_bytes = (basis_bits + 7) // 8
    fixed_fields = (
        FIXED_CPS3_PREFIX_BYTES + 8 * dimensions + 32 + dimensions + basis_bytes
    )
    if len(blob) <= fixed_fields:
        raise CarrierRepackError("truncated fixed CPS3 coefficient payload")
    coefficient_bytes = len(blob) - fixed_fields
    coefficient_bits = (coefficient_bytes - 1) * 8 + valid_bits
    restored = (
        CPS3_MAGIC
        + bytes((_CPS3_VERSION, _CPS3_PIECEWISE, 1, 16))
        + blob[:3]
        + _u24(coefficient_bits)
        + blob[4:]
    )
    # This checks Rice code boundaries, payload length, and padding bits.
    decode_cps3(restored, frames=frames, dimensions=dimensions)
    return restored
