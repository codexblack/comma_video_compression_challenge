"""Causal, table-only residual calibration for frozen IntegerHPAC logits."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .bits import BitPackingError, pack_signed, packed_length, unpack_signed

NUM_CLASSES = 5
RESIDUAL_MAGIC = b"RCL1"
RESIDUAL_VERSION = 1

# These identifiers are part of the on-wire format.  A decoder is therefore
# never required to infer a table's conditioning feature from external state.
FEATURE_IDS = {
    "global_bias": 1,
    "predicted_margin": 2,
    "previous_predicted": 3,
    "boundary_predicted": 4,
    "row_predicted": 5,
    "segment_predicted": 6,
    "temperature_log": 7,
}
FEATURE_NAMES = {value: key for key, value in FEATURE_IDS.items()}
MODE_IDS = {None: 0, "temperature": 1}
MODE_NAMES = {value: key for key, value in MODE_IDS.items()}
PRECISION_IDS = {4: 0, 6: 1, 8: 2}
PRECISION_BITS = {value: key for key, value in PRECISION_IDS.items()}


@dataclass
class FeatureStatistics:
    states: int
    target: np.ndarray
    expected: np.ndarray


@dataclass(frozen=True)
class QuantizedTable:
    name: str
    bits: int
    codes: np.ndarray
    scale: float
    values: np.ndarray

    @property
    def storage_bytes(self) -> int:
        # One field descriptor (bits, state count, fp16 scale) plus exact
        # signed packed codes. The candidate header is charged once below.
        return 1 + 2 + 2 + len(pack_signed(self.codes.reshape(-1), self.bits))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            pack_signed(self.codes.reshape(-1), self.bits)
            + np.asarray([self.scale], dtype="<f2").tobytes()
        ).hexdigest()


def new_statistics(states: int) -> FeatureStatistics:
    return FeatureStatistics(
        states,
        np.zeros((states, NUM_CLASSES), dtype=np.float64),
        np.zeros((states, NUM_CLASSES), dtype=np.float64),
    )


def update_statistics(
    statistics: FeatureStatistics,
    context: np.ndarray,
    symbols: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    context = np.asarray(context, dtype=np.int64).reshape(-1)
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if context.size != symbols.size or probabilities.shape != (
        symbols.size,
        NUM_CLASSES,
    ):
        raise ValueError("residual statistic dimensions do not match")
    if (
        np.any(context < 0)
        or np.any(context >= statistics.states)
        or np.any(symbols < 0)
        or np.any(symbols >= NUM_CLASSES)
    ):
        raise ValueError("residual statistic context or class out of range")
    statistics.target += np.bincount(
        context * NUM_CLASSES + symbols, minlength=statistics.states * NUM_CLASSES
    ).reshape(statistics.states, NUM_CLASSES)
    for klass in range(NUM_CLASSES):
        statistics.expected[:, klass] += np.bincount(
            context, weights=probabilities[:, klass], minlength=statistics.states
        )


def fit_log_ratio(statistics: FeatureStatistics, smoothing: float = 0.5) -> np.ndarray:
    values = np.log((statistics.target + smoothing) / (statistics.expected + smoothing))
    return values - values.mean(axis=1, keepdims=True)


def quantize_table(name: str, values: np.ndarray, bits: int) -> QuantizedTable:
    if bits not in (4, 6, 8):
        raise ValueError("Phase-2C residual tables support int4/int6/int8")
    limit = (1 << (bits - 1)) - 1
    scale = float(np.max(np.abs(values)) / limit) if np.any(values) else 1.0
    # Store the exact deployed scalar representation, then regenerate values
    # from it so evaluation charges what a decoder would use.
    scale = float(np.asarray([scale], dtype="<f2")[0])
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    codes = np.clip(np.rint(values / scale), -limit, limit).astype(np.int8)
    return QuantizedTable(name, bits, codes, scale, codes.astype(np.float32) * scale)


def margin_buckets(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError("expected Nx5 HPAC logits")
    ordered = np.partition(logits, -2, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return np.digitize(margin, np.asarray((0.25, 0.75, 1.50), dtype=np.float32)).astype(
        np.int64
    )


def contexts(
    predicted: np.ndarray,
    margin: np.ndarray,
    previous: np.ndarray,
    boundary: np.ndarray,
    rows: np.ndarray,
    segment: int,
) -> dict[str, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.int64)
    return {
        "global_bias": np.zeros_like(predicted),
        "predicted_margin": predicted * 4 + np.asarray(margin, dtype=np.int64),
        "previous_predicted": np.asarray(previous, dtype=np.int64) * NUM_CLASSES
        + predicted,
        "boundary_predicted": np.asarray(boundary, dtype=np.int64) * NUM_CLASSES
        + predicted,
        "row_predicted": np.asarray(rows, dtype=np.int64) * NUM_CLASSES + predicted,
        "segment_predicted": int(segment) * NUM_CLASSES + predicted,
    }


def table_bytes(tables: tuple[QuantizedTable, ...]) -> int:
    return 8 + sum(table.storage_bytes for table in tables)


def serialize_tables(
    tables: tuple[QuantizedTable, ...], mode: str | None = None
) -> bytes:
    """Serialize the charged residual payload without implicit metadata."""
    if not tables or len(tables) > 255 or mode not in MODE_IDS:
        raise ValueError("invalid residual table collection")
    result = bytearray(
        RESIDUAL_MAGIC + bytes((RESIDUAL_VERSION, len(tables), MODE_IDS[mode], 0))
    )
    for table in tables:
        if (
            table.name not in FEATURE_IDS
            or table.codes.ndim != 2
            or table.codes.shape[1] != NUM_CLASSES
        ):
            raise ValueError("unsupported residual table")
        states = table.codes.shape[0]
        if not 1 <= states <= 65535:
            raise ValueError("residual table state count out of range")
        result.append((FEATURE_IDS[table.name] << 2) | PRECISION_IDS[table.bits])
        result.extend(int(states).to_bytes(2, "little"))
        result.extend(np.asarray([table.scale], dtype="<f2").tobytes())
        result.extend(pack_signed(table.codes.reshape(-1), table.bits))
    if len(result) != table_bytes(tables):
        raise AssertionError("residual serialization byte accounting diverged")
    return bytes(result)


def deserialize_tables(blob: bytes) -> tuple[tuple[QuantizedTable, ...], str | None]:
    """Parse a complete RCL1 payload and reject truncation or trailing data."""
    if len(blob) < 8 or blob[:4] != RESIDUAL_MAGIC:
        raise BitPackingError("invalid residual table magic")
    version, count, mode_id, reserved = blob[4:8]
    if (
        version != RESIDUAL_VERSION
        or not count
        or mode_id not in MODE_NAMES
        or reserved
    ):
        raise BitPackingError("invalid residual table header")
    offset = 8
    tables = []
    for _ in range(count):
        if offset + 5 > len(blob):
            raise BitPackingError("truncated residual table descriptor")
        descriptor = blob[offset]
        feature_id, precision_id = descriptor >> 2, descriptor & 0x03
        bits = PRECISION_BITS.get(precision_id)
        states = int.from_bytes(blob[offset + 1 : offset + 3], "little")
        scale = float(np.frombuffer(blob[offset + 3 : offset + 5], dtype="<f2")[0])
        offset += 5
        if (
            feature_id not in FEATURE_NAMES
            or bits is None
            or not states
            or not math.isfinite(scale)
            or scale <= 0.0
        ):
            raise BitPackingError("invalid residual table descriptor")
        count_codes = states * NUM_CLASSES
        size = packed_length(count_codes, bits)
        if offset + size > len(blob):
            raise BitPackingError("truncated residual table codes")
        codes = np.asarray(
            unpack_signed(blob[offset : offset + size], count_codes, bits),
            dtype=np.int8,
        ).reshape(states, NUM_CLASSES)
        offset += size
        tables.append(
            QuantizedTable(
                FEATURE_NAMES[feature_id],
                bits,
                codes,
                scale,
                codes.astype(np.float32) * scale,
            )
        )
    if offset != len(blob) or len({table.name for table in tables}) != len(tables):
        raise BitPackingError("residual table payload has trailing or duplicate data")
    return tuple(tables), MODE_NAMES[mode_id]


def rcl1_payload_length(blob: bytes) -> int:
    """Return the exact length of one RCL1 payload at the start of ``blob``.

    Production archives concatenate the self-describing generic RCL1 table
    with a range-coded token stream.  This scanner intentionally validates
    only enough syntax to locate the table boundary; ``deserialize_tables``
    remains the strict full parser for the extracted slice.
    """
    if len(blob) < 8 or blob[:4] != RESIDUAL_MAGIC:
        raise BitPackingError("invalid residual table magic")
    version, count, mode_id, reserved = blob[4:8]
    if (
        version != RESIDUAL_VERSION
        or not count
        or mode_id not in MODE_NAMES
        or reserved
    ):
        raise BitPackingError("invalid residual table header")
    offset = 8
    seen: set[int] = set()
    for _ in range(count):
        if offset + 5 > len(blob):
            raise BitPackingError("truncated residual table descriptor")
        descriptor = blob[offset]
        feature_id, precision_id = descriptor >> 2, descriptor & 0x03
        bits = PRECISION_BITS.get(precision_id)
        states = int.from_bytes(blob[offset + 1 : offset + 3], "little")
        scale = float(np.frombuffer(blob[offset + 3 : offset + 5], dtype="<f2")[0])
        offset += 5
        if (
            feature_id not in FEATURE_NAMES
            or feature_id in seen
            or bits is None
            or not states
            or not math.isfinite(scale)
            or scale <= 0.0
        ):
            raise BitPackingError("invalid residual table descriptor")
        seen.add(feature_id)
        code_bytes = packed_length(states * NUM_CLASSES, bits)
        if offset + code_bytes > len(blob):
            raise BitPackingError("truncated residual table codes")
        offset += code_bytes
    return offset
