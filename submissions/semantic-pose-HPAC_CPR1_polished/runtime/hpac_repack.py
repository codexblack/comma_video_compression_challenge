"""Lossless representation-only repacking for deployed IntegerHPAC bytes.

The deployed CPR1 HPAC blob is a fixed sequence of masked int8 weights,
int16 biases, and int8 exponents.  This module never changes an integer value;
it merely applies reversible ordering and packing transforms before the archive
container's existing LZMA stage.
"""

from __future__ import annotations

import heapq
import struct
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

MAGIC = b"HPR1"
VERSION = 1
SELF_MAGIC = b"IHS1"
SELF_REPACK_MAGIC = b"HPS2"
FRAME_RAW = 0
FRAME_DIM_MAJOR = 1
FRAME_DELTA = 2
FRAME_SECOND = 3
FRAME_DIM_DELTA = 4
FRAME_RICE = 5
FRAME_HUFFMAN = 6
FRAME_RANGE = 7
FRAME_NAMES = {
    FRAME_RAW: "raw_frame_major",
    FRAME_DIM_MAJOR: "raw_dimension_major",
    FRAME_DELTA: "first_order_temporal_delta",
    FRAME_SECOND: "second_order_temporal_prediction",
    FRAME_DIM_DELTA: "per_dimension_delta",
    FRAME_RICE: "per_dimension_delta_zigzag_rice",
    FRAME_HUFFMAN: "canonical_huffman_delta",
    FRAME_RANGE: "range_coded_delta",
}


class RepackError(ValueError):
    """A repacked HPAC stream is malformed or not exactly recoverable."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    count: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize

    @property
    def byte_count(self) -> int:
        return self.count * self.itemsize

    @property
    def is_bias(self) -> bool:
        return self.name.endswith(".bias")

    @property
    def is_exponent(self) -> bool:
        return self.name.endswith(".exponent")


@dataclass(frozen=True)
class IntegerHpacLayout:
    fields: tuple[FieldSpec, ...]

    @property
    def raw_bytes(self) -> int:
        return sum(field.byte_count for field in self.fields)

    @property
    def bias_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(field for field in self.fields if field.is_bias)

    @property
    def frame_field(self) -> FieldSpec:
        matches = [field for field in self.fields if field.name == "frame_embed.weight"]
        if len(matches) != 1:
            raise RepackError("expected one frame embedding field")
        return matches[0]


@dataclass(frozen=True)
class SelfCompressedHpacLayout:
    """Layout of the deployed IHS1 integer-model representation."""

    depth_count: int
    weight_bytes: int
    tail_fields: tuple[FieldSpec, ...]
    module_ranges: tuple[tuple[int, int], ...]

    @property
    def depth_bytes(self) -> int:
        return (self.depth_count + 1) // 2

    @property
    def raw_bytes(self) -> int:
        return (
            4
            + self.depth_bytes
            + self.weight_bytes
            + sum(field.byte_count for field in self.tail_fields)
        )

    @property
    def bias_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(field for field in self.tail_fields if field.is_bias)


def _pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    if not 1 <= bits <= 16:
        raise RepackError("invalid unsigned bit width")
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(values < 0) or np.any(values >= (1 << bits)):
        raise RepackError("unsigned packing range violation")
    output = np.zeros((values.size * bits + 7) // 8, dtype=np.uint8)
    for index, value in enumerate(values):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = int(value) << shift
        output[byte] |= word & 0xFF
        if shift + bits > 8:
            output[byte + 1] |= (word >> 8) & 0xFF
        if shift + bits > 16:
            output[byte + 2] |= (word >> 16) & 0xFF
    return output.tobytes()


def _unpack_unsigned(raw: bytes, count: int, bits: int) -> np.ndarray:
    expected = (count * bits + 7) // 8
    if len(raw) != expected:
        raise RepackError("truncated packed integer field")
    source = np.frombuffer(raw, dtype=np.uint8)
    values = np.empty(count, dtype=np.int64)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = int(source[byte])
        if byte + 1 < len(source):
            word |= int(source[byte + 1]) << 8
        if byte + 2 < len(source):
            word |= int(source[byte + 2]) << 16
        values[index] = (word >> shift) & ((1 << bits) - 1)
    return values


def _signed_width(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.int64)
    for bits in range(1, 17):
        if values.min(initial=0) >= -(1 << (bits - 1)) and values.max(initial=0) < (
            1 << (bits - 1)
        ):
            return bits
    raise RepackError("int16 bias does not fit a signed width")


def _pack_signed(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < -(1 << (bits - 1))) or np.any(values >= (1 << (bits - 1))):
        raise RepackError("signed packing range violation")
    return _pack_unsigned(np.where(values < 0, values + (1 << bits), values), bits)


def _unpack_signed(raw: bytes, count: int, bits: int) -> np.ndarray:
    values = _unpack_unsigned(raw, count, bits)
    sign = 1 << (bits - 1)
    return np.where(values >= sign, values - (1 << bits), values).astype(np.int16)


def _unpack_nibbles(raw: bytes, count: int) -> np.ndarray:
    if len(raw) != (count + 1) // 2:
        raise RepackError("truncated row-depth metadata")
    packed = np.frombuffer(raw, dtype=np.uint8)
    output = np.empty(len(packed) * 2, dtype=np.uint8)
    output[0::2] = packed & 15
    output[1::2] = packed >> 4
    return output[:count]


def _pack_nibbles(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(values > 15):
        raise RepackError("row depth does not fit four bits")
    padded = np.pad(values, (0, len(values) % 2), constant_values=0)
    return (padded[0::2] | (padded[1::2] << 4)).tobytes()


def _frame_matrix(raw: bytes) -> np.ndarray:
    if len(raw) != 600 * 8:
        raise RepackError("expected a 600x8 int8 frame embedding")
    return np.frombuffer(raw, dtype=np.uint8).copy().reshape(600, 8)


def _delta_forward(values: np.ndarray, second: bool = False) -> np.ndarray:
    output = values.copy()
    if second:
        output[2:] = (
            values[2:].astype(np.int16)
            - 2 * values[1:-1].astype(np.int16)
            + values[:-2].astype(np.int16)
        ) % 256
    else:
        output[1:] = (values[1:].astype(np.int16) - values[:-1].astype(np.int16)) % 256
    return output.astype(np.uint8)


def _delta_inverse(values: np.ndarray, second: bool = False) -> np.ndarray:
    output = values.copy()
    if second:
        for index in range(2, len(output)):
            output[index] = (
                values[index].astype(np.int16)
                + 2 * output[index - 1].astype(np.int16)
                - output[index - 2].astype(np.int16)
            ) % 256
    else:
        for index in range(1, len(output)):
            output[index] = (
                values[index].astype(np.int16) + output[index - 1].astype(np.int16)
            ) % 256
    return output


def _rice_bits(values: np.ndarray, parameter: int) -> int:
    unsigned = np.where(values >= 0, 2 * values, -2 * values - 1)
    return int(np.sum((unsigned >> parameter) + 1 + parameter))


def _write_bits(values: Iterable[tuple[int, int]]) -> bytes:
    out = bytearray()
    accumulator = 0
    used = 0
    for value, bits in values:
        accumulator |= int(value) << used
        used += bits
        while used >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            used -= 8
    if used:
        out.append(accumulator & 0xFF)
    return bytes(out)


class _BitReader:
    def __init__(self, raw: bytes):
        self.raw, self.offset, self.accumulator, self.available = raw, 0, 0, 0

    def read(self, bits: int) -> int:
        while self.available < bits:
            if self.offset >= len(self.raw):
                raise RepackError("truncated entropy-coded frame embedding")
            self.accumulator |= self.raw[self.offset] << self.available
            self.available += 8
            self.offset += 1
        value = self.accumulator & ((1 << bits) - 1)
        self.accumulator >>= bits
        self.available -= bits
        return value


def _encode_rice(matrix: np.ndarray) -> bytes:
    signed = matrix.view(np.int8).astype(np.int16)
    first = signed[0].astype(np.int8).tobytes()
    parameters = []
    pairs: list[tuple[int, int]] = []
    for dimension in range(8):
        delta = np.diff(signed[:, dimension]).astype(np.int64)
        parameter = min(range(8), key=lambda candidate: _rice_bits(delta, candidate))
        parameters.append(parameter)
        for value in delta:
            unsigned = int(2 * value if value >= 0 else -2 * value - 1)
            quotient, remainder = divmod(unsigned, 1 << parameter)
            pairs.extend(((0, 1),) * quotient)
            pairs.append((1, 1))
            if parameter:
                pairs.append((remainder, parameter))
    bits = _write_bits(pairs)
    return first + bytes(parameters) + struct.pack("<I", len(bits)) + bits


def _decode_rice(raw: bytes) -> np.ndarray:
    if len(raw) < 20:
        raise RepackError("truncated Rice frame embedding")
    first = np.frombuffer(raw[:8], dtype=np.int8).astype(np.int16)
    parameters = raw[8:16]
    size = struct.unpack_from("<I", raw, 16)[0]
    if len(raw) != 20 + size or any(parameter > 7 for parameter in parameters):
        raise RepackError("invalid Rice frame embedding")
    reader = _BitReader(raw[20:])
    output = np.empty((600, 8), dtype=np.int16)
    output[0] = first
    for dimension, parameter in enumerate(parameters):
        for frame in range(1, 600):
            quotient = 0
            while reader.read(1) == 0:
                quotient += 1
            remainder = reader.read(parameter) if parameter else 0
            unsigned = (quotient << parameter) | remainder
            delta = unsigned // 2 if not (unsigned & 1) else -(unsigned // 2) - 1
            output[frame, dimension] = output[frame - 1, dimension] + delta
    if np.any(output < -128) or np.any(output > 127):
        raise RepackError("Rice frame embedding reconstructs outside int8")
    return output.astype(np.int8).view(np.uint8)


def _huffman_lengths(frequencies: np.ndarray) -> np.ndarray:
    queue = [(int(value), index) for index, value in enumerate(frequencies) if value]
    if not queue:
        raise RepackError("empty Huffman alphabet")
    if len(queue) == 1:
        result = np.zeros(256, dtype=np.uint8)
        result[queue[0][1]] = 1
        return result
    heapq.heapify(queue)
    parent: dict[int, tuple[int, int]] = {}
    next_node = 256
    while len(queue) > 1:
        left = heapq.heappop(queue)
        right = heapq.heappop(queue)
        parent[left[1]] = (next_node, 0)
        parent[right[1]] = (next_node, 1)
        heapq.heappush(queue, (left[0] + right[0], next_node))
        next_node += 1
    lengths = np.zeros(256, dtype=np.uint8)
    for symbol in np.flatnonzero(frequencies):
        node, length = int(symbol), 0
        while node in parent:
            node = parent[node][0]
            length += 1
        lengths[symbol] = length
    return lengths


def _canonical_codes(lengths: np.ndarray) -> dict[int, tuple[int, int]]:
    code = 0
    previous = 0
    result: dict[int, tuple[int, int]] = {}
    for length, symbol in sorted(
        (int(length), index) for index, length in enumerate(lengths) if length
    ):
        code <<= length - previous
        result[symbol] = (code, length)
        code += 1
        previous = length
    return result


def _reverse_bits(value: int, width: int) -> int:
    output = 0
    for _ in range(width):
        output = (output << 1) | (value & 1)
        value >>= 1
    return output


def _encode_huffman(matrix: np.ndarray) -> bytes:
    signed = matrix.view(np.int8).astype(np.int16)
    delta = np.diff(signed, axis=0).reshape(-1)
    symbols = (delta + 128).astype(np.uint8)
    lengths = _huffman_lengths(np.bincount(symbols, minlength=256))
    codes = _canonical_codes(lengths)
    bits = _write_bits(
        (
            _reverse_bits(codes[int(symbol)][0], codes[int(symbol)][1]),
            codes[int(symbol)][1],
        )
        for symbol in symbols
    )
    return (
        signed[0].astype(np.int8).tobytes()
        + lengths.tobytes()
        + struct.pack("<I", len(bits))
        + bits
    )


def _decode_huffman(raw: bytes) -> np.ndarray:
    if len(raw) < 268:
        raise RepackError("truncated Huffman frame embedding")
    first = np.frombuffer(raw[:8], dtype=np.int8).astype(np.int16)
    lengths = np.frombuffer(raw[8:264], dtype=np.uint8)
    size = struct.unpack_from("<I", raw, 264)[0]
    if len(raw) != 268 + size:
        raise RepackError("invalid Huffman frame embedding")
    codes = _canonical_codes(lengths)
    decode = {(code, length): symbol for symbol, (code, length) in codes.items()}
    reader = _BitReader(raw[268:])
    symbols = []
    code = 0
    for _ in range(599 * 8):
        for length in range(1, int(lengths.max()) + 1):
            code = (code << 1) | reader.read(1)
            symbol = decode.get((code, length))
            if symbol is not None:
                symbols.append(symbol - 128)
                code = 0
                break
        else:
            raise RepackError("invalid canonical Huffman code")
    delta = np.asarray(symbols, dtype=np.int16).reshape(599, 8)
    output = np.empty((600, 8), dtype=np.int16)
    output[0] = first
    for frame in range(1, 600):
        output[frame] = output[frame - 1] + delta[frame - 1]
    if np.any(output < -128) or np.any(output > 127):
        raise RepackError("Huffman frame embedding reconstructs outside int8")
    return output.astype(np.int8).view(np.uint8)


def _encode_range(matrix: np.ndarray) -> bytes:
    import constriction

    signed = matrix.view(np.int8).astype(np.int16)
    delta = np.diff(signed, axis=0).reshape(-1)
    symbols = (delta + 128).astype(np.int32)
    values, counts = np.unique(symbols, return_counts=True)
    table = (
        (counts / counts.sum()).astype(np.float32)[None].repeat(len(symbols), axis=0)
    )
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(
        np.searchsorted(values, symbols).astype(np.int32),
        constriction.stream.model.Categorical(perfect=False),
        table,
    )
    coded = encoder.get_compressed().tobytes()
    return (
        signed[0].astype(np.int8).tobytes()
        + struct.pack("<B", len(values))
        + values.astype(np.uint8).tobytes()
        + counts.astype("<u2").tobytes()
        + struct.pack("<I", len(coded))
        + coded
    )


def _decode_range(raw: bytes) -> np.ndarray:
    import constriction

    if len(raw) < 9:
        raise RepackError("truncated range-coded frame embedding")
    first = np.frombuffer(raw[:8], dtype=np.int8).astype(np.int16)
    count = raw[8]
    offset = 9
    if count == 0 or len(raw) < offset + count * 3 + 4:
        raise RepackError("invalid range-coded frame alphabet")
    values = np.frombuffer(raw[offset : offset + count], dtype=np.uint8).astype(
        np.int16
    )
    offset += count
    frequencies = np.frombuffer(raw[offset : offset + 2 * count], dtype="<u2")
    offset += 2 * count
    size = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    if offset + size != len(raw) or size % 4:
        raise RepackError("invalid range-coded frame payload")
    table = (
        (frequencies / frequencies.sum())
        .astype(np.float32)[None]
        .repeat(599 * 8, axis=0)
    )
    decoder = constriction.stream.queue.RangeDecoder(
        np.frombuffer(raw[offset:], dtype="<u4")
    )
    symbols = decoder.decode(
        constriction.stream.model.Categorical(perfect=False), table
    )
    delta = values[symbols].reshape(599, 8) - 128
    output = np.empty((600, 8), dtype=np.int16)
    output[0] = first
    for frame in range(1, 600):
        output[frame] = output[frame - 1] + delta[frame - 1]
    if np.any(output < -128) or np.any(output > 127):
        raise RepackError("range frame embedding reconstructs outside int8")
    return output.astype(np.int8).view(np.uint8)


def _encode_frame(raw: bytes, kind: int) -> bytes:
    matrix = _frame_matrix(raw)
    if kind == FRAME_RAW:
        return raw
    if kind == FRAME_DIM_MAJOR:
        return matrix.T.tobytes()
    if kind == FRAME_DELTA:
        return _delta_forward(matrix).tobytes()
    if kind == FRAME_SECOND:
        return _delta_forward(matrix, second=True).tobytes()
    if kind == FRAME_DIM_DELTA:
        return _delta_forward(matrix.T).tobytes()
    if kind == FRAME_RICE:
        return _encode_rice(matrix)
    if kind == FRAME_HUFFMAN:
        return _encode_huffman(matrix)
    if kind == FRAME_RANGE:
        return _encode_range(matrix)
    raise RepackError("unknown frame representation")


def _decode_frame(raw: bytes, kind: int) -> bytes:
    if kind == FRAME_RAW:
        if len(raw) != 4800:
            raise RepackError("invalid raw frame size")
        return raw
    if kind == FRAME_DIM_MAJOR:
        if len(raw) != 4800:
            raise RepackError("invalid dimension-major frame size")
        return np.frombuffer(raw, dtype=np.uint8).reshape(8, 600).T.copy().tobytes()
    if kind == FRAME_DELTA:
        return _delta_inverse(_frame_matrix(raw)).tobytes()
    if kind == FRAME_SECOND:
        return _delta_inverse(_frame_matrix(raw), second=True).tobytes()
    if kind == FRAME_DIM_DELTA:
        if len(raw) != 4800:
            raise RepackError("invalid dimension-delta frame size")
        return (
            _delta_inverse(np.frombuffer(raw, dtype=np.uint8).reshape(8, 600))
            .T.copy()
            .tobytes()
        )
    if kind == FRAME_RICE:
        return _decode_rice(raw).tobytes()
    if kind == FRAME_HUFFMAN:
        return _decode_huffman(raw).tobytes()
    if kind == FRAME_RANGE:
        return _decode_range(raw).tobytes()
    raise RepackError("unknown frame representation")


DEPTH_RAW = 0
DEPTH_MODULE_DEFAULT = 1
DEPTH_RLE = 2
DEPTH_DELTA = 3
DEPTH_HUFFMAN = 4
DEPTH_NAMES = {
    DEPTH_RAW: "current_4bit_depths",
    DEPTH_MODULE_DEFAULT: "module_default_plus_exceptions",
    DEPTH_RLE: "run_length_depths",
    DEPTH_DELTA: "adjacent_row_delta_depths",
    DEPTH_HUFFMAN: "canonical_huffman_depths",
}


def _encode_depths(
    values: np.ndarray, layout: SelfCompressedHpacLayout, kind: int
) -> bytes:
    if kind == DEPTH_RAW:
        return _pack_nibbles(values)
    if kind == DEPTH_MODULE_DEFAULT:
        defaults = []
        exceptions: list[tuple[int, int]] = []
        for start, end in layout.module_ranges:
            local = values[start:end]
            counts = np.bincount(local, minlength=16)
            default = int(counts.argmax())
            defaults.append(default)
            exceptions.extend(
                (start + index, int(value))
                for index, value in enumerate(local)
                if value != default
            )
        return (
            _pack_nibbles(np.asarray(defaults, dtype=np.uint8))
            + struct.pack("<H", len(exceptions))
            + b"".join(struct.pack("<HB", index, value) for index, value in exceptions)
        )
    if kind == DEPTH_RLE:
        runs: list[tuple[int, int]] = []
        start = 0
        while start < len(values):
            end = start + 1
            while (
                end < len(values)
                and values[end] == values[start]
                and end - start < 65535
            ):
                end += 1
            runs.append((end - start, int(values[start])))
            start = end
        return struct.pack("<H", len(runs)) + b"".join(
            struct.pack("<HB", length, value) for length, value in runs
        )
    if kind == DEPTH_DELTA:
        delta = np.diff(values.astype(np.int16))
        return bytes([int(values[0])]) + delta.astype(np.int8).tobytes()
    if kind == DEPTH_HUFFMAN:
        lengths = _huffman_lengths(np.bincount(values, minlength=256))
        codes = _canonical_codes(lengths)
        bits = _write_bits(
            (
                _reverse_bits(codes[int(value)][0], codes[int(value)][1]),
                codes[int(value)][1],
            )
            for value in values
        )
        return lengths.tobytes() + struct.pack("<I", len(bits)) + bits
    raise RepackError("unknown row-depth representation")


def _decode_depths(
    raw: bytes, layout: SelfCompressedHpacLayout, kind: int
) -> np.ndarray:
    count = layout.depth_count
    if kind == DEPTH_RAW:
        return _unpack_nibbles(raw, count)
    if kind == DEPTH_MODULE_DEFAULT:
        default_size = (len(layout.module_ranges) + 1) // 2
        if len(raw) < default_size + 2:
            raise RepackError("truncated depth defaults")
        values = np.empty(count, dtype=np.uint8)
        defaults = _unpack_nibbles(raw[:default_size], len(layout.module_ranges))
        exception_count = struct.unpack_from("<H", raw, default_size)[0]
        offset = default_size + 2
        for (start, end), default in zip(layout.module_ranges, defaults, strict=True):
            values[start:end] = default
        if len(raw) != offset + exception_count * 3:
            raise RepackError("invalid depth exceptions")
        for _ in range(exception_count):
            index, value = struct.unpack_from("<HB", raw, offset)
            offset += 3
            if index >= count or value > 15:
                raise RepackError("invalid depth exception")
            values[index] = value
        return values
    if kind == DEPTH_RLE:
        if len(raw) < 2:
            raise RepackError("truncated depth runs")
        run_count = struct.unpack_from("<H", raw)[0]
        if len(raw) != 2 + 3 * run_count:
            raise RepackError("invalid depth runs")
        values: list[int] = []
        offset = 2
        for _ in range(run_count):
            length, value = struct.unpack_from("<HB", raw, offset)
            offset += 3
            if not length or value > 15:
                raise RepackError("invalid depth run")
            values.extend([value] * length)
        if len(values) != count:
            raise RepackError("depth run count mismatch")
        return np.asarray(values, dtype=np.uint8)
    if kind == DEPTH_DELTA:
        if len(raw) != count:
            raise RepackError("invalid depth delta payload")
        values = np.empty(count, dtype=np.int16)
        values[0] = raw[0]
        values[1:] = np.frombuffer(raw[1:], dtype=np.int8)
        values = np.cumsum(values)
        if np.any(values < 0) or np.any(values > 15):
            raise RepackError("depth delta escapes nibble range")
        return values.astype(np.uint8)
    if kind == DEPTH_HUFFMAN:
        if len(raw) < 260:
            raise RepackError("truncated depth Huffman payload")
        lengths = np.frombuffer(raw[:256], dtype=np.uint8)
        size = struct.unpack_from("<I", raw, 256)[0]
        if len(raw) != 260 + size:
            raise RepackError("invalid depth Huffman payload")
        codes = _canonical_codes(lengths)
        decode_map = {
            (code, length): symbol for symbol, (code, length) in codes.items()
        }
        reader = _BitReader(raw[260:])
        values = []
        for _ in range(count):
            code = 0
            for length in range(1, int(lengths.max()) + 1):
                code = (code << 1) | reader.read(1)
                symbol = decode_map.get((code, length))
                if symbol is not None:
                    values.append(symbol)
                    break
            else:
                raise RepackError("invalid depth Huffman code")
        if any(value > 15 for value in values):
            raise RepackError("depth Huffman value outside nibble range")
        return np.asarray(values, dtype=np.uint8)
    raise RepackError("unknown row-depth representation")


def _self_segments(
    raw: bytes, layout: SelfCompressedHpacLayout
) -> tuple[np.ndarray, bytes, list[bytes]]:
    if len(raw) != layout.raw_bytes or not raw.startswith(SELF_MAGIC):
        raise RepackError("invalid deployed IHS1 payload")
    offset = 4
    depths = _unpack_nibbles(
        raw[offset : offset + layout.depth_bytes], layout.depth_count
    )
    offset += layout.depth_bytes
    weights = raw[offset : offset + layout.weight_bytes]
    offset += layout.weight_bytes
    tail = []
    for field in layout.tail_fields:
        tail.append(raw[offset : offset + field.byte_count])
        offset += field.byte_count
    if offset != len(raw):
        raise RepackError("IHS1 trailing bytes")
    return depths, weights, tail


def encode_self(
    raw: bytes,
    layout: SelfCompressedHpacLayout,
    *,
    frame_kind: int,
    depth_kind: int,
    pack_exponents: bool,
    pack_biases: bool,
) -> bytes:
    depths, weights, tail = _self_segments(raw, layout)
    bias_widths = []
    transformed = []
    for field, value in zip(layout.tail_fields, tail, strict=True):
        if field.name == "frame_embed.weight":
            value = _encode_frame(value, frame_kind)
        elif field.is_exponent and pack_exponents:
            codes = np.frombuffer(value, dtype=np.int8).astype(np.int16)
            if np.any(codes < -6) or np.any(codes > 0):
                raise RepackError("exponent lies outside [-6, 0]")
            value = _pack_unsigned(codes + 6, 3)
        elif field.is_bias and pack_biases:
            width = _signed_width(np.frombuffer(value, dtype="<i2"))
            bias_widths.append(width)
            value = _pack_signed(np.frombuffer(value, dtype="<i2"), width)
        transformed.append((field, value))
    depth_blob = _encode_depths(depths, layout, depth_kind)
    variable_frame = frame_kind in (FRAME_RICE, FRAME_HUFFMAN, FRAME_RANGE)
    output = bytearray(
        struct.pack(
            "<4sBBBBB",
            SELF_REPACK_MAGIC,
            VERSION,
            frame_kind,
            depth_kind,
            int(pack_exponents),
            int(pack_biases),
        )
    )
    output.extend(bytes(bias_widths))
    output.extend(struct.pack("<I", len(depth_blob)))
    output.extend(depth_blob)
    output.extend(weights)
    for field, value in transformed:
        if field.name == "frame_embed.weight" and variable_frame:
            output.extend(struct.pack("<I", len(value)))
        output.extend(value)
    return bytes(output)


def decode_self(blob: bytes, layout: SelfCompressedHpacLayout) -> bytes:
    if len(blob) < 9:
        raise RepackError("truncated HPS2 header")
    magic, version, frame_kind, depth_kind, exponent_flag, bias_flag = (
        struct.unpack_from("<4sBBBBB", blob)
    )
    if (
        magic != SELF_REPACK_MAGIC
        or version != VERSION
        or frame_kind not in FRAME_NAMES
        or depth_kind not in DEPTH_NAMES
        or exponent_flag not in (0, 1)
        or bias_flag not in (0, 1)
    ):
        raise RepackError("invalid HPS2 header")
    offset = 9
    biases = layout.bias_fields
    if bias_flag:
        if len(blob) < offset + len(biases):
            raise RepackError("truncated HPS2 bias widths")
        widths = list(blob[offset : offset + len(biases)])
        offset += len(biases)
        if any(width < 1 or width > 16 for width in widths):
            raise RepackError("invalid HPS2 bias width")
    else:
        widths = []
    if len(blob) < offset + 4:
        raise RepackError("truncated HPS2 depth length")
    depth_size = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    if len(blob) < offset + depth_size + layout.weight_bytes:
        raise RepackError("truncated HPS2 depth/weight section")
    depths = _decode_depths(blob[offset : offset + depth_size], layout, depth_kind)
    offset += depth_size
    weights = blob[offset : offset + layout.weight_bytes]
    offset += layout.weight_bytes
    output = bytearray(SELF_MAGIC)
    output.extend(_pack_nibbles(depths))
    output.extend(weights)
    bias_index = 0
    variable = frame_kind in (FRAME_RICE, FRAME_HUFFMAN, FRAME_RANGE)
    for field in layout.tail_fields:
        if field.name == "frame_embed.weight":
            size = (
                struct.unpack_from("<I", blob, offset)[0]
                if variable and len(blob) >= offset + 4
                else field.byte_count
            )
            if variable:
                offset += 4
            if len(blob) < offset + size:
                raise RepackError("truncated HPS2 frame field")
            output.extend(_decode_frame(blob[offset : offset + size], frame_kind))
            offset += size
        elif field.is_exponent and exponent_flag:
            size = (field.count * 3 + 7) // 8
            if len(blob) < offset + size:
                raise RepackError("truncated HPS2 exponent field")
            output.extend(
                (_unpack_unsigned(blob[offset : offset + size], field.count, 3) - 6)
                .astype(np.int8)
                .tobytes()
            )
            offset += size
        elif field.is_bias and bias_flag:
            width = widths[bias_index]
            bias_index += 1
            size = (field.count * width + 7) // 8
            if len(blob) < offset + size:
                raise RepackError("truncated HPS2 bias field")
            output.extend(
                _unpack_signed(blob[offset : offset + size], field.count, width)
                .astype("<i2")
                .tobytes()
            )
            offset += size
        else:
            if len(blob) < offset + field.byte_count:
                raise RepackError(f"truncated HPS2 field {field.name}")
            output.extend(blob[offset : offset + field.byte_count])
            offset += field.byte_count
    if offset != len(blob):
        raise RepackError("HPS2 trailing bytes")
    return bytes(output)


def encode(
    raw: bytes,
    layout: IntegerHpacLayout,
    *,
    frame_kind: int,
    pack_exponents: bool,
    pack_biases: bool,
) -> bytes:
    if len(raw) != layout.raw_bytes:
        raise RepackError("wrong raw HPAC byte count")
    bias_widths: list[int] = []
    fields: list[tuple[FieldSpec, bytes]] = []
    offset = 0
    for field in layout.fields:
        value = raw[offset : offset + field.byte_count]
        offset += field.byte_count
        if field.name == "frame_embed.weight":
            value = _encode_frame(value, frame_kind)
        elif field.is_exponent and pack_exponents:
            values = np.frombuffer(value, dtype=np.int8).astype(np.int16)
            if np.any(values < -6) or np.any(values > 0):
                raise RepackError("exponent lies outside [-6, 0]")
            value = _pack_unsigned(values + 6, 3)
        elif field.is_bias and pack_biases:
            values = np.frombuffer(value, dtype="<i2")
            width = _signed_width(values)
            bias_widths.append(width)
            value = _pack_signed(values, width)
        fields.append((field, value))
    frame_variable = frame_kind in (FRAME_RICE, FRAME_HUFFMAN, FRAME_RANGE)
    header = struct.pack(
        "<4sBBBB", MAGIC, VERSION, frame_kind, int(pack_exponents), int(pack_biases)
    ) + bytes(bias_widths)
    output = bytearray(header)
    for field, value in fields:
        if field.name == "frame_embed.weight" and frame_variable:
            output.extend(struct.pack("<I", len(value)))
        output.extend(value)
    return bytes(output)


def decode(blob: bytes, layout: IntegerHpacLayout) -> bytes:
    if len(blob) < 8:
        raise RepackError("truncated HPR1 header")
    magic, version, frame_kind, exponent_flag, bias_flag = struct.unpack_from(
        "<4sBBBB", blob
    )
    if (
        magic != MAGIC
        or version != VERSION
        or frame_kind not in FRAME_NAMES
        or exponent_flag not in (0, 1)
        or bias_flag not in (0, 1)
    ):
        raise RepackError("invalid HPR1 header")
    offset = 8
    bias_fields = layout.bias_fields
    if bias_flag:
        if len(blob) < offset + len(bias_fields):
            raise RepackError("truncated bias width table")
        widths = list(blob[offset : offset + len(bias_fields)])
        offset += len(bias_fields)
        if any(width < 1 or width > 16 for width in widths):
            raise RepackError("invalid bias width")
    else:
        widths = []
    bias_index = 0
    output = bytearray()
    variable = frame_kind in (FRAME_RICE, FRAME_HUFFMAN, FRAME_RANGE)
    for field in layout.fields:
        if field.name == "frame_embed.weight":
            size = (
                struct.unpack_from("<I", blob, offset)[0]
                if variable and len(blob) >= offset + 4
                else field.byte_count
            )
            if variable:
                offset += 4
            if len(blob) < offset + size:
                raise RepackError("truncated frame embedding")
            output.extend(_decode_frame(blob[offset : offset + size], frame_kind))
            offset += size
        elif field.is_exponent and exponent_flag:
            size = (field.count * 3 + 7) // 8
            if len(blob) < offset + size:
                raise RepackError("truncated packed exponents")
            output.extend(
                (_unpack_unsigned(blob[offset : offset + size], field.count, 3) - 6)
                .astype(np.int8)
                .tobytes()
            )
            offset += size
        elif field.is_bias and bias_flag:
            width = widths[bias_index]
            bias_index += 1
            size = (field.count * width + 7) // 8
            if len(blob) < offset + size:
                raise RepackError("truncated packed bias")
            output.extend(
                _unpack_signed(blob[offset : offset + size], field.count, width)
                .astype("<i2")
                .tobytes()
            )
            offset += size
        else:
            if len(blob) < offset + field.byte_count:
                raise RepackError(f"truncated field {field.name}")
            output.extend(blob[offset : offset + field.byte_count])
            offset += field.byte_count
    if offset != len(blob):
        raise RepackError("HPR1 trailing bytes")
    return bytes(output)
