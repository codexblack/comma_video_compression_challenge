"""Exact, charged payloads for deterministic frame-0 catalog selection.

The codec deliberately stores every global mode definition and the complete
per-pair selector stream.  It uses a canonical Huffman code derived from the
stored frequencies, so the decoder needs no external table or model.
"""

from __future__ import annotations

import heapq
import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

MAGIC = b"F0S1"
VERSION = 1
SPARSE_MAGIC = b"F0E1"
SPARSE_VERSION = 1
_HEADER = struct.Struct("<4sBBHHH")
_SPARSE_HEADER = struct.Struct("<4sBH")
_MODE = struct.Struct("<Bbbbb")
_MAX_MODES = 16

IDENTITY = 0
COEFFICIENT = 1
COEFFICIENT_PAIR = 2
LUMA = 3
CHANNEL = 4
ROLL = 5
TILE = 6
_KINDS = frozenset((IDENTITY, COEFFICIENT, COEFFICIENT_PAIR, LUMA, CHANNEL, ROLL, TILE))


class Frame0SelectorError(ValueError):
    """A frame-0 selector payload or mode definition is invalid."""


@dataclass(frozen=True, order=True)
class SelectorMode:
    """A fixed-width, integer-only deterministic frame-0 perturbation."""

    kind: int
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0

    def encoded(self) -> bytes:
        _validate_mode(self)
        return _MODE.pack(self.kind, self.a, self.b, self.c, self.d)


# F0E1 is deliberately a fixed catalog. Identity is implicit in the sparse
# event stream; the remaining seven modes are the non-carrier pixel modes
# already proven deterministic by F0S1.
SPARSE_PIXEL_MODES = (
    SelectorMode(IDENTITY),
    SelectorMode(LUMA, 1),
    SelectorMode(LUMA, -1),
    SelectorMode(CHANNEL, 1, 0, -1),
    SelectorMode(ROLL, 1, 0),
    SelectorMode(ROLL, 0, 1),
    SelectorMode(TILE, 0, 1),
    SelectorMode(TILE, 3, 1),
)


def _validate_mode(mode: SelectorMode) -> None:
    if not isinstance(mode, SelectorMode) or mode.kind not in _KINDS:
        raise Frame0SelectorError("unsupported selector mode")
    if any(
        not isinstance(value, int) or not -128 <= value <= 127
        for value in (mode.a, mode.b, mode.c, mode.d)
    ):
        raise Frame0SelectorError("selector operands must be signed bytes")
    if mode.kind == IDENTITY and (mode.a, mode.b, mode.c, mode.d) != (0, 0, 0, 0):
        raise Frame0SelectorError("identity mode has operands")
    if mode.kind == COEFFICIENT and not (
        0 <= mode.a < 12 and -2 <= mode.b <= 2 and mode.b and mode.c == mode.d == 0
    ):
        raise Frame0SelectorError("invalid single-coefficient selector mode")
    if mode.kind == COEFFICIENT_PAIR and not (
        0 <= mode.a < 12
        and 0 <= mode.c < 12
        and mode.a != mode.c
        and -2 <= mode.b <= 2
        and -2 <= mode.d <= 2
        and mode.b
        and mode.d
    ):
        raise Frame0SelectorError("invalid coefficient-pair selector mode")
    if mode.kind == LUMA and not (
        -8 <= mode.a <= 8 and mode.a and mode.b == mode.c == mode.d == 0
    ):
        raise Frame0SelectorError("invalid luma selector mode")
    if mode.kind == CHANNEL and not (
        any((mode.a, mode.b, mode.c))
        and all(-8 <= item <= 8 for item in (mode.a, mode.b, mode.c))
        and mode.d == 0
    ):
        raise Frame0SelectorError("invalid channel selector mode")
    if mode.kind == ROLL and not (
        -1 <= mode.a <= 1
        and -1 <= mode.b <= 1
        and (mode.a or mode.b)
        and mode.c == mode.d == 0
    ):
        raise Frame0SelectorError("invalid roll selector mode")
    if mode.kind == TILE and not (
        0 <= mode.a <= 3 and -4 <= mode.b <= 4 and mode.b and mode.c == mode.d == 0
    ):
        raise Frame0SelectorError("invalid tile selector mode")


def _huffman_codes(frequencies: Sequence[int]) -> dict[int, tuple[int, int]]:
    """Return canonical ``symbol -> (code, bit_length)`` mappings."""
    active = [
        (int(count), symbol)
        for symbol, count in enumerate(frequencies)
        if int(count) > 0
    ]
    if not active:
        raise Frame0SelectorError("selector index stream is empty")
    if len(active) == 1:
        return {active[0][1]: (0, 1)}
    heapq.heapify(active)
    tree: dict[int, tuple[int, int]] = {}
    next_node = len(frequencies)
    while len(active) > 1:
        count_a, node_a = heapq.heappop(active)
        count_b, node_b = heapq.heappop(active)
        tree[next_node] = (node_a, node_b)
        heapq.heappush(active, (count_a + count_b, next_node))
        next_node += 1
    lengths: dict[int, int] = {}
    stack = [(active[0][1], 0)]
    while stack:
        node, depth = stack.pop()
        if node < len(frequencies):
            lengths[node] = depth
            continue
        left, right = tree[node]
        stack.append((right, depth + 1))
        stack.append((left, depth + 1))
    code = 0
    previous_length = 0
    result: dict[int, tuple[int, int]] = {}
    for length, symbol in sorted(
        (length, symbol) for symbol, length in lengths.items()
    ):
        code <<= length - previous_length
        result[symbol] = (code, length)
        code += 1
        previous_length = length
    return result


def _pack_codes(
    indices: Iterable[int], codes: dict[int, tuple[int, int]]
) -> tuple[bytes, int]:
    value = 0
    count = 0
    output = bytearray()
    for index in indices:
        code, width = codes[int(index)]
        value = (value << width) | code
        count += width
        while count >= 8:
            count -= 8
            output.append((value >> count) & 0xFF)
    if count:
        output.append((value << (8 - count)) & 0xFF)
    return bytes(output), sum(
        width for _, width in (codes[int(index)] for index in indices)
    )


def _decode_codes(
    payload: bytes, bit_count: int, frequencies: Sequence[int], frames: int
) -> np.ndarray:
    if (
        bit_count <= 0
        or bit_count > len(payload) * 8
        or (bit_count % 8 and payload[-1] & ((1 << (8 - bit_count % 8)) - 1))
    ):
        raise Frame0SelectorError("invalid selector code bit count or padding")
    codes = _huffman_codes(frequencies)
    reverse = {(code, width): symbol for symbol, (code, width) in codes.items()}
    max_width = max(width for _, width in codes.values())
    decoded: list[int] = []
    value = 0
    width = 0
    for cursor in range(bit_count):
        byte, shift = payload[cursor // 8], 7 - cursor % 8
        value = (value << 1) | ((byte >> shift) & 1)
        width += 1
        symbol = reverse.get((value, width))
        if symbol is not None:
            decoded.append(symbol)
            value = 0
            width = 0
        elif width > max_width:
            raise Frame0SelectorError("invalid canonical selector code")
    if value or width or len(decoded) != frames:
        raise Frame0SelectorError("truncated or overlong selector index stream")
    decoded_array = np.asarray(decoded, dtype=np.uint8)
    if not np.array_equal(
        np.bincount(decoded_array, minlength=len(frequencies)),
        np.asarray(frequencies, dtype=np.int64),
    ):
        raise Frame0SelectorError("selector frequencies do not match index stream")
    return decoded_array


def encode_selector(modes: Sequence[SelectorMode], indices: np.ndarray) -> bytes:
    """Encode modes and one exact catalog choice for each frame/pair."""
    if not 1 <= len(modes) <= _MAX_MODES:
        raise Frame0SelectorError("selector needs one to sixteen modes")
    if len(set(modes)) != len(modes):
        raise Frame0SelectorError("selector modes must be unique")
    for mode in modes:
        _validate_mode(mode)
    choices = np.asarray(indices)
    if (
        choices.ndim != 1
        or not np.issubdtype(choices.dtype, np.integer)
        or not 1 <= choices.size <= 0xFFFF
    ):
        raise Frame0SelectorError("selector indices must be a nonempty integer vector")
    if np.any(choices < 0) or np.any(choices >= len(modes)):
        raise Frame0SelectorError("selector index is out of range")
    frequencies = np.bincount(
        choices.astype(np.int64, copy=False), minlength=len(modes)
    ).astype("<u2")
    codes = _huffman_codes(frequencies)
    packed, bit_count = _pack_codes(choices, codes)
    if bit_count > 0xFFFF:
        raise Frame0SelectorError("selector Huffman stream exceeds u16 bit count")
    return (
        _HEADER.pack(
            MAGIC, VERSION, len(modes), choices.size, len(modes) * _MODE.size, bit_count
        )
        + b"".join(mode.encoded() for mode in modes)
        + frequencies.tobytes()
        + packed
    )


def _combination_rank(positions: np.ndarray) -> int:
    """Return the canonical colexicographic rank of sorted frame positions."""
    return sum(
        math.comb(int(position), index + 1) for index, position in enumerate(positions)
    )


def _combination_unrank(rank: int, count: int, frames: int) -> np.ndarray:
    if not 0 <= count <= frames or not 0 <= rank < math.comb(frames, count):
        raise Frame0SelectorError("sparse selector combination rank is out of range")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    positions = np.empty(count, dtype=np.int64)
    upper = frames - 1
    remaining = rank
    for width in range(count, 0, -1):
        low, high = width - 1, upper
        while low < high:
            middle = (low + high + 1) // 2
            if math.comb(middle, width) <= remaining:
                low = middle
            else:
                high = middle - 1
        positions[width - 1] = low
        remaining -= math.comb(low, width)
        upper = low - 1
    if remaining:
        raise Frame0SelectorError("non-canonical sparse selector combination rank")
    return positions


def _pack_sparse_labels(labels: np.ndarray) -> bytes:
    value = 0
    bits = 0
    output = bytearray()
    for label in labels:
        value = (value << 3) | int(label)
        bits += 3
        while bits >= 8:
            bits -= 8
            output.append((value >> bits) & 0xFF)
    if bits:
        output.append((value << (8 - bits)) & 0xFF)
    return bytes(output)


def _unpack_sparse_labels(payload: bytes, count: int) -> np.ndarray:
    bit_count = count * 3
    if len(payload) != (bit_count + 7) // 8:
        raise Frame0SelectorError("invalid sparse selector label length")
    if bit_count % 8 and payload[-1] & ((1 << (8 - bit_count % 8)) - 1):
        raise Frame0SelectorError("nonzero sparse selector label padding")
    result = np.empty(count, dtype=np.uint8)
    cursor = 0
    for index in range(count):
        label = 0
        for _ in range(3):
            byte, shift = payload[cursor // 8], 7 - cursor % 8
            label = (label << 1) | ((byte >> shift) & 1)
            cursor += 1
        if label >= len(SPARSE_PIXEL_MODES) - 1:
            raise Frame0SelectorError("sparse selector label is out of range")
        result[index] = label + 1
    return result


def encode_sparse_selector(indices: np.ndarray) -> bytes:
    """Encode sparse choices from the fixed eight-mode pixel catalog.

    Nonidentity frame positions use an exact combinatorial rank. Their seven
    possible labels use fixed three-bit codes. The 600-frame challenge shape
    and every mode definition are schema constants rather than charged data.
    """
    choices = np.asarray(indices)
    if choices.shape != (600,) or not np.issubdtype(choices.dtype, np.integer):
        raise Frame0SelectorError(
            "sparse selector requires exactly 600 integer choices"
        )
    if np.any(choices < 0) or np.any(choices >= len(SPARSE_PIXEL_MODES)):
        raise Frame0SelectorError("sparse selector choice is out of range")
    positions = np.flatnonzero(choices).astype(np.int64, copy=False)
    count = int(positions.size)
    if count == 0:
        raise Frame0SelectorError(
            "an all-identity stream must omit the sparse selector"
        )
    limit = math.comb(600, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    rank = _combination_rank(positions)
    if not 0 <= rank < limit:
        raise Frame0SelectorError("sparse selector rank construction failed")
    labels = choices[positions].astype(np.uint8, copy=False) - 1
    return (
        _SPARSE_HEADER.pack(SPARSE_MAGIC, SPARSE_VERSION, count)
        + rank.to_bytes(rank_bytes, "big")
        + _pack_sparse_labels(labels)
    )


def _decode_sparse_selector(
    payload: bytes,
) -> tuple[tuple[SelectorMode, ...], np.ndarray]:
    if len(payload) < _SPARSE_HEADER.size:
        raise Frame0SelectorError("truncated sparse selector header")
    magic, version, count = _SPARSE_HEADER.unpack_from(payload)
    if magic != SPARSE_MAGIC or version != SPARSE_VERSION or not 1 <= count <= 600:
        raise Frame0SelectorError("invalid sparse selector header")
    limit = math.comb(600, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    expected = _SPARSE_HEADER.size + rank_bytes + label_bytes
    if len(payload) != expected:
        raise Frame0SelectorError("truncated or trailing sparse selector payload")
    offset = _SPARSE_HEADER.size
    rank = int.from_bytes(payload[offset : offset + rank_bytes], "big")
    positions = _combination_unrank(rank, count, 600)
    labels = _unpack_sparse_labels(payload[offset + rank_bytes :], count)
    choices = np.zeros(600, dtype=np.uint8)
    choices[positions] = labels
    return SPARSE_PIXEL_MODES, choices


def decode_selector(payload: bytes) -> tuple[tuple[SelectorMode, ...], np.ndarray]:
    """Strictly decode one selector payload, rejecting all trailing data."""
    if payload.startswith(SPARSE_MAGIC):
        return _decode_sparse_selector(payload)
    if len(payload) < _HEADER.size:
        raise Frame0SelectorError("truncated selector header")
    magic, version, mode_count, frames, mode_bytes, bit_count = _HEADER.unpack_from(
        payload
    )
    if (
        magic != MAGIC
        or version != VERSION
        or not 1 <= mode_count <= _MAX_MODES
        or not frames
        or mode_bytes != mode_count * _MODE.size
    ):
        raise Frame0SelectorError("invalid selector header")
    offset = _HEADER.size
    required = offset + mode_bytes + 2 * mode_count + (bit_count + 7) // 8
    if len(payload) != required:
        raise Frame0SelectorError("truncated or trailing selector payload")
    modes = tuple(
        SelectorMode(*_MODE.unpack_from(payload, offset + _MODE.size * item))
        for item in range(mode_count)
    )
    if len(set(modes)) != len(modes):
        raise Frame0SelectorError("duplicate selector mode")
    for mode in modes:
        _validate_mode(mode)
    offset += mode_bytes
    frequencies = np.frombuffer(
        payload[offset : offset + 2 * mode_count], dtype="<u2"
    ).astype(np.int64)
    if int(frequencies.sum()) != frames:
        raise Frame0SelectorError("selector frequencies do not sum to frame count")
    offset += 2 * mode_count
    return modes, _decode_codes(payload[offset:], bit_count, frequencies, frames)


def apply_pixel_mode(frames: np.ndarray, mode: SelectorMode) -> np.ndarray:
    """Apply one non-coefficient mode with integer, CPU-stable operations.

    ``COEFFICIENT`` variants must be rendered from their changed carrier codes
    before this function is called; pixel modes never use interpolation.
    """
    _validate_mode(mode)
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise Frame0SelectorError("pixel selector needs BxHxWx3 uint8 frames")
    if mode.kind in (COEFFICIENT, COEFFICIENT_PAIR):
        raise Frame0SelectorError("coefficient selector mode needs carrier rendering")
    if mode.kind == IDENTITY:
        return values.copy()
    if mode.kind == ROLL:
        return np.roll(values, shift=(mode.b, mode.a), axis=(1, 2))
    if mode.kind == TILE:
        height, width = values.shape[1:3]
        yy, xx = np.indices((height, width), dtype=np.int32)
        if mode.a == 0:
            signs = ((yy + xx) & 1) * 2 - 1
        elif mode.a == 1:
            signs = (yy & 1) * 2 - 1
        elif mode.a == 2:
            signs = (xx & 1) * 2 - 1
        else:
            signs = (((yy >> 2) + (xx >> 2)) & 1) * 2 - 1
        delta = signs[None, :, :, None] * mode.b
    elif mode.kind == LUMA:
        delta = mode.a
    elif mode.kind == CHANNEL:
        delta = np.asarray((mode.a, mode.b, mode.c), dtype=np.int16).reshape(1, 1, 1, 3)
    else:  # pragma: no cover - _validate_mode guards this closed set.
        raise AssertionError("unreachable selector pixel mode")
    return np.clip(values.astype(np.int16) + delta, 0, 255).astype(np.uint8)
