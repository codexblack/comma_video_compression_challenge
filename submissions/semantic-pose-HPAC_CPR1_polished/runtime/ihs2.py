"""Exact fixed-schema IHS2 representation for the deployed CPR1 IntegerHPAC.

IHS2 is a storage representation, not a new model.  It reconstructs the
byte-identical deployed IHS1 blob before the pinned CPR1 loader is called.
The only constants in this module describe the already-deployed architecture;
all model values, original depths, and reduced depths are carried in IHS2.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from .hpac_repack import FieldSpec

IHS1_MAGIC = b"IHS1"
IHS2_MAGIC = b"IHS2"
IHS2_VERSION = 1

# The three low bits select the frame field.  Zero intentionally means the
# original raw int8 frame field, so isolated exponent/depth candidates need no
# special side format.
FRAME_RAW = 0
FRAME_E0L0 = 1
FRAME_E0L1 = 2
FRAME_E1L0 = 3
FRAME_E1L1 = 4
FRAME_E2L0 = 5
FRAME_E2L1 = 6
FRAME_FORMATS = {
    FRAME_RAW: "raw_i8",
    FRAME_E0L0: "E0L0_twos_complement_frame_major",
    FRAME_E0L1: "E0L1_twos_complement_dimension_major",
    FRAME_E1L0: "E1L0_biased_frame_major",
    FRAME_E1L1: "E1L1_biased_dimension_major",
    FRAME_E2L0: "E2L0_zigzag_frame_major",
    FRAME_E2L1: "E2L1_zigzag_dimension_major",
}

FLAG_EXPONENTS_3BIT = 1 << 3
FLAG_TIGHT_ROWS = 1 << 4
_KNOWN_FLAG_MASK = 0x1F


class IHS2Error(ValueError):
    """An IHS1/IHS2 representation is invalid or cannot be reconstructed."""


@dataclass(frozen=True)
class IHS2Layout:
    """Known CPR1 HPAC structure; none of these entries are learned values."""

    row_counts: tuple[int, ...]
    module_ranges: tuple[tuple[str, int, int], ...]
    tail_fields: tuple[FieldSpec, ...]

    @property
    def depth_count(self) -> int:
        return len(self.row_counts)

    @property
    def depth_bytes(self) -> int:
        return (self.depth_count + 1) // 2

    @property
    def frame_field(self) -> FieldSpec:
        fields = [
            field for field in self.tail_fields if field.name == "frame_embed.weight"
        ]
        if len(fields) != 1 or fields[0].count != 600 * 8 or fields[0].dtype != "i1":
            raise IHS2Error("IHS2 requires one 600x8 int8 frame embedding")
        return fields[0]

    @property
    def exponent_count(self) -> int:
        return sum(field.count for field in self.tail_fields if field.is_exponent)

    @property
    def non_exponent_tail_bytes(self) -> int:
        return sum(
            field.byte_count
            for field in self.tail_fields
            if not field.is_exponent and field.name != "frame_embed.weight"
        )

    @property
    def raw_tail_bytes(self) -> int:
        return sum(field.byte_count for field in self.tail_fields)


@dataclass(frozen=True)
class IHS1Segments:
    original_depths: np.ndarray
    rows: tuple[np.ndarray, ...]
    frame: bytes
    non_exponent_tail: tuple[bytes, ...]
    exponents: np.ndarray


def layout_from_model(model) -> IHS2Layout:
    """Derive the fixed deployed layout from an unmodified CPR1 model object."""
    integer = importlib.import_module("hpac_integer")
    compressed_types = (integer.IntegerConv2d, integer.IntegerLinear)
    modules = dict(model.named_modules())
    row_counts: list[int] = []
    module_ranges: list[tuple[str, int, int]] = []
    for name, module in model.named_modules():
        if not isinstance(module, compressed_types):
            continue
        start = len(row_counts)
        if isinstance(module, integer.IntegerConv2d):
            mask = module.mask.to(bool).expand_as(module.weight)
            row_counts.extend(
                int(mask[index].sum().item()) for index in range(module.weight.shape[0])
            )
        else:
            row_counts.extend(
                int(module.weight[index].numel())
                for index in range(module.weight.shape[0])
            )
        module_ranges.append((name, start, len(row_counts)))
    fields: list[FieldSpec] = []
    for name, parameter in model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        module = modules[module_name]
        if field == "weight" and isinstance(module, compressed_types):
            continue
        fields.append(
            FieldSpec(
                name,
                parameter.numel(),
                "<i2" if field == "bias" else "i1",
                tuple(parameter.shape),
            )
        )
    layout = IHS2Layout(tuple(row_counts), tuple(module_ranges), tuple(fields))
    # Fail before writing an artifact if the pinned architecture changed.
    if layout.depth_count != 517 or layout.exponent_count != 517:
        raise IHS2Error("unexpected CPR1 HPAC structural schema")
    if layout.frame_field.count != 4_800:
        raise IHS2Error("unexpected CPR1 HPAC frame embedding")
    return layout


def layout_from_runtime(runtime) -> IHS2Layout:
    """Build a value-free CPR1 model shell and inspect its pinned structure."""
    model = runtime.IntegerHPAC(
        num_pairs=runtime.N,
        num_classes=runtime.NUM_CLASSES,
        patch=runtime.HPAC_PATCH,
        delta=runtime.HPAC_DELTA,
        channels=runtime.HPAC_CHANNELS,
        frame_dim=runtime.HPAC_FILM_DIM,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=127,
        activation_bound=127,
        use_weight_scales=True,
        weight_exponent_min=-6,
        use_spm=True,
        use_norm_gates=False,
    ).eval()
    return layout_from_model(model)


def _pack_nibbles(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(values > 15):
        raise IHS2Error("nibble value outside 0..15")
    padded = np.pad(values, (0, values.size % 2), constant_values=0)
    return (padded[0::2] | (padded[1::2] << 4)).tobytes()


def _unpack_nibbles(raw: bytes, count: int) -> np.ndarray:
    if len(raw) != (count + 1) // 2:
        raise IHS2Error("truncated nibble field")
    if count % 2 and raw[-1] >> 4:
        raise IHS2Error("non-zero odd-nibble padding")
    source = np.frombuffer(raw, dtype=np.uint8)
    values = np.empty(source.size * 2, dtype=np.uint8)
    values[0::2] = source & 0x0F
    values[1::2] = source >> 4
    return values[:count]


def _pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    if not 1 <= bits <= 15:
        raise IHS2Error("invalid packed bit width")
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(values < 0) or np.any(values >= (1 << bits)):
        raise IHS2Error("unsigned value outside packed range")
    output = bytearray((values.size * bits + 7) // 8)
    offset = 0
    for value in values:
        byte, shift = divmod(offset, 8)
        word = int(value) << shift
        output[byte] |= word & 0xFF
        if shift + bits > 8:
            output[byte + 1] |= (word >> 8) & 0xFF
        if shift + bits > 16:
            output[byte + 2] |= (word >> 16) & 0xFF
        offset += bits
    return bytes(output)


def _unpack_unsigned(raw: bytes, count: int, bits: int) -> np.ndarray:
    expected = (count * bits + 7) // 8
    if len(raw) != expected:
        raise IHS2Error("truncated packed integer field")
    if count * bits % 8 and raw[-1] >> (count * bits % 8):
        raise IHS2Error("non-zero packed-field padding")
    values = np.empty(count, dtype=np.int16)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        if byte + 2 < len(raw):
            word |= raw[byte + 2] << 16
        values[index] = (word >> shift) & ((1 << bits) - 1)
    return values


def _read_signed_rows(
    raw: bytes, depths: np.ndarray, layout: IHS2Layout
) -> tuple[np.ndarray, ...]:
    total_bits = int(
        sum(
            int(depth) * count
            for depth, count in zip(depths, layout.row_counts, strict=True)
        )
    )
    expected = (total_bits + 7) // 8
    if len(raw) != expected:
        raise IHS2Error("invalid IHS1 weight bitstream length")
    if total_bits % 8 and raw[-1] >> (total_bits % 8):
        raise IHS2Error("IHS1 non-zero weight padding")
    rows: list[np.ndarray] = []
    offset = 0
    for depth, count in zip(depths, layout.row_counts, strict=True):
        depth = int(depth)
        if depth == 0:
            rows.append(np.zeros(count, dtype=np.int16))
            continue
        values = np.empty(count, dtype=np.int16)
        sign = 1 << (depth - 1)
        for index in range(count):
            byte, shift = divmod(offset, 8)
            word = raw[byte]
            if byte + 1 < len(raw):
                word |= raw[byte + 1] << 8
            if byte + 2 < len(raw):
                word |= raw[byte + 2] << 16
            unsigned = (word >> shift) & ((1 << depth) - 1)
            values[index] = unsigned - (1 << depth) if unsigned & sign else unsigned
            offset += depth
        rows.append(values)
    if offset != total_bits:
        raise IHS2Error("IHS1 row bit accounting mismatch")
    return tuple(rows)


def _write_signed_rows(
    rows: tuple[np.ndarray, ...], depths: np.ndarray, layout: IHS2Layout
) -> bytes:
    if len(rows) != layout.depth_count or len(depths) != layout.depth_count:
        raise IHS2Error("row/depth count mismatch")
    total_bits = int(
        sum(
            int(depth) * count
            for depth, count in zip(depths, layout.row_counts, strict=True)
        )
    )
    output = bytearray((total_bits + 7) // 8)
    offset = 0
    for row, depth, count in zip(rows, depths, layout.row_counts, strict=True):
        values = np.asarray(row, dtype=np.int16).reshape(-1)
        depth = int(depth)
        if values.size != count:
            raise IHS2Error("row parameter count mismatch")
        if depth == 0:
            if np.any(values):
                raise IHS2Error("zero-bit row contains a non-zero value")
            continue
        minimum, maximum = -(1 << (depth - 1)), (1 << (depth - 1)) - 1
        if np.any(values < minimum) or np.any(values > maximum):
            raise IHS2Error("row value outside declared signed depth")
        for value in values:
            byte, shift = divmod(offset, 8)
            word = int(value) & ((1 << depth) - 1)
            packed = word << shift
            output[byte] |= packed & 0xFF
            if shift + depth > 8:
                output[byte + 1] |= (packed >> 8) & 0xFF
            if shift + depth > 16:
                output[byte + 2] |= (packed >> 16) & 0xFF
            offset += depth
    if offset != total_bits:
        raise IHS2Error("row packing bit accounting mismatch")
    return bytes(output)


def minimum_signed_depth(values: np.ndarray) -> int:
    """Canonical exact signed width, with an explicit zero-bit all-zero row."""
    values = np.asarray(values, dtype=np.int16).reshape(-1)
    if not values.size or not np.any(values):
        return 0
    minimum, maximum = int(values.min()), int(values.max())
    for bits in range(1, 16):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return bits
    raise IHS2Error("row cannot be represented in deployed signed depth range")


def parse_ihs1(raw: bytes, layout: IHS2Layout) -> IHS1Segments:
    if not raw.startswith(IHS1_MAGIC):
        raise IHS2Error("expected IHS1 input")
    if len(raw) < len(IHS1_MAGIC) + layout.depth_bytes:
        raise IHS2Error("truncated IHS1 depth metadata")
    offset = len(IHS1_MAGIC)
    depths = _unpack_nibbles(
        raw[offset : offset + layout.depth_bytes], layout.depth_count
    ).copy()
    offset += layout.depth_bytes
    weight_bits = int(
        sum(
            int(depth) * count
            for depth, count in zip(depths, layout.row_counts, strict=True)
        )
    )
    weight_bytes = (weight_bits + 7) // 8
    if len(raw) < offset + weight_bytes + layout.raw_tail_bytes:
        raise IHS2Error("truncated IHS1 payload")
    rows = _read_signed_rows(raw[offset : offset + weight_bytes], depths, layout)
    offset += weight_bytes
    frame = b""
    non_exponent: list[bytes] = []
    exponents: list[np.ndarray] = []
    for field in layout.tail_fields:
        value = raw[offset : offset + field.byte_count]
        if len(value) != field.byte_count:
            raise IHS2Error(f"truncated IHS1 field {field.name}")
        offset += field.byte_count
        if field.name == "frame_embed.weight":
            frame = value
        elif field.is_exponent:
            exponents.append(np.frombuffer(value, dtype=np.int8).copy())
        else:
            non_exponent.append(value)
    if offset != len(raw):
        raise IHS2Error("IHS1 trailing data")
    if len(frame) != layout.frame_field.byte_count:
        raise IHS2Error("IHS1 missing frame embedding")
    return IHS1Segments(
        depths, rows, frame, tuple(non_exponent), np.concatenate(exponents)
    )


def _encode_frame(raw: bytes, frame_format: int) -> bytes:
    if frame_format == FRAME_RAW:
        if len(raw) != 600 * 8:
            raise IHS2Error("invalid raw frame embedding length")
        return raw
    if frame_format not in FRAME_FORMATS:
        raise IHS2Error("invalid frame format")
    matrix = np.frombuffer(raw, dtype=np.int8).reshape(600, 8)
    if matrix.min() < -8 or matrix.max() > 7:
        raise IHS2Error("frame embedding cannot be represented as exact signed int4")
    values = (
        matrix.T.reshape(-1)
        if frame_format in (FRAME_E0L1, FRAME_E1L1, FRAME_E2L1)
        else matrix.reshape(-1)
    )
    signed = values.astype(np.int16)
    if frame_format in (FRAME_E0L0, FRAME_E0L1):
        codes = signed & 0xF
    elif frame_format in (FRAME_E1L0, FRAME_E1L1):
        codes = signed + 8
    else:
        codes = np.where(signed >= 0, 2 * signed, -2 * signed - 1)
    return _pack_nibbles(codes.astype(np.uint8))


def _decode_frame(raw: bytes, frame_format: int) -> bytes:
    if frame_format == FRAME_RAW:
        if len(raw) != 600 * 8:
            raise IHS2Error("invalid raw frame embedding length")
        return raw
    if frame_format not in FRAME_FORMATS:
        raise IHS2Error("invalid frame format")
    codes = _unpack_nibbles(raw, 600 * 8).astype(np.int16)
    if frame_format in (FRAME_E0L0, FRAME_E0L1):
        values = np.where(codes >= 8, codes - 16, codes)
    elif frame_format in (FRAME_E1L0, FRAME_E1L1):
        values = codes - 8
    else:
        values = np.where(codes & 1, -(codes // 2) - 1, codes // 2)
    if np.any(values < -8) or np.any(values > 7):
        raise IHS2Error("invalid decoded int4 frame code")
    matrix = (
        values.reshape(8, 600).T
        if frame_format in (FRAME_E0L1, FRAME_E1L1, FRAME_E2L1)
        else values.reshape(600, 8)
    )
    return matrix.astype(np.int8).tobytes()


def encode_ihs2(
    raw_ihs1: bytes,
    layout: IHS2Layout,
    *,
    frame_format: int = FRAME_RAW,
    pack_exponents: bool = False,
    tighten_rows: bool = False,
) -> bytes:
    """Encode a deterministic exact IHS2 blob from an IHS1 model payload."""
    if frame_format not in FRAME_FORMATS:
        raise IHS2Error("invalid IHS2 frame format")
    segments = parse_ihs1(raw_ihs1, layout)
    if np.any(segments.exponents < -6) or np.any(segments.exponents > 0):
        raise IHS2Error("exponent outside exact IHS2 alphabet [-6, 0]")
    flags = (
        frame_format
        | (FLAG_EXPONENTS_3BIT if pack_exponents else 0)
        | (FLAG_TIGHT_ROWS if tighten_rows else 0)
    )
    depths = (
        np.asarray([minimum_signed_depth(row) for row in segments.rows], dtype=np.uint8)
        if tighten_rows
        else segments.original_depths
    )
    if np.any(depths > segments.original_depths):
        raise IHS2Error("canonical depth unexpectedly exceeds deployed depth")
    output = bytearray(IHS2_MAGIC + bytes((IHS2_VERSION, flags)))
    output.extend(_pack_nibbles(segments.original_depths))
    if tighten_rows:
        output.extend(_pack_nibbles(depths))
    output.extend(_write_signed_rows(segments.rows, depths, layout))
    output.extend(_encode_frame(segments.frame, frame_format))
    for value in segments.non_exponent_tail:
        output.extend(value)
    if pack_exponents:
        output.extend(_pack_unsigned(segments.exponents.astype(np.int16) + 6, 3))
    else:
        output.extend(segments.exponents.astype(np.int8).tobytes())
    return bytes(output)


def _decode_ihs2_v1(blob: bytes, layout: IHS2Layout) -> bytes:
    """Strictly reconstruct the original byte-level IHS1 model representation."""
    if len(blob) < 6:
        raise IHS2Error("truncated IHS2 header")
    if blob[:4] != IHS2_MAGIC:
        raise IHS2Error("invalid IHS2 magic")
    version, flags = blob[4], blob[5]
    if version != IHS2_VERSION:
        raise IHS2Error("unsupported IHS2 version")
    if flags & ~_KNOWN_FLAG_MASK:
        raise IHS2Error("IHS2 reserved flags are non-zero")
    frame_format = flags & 0x07
    if frame_format not in FRAME_FORMATS:
        raise IHS2Error("IHS2 reserved frame format")
    pack_exponents = bool(flags & FLAG_EXPONENTS_3BIT)
    tighten_rows = bool(flags & FLAG_TIGHT_ROWS)
    offset = 6
    if len(blob) < offset + layout.depth_bytes:
        raise IHS2Error("truncated IHS2 original depths")
    original_depths = _unpack_nibbles(
        blob[offset : offset + layout.depth_bytes], layout.depth_count
    )
    offset += layout.depth_bytes
    if tighten_rows:
        if len(blob) < offset + layout.depth_bytes:
            raise IHS2Error("truncated IHS2 tightened depths")
        stored_depths = _unpack_nibbles(
            blob[offset : offset + layout.depth_bytes], layout.depth_count
        )
        offset += layout.depth_bytes
        if np.any(stored_depths > original_depths):
            raise IHS2Error("tightened depth exceeds original declared depth")
    else:
        stored_depths = original_depths
    weight_bits = int(
        sum(
            int(depth) * count
            for depth, count in zip(stored_depths, layout.row_counts, strict=True)
        )
    )
    weight_bytes = (weight_bits + 7) // 8
    if len(blob) < offset + weight_bytes:
        raise IHS2Error("truncated IHS2 weights")
    rows = _read_signed_rows(
        blob[offset : offset + weight_bytes], stored_depths, layout
    )
    offset += weight_bytes
    if tighten_rows:
        canonical = np.asarray(
            [minimum_signed_depth(row) for row in rows], dtype=np.uint8
        )
        if not np.array_equal(canonical, stored_depths):
            raise IHS2Error("non-canonical or inconsistent tightened row depths")
    frame_bytes = 600 * 8 if frame_format == FRAME_RAW else (600 * 8 + 1) // 2
    if len(blob) < offset + frame_bytes:
        raise IHS2Error("truncated IHS2 frame embedding")
    frame = _decode_frame(blob[offset : offset + frame_bytes], frame_format)
    offset += frame_bytes
    non_exponent: list[bytes] = []
    for field in layout.tail_fields:
        if field.name == "frame_embed.weight" or field.is_exponent:
            continue
        if len(blob) < offset + field.byte_count:
            raise IHS2Error(f"truncated IHS2 field {field.name}")
        non_exponent.append(blob[offset : offset + field.byte_count])
        offset += field.byte_count
    exponent_bytes = (
        (layout.exponent_count * 3 + 7) // 8
        if pack_exponents
        else layout.exponent_count
    )
    if len(blob) != offset + exponent_bytes:
        raise IHS2Error("IHS2 truncated or trailing exponent section")
    if pack_exponents:
        exponent_codes = _unpack_unsigned(blob[offset:], layout.exponent_count, 3)
        if np.any(exponent_codes == 7):
            raise IHS2Error("IHS2 reserved exponent code 7")
        exponents = (exponent_codes - 6).astype(np.int8)
    else:
        exponents = np.frombuffer(blob[offset:], dtype=np.int8).copy()
    if np.any(exponents < -6) or np.any(exponents > 0):
        raise IHS2Error("IHS2 exponent outside exact alphabet")
    output = bytearray(IHS1_MAGIC)
    output.extend(_pack_nibbles(original_depths))
    output.extend(_write_signed_rows(rows, original_depths, layout))
    nonexp_index = 0
    exponent_offset = 0
    for field in layout.tail_fields:
        if field.name == "frame_embed.weight":
            output.extend(frame)
        elif field.is_exponent:
            output.extend(
                exponents[exponent_offset : exponent_offset + field.count]
                .astype(np.int8)
                .tobytes()
            )
            exponent_offset += field.count
        else:
            output.extend(non_exponent[nonexp_index])
            nonexp_index += 1
    if exponent_offset != layout.exponent_count or nonexp_index != len(non_exponent):
        raise IHS2Error("IHS2 tail reconstruction accounting mismatch")
    return bytes(output)


def encode_ihs2_v2(
    raw_ihs1: bytes,
    layout: IHS2Layout,
    *,
    frame_format: int = FRAME_RAW,
    pack_exponents: bool = False,
    tighten_rows: bool = False,
    depth_codec: int = 0,
    section_order: int = 0,
) -> bytes:
    """Encode the advanced IHS2 v2 exact schema on demand.

    Kept in a companion module so the production v1 path stays small and its
    byte-level behavior cannot drift while advanced codecs are evaluated.
    """
    from .ihs2_advanced import encode_v2

    return encode_v2(
        raw_ihs1,
        layout,
        frame_format=frame_format,
        pack_exponents=pack_exponents,
        tighten_rows=tighten_rows,
        depth_codec=depth_codec,
        section_order=section_order,
    )


def encode_ihs2_v3(
    raw_ihs1: bytes,
    layout: IHS2Layout,
    *,
    frame_format: int = FRAME_RAW,
    pack_exponents: bool = False,
    tighten_rows: bool = False,
    pack_biases: bool = False,
) -> bytes:
    """Encode Gate-A exact bias packing with all learned widths charged."""
    from .ihs2_gate_a import encode_v3

    return encode_v3(
        raw_ihs1,
        layout,
        frame_format=frame_format,
        pack_exponents=pack_exponents,
        tighten_rows=tighten_rows,
        pack_biases=pack_biases,
    )


def decode_ihs2(blob: bytes, layout: IHS2Layout) -> bytes:
    """Decode every supported IHS2 version, rejecting unknown versions."""
    if len(blob) < 5:
        raise IHS2Error("truncated IHS2 header")
    if blob[:4] != IHS2_MAGIC:
        raise IHS2Error("invalid IHS2 magic")
    if blob[4] == IHS2_VERSION:
        return _decode_ihs2_v1(blob, layout)
    if blob[4] == 2:
        from .ihs2_advanced import decode_v2

        return decode_v2(blob, layout)
    if blob[4] == 3:
        from .ihs2_gate_a import decode_v3

        return decode_v3(blob, layout)
    raise IHS2Error("unsupported IHS2 version")


def materialize_ihs1(blob: bytes, runtime) -> bytes:
    """Select the unambiguous stored representation for the production path."""
    if blob.startswith(IHS1_MAGIC):
        return blob
    if blob.startswith(IHS2_MAGIC):
        return decode_ihs2(blob, layout_from_runtime(runtime))
    raise IHS2Error("unknown HPAC representation magic")
