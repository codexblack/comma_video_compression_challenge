"""Exact fixed-schema adaptive-rANS storage for CPR1 W4 renderer symbols."""

from __future__ import annotations

import struct
from math import factorial

import numpy as np

from ..baseline import SEMANTIC_SCHEMA, TensorStorage, decoded_state_sha256
from ..bits import BitPackingError, unpack_signed
from .adaptive_ans import ANSError, decode_adaptive, encode_adaptive

WANS1_MAGIC = b"WANS"
WANS1_VERSION = 1
_W4_INDEXES = tuple(
    index for index, schema in enumerate(SEMANTIC_SCHEMA) if not schema.is_fp16
)
_W4_COUNT = len(_W4_INDEXES)
_MASK_BYTES = (_W4_COUNT + 7) // 8
_PRIOR_BYTES = (_W4_COUNT * 2 + 7) // 8
_OFFSET_BYTES = 2 * (_W4_COUNT - 1)
_HEADER_BYTES = 5 + _MASK_BYTES + _PRIOR_BYTES + _OFFSET_BYTES
# F11's global encoder selected this exact fixed prefix.  F12 moves the
# information into its fixed schema so it need not be repeated in every model
# payload.  The 30-byte offset table remains payload data because the rANS
# streams have variable lengths.
F11_FIXED_PREFIX = WANS1_MAGIC + bytes((WANS1_VERSION, 0xB7, 0xFD, 0, 0, 0, 0))
F11_PREFIX_BYTES = len(F11_FIXED_PREFIX)
F12_ORDER_BYTES = 6  # ceil(log2(16!)); encodes every permutation of W4 streams.


class RendererWeightCodecError(ValueError):
    """A WANS1 renderer payload is malformed or non-canonical."""


def _require_records(records: tuple[TensorStorage, ...]) -> None:
    if (
        len(records) != len(SEMANTIC_SCHEMA)
        or tuple(record.schema for record in records) != SEMANTIC_SCHEMA
    ):
        raise RendererWeightCodecError(
            "renderer records do not match the fixed CPR1 schema"
        )
    for record in records:
        if record.schema.is_fp16:
            if record.format != "fp16" or record.raw_fp16 is None:
                raise RendererWeightCodecError(
                    f"missing exact fp16 bytes for {record.schema.name}"
                )
        elif record.format != "w4" or record.codes is None or record.raw_scales is None:
            raise RendererWeightCodecError(
                f"missing exact W4 values for {record.schema.name}"
            )


def _fixed_metadata_bytes() -> int:
    return sum(
        schema.count * 2 if schema.is_fp16 else schema.scale_count * 2
        for schema in SEMANTIC_SCHEMA
    )


def _pack_priors(priors: list[int]) -> bytes:
    if len(priors) != _W4_COUNT or any(not 0 <= prior <= 3 for prior in priors):
        raise RendererWeightCodecError("invalid WANS1 prior catalog index")
    output = bytearray(_PRIOR_BYTES)
    for index, prior in enumerate(priors):
        output[index // 4] |= prior << (2 * (index % 4))
    return bytes(output)


def _unpack_priors(raw: bytes) -> list[int]:
    if len(raw) != _PRIOR_BYTES:
        raise RendererWeightCodecError("truncated WANS1 prior table")
    return [(raw[index // 4] >> (2 * (index % 4))) & 3 for index in range(_W4_COUNT)]


def encode_wans1(
    records: tuple[TensorStorage, ...], *, strategy: str = "per_tensor"
) -> tuple[bytes, dict[str, object]]:
    """Encode exact semantic records, selecting rANS only where it is smaller.

    ``global`` selects one prior for every rANS-enabled tensor. ``per_tensor``
    independently chooses its catalog prior. Both retain the original nibble
    stream for a tensor when adaptive rANS is not smaller.
    """
    _require_records(records)
    if strategy not in {"global", "per_tensor"}:
        raise RendererWeightCodecError("unsupported WANS1 strategy")
    metadata = bytearray()
    raw_streams: list[bytes] = []
    trials: list[list[bytes]] = []
    for record in records:
        if record.schema.is_fp16:
            metadata.extend(record.raw_fp16 or b"")
            continue
        assert record.codes is not None and record.raw_scales is not None
        metadata.extend(record.raw_scales)
        raw_stream = np.asarray(record.codes, dtype=np.int8).reshape(-1)
        raw_streams.append(
            np.asarray(raw_stream).view(np.uint8).tobytes()
        )  # marker, replaced below
        symbols = (raw_stream.astype(np.int16) + 8).astype(np.uint8)
        trials.append(
            [encode_adaptive(symbols, prior_index=prior) for prior in range(4)]
        )
    # Stored raw W4 streams preserve CPR1's MSB-first nibble packing, not the
    # byte layout of the decoded int8 values.
    raw_streams = [_raw_code_stream(records[index]) for index in _W4_INDEXES]
    if strategy == "global":
        prior = min(
            range(4),
            key=lambda candidate: sum(
                min(len(raw), len(trial[candidate]))
                for raw, trial in zip(raw_streams, trials, strict=True)
            ),
        )
        selected_priors = [prior] * _W4_COUNT
    else:
        selected_priors = [
            min(range(4), key=lambda candidate: len(trial[candidate]))
            for trial in trials
        ]
    modes: list[bool] = []
    streams: list[bytes] = []
    priors: list[int] = []
    for raw, trial, prior in zip(raw_streams, trials, selected_priors, strict=True):
        encoded = trial[prior]
        enabled = len(encoded) < len(raw)
        modes.append(enabled)
        streams.append(encoded if enabled else raw)
        priors.append(prior if enabled else 0)
    mask = bytearray(_MASK_BYTES)
    for index, enabled in enumerate(modes):
        if enabled:
            mask[index // 8] |= 1 << (index % 8)
    offsets = np.cumsum([len(stream) for stream in streams], dtype=np.int64)
    if offsets.size and int(offsets[-1]) >= 1 << 16:
        raise RendererWeightCodecError("WANS1 stream area exceeds fixed u16 offsets")
    header = (
        WANS1_MAGIC
        + bytes((WANS1_VERSION,))
        + bytes(mask)
        + _pack_priors(priors)
        + b"".join(struct.pack("<H", int(value)) for value in offsets[:-1])
    )
    blob = header + bytes(metadata) + b"".join(streams)
    decoded = decode_wans1(blob)
    if decoded_state_sha256(decoded) != decoded_state_sha256(records):
        raise RendererWeightCodecError("WANS1 encoder failed semantic-state parity")
    return blob, inspect_wans1(blob)


def _raw_code_stream(record: TensorStorage) -> bytes:
    assert record.codes is not None
    # Reuse the deployed nibble representation exactly, including odd-count
    # zero padding; importing locally avoids a second bit-packing contract.
    from ..bits import pack_signed

    return pack_signed(record.codes.reshape(-1), 4)


def _stream_order_rank(order: tuple[int, ...]) -> int:
    if len(order) != _W4_COUNT or set(order) != set(range(_W4_COUNT)):
        raise RendererWeightCodecError("WANS stream order is not a permutation")
    available = list(range(_W4_COUNT))
    rank = 0
    for position, value in enumerate(order):
        index = available.index(value)
        rank += index * factorial(_W4_COUNT - position - 1)
        available.pop(index)
    return rank


def _stream_order_unrank(rank: int) -> tuple[int, ...]:
    if not 0 <= rank < factorial(_W4_COUNT):
        raise RendererWeightCodecError("WANS stream-order rank is out of range")
    available = list(range(_W4_COUNT))
    output: list[int] = []
    for position in range(_W4_COUNT):
        unit = factorial(_W4_COUNT - position - 1)
        index, rank = divmod(rank, unit)
        output.append(available.pop(index))
    return tuple(output)


def pack_f12_stream_order(order: tuple[int, ...]) -> bytes:
    """Store an arbitrary W4-stream permutation in its minimal fixed field."""
    return _stream_order_rank(order).to_bytes(F12_ORDER_BYTES, "little")


def unpack_f12_stream_order(raw: bytes) -> tuple[int, ...]:
    """Recover the exact F12 W4-stream permutation."""
    if len(raw) != F12_ORDER_BYTES:
        raise RendererWeightCodecError("invalid F12 WANS stream-order field")
    return _stream_order_unrank(int.from_bytes(raw, "little"))


def _streams_from_header(header: bytes, stream_area: bytes) -> list[bytes]:
    if len(header) != _OFFSET_BYTES:
        raise RendererWeightCodecError("invalid WANS offset table")
    ends = [
        struct.unpack_from("<H", header, 2 * item)[0] for item in range(_W4_COUNT - 1)
    ] + [len(stream_area)]
    starts = [0] + ends[:-1]
    if any(end <= start for start, end in zip(starts, ends, strict=True)):
        raise RendererWeightCodecError("non-monotonic WANS stream offsets")
    return [stream_area[start:end] for start, end in zip(starts, ends, strict=True)]


def _offsets_for_streams(streams: list[bytes]) -> bytes:
    if len(streams) != _W4_COUNT or any(not stream for stream in streams):
        raise RendererWeightCodecError("invalid WANS stream sequence")
    ends = np.cumsum([len(stream) for stream in streams], dtype=np.int64)
    if int(ends[-1]) >= 1 << 16:
        raise RendererWeightCodecError("WANS stream area exceeds fixed u16 offsets")
    return b"".join(struct.pack("<H", int(value)) for value in ends[:-1])


def encode_f12_wans_body(blob: bytes, order: tuple[int, ...]) -> bytes:
    """Remove F11-fixed WANS metadata and place streams in ``order``.

    This changes only the physical placement of independently decodable W4
    streams.  ``decode_f12_wans_body`` restores the ordinary WANS1 bytes
    before the semantic decoder is invoked.
    """
    info = inspect_wans1(blob)
    if blob[:F11_PREFIX_BYTES] != F11_FIXED_PREFIX:
        raise RendererWeightCodecError("F12 requires F11's fixed WANS modes and priors")
    if tuple(order) != unpack_f12_stream_order(pack_f12_stream_order(order)):
        raise RendererWeightCodecError("F12 WANS stream-order round trip failed")
    metadata_start = _HEADER_BYTES
    metadata_end = metadata_start + int(info["fixed_metadata_bytes"])
    streams = _streams_from_header(
        blob[F11_PREFIX_BYTES:_HEADER_BYTES], blob[metadata_end:]
    )
    ordered = [streams[index] for index in order]
    return (
        _offsets_for_streams(ordered)
        + blob[metadata_start:metadata_end]
        + b"".join(ordered)
    )


def decode_f12_wans_body(blob: bytes, order: tuple[int, ...]) -> bytes:
    """Restore canonical WANS1 bytes from an F12 stream-ordered body."""
    if len(blob) < _OFFSET_BYTES + _fixed_metadata_bytes() + _W4_COUNT:
        raise RendererWeightCodecError("truncated F12 WANS body")
    if tuple(order) != unpack_f12_stream_order(pack_f12_stream_order(order)):
        raise RendererWeightCodecError("F12 WANS stream-order round trip failed")
    offsets = blob[:_OFFSET_BYTES]
    metadata_end = _OFFSET_BYTES + _fixed_metadata_bytes()
    metadata, stream_area = blob[_OFFSET_BYTES:metadata_end], blob[metadata_end:]
    ordered = _streams_from_header(offsets, stream_area)
    streams: list[bytes | None] = [None] * _W4_COUNT
    for index, value in zip(order, ordered, strict=True):
        streams[index] = value
    if any(
        value is None for value in streams
    ):  # pragma: no cover - permutation checks above.
        raise RendererWeightCodecError("F12 WANS stream order omitted an entry")
    canonical = [value for value in streams if value is not None]
    restored = (
        F11_FIXED_PREFIX
        + _offsets_for_streams(canonical)
        + metadata
        + b"".join(canonical)
    )
    decode_wans1(restored)
    return restored


def inspect_wans1(blob: bytes) -> dict[str, object]:
    """Validate fixed framing and return charged stream accounting."""
    if len(blob) < _HEADER_BYTES or blob[:4] != WANS1_MAGIC or blob[4] != WANS1_VERSION:
        raise RendererWeightCodecError("invalid WANS1 header")
    offset = 5
    mask = blob[offset : offset + _MASK_BYTES]
    offset += _MASK_BYTES
    if _W4_COUNT % 8 and mask[-1] & ((1 << (8 - _W4_COUNT % 8)) - 1):
        raise RendererWeightCodecError("nonzero WANS1 mode-mask padding")
    priors = _unpack_priors(blob[offset : offset + _PRIOR_BYTES])
    offset += _PRIOR_BYTES
    ends = [
        struct.unpack_from("<H", blob, offset + 2 * item)[0]
        for item in range(_W4_COUNT - 1)
    ]
    offset += _OFFSET_BYTES
    metadata_bytes = _fixed_metadata_bytes()
    if len(blob) < offset + metadata_bytes:
        raise RendererWeightCodecError("truncated WANS1 metadata")
    stream_bytes = len(blob) - offset - metadata_bytes
    ends.append(stream_bytes)
    starts = [0] + ends[:-1]
    if any(end <= start for start, end in zip(starts, ends, strict=True)):
        raise RendererWeightCodecError("non-monotonic WANS1 stream offsets")
    modes = [bool(mask[item // 8] & (1 << (item % 8))) for item in range(_W4_COUNT)]
    if any(prior and not mode for prior, mode in zip(priors, modes, strict=True)):
        raise RendererWeightCodecError("WANS1 raw stream has a noncanonical prior")
    return {
        "header_bytes": _HEADER_BYTES,
        "mode_mask_bytes": _MASK_BYTES,
        "prior_table_bytes": _PRIOR_BYTES,
        "offset_table_bytes": _OFFSET_BYTES,
        "fixed_metadata_bytes": metadata_bytes,
        "stream_bytes": stream_bytes,
        "raw_stream_count": int(sum(not mode for mode in modes)),
        "ans_stream_count": int(sum(modes)),
        "modes": modes,
        "priors": priors,
        "stream_lengths": [
            end - start for start, end in zip(starts, ends, strict=True)
        ],
        "semantic_bytes": len(blob),
    }


def decode_wans1(blob: bytes) -> tuple[TensorStorage, ...]:
    """Strictly restore fixed-schema CPR1 semantic records from WANS1."""
    info = inspect_wans1(blob)
    offset = _HEADER_BYTES
    fixed = blob[offset : offset + int(info["fixed_metadata_bytes"])]
    offset += int(info["fixed_metadata_bytes"])
    streams = blob[offset:]
    metadata_offset = 0
    records: list[TensorStorage] = []
    w4_position = 0
    stream_offset = 0
    modes = list(info["modes"])
    priors = list(info["priors"])
    lengths = list(info["stream_lengths"])
    for schema in SEMANTIC_SCHEMA:
        if schema.is_fp16:
            size = schema.count * 2
            raw = fixed[metadata_offset : metadata_offset + size]
            metadata_offset += size
            if len(raw) != size:
                raise RendererWeightCodecError(f"truncated WANS1 fp16 {schema.name}")
            values = (
                np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(schema.shape)
            )
            records.append(
                TensorStorage(schema, "fp16", values, None, None, raw_fp16=raw)
            )
            continue
        scale_size = schema.scale_count * 2
        raw_scales = fixed[metadata_offset : metadata_offset + scale_size]
        metadata_offset += scale_size
        if len(raw_scales) != scale_size:
            raise RendererWeightCodecError(f"truncated WANS1 scales {schema.name}")
        stream = streams[stream_offset : stream_offset + lengths[w4_position]]
        stream_offset += lengths[w4_position]
        if len(stream) != lengths[w4_position]:
            raise RendererWeightCodecError(
                f"truncated WANS1 symbol stream {schema.name}"
            )
        if modes[w4_position]:
            try:
                symbols = decode_adaptive(
                    stream, schema.count, prior_index=priors[w4_position]
                )
            except ANSError as error:
                raise RendererWeightCodecError(
                    f"invalid WANS1 ANS stream {schema.name}: {error}"
                ) from error
            if np.any(symbols == 0):
                raise RendererWeightCodecError(
                    f"WANS1 reserved -8 symbol in {schema.name}"
                )
            codes = symbols.astype(np.int16) - 8
        else:
            expected = (schema.count + 1) // 2
            if len(stream) != expected:
                raise RendererWeightCodecError(
                    f"invalid WANS1 raw stream length {schema.name}"
                )
            try:
                codes = np.asarray(
                    unpack_signed(stream, schema.count, 4), dtype=np.int8
                )
            except BitPackingError as error:
                raise RendererWeightCodecError(
                    f"invalid WANS1 raw stream {schema.name}: {error}"
                ) from error
            if np.any(codes == -8):
                raise RendererWeightCodecError(
                    f"WANS1 reserved -8 code in {schema.name}"
                )
        scales = np.frombuffer(raw_scales, dtype="<f2").astype(np.float32)
        scale_shape = [1] * len(schema.shape)
        scale_shape[-1 if schema.name.endswith("embed.weight") else 0] = (
            schema.scale_count
        )
        codes = codes.astype(np.int8).reshape(schema.shape)
        records.append(
            TensorStorage(
                schema,
                "w4",
                codes.astype(np.float32) * scales.reshape(scale_shape),
                scales,
                codes,
                raw_scales=raw_scales,
            )
        )
        w4_position += 1
    if (
        metadata_offset != len(fixed)
        or stream_offset != len(streams)
        or w4_position != _W4_COUNT
    ):
        raise RendererWeightCodecError("WANS1 field accounting mismatch")
    return tuple(records)
