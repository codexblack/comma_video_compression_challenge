"""Charged, integer rank-one adapters for the frozen CPR1 HPAC model.

The base HPAC representation stays byte-for-byte in an archive.  ``LRA1``
wraps that representation and carries a small ``LAD1`` side payload.  Each
entry is an integer rank-one update to one already-deployed matrix:

``W' = W + round_away_from_zero(B @ A / 2**shift)``.

This is deliberately a weight-domain spelling of ``W'x = Wx + B(Ax)``.  It
means both the encoder and inflater enter the pinned integer model with the
same integer weight codes, without a float-only LoRA execution path.  The
power-of-two divisor, every signed code, and entry metadata are on wire.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from .bits import BitPackingError, pack_signed, packed_length, unpack_signed

LORA_MAGIC = b"LRA1"
ADAPTER_MAGIC = b"LAD1"
LORA_VERSION = 1

# These are structural constants of the pinned public CPR1 IntegerHPAC, not
# learned values.  Keeping them fixed prevents learned dimensions/names from
# being smuggled in as uncharged metadata.
PLACEMENT_SPECS: dict[str, tuple[int, int, int]] = {
    "head": (1, 5, 64),
    "conv_past": (2, 64, 45),
    "frame_shift": (3, 64, 8),
}
PLACEMENT_NAMES = {
    identifier: name for name, (identifier, _, _) in PLACEMENT_SPECS.items()
}


class LoRAError(ValueError):
    """An adapter payload is malformed or cannot be applied exactly."""


@dataclass(frozen=True)
class IntegerRankOne:
    """One fully charged integer rank-one update."""

    placement: str
    bits: int
    shift: int
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if self.placement not in PLACEMENT_SPECS:
            raise LoRAError(f"unknown LoRA placement {self.placement!r}")
        if self.bits not in (4, 6, 8):
            raise LoRAError("LoRA code precision must be int4, int6, or int8")
        if not 0 <= self.shift <= 31:
            raise LoRAError("LoRA power-of-two shift must be in 0..31")
        _, out_features, in_features = PLACEMENT_SPECS[self.placement]
        a = np.asarray(self.a)
        b = np.asarray(self.b)
        if a.shape != (1, in_features) or b.shape != (out_features, 1):
            raise LoRAError(
                f"{self.placement} rank-one shapes must be (1, {in_features}) and ({out_features}, 1)"
            )
        if a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
            raise LoRAError("LoRA codes must be integer arrays")
        limit = 1 << (self.bits - 1)
        if (
            np.any(a < -limit)
            or np.any(a >= limit)
            or np.any(b < -limit)
            or np.any(b >= limit)
        ):
            raise LoRAError("LoRA code outside declared signed precision")

    @property
    def codes(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(self.a, dtype=np.int16).reshape(-1),
                np.asarray(self.b, dtype=np.int16).reshape(-1),
            )
        )

    @property
    def raw_code_bytes(self) -> int:
        return packed_length(self.codes.size, self.bits)


@dataclass(frozen=True)
class IntegerLoRA:
    """A nonempty set of unique rank-one updates."""

    entries: tuple[IntegerRankOne, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise LoRAError("an adapter must have at least one rank-one entry")
        names = [entry.placement for entry in self.entries]
        if len(names) != len(set(names)):
            raise LoRAError("duplicate LoRA placement")

    @property
    def raw_code_bytes(self) -> int:
        return sum(entry.raw_code_bytes for entry in self.entries)


def round_shift_away_from_zero(values: np.ndarray, shift: int) -> np.ndarray:
    """Round signed integers after division by ``2**shift`` deterministically."""
    values = np.asarray(values, dtype=np.int64)
    if not 0 <= shift <= 31:
        raise LoRAError("invalid integer requantization shift")
    if shift == 0:
        return values.copy()
    magnitude = np.abs(values)
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return np.where(values < 0, -rounded, rounded)


def effective_delta(entry: IntegerRankOne) -> np.ndarray:
    """Materialize the exact integer matrix added to the frozen weight codes."""
    product = np.asarray(entry.b, dtype=np.int64) @ np.asarray(entry.a, dtype=np.int64)
    result = round_shift_away_from_zero(product, entry.shift)
    if np.any(result < np.iinfo(np.int16).min) or np.any(
        result > np.iinfo(np.int16).max
    ):
        raise LoRAError("LoRA accumulation exceeds int16 materialization range")
    return result.astype(np.int16)


def serialize_adapter(adapter: IntegerLoRA) -> bytes:
    """Serialize every learned adapter value and reject ambiguous layouts."""
    output = bytearray(ADAPTER_MAGIC + bytes((LORA_VERSION, len(adapter.entries))))
    for entry in adapter.entries:
        identifier, _, _ = PLACEMENT_SPECS[entry.placement]
        output.extend(bytes((identifier, entry.bits, entry.shift, 0)))
        output.extend(pack_signed(entry.codes, entry.bits))
    return bytes(output)


def deserialize_adapter(blob: bytes) -> IntegerLoRA:
    """Strictly decode one complete ``LAD1`` adapter payload."""
    if len(blob) < 6 or blob[:4] != ADAPTER_MAGIC:
        raise LoRAError("invalid adapter magic or truncated header")
    version, count = blob[4], blob[5]
    if version != LORA_VERSION or not count or count > len(PLACEMENT_SPECS):
        raise LoRAError("unsupported adapter version or entry count")
    offset = 6
    entries: list[IntegerRankOne] = []
    seen: set[str] = set()
    for _ in range(count):
        if offset + 4 > len(blob):
            raise LoRAError("truncated adapter entry header")
        identifier, bits, shift, reserved = blob[offset : offset + 4]
        offset += 4
        placement = PLACEMENT_NAMES.get(identifier)
        if (
            placement is None
            or placement in seen
            or bits not in (4, 6, 8)
            or shift > 31
            or reserved
        ):
            raise LoRAError("invalid, duplicate, or reserved adapter entry")
        _, out_features, in_features = PLACEMENT_SPECS[placement]
        count_codes = out_features + in_features
        size = packed_length(count_codes, bits)
        if offset + size > len(blob):
            raise LoRAError("truncated adapter codes")
        try:
            codes = np.asarray(
                unpack_signed(blob[offset : offset + size], count_codes, bits),
                dtype=np.int8,
            )
        except BitPackingError as error:
            raise LoRAError(f"invalid packed adapter codes: {error}") from error
        offset += size
        entries.append(
            IntegerRankOne(
                placement,
                bits,
                shift,
                codes[:in_features].reshape(1, in_features),
                codes[in_features:].reshape(out_features, 1),
            )
        )
        seen.add(placement)
    if offset != len(blob):
        raise LoRAError("adapter has trailing data")
    return IntegerLoRA(tuple(entries))


def wrap_hpac(base_representation: bytes, adapter: IntegerLoRA) -> bytes:
    """Put a charged adapter beside an arbitrary IHS1/IHS2 base representation."""
    if not base_representation:
        raise LoRAError("cannot wrap an empty HPAC representation")
    if len(base_representation) >= 1 << 32:
        raise LoRAError("base HPAC representation is too large")
    return (
        LORA_MAGIC
        + struct.pack("<I", len(base_representation))
        + base_representation
        + serialize_adapter(adapter)
    )


def unwrap_hpac(blob: bytes) -> tuple[bytes, IntegerLoRA | None, bytes]:
    """Return base representation, parsed adapter, and charged adapter bytes."""
    if not blob.startswith(LORA_MAGIC):
        return blob, None, b""
    if len(blob) < 8:
        raise LoRAError("truncated LRA1 wrapper")
    base_bytes = struct.unpack_from("<I", blob, 4)[0]
    start, end = 8, 8 + base_bytes
    if not base_bytes or end >= len(blob):
        raise LoRAError("invalid LRA1 base representation length")
    adapter_blob = blob[end:]
    return blob[start:end], deserialize_adapter(adapter_blob), adapter_blob


def adapter_sha256(adapter: IntegerLoRA | None) -> str:
    return (
        ""
        if adapter is None
        else hashlib.sha256(serialize_adapter(adapter)).hexdigest()
    )


def adapter_report(blob: bytes) -> dict[str, object]:
    """Expose only charged accounting for reports and candidate registries."""
    base, adapter, adapter_blob = unwrap_hpac(blob)
    return {
        "base_representation_bytes": len(base),
        "adapter_bytes": len(adapter_blob),
        "adapter_raw_code_bytes": 0 if adapter is None else adapter.raw_code_bytes,
        "adapter_sha256": adapter_sha256(adapter),
        "adapter_entries": []
        if adapter is None
        else [
            {
                "placement": entry.placement,
                "rank": 1,
                "bits": entry.bits,
                "shift": entry.shift,
                "raw_code_bytes": entry.raw_code_bytes,
            }
            for entry in adapter.entries
        ],
    }


def _module_for_placement(model, placement: str):
    try:
        return getattr(model, placement)
    except AttributeError as error:
        raise LoRAError(
            f"deployed model does not expose placement {placement}"
        ) from error


def apply_adapter(model, adapter: IntegerLoRA | None) -> None:
    """Apply exact integer deltas to an in-memory model without touching its blob.

    ``codes()`` defines the deployed integer state.  We start from those codes,
    check the explicit convolution/linear bounds, and replace only the working
    model parameter that the pinned integer path subsequently rounds.  The
    caller retains the original model bytes and can independently hash them.
    """
    if adapter is None:
        return
    import torch

    with torch.no_grad():
        for entry in adapter.entries:
            module = _module_for_placement(model, entry.placement)
            weight, _, _ = module.codes()
            flat = weight.detach().round().to(torch.int64).reshape(weight.shape[0], -1)
            expected_out, expected_in = effective_delta(entry).shape
            if tuple(flat.shape) != (expected_out, expected_in):
                raise LoRAError(
                    f"deployed {entry.placement} weight shape is incompatible with LRA1"
                )
            delta = torch.from_numpy(effective_delta(entry)).to(
                device=flat.device, dtype=torch.int64
            )
            updated = flat + delta
            bound = int(module.weight_bound)
            if bool((updated < -bound).any()) or bool((updated > bound).any()):
                raise LoRAError(
                    f"{entry.placement} LoRA update exceeds declared integer weight bounds"
                )
            module.weight.copy_(
                updated.reshape_as(module.weight).to(dtype=module.weight.dtype)
            )


def materialize_hpac(blob: bytes, runtime) -> tuple[bytes, IntegerLoRA | None]:
    """Decode IHS storage and its optional adapter, retaining the base separately."""
    from .ihs2 import materialize_ihs1

    base, adapter, _ = unwrap_hpac(blob)
    return materialize_ihs1(base, runtime), adapter
