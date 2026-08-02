"""Deterministic Phase-2D archives for a frozen HPAC residual table.

The baseline model section is unchanged.  A residual candidate replaces only
the token section with ``[table][range-stream]``; the generic table is RCL1
and the compact production table uses the fixed RCF1 schema.
"""

from __future__ import annotations

import hashlib
import io
import lzma
import os
import struct
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baseline import BaselinePayload, encode_legacy_w4, sha256_bytes
from .bits import BitPackingError, pack_signed, packed_length, unpack_signed
from .carrier_repack import (
    decode_cvh1,
    decode_cvq1,
    pack_frame0_selector_carrier,
    restore_fixed_cps3,
    split_frame0_selector_carrier,
    strip_fixed_cps3,
)
from .entropy.coefficient_ar1_codec import (
    CAP1_MAGIC,
    CoefficientAr1CodecError,
    decode_cap1,
)
from .entropy.renderer_weight_codec import (
    F12_ORDER_BYTES,
    RendererWeightCodecError,
    decode_f12_wans_body,
    decode_wans1,
    encode_f12_wans_body,
    pack_f12_stream_order,
    unpack_f12_stream_order,
)
from .entropy_audit import boundary_buckets
from .frame0_selector import MAGIC as FRAME0_SELECTOR_MAGIC
from .frame0_selector import SPARSE_MAGIC as FRAME0_SPARSE_SELECTOR_MAGIC
from .frame0_selector import SPARSE_VERSION as FRAME0_SPARSE_SELECTOR_VERSION
from .frame0_selector import VERSION as FRAME0_SELECTOR_VERSION
from .frame0_selector import decode_selector
from .residual_calibration import (
    NUM_CLASSES,
    QuantizedTable,
    deserialize_tables,
    margin_buckets,
    rcl1_payload_length,
    serialize_tables,
)


class ResidualArchiveError(ValueError):
    """A Phase-2D residual archive is malformed or internally inconsistent."""


RCL1_SCHEMA = "rcl1"
FIXED_SCHEMA = "fixed_boundary_int6"
BASELINE_SCHEMA = "baseline"
FIXED_MAGIC = b"RCF1"
FIXED_STATES = 25
FIXED_BITS = 6
FIXED_MODEL_MAGIC = b"M2F1"
COMPACT_MODEL_MAGIC = b"M3F1"
FIXED_SEMANTIC_BYTES = 40_252
FIXED_CARRIER_BYTES = 23_054
FIXED_CPS3_CARRIER_BYTES = 23_056
FIXED_COMPACT_CPS3_BYTES = 23_046
FIXED_IHS2_BYTES = 16_599
FIXED_IHS2_HEADER = b"IHS2\x03\x31"
FIXED_IHS2_BODY_BYTES = FIXED_IHS2_BYTES - len(FIXED_IHS2_HEADER)
FIXED_SELECTOR_MAGIC = b"F7S1"
FIXED_VQ_MAGIC = b"F8V1"
FIXED_WANS_MAGIC = b"F9W1"
FIXED_AR1_MAGIC = b"F10C"
FIXED_WANS_AR1_MAGIC = b"F11A"
FIXED_WANS_AR1_COMPACT_MAGIC = b"F12A"
FIXED_WANS_AR1_PINNED_MAGIC = b"F13A"
FIXED_WANS_AR1_RATE_MAGIC = b"F14R"
FIXED_WANS_AR1_RC64_MAGIC = b"F16R"
FIXED_WANS_AR1_RC64_SELECTOR_MAGIC = b"F21S"
FIXED_WANS_AR1_RC64_SPARSE_SELECTOR_MAGIC = b"F24S"
F12_MAGIC_PREFIX = b"F12"
XZ_MAGIC = b"\xfd7zXZ\x00"
F12_CAP1_PREFIX = CAP1_MAGIC + bytes((1, 0, 0, 0))
F12_SELECTOR_PREFIX = FRAME0_SELECTOR_MAGIC + bytes((FRAME0_SELECTOR_VERSION,))
F24_SPARSE_SELECTOR_PREFIX = FRAME0_SPARSE_SELECTOR_MAGIC + bytes(
    (FRAME0_SPARSE_SELECTOR_VERSION,)
)
_F12_CAP_FIELDS = ("predictor", "scales", "lengths", "ks", "basis", "rice")


def _f12_cap_orders() -> tuple[tuple[str, ...], ...]:
    canonical = _F12_CAP_FIELDS
    orders = [canonical]
    for left in range(len(canonical)):
        for right in range(left + 1, len(canonical)):
            swapped = list(canonical)
            swapped[left], swapped[right] = swapped[right], swapped[left]
            orders.append(tuple(swapped))
    return tuple(orders)


F12_CAP_FIELD_ORDERS = _f12_cap_orders()
# The F13 schema is a single lossless layout selected from F12's measured
# order search.  Because it is schema, not candidate metadata, its semantic
# stream rank and byte length are derived rather than stored.
F13_WANS_STREAM_ORDER = (1, 15, 4, 0, 11, 5, 2, 9, 3, 6, 10, 12, 14, 7, 8, 13)
F13_CAP_FIELD_ORDER = 1
F13_WANS_BODY_BYTES = 36_040
FIXED_PRODUCTION_ORDERS = (
    ("semantic", "carrier", "hpac"),
    ("semantic", "hpac", "carrier"),
    ("carrier", "semantic", "hpac"),
    ("carrier", "hpac", "semantic"),
    ("hpac", "semantic", "carrier"),
    ("hpac", "carrier", "semantic"),
)
LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


@dataclass(frozen=True)
class ResidualArchiveParts:
    semantic_blob: bytes
    carrier_blob: bytes
    hpac_blob: bytes
    token_stream: bytes
    table: QuantizedTable | None
    schema: str
    residual_payload: bytes
    compressed_models: bytes
    token_codec: str = "range32"


def _require_fixed_table(table: QuantizedTable) -> None:
    if (
        table.name != "boundary_predicted"
        or table.bits != FIXED_BITS
        or table.codes.shape != (FIXED_STATES, NUM_CLASSES)
    ):
        raise ResidualArchiveError(
            "fixed residual schema requires one 25x5 int6 boundary_predicted table"
        )
    if not np.isfinite(table.scale) or table.scale <= 0.0:
        raise ResidualArchiveError("fixed residual table scale is invalid")


def serialize_fixed_boundary_int6(table: QuantizedTable) -> bytes:
    """Serialize learned data while omitting fixed-schema metadata."""
    _require_fixed_table(table)
    return (
        FIXED_MAGIC
        + np.asarray([table.scale], dtype="<f2").tobytes()
        + pack_signed(table.codes.reshape(-1), FIXED_BITS)
    )


def deserialize_fixed_boundary_int6(blob: bytes) -> QuantizedTable:
    expected = (
        len(FIXED_MAGIC) + 2 + packed_length(FIXED_STATES * NUM_CLASSES, FIXED_BITS)
    )
    if len(blob) != expected or not blob.startswith(FIXED_MAGIC):
        raise ResidualArchiveError("invalid fixed residual payload length or magic")
    scale = float(np.frombuffer(blob[4:6], dtype="<f2")[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ResidualArchiveError("invalid fixed residual scale")
    try:
        codes = np.asarray(
            unpack_signed(blob[6:], FIXED_STATES * NUM_CLASSES, FIXED_BITS),
            dtype=np.int8,
        ).reshape(FIXED_STATES, NUM_CLASSES)
    except BitPackingError as error:
        raise ResidualArchiveError(f"invalid fixed residual codes: {error}") from error
    return QuantizedTable(
        "boundary_predicted",
        FIXED_BITS,
        codes,
        scale,
        codes.astype(np.float32) * scale,
    )


def _fixed_production_magic(order: int) -> bytes:
    if not 0 <= order < len(FIXED_PRODUCTION_ORDERS):
        raise ResidualArchiveError("fixed production model order is invalid")
    return b"F6A" + bytes((ord("0") + order,))


def _fixed_production_order(magic: bytes) -> int | None:
    if (
        len(magic) != 4
        or not magic.startswith(b"F6A")
        or not ord("0") <= magic[3] < ord("0") + len(FIXED_PRODUCTION_ORDERS)
    ):
        return None
    return magic[3] - ord("0")


def _f12_magic(cap_order: int) -> bytes:
    if not 0 <= cap_order < len(F12_CAP_FIELD_ORDERS):
        raise ResidualArchiveError("fixed F12 CAP1 field order is invalid")
    return F12_MAGIC_PREFIX + bytes((ord("A") + cap_order,))


def _f12_cap_order(magic: bytes) -> int | None:
    if (
        len(magic) != 4
        or not magic.startswith(F12_MAGIC_PREFIX)
        or not ord("A") <= magic[3] < ord("A") + len(F12_CAP_FIELD_ORDERS)
    ):
        return None
    return magic[3] - ord("A")


def _f12_cap1_body_bytes(raw: bytes) -> int:
    """Return the fixed-schema CAP1 body length excluding its 8-byte prefix."""
    if len(raw) < 6:
        raise ResidualArchiveError("truncated F12 CAP1 bit counts")
    basis_bits = int.from_bytes(raw[:3], "little")
    residual_bits = int.from_bytes(raw[3:6], "little")
    if not basis_bits or not residual_bits:
        raise ResidualArchiveError("invalid F12 CAP1 bit counts")
    # CAP1's remaining fixed fields are AR metadata (12 * 3), scales (12 *
    # 8), basis lengths (32), and Rice parameters (12).  The bit counts make
    # both entropy streams self-delimiting, so no outer carrier length is due.
    return 6 + 36 + 96 + 32 + 12 + (basis_bits + 7) // 8 + (residual_bits + 7) // 8


def _split_f12_cap1_body(raw: bytes, cap_order: int) -> tuple[bytes, dict[str, bytes]]:
    size = _f12_cap1_body_bytes(raw)
    if len(raw) != size:
        raise ResidualArchiveError("invalid F12 CAP1 body length")
    basis_bits = int.from_bytes(raw[:3], "little")
    residual_bits = int.from_bytes(raw[3:6], "little")
    sizes = {
        "predictor": 36,
        "scales": 96,
        "lengths": 32,
        "ks": 12,
        "basis": (basis_bits + 7) // 8,
        "rice": (residual_bits + 7) // 8,
    }
    offset = 6
    fields: dict[str, bytes] = {}
    for name in F12_CAP_FIELD_ORDERS[cap_order]:
        size = sizes[name]
        fields[name] = raw[offset : offset + size]
        offset += size
    if offset != len(raw):  # pragma: no cover - guarded by the exact sum above.
        raise ResidualArchiveError("F12 CAP1 field accounting mismatch")
    return raw[:6], fields


def _encode_f12_cap1_body(cap1: bytes, cap_order: int) -> bytes:
    if not cap1.startswith(F12_CAP1_PREFIX):
        raise ResidualArchiveError("F12 CAP1 prefix is invalid")
    raw = cap1[len(F12_CAP1_PREFIX) :]
    counts, fields = _split_f12_cap1_body(raw, 0)
    return counts + b"".join(fields[name] for name in F12_CAP_FIELD_ORDERS[cap_order])


def _decode_f12_cap1_body(raw: bytes, cap_order: int) -> bytes:
    counts, fields = _split_f12_cap1_body(raw, cap_order)
    return F12_CAP1_PREFIX + counts + b"".join(fields[name] for name in _F12_CAP_FIELDS)


def _models(
    payload: BaselinePayload,
    hpac_blob: bytes | None = None,
    *,
    semantic_blob: bytes | None = None,
    carrier_blob: bytes | None = None,
    fixed_model_lengths: bool = False,
    fixed_schema_order: int | None = None,
    fixed_selector_schema: bool = False,
    fixed_vq_schema: bool = False,
    fixed_wans_schema: bool = False,
    fixed_ar1_schema: bool = False,
    fixed_wans_ar1_schema: bool = False,
    fixed_wans_ar1_compact_schema: bool = False,
    fixed_wans_ar1_pinned_schema: bool = False,
    fixed_wans_ar1_rate_schema: bool = False,
    fixed_wans_ar1_rc64_schema: bool = False,
    f12_wans_stream_order: tuple[int, ...] | None = None,
    f12_cap_field_order: int = 0,
) -> bytes:
    """Assemble model bytes while allowing an exact IHS2 HPAC representation."""
    hpac = payload.hpac_blob if hpac_blob is None else hpac_blob
    semantic = payload.semantic_blob if semantic_blob is None else semantic_blob
    carrier = payload.carrier_blob if carrier_blob is None else carrier_blob
    if not hpac:
        raise ResidualArchiveError("HPAC representation cannot be empty")
    if fixed_wans_ar1_rate_schema or fixed_wans_ar1_rc64_schema:
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError(
                "fixed F14 rate-only schema requires frozen IHS2 HPAC"
            )
        base_carrier, selector = split_frame0_selector_carrier(carrier)
        if not base_carrier.startswith(F12_CAP1_PREFIX):
            raise ResidualArchiveError("fixed F14/F16 schema requires a CAP1 carrier")
        if fixed_wans_ar1_rate_schema and selector is not None:
            raise ResidualArchiveError(
                "fixed F14 rate-only schema requires a selector-free carrier"
            )
        if selector is not None and not selector.startswith(
            (
                F12_SELECTOR_PREFIX,
                F24_SPARSE_SELECTOR_PREFIX,
            )
        ):
            raise ResidualArchiveError(
                "fixed RC64 selector schema requires an F0S1 or F0E1 selector"
            )
        try:
            semantic_body = encode_f12_wans_body(semantic, F13_WANS_STREAM_ORDER)
            if len(semantic_body) != F13_WANS_BODY_BYTES:
                raise ResidualArchiveError("fixed F14 semantic body length changed")
            if encode_legacy_w4(decode_wans1(semantic)) != payload.semantic_blob:
                raise ResidualArchiveError(
                    "fixed F14 semantic data does not restore deployed renderer bytes"
                )
            decode_cap1(base_carrier, frames=600, dimensions=12)
            if selector is not None:
                decode_selector(selector)
        except (
            RendererWeightCodecError,
            CoefficientAr1CodecError,
            ValueError,
        ) as error:
            raise ResidualArchiveError(
                f"invalid fixed F14 rate-only payload: {error}"
            ) from error
        return (
            (
                FIXED_WANS_AR1_RC64_SPARSE_SELECTOR_MAGIC
                if selector is not None
                and selector.startswith(F24_SPARSE_SELECTOR_PREFIX)
                else FIXED_WANS_AR1_RC64_SELECTOR_MAGIC
                if selector is not None
                else FIXED_WANS_AR1_RC64_MAGIC
                if fixed_wans_ar1_rc64_schema
                else FIXED_WANS_AR1_RATE_MAGIC
            )
            + hpac[len(FIXED_IHS2_HEADER) :]
            + semantic_body
            + _encode_f12_cap1_body(base_carrier, F13_CAP_FIELD_ORDER)
            + (
                selector[
                    len(F24_SPARSE_SELECTOR_PREFIX)
                    if selector.startswith(F24_SPARSE_SELECTOR_PREFIX)
                    else len(F12_SELECTOR_PREFIX) :
                ]
                if selector is not None
                else b""
            )
        )
    if fixed_wans_ar1_pinned_schema:
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError("fixed F13 schema requires frozen IHS2 HPAC")
        base_carrier, selector = split_frame0_selector_carrier(carrier)
        if (
            selector is None
            or not base_carrier.startswith(F12_CAP1_PREFIX)
            or not selector.startswith(F12_SELECTOR_PREFIX)
        ):
            raise ResidualArchiveError(
                "fixed F13 schema requires F11 CAP1 carrier and selector"
            )
        try:
            semantic_body = encode_f12_wans_body(semantic, F13_WANS_STREAM_ORDER)
            if len(semantic_body) != F13_WANS_BODY_BYTES:
                raise ResidualArchiveError("fixed F13 semantic body length changed")
            if encode_legacy_w4(decode_wans1(semantic)) != payload.semantic_blob:
                raise ResidualArchiveError(
                    "fixed F13 semantic data does not restore deployed renderer bytes"
                )
            decode_cap1(base_carrier, frames=600, dimensions=12)
            decode_selector(selector)
        except (
            RendererWeightCodecError,
            CoefficientAr1CodecError,
            ValueError,
        ) as error:
            raise ResidualArchiveError(f"invalid fixed F13 payload: {error}") from error
        return (
            FIXED_WANS_AR1_PINNED_MAGIC
            + hpac[len(FIXED_IHS2_HEADER) :]
            + semantic_body
            + _encode_f12_cap1_body(base_carrier, F13_CAP_FIELD_ORDER)
            + selector[len(F12_SELECTOR_PREFIX) :]
        )
    if fixed_wans_ar1_compact_schema:
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError("fixed F12 schema requires frozen IHS2 HPAC")
        base_carrier, selector = split_frame0_selector_carrier(carrier)
        if (
            selector is None
            or not base_carrier.startswith(F12_CAP1_PREFIX)
            or not selector.startswith(F12_SELECTOR_PREFIX)
        ):
            raise ResidualArchiveError(
                "fixed F12 schema requires F11 CAP1 carrier and selector"
            )
        order = (
            tuple(range(16))
            if f12_wans_stream_order is None
            else tuple(f12_wans_stream_order)
        )
        try:
            semantic_body = encode_f12_wans_body(semantic, order)
            if encode_legacy_w4(decode_wans1(semantic)) != payload.semantic_blob:
                raise ResidualArchiveError(
                    "fixed F12 semantic data does not restore deployed renderer bytes"
                )
            decode_cap1(base_carrier, frames=600, dimensions=12)
            decode_selector(selector)
        except (
            RendererWeightCodecError,
            CoefficientAr1CodecError,
            ValueError,
        ) as error:
            raise ResidualArchiveError(f"invalid fixed F12 payload: {error}") from error
        if len(semantic_body) >= 1 << 16:
            raise ResidualArchiveError("fixed F12 semantic payload exceeds u16 length")
        return (
            _f12_magic(f12_cap_field_order)
            + struct.pack("<H", len(semantic_body))
            + pack_f12_stream_order(order)
            + hpac[len(FIXED_IHS2_HEADER) :]
            + semantic_body
            + _encode_f12_cap1_body(base_carrier, f12_cap_field_order)
            + selector[len(F12_SELECTOR_PREFIX) :]
        )
    if fixed_wans_ar1_schema:
        if len(semantic) >= 1 << 16:
            raise ResidualArchiveError(
                "fixed WANS+AR1 semantic payload exceeds u16 length"
            )
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError(
                "fixed WANS+AR1 schema requires frozen IHS2 HPAC"
            )
        base_carrier, selector = split_frame0_selector_carrier(carrier)
        if (
            selector is None
            or not base_carrier.startswith(CAP1_MAGIC)
            or len(base_carrier) >= 1 << 16
            or len(selector) >= 1 << 16
        ):
            raise ResidualArchiveError(
                "fixed WANS+AR1 schema requires CAP1 carrier and P9 selector"
            )
        try:
            if encode_legacy_w4(decode_wans1(semantic)) != payload.semantic_blob:
                raise ResidualArchiveError(
                    "fixed WANS+AR1 semantic data does not restore P9 renderer bytes"
                )
            decode_cap1(base_carrier, frames=600, dimensions=12)
        except (RendererWeightCodecError, CoefficientAr1CodecError) as error:
            raise ResidualArchiveError(
                f"invalid fixed WANS+AR1 payload: {error}"
            ) from error
        return (
            FIXED_WANS_AR1_MAGIC
            + struct.pack("<HHH", len(semantic), len(base_carrier), len(selector))
            + hpac[len(FIXED_IHS2_HEADER) :]
            + semantic
            + base_carrier
            + selector
        )
    if fixed_ar1_schema:
        if len(semantic) != FIXED_SEMANTIC_BYTES:
            raise ResidualArchiveError(
                "fixed AR1 schema does not match frozen semantic length"
            )
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError(
                "fixed AR1 schema requires the frozen IHS2 HPAC representation"
            )
        base_carrier, selector = split_frame0_selector_carrier(carrier)
        if (
            selector is None
            or not base_carrier.startswith(CAP1_MAGIC)
            or len(base_carrier) >= 1 << 16
            or len(selector) >= 1 << 16
        ):
            raise ResidualArchiveError(
                "fixed AR1 schema requires a CAP1 carrier and P9 selector"
            )
        try:
            decode_cap1(base_carrier, frames=600, dimensions=12)
        except CoefficientAr1CodecError as error:
            raise ResidualArchiveError(f"invalid fixed AR1 carrier: {error}") from error
        return (
            FIXED_AR1_MAGIC
            + struct.pack("<HH", len(base_carrier), len(selector))
            + hpac[len(FIXED_IHS2_HEADER) :]
            + semantic
            + base_carrier
            + selector
        )
    if fixed_wans_schema:
        if len(semantic) >= 1 << 16:
            raise ResidualArchiveError(
                "fixed WANS schema semantic payload exceeds u16 length"
            )
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError(
                "fixed WANS schema requires the frozen IHS2 HPAC representation"
            )
        base_carrier, selector = split_frame0_selector_carrier(carrier)
        if (
            selector is None
            or len(base_carrier) != FIXED_CPS3_CARRIER_BYTES
            or len(selector) >= 1 << 16
        ):
            raise ResidualArchiveError(
                "fixed WANS schema requires P9's CPS3 carrier and selector"
            )
        compact_carrier = strip_fixed_cps3(base_carrier, frames=600, dimensions=12)
        if len(compact_carrier) != FIXED_COMPACT_CPS3_BYTES:
            raise ResidualArchiveError("fixed WANS CPS3 body length is unexpected")
        try:
            if encode_legacy_w4(decode_wans1(semantic)) != payload.semantic_blob:
                raise ResidualArchiveError(
                    "fixed WANS semantic data does not restore P9 renderer bytes"
                )
        except RendererWeightCodecError as error:
            raise ResidualArchiveError(
                f"invalid fixed WANS semantic payload: {error}"
            ) from error
        return (
            FIXED_WANS_MAGIC
            + struct.pack("<HH", len(semantic), len(selector))
            + hpac[len(FIXED_IHS2_HEADER) :]
            + semantic
            + compact_carrier
            + selector
        )
    if fixed_vq_schema:
        if len(semantic) != FIXED_SEMANTIC_BYTES:
            raise ResidualArchiveError(
                "fixed VQ schema does not match frozen semantic length"
            )
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError(
                "fixed VQ schema requires the frozen IHS2 HPAC representation"
            )
        try:
            (decode_cvq1 if carrier.startswith(b"CVQ1") else decode_cvh1)(
                carrier, frames=600, dimensions=12
            )
        except ValueError as error:
            raise ResidualArchiveError(f"invalid fixed VQ carrier: {error}") from error
        # F8V1 keeps the proven F6 field order.  CVQ1 is self-delimiting in
        # this fixed container because it occupies the remaining model bytes.
        return FIXED_VQ_MAGIC + hpac[len(FIXED_IHS2_HEADER) :] + semantic + carrier
    if fixed_selector_schema:
        if len(semantic) != FIXED_SEMANTIC_BYTES:
            raise ResidualArchiveError(
                "fixed selector schema does not match frozen semantic length"
            )
        base_carrier, selector = split_frame0_selector_carrier(carrier)
        if selector is None or len(base_carrier) != FIXED_CPS3_CARRIER_BYTES:
            raise ResidualArchiveError(
                "fixed selector schema requires one CPS3 carrier plus a "
                "selector payload"
            )
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError(
                "fixed selector schema requires the frozen IHS2 HPAC representation"
            )
        if len(selector) >= 1 << 16:
            raise ResidualArchiveError("fixed selector payload exceeds u16 length")
        compact_carrier = strip_fixed_cps3(base_carrier, frames=600, dimensions=12)
        if len(compact_carrier) != FIXED_COMPACT_CPS3_BYTES:
            raise ResidualArchiveError("fixed selector CPS3 body length is unexpected")
        # F7S1 keeps F6's empirically best HPAC/semantic/carrier ordering and
        # stores the selector immediately after its carrier, its only consumer.
        return (
            FIXED_SELECTOR_MAGIC
            + struct.pack("<H", len(selector))
            + hpac[len(FIXED_IHS2_HEADER) :]
            + semantic
            + compact_carrier
            + selector
        )
    if fixed_schema_order is not None:
        if len(semantic) != FIXED_SEMANTIC_BYTES:
            raise ResidualArchiveError(
                "fixed production schema does not match frozen semantic length"
            )
        if len(carrier) != FIXED_CPS3_CARRIER_BYTES:
            raise ResidualArchiveError(
                "fixed production schema requires the frozen 23,056-byte CPS3 carrier"
            )
        if len(hpac) != FIXED_IHS2_BYTES or not hpac.startswith(FIXED_IHS2_HEADER):
            raise ResidualArchiveError(
                "fixed production schema requires the frozen IHS2 HPAC representation"
            )
        compact_carrier = strip_fixed_cps3(carrier, frames=600, dimensions=12)
        if len(compact_carrier) != FIXED_COMPACT_CPS3_BYTES:
            raise ResidualArchiveError(
                "fixed production CPS3 body length is unexpected"
            )
        fields = {
            "semantic": semantic,
            "carrier": compact_carrier,
            "hpac": hpac[len(FIXED_IHS2_HEADER) :],
        }
        return _fixed_production_magic(fixed_schema_order) + b"".join(
            fields[name] for name in FIXED_PRODUCTION_ORDERS[fixed_schema_order]
        )
    if fixed_model_lengths:
        if len(semantic) != FIXED_SEMANTIC_BYTES:
            raise ResidualArchiveError(
                "fixed model schema does not match frozen semantic length"
            )
        if len(carrier) == FIXED_CARRIER_BYTES:
            return FIXED_MODEL_MAGIC + semantic + carrier + hpac
        if len(carrier) >= 1 << 16:
            raise ResidualArchiveError(
                "compact model schema carrier length exceeds u16"
            )
        return (
            COMPACT_MODEL_MAGIC
            + struct.pack("<H", len(carrier))
            + semantic
            + carrier
            + hpac
        )
    return struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier + hpac


def _split_models(models: bytes) -> tuple[bytes, bytes, bytes, bool]:
    if models.startswith(
        (
            FIXED_WANS_AR1_RATE_MAGIC,
            FIXED_WANS_AR1_RC64_MAGIC,
            FIXED_WANS_AR1_RC64_SELECTOR_MAGIC,
            FIXED_WANS_AR1_RC64_SPARSE_SELECTOR_MAGIC,
        )
    ):
        has_sparse_selector = models.startswith(
            FIXED_WANS_AR1_RC64_SPARSE_SELECTOR_MAGIC
        )
        has_selector = has_sparse_selector or models.startswith(
            FIXED_WANS_AR1_RC64_SELECTOR_MAGIC
        )
        minimum = (
            4 + FIXED_IHS2_BODY_BYTES + F13_WANS_BODY_BYTES + 6 + int(has_selector)
        )
        if len(models) < minimum:
            raise ResidualArchiveError(
                "truncated fixed F14/F16 rate-only model payload"
            )
        offset = 4
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic_end = offset + F13_WANS_BODY_BYTES
        try:
            semantic = decode_f12_wans_body(
                models[offset:semantic_end], F13_WANS_STREAM_ORDER
            )
            decode_wans1(semantic)
        except RendererWeightCodecError as error:
            raise ResidualArchiveError(
                f"invalid fixed F14/F16 semantic payload: {error}"
            ) from error
        offset = semantic_end
        cap1_bytes = _f12_cap1_body_bytes(models[offset:])
        cap1_end = offset + cap1_bytes
        if (has_selector and cap1_end >= len(models)) or (
            not has_selector and cap1_end != len(models)
        ):
            raise ResidualArchiveError(
                "invalid fixed F14/F16/F21 carrier payload length"
            )
        cap1 = _decode_f12_cap1_body(models[offset:cap1_end], F13_CAP_FIELD_ORDER)
        try:
            decode_cap1(cap1, frames=600, dimensions=12)
            if has_selector:
                selector = (
                    F24_SPARSE_SELECTOR_PREFIX
                    if has_sparse_selector
                    else F12_SELECTOR_PREFIX
                ) + models[cap1_end:]
                decode_selector(selector)
                carrier = pack_frame0_selector_carrier(cap1, selector)
            else:
                carrier = cap1
        except (CoefficientAr1CodecError, ValueError) as error:
            raise ResidualArchiveError(
                f"invalid fixed F14/F16/F21 carrier payload: {error}"
            ) from error
        return semantic, carrier, hpac, True
    if models.startswith(FIXED_WANS_AR1_PINNED_MAGIC):
        minimum = 4 + FIXED_IHS2_BODY_BYTES + F13_WANS_BODY_BYTES + 6 + 1
        if len(models) < minimum:
            raise ResidualArchiveError("truncated fixed F13 model payload")
        offset = 4
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic_end = offset + F13_WANS_BODY_BYTES
        try:
            semantic = decode_f12_wans_body(
                models[offset:semantic_end], F13_WANS_STREAM_ORDER
            )
            decode_wans1(semantic)
        except RendererWeightCodecError as error:
            raise ResidualArchiveError(
                f"invalid fixed F13 semantic payload: {error}"
            ) from error
        offset = semantic_end
        cap1_bytes = _f12_cap1_body_bytes(models[offset:])
        cap1_end = offset + cap1_bytes
        if cap1_end >= len(models):
            raise ResidualArchiveError("truncated fixed F13 carrier payload")
        cap1 = _decode_f12_cap1_body(models[offset:cap1_end], F13_CAP_FIELD_ORDER)
        selector = F12_SELECTOR_PREFIX + models[cap1_end:]
        try:
            decode_cap1(cap1, frames=600, dimensions=12)
            decode_selector(selector)
            carrier = pack_frame0_selector_carrier(cap1, selector)
        except (CoefficientAr1CodecError, ValueError) as error:
            raise ResidualArchiveError(
                f"invalid fixed F13 carrier payload: {error}"
            ) from error
        return semantic, carrier, hpac, True
    f12_cap_order = _f12_cap_order(models[:4])
    if f12_cap_order is not None:
        minimum = 4 + 2 + F12_ORDER_BYTES + FIXED_IHS2_BODY_BYTES + 6 + 1
        if len(models) < minimum:
            raise ResidualArchiveError("truncated fixed F12 model payload")
        semantic_bytes = struct.unpack_from("<H", models, 4)[0]
        try:
            order = unpack_f12_stream_order(models[6 : 6 + F12_ORDER_BYTES])
        except RendererWeightCodecError as error:
            raise ResidualArchiveError(
                f"invalid fixed F12 semantic order: {error}"
            ) from error
        offset = 6 + F12_ORDER_BYTES
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic_end = offset + semantic_bytes
        if not semantic_bytes or semantic_end >= len(models):
            raise ResidualArchiveError("invalid fixed F12 semantic payload length")
        try:
            semantic = decode_f12_wans_body(models[offset:semantic_end], order)
            decode_wans1(semantic)
        except RendererWeightCodecError as error:
            raise ResidualArchiveError(
                f"invalid fixed F12 semantic payload: {error}"
            ) from error
        offset = semantic_end
        cap1_bytes = _f12_cap1_body_bytes(models[offset:])
        cap1_end = offset + cap1_bytes
        if cap1_end >= len(models):
            raise ResidualArchiveError("truncated fixed F12 carrier payload")
        cap1 = _decode_f12_cap1_body(models[offset:cap1_end], f12_cap_order)
        selector = F12_SELECTOR_PREFIX + models[cap1_end:]
        try:
            decode_cap1(cap1, frames=600, dimensions=12)
            decode_selector(selector)
            carrier = pack_frame0_selector_carrier(cap1, selector)
        except (CoefficientAr1CodecError, ValueError) as error:
            raise ResidualArchiveError(
                f"invalid fixed F12 carrier payload: {error}"
            ) from error
        return semantic, carrier, hpac, True
    if models.startswith(FIXED_WANS_AR1_MAGIC):
        if len(models) < 10:
            raise ResidualArchiveError("truncated fixed WANS+AR1 model payload")
        semantic_bytes, carrier_bytes, selector_bytes = struct.unpack_from(
            "<HHH", models, 4
        )
        expected = (
            10 + FIXED_IHS2_BODY_BYTES + semantic_bytes + carrier_bytes + selector_bytes
        )
        if (
            not semantic_bytes
            or not carrier_bytes
            or not selector_bytes
            or len(models) != expected
        ):
            raise ResidualArchiveError("invalid fixed WANS+AR1 model payload length")
        offset = 10
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic = models[offset : offset + semantic_bytes]
        offset += semantic_bytes
        carrier = models[offset : offset + carrier_bytes]
        selector = models[offset + carrier_bytes :]
        try:
            decode_wans1(semantic)
            decode_cap1(carrier, frames=600, dimensions=12)
            carrier = pack_frame0_selector_carrier(carrier, selector)
        except (
            RendererWeightCodecError,
            CoefficientAr1CodecError,
            ValueError,
        ) as error:
            raise ResidualArchiveError(
                f"invalid fixed WANS+AR1 payload: {error}"
            ) from error
        return semantic, carrier, hpac, True
    if models.startswith(FIXED_AR1_MAGIC):
        if len(models) < 8:
            raise ResidualArchiveError("truncated fixed AR1 model payload")
        carrier_bytes, selector_bytes = struct.unpack_from("<HH", models, 4)
        expected = (
            8
            + FIXED_IHS2_BODY_BYTES
            + FIXED_SEMANTIC_BYTES
            + carrier_bytes
            + selector_bytes
        )
        if not carrier_bytes or not selector_bytes or len(models) != expected:
            raise ResidualArchiveError("invalid fixed AR1 model payload length")
        offset = 8
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic = models[offset : offset + FIXED_SEMANTIC_BYTES]
        offset += FIXED_SEMANTIC_BYTES
        carrier = models[offset : offset + carrier_bytes]
        selector = models[offset + carrier_bytes :]
        try:
            decode_cap1(carrier, frames=600, dimensions=12)
            carrier = pack_frame0_selector_carrier(carrier, selector)
        except (CoefficientAr1CodecError, ValueError) as error:
            raise ResidualArchiveError(f"invalid fixed AR1 payload: {error}") from error
        return semantic, carrier, hpac, True
    if models.startswith(FIXED_WANS_MAGIC):
        if len(models) < 8:
            raise ResidualArchiveError("truncated fixed WANS model payload")
        semantic_bytes, selector_bytes = struct.unpack_from("<HH", models, 4)
        expected = (
            8
            + FIXED_IHS2_BODY_BYTES
            + semantic_bytes
            + FIXED_COMPACT_CPS3_BYTES
            + selector_bytes
        )
        if not semantic_bytes or not selector_bytes or len(models) != expected:
            raise ResidualArchiveError("invalid fixed WANS model payload length")
        offset = 8
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic = models[offset : offset + semantic_bytes]
        offset += semantic_bytes
        compact_carrier = models[offset : offset + FIXED_COMPACT_CPS3_BYTES]
        selector = models[offset + FIXED_COMPACT_CPS3_BYTES :]
        try:
            decode_wans1(semantic)
            carrier = pack_frame0_selector_carrier(
                restore_fixed_cps3(compact_carrier, frames=600, dimensions=12),
                selector,
            )
        except (RendererWeightCodecError, ValueError) as error:
            raise ResidualArchiveError(
                f"invalid fixed WANS payload: {error}"
            ) from error
        return semantic, carrier, hpac, True
    if models.startswith(FIXED_VQ_MAGIC):
        offset = len(FIXED_VQ_MAGIC)
        minimum = offset + FIXED_IHS2_BODY_BYTES + FIXED_SEMANTIC_BYTES + 1
        if len(models) < minimum:
            raise ResidualArchiveError("truncated fixed VQ model payload")
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic = models[offset : offset + FIXED_SEMANTIC_BYTES]
        carrier = models[offset + FIXED_SEMANTIC_BYTES :]
        try:
            (decode_cvq1 if carrier.startswith(b"CVQ1") else decode_cvh1)(
                carrier, frames=600, dimensions=12
            )
        except ValueError as error:
            raise ResidualArchiveError(f"invalid fixed VQ carrier: {error}") from error
        return semantic, carrier, hpac, True
    if models.startswith(FIXED_SELECTOR_MAGIC):
        if len(models) < 6:
            raise ResidualArchiveError("truncated fixed selector model payload")
        selector_bytes = struct.unpack_from("<H", models, 4)[0]
        expected = (
            6
            + FIXED_IHS2_BODY_BYTES
            + FIXED_SEMANTIC_BYTES
            + FIXED_COMPACT_CPS3_BYTES
            + selector_bytes
        )
        if not selector_bytes or len(models) != expected:
            raise ResidualArchiveError("invalid fixed selector model payload length")
        offset = 6
        hpac = FIXED_IHS2_HEADER + models[offset : offset + FIXED_IHS2_BODY_BYTES]
        offset += FIXED_IHS2_BODY_BYTES
        semantic = models[offset : offset + FIXED_SEMANTIC_BYTES]
        offset += FIXED_SEMANTIC_BYTES
        compact_carrier = models[offset : offset + FIXED_COMPACT_CPS3_BYTES]
        selector = models[offset + FIXED_COMPACT_CPS3_BYTES :]
        try:
            carrier = pack_frame0_selector_carrier(
                restore_fixed_cps3(compact_carrier, frames=600, dimensions=12),
                selector,
            )
        except ValueError as error:
            raise ResidualArchiveError(
                f"invalid fixed selector payload: {error}"
            ) from error
        return semantic, carrier, hpac, True
    fixed_order = _fixed_production_order(models[:4])
    if fixed_order is not None:
        expected = (
            4 + FIXED_SEMANTIC_BYTES + FIXED_COMPACT_CPS3_BYTES + FIXED_IHS2_BODY_BYTES
        )
        if len(models) != expected:
            raise ResidualArchiveError("invalid fixed production model payload length")
        offset = 4
        fields: dict[str, bytes] = {}
        sizes = {
            "semantic": FIXED_SEMANTIC_BYTES,
            "carrier": FIXED_COMPACT_CPS3_BYTES,
            "hpac": FIXED_IHS2_BODY_BYTES,
        }
        for name in FIXED_PRODUCTION_ORDERS[fixed_order]:
            size = sizes[name]
            fields[name] = models[offset : offset + size]
            offset += size
        try:
            carrier = restore_fixed_cps3(fields["carrier"], frames=600, dimensions=12)
        except ValueError as error:
            raise ResidualArchiveError(
                f"invalid fixed production CPS3 payload: {error}"
            ) from error
        return (
            fields["semantic"],
            carrier,
            FIXED_IHS2_HEADER + fields["hpac"],
            True,
        )
    if models.startswith(FIXED_MODEL_MAGIC):
        semantic_start = len(FIXED_MODEL_MAGIC)
        semantic_end = semantic_start + FIXED_SEMANTIC_BYTES
        carrier_end = semantic_end + FIXED_CARRIER_BYTES
        if len(models) <= carrier_end:
            raise ResidualArchiveError("truncated fixed-schema model payload")
        return (
            models[semantic_start:semantic_end],
            models[semantic_end:carrier_end],
            models[carrier_end:],
            False,
        )
    if models.startswith(COMPACT_MODEL_MAGIC):
        if len(models) < len(COMPACT_MODEL_MAGIC) + 2 + FIXED_SEMANTIC_BYTES + 1:
            raise ResidualArchiveError("truncated compact-schema model payload")
        carrier_bytes = struct.unpack_from("<H", models, len(COMPACT_MODEL_MAGIC))[0]
        semantic_start = len(COMPACT_MODEL_MAGIC) + 2
        semantic_end = semantic_start + FIXED_SEMANTIC_BYTES
        carrier_end = semantic_end + carrier_bytes
        if not carrier_bytes or len(models) <= carrier_end:
            raise ResidualArchiveError("invalid compact-schema carrier length")
        return (
            models[semantic_start:semantic_end],
            models[semantic_end:carrier_end],
            models[carrier_end:],
            False,
        )
    if len(models) < 8:
        raise ResidualArchiveError("truncated model payload")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if semantic_bytes <= 0 or carrier_bytes <= 0 or carrier_end >= len(models):
        raise ResidualArchiveError("invalid model component lengths")
    return (
        models[8:semantic_end],
        models[semantic_end:carrier_end],
        models[carrier_end:],
        False,
    )


def _token_section(
    table: QuantizedTable | None, token_stream: bytes, schema: str
) -> tuple[bytes, bytes]:
    if schema == BASELINE_SCHEMA:
        if table is not None:
            raise ResidualArchiveError(
                "baseline schema cannot contain a residual table"
            )
        return b"", token_stream
    if table is None:
        raise ResidualArchiveError("residual archive requires a table")
    if schema == RCL1_SCHEMA:
        residual = serialize_tables((table,))
    elif schema == FIXED_SCHEMA:
        residual = serialize_fixed_boundary_int6(table)
    else:
        raise ResidualArchiveError(f"unknown residual schema {schema}")
    return residual, residual + token_stream


def _zip_bytes(
    outer_payload: bytes,
    *,
    member_name: str = "p",
    compression: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> bytes:
    """Build the charged one-member ZIP in memory with fixed metadata."""
    output = io.BytesIO()
    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(
        output,
        "w",
        compression=compression,
        compresslevel=compresslevel,
        allowZip64=False,
    ) as archive:
        archive.writestr(info, outer_payload)
    return output.getvalue()


def _write_zip(output: Path, zip_payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(zip_payload)


def build_residual_archive_bytes(
    payload: BaselinePayload,
    table: QuantizedTable | None,
    token_stream: bytes,
    schema: str,
    *,
    hpac_blob: bytes | None = None,
    semantic_blob: bytes | None = None,
    carrier_blob: bytes | None = None,
    lzma_filters: list[dict[str, object]] | None = None,
    fixed_model_lengths: bool = False,
    fixed_schema_order: int | None = None,
    fixed_selector_schema: bool = False,
    fixed_vq_schema: bool = False,
    fixed_wans_schema: bool = False,
    fixed_ar1_schema: bool = False,
    fixed_wans_ar1_schema: bool = False,
    fixed_wans_ar1_compact_schema: bool = False,
    fixed_wans_ar1_pinned_schema: bool = False,
    fixed_wans_ar1_rate_schema: bool = False,
    fixed_wans_ar1_rc64_schema: bool = False,
    f12_wans_stream_order: tuple[int, ...] | None = None,
    f12_cap_field_order: int = 0,
    model_compression: str = "xz",
    zip_compression: int = zipfile.ZIP_STORED,
    zip_compresslevel: int | None = None,
    zip_member_name: str = "p",
) -> tuple[bytes, dict[str, object]]:
    """Return exact deterministic archive bytes and their charged ledger."""
    if len(token_stream) == 0 or (
        len(token_stream) % 4 and not fixed_wans_ar1_rc64_schema
    ):
        raise ResidualArchiveError(
            "range-coded token stream must be non-empty u32 words"
        )
    residual, section = _token_section(table, token_stream, schema)
    stored_hpac = payload.hpac_blob if hpac_blob is None else hpac_blob
    stored_semantic = payload.semantic_blob if semantic_blob is None else semantic_blob
    filters = LZMA_FILTERS if lzma_filters is None else lzma_filters
    stored_carrier = payload.carrier_blob if carrier_blob is None else carrier_blob
    if model_compression not in ("xz", "raw"):
        raise ResidualArchiveError("model compression must be xz or raw")
    if (
        int(fixed_schema_order is not None)
        + int(fixed_selector_schema)
        + int(fixed_vq_schema)
        + int(fixed_wans_schema)
        + int(fixed_ar1_schema)
        + int(fixed_wans_ar1_schema)
        + int(fixed_wans_ar1_compact_schema)
        + int(fixed_wans_ar1_pinned_schema)
        + int(fixed_wans_ar1_rate_schema)
        + int(fixed_wans_ar1_rc64_schema)
        > 1
    ):
        raise ResidualArchiveError("only one fixed model schema may be selected")
    if (
        fixed_schema_order is not None
        or fixed_selector_schema
        or fixed_vq_schema
        or fixed_wans_schema
        or fixed_ar1_schema
        or fixed_wans_ar1_schema
        or fixed_wans_ar1_compact_schema
        or fixed_wans_ar1_pinned_schema
        or fixed_wans_ar1_rate_schema
        or fixed_wans_ar1_rc64_schema
    ) and (schema != FIXED_SCHEMA or table is None):
        raise ResidualArchiveError(
            "fixed production schema requires the fixed residual table"
        )
    model_payload = _models(
        payload,
        stored_hpac,
        semantic_blob=stored_semantic,
        carrier_blob=stored_carrier,
        fixed_model_lengths=fixed_model_lengths,
        fixed_schema_order=fixed_schema_order,
        fixed_selector_schema=fixed_selector_schema,
        fixed_vq_schema=fixed_vq_schema,
        fixed_wans_schema=fixed_wans_schema,
        fixed_ar1_schema=fixed_ar1_schema,
        fixed_wans_ar1_schema=fixed_wans_ar1_schema,
        fixed_wans_ar1_compact_schema=fixed_wans_ar1_compact_schema,
        fixed_wans_ar1_pinned_schema=fixed_wans_ar1_pinned_schema,
        fixed_wans_ar1_rate_schema=fixed_wans_ar1_rate_schema,
        fixed_wans_ar1_rc64_schema=fixed_wans_ar1_rc64_schema,
        f12_wans_stream_order=f12_wans_stream_order,
        f12_cap_field_order=f12_cap_field_order,
    )
    compression_format = (
        lzma.FORMAT_XZ if model_compression == "xz" else lzma.FORMAT_RAW
    )
    compressed = lzma.compress(
        model_payload, format=compression_format, filters=filters
    )
    if (
        fixed_schema_order is None
        and not fixed_selector_schema
        and not fixed_vq_schema
        and not fixed_wans_schema
        and not fixed_ar1_schema
        and not fixed_wans_ar1_schema
        and not fixed_wans_ar1_compact_schema
        and not fixed_wans_ar1_pinned_schema
        and not fixed_wans_ar1_rate_schema
        and not fixed_wans_ar1_rc64_schema
    ):
        outer_payload = struct.pack("<I", len(compressed)) + compressed + section
        outer_layout = "legacy_explicit_lengths"
    else:
        # The F6A model magic fixes the carrier/IHS2 layouts and the residual
        # table schema. Its length is therefore derived from this schema.
        outer_payload = compressed + residual[len(FIXED_MAGIC) :] + token_stream
        outer_layout = f"fixed_implicit_{model_compression}"
    archive_payload = _zip_bytes(
        outer_payload,
        member_name=zip_member_name,
        compression=zip_compression,
        compresslevel=zip_compresslevel,
    )
    charged_hpac = stored_hpac
    charged_semantic = stored_semantic
    charged_carrier = stored_carrier
    has_rc64_selector = False
    has_rc64_sparse_selector = False
    if fixed_schema_order is not None:
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_carrier = strip_fixed_cps3(stored_carrier, frames=600, dimensions=12)
    elif fixed_selector_schema:
        base_carrier, selector = split_frame0_selector_carrier(stored_carrier)
        if selector is None:  # pragma: no cover - _models already rejects this.
            raise ResidualArchiveError(
                "fixed selector schema lost its selector payload"
            )
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_carrier = (
            strip_fixed_cps3(base_carrier, frames=600, dimensions=12) + selector
        )
    elif fixed_vq_schema:
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
    elif fixed_wans_schema:
        base_carrier, selector = split_frame0_selector_carrier(stored_carrier)
        if selector is None:  # pragma: no cover - _models already rejects this.
            raise ResidualArchiveError("fixed WANS schema lost its selector payload")
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_carrier = (
            strip_fixed_cps3(base_carrier, frames=600, dimensions=12) + selector
        )
    elif fixed_ar1_schema:
        base_carrier, selector = split_frame0_selector_carrier(stored_carrier)
        if selector is None:  # pragma: no cover - _models already rejects this.
            raise ResidualArchiveError("fixed AR1 schema lost its selector payload")
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_carrier = base_carrier + selector
    elif fixed_wans_ar1_schema:
        base_carrier, selector = split_frame0_selector_carrier(stored_carrier)
        if selector is None:  # pragma: no cover - _models already rejects this.
            raise ResidualArchiveError(
                "fixed WANS+AR1 schema lost its selector payload"
            )
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_carrier = base_carrier + selector
    elif fixed_wans_ar1_compact_schema:
        base_carrier, selector = split_frame0_selector_carrier(stored_carrier)
        if selector is None:  # pragma: no cover - _models already rejects this.
            raise ResidualArchiveError("fixed F12 schema lost its selector payload")
        order = (
            tuple(range(16))
            if f12_wans_stream_order is None
            else tuple(f12_wans_stream_order)
        )
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_semantic = encode_f12_wans_body(stored_semantic, order)
        charged_carrier = (
            base_carrier[len(F12_CAP1_PREFIX) :] + selector[len(F12_SELECTOR_PREFIX) :]
        )
    elif fixed_wans_ar1_pinned_schema:
        base_carrier, selector = split_frame0_selector_carrier(stored_carrier)
        if selector is None:  # pragma: no cover - _models already rejects this.
            raise ResidualArchiveError("fixed F13 schema lost its selector payload")
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_semantic = encode_f12_wans_body(stored_semantic, F13_WANS_STREAM_ORDER)
        charged_carrier = (
            _encode_f12_cap1_body(base_carrier, F13_CAP_FIELD_ORDER)
            + selector[len(F12_SELECTOR_PREFIX) :]
        )
    elif fixed_wans_ar1_rate_schema or fixed_wans_ar1_rc64_schema:
        base_carrier, selector = split_frame0_selector_carrier(stored_carrier)
        if (
            fixed_wans_ar1_rate_schema and selector is not None
        ):  # pragma: no cover - _models rejects this.
            raise ResidualArchiveError(
                "fixed F14 rate-only schema gained a selector payload"
            )
        has_rc64_selector = selector is not None
        has_rc64_sparse_selector = selector is not None and selector.startswith(
            F24_SPARSE_SELECTOR_PREFIX
        )
        charged_hpac = stored_hpac[len(FIXED_IHS2_HEADER) :]
        charged_semantic = encode_f12_wans_body(stored_semantic, F13_WANS_STREAM_ORDER)
        charged_carrier = _encode_f12_cap1_body(base_carrier, F13_CAP_FIELD_ORDER) + (
            selector[
                len(F24_SPARSE_SELECTOR_PREFIX)
                if has_rc64_sparse_selector
                else len(F12_SELECTOR_PREFIX) :
            ]
            if selector is not None
            else b""
        )
    return archive_payload, {
        "schema": schema,
        "payload_bytes": len(outer_payload),
        "zip_member_bytes": len(outer_payload),
        "archive_bytes": len(archive_payload),
        "zip_method": "ZIP_STORED"
        if zip_compression == zipfile.ZIP_STORED
        else "ZIP_DEFLATED"
        if zip_compression == zipfile.ZIP_DEFLATED
        else str(zip_compression),
        "zip_member_name": zip_member_name,
        "archive_sha256": sha256_bytes(archive_payload),
        "payload_sha256": sha256_bytes(outer_payload),
        "token_stream_section_bytes": len(token_stream),
        "residual_section_bytes": len(residual),
        "all_other_section_bytes": (
            0
            if (
                fixed_schema_order is not None
                or fixed_vq_schema
                or fixed_wans_schema
                or fixed_ar1_schema
                or fixed_wans_ar1_schema
                or fixed_wans_ar1_compact_schema
                or fixed_wans_ar1_pinned_schema
                or fixed_wans_ar1_rate_schema
                or fixed_wans_ar1_rc64_schema
            )
            else 4
        )
        + len(compressed),
        "residual_payload_sha256": sha256_bytes(residual) if residual else "",
        "token_stream_sha256": sha256_bytes(token_stream),
        "compressed_models_sha256": sha256_bytes(compressed),
        "stored_hpac_bytes": len(charged_hpac),
        "stored_hpac_sha256": sha256_bytes(charged_hpac),
        "rehydrated_hpac_bytes": len(stored_hpac),
        "rehydrated_hpac_sha256": sha256_bytes(stored_hpac),
        "stored_semantic_bytes": len(charged_semantic),
        "stored_semantic_sha256": sha256_bytes(charged_semantic),
        "rehydrated_semantic_bytes": len(payload.semantic_blob),
        "rehydrated_semantic_sha256": sha256_bytes(payload.semantic_blob),
        "stored_carrier_bytes": len(charged_carrier),
        "stored_carrier_sha256": sha256_bytes(charged_carrier),
        "rehydrated_carrier_bytes": len(stored_carrier),
        "rehydrated_carrier_sha256": sha256_bytes(stored_carrier),
        "model_container_schema": (
            f"fixed_F6A{fixed_schema_order}"
            if fixed_schema_order is not None
            else "fixed_F7S1"
            if fixed_selector_schema
            else "fixed_F8V1"
            if fixed_vq_schema
            else "fixed_F9W1"
            if fixed_wans_schema
            else "fixed_F10C"
            if fixed_ar1_schema
            else "fixed_F11A"
            if fixed_wans_ar1_schema
            else "fixed_F12A"
            if fixed_wans_ar1_compact_schema
            else "fixed_F13A"
            if fixed_wans_ar1_pinned_schema
            else "fixed_F14R"
            if fixed_wans_ar1_rate_schema
            else "fixed_F24S"
            if fixed_wans_ar1_rc64_schema and has_rc64_sparse_selector
            else "fixed_F21S"
            if fixed_wans_ar1_rc64_schema and has_rc64_selector
            else "fixed_F16R"
            if fixed_wans_ar1_rc64_schema
            else "fixed_M2F1"
            if fixed_model_lengths and len(stored_carrier) == FIXED_CARRIER_BYTES
            else "compact_M3F1"
            if fixed_model_lengths
            else "u32_lengths"
        ),
        "outer_layout": outer_layout,
        "model_compression": model_compression,
        "token_codec": "rc64" if fixed_wans_ar1_rc64_schema else "range32",
        "selector_container_header_bytes": 2
        if fixed_selector_schema
        else 2
        if fixed_wans_schema
        else 2
        if fixed_ar1_schema
        else 2
        if fixed_wans_ar1_schema
        else 8
        if fixed_wans_ar1_compact_schema
        else 0,
        "selector_prefix_elided_bytes": len(F24_SPARSE_SELECTOR_PREFIX)
        if has_rc64_sparse_selector
        else len(F12_SELECTOR_PREFIX)
        if has_rc64_selector
        else 0,
    }


def build_residual_archive(
    payload: BaselinePayload,
    table: QuantizedTable | None,
    token_stream: bytes,
    schema: str,
    output: Path,
    *,
    hpac_blob: bytes | None = None,
    semantic_blob: bytes | None = None,
    carrier_blob: bytes | None = None,
    lzma_filters: list[dict[str, object]] | None = None,
    fixed_model_lengths: bool = False,
    fixed_schema_order: int | None = None,
    fixed_selector_schema: bool = False,
    fixed_vq_schema: bool = False,
    fixed_wans_schema: bool = False,
    fixed_ar1_schema: bool = False,
    fixed_wans_ar1_schema: bool = False,
    fixed_wans_ar1_compact_schema: bool = False,
    fixed_wans_ar1_pinned_schema: bool = False,
    fixed_wans_ar1_rate_schema: bool = False,
    fixed_wans_ar1_rc64_schema: bool = False,
    f12_wans_stream_order: tuple[int, ...] | None = None,
    f12_cap_field_order: int = 0,
    model_compression: str = "xz",
    zip_compression: int = zipfile.ZIP_STORED,
    zip_compresslevel: int | None = None,
    zip_member_name: str = "p",
) -> dict[str, object]:
    """Build a deterministic one-member production candidate archive."""
    archive_payload, report = build_residual_archive_bytes(
        payload,
        table,
        token_stream,
        schema,
        hpac_blob=hpac_blob,
        semantic_blob=semantic_blob,
        carrier_blob=carrier_blob,
        lzma_filters=lzma_filters,
        fixed_model_lengths=fixed_model_lengths,
        fixed_schema_order=fixed_schema_order,
        fixed_selector_schema=fixed_selector_schema,
        fixed_vq_schema=fixed_vq_schema,
        fixed_wans_schema=fixed_wans_schema,
        fixed_ar1_schema=fixed_ar1_schema,
        fixed_wans_ar1_schema=fixed_wans_ar1_schema,
        fixed_wans_ar1_compact_schema=fixed_wans_ar1_compact_schema,
        fixed_wans_ar1_pinned_schema=fixed_wans_ar1_pinned_schema,
        fixed_wans_ar1_rate_schema=fixed_wans_ar1_rate_schema,
        fixed_wans_ar1_rc64_schema=fixed_wans_ar1_rc64_schema,
        f12_wans_stream_order=f12_wans_stream_order,
        f12_cap_field_order=f12_cap_field_order,
        model_compression=model_compression,
        zip_compression=zip_compression,
        zip_compresslevel=zip_compresslevel,
        zip_member_name=zip_member_name,
    )
    _write_zip(output, archive_payload)
    return report


def read_residual_archive(archive_path: Path) -> ResidualArchiveParts:
    """Strictly parse a baseline, generic RCL1, or fixed-schema archive."""
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != 1 or names[0] not in ("p", ""):
            raise ResidualArchiveError(
                "archive must contain exactly one supported member"
            )
        outer = archive.read(names[0])
    if len(outer) < 4:
        raise ResidualArchiveError("truncated outer payload")
    compressed: bytes
    models: bytes
    section: bytes
    if outer.startswith(XZ_MAGIC):
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
        models = decoder.decompress(outer)
        section = decoder.unused_data
        compressed = outer[: len(outer) - len(section)]
        if not decoder.eof or not section:
            raise ResidualArchiveError("truncated implicit XZ model section")
    else:
        compressed_bytes = struct.unpack_from("<I", outer)[0]
        legacy = False
        if 0 < compressed_bytes and 4 + compressed_bytes < len(outer):
            candidate = outer[4 : 4 + compressed_bytes]
            try:
                models = lzma.decompress(candidate)
                section = outer[4 + compressed_bytes :]
                compressed = candidate
                legacy = True
            except lzma.LZMAError:
                # Generic raw-LZMA archives retain this explicit compressed
                # length.  Decode that exact slice before attempting the F6
                # implicit raw layout, whose stream starts at byte zero.
                try:
                    models = lzma.decompress(
                        candidate, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS
                    )
                    section = outer[4 + compressed_bytes :]
                    compressed = candidate
                    legacy = True
                except lzma.LZMAError:
                    legacy = False
        if not legacy:
            try:
                decoder = lzma.LZMADecompressor(
                    format=lzma.FORMAT_RAW, filters=LZMA_FILTERS
                )
                models = decoder.decompress(outer)
                section = decoder.unused_data
                compressed = outer[: len(outer) - len(section)]
            except lzma.LZMAError as error:
                raise ResidualArchiveError(
                    "invalid compressed model section"
                ) from error
            if not decoder.eof or not section:
                raise ResidualArchiveError("truncated implicit raw model section")
    token_codec = (
        "rc64"
        if models.startswith(
            (
                FIXED_WANS_AR1_RC64_MAGIC,
                FIXED_WANS_AR1_RC64_SELECTOR_MAGIC,
                FIXED_WANS_AR1_RC64_SPARSE_SELECTOR_MAGIC,
            )
        )
        else "range32"
    )
    semantic, carrier, hpac, implicit_fixed = _split_models(models)
    if implicit_fixed:
        fixed_size = (
            len(FIXED_MAGIC) + 2 + packed_length(FIXED_STATES * NUM_CLASSES, FIXED_BITS)
        )
        compact_size = fixed_size - len(FIXED_MAGIC)
        if len(section) <= compact_size:
            raise ResidualArchiveError("truncated fixed production residual section")
        residual = FIXED_MAGIC + section[:compact_size]
        table, tokens, schema = (
            deserialize_fixed_boundary_int6(residual),
            section[compact_size:],
            FIXED_SCHEMA,
        )
    elif section.startswith(b"RCL1"):
        try:
            table_size = rcl1_payload_length(section)
            tables, mode = deserialize_tables(section[:table_size])
        except BitPackingError as error:
            raise ResidualArchiveError(
                f"invalid RCL1 residual section: {error}"
            ) from error
        if mode is not None or len(tables) != 1:
            raise ResidualArchiveError(
                "production RCL1 section must contain one additive table"
            )
        table, tokens, schema, residual = (
            tables[0],
            section[table_size:],
            RCL1_SCHEMA,
            section[:table_size],
        )
    elif section.startswith(FIXED_MAGIC):
        fixed_size = (
            len(FIXED_MAGIC) + 2 + packed_length(FIXED_STATES * NUM_CLASSES, FIXED_BITS)
        )
        table, tokens, schema, residual = (
            deserialize_fixed_boundary_int6(section[:fixed_size]),
            section[fixed_size:],
            FIXED_SCHEMA,
            section[:fixed_size],
        )
    else:
        table, tokens, schema, residual = None, section, BASELINE_SCHEMA, b""
    if not tokens or (token_codec == "range32" and len(tokens) % 4):
        raise ResidualArchiveError("invalid token stream section")
    return ResidualArchiveParts(
        semantic,
        carrier,
        hpac,
        tokens,
        table,
        schema,
        residual,
        compressed,
        token_codec,
    )


def _sparse_class(code_dir: Path):
    import sys

    sys.path.insert(0, str(code_dir))
    try:
        from hpac_integer_sparse import SparseIntegerHPAC

        return SparseIntegerHPAC
    finally:
        sys.path.pop(0)


def _probability_table_from_quantized_logits(
    logits: np.ndarray,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _probability_table(logits: np.ndarray, precision: int) -> np.ndarray:
    quantized = np.clip(
        np.rint(np.asarray(logits, dtype=np.float32) * precision),
        -32768,
        32767,
    ).astype(np.int16)
    normalized = quantized.astype(np.float32) / precision
    return _probability_table_from_quantized_logits(normalized)


def _residual_feature(
    name: str,
    predicted: np.ndarray,
    base_logits: np.ndarray,
    boundary: np.ndarray,
    row_buckets: np.ndarray,
    frame: int,
    frame_count: int,
) -> np.ndarray:
    if name == "global_bias":
        return np.zeros_like(predicted)
    if name == "predicted_margin":
        return predicted * 4 + margin_buckets(base_logits)
    if name == "previous_predicted":
        return predicted
    if name == "boundary_predicted":
        return boundary.astype(np.int64) * NUM_CLASSES + predicted
    if name == "row_predicted":
        return row_buckets * NUM_CLASSES + predicted
    if name == "segment_predicted":
        return frame * 8 // frame_count * NUM_CLASSES + predicted
    raise ResidualArchiveError(f"unsupported residual feature: {name}")


def decode_production_tokens(
    parts: ResidualArchiveParts,
    runtime,
    code_dir: Path,
    device,
) -> tuple[object, dict[str, object]]:
    """Decode a production archive and hash exactly what the coder consumes.

    The CDF hash is the bytewise sequence of canonical float32 probability
    rows handed to the fixed ``Categorical(perfect=False)`` constructor. Those
    rows are the complete, deterministic inputs to constriction's integer CDF
    construction; the encoder records the same sequence independently.
    """
    import constriction
    import torch

    started = time.time()
    sparse_class = _sparse_class(code_dir)
    from .hpac_inference import (
        configure_cuda_reproducibility,
        optimize_sparse_evaluator,
    )
    from .hpac_lora import adapter_sha256, apply_adapter, materialize_hpac

    configure_cuda_reproducibility()
    base_hpac, adapter = materialize_hpac(parts.hpac_blob, runtime)
    model = runtime.load_hpac(base_hpac, device)
    apply_adapter(model, adapter)
    masks = runtime.group_masks(device)
    sparse = sparse_class(model, runtime.EVAL_H, runtime.EVAL_W)
    if parts.token_codec == "rc64":
        from .entropy.rc64 import NativeDecoder

        library = os.environ.get("CPR1_RC64_LIBRARY")
        if not library:
            raise ResidualArchiveError("RC64 decoding requires CPR1_RC64_LIBRARY")
        decoder = NativeDecoder(Path(library), parts.token_stream)
        family = None
    else:
        decoder = constriction.stream.queue.RangeDecoder(
            np.frombuffer(parts.token_stream, dtype="<u4")
        )
        family = constriction.stream.model.Categorical(perfect=False)
    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append(
            (
                torch.from_numpy(flat_positions).to(device),
                flat_positions,
                (flat_positions // runtime.EVAL_W) * 8 // runtime.EVAL_H,
            )
        )

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros(
            (1, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.long,
            device=device,
        )
        tokens = torch.empty(
            (runtime.N, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.uint8,
        )
        for frame in range(runtime.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(
                    runtime.EVAL_H * runtime.EVAL_W,
                    4,
                    dtype=np.uint8,
                )
            for group, plan in enumerate(group_plans):
                device_positions, flat_positions, row_buckets = plan
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                # Softmax is strictly order-preserving, so constructing the
                # float64 base table only to take its argmax is redundant.
                # NumPy argmax keeps the same first-index tie behavior.
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = _residual_feature(
                    parts.table.name,
                    predicted,
                    base_logits,
                    boundary[flat_positions].astype(np.int64),
                    row_buckets,
                    frame,
                    runtime.N,
                )
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                probability = _probability_table(
                    corrected,
                    runtime.HPAC_LOGIT_PRECISION,
                )
                cdf_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )
                symbols = (
                    decoder.decode(probability)
                    if parts.token_codec == "rc64"
                    else decoder.decode(family, probability)
                ).astype(np.int64)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(
                    device
                )
            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            previous = current
    elapsed = time.time() - started
    del model
    return tokens, {
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "decoded_token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "decode_runtime_seconds": elapsed,
        "adapter_sha256": adapter_sha256(adapter),
        "token_codec": parts.token_codec,
        **(
            {"decoder_bit_position": decoder.bit_position}
            if parts.token_codec == "rc64"
            else {}
        ),
    }
