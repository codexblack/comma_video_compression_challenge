"""Read-only CPR1 artifact parsing and frozen-component hashes.

The parser implements only the public on-wire layout needed for the semantic
renderer. It is independent harness code; it does not copy or modify the CPR1
submission runtime.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bits import BitPackingError, pack_signed, unpack_signed

ARCHIVE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
ARCHIVE_BYTES = 191_052
CHALLENGE_COMMIT = "d3f688f84f555c5aaebee7d2c4203efc8a9051e2"
RECIPE_COMMIT = "2f94596bb0136d342254022a5c9584756eae0468"


@dataclass(frozen=True)
class TensorSchema:
    name: str
    shape: tuple[int, ...]

    @property
    def count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def is_fp16(self) -> bool:
        return len(self.shape) < 2

    @property
    def scale_count(self) -> int:
        if self.is_fp16:
            return 0
        return self.shape[-1] if self.name.endswith("embed.weight") else self.shape[0]


def _schemas() -> tuple[TensorSchema, ...]:
    items: list[tuple[str, tuple[int, ...]]] = [
        ("token_embed.weight", (5, 96)),
        ("frame_embed.weight", (600, 8)),
        ("coord_mix.weight", (96, 100, 1, 1)),
        ("coord_mix.bias", (96,)),
    ]
    for block in range(4):
        prefix = f"blocks.{block}"
        items.extend(
            [
                (f"{prefix}.dw.weight", (96, 1, 3, 3)),
                (f"{prefix}.dw.bias", (96,)),
                (f"{prefix}.pw.weight", (96, 96, 1, 1)),
                (f"{prefix}.pw.bias", (96,)),
                (f"{prefix}.norm.weight", (96,)),
                (f"{prefix}.norm.bias", (96,)),
                (f"{prefix}.film.weight", (192, 8)),
                (f"{prefix}.film.bias", (192,)),
            ]
        )
    items.extend([("head.weight", (3, 96, 3, 3)), ("head.bias", (3,))])
    return tuple(TensorSchema(name, shape) for name, shape in items)


SEMANTIC_SCHEMA = _schemas()


def schema_sha256() -> str:
    document = [{"name": item.name, "shape": item.shape} for item in SEMANTIC_SCHEMA]
    return hashlib.sha256(
        json.dumps(document, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class TensorStorage:
    schema: TensorSchema
    format: str
    values: np.ndarray
    scales: np.ndarray | None
    codes: np.ndarray | None
    raw_fp16: bytes | None = None
    raw_scales: bytes | None = None

    @property
    def raw_bytes(self) -> int:
        if self.format == "fp16":
            return len(self.raw_fp16 or b"")
        assert self.codes is not None and self.scales is not None
        return len(self.raw_scales or b"") + (self.codes.size + 1) // 2


@dataclass
class BaselinePayload:
    archive_path: Path
    semantic_blob: bytes
    carrier_blob: bytes
    hpac_blob: bytes
    token_stream: bytes
    records: tuple[TensorStorage, ...]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def default_archive(root: Path) -> Path:
    return root / "third_party" / "cpr1" / "artifacts" / "final" / "archive.zip"


def parse_archive(archive_path: Path) -> tuple[bytes, bytes, bytes, bytes]:
    """Return semantic, carrier, HPAC, and immutable token payloads."""
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != ["p"]:
            raise ValueError("CPR1 archive must contain exactly member 'p'")
        payload = archive.read("p")
    if len(payload) < 4:
        raise ValueError("truncated outer payload")
    model_bytes = struct.unpack_from("<I", payload)[0]
    if model_bytes <= 0 or 4 + model_bytes >= len(payload):
        raise ValueError("invalid compressed-model length")
    models = lzma.decompress(payload[4 : 4 + model_bytes])
    tokens = payload[4 + model_bytes :]
    if len(models) < 8:
        raise ValueError("truncated model payload")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models)
    end_semantic = 8 + semantic_bytes
    end_carrier = end_semantic + carrier_bytes
    if semantic_bytes <= 0 or carrier_bytes <= 0 or end_carrier >= len(models):
        raise ValueError("invalid semantic/carrier component lengths")
    return (
        models[8:end_semantic],
        models[end_semantic:end_carrier],
        models[end_carrier:],
        tokens,
    )


def decode_legacy_w4(blob: bytes) -> tuple[TensorStorage, ...]:
    """Decode deployed int4 weights while retaining exact stored codes."""
    offset = 0
    records: list[TensorStorage] = []
    for schema in SEMANTIC_SCHEMA:
        if schema.is_fp16:
            size = schema.count * 2
            raw = blob[offset : offset + size]
            if len(raw) != size:
                raise ValueError(f"truncated fp16 tensor {schema.name}")
            values = (
                np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(schema.shape)
            )
            records.append(
                TensorStorage(schema, "fp16", values, None, None, raw_fp16=raw)
            )
            offset += size
            continue
        scale_size = schema.scale_count * 2
        raw_scales = blob[offset : offset + scale_size]
        if len(raw_scales) != scale_size:
            raise ValueError(f"truncated scale table {schema.name}")
        scales = np.frombuffer(raw_scales, dtype="<f2").astype(np.float32)
        offset += scale_size
        code_size = (schema.count + 1) // 2
        raw_codes = blob[offset : offset + code_size]
        if len(raw_codes) != code_size:
            raise ValueError(f"truncated code stream {schema.name}")
        try:
            codes = np.asarray(unpack_signed(raw_codes, schema.count, 4), dtype=np.int8)
        except BitPackingError as error:
            raise ValueError(f"invalid code stream {schema.name}: {error}") from error
        if np.any(codes == -8):
            raise ValueError(f"reserved int4 code -8 in {schema.name}")
        codes = codes.reshape(schema.shape)
        scale_shape = [1] * len(schema.shape)
        scale_shape[-1 if schema.name.endswith("embed.weight") else 0] = (
            schema.scale_count
        )
        values = codes.astype(np.float32) * scales.reshape(scale_shape)
        records.append(
            TensorStorage(schema, "w4", values, scales, codes, raw_scales=raw_scales)
        )
        offset += code_size
    if offset != len(blob):
        raise ValueError(f"semantic payload has {len(blob) - offset} trailing bytes")
    return tuple(records)


def encode_legacy_w4(records: tuple[TensorStorage, ...]) -> bytes:
    if tuple(item.schema for item in records) != SEMANTIC_SCHEMA:
        raise ValueError("records do not match the fixed CPR1 semantic schema")
    output = bytearray()
    for item in records:
        if item.format == "fp16":
            output.extend(
                item.raw_fp16 or np.asarray(item.values, dtype="<f2").tobytes()
            )
            continue
        if item.format != "w4" or item.codes is None or item.scales is None:
            raise ValueError(f"{item.schema.name} is not an int4 storage record")
        if np.any(item.codes < -7) or np.any(item.codes > 7):
            raise ValueError(f"int4 range violation in {item.schema.name}")
        output.extend(item.raw_scales or np.asarray(item.scales, dtype="<f2").tobytes())
        output.extend(pack_signed(item.codes.reshape(-1), 4))
    return bytes(output)


def decoded_state_sha256(records: tuple[TensorStorage, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.schema.name.encode("utf-8") + b"\0")
        digest.update(np.ascontiguousarray(record.values, dtype="<f4").tobytes())
    return digest.hexdigest()


def load_baseline(archive_path: Path) -> BaselinePayload:
    semantic, carrier, hpac, tokens = parse_archive(archive_path)
    return BaselinePayload(
        archive_path=archive_path,
        semantic_blob=semantic,
        carrier_blob=carrier,
        hpac_blob=hpac,
        token_stream=tokens,
        records=decode_legacy_w4(semantic),
    )


def frozen_hashes(payload: BaselinePayload) -> dict[str, str]:
    archive = payload.archive_path.read_bytes()
    return {
        "archive_sha256": sha256_bytes(archive),
        "semantic_blob_sha256": sha256_bytes(payload.semantic_blob),
        "carrier_blob_sha256": sha256_bytes(payload.carrier_blob),
        "hpac_blob_sha256": sha256_bytes(payload.hpac_blob),
        "token_stream_sha256": sha256_bytes(payload.token_stream),
        "semantic_state_sha256": decoded_state_sha256(payload.records),
    }
