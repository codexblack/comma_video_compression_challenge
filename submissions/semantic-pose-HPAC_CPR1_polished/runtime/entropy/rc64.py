"""Experimental high-precision arithmetic coding for frozen HPAC tokens.

RC64 changes only the exact representation of a known symbol sequence.  The
probabilities remain the canonical float32 rows already hashed by the frozen
HPAC path.  They are mapped deterministically to five positive integer
frequencies summing to ``2**31`` and consumed by a 63-bit arithmetic coder.

The pure-Python classes are deliberately small reference implementations for
tests.  Full 117-million-symbol audits use the ABI-compatible C backend.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Final

import numpy as np

ALPHABET: Final = 5
TOTAL: Final = 1 << 31
TOP: Final = (1 << 63) - 1
FIRST_QTR: Final = 1 << 61
HALF: Final = 1 << 62
THIRD_QTR: Final = 3 << 61


class Rc64Error(ValueError):
    """An RC64 probability row, stream, or backend operation is invalid."""


def quantize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return deterministic positive u31 frequencies for five-class rows.

    Flooring all entries and assigning the small integer balance to the most
    probable symbol avoids a platform-sensitive fractional sort.  The input is
    first interpreted as float32 because that is the production CDF contract.
    """
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ALPHABET or not values.size:
        raise Rc64Error("RC64 probabilities must have shape [N, 5]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise Rc64Error("RC64 probabilities must be finite and positive")
    row_sums = values.astype(np.float64).sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 2e-5):
        raise Rc64Error("RC64 probability rows must sum to one")
    frequencies = np.floor(values.astype(np.float64) * TOTAL).astype(np.int64)
    np.maximum(frequencies, 1, out=frequencies)
    winners = values.argmax(axis=1)
    balance = TOTAL - frequencies.sum(axis=1)
    frequencies[np.arange(values.shape[0]), winners] += balance
    if np.any(frequencies <= 0) or np.any(frequencies >= TOTAL):
        raise Rc64Error("RC64 frequency normalization failed")
    if np.any(frequencies.sum(axis=1) != TOTAL):
        raise Rc64Error("RC64 frequency rows do not total 2**31")
    return np.ascontiguousarray(frequencies, dtype=np.uint32)


class _BitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.partial = 0
        self.bits = 0

    def put(self, bit: int) -> None:
        self.partial = (self.partial << 1) | (bit & 1)
        self.bits += 1
        if self.bits == 8:
            self.output.append(self.partial)
            self.partial = 0
            self.bits = 0

    def finish(self) -> bytes:
        if self.bits:
            self.output.append(self.partial << (8 - self.bits))
            self.partial = 0
            self.bits = 0
        return bytes(self.output)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        if not payload:
            raise Rc64Error("RC64 stream is empty")
        self.payload = payload
        self.position = 0

    def read(self) -> int:
        byte = self.position >> 3
        bit = self.position & 7
        self.position += 1
        return 0 if byte >= len(self.payload) else (self.payload[byte] >> (7 - bit)) & 1


class ReferenceEncoder:
    """Slow streaming reference encoder used for deterministic unit tests."""

    def __init__(self) -> None:
        self.low = 0
        self.high = TOP
        self.pending = 0
        self.writer = _BitWriter()
        self.finished = False

    def _put_with_pending(self, bit: int) -> None:
        self.writer.put(bit)
        while self.pending:
            self.writer.put(bit ^ 1)
            self.pending -= 1

    def encode_frequencies(self, symbols: np.ndarray, frequencies: np.ndarray) -> None:
        if self.finished:
            raise Rc64Error("RC64 encoder is already finished")
        source = np.asarray(symbols, dtype=np.int32).reshape(-1)
        rows = np.asarray(frequencies, dtype=np.uint32)
        if rows.shape != (source.size, ALPHABET):
            raise Rc64Error("RC64 symbol/frequency shapes disagree")
        if np.any(source < 0) or np.any(source >= ALPHABET):
            raise Rc64Error("RC64 symbol is outside the five-class alphabet")
        if np.any(rows == 0) or np.any(rows.astype(np.uint64).sum(axis=1) != TOTAL):
            raise Rc64Error("RC64 frequency row is invalid")
        for symbol, row in zip(source, rows, strict=True):
            cumulative = np.cumsum(row, dtype=np.uint64)
            lower = 0 if symbol == 0 else int(cumulative[symbol - 1])
            upper = int(cumulative[symbol])
            width = self.high - self.low + 1
            old_low = self.low
            self.low = old_low + width * lower // TOTAL
            self.high = old_low + width * upper // TOTAL - 1
            while True:
                if self.high < HALF:
                    self._put_with_pending(0)
                elif self.low >= HALF:
                    self._put_with_pending(1)
                    self.low -= HALF
                    self.high -= HALF
                elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                    self.pending += 1
                    self.low -= FIRST_QTR
                    self.high -= FIRST_QTR
                else:
                    break
                self.low <<= 1
                self.high = (self.high << 1) | 1

    def encode(self, symbols: np.ndarray, probabilities: np.ndarray) -> None:
        self.encode_frequencies(symbols, quantize_probabilities(probabilities))

    def finish(self) -> bytes:
        if not self.finished:
            self.pending += 1
            self._put_with_pending(0 if self.low < FIRST_QTR else 1)
            self.finished = True
        return self.writer.finish()


class ReferenceDecoder:
    """Slow streaming reference decoder used for deterministic unit tests."""

    def __init__(self, payload: bytes) -> None:
        self.reader = _BitReader(payload)
        self.low = 0
        self.high = TOP
        self.code = 0
        for _ in range(63):
            self.code = (self.code << 1) | self.reader.read()

    def decode_frequencies(self, frequencies: np.ndarray) -> np.ndarray:
        rows = np.asarray(frequencies, dtype=np.uint32)
        if rows.ndim != 2 or rows.shape[1] != ALPHABET or not rows.size:
            raise Rc64Error("RC64 frequency rows must have shape [N, 5]")
        if np.any(rows == 0) or np.any(rows.astype(np.uint64).sum(axis=1) != TOTAL):
            raise Rc64Error("RC64 frequency row is invalid")
        output = np.empty(rows.shape[0], dtype=np.int32)
        for index, row in enumerate(rows):
            width = self.high - self.low + 1
            scaled = ((self.code - self.low + 1) * TOTAL - 1) // width
            cumulative = np.cumsum(row, dtype=np.uint64)
            symbol = int(np.searchsorted(cumulative, scaled, side="right"))
            if symbol >= ALPHABET:
                raise Rc64Error("RC64 decoder selected no symbol")
            lower = 0 if symbol == 0 else int(cumulative[symbol - 1])
            upper = int(cumulative[symbol])
            old_low = self.low
            self.low = old_low + width * lower // TOTAL
            self.high = old_low + width * upper // TOTAL - 1
            while True:
                if self.high < HALF:
                    pass
                elif self.low >= HALF:
                    self.code -= HALF
                    self.low -= HALF
                    self.high -= HALF
                elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                    self.code -= FIRST_QTR
                    self.low -= FIRST_QTR
                    self.high -= FIRST_QTR
                else:
                    break
                self.low <<= 1
                self.high = (self.high << 1) | 1
                self.code = (self.code << 1) | self.reader.read()
            output[index] = symbol
        return output

    def decode(self, probabilities: np.ndarray) -> np.ndarray:
        return self.decode_frequencies(quantize_probabilities(probabilities))


def compile_backend(output: Path, *, compiler: str = "cc") -> Path:
    """Compile the bundled C backend into ``output`` with GCC or Clang."""
    destination = Path(output).resolve()
    source = Path(__file__).with_name("rc64_backend.c").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-std=c11",
        "-O3",
        "-fPIC",
        "-shared",
        str(source),
        "-o",
        str(destination),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise Rc64Error(
            f"RC64 backend compilation failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    return destination


def _library(path: Path):
    library = ctypes.CDLL(str(Path(path).resolve()))
    u32_pointer = ctypes.POINTER(ctypes.c_uint32)
    i32_pointer = ctypes.POINTER(ctypes.c_int32)
    u8_pointer = ctypes.POINTER(ctypes.c_uint8)
    f32_pointer = ctypes.POINTER(ctypes.c_float)
    library.rc64_encoder_create.restype = ctypes.c_void_p
    library.rc64_encoder_destroy.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_encode.argtypes = [
        ctypes.c_void_p,
        i32_pointer,
        u32_pointer,
        ctypes.c_size_t,
    ]
    library.rc64_encoder_encode.restype = ctypes.c_int
    library.rc64_encoder_finish.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_finish.restype = ctypes.c_int
    library.rc64_encoder_data.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_data.restype = u8_pointer
    library.rc64_encoder_size.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_size.restype = ctypes.c_size_t
    library.rc64_decoder_create.argtypes = [u8_pointer, ctypes.c_size_t]
    library.rc64_decoder_create.restype = ctypes.c_void_p
    library.rc64_decoder_destroy.argtypes = [ctypes.c_void_p]
    library.rc64_decoder_decode.argtypes = [
        ctypes.c_void_p,
        u32_pointer,
        ctypes.c_size_t,
        i32_pointer,
    ]
    library.rc64_decoder_decode.restype = ctypes.c_int
    library.rc64_decoder_decode_probabilities.argtypes = [
        ctypes.c_void_p,
        f32_pointer,
        ctypes.c_size_t,
        i32_pointer,
    ]
    library.rc64_decoder_decode_probabilities.restype = ctypes.c_int
    library.rc64_decoder_bit_position.argtypes = [ctypes.c_void_p]
    library.rc64_decoder_bit_position.restype = ctypes.c_size_t
    library.rc64_total_frequency.restype = ctypes.c_uint64
    if library.rc64_total_frequency() != TOTAL:
        raise Rc64Error("RC64 backend uses an incompatible frequency total")
    return library


class NativeEncoder:
    """Streaming wrapper around the compiled RC64 encoder."""

    def __init__(self, library_path: Path) -> None:
        self.library = _library(library_path)
        self.context = self.library.rc64_encoder_create()
        if not self.context:
            raise Rc64Error("RC64 encoder allocation failed")
        self.finished = False

    def close(self) -> None:
        if self.context:
            self.library.rc64_encoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        self.close()

    def encode_frequencies(self, symbols: np.ndarray, frequencies: np.ndarray) -> None:
        source = np.ascontiguousarray(symbols, dtype=np.int32).reshape(-1)
        rows = np.ascontiguousarray(frequencies, dtype=np.uint32)
        if rows.shape != (source.size, ALPHABET):
            raise Rc64Error("RC64 symbol/frequency shapes disagree")
        code = self.library.rc64_encoder_encode(
            self.context,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            source.size,
        )
        if code:
            raise Rc64Error(f"native RC64 encode failed with status {code}")

    def encode(self, symbols: np.ndarray, probabilities: np.ndarray) -> None:
        self.encode_frequencies(symbols, quantize_probabilities(probabilities))

    def finish(self) -> bytes:
        if not self.finished:
            code = self.library.rc64_encoder_finish(self.context)
            if code:
                raise Rc64Error(f"native RC64 finish failed with status {code}")
            self.finished = True
        size = self.library.rc64_encoder_size(self.context)
        pointer = self.library.rc64_encoder_data(self.context)
        if not size or not pointer:
            raise Rc64Error("native RC64 encoder returned an empty stream")
        return ctypes.string_at(pointer, size)


class NativeDecoder:
    """Streaming wrapper around the compiled RC64 decoder."""

    def __init__(self, library_path: Path, payload: bytes) -> None:
        self.library = _library(library_path)
        self.payload = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        self.context = self.library.rc64_decoder_create(self.payload, len(payload))
        if not self.context:
            raise Rc64Error("RC64 decoder allocation failed")

    def close(self) -> None:
        if self.context:
            self.library.rc64_decoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        self.close()

    def decode_frequencies(self, frequencies: np.ndarray) -> np.ndarray:
        rows = np.ascontiguousarray(frequencies, dtype=np.uint32)
        if rows.ndim != 2 or rows.shape[1] != ALPHABET or not rows.size:
            raise Rc64Error("RC64 frequency rows must have shape [N, 5]")
        output = np.empty(rows.shape[0], dtype=np.int32)
        code = self.library.rc64_decoder_decode(
            self.context,
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            rows.shape[0],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if code:
            raise Rc64Error(f"native RC64 decode failed with status {code}")
        return output

    def decode(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ALPHABET or not values.size:
            raise Rc64Error("RC64 probabilities must have shape [N, 5]")
        output = np.empty(values.shape[0], dtype=np.int32)
        code = self.library.rc64_decoder_decode_probabilities(
            self.context,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.shape[0],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if code:
            raise Rc64Error(f"native RC64 probability decode failed with status {code}")
        return output

    @property
    def bit_position(self) -> int:
        return int(self.library.rc64_decoder_bit_position(self.context))
