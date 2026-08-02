"""Frozen-token temporal entropy opportunity audit for Phase 2.

This module deliberately treats the deployed CPR1 HPAC payload as read-only.
It can replay its exact causal probability tables to measure the present token
stream, then compares deliberately offline prior/motion/SSM *estimates*.  None
of the experiments here changes the production HPAC decoder or token stream.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NUM_CLASSES = 5


class EntropyAuditError(RuntimeError):
    """The frozen-token audit cannot establish a reproducible measurement."""


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probability = counts[counts > 0].astype(np.float64) / total
    return float(-(probability * np.log2(probability)).sum())


def quantiles(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {str(q): 0.0 for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}
    return {
        str(q): float(np.percentile(values, q))
        for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)
    }


def boundary_buckets(previous: np.ndarray, max_distance: int = 4) -> np.ndarray:
    """Return distance-to-semantic-boundary buckets for one HxW token frame.

    A bucket of zero is an edge touching a different class.  Distances larger
    than ``max_distance`` are saturated.  The morphology is NumPy-only so the
    definition is deterministic across the constrained development rails.
    """
    if previous.ndim != 2:
        raise EntropyAuditError("boundary source must be one HxW token frame")
    edge = np.zeros(previous.shape, dtype=bool)
    edge[1:] |= previous[1:] != previous[:-1]
    edge[:-1] |= previous[:-1] != previous[1:]
    edge[:, 1:] |= previous[:, 1:] != previous[:, :-1]
    edge[:, :-1] |= previous[:, :-1] != previous[:, 1:]
    result = np.full(previous.shape, max_distance, dtype=np.uint8)
    active = edge.copy()
    result[active] = 0
    for distance in range(1, max_distance):
        grown = active.copy()
        grown[1:] |= active[:-1]
        grown[:-1] |= active[1:]
        grown[:, 1:] |= active[:, :-1]
        grown[:, :-1] |= active[:, 1:]
        active = grown
        result[(result == max_distance) & active] = distance
    return result


def transition_counts(tokens: np.ndarray) -> np.ndarray:
    """Return CxC causal previous-token/current-token counts."""
    if tokens.ndim != 3 or tokens.shape[0] < 2:
        raise EntropyAuditError("tokens must be at least two FxHxW frames")
    previous = tokens[:-1].reshape(-1)
    current = tokens[1:].reshape(-1)
    return np.bincount(
        previous * NUM_CLASSES + current, minlength=NUM_CLASSES**2
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def conditional_entropy(tokens: np.ndarray, predictors: np.ndarray) -> float:
    """Empirical H(tokens | predictors) in bits/token for matching arrays."""
    if tokens.shape != predictors.shape:
        raise EntropyAuditError("conditional-entropy arrays must have identical shapes")
    if tokens.size == 0:
        return 0.0
    flat_t = tokens.reshape(-1).astype(np.int64)
    flat_p = predictors.reshape(-1).astype(np.int64)
    cardinality = int(flat_p.max(initial=0)) + 1
    counts = np.bincount(
        flat_p * NUM_CLASSES + flat_t, minlength=cardinality * NUM_CLASSES
    ).reshape(cardinality, NUM_CLASSES)
    total = int(counts.sum())
    likelihood = 0.0
    for row in counts:
        row_total = int(row.sum())
        if row_total:
            likelihood += row_total * _entropy_from_counts(row)
    return float(likelihood / total)


def _context_entropy(
    tokens: np.ndarray, context: np.ndarray, table_bytes: int, label: str
) -> dict[str, Any]:
    if tokens.shape != context.shape:
        raise EntropyAuditError("prior token and context tensors must match")
    flat_tokens = tokens.reshape(-1).astype(np.int64)
    flat_context = context.reshape(-1).astype(np.int64)
    categories = int(flat_context.max(initial=0)) + 1
    counts = np.bincount(
        flat_context * NUM_CLASSES + flat_tokens, minlength=categories * NUM_CLASSES
    ).reshape(categories, NUM_CLASSES)
    entropy = 0.0
    for row in counts:
        entropy += int(row.sum()) * _entropy_from_counts(row)
    return {
        "prior": label,
        "context_states": categories,
        "quantized_table_bytes": table_bytes,
        "offline_empirical_bits_per_token": float(entropy / flat_tokens.size),
        "offline_estimated_stream_bytes": math.ceil(entropy / 8.0) + table_bytes,
        "method": "offline empirical conditional entropy; not a deployable re-encode",
    }


def simple_prior_study(tokens: np.ndarray) -> list[dict[str, Any]]:
    """Measure P0--P4 specified in the Phase-2 audit handoff.

    The code lengths use full-corpus empirical tables, so they are opportunity
    estimates, not valid encoder claims.  Their table byte charges are explicit
    int8-logit storage estimates.
    """
    if tokens.ndim != 3 or tokens.dtype != np.uint8:
        raise EntropyAuditError("expected uint8 FxHxW tokens")
    frames, height, width = tokens.shape
    previous = tokens[:-1]
    current = tokens[1:]
    row16 = np.broadcast_to(
        (np.arange(height) * 16 // height)[:, None], (height, width)
    )[None]
    row16 = np.broadcast_to(row16, previous.shape)
    boundary = np.stack([boundary_buckets(frame) for frame in tokens[:-1]], axis=0)
    segment16 = np.broadcast_to(
        (np.arange(1, frames) * 16 // frames)[:, None, None], previous.shape
    )
    p0 = _context_entropy(
        current, previous, NUM_CLASSES * NUM_CLASSES, "P0 previous-class transition"
    )
    p1_context = previous.astype(np.int64) * 16 + row16
    p1 = _context_entropy(
        current,
        p1_context,
        NUM_CLASSES * 16 * NUM_CLASSES,
        "P1 previous class + 16 vertical row bands",
    )
    p2_context = previous.astype(np.int64) * 5 + boundary
    p2 = _context_entropy(
        current,
        p2_context,
        NUM_CLASSES * 5 * NUM_CLASSES,
        "P2 previous class + boundary-distance bucket",
    )
    p3_context = previous.astype(np.int64) * 16 + segment16
    p3 = _context_entropy(
        current,
        p3_context,
        NUM_CLASSES * 16 * NUM_CLASSES,
        "P3 previous class + 16 temporal segments",
    )
    # P4 is an additive logit bank. Its byte cost is the four separately
    # quantized [feature-state, class] matrices rather than a cross product.
    factors = (previous, row16, boundary, segment16)
    logits = np.zeros(previous.shape + (NUM_CLASSES,), dtype=np.float64)
    for factor in factors:
        states = int(factor.max(initial=0)) + 1
        counts = np.bincount(
            factor.reshape(-1).astype(np.int64) * NUM_CLASSES + current.reshape(-1),
            minlength=states * NUM_CLASSES,
        ).reshape(states, NUM_CLASSES)
        probs = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 0.5 * NUM_CLASSES)
        logits += np.log(probs[factor])
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    selected = probs.reshape(-1, NUM_CLASSES)[
        np.arange(current.size), current.reshape(-1)
    ]
    p4_bits = float(-np.log2(selected).sum())
    p4_table = (NUM_CLASSES + 16 + 5 + 16) * NUM_CLASSES
    p4 = {
        "prior": "P4 additive [previous class, row band, boundary state, temporal segment]",
        "context_states": "additive",
        "quantized_table_bytes": p4_table,
        "offline_empirical_bits_per_token": p4_bits / current.size,
        "offline_estimated_stream_bytes": math.ceil(p4_bits / 8.0) + p4_table,
        "method": "offline additive int8-logit opportunity estimate; not a deployable re-encode",
    }
    return [p0, p1, p2, p3, p4]


def _translate(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate with edge replication, avoiding circular roll artefacts."""
    height, width = frame.shape
    rows = np.clip(np.arange(height) - dy, 0, height - 1)
    cols = np.clip(np.arange(width) - dx, 0, width - 1)
    return frame[np.ix_(rows, cols)]


def _vertical_scale(frame: np.ndarray, scale: float) -> np.ndarray:
    """Nearest-neighbour vertical scale about image centre with edge padding."""
    height = frame.shape[0]
    source_rows = np.clip(
        np.rint(
            (np.arange(height) - (height - 1) / 2) / scale + (height - 1) / 2
        ).astype(np.int64),
        0,
        height - 1,
    )
    return frame[source_rows]


def _huffman_bits(symbols: Iterable[int]) -> int:
    values = np.asarray(list(symbols), dtype=np.int64)
    if not values.size:
        return 0
    counts = np.bincount(values)
    # A compact deterministic Huffman length calculation without a bitstream.
    import heapq

    queue = [int(value) for value in counts if value]
    if len(queue) == 1:
        return int(values.size)
    heapq.heapify(queue)
    total = 0
    while len(queue) > 1:
        left = heapq.heappop(queue)
        right = heapq.heappop(queue)
        merged = left + right
        total += merged
        heapq.heappush(queue, merged)
    return total


@dataclass(frozen=True)
class MotionResult:
    mode: str
    codebook: list[tuple[int, int]]
    mode_ids: np.ndarray
    warped: np.ndarray
    side_bytes_fixed: int
    side_bytes_huffman: int


def _best_shift(
    previous: np.ndarray, current: np.ndarray, shifts: list[tuple[int, int]]
) -> int:
    best = 0
    best_error = previous.size + 1
    for index, (dx, dy) in enumerate(shifts):
        error = int((_translate(previous, dx, dy) != current).sum())
        if error < best_error:
            best, best_error = index, error
    return best


def motion_oracle(
    tokens: np.ndarray, baseline_hpac_nll_bpt: float
) -> list[dict[str, Any]]:
    """Run the required M0--M4 causal previous-token warp opportunity audit."""
    if tokens.ndim != 3:
        raise EntropyAuditError("motion oracle expects FxHxW tokens")
    frames = tokens.shape[0]
    shifts = [
        (dx, dy) for dy in (-8, -4, -2, 0, 2, 4, 8) for dx in (-8, -4, -2, 0, 2, 4, 8)
    ]
    previous, current = tokens[:-1], tokens[1:]
    results: list[MotionResult] = []
    identity = np.array([_translate(frame, 0, 0) for frame in previous], dtype=np.uint8)
    results.append(
        MotionResult(
            "M0 identity",
            [(0, 0)],
            np.zeros(frames - 1, dtype=np.uint8),
            identity,
            0,
            0,
        )
    )
    ids = np.array(
        [_best_shift(p, c, shifts) for p, c in zip(previous, current, strict=True)],
        dtype=np.uint8,
    )
    warped = np.array(
        [_translate(p, *shifts[index]) for p, index in zip(previous, ids, strict=True)],
        dtype=np.uint8,
    )
    results.append(
        MotionResult(
            "M1 7x7 translation",
            shifts,
            ids,
            warped,
            math.ceil((frames - 1) * math.log2(len(shifts)) / 8),
            math.ceil(_huffman_bits(ids) / 8),
        )
    )
    scales = (0.94, 0.97, 1.00, 1.03, 1.06)
    m2_transforms = [(dx, dy, scale) for scale in scales for dx, dy in shifts]
    m2_ids = np.zeros(frames - 1, dtype=np.uint16)
    m2_warp = np.empty_like(current)
    # Select each scale on the full token grid.  This is an oracle search,
    # not a deployable motion estimator, and the selected code is charged.
    for frame_index, (p, c) in enumerate(zip(previous, current, strict=True)):
        best_id, best_error, best_warp = 0, p.size + 1, p
        for scale_index, scale in enumerate(scales):
            scaled = _vertical_scale(p, scale)
            shift_id = _best_shift(scaled, c, shifts)
            candidate = _translate(scaled, *shifts[shift_id])
            error = int((candidate != c).sum())
            if error < best_error:
                best_id, best_error, best_warp = (
                    scale_index * len(shifts) + shift_id,
                    error,
                    candidate,
                )
        m2_ids[frame_index] = best_id
        m2_warp[frame_index] = best_warp
    results.append(
        MotionResult(
            "M2 translation + vertical scale centre",
            m2_transforms,
            m2_ids,
            m2_warp,
            math.ceil((frames - 1) * math.log2(len(m2_transforms)) / 8),
            math.ceil(_huffman_bits(m2_ids) / 8),
        )
    )
    band_ids = np.zeros((frames - 1, 4), dtype=np.uint8)
    band_warp = np.empty_like(current)
    height = tokens.shape[1]
    for frame_index, (p, c) in enumerate(zip(previous, current, strict=True)):
        for band in range(4):
            start, end = band * height // 4, (band + 1) * height // 4
            chosen = _best_shift(p[start:end], c[start:end], shifts)
            band_ids[frame_index, band] = chosen
            band_warp[frame_index, start:end] = _translate(p, *shifts[chosen])[
                start:end
            ]
    results.append(
        MotionResult(
            "M3 four horizontal-band translations",
            shifts,
            band_ids.reshape(-1),
            band_warp,
            math.ceil((frames - 1) * 4 * math.log2(len(shifts)) / 8),
            math.ceil(_huffman_bits(band_ids.reshape(-1)) / 8),
        )
    )
    common = np.bincount(ids, minlength=len(shifts)).argsort()[-16:]
    common.sort()
    compact = [shifts[index] for index in common]
    m4_ids = np.array(
        [_best_shift(p, c, compact) for p, c in zip(previous, current, strict=True)],
        dtype=np.uint8,
    )
    m4_warp = np.array(
        [
            _translate(p, *compact[index])
            for p, index in zip(previous, m4_ids, strict=True)
        ],
        dtype=np.uint8,
    )
    results.append(
        MotionResult(
            "M4 learned global shared 16-transform codebook",
            compact,
            m4_ids,
            m4_warp,
            len(compact) * 2 + math.ceil((frames - 1) * 4 / 8),
            len(compact) * 2 + math.ceil(_huffman_bits(m4_ids) / 8),
        )
    )
    if not math.isfinite(baseline_hpac_nll_bpt) or baseline_hpac_nll_bpt < 0:
        raise EntropyAuditError(
            "motion study requires a non-negative measured HPAC NLL"
        )
    rows: list[dict[str, Any]] = []
    for result in results:
        changed = result.warped != current
        conditional = conditional_entropy(current, result.warped)
        reduction_bits = (baseline_hpac_nll_bpt - conditional) * current.size
        rows.append(
            {
                "mode": result.mode,
                "transform_count": len(result.codebook),
                "mode_entropy_bits": _entropy_from_counts(
                    np.bincount(result.mode_ids, minlength=max(1, len(result.codebook)))
                ),
                "mode_stream_fixed_bytes": result.side_bytes_fixed,
                "mode_stream_huffman_bytes": result.side_bytes_huffman,
                "warped_change_rate": float(changed.mean()),
                "conditional_entropy_bits_per_token": conditional,
                "estimated_hpac_nll_reduction_bytes": reduction_bits / 8.0,
                "estimated_net_saving_bytes_fixed": reduction_bits / 8.0
                - result.side_bytes_fixed,
                "estimated_net_saving_bytes_huffman": reduction_bits / 8.0
                - result.side_bytes_huffman,
                "interpretation": "offline entropy bound; no HPAC decoder or token stream was modified",
            }
        )
    return rows


def ssm_feasibility(tokens: np.ndarray) -> list[dict[str, Any]]:
    """Produce a conservative structural feasibility ledger for tiny diagonal SSMs.

    No learned recurrent model is smuggled into this study.  The fixed-point
    recurrence is simulated on per-frame class histograms solely to validate
    canonical state ranges and byte charges.  Therefore every row is rejected
    as insufficient evidence for a production gate.
    """
    histogram = np.stack(
        [np.bincount(frame.reshape(-1), minlength=NUM_CLASSES) for frame in tokens],
        axis=0,
    ).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for rank in (2, 4, 8):
        state = np.zeros(rank, dtype=np.int16)
        maximum = 0
        for item in histogram:
            signal = int(item[0] - item[-1]) // max(1, item.size)
            state = np.clip(
                (state.astype(np.int32) * 3 + signal * (np.arange(rank) + 1)) // 4,
                -127,
                127,
            ).astype(np.int16)
            maximum = max(maximum, int(np.abs(state).max(initial=0)))
        for inputs in (
            "previous token",
            "warped previous token",
            "row bands",
            "temporal segments",
        ):
            bytes_int4 = rank * (5 + 5 + 1) // 2 + rank + 5
            bytes_int6 = math.ceil(rank * (5 + 5 + 1) * 6 / 8) + rank + 5
            bytes_int8 = rank * (5 + 5 + 1) + rank + 5
            rows.append(
                {
                    "rank": rank,
                    "inputs": inputs,
                    "integer_state": "int8",
                    "canonical_integer_logits": True,
                    "maximum_abs_simulated_state": maximum,
                    "model_bytes_int4": bytes_int4,
                    "model_bytes_int6": bytes_int6,
                    "model_bytes_int8": bytes_int8,
                    "estimated_runtime_microseconds_per_frame": float(2 * rank * 5),
                    "hpac_nll_bytes_saved": 0.0,
                    "net_saving_bytes": -float(bytes_int4),
                    "status": "rejected_no_trained_causal_model",
                    "reason": "fixed-point structural simulation only; no measured NLL improvement or exact re-encode",
                }
            )
    return rows


def adapter_feasibility() -> list[dict[str, Any]]:
    """Charge all requested HPAC adapter placements before any retraining.

    The production HPAC model is frozen in Phase 2.  These zero-initialized
    entries therefore establish overhead and exact-reencode requirements; they
    do not pretend that an untrained adapter supplies an NLL improvement.
    """
    placements = (
        ("global frame-conditioned bias", 8, 5),
        ("global + 4 temporal segments", 8 * 4, 5),
        ("global + 8 temporal segments", 8 * 8, 5),
        ("small causal student", 64, 5),
        ("low-rank conv_past", 5, 64),
        ("low-rank frame conditioning", 8, 64),
        ("low-rank head", 64, 5),
        ("low-rank logits", 64, 5),
    )
    rows: list[dict[str, Any]] = []
    for placement, inputs, outputs in placements:
        for rank in (1, 2, 4):
            parameter_count = rank * (inputs + outputs) + outputs
            for bits in (4, 6, 8):
                storage = math.ceil(parameter_count * bits / 8)
                rows.append(
                    {
                        "placement": placement,
                        "rank": rank,
                        "weight_bits": bits,
                        "parameter_count": parameter_count,
                        "charged_adapter_bytes": storage,
                        "measured_hpac_nll_delta_bits": 0.0,
                        "estimated_net_saving_bytes": -float(storage),
                        "exact_token_reencode": False,
                        "status": "not_trained_no_candidate",
                        "reason": "Phase-2 frozen-production feasibility ledger; train/re-encode only after a separately justified >=5 KiB opportunity",
                    }
                )
    return rows


def strata_aggregates(
    tokens: np.ndarray, probability_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate saved exact-replay NLL maps over required causal strata."""
    if tokens.ndim != 3:
        raise EntropyAuditError("strata aggregation expects FxHxW tokens")
    frames, height, width = tokens.shape
    sums: dict[str, dict[int, list[float]]] = {
        "class": {},
        "row_band_8": {},
        "row_band_16": {},
        "boundary_distance": {},
        "changed": {},
        "temporal_segment_4": {},
        "temporal_segment_8": {},
        "temporal_segment_16": {},
    }

    def add(name: str, bucket: np.ndarray | int, nll: np.ndarray) -> None:
        values = (
            np.full(nll.shape, bucket, dtype=np.int64)
            if isinstance(bucket, int)
            else bucket.astype(np.int64, copy=False)
        )
        flat_bucket = values.reshape(-1)
        flat_nll = nll.reshape(-1).astype(np.float64)
        for key in np.unique(flat_bucket):
            selected = flat_bucket == key
            total = sums[name].setdefault(int(key), [0.0, 0.0])
            total[0] += float(flat_nll[selected].sum())
            total[1] += float(selected.sum())

    row8 = np.broadcast_to((np.arange(height) * 8 // height)[:, None], (height, width))
    row16 = np.broadcast_to(
        (np.arange(height) * 16 // height)[:, None], (height, width)
    )
    for frame in range(frames):
        path = probability_dir / f"frame_{frame:03d}.npz"
        if not path.is_file():
            raise EntropyAuditError(f"missing exact-replay probability summary: {path}")
        with np.load(path) as stored:
            nll = stored["nll_bits"].astype(np.float32)
        add("class", tokens[frame], nll)
        add("row_band_8", row8, nll)
        add("row_band_16", row16, nll)
        add("temporal_segment_4", frame * 4 // frames, nll)
        add("temporal_segment_8", frame * 8 // frames, nll)
        add("temporal_segment_16", frame * 16 // frames, nll)
        if frame == 0:
            add("boundary_distance", 4, nll)
            add("changed", 1, nll)
        else:
            add("boundary_distance", boundary_buckets(tokens[frame - 1]), nll)
            add("changed", (tokens[frame] != tokens[frame - 1]).astype(np.uint8), nll)
    return {
        name: [
            {
                "bucket": bucket,
                "token_count": int(values[1]),
                "hpac_nll_bits": values[0],
                "hpac_nll_bits_per_token": values[0] / values[1],
            }
            for bucket, values in sorted(buckets.items())
        ]
        for name, buckets in sums.items()
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EntropyAuditError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
