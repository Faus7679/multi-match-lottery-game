"""
Position-based machine-learning number selection for Maryland Multi-Match.

Each draw is sorted, so it has six positions (n1 <= ... <= n6) with very
different ranges -- n1 is usually small, n6 usually large. Instead of ranking
numbers regardless of where they land (see ml_model.py), this module trains
one logistic regression *per position* that scores every candidate number
for that slot, then builds lines by choosing position by position from each
position's own probabilities. Picks are not forced into ascending order.

Features for (draw t, position p, candidate number n) are built causally --
only draws before t are used:

  - all-time frequency of n at position p
  - frequency of n at position p over the last two windows (recent form)
  - draws since n last landed at position p (capped)
  - frequency of n at *any* position over the last window
  - distance from n to the number that held position p in the previous draw

Training is tuned by the same walk-forward cross-validation as ml_model.py
(grid search over regularization strength; every fold trains strictly on draws
before its test window). Reported accuracy is the top-1 hit rate per position
-- how often the model's single best number for that slot was the number
actually drawn there -- next to a frequency-only baseline (always guess the
number historically most common at that position). Multi-Match draws are
independent random events, so expect the model to land near that baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

import numpy as np

from ml_model import (
    C_GRID,
    GAP_CAP,
    MIN_DRAWS_REQUIRED,
    REC_WINDOWS,
    WARMUP,
    InsufficientHistory,
    SKLEARN_AVAILABLE,
    _fit,
    _fold_bounds,
    _history_frame,
    _last_seen_before,
)

if SKLEARN_AVAILABLE:
    from sklearn.metrics import roc_auc_score

ANY_POSITION_WINDOW = REC_WINDOWS[1]


@dataclass(frozen=True)
class PositionalReport:
    position_probs: tuple[tuple[float, ...], ...]  # [position][number], index 0 unused
    best_line: tuple[int, ...]                      # most likely number at each position, in position order
    position_accuracy: tuple[float | None, ...]     # top-1 hit rate per position across CV folds
    position_baseline: tuple[float | None, ...]     # same, for a frequency-only guess
    auc: float | None                               # mean ROC-AUC across positions and folds
    best_C: float | None
    n_draws: int
    holdout_size: int


def _position_features(values: np.ndarray, position: int, pool_size: int):
    """Feature tensor for every t in WARMUP..n_draws (inclusive -- the last row is
    the not-yet-played next draw) and every candidate number 1..pool_size.
    Returns (X, y, ts) with rows ordered draw-major, number-minor; y is 0 for the
    unplayed final draw."""
    n_draws = values.shape[0]
    width = pool_size + 1

    pos_presence = np.zeros((n_draws + 1, width))
    pos_presence[np.arange(n_draws), values[:, position]] = 1.0
    any_presence = np.zeros((n_draws + 1, width))
    for col in range(values.shape[1]):
        any_presence[np.arange(n_draws), values[:, col]] = 1.0

    pos_cum = np.vstack([np.zeros((1, width)), np.cumsum(pos_presence[:n_draws], axis=0)])
    any_cum = np.vstack([np.zeros((1, width)), np.cumsum(any_presence[:n_draws], axis=0)])
    last_seen = _last_seen_before(pos_presence)

    ts = np.arange(WARMUP, n_draws + 1)
    freq_all = pos_cum[ts] / ts[:, None]
    rec = [
        (pos_cum[ts] - pos_cum[np.maximum(ts - w, 0)]) / np.minimum(w, ts)[:, None]
        for w in REC_WINDOWS
    ]
    gap = np.where(last_seen[ts] >= 0, ts[:, None] - last_seen[ts], ts[:, None])
    gap_norm = np.minimum(gap, GAP_CAP) / GAP_CAP
    w = ANY_POSITION_WINDOW
    any_rec = (any_cum[ts] - any_cum[np.maximum(ts - w, 0)]) / np.minimum(w, ts)[:, None]
    numbers = np.arange(width)
    prev_dist = np.abs(numbers[None, :] - values[ts - 1, position][:, None]) / pool_size

    features = np.stack([freq_all, rec[0], rec[1], gap_norm, any_rec, prev_dist], axis=2)
    features = features[:, 1:, :]                      # drop the unused number-0 column
    X = features.reshape(-1, features.shape[2])
    y = pos_presence[ts][:, 1:].reshape(-1)
    return X, y, np.repeat(ts, pool_size)


def _top1_hits(scores: np.ndarray, y: np.ndarray, pool_size: int) -> float:
    """Fraction of draws where the highest-scored number is the one drawn at this
    position. Rows are draw-major, so reshaping gives one row per draw."""
    s = scores.reshape(-1, pool_size)
    truth = y.reshape(-1, pool_size)
    return float(truth[np.arange(len(s)), s.argmax(axis=1)].mean())


def _walk_forward_eval(data, n_draws, pool_size, C, seed):
    """Per-position mean AUC / top-1 accuracy / frequency-baseline accuracy over
    the rolling folds, plus draws evaluated per position."""
    per_pos = []
    n_evaluated = 0
    for X, y, ts in data:
        aucs, hits, base = [], [], []
        n_evaluated = 0
        for test_start, test_end in _fold_bounds(n_draws):
            train = ts < test_start
            test = (ts >= test_start) & (ts < test_end)
            model = _fit(X[train], y[train], C, seed)
            preds = model.predict_proba(X[test])[:, 1]
            aucs.append(roc_auc_score(y[test], preds))
            hits.append(_top1_hits(preds, y[test], pool_size))
            base.append(_top1_hits(X[test][:, 0], y[test], pool_size))
            n_evaluated += test_end - test_start
        per_pos.append((float(np.mean(aucs)), float(np.mean(hits)), float(np.mean(base))))
    return per_pos, n_evaluated


def train_positional_model(
    history: Iterable,
    pool_size: int,
    draw_size: int,
    seed: int | None = None,
) -> PositionalReport:
    """Train one model per draw position and return next-draw probabilities for
    every number at every position. Raises InsufficientHistory if scikit-learn/
    pandas aren't installed or there isn't enough history to train reliably."""
    if not SKLEARN_AVAILABLE:
        raise InsufficientHistory("scikit-learn/pandas are not installed.")
    history = tuple(history)
    if len(history) < MIN_DRAWS_REQUIRED:
        raise InsufficientHistory(
            f"Need at least {MIN_DRAWS_REQUIRED} draws of history to train reliably; "
            f"have {len(history)}."
        )

    df = _history_frame(history, draw_size)
    values = df[[f"Num{i}" for i in range(1, draw_size + 1)]].values.astype(int)
    n_draws = values.shape[0]

    data = [_position_features(values, p, pool_size) for p in range(draw_size)]
    # The final draw's rows have no outcome yet: they are scored, never trained on.
    trainable = [(X[ts < n_draws], y[ts < n_draws], ts[ts < n_draws]) for X, y, ts in data]

    best_C, best_auc, best_eval, n_eval = C_GRID[0], -1.0, None, 0
    for C in C_GRID:
        per_pos, n_evaluated = _walk_forward_eval(trainable, n_draws, pool_size, C, seed)
        mean_auc = float(np.mean([r[0] for r in per_pos]))
        if mean_auc > best_auc:
            best_C, best_auc, best_eval, n_eval = C, mean_auc, per_pos, n_evaluated

    # Production models refit on all history at the tuned C.
    probs = []
    for (X, y, ts), (X_tr, y_tr, _) in zip(data, trainable):
        model = _fit(X_tr, y_tr, best_C, seed)
        row = np.zeros(pool_size + 1)
        row[1:] = model.predict_proba(X[ts == n_draws])[:, 1]
        probs.append(tuple(float(v) for v in row))

    return PositionalReport(
        position_probs=tuple(probs),
        best_line=_greedy_line(probs, pool_size, draw_size),
        position_accuracy=tuple(r[1] for r in best_eval),
        position_baseline=tuple(r[2] for r in best_eval),
        auc=best_auc,
        best_C=best_C,
        n_draws=n_draws,
        holdout_size=n_eval,
    )


def _greedy_line(probs, pool_size: int, draw_size: int) -> tuple[int, ...]:
    """Most likely number for each position, in position order (n1..n6). Not forced
    ascending; a number already taken by an earlier position is skipped."""
    line: list[int] = []
    for p in range(draw_size):
        pick = max((n for n in range(1, pool_size + 1) if n not in line), key=lambda n: probs[p][n])
        line.append(pick)
    return tuple(line)


def generate_positional_line(report: PositionalReport, rng: Random, pool_size: int, draw_size: int) -> tuple[int, ...]:
    """One line, n1..n6 in position order. Each position draws from its own
    probabilities over every number not already used in the line -- there is no
    ascending constraint, so the result is returned exactly as picked."""
    line: list[int] = []
    for p in range(draw_size):
        candidates = [n for n in range(1, pool_size + 1) if n not in line]
        weights = [max(report.position_probs[p][n], 1e-6) for n in candidates]
        line.append(rng.choices(candidates, weights=weights, k=1)[0])
    return tuple(line)


def generate_positional_tickets(
    report: PositionalReport,
    num_tickets: int,
    lines_per_ticket: int,
    pool_size: int,
    draw_size: int,
    seed: int | None = None,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Tickets whose lines are sampled position by position from each position's
    predicted probabilities, in position order rather than sorted. Higher-probability
    numbers are likelier, but every number keeps a nonzero chance -- a weighting,
    not a guarantee."""
    rng = Random(seed)
    tickets = []
    for _ in range(num_tickets):
        ticket: list[tuple[int, ...]] = []
        while len(ticket) < lines_per_ticket:
            line = generate_positional_line(report, rng, pool_size, draw_size)
            # Same six numbers in a different order is still the same line to play.
            if all(set(line) != set(other) for other in ticket):
                ticket.append(line)
        tickets.append(tuple(ticket))
    return tuple(tickets)
