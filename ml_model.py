"""
Machine-learning number ranking for Maryland Multi-Match.

Trains a logistic regression per number in the pool on features built
*causally* from draw history -- every feature for a training row at draw t
uses only draws before t, so the model never sees the future it is scored
against:

  - all-time frequency
  - rolling frequency over the last two windows (recent form)
  - overdue gap since the number last appeared
  - pairing momentum with the numbers drawn immediately before it

Held-out ROC-AUC is reported so the model's real skill is visible. Multi-Match
draws are independent random events -- an AUC near 0.5 (coin flip) is
expected, and no model, this one included, can predict them. This module
ranks numbers by historical pattern; it does not, and cannot, produce
"accurate" winning numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Iterable

import numpy as np

try:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when deps are missing
    SKLEARN_AVAILABLE = False

WARMUP = 40          # draws of history required before a row is trainable
HOLDOUT_N = 20        # most recent draws held out for AUC evaluation
REC_WINDOWS = (10, 20)
GAP_CAP = 60
MIN_DRAWS_REQUIRED = WARMUP + HOLDOUT_N + 10


class InsufficientHistory(Exception):
    """Raised when ML deps are missing or there isn't enough history to train on."""


@dataclass(frozen=True)
class MLReport:
    ranked_numbers: tuple[tuple[int, float], ...]  # (number, probability) desc by probability
    auc: float | None
    n_draws: int
    holdout_size: int


def _history_frame(history: Iterable, draw_size: int) -> "pd.DataFrame":
    rows = sorted(history, key=lambda r: r.draw_date)
    cols = [f"Num{i}" for i in range(1, draw_size + 1)]
    data = [
        {"draw_date": r.draw_date, **dict(zip(cols, sorted(r.numbers)))}
        for r in rows
    ]
    return pd.DataFrame(data)


def _presence_matrix(values: np.ndarray, pool_size: int) -> np.ndarray:
    n_draws = values.shape[0]
    presence = np.zeros((n_draws, pool_size + 1), dtype=np.float64)
    for t in range(n_draws):
        presence[t, values[t]] = 1.0
    return presence


def _last_seen_before(presence: np.ndarray) -> np.ndarray:
    n_draws, width = presence.shape
    out = np.full((n_draws, width), -1, dtype=np.int64)
    seen = np.full(width, -1, dtype=np.int64)
    for t in range(n_draws):
        out[t] = seen
        seen[presence[t] > 0] = t
    return out


def _pair_momentum(main_matrix: np.ndarray, pool_size: int):
    n_draws = main_matrix.shape[0]
    pair_counts = np.zeros((pool_size + 1, pool_size + 1), dtype=np.float64)
    out = np.zeros((n_draws, pool_size + 1), dtype=np.float64)
    for t in range(1, n_draws):
        prev = main_matrix[t - 1]
        for a, b in combinations(sorted(prev), 2):
            pair_counts[a, b] += 1
            pair_counts[b, a] += 1
        out[t] = pair_counts[:, prev].sum(axis=1)
    return out, pair_counts


def _build_features(df: "pd.DataFrame", pool_size: int, draw_size: int):
    cols = [f"Num{i}" for i in range(1, draw_size + 1)]
    main_matrix = df[cols].values.astype(int)
    n_draws = main_matrix.shape[0]

    presence = _presence_matrix(main_matrix, pool_size)
    cum = np.vstack([np.zeros((1, pool_size + 1)), np.cumsum(presence, axis=0)])
    last_seen_before = _last_seen_before(presence)
    pair_before, pair_counts_final = _pair_momentum(main_matrix, pool_size)

    ts = np.arange(WARMUP, n_draws)
    freq_all = cum[ts] / ts[:, None]
    rec = {
        w: (cum[ts] - cum[np.maximum(ts - w, 0)]) / np.minimum(w, ts)[:, None]
        for w in REC_WINDOWS
    }
    gap = np.where(last_seen_before[ts] >= 0, ts[:, None] - last_seen_before[ts], ts[:, None])
    gap_norm = np.minimum(gap, GAP_CAP) / GAP_CAP
    pair = pair_before[ts]
    target = presence[ts]

    X_blocks, y_blocks, ts_blocks = [], [], []
    for n in range(1, pool_size + 1):
        Xn = np.stack(
            [freq_all[:, n], rec[REC_WINDOWS[0]][:, n], rec[REC_WINDOWS[1]][:, n],
             gap_norm[:, n], pair[:, n]],
            axis=1,
        )
        X_blocks.append(Xn)
        y_blocks.append(target[:, n])
        ts_blocks.append(ts)
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    ts_full = np.concatenate(ts_blocks)

    last_draw = main_matrix[-1]
    for a, b in combinations(sorted(last_draw), 2):
        pair_counts_final[a, b] += 1
        pair_counts_final[b, a] += 1
    pair_next = pair_counts_final[:, last_draw].sum(axis=1)

    next_feats = {}
    for n in range(1, pool_size + 1):
        f_all = cum[n_draws, n] / n_draws
        f_rec = {
            w: (cum[n_draws, n] - cum[max(n_draws - w, 0), n]) / min(w, n_draws)
            for w in REC_WINDOWS
        }
        seen = last_seen_before[-1, n] if last_seen_before[-1, n] >= 0 else -1
        seen = n_draws - 1 if n in last_draw else seen
        g = (n_draws - seen) if seen >= 0 else n_draws
        g_norm = min(g, GAP_CAP) / GAP_CAP
        next_feats[n] = [f_all, f_rec[REC_WINDOWS[0]], f_rec[REC_WINDOWS[1]], g_norm, pair_next[n]]

    return X, y, ts_full, next_feats, n_draws


def _train_and_score(X, y, ts_full, n_draws, next_feats, seed):
    split_t = max(n_draws - HOLDOUT_N, WARMUP + 1)
    train_mask = ts_full < split_t
    test_mask = ~train_mask

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(X[train_mask], y[train_mask])

    auc = None
    if test_mask.sum() > 0 and len(set(y[test_mask])) > 1:
        preds = model.predict_proba(X[test_mask])[:, 1]
        auc = roc_auc_score(y[test_mask], preds)

    prob = {n: model.predict_proba([feats])[0, 1] for n, feats in next_feats.items()}
    return prob, auc


def train_number_model(
    history: Iterable,
    pool_size: int,
    draw_size: int,
    seed: int | None = None,
) -> MLReport:
    """Train the per-number model on history and return a ranked report for the
    not-yet-played next draw. Raises InsufficientHistory if scikit-learn/pandas
    aren't installed or there isn't enough history to train reliably."""
    if not SKLEARN_AVAILABLE:
        raise InsufficientHistory("scikit-learn/pandas are not installed.")
    history = tuple(history)
    if len(history) < MIN_DRAWS_REQUIRED:
        raise InsufficientHistory(
            f"Need at least {MIN_DRAWS_REQUIRED} draws of history to train reliably; "
            f"have {len(history)}."
        )

    df = _history_frame(history, draw_size)
    X, y, ts_full, next_feats, n_draws = _build_features(df, pool_size, draw_size)
    prob, auc = _train_and_score(X, y, ts_full, n_draws, next_feats, seed)
    ranked = tuple(sorted(prob.items(), key=lambda item: (-item[1], item[0])))
    holdout_size = min(HOLDOUT_N, n_draws - WARMUP)
    return MLReport(ranked_numbers=ranked, auc=auc, n_draws=n_draws, holdout_size=holdout_size)


def _weighted_sample(rng: Random, population: list, weights: list, k: int) -> list:
    pool = list(zip(population, weights))
    result = []
    while len(result) < k:
        total = sum(w for _, w in pool)
        r = rng.uniform(0, total)
        running = 0.0
        for i, (item, w) in enumerate(pool):
            running += w
            if r <= running:
                result.append(item)
                pool.pop(i)
                break
    return result


def generate_ml_line(report: MLReport, rng: Random, draw_size: int) -> tuple[int, ...]:
    population = [n for n, _ in report.ranked_numbers]
    weights = [max(p, 1e-6) for _, p in report.ranked_numbers]
    return tuple(sorted(_weighted_sample(rng, population, weights, draw_size)))


def generate_ml_tickets(
    report: MLReport,
    num_tickets: int,
    lines_per_ticket: int,
    draw_size: int,
    seed: int | None = None,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Weighted-random tickets sampled from the model's predicted probabilities.
    A higher-probability number is more likely to be picked, but every number
    keeps a nonzero chance -- this is a weighting, not a guarantee."""
    rng = Random(seed)
    tickets = []
    for _ in range(num_tickets):
        ticket: list[tuple[int, ...]] = []
        while len(ticket) < lines_per_ticket:
            line = generate_ml_line(report, rng, draw_size)
            if line not in ticket:
                ticket.append(line)
        tickets.append(tuple(ticket))
    return tuple(tickets)
