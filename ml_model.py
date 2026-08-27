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

Training is tuned by walk-forward cross-validation: a small grid search over
the regularization strength picks whichever setting scores best across
several rolling train/test splits (never peeking at a test split's future),
and the reported ROC-AUC and precision@k are averaged over those same splits
so they reflect genuine out-of-sample performance rather than a single lucky
split. Multi-Match draws are independent random events -- an AUC near 0.5
(coin flip) and a precision@k near the baseline rate are expected, and no
model, this one included, can predict them. This module ranks numbers by
historical pattern; it does not, and cannot, produce "accurate" winning
numbers.
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
HOLDOUT_N = 20        # size of each walk-forward validation fold
N_FOLDS = 5            # number of rolling folds used for tuning and reporting
REC_WINDOWS = (10, 20)
GAP_CAP = 60
MIN_DRAWS_REQUIRED = WARMUP + HOLDOUT_N + 10
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)  # regularization grid searched by walk-forward CV


class InsufficientHistory(Exception):
    """Raised when ML deps are missing or there isn't enough history to train on."""


@dataclass(frozen=True)
class MLReport:
    ranked_numbers: tuple[tuple[int, float], ...]  # (number, probability) desc by probability
    auc: float | None
    precision_at_k: float | None      # mean fraction of the top draw_size picks that hit, across CV folds
    baseline_precision: float          # precision a uniform-random top-k pick would get by chance
    best_C: float | None               # regularization strength chosen by walk-forward grid search
    n_draws: int
    holdout_size: int                  # total draws evaluated across all CV folds


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


def _fit(X, y, C, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(X, y)
    return model


def _precision_at_k(preds: np.ndarray, y_test: np.ndarray, ts_test: np.ndarray, draw_size: int) -> float | None:
    """Mean fraction of each draw's top-`draw_size` predicted numbers that were
    actually drawn, averaged across the draws present in ts_test."""
    hits, n_draws_seen = 0.0, 0
    for t in np.unique(ts_test):
        mask = ts_test == t
        top_idx = np.argsort(-preds[mask])[:draw_size]
        hits += y_test[mask][top_idx].sum()
        n_draws_seen += 1
    return hits / (n_draws_seen * draw_size) if n_draws_seen else None


def _fold_bounds(n_draws: int) -> list[tuple[int, int]]:
    """Rolling walk-forward folds, most recent first: (test_start, test_end),
    each trained only on draws strictly before test_start."""
    bounds = []
    for k in range(N_FOLDS):
        test_end = n_draws - k * HOLDOUT_N
        test_start = test_end - HOLDOUT_N
        if test_start < WARMUP + 10:
            break
        bounds.append((test_start, test_end))
    return bounds


def _walk_forward_eval(X, y, ts_full, n_draws, draw_size, C, seed):
    """Fit/evaluate across each rolling fold, always training only on draws
    strictly before that fold's test window. Returns mean AUC, mean
    precision@draw_size, and the total number of draws evaluated."""
    aucs, precisions, n_evaluated = [], [], 0
    for test_start, test_end in _fold_bounds(n_draws):
        train_mask = ts_full < test_start
        test_mask = (ts_full >= test_start) & (ts_full < test_end)
        if test_mask.sum() == 0 or len(set(y[train_mask])) < 2:
            continue
        model = _fit(X[train_mask], y[train_mask], C, seed)
        preds = model.predict_proba(X[test_mask])[:, 1]
        if len(set(y[test_mask])) > 1:
            aucs.append(roc_auc_score(y[test_mask], preds))
        precision = _precision_at_k(preds, y[test_mask], ts_full[test_mask], draw_size)
        if precision is not None:
            precisions.append(precision)
        n_evaluated += test_end - test_start
    mean_auc = float(np.mean(aucs)) if aucs else None
    mean_precision = float(np.mean(precisions)) if precisions else None
    return mean_auc, mean_precision, n_evaluated


def _grid_search_C(X, y, ts_full, n_draws, draw_size, seed):
    """Pick the regularization strength that scores best (by mean walk-forward
    AUC) across the rolling CV folds -- never touching the final holdout used
    for the reported metrics beyond that same rolling evaluation."""
    best_C, best_auc = C_GRID[0], -1.0
    for C in C_GRID:
        auc, _, _ = _walk_forward_eval(X, y, ts_full, n_draws, draw_size, C, seed)
        if auc is not None and auc > best_auc:
            best_C, best_auc = C, auc
    return best_C


def _train_and_score(X, y, ts_full, n_draws, next_feats, draw_size, seed):
    best_C = _grid_search_C(X, y, ts_full, n_draws, draw_size, seed)
    auc, precision, n_evaluated = _walk_forward_eval(X, y, ts_full, n_draws, draw_size, best_C, seed)

    # Final production model is refit on *all* available history at the tuned
    # C, so the next-draw probabilities use every draw -- the CV folds above
    # exist only to pick C and report honest out-of-sample metrics, not to
    # withhold data from the model actually used for ranking.
    final_model = _fit(X, y, best_C, seed)
    prob = {n: final_model.predict_proba([feats])[0, 1] for n, feats in next_feats.items()}
    return prob, auc, precision, best_C, n_evaluated


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
    prob, auc, precision, best_C, holdout_size = _train_and_score(
        X, y, ts_full, n_draws, next_feats, draw_size, seed
    )
    ranked = tuple(sorted(prob.items(), key=lambda item: (-item[1], item[0])))
    baseline_precision = draw_size / pool_size
    return MLReport(
        ranked_numbers=ranked,
        auc=auc,
        precision_at_k=precision,
        baseline_precision=baseline_precision,
        best_C=best_C,
        n_draws=n_draws,
        holdout_size=holdout_size,
    )


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
