from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from random import Random
from typing import Iterable

NUMBER_RANGE = range(1, 44)
DRAW_SIZE = 6
LINES_PER_TICKET = 3
SUPPORTED_DRAW_DAYS = ("Monday", "Thursday")


class Style:
    """ANSI styling for the winning numbers. Falls back to plain text when the
    terminal doesn't support color (piped output, NO_COLOR, unsupported Windows console)."""

    BOLD = ""
    YELLOW = ""
    GREEN = ""
    RESET = ""

    def __init__(self, enabled: bool) -> None:
        if enabled:
            self.BOLD = "\033[1m"
            self.YELLOW = "\033[93m"
            self.GREEN = "\033[92m"
            self.RESET = "\033[0m"

    @classmethod
    def detect(cls) -> "Style":
        return cls(_supports_color())


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        return _enable_windows_ansi()
    return True


def _enable_windows_ansi() -> bool:
    """Turn on ANSI escape processing for the current Windows console, if possible."""
    try:
        import ctypes

        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if not kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING):
            return False
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class DrawRecord:
    draw_date: date
    numbers: tuple[int, ...]

    @classmethod
    def fromisoformat(cls, draw_date: str, numbers: Iterable[int]) -> "DrawRecord":
        return cls(datetime.strptime(draw_date, "%Y-%m-%d").date(), tuple(numbers))

    def __post_init__(self) -> None:
        normalized_numbers = normalize_line(self.numbers)
        if self.weekday not in SUPPORTED_DRAW_DAYS:
            raise ValueError("Maryland Multi-Match draws happen on Monday and Thursday.")
        object.__setattr__(self, "numbers", normalized_numbers)

    @property
    def weekday(self) -> str:
        return self.draw_date.strftime("%A")


@dataclass(frozen=True)
class DayAnalysis:
    draw_day: str
    recommended_line: tuple[int, ...]
    hottest_numbers: tuple[int, ...]
    overdue_numbers: tuple[int, ...]
    scorecard: tuple[tuple[int, float], ...]


def normalize_line(numbers: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(numbers)))
    if len(normalized) != DRAW_SIZE:
        raise ValueError(f"Each line must contain {DRAW_SIZE} unique numbers.")
    invalid_numbers = [number for number in normalized if number not in NUMBER_RANGE]
    if invalid_numbers:
        raise ValueError("Numbers must be between 1 and 43.")
    return normalized


def sample_maryland_history() -> tuple[DrawRecord, ...]:
    sample_rows = (
        # 2025 — full year (104 draws)
        ("2025-01-02", (6, 18, 22, 25, 36, 39)),
        ("2025-01-06", (2, 12, 18, 19, 26, 39)),
        ("2025-01-09", (3, 15, 28, 32, 38, 41)),
        ("2025-01-13", (9, 16, 20, 35, 37, 43)),
        ("2025-01-16", (9, 13, 27, 32, 40, 43)),
        ("2025-01-20", (5, 9, 14, 17, 34, 42)),
        ("2025-01-23", (20, 21, 23, 28, 35, 41)),
        ("2025-01-27", (7, 12, 21, 25, 32, 35)),
        ("2025-01-30", (3, 12, 15, 17, 26, 40)),
        ("2025-02-03", (7, 11, 21, 23, 39, 40)),
        ("2025-02-06", (1, 7, 16, 28, 33, 42)),
        ("2025-02-10", (4, 10, 13, 23, 27, 29)),
        ("2025-02-13", (4, 11, 16, 29, 33, 40)),
        ("2025-02-17", (11, 19, 28, 29, 33, 37)),
        ("2025-02-20", (10, 12, 30, 34, 35, 37)),
        ("2025-02-24", (3, 6, 16, 22, 26, 33)),
        ("2025-02-27", (9, 12, 21, 22, 27, 41)),
        ("2025-03-03", (1, 2, 6, 10, 12, 15)),
        ("2025-03-06", (6, 11, 13, 21, 23, 24)),
        ("2025-03-10", (14, 20, 33, 35, 36, 43)),
        ("2025-03-13", (15, 17, 22, 25, 34, 40)),
        ("2025-03-17", (8, 9, 11, 12, 23, 25)),
        ("2025-03-20", (3, 7, 28, 30, 32, 41)),
        ("2025-03-24", (11, 21, 24, 27, 28, 30)),
        ("2025-03-27", (8, 16, 24, 29, 30, 36)),
        ("2025-03-31", (2, 7, 9, 21, 30, 34)),
        ("2025-04-03", (10, 11, 12, 15, 16, 21)),
        ("2025-04-07", (3, 9, 18, 24, 26, 33)),
        ("2025-04-10", (5, 27, 29, 35, 39, 42)),
        ("2025-04-14", (5, 10, 18, 23, 26, 40)),
        ("2025-04-17", (1, 17, 23, 36, 37, 41)),
        ("2025-04-21", (4, 30, 34, 36, 40, 41)),
        ("2025-04-24", (5, 22, 31, 32, 36, 42)),
        ("2025-04-28", (6, 10, 15, 17, 20, 43)),
        ("2025-05-01", (4, 7, 19, 21, 22, 36)),
        ("2025-05-05", (13, 30, 36, 38, 41, 42)),
        ("2025-05-08", (4, 6, 9, 10, 21, 40)),
        ("2025-05-12", (6, 7, 11, 14, 35, 37)),
        ("2025-05-15", (4, 9, 17, 25, 30, 37)),
        ("2025-05-19", (13, 16, 23, 32, 37, 43)),
        ("2025-05-22", (15, 18, 24, 32, 35, 42)),
        ("2025-05-26", (1, 6, 13, 20, 22, 33)),
        ("2025-05-29", (11, 12, 13, 20, 22, 43)),
        ("2025-06-02", (5, 7, 9, 17, 34, 43)),
        ("2025-06-05", (4, 28, 30, 32, 33, 37)),
        ("2025-06-09", (20, 22, 27, 31, 32, 41)),
        ("2025-06-12", (7, 18, 19, 28, 29, 30)),
        ("2025-06-16", (2, 3, 4, 17, 32, 42)),
        ("2025-06-19", (11, 15, 17, 23, 26, 35)),
        ("2025-06-23", (1, 2, 6, 19, 27, 42)),
        ("2025-06-26", (1, 5, 12, 19, 25, 36)),
        ("2025-06-30", (4, 13, 14, 20, 25, 31)),
        ("2025-07-03", (13, 22, 24, 28, 34, 42)),
        ("2025-07-07", (11, 17, 19, 26, 30, 35)),
        ("2025-07-10", (1, 6, 13, 16, 24, 39)),
        ("2025-07-14", (9, 14, 18, 24, 36, 40)),
        ("2025-07-17", (5, 6, 9, 11, 20, 41)),
        ("2025-07-21", (8, 9, 13, 15, 22, 36)),
        ("2025-07-24", (4, 23, 24, 25, 40, 43)),
        ("2025-07-28", (10, 21, 31, 38, 41, 42)),
        ("2025-07-31", (3, 7, 15, 22, 24, 32)),
        ("2025-08-04", (12, 18, 20, 32, 34, 39)),
        ("2025-08-07", (6, 12, 21, 22, 24, 28)),
        ("2025-08-11", (10, 15, 21, 27, 29, 42)),
        ("2025-08-14", (3, 9, 11, 15, 17, 39)),
        ("2025-08-18", (1, 14, 20, 25, 30, 36)),
        ("2025-08-21", (1, 2, 9, 20, 25, 34)),
        ("2025-08-25", (2, 3, 6, 24, 38, 39)),
        ("2025-08-28", (13, 14, 16, 27, 31, 33)),
        ("2025-09-01", (3, 4, 5, 10, 24, 25)),
        ("2025-09-04", (3, 22, 23, 29, 36, 39)),
        ("2025-09-08", (2, 19, 25, 28, 29, 33)),
        ("2025-09-11", (5, 26, 29, 30, 32, 40)),
        ("2025-09-15", (5, 11, 23, 27, 28, 33)),
        ("2025-09-18", (6, 14, 16, 24, 38, 40)),
        ("2025-09-22", (2, 24, 25, 34, 39, 42)),
        ("2025-09-25", (9, 10, 21, 23, 26, 30)),
        ("2025-09-29", (2, 8, 24, 30, 31, 33)),
        ("2025-10-02", (14, 17, 20, 21, 28, 36)),
        ("2025-10-06", (1, 3, 4, 5, 10, 19)),
        ("2025-10-09", (2, 8, 39, 40, 41, 43)),
        ("2025-10-13", (7, 8, 18, 20, 24, 41)),
        ("2025-10-16", (24, 31, 34, 38, 39, 40)),
        ("2025-10-20", (9, 24, 29, 31, 36, 40)),
        ("2025-10-23", (4, 12, 13, 18, 38, 41)),
        ("2025-10-27", (8, 18, 26, 29, 30, 41)),
        ("2025-10-30", (18, 19, 23, 26, 33, 40)),
        ("2025-11-03", (12, 13, 20, 24, 25, 28)),
        ("2025-11-06", (16, 18, 20, 24, 27, 38)),
        ("2025-11-10", (2, 4, 10, 16, 31, 32)),
        ("2025-11-13", (8, 11, 12, 16, 17, 43)),
        ("2025-11-17", (2, 4, 13, 23, 30, 42)),
        ("2025-11-20", (3, 8, 20, 23, 38, 41)),
        ("2025-11-24", (11, 14, 34, 35, 41, 43)),
        ("2025-11-27", (5, 13, 16, 28, 29, 35)),
        ("2025-12-01", (1, 3, 10, 13, 38, 40)),
        ("2025-12-04", (11, 19, 21, 30, 32, 34)),
        ("2025-12-08", (2, 15, 18, 32, 39, 43)),
        ("2025-12-11", (8, 12, 16, 31, 41, 42)),
        ("2025-12-15", (7, 13, 14, 17, 29, 38)),
        ("2025-12-18", (10, 23, 25, 29, 39, 41)),
        ("2025-12-22", (6, 8, 17, 29, 34, 43)),
        ("2025-12-25", (1, 5, 8, 18, 25, 41)),
        ("2025-12-29", (10, 22, 25, 29, 38, 40)),
        # 2026 — Jan 1 through May 28 (43 draws)
        ("2026-01-01", (14, 19, 20, 22, 27, 30)),
        ("2026-01-05", (2, 3, 8, 10, 25, 42)),
        ("2026-01-08", (31, 34, 35, 37, 40, 41)),
        ("2026-01-12", (6, 26, 35, 37, 38, 43)),
        ("2026-01-15", (2, 5, 11, 12, 14, 34)),
        ("2026-01-19", (1, 3, 8, 11, 22, 26)),
        ("2026-01-22", (4, 6, 10, 27, 39, 41)),
        ("2026-01-26", (2, 11, 22, 28, 29, 30)),
        ("2026-01-29", (7, 8, 10, 24, 32, 36)),
        ("2026-02-02", (9, 15, 22, 27, 39, 40)),
        ("2026-02-05", (2, 11, 13, 27, 34, 37)),
        ("2026-02-09", (1, 8, 27, 29, 30, 40)),
        ("2026-02-12", (12, 18, 30, 31, 36, 42)),
        ("2026-02-16", (11, 13, 16, 19, 24, 29)),
        ("2026-02-19", (4, 10, 12, 14, 19, 32)),
        ("2026-02-23", (8, 13, 18, 20, 23, 26)),
        ("2026-02-26", (1, 23, 25, 32, 35, 42)),
        ("2026-03-02", (1, 16, 25, 27, 28, 34)),
        ("2026-03-05", (4, 6, 25, 32, 36, 42)),
        ("2026-03-09", (3, 6, 8, 16, 23, 32)),
        ("2026-03-12", (9, 15, 24, 26, 27, 35)),
        ("2026-03-16", (6, 11, 19, 32, 34, 39)),
        ("2026-03-19", (3, 14, 15, 19, 27, 38)),
        ("2026-03-23", (7, 10, 14, 18, 25, 31)),
        ("2026-03-26", (1, 16, 23, 24, 26, 29)),
        ("2026-03-30", (1, 3, 5, 16, 20, 28)),
        ("2026-04-02", (7, 16, 22, 24, 40, 43)),
        ("2026-04-06", (7, 8, 24, 27, 33, 42)),
        ("2026-04-09", (5, 14, 19, 39, 41, 43)),
        ("2026-04-13", (20, 28, 31, 35, 38, 42)),
        ("2026-04-16", (2, 6, 8, 12, 29, 38)),
        ("2026-04-20", (6, 11, 19, 31, 32, 38)),
        ("2026-04-23", (11, 23, 28, 33, 34, 35)),
        ("2026-04-27", (1, 15, 17, 41, 42, 43)),
        ("2026-04-30", (4, 6, 16, 18, 19, 34)),
        ("2026-05-04", (17, 24, 26, 27, 29, 32)),
        ("2026-05-07", (11, 17, 23, 29, 38, 42)),
        ("2026-05-11", (9, 12, 15, 20, 22, 31)),
        ("2026-05-14", (3, 9, 23, 30, 34, 38)),
        ("2026-05-18", (4, 6, 13, 16, 22, 38)),
        ("2026-05-21", (7, 9, 14, 28, 34, 41)),
        ("2026-05-25", (3, 7, 17, 27, 39, 41)),
        ("2026-05-28", (2, 9, 11, 25, 29, 37)),
        ("2026-06-01", (1, 4, 12, 16, 18, 21)),
        ("2026-06-04", (3, 6, 19, 21, 32, 33)),
        ("2026-06-08", (10, 21, 22, 25, 26, 32)),
        ("2026-06-11", (4, 6, 11, 28, 29, 34)),
        ("2026-06-15", (8, 17, 18, 19, 20, 24)),
        ("2026-06-18", (8, 11, 20, 26, 30, 43)),
        ("2026-06-22", (1, 3, 9, 21, 22, 33)),
        ("2026-06-25", (10, 15, 16, 21, 36, 40)),
    )
    return tuple(
        DrawRecord(datetime.strptime(draw_date, "%Y-%m-%d").date(), numbers)
        for draw_date, numbers in sample_rows
    )


def dynamic_scores(
    history: Iterable[DrawRecord],
    draw_day: str,
    freq_weight: float = 2.0,
    recent_weight: float = 1.0,
    gap_weight: float = 1.0,
) -> dict[int, float]:
    validate_draw_day(draw_day)
    history_by_day = tuple(record for record in history if record.weekday == draw_day)
    if not history_by_day:
        raise ValueError(f"No history is available for {draw_day}.")

    recent_window = history_by_day[-3:]
    scores: dict[int, float] = {}
    history_size = len(history_by_day)
    for number in NUMBER_RANGE:
        day_hits = sum(number in record.numbers for record in history_by_day)
        recent_hits = sum(number in record.numbers for record in recent_window)
        total_gap = next(
            (offset for offset, record in enumerate(reversed(history_by_day), start=1) if number in record.numbers),
            history_size + 1,
        )
        exploration_bonus = 0.25 if day_hits == 0 else 0.0
        scores[number] = (
            day_hits * freq_weight
            + recent_hits * recent_weight
            + total_gap * gap_weight
            + exploration_bonus
        )
    return scores


def predict_winning_line(
    history: Iterable[DrawRecord],
    draw_day: str,
    freq_weight: float = 2.0,
    recent_weight: float = 1.0,
    gap_weight: float = 1.0,
) -> tuple[int, ...]:
    scores = dynamic_scores(history, draw_day, freq_weight, recent_weight, gap_weight)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(sorted(number for number, _ in ranked[:DRAW_SIZE]))


def analyze_draw_day(history: Iterable[DrawRecord], draw_day: str) -> DayAnalysis:
    scores = dynamic_scores(history, draw_day)
    day_history = tuple(record for record in history if record.weekday == draw_day)
    frequency_rank = sorted(
        NUMBER_RANGE,
        key=lambda number: (-sum(number in record.numbers for record in day_history), number),
    )
    overdue_rank = sorted(
        NUMBER_RANGE,
        key=lambda number: (
            -next(
                (offset for offset, record in enumerate(reversed(day_history), start=1) if number in record.numbers),
                len(day_history) + 1,
            ),
            number,
        ),
    )
    ranked_scores = tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:10])
    return DayAnalysis(
        draw_day=draw_day,
        recommended_line=predict_winning_line(day_history, draw_day),
        hottest_numbers=tuple(sorted(frequency_rank[:DRAW_SIZE])),
        overdue_numbers=tuple(sorted(overdue_rank[:DRAW_SIZE])),
        scorecard=ranked_scores,
    )


def next_draw_day() -> tuple[str, date]:
    """Return the next Multi-Match draw day, treating today as "next" if today is a draw day."""
    today = date.today()
    draw_weekdays = {"Monday": 0, "Thursday": 3}
    days_ahead = {
        name: (target - today.weekday()) % 7
        for name, target in draw_weekdays.items()
    }
    name = min(days_ahead, key=days_ahead.__getitem__)
    return name, today + timedelta(days=days_ahead[name])


def build_ticket(history: Iterable[DrawRecord], draw_day: str, seed: int | None = None) -> tuple[tuple[int, ...], ...]:
    prediction = predict_winning_line(history, draw_day)
    rng = Random(seed)
    ticket = [prediction]
    while len(ticket) < LINES_PER_TICKET:
        ticket.append(generate_quick_pick(rng))
    return tuple(ticket)


def generate_quick_pick(rng: Random) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(tuple(NUMBER_RANGE), DRAW_SIZE)))


def evaluate_ticket(
    ticket: Iterable[Iterable[int]],
    winning_numbers: Iterable[int],
) -> dict[str, object]:
    normalized_winning_numbers = normalize_line(winning_numbers)
    normalized_ticket = tuple(normalize_line(line) for line in ticket)
    per_line_matches = tuple(
        len(set(line).intersection(normalized_winning_numbers))
        for line in normalized_ticket
    )
    return {
        "winning_numbers": normalized_winning_numbers,
        "per_line_matches": per_line_matches,
        "best_line_match_count": max(per_line_matches, default=0),
        "total_matched_numbers": sum(per_line_matches),
    }


def validate_draw_day(draw_day: str) -> None:
    if draw_day not in SUPPORTED_DRAW_DAYS:
        raise ValueError("Draw day must be Monday or Thursday.")


def backtest(
    history: tuple[DrawRecord, ...],
    min_day_draws: int = 3,
    freq_weight: float = 2.0,
    recent_weight: float = 1.0,
    gap_weight: float = 1.0,
) -> dict[str, dict]:
    results: dict[str, list[int]] = {day: [] for day in SUPPORTED_DRAW_DAYS}
    for i, record in enumerate(history):
        day_prior = tuple(r for r in history[:i] if r.weekday == record.weekday)
        if len(day_prior) < min_day_draws:
            continue
        predicted = predict_winning_line(day_prior, record.weekday, freq_weight, recent_weight, gap_weight)
        results[record.weekday].append(len(set(predicted) & set(record.numbers)))

    random_baseline = DRAW_SIZE * DRAW_SIZE / len(NUMBER_RANGE)
    return {
        day: {
            "draws_tested": len(hits),
            "avg_hits": sum(hits) / len(hits),
            "random_baseline": random_baseline,
            "lift": sum(hits) / len(hits) - random_baseline,
            "hit_distribution": {k: hits.count(k) for k in range(DRAW_SIZE + 1)},
        }
        for day, hits in results.items()
        if hits
    }


def format_backtest(results: dict[str, dict]) -> str:
    lines = ["Backtest  (walk-forward: predict then verify)"]
    for day, stats in results.items():
        dist = stats["hit_distribution"]
        dist_str = "  ".join(f"{k}:{dist.get(k, 0)}" for k in range(DRAW_SIZE + 1))
        lines += [
            f"\n  {day} - {stats['draws_tested']} draws tested",
            f"    Avg hits:        {stats['avg_hits']:.3f}",
            f"    Random baseline: {stats['random_baseline']:.3f}",
            f"    Lift:            {stats['lift']:+.3f}",
            f"    Distribution:    {dist_str}",
        ]
    return "\n".join(lines)


def grid_search_weights(history: tuple[DrawRecord, ...]) -> list[dict]:
    freq_values = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)
    recent_values = (0.0, 1.0, 2.0, 3.0, 5.0)
    gap_values = (0.0, 0.25, 0.5, 1.0, 2.0)
    results = []
    for fw in freq_values:
        for rw in recent_values:
            for gw in gap_values:
                bt = backtest(history, freq_weight=fw, recent_weight=rw, gap_weight=gw)
                combined_lift = sum(stats["lift"] for stats in bt.values())
                results.append({
                    "freq_weight": fw,
                    "recent_weight": rw,
                    "gap_weight": gw,
                    "combined_lift": combined_lift,
                    "by_day": bt,
                })
    return sorted(results, key=lambda r: -r["combined_lift"])


def format_grid_search(results: list[dict], top_n: int = 10) -> str:
    header = f"{'freq':>5}  {'recent':>6}  {'gap':>5}  {'Mon lift':>9}  {'Thu lift':>9}  {'combined':>9}"
    lines = [f"Grid search - top {top_n} weight combinations (150 total)", f"  {header}"]
    for r in results[:top_n]:
        by_day = r["by_day"]
        mon_lift = by_day.get("Monday", {}).get("lift", float("nan"))
        thu_lift = by_day.get("Thursday", {}).get("lift", float("nan"))
        lines.append(
            f"  {r['freq_weight']:>5.1f}  {r['recent_weight']:>6.1f}  {r['gap_weight']:>5.2f}"
            f"  {mon_lift:>+9.3f}  {thu_lift:>+9.3f}  {r['combined_lift']:>+9.3f}"
        )
    return "\n".join(lines)


def format_analysis(analysis: DayAnalysis) -> str:
    recommended = ", ".join(str(number) for number in analysis.recommended_line)
    hottest = ", ".join(str(number) for number in analysis.hottest_numbers)
    overdue = ", ".join(str(number) for number in analysis.overdue_numbers)
    score_lines = "\n".join(
        f"    {number:>2}: {score:.2f}"
        for number, score in analysis.scorecard
    )
    return (
        f"{analysis.draw_day} analysis\n"
        f"  Recommended line: {recommended}\n"
        f"  Hottest numbers: {hottest}\n"
        f"  Overdue numbers: {overdue}\n"
        f"  Top dynamic scores:\n{score_lines}"
    )


def _try_live_history(static: tuple) -> tuple[tuple, str]:
    """Attempt to fetch live data and merge with static history. Always returns a valid history."""
    try:
        from live_data import fetch_live_results, merge_history
        live, err, source = fetch_live_results()
        if live:
            merged = merge_history(static=static, live=live)
            label = f"{source.upper()}  mdlottery.com  ({merged[-1].draw_date})"
            return merged, label
        return static, f"static fallback  (live: {err})"
    except ImportError:
        return static, "static  (live_data.py not found)"
    except Exception as exc:
        return static, f"static fallback  (error: {exc})"


def main() -> None:
    style = Style.detect()
    static = sample_maryland_history()
    history, data_label = _try_live_history(static)

    draw_day, draw_date = next_draw_day()
    today = date.today()
    is_draw_today = draw_date == today

    ticket = build_ticket(history, draw_day)
    smart_line = ticket[0]
    analysis = analyze_draw_day(history, draw_day)
    day_history = tuple(r for r in history if r.weekday == draw_day)

    W = 62
    print("=" * W)
    print("  Maryland Multi-Match - LIVE SMART PICK")
    print("=" * W)
    print(f"  Data   : {data_label}")
    print(f"  History: {len(history)} draws  ({history[0].draw_date} to {history[-1].draw_date})")
    print()

    draw_tag = f"  {style.BOLD}{style.YELLOW}[DRAW IS TODAY - buy before cutoff!]{style.RESET}" if is_draw_today else ""
    print(f"  Next draw: {draw_day}, {draw_date.strftime('%B %d, %Y')}{draw_tag}")
    print()

    smart_pick_str = " - ".join(f"{n:02d}" for n in smart_line)
    box_inner = f"   SMART PICK  >>  {smart_pick_str}   "
    border = "-" * len(box_inner)
    print(f"  {style.BOLD}{style.YELLOW}+{border}+{style.RESET}")
    print(f"  {style.BOLD}{style.YELLOW}|{box_inner}|{style.RESET}")
    print(f"  {style.BOLD}{style.YELLOW}+{border}+{style.RESET}")
    print()
    print(f"  Full ticket ({len(ticket)} lines):")
    for idx, line in enumerate(ticket, start=1):
        line_str = ", ".join(f"{n:02d}" for n in line)
        if idx == 1:
            print(f"    Line {idx}: {style.BOLD}{style.YELLOW}{line_str}{style.RESET}  <- smart pick")
        else:
            print(f"    Line {idx}: {line_str}  (quick pick)")

    print()
    print("-" * W)
    print(f"  {draw_day} analysis")
    print("-" * W)
    recommended = ", ".join(f"{n:02d}" for n in analysis.recommended_line)
    hottest = ", ".join(f"{n:02d}" for n in analysis.hottest_numbers)
    overdue = ", ".join(f"{n:02d}" for n in analysis.overdue_numbers)
    print(f"  Recommended : {style.BOLD}{style.YELLOW}{recommended}{style.RESET}")
    print(f"  Hottest     : {hottest}")
    print(f"  Overdue     : {overdue}")
    print()
    print(f"  Top dynamic scores:")
    for number, score in analysis.scorecard:
        bar = "#" * int(score // 1)
        print(f"    {number:>2}: {score:>6.2f}  {bar}")

    print()
    print("-" * W)
    print(f"  Recent {draw_day} draws")
    print("-" * W)
    for r in day_history[-5:]:
        is_latest = r == day_history[-1]
        numbers_str = " - ".join(f"{n:02d}" for n in r.numbers)
        if is_latest:
            print(f"    {r.draw_date}  {style.BOLD}{style.GREEN}{numbers_str}{style.RESET}  <- most recent (actual winning numbers)")
        else:
            print(f"    {r.draw_date}  {numbers_str}")

    print()
    print("=" * W)
    print()

    monday_analysis = analyze_draw_day(history, "Monday")
    thursday_analysis = analyze_draw_day(history, "Thursday")
    print("Full analysis (both draw days)")
    print("=" * 40)
    print(format_analysis(monday_analysis))
    print()
    print(format_analysis(thursday_analysis))
    print()
    print(format_backtest(backtest(history)))
    print()
    print(format_grid_search(grid_search_weights(history)))


if __name__ == "__main__":
    main()
