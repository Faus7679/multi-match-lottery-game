from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from random import Random
from typing import Iterable

NUMBER_RANGE = range(1, 44)
DRAW_SIZE = 6
LINES_PER_TICKET = 3
SUPPORTED_DRAW_DAYS = ("Monday", "Thursday")


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
    )
    return tuple(
        DrawRecord(datetime.strptime(draw_date, "%Y-%m-%d").date(), numbers)
        for draw_date, numbers in sample_rows
    )


def dynamic_scores(history: Iterable[DrawRecord], draw_day: str) -> dict[int, float]:
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
        scores[number] = (day_hits * 4.0) + (recent_hits * 2.5) + float(total_gap) + exploration_bonus
    return scores


def predict_winning_line(history: Iterable[DrawRecord], draw_day: str) -> tuple[int, ...]:
    scores = dynamic_scores(history, draw_day)
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


def build_ticket(history: Iterable[DrawRecord], draw_day: str, seed: int = 7) -> tuple[tuple[int, ...], ...]:
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


def main() -> None:
    history = sample_maryland_history()
    monday_analysis = analyze_draw_day(history, "Monday")
    thursday_analysis = analyze_draw_day(history, "Thursday")

    print("Maryland Multi-Match dynamic strategy demo")
    print("=" * 40)
    print("Rules: choose 6 numbers from 1-43. Drawings happen on Monday and Thursday.")
    print()
    print(format_analysis(monday_analysis))
    print()
    print(format_analysis(thursday_analysis))
    print()
    thursday_ticket = build_ticket(history, "Thursday")
    print("Suggested Thursday ticket:")
    for index, line in enumerate(thursday_ticket, start=1):
        print(f"  Line {index}: {', '.join(str(number) for number in line)}")


if __name__ == "__main__":
    main()
