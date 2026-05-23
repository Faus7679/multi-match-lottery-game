import unittest

from lottery_game import (
    DrawRecord,
    analyze_draw_day,
    build_ticket,
    evaluate_ticket,
    predict_winning_line,
    sample_maryland_history,
)


class LotteryGameTests(unittest.TestCase):
    def test_draw_record_rejects_invalid_draw_day(self) -> None:
        with self.assertRaisesRegex(ValueError, "Monday and Thursday"):
            DrawRecord.fromisoformat("2026-05-19", (1, 2, 3, 4, 5, 6))

    def test_monday_prediction_favors_repeat_monday_numbers(self) -> None:
        history = sample_maryland_history()

        self.assertEqual(predict_winning_line(history, "Monday"), (6, 15, 17, 22, 31, 32))

    def test_thursday_analysis_returns_ranked_scores(self) -> None:
        history = sample_maryland_history()

        analysis = analyze_draw_day(history, "Thursday")

        self.assertEqual(analysis.recommended_line, (9, 11, 23, 28, 34, 38))
        self.assertEqual(analysis.hottest_numbers, (9, 11, 23, 28, 34, 38))
        self.assertEqual(len(analysis.scorecard), 10)

    def test_build_ticket_starts_with_prediction(self) -> None:
        history = sample_maryland_history()

        ticket = build_ticket(history, "Thursday", seed=1)

        self.assertEqual(ticket[0], (9, 11, 23, 28, 34, 38))
        self.assertEqual(len(ticket), 3)
        for line in ticket:
            self.assertEqual(len(line), 6)
            self.assertEqual(line, tuple(sorted(line)))

    def test_evaluate_ticket_counts_matches_per_line(self) -> None:
        ticket = (
            (1, 2, 3, 4, 5, 6),
            (7, 8, 9, 10, 11, 12),
            (13, 14, 15, 16, 17, 18),
        )

        result = evaluate_ticket(ticket, (1, 3, 5, 7, 9, 11))

        self.assertEqual(result["per_line_matches"], (3, 3, 0))
        self.assertEqual(result["best_line_match_count"], 3)
        self.assertEqual(result["total_matched_numbers"], 6)


if __name__ == "__main__":
    unittest.main()
