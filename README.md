# multi-match-lottery-game

This repository contains a small Python implementation of a Maryland-style Multi-Match lottery helper.
It also reports historical Monday/Thursday draw statistics (hottest numbers, overdue numbers) for interest.

**Reality check:** real Multi-Match drawings are independent random events. No model — machine
learning included — can predict them. When enough history is available, ticket lines are generated
by an ML model that *weights* numbers by historical pattern (see "Machine learning" below), but the
model's own reported held-out accuracy sits at chance level, and every ticket line is still, in the
end, a weighted random draw. Nothing in this repo should be read as a claim of "accurate" prediction.

## Maryland Multi-Match assumptions used here

- Choose **6 unique numbers**
- Numbers come from **1 to 43**
- Drawings are modeled for **Monday** and **Thursday**
- A ticket contains **3 lines**, with no duplicate lines within a ticket; lines are either sampled
  from the ML model's weights (when available) or plain random

## Files

- `lottery_game.py`
  - main game logic
  - random line/ticket generation
  - Monday/Thursday historical statistics (hottest/overdue numbers)
  - prints the random pick and the most recent actual winning numbers in bold/color when the terminal supports it
- `live_data.py`
  - fetches recent draw results from mdlottery.com, with local caching and a static-data fallback
- `analysis.py`
  - deep statistical report: frequency, pairs, sums, gaps, momentum, and ticket recommendations
- `ml_model.py`
  - optional ML ticket weighting: a logistic regression per number, trained on causal historical
    features, with honest held-out ROC-AUC reporting (see "Machine learning" below)
- `test_lottery_game.py`
  - focused unit tests

## How it works

### Step 1: Load draw history

`sample_maryland_history()` includes recent sample Monday and Thursday Maryland Multi-Match results.

### Step 2: Separate draws by day

The historical statistics are computed independently for:

- Monday
- Thursday

This keeps the stats day-specific instead of mixing both draw schedules together.

### Step 3: Pick a random line

`predict_winning_line()` draws 6 unique numbers from 1 to 43 at random (optionally seeded for reproducible output). It does not use draw history — a real drawing is independent of past results.

### Step 4: Build a Multi-Match ticket

`build_ticket()` creates a 3-line ticket of unique, randomly drawn lines.

`generate_smart_tickets()` calls this pattern multiple times to produce a plain-random ticket set.
`generate_ml_smart_tickets()` is what `main()` actually calls: it tries the ML-weighted pick
described below, and falls back to `generate_smart_tickets()`'s plain random lines if scikit-learn/
pandas aren't installed or there isn't enough history to train on.

### Step 5: Check tickets against the actual draw once it's published

`find_actual_result()` looks in the (live-merged) history for an official record matching the
upcoming draw day and date. If mdlottery.com has already published that draw's numbers — e.g. you
run the script after the draw has happened — `main()` shows the real winning numbers instead of a
random guess, and scores every generated ticket against them with `evaluate_ticket()` (per-line
matches, best line, total matched numbers). If the draw hasn't happened yet, the tickets are shown
as plain random picks with no accuracy claim, since a real Multi-Match drawing can't be predicted.

### Step 6: Review the Monday and Thursday history

`analyze_draw_day()` returns:

- a random recommended line
- the hottest numbers (historically most frequent)
- the most overdue numbers (longest since last appearing)

This gives a simple historical-stats view for Maryland Monday and Thursday draw behavior while treating the actual next draw as uncertain and random.

## Machine learning

`ml_model.py` trains a **logistic regression per number** (pool 1–43) on features built *causally*
from draw history — every feature for a training row at draw *t* uses only draws strictly before
*t*, so the model never sees the future it's scored against:

- all-time frequency
- rolling frequency over two recent windows (10 and 20 draws)
- overdue gap since the number last appeared
- pairing momentum with the numbers drawn immediately before it

`train_number_model()` reports **held-out ROC-AUC** on the most recent 20 draws so the model's real
skill is visible rather than implied. Because Multi-Match draws are independent random events, that
AUC should sit near 0.5 (a coin flip) — and it does in practice. `generate_ml_smart_tickets()` uses
the model's predicted probabilities only as *sampling weights* (every number keeps a nonzero chance),
then falls back to a plain random pick if:

- `pandas`/`scikit-learn` aren't installed, or
- there's fewer than 70 draws of history to train on reliably

`main()` prints which path ran, the held-out AUC when the ML path trains successfully, and a
disclaimer repeating that near-chance AUC means no real predictive edge exists.

## Run the demo

```bash
cd multi-match-lottery-game
pip install -r requirements.txt   # optional: enables the ML-weighted pick
python lottery_game.py
```

Example output includes:

- Monday analysis
- Thursday analysis
- a suggested Thursday ticket
- the ML model's held-out AUC and top-ranked numbers, when the ML path is available

## Run the tests

```bash
cd multi-match-lottery-game
python -m unittest discover
```

## Notes

- This project is an educational analysis tool, not a guarantee of lottery results.
- The built-in history can be replaced with newer Maryland draw data if you want to update the hottest/overdue statistics.
