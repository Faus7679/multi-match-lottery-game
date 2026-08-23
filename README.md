# multi-match-lottery-game

This repository contains a small Python implementation of a Maryland-style Multi-Match lottery helper.
Every line it hands you — the "recommended" line and every Smart/Random Ticket line — is a random pick,
because real Multi-Match drawings are independent random events that historical patterns cannot predict.
It also reports historical Monday/Thursday draw statistics (hottest numbers, overdue numbers) for interest.

## Maryland Multi-Match assumptions used here

- Choose **6 unique numbers**
- Numbers come from **1 to 43**
- Drawings are modeled for **Monday** and **Thursday**
- A ticket contains **3 lines**, all randomly generated with no duplicate lines within a ticket

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

`generate_smart_tickets()` calls this pattern multiple times to produce the "Random Tickets" shown by `main()`: every line, on every ticket, is freshly randomized on each run of the script.

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

## Run the demo

```bash
cd multi-match-lottery-game
python lottery_game.py
```

Example output includes:

- Monday analysis
- Thursday analysis
- a suggested Thursday ticket

## Run the tests

```bash
cd multi-match-lottery-game
python -m unittest discover
```

## Notes

- This project is an educational analysis tool, not a guarantee of lottery results.
- The built-in history can be replaced with newer Maryland draw data if you want to update the hottest/overdue statistics.
