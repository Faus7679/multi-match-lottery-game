# multi-match-lottery-game

This repository contains a small Python implementation of a Maryland-style Multi-Match lottery helper.
It uses a dynamic scoring system to analyze historical Monday and Thursday draw patterns and suggest a predicted line for the next draw.

## Maryland Multi-Match assumptions used here

- Choose **6 unique numbers**
- Numbers come from **1 to 43**
- Drawings are modeled for **Monday** and **Thursday**
- A ticket contains **3 lines**
  - line 1 uses the dynamic prediction
  - lines 2 and 3 are quick-pick lines

## Files

- `lottery_game.py`
  - main game logic
  - dynamic scoring
  - Monday/Thursday analysis
  - ticket generation
  - prints the smart pick and the most recent actual winning numbers in bold/color when the terminal supports it
- `live_data.py`
  - fetches recent draw results from mdlottery.com, with local caching and a static-data fallback
- `analysis.py`
  - deep statistical report: frequency, pairs, sums, gaps, momentum, backtest, and ticket recommendations
- `test_lottery_game.py`
  - focused unit tests

## How the dynamic system works

### Step 1: Load draw history

`sample_maryland_history()` includes recent sample Monday and Thursday Maryland Multi-Match results.

### Step 2: Separate draws by day

The analysis is performed independently for:

- Monday
- Thursday

This keeps the strategy day-specific instead of mixing both draw schedules together.

### Step 3: Score every number from 1 to 43

For the requested draw day, each number gets a dynamic score based on:

- **day hits**: how often the number appeared on that weekday
- **recent hits**: how often it appeared in the most recent three draws for that weekday
- **gap score**: how long it has been since the number last appeared on that weekday
- **exploration bonus**: a small bonus for numbers that have not appeared yet in the filtered history

### Step 4: Predict the next winning line

`predict_winning_line()` sorts the scores and returns the top 6 numbers.

### Step 5: Build a Multi-Match ticket

`build_ticket()` creates a 3-line ticket:

1. the predicted line
2. quick-pick line
3. quick-pick line

### Step 6: Review the Monday and Thursday strategy

`analyze_draw_day()` returns:

- the recommended line
- the hottest numbers
- the most overdue numbers
- the top dynamic scores

This gives a simple strategy view for Maryland Monday and Thursday draw behavior while still treating lottery outcomes as uncertain and random.

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
- The built-in history can be replaced with newer Maryland draw data if you want to update the predictions.
