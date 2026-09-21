"""
One-off downloader: pulls the full Maryland Multi-Match draw history (2010 -> present)
from lottery.net's year-archive pages and writes data/multi_match_history.csv.

lottery.net's archive is the earliest publicly available full-history source (it starts
April 2010; mdlottery.com's own site only exposes a rolling recent window). Run this
again later to refresh with newer draws.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
FIRST_YEAR = 2010
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "multi_match_history.csv")

ROW_RE = re.compile(
    r'<td class="colour noBefore">([A-Za-z]+ \d{1,2}, \d{4})</td>.*?<ul class="balls">(.*?)</ul>',
    re.DOTALL,
)
BALL_RE = re.compile(r'<li class="ball md-multi-match ball">(\d{1,2})</li>')


def fetch_year(year: int) -> str:
    url = f"https://www.lottery.net/maryland/multi-match/numbers/{year}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_year(html: str) -> list[tuple]:
    records = []
    for date_str, balls_html in ROW_RE.findall(html):
        draw_date = datetime.strptime(date_str, "%B %d, %Y").date()
        numbers = tuple(sorted(int(n) for n in BALL_RE.findall(balls_html)))
        if len(numbers) != 6:
            print(f"  ! skipping {draw_date}: found {len(numbers)} numbers", file=sys.stderr)
            continue
        records.append((draw_date, numbers))
    return records


def main() -> None:
    last_year = datetime.now().year
    all_records: dict = {}

    for year in range(FIRST_YEAR, last_year + 1):
        print(f"Fetching {year}...")
        try:
            html = fetch_year(year)
        except Exception as exc:
            print(f"  ! failed to fetch {year}: {exc}", file=sys.stderr)
            continue
        year_records = parse_year(html)
        print(f"  {len(year_records)} draws")
        for draw_date, numbers in year_records:
            all_records[draw_date] = numbers
        time.sleep(1)  # be polite to the server

    sorted_dates = sorted(all_records)
    if not sorted_dates:
        print("No records fetched; aborting.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "n1", "n2", "n3", "n4", "n5", "n6"])
        for draw_date in sorted_dates:
            writer.writerow([draw_date.isoformat(), *all_records[draw_date]])

    print(f"\nWrote {len(sorted_dates)} draws ({sorted_dates[0]} to {sorted_dates[-1]}) to {OUT_PATH}")


if __name__ == "__main__":
    main()
