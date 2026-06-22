"""
Fetches live Maryland Multi-Match draw results from mdlottery.com.

Falls back gracefully if the site is unreachable or the page structure changes.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import datetime

RESULTS_URL = "https://www.mdlottery.com/player-tools/winning-numbers/"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_live_results() -> tuple:
    """
    Download and parse recent Multi-Match draw records from mdlottery.com.

    Returns (records, error_msg).  records is a tuple[DrawRecord, ...] on
    success, None on failure.  error_msg is a str description on failure,
    None on success.
    """
    from lottery_game import DrawRecord, SUPPORTED_DRAW_DAYS  # local import avoids circular dep

    try:
        req = urllib.request.Request(RESULTS_URL, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return None, f"network error: {exc}"

    records = _parse_html(html, DrawRecord, SUPPORTED_DRAW_DAYS)
    if not records:
        return None, "parse failed: Multi-Match section not found or empty"
    return records, None


def _parse_html(html: str, DrawRecord, SUPPORTED_DRAW_DAYS) -> tuple | None:
    """Extract Multi-Match draw records from the full page HTML."""
    lower = html.lower()

    # Locate the Multi-Match section by its anchor id
    start = lower.find('id="multi-match"')
    if start == -1:
        start = lower.find("multi-match")
    if start == -1:
        return None

    chunk = html[start:]

    # Bound the search to this section only (stop at the next game section)
    next_id = re.search(r'id="(?!multi-match)[a-z][a-z0-9-]*"', chunk[20:])
    if next_id:
        chunk = chunk[: 20 + next_id.start()]

    # Strip HTML tags so only text content remains
    text = re.sub(r"<[^>]+>", " ", chunk)
    text = re.sub(r"[ \t]+", " ", text)

    date_re = re.compile(r"(\d{2}/\d{2}/(?:\d{2}|\d{4}))")
    # Match 1-2 digit numbers not adjacent to other digits
    num_re = re.compile(r"(?<!\d)(\d{1,2})(?!\d)")

    records: list = []
    for dm in date_re.finditer(text):
        date_str = dm.group(1)
        y_part = date_str.split("/")[2]
        try:
            fmt = "%m/%d/%y" if len(y_part) == 2 else "%m/%d/%Y"
            draw_date = datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

        if draw_date.strftime("%A") not in SUPPORTED_DRAW_DAYS:
            continue

        # Collect first 6 valid lottery numbers (1-43) in the 300-char window after the date
        window = text[dm.end() : dm.end() + 300]
        candidates: list[int] = []
        for nm in num_re.finditer(window):
            n = int(nm.group(1))
            if 1 <= n <= 43:
                candidates.append(n)
            if len(candidates) == 6:
                break

        if len(candidates) == 6:
            try:
                records.append(DrawRecord(draw_date, tuple(sorted(candidates))))
            except (ValueError, TypeError):
                pass

    if not records:
        return None

    # Deduplicate by draw_date and sort chronologically
    seen: dict = {}
    for r in records:
        seen.setdefault(r.draw_date, r)
    return tuple(sorted(seen.values(), key=lambda r: r.draw_date))


def merge_history(static: tuple, live: tuple) -> tuple:
    """Combine static and live records; live records win on date conflicts."""
    if live is None:
        return static
    live_dates = {r.draw_date for r in live}
    merged = list(live) + [r for r in static if r.draw_date not in live_dates]
    merged.sort(key=lambda r: r.draw_date)
    return tuple(merged)
