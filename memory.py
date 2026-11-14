"""The bot's two-file memory.

data/ledger.csv    every BUY, SELL and SKIP with its real outcome (replays and paper trades)
data/learnings.md  plain-English lessons drawn from those outcomes

Nothing here invents data: rows are only written from trades that a replay over real price
history, or the paper account, actually produced.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LEDGER = DATA_DIR / "ledger.csv"
LEARNINGS = DATA_DIR / "learnings.md"

HEADER = ["timestamp", "symbol", "action", "price", "quantity", "reason", "mode", "outcome", "pnl"]

LEARNINGS_INTRO = """# Learnings

Lessons the bot has drawn from real trade outcomes recorded in `data/ledger.csv`.
The bot reads this file and the ledger before every buy or sell. A warning below makes it
skip that setup on that symbol; it is lifted automatically once the record improves.
Rewritten from the ledger by `python bot.py --replay-raw` and after paper trades close.
"""
NO_LESSONS = "\n## Warnings in force\n\nNone yet. No setup has a losing record in the ledger.\n"
WARN_TAG = re.compile(r"<!-- warn:([^|]+)\|(.+?)\|(\d{4}-\d{2}-\d{2}) -->")


def ensure() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not LEDGER.exists():
        write_ledger([])
    if not LEARNINGS.exists():
        LEARNINGS.write_text(LEARNINGS_INTRO + NO_LESSONS, encoding="utf-8")


def row(timestamp, symbol, action, price, quantity, reason, mode, outcome="", pnl="") -> dict:
    fmt = lambda v, n: "" if v in ("", None) else f"{float(v):.{n}f}"   # noqa: E731
    return {"timestamp": str(timestamp), "symbol": symbol, "action": action, "price": fmt(price, 2),
            "quantity": "" if quantity in ("", None) else f"{float(quantity):g}", "reason": reason, "mode": mode,
            "outcome": outcome, "pnl": fmt(pnl, 2)}


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("timestamp")]


def write_ledger(rows: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def append_rows(new_rows: list[dict], replace_mode: str | None = None) -> int:
    """Add rows to the ledger, skipping exact repeats. `replace_mode` first drops that mode's old rows,
    so re-running a replay refreshes its rows instead of piling up copies. Returns how many were added."""
    rows = [r for r in read_ledger() if r["mode"] != replace_mode]
    seen = {(r["timestamp"], r["symbol"], r["action"], r["mode"]) for r in rows}
    added = 0
    for r in new_rows:
        key = (r["timestamp"], r["symbol"], r["action"], r["mode"])
        if key not in seen:
            seen.add(key)
            rows.append(r)
            added += 1
    write_ledger(rows)
    return added


def read_learnings() -> str:
    return LEARNINGS.read_text(encoding="utf-8") if LEARNINGS.exists() else ""


def parse_warnings(text: str) -> list[dict]:
    """The warnings in learnings.md as [{symbol, setup, learned}]."""
    return [{"symbol": m.group(1), "setup": m.group(2), "learned": m.group(3)} for m in WARN_TAG.finditer(text)]


def write_learnings(lessons: list[dict], extra: str = "") -> None:
    """lessons: [{symbol, setup, text, learned}] - text is the plain-English sentence.
    extra: further plain-English sections (for example the cool-off rule), placed above the warnings."""
    DATA_DIR.mkdir(exist_ok=True)
    if not lessons:
        LEARNINGS.write_text(LEARNINGS_INTRO + extra + NO_LESSONS, encoding="utf-8")
        return
    lines = [LEARNINGS_INTRO + extra, "\n## Warnings in force\n"]
    for item in sorted(lessons, key=lambda x: (x["symbol"], x["setup"])):
        lines.append(f"- {item['text']} <!-- warn:{item['symbol']}|{item['setup']}|{item['learned']} -->")
    LEARNINGS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reset() -> tuple[int, int]:
    """Wipe the memory back to empty files. Returns (ledger rows removed, warnings removed)."""
    rows, warnings = len(read_ledger()), len(parse_warnings(read_learnings()))
    write_ledger([])
    LEARNINGS.write_text(LEARNINGS_INTRO + NO_LESSONS, encoding="utf-8")
    return rows, warnings
