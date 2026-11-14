"""Decide whether a new signal repeats a setup that has really lost before.

Before a buy the bot asks three questions of its memory:
  1. Has this symbol lost on a similar setup before?        (data/ledger.csv)
  2. Does learnings.md warn about this setup?               (data/learnings.md)
  3. Is this signal repeating a known bad pattern?          (1 or 2 says yes)

"Similar setup" = same symbol and same market conditions at the buy signal:
  - is the fast EMA above or below the slow EMA, and
  - how far the price is above its long average (under 3%, 3-10%, over 10%).

One loss is not a pattern - most trades of a trend strategy lose a little and a few win a lot.
A setup is "known bad" only when it has lost at least `min_losses` times, more often than it
has won, and has lost money overall. A later win lifts the warning.

A second lesson came out of the replayed losses themselves: a symbol that has just lost tends to
be chopping around, and buying its next dip straight away did worse than waiting. So after a
losing trade the bot waits `cooldown_days` before buying that symbol again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

LEARN_MODES = ("replay:raw", "paper")       # outcomes the bot may learn from (never from its own memory replay)
SETUP_IN_REASON = re.compile(r"setup=([a-z_]+\|[a-z]+)")

EMA_TEXT = {"ema_up": "the fast EMA is above the slow EMA", "ema_dn": "the fast EMA is still below the slow EMA"}
DIST_TEXT = {"near": "the price is less than 3% above its long average",
             "mid": "the price is 3-10% above its long average",
             "far": "the price is more than 10% above its long average"}


def setup_key(close: float, sma: float, ema_fast: float, ema_slow: float) -> str:
    gap = close / sma - 1 if sma and sma == sma else 0.0
    return f"{'ema_up' if ema_fast >= ema_slow else 'ema_dn'}|{'near' if gap < 0.03 else 'mid' if gap < 0.10 else 'far'}"


def describe(setup: str) -> str:
    ema, dist = setup.split("|")
    return f"dip recovery while {EMA_TEXT.get(ema, ema)} and {DIST_TEXT.get(dist, dist)}"


def setup_of(reason: str) -> str | None:
    found = SETUP_IN_REASON.search(reason or "")
    return found.group(1) if found else None


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str
    losses: int = 0
    wins: int = 0
    pnl: float = 0.0
    learnings_warns: bool = False


class Memory:
    """The ledger's closed trades and the warnings in learnings.md, ready to be asked about a signal."""

    def __init__(self, ledger_rows: list[dict], warnings: list[dict], min_losses: int = 2, cooldown_days: int = 0):
        self.min_losses = min_losses
        self.cooldown_days = cooldown_days
        self.by_symbol: dict[str, list[tuple[str, float, str]]] = {}
        self.closed: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
        for r in ledger_rows:
            if r["action"] != "SELL" or r["mode"] not in LEARN_MODES or r["outcome"] not in ("win", "loss"):
                continue
            pnl = float(r["pnl"]) if r["pnl"] else 0.0
            self.by_symbol.setdefault(r["symbol"], []).append((r["timestamp"][:10], pnl, r["outcome"]))
            setup = setup_of(r["reason"])
            if setup:
                self.closed.setdefault((r["symbol"], setup), []).append((r["timestamp"][:10], pnl, r["outcome"]))
        for trades in list(self.closed.values()) + list(self.by_symbol.values()):
            trades.sort()
        self.warnings = {(w["symbol"], w["setup"]): w["learned"] for w in warnings}

    @property
    def any_losses(self) -> bool:
        return any(outcome == "loss" for trades in self.closed.values() for _, _, outcome in trades)

    def record(self, symbol: str, setup: str, before: str) -> tuple[int, int, float]:
        """(losses, wins, total P&L) of this symbol + setup among trades that closed before the date."""
        trades = [t for t in self.closed.get((symbol, setup), []) if t[0] < before]
        return (sum(t[2] == "loss" for t in trades), sum(t[2] == "win" for t in trades), sum(t[1] for t in trades))

    def is_bad(self, losses: int, wins: int, pnl: float) -> bool:
        return losses >= self.min_losses and losses > wins and pnl < 0

    def recent_loss(self, symbol: str, as_of: str) -> tuple[str, float] | None:
        """The symbol's latest trade, if it was a loss that closed within the cool-off window before as_of."""
        if self.cooldown_days <= 0:
            return None
        earliest = (date.fromisoformat(as_of) - timedelta(days=self.cooldown_days)).isoformat()
        recent = [t for t in self.by_symbol.get(symbol, []) if earliest <= t[0] < as_of]
        if recent and all(t[2] == "loss" for t in recent):
            return recent[-1][0], recent[-1][1]
        return None

    def check(self, symbol: str, setup: str, as_of: str) -> Decision:
        """as_of: date of the signal (YYYY-MM-DD). Only outcomes known before that date are used."""
        cooling = self.recent_loss(symbol, as_of)
        if cooling:
            l, w, p = self.record(symbol, setup, as_of)
            return Decision(False, f"{symbol} lost ${-cooling[1]:,.0f} on a trade that closed {cooling[0]}; the bot waits "
                                   f"{self.cooldown_days} days after a loss before buying the same symbol again", l, w, p, False)
        losses, wins, pnl = self.record(symbol, setup, as_of)
        learned = self.warnings.get((symbol, setup))
        warned = learned is not None and learned < as_of
        if self.is_bad(losses, wins, pnl):
            why = (f"{symbol} has lost {losses} of {losses + wins} times on this setup ({describe(setup)}), "
                   f"${pnl:,.0f} in total" + (", and learnings.md warns about it" if warned else ""))
            return Decision(False, why, losses, wins, pnl, warned)
        if warned:
            return Decision(False, f"learnings.md warns about this setup on {symbol} ({describe(setup)})",
                            losses, wins, pnl, True)
        if losses:
            note = f"{symbol} has lost {losses} time(s) and won {wins} on this setup - not enough to call it a bad pattern"
        else:
            note = f"no prior loss for {symbol} on this setup"
        return Decision(True, note, losses, wins, pnl, False)


def cooling_off(ledger_rows: list[dict], cooldown_days: int, today: str) -> list[tuple[str, str, float]]:
    """Symbols inside the cool-off window today: [(symbol, date the losing trade closed, its P&L)]."""
    mem = Memory(ledger_rows, [], cooldown_days=cooldown_days)
    found = [(s, *hit) for s in sorted(mem.by_symbol) if (hit := mem.recent_loss(s, today))]
    return found


def derive_lessons(ledger_rows: list[dict], min_losses: int = 2) -> list[dict]:
    """The warnings the ledger supports right now, each with a plain-English sentence."""
    memory = Memory(ledger_rows, [], min_losses)
    lessons = []
    for (symbol, setup), trades in memory.closed.items():
        losses = sum(t[2] == "loss" for t in trades)
        wins = len(trades) - losses
        pnl = sum(t[1] for t in trades)
        if not memory.is_bad(losses, wins, pnl):
            continue
        last_loss = max(t[0] for t in trades if t[2] == "loss")
        text = (f"**{symbol}: {describe(setup)}.** This setup lost {losses} of {losses + wins} times on {symbol} "
                f"(${pnl:,.0f} in total; most recent loss {last_loss}). "
                f"Skip it on {symbol} until the record improves.")
        lessons.append({"symbol": symbol, "setup": setup, "text": text, "learned": last_loss})
    return lessons
