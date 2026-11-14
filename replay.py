"""Replay the daily strategy over real price history, with and without the memory.

replay:raw     python bot.py --replay-raw     the strategy exactly as written, ignoring memory (the honest baseline).
                                              Its real outcomes are written to data/ledger.csv and any setup with a
                                              real losing record becomes a plain-English lesson in data/learnings.md.
replay:memory  python bot.py --replay-memory  the same replay, but before each buy it asks the memory whether that
                                              symbol + setup has a losing record. It only uses outcomes that were
                                              already known on that date, so it cannot peek at the future.

Prices come from yfinance; nothing is invented, seeded or forced to fail.
"""

from __future__ import annotations

import logging
import math

import adaptive_filter
import bot
import memory

log = logging.getLogger("bot")
DAY = "%Y-%m-%d"


def ledger_rows(trades, mode: str) -> list[dict]:
    rows = []
    for sym, entry_day, exit_day, shares, entry, exit_price, pnl, reason, setup in trades:
        rows.append(memory.row(entry_day.strftime(DAY), sym, "BUY", entry, shares,
                               f"dip recovery in an uptrend; setup={setup}", mode, "open"))
        if reason == "open":
            continue                              # still open when the data ended: no real outcome yet
        rows.append(memory.row(exit_day.strftime(DAY), sym, "SELL", exit_price, shares,
                               f"{reason}; setup={setup}; entry={entry_day.strftime(DAY)}", mode,
                               "win" if pnl > 0 else "loss", pnl))
    return rows


def summary(trades, curve, start_equity: float) -> dict:
    closed = [t for t in trades if t[7] != "open"]
    wins = [t for t in closed if t[6] > 0]
    years = len(curve) / 252
    return {"trades": len(trades), "closed": len(closed), "wins": len(wins), "losses": len(closed) - len(wins),
            "trade_pnl": sum(t[6] for t in trades), "final": float(curve.iloc[-1]),
            "per_year": ((curve.iloc[-1] / start_equity) ** (1 / years) - 1) * 100,
            "worst_drop": float(((curve / curve.cummax()) - 1).min() * 100)}


def describe(label: str, s: dict) -> str:
    return (f"{label}: {s['trades']} trades ({s['wins']} won, {s['losses']} lost, {s['trades'] - s['closed']} still open) | "
            f"strategy trades P&L ${s['trade_pnl']:+,.2f} | account ${s['final']:,.0f} | "
            f"{s['per_year']:.1f}% a year | worst drop {s['worst_drop']:.1f}%")


def trade_line(t) -> str:
    return (f"  {t[0]:<5} {t[1].strftime(DAY)} -> {t[2].strftime(DAY)}  {t[3]:>4} sh  in {t[4]:>8.2f}  out {t[5]:>8.2f}  "
            f"{t[7]:<11} {t[6]:>+9.2f}  [{t[8]}]")


def replay_raw(cfg, equity: float, history: str) -> None:
    memory.ensure()
    frames, start = bot.replay_frames(cfg, history)
    trades, curve = bot.simulate_daily(frames, cfg, equity, start_day=start)       # no entry_filter: memory is ignored
    log.info("REPLAY RAW (memory ignored): %d symbols, %s to %s, starting equity $%s.",
             len(frames), curve.index[0].strftime(DAY), curve.index[-1].strftime(DAY), f"{equity:,.0f}")
    for t in trades[-8:]:
        log.info(trade_line(t))
    log.info(describe("RAW", summary(trades, curve, equity)))

    rows = ledger_rows(trades, "replay:raw")
    memory.append_rows(rows, replace_mode="replay:raw")
    losses = sum(r["outcome"] == "loss" for r in rows)
    log.info("Ledger: wrote %d rows of real replay outcomes to data/ledger.csv (%d losing trades).", len(rows), losses)
    lessons = bot.refresh_learnings(cfg)
    if lessons:
        log.info("Learnings: %d setup(s) have a real losing record - written to data/learnings.md. For example:", len(lessons))
        for item in sorted(lessons, key=lambda x: x["learned"])[-3:]:
            log.info("  %s", item["text"].replace("**", ""))
    else:
        log.info("Learnings: no setup has lost at least %d times with a losing record, so no lesson was written.",
                 cfg.memory_min_losses)


def replay_memory(cfg, equity: float, history: str) -> None:
    memory.ensure()
    mem = bot.load_memory(cfg)
    if not mem.any_losses:
        log.info("REPLAY MEMORY: the ledger holds no real losing trade yet, so there is nothing to learn from. "
                 "Run replay:raw first:  python bot.py --replay-raw")
        return

    frames, start = bot.replay_frames(cfg, history)
    raw_trades, raw_curve = bot.simulate_daily(frames, cfg, equity, start_day=start)
    skipped = []

    def ask_memory(sym, signal_day, setup) -> bool:
        verdict = mem.check(sym, setup, signal_day.strftime(DAY))      # only outcomes known before signal_day
        if not verdict.allow:
            skipped.append((signal_day, sym, setup, verdict))
        return verdict.allow

    trades, curve = bot.simulate_daily(frames, cfg, equity, start_day=start, entry_filter=ask_memory)
    log.info("REPLAY MEMORY (walk-forward: each decision only uses outcomes already known on that day): "
             "%d symbols, %s to %s.", len(frames), curve.index[0].strftime(DAY), curve.index[-1].strftime(DAY))

    raw_by_symbol = {}
    for t in raw_trades:
        raw_by_symbol.setdefault(t[0], []).append(t)
    rows, avoided = [], 0.0
    for signal_day, sym, setup, verdict in skipped:
        nxt = [t for t in raw_by_symbol.get(sym, []) if signal_day < t[1] <= signal_day + bot.pd.Timedelta(days=5)]
        what = f"the raw bot's trade there made ${nxt[0][6]:+,.2f}" if nxt else "the raw bot did not trade it either"
        avoided += nxt[0][6] if nxt else 0.0
        rows.append(memory.row(signal_day.strftime(DAY), sym, "SKIP", "", "",
                               f"memory warning: {verdict.reason}; setup={setup}", "replay:memory", "skipped"))
        if len(rows) <= 8:
            log.info("  SKIP %s on %s: %s -> %s", sym, signal_day.strftime(DAY), verdict.reason, what)
    if len(skipped) > 8:
        log.info("  ... and %d more skips", len(skipped) - 8)
    memory.append_rows(rows, replace_mode="replay:memory")

    raw, new = summary(raw_trades, raw_curve, equity), summary(trades, curve, equity)
    log.info(describe("RAW   ", raw))
    log.info(describe("MEMORY", new))
    if not skipped:
        log.info("Memory had no real prior warning to act on at any point, so both runs are identical. "
                 "Keep paper testing.")
    else:
        log.info(f"Memory skipped {len(skipped)} buy signal(s). The raw trades at those signals made "
                 f"${avoided:+,.2f} in total, so skipping them {'avoided losses' if avoided < 0 else 'gave up gains'}. "
                 f"Account difference: ${new['final'] - raw['final']:+,.0f} "
                 f"({new['per_year'] - raw['per_year']:+.2f} points a year).")

    # What memory says about the latest real bar
    latest = max(f.index[-1] for f in frames.values())
    ranks = bot.trend_ranks({s: f["Close"] for s, f in frames.items() if s != cfg.core_symbol or cfg.core_pct <= 0}, cfg)
    found = False
    for sym in sorted(frames):
        f = frames[sym]
        if (cfg.core_pct > 0 and sym == cfg.core_symbol) or f.index[-1] != latest:
            continue
        bar = bot.add_daily_signals(f, cfg).iloc[-1]
        if not bool(bar["entry"]) or (ranks and ranks.get(sym, 2.0) > cfg.rank_top_pct / 100):
            continue
        found = True
        verdict = mem.check(sym, bot.bar_setup(bar), latest.strftime(DAY))
        log.info("Latest bar %s, %s: %s -> %s", latest.strftime(DAY), sym, verdict.reason,
                 "BUY allowed" if verdict.allow else "SKIP")
    if not found:
        log.info("Latest bar %s: no buy signal in today's picks -> HOLD. Keep paper testing.", latest.strftime(DAY))


def reset() -> None:
    rows, warnings = memory.reset()
    log.info("Memory reset: removed %d ledger row(s) and %d warning(s). data/ledger.csv and data/learnings.md are empty again.",
             rows, warnings)
