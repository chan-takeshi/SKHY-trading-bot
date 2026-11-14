"""Trading bot for an Alpaca PAPER account. config.toml's [strategy] mode picks one of two strategies.

mode = "daily" (see the "Daily strategy" section further down)
  Once a day, just after the open, on a list of symbols: buy a dip that is recovering inside an
  uptrend, hold while the trend lasts, sell when it ends or the stop-loss is hit. Then the bot exits.
    python bot.py                      trade today's signals (waits for the open if started before it)
    python bot.py --dry-run            show today's signals and orders without sending anything
    python bot.py --backtest           replay the strategy (add --history 5y to change the span)

mode = "intraday": mean reversion on one symbol's 15-minute bars, described below.

Setup it looks for, on completed bars only:
  1. Uptrend      - the close is above the long EMA.
  2. Oversold     - the previous bar closed below the lower Bollinger Band.
  3. Reversal     - the latest bar closed back above the lower band.

When all three line up it buys with a bracket order: take-profit at the middle
Bollinger Band, stop-loss a multiple of ATR below entry, and a share count
sized so a stopped-out trade loses a fixed percentage of account equity.

Two [execution] settings in config.toml shape how trades are handled:
  entry_order = "limit"         buy at the signal bar's close; cancel if unfilled after one bar
  flatten_before_close = true   sell out during the session's last bar, so nothing is held overnight

Usage:
  python bot.py              run continuously, checking after every bar closes
  python bot.py --once       run one check and exit
  python bot.py --dry-run    do everything except send, cancel or close orders
  python bot.py --backscan   list every setup found in the fetched history
  python bot.py --backtest   replay the strategy over the fetched history
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import tomllib
import urllib.request
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from dotenv import load_dotenv

import adaptive_filter
import memory

BASE_DIR = Path(__file__).resolve().parent
NY = "America/New_York"
POST_CLOSE_DELAY = pd.Timedelta(seconds=20)   # give Yahoo time to publish the finished bar
MAX_SIGNAL_AGE = pd.Timedelta(minutes=5)      # never chase a signal older than this
FRESH_BAR_ATTEMPTS = 4
FRESH_BAR_WAIT_SECONDS = 10
CANCEL_WAIT_SECONDS = 15                      # how long to wait for cancelled orders to clear
CHECK_RETRIES = 3                             # retries within a bar when a check fails
RETRY_SECONDS = 20
CONNECT_ATTEMPTS = 10                         # startup tries; the network can lag after the PC wakes
CONNECT_RETRY_SECONDS = 30
STAY_AWAKE_BEFORE_OPEN = pd.Timedelta(hours=3)  # keep the PC awake once the open is this close

log = logging.getLogger("bot")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    symbol: str
    interval: str
    lookback_days: int
    ema_period: int
    bb_period: int
    bb_std: float
    atr_period: int
    risk_per_trade_pct: float
    stop_atr_multiple: float
    max_position_pct: float
    entry_order: str
    flatten_before_close: bool
    time_in_force: str
    slippage_pct: float
    # Daily strategy (mode = "daily"); the defaults keep older config files working.
    mode: str = "intraday"
    daily_symbols: tuple[str, ...] = ()
    trend_sma: int = 200
    fast_ema: int = 20
    slow_ema: int = 50
    daily_stop_atr: float = 3.0
    daily_risk_pct: float = 1.0
    daily_max_position_pct: float = 20.0
    daily_history: str = "2y"
    core_symbol: str = ""            # a plain holding kept at core_pct of the account, alongside the strategy
    core_pct: float = 0.0
    core_band_pct: float = 5.0       # only top up or trim the core when it is this far (in % of equity) from target
    rank_lookback: int = 252         # daily picks: rank the list by its return over this many days...
    rank_top_pct: float = 100.0      # ...and only buy dips in this top share of it (100 = no ranking)
    memory_enabled: bool = False     # consult data/ledger.csv and data/learnings.md before trading
    memory_min_losses: int = 2       # real losses on the same symbol + setup before it counts as a bad pattern
    memory_cooldown_days: int = 0    # days to wait after a losing trade before buying the same symbol again (0 = off)

    @property
    def bar_length(self) -> pd.Timedelta:
        unit = self.interval[-1]
        count = int(self.interval[:-1])
        return pd.Timedelta(minutes=count * (60 if unit == "h" else 1))


def load_config(path: Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    try:
        cfg = Config(
            symbol=str(raw["market"]["symbol"]).upper().strip(),
            interval=str(raw["market"]["interval"]).lower().strip(),
            lookback_days=int(raw["market"]["lookback_days"]),
            ema_period=int(raw["indicators"]["ema_period"]),
            bb_period=int(raw["indicators"]["bb_period"]),
            bb_std=float(raw["indicators"]["bb_std"]),
            atr_period=int(raw["indicators"]["atr_period"]),
            risk_per_trade_pct=float(raw["risk"]["risk_per_trade_pct"]),
            stop_atr_multiple=float(raw["risk"]["stop_atr_multiple"]),
            max_position_pct=float(raw["risk"]["max_position_pct"]),
            entry_order=str(raw["execution"]["entry_order"]).lower().strip(),
            flatten_before_close=bool(raw["execution"]["flatten_before_close"]),
            time_in_force=str(raw["execution"]["time_in_force"]).lower().strip(),
            slippage_pct=float(raw["backtest"]["slippage_pct"]),
        )
        daily = raw.get("daily", {})
        cfg = replace(
            cfg,
            mode=str(raw.get("strategy", {}).get("mode", "intraday")).lower().strip(),
            daily_symbols=tuple(str(s).upper().strip() for s in daily.get("symbols", [])),
            trend_sma=int(daily.get("trend_sma", cfg.trend_sma)),
            fast_ema=int(daily.get("fast_ema", cfg.fast_ema)),
            slow_ema=int(daily.get("slow_ema", cfg.slow_ema)),
            daily_stop_atr=float(daily.get("stop_atr_multiple", cfg.daily_stop_atr)),
            daily_risk_pct=float(daily.get("risk_per_trade_pct", cfg.daily_risk_pct)),
            daily_max_position_pct=float(daily.get("max_position_pct", cfg.daily_max_position_pct)),
            daily_history=str(daily.get("history", cfg.daily_history)).lower().strip(),
            core_symbol=str(daily.get("core_symbol", "")).upper().strip(),
            core_pct=float(daily.get("core_pct", 0.0)),
            core_band_pct=float(daily.get("core_rebalance_band_pct", cfg.core_band_pct)),
            rank_lookback=int(daily.get("rank_lookback", cfg.rank_lookback)),
            rank_top_pct=float(daily.get("rank_top_pct", cfg.rank_top_pct)),
            memory_enabled=bool(raw.get("memory", {}).get("enabled", False)),
            memory_min_losses=int(raw.get("memory", {}).get("min_losses", cfg.memory_min_losses)),
            memory_cooldown_days=max(0, int(raw.get("memory", {}).get("cooldown_days", 0))),
        )
    except KeyError as missing:
        raise ValueError(f"{path.name} is missing the setting {missing}") from None

    problems = []
    if cfg.mode not in ("intraday", "daily"):
        problems.append('strategy.mode must be "intraday" or "daily"')
    if cfg.mode == "daily" and not cfg.daily_symbols:
        problems.append("daily.symbols must list at least one symbol")
    for name in ("trend_sma", "fast_ema", "slow_ema"):
        if getattr(cfg, name) < 1:
            problems.append(f"daily {name} must be at least 1")
    if cfg.fast_ema >= cfg.slow_ema:
        problems.append("daily.fast_ema must be shorter than daily.slow_ema")
    if cfg.rank_lookback < 1 or not 0 < cfg.rank_top_pct <= 100:
        problems.append("daily.rank_lookback must be at least 1 and daily.rank_top_pct between 0 and 100")
    if cfg.memory_min_losses < 1:
        problems.append("memory.min_losses must be at least 1")
    if not 0 <= cfg.core_pct < 100:
        problems.append("daily.core_pct must be from 0 up to (not including) 100")
    if cfg.core_pct > 0 and not cfg.core_symbol:
        problems.append("daily.core_symbol is needed when core_pct is above 0")
    if cfg.core_band_pct <= 0:
        problems.append("daily.core_rebalance_band_pct must be greater than 0")
    if cfg.daily_stop_atr <= 0:
        problems.append("daily.stop_atr_multiple must be greater than 0")
    for name in ("daily_risk_pct", "daily_max_position_pct"):
        if not 0 < getattr(cfg, name) <= 100:
            problems.append(f"{name} must be between 0 and 100")
    if not cfg.symbol:
        problems.append("market.symbol is empty")
    if cfg.interval[-1:] not in ("m", "h") or not cfg.interval[:-1].isdigit():
        problems.append('market.interval must look like "15m" or "1h"')
    for name in ("lookback_days", "ema_period", "bb_period", "atr_period"):
        if getattr(cfg, name) < 1:
            problems.append(f"{name} must be at least 1")
    for name in ("bb_std", "stop_atr_multiple"):
        if getattr(cfg, name) <= 0:
            problems.append(f"{name} must be greater than 0")
    for name in ("risk_per_trade_pct", "max_position_pct"):
        if not 0 < getattr(cfg, name) <= 100:
            problems.append(f"{name} must be between 0 and 100")
    if cfg.entry_order not in ("limit", "market"):
        problems.append('execution.entry_order must be "limit" or "market"')
    if cfg.time_in_force not in ("day", "gtc"):
        problems.append('execution.time_in_force must be "day" or "gtc"')
    if cfg.slippage_pct < 0:
        problems.append("backtest.slippage_pct cannot be negative")
    if problems:
        raise ValueError("Invalid config: " + "; ".join(problems))
    return cfg


# --------------------------------------------------------------------------
# Market data and indicators
# --------------------------------------------------------------------------

def fetch_bars(cfg: Config) -> pd.DataFrame:
    """Download recent bars and keep only the ones that have finished."""
    df = yf.Ticker(cfg.symbol).history(
        period=f"{cfg.lookback_days}d",
        interval=cfg.interval,
        auto_adjust=False,
        prepost=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no {cfg.interval} data for {cfg.symbol}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = df.index.tz_localize(NY) if df.index.tz is None else df.index.tz_convert(NY)

    # The newest row is usually a bar still in progress; its close is not final.
    now = pd.Timestamp.now(tz=NY)
    return df[df.index + cfg.bar_length <= now]


def fetch_fresh_bars(cfg: Config) -> pd.DataFrame:
    """Fetch bars, retrying briefly until the bar that just closed shows up."""
    df = fetch_bars(cfg)
    for _ in range(FRESH_BAR_ATTEMPTS - 1):
        expected_close = last_bar_boundary(pd.Timestamp.now(tz=NY), cfg.bar_length)
        if not df.empty and df.index[-1] + cfg.bar_length >= expected_close:
            break
        time.sleep(FRESH_BAR_WAIT_SECONDS)
        df = fetch_bars(cfg)
    return df


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    out["ema"] = close.ewm(span=cfg.ema_period, adjust=False, min_periods=cfg.ema_period).mean()

    out["bb_mid"] = close.rolling(cfg.bb_period).mean()
    band = cfg.bb_std * close.rolling(cfg.bb_period).std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + band
    out["bb_lower"] = out["bb_mid"] - band

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    # Wilder's smoothing, the standard ATR definition.
    out["atr"] = true_range.ewm(alpha=1 / cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    return out


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["uptrend"] = out["Close"] > out["ema"]
    out["was_below"] = out["Close"].shift(1) < out["bb_lower"].shift(1)
    out["back_inside"] = out["Close"] > out["bb_lower"]
    out["signal"] = out["uptrend"] & out["was_below"] & out["back_inside"]
    return out


# --------------------------------------------------------------------------
# Position sizing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TradePlan:
    symbol: str
    entry: float                      # the signal bar's close; the limit price for limit entries
    take_profit: float
    stop_loss: float
    atr: float
    shares: int
    capped: bool = False              # True when a cap, not the ATR, set the share count
    reject_reason: str | None = None

    @property
    def risk_dollars(self) -> float:
        return self.shares * (self.entry - self.stop_loss)

    @property
    def reward_dollars(self) -> float:
        return self.shares * (self.take_profit - self.entry)


def build_plan(bar, equity: float, buying_power: float, cfg: Config) -> TradePlan:
    entry = round(float(bar["Close"]), 2)
    atr = float(bar["atr"])
    stop_distance = cfg.stop_atr_multiple * atr
    take_profit = round(float(bar["bb_mid"]), 2)
    stop_loss = round(entry - stop_distance, 2)

    # Volatility sizing: wider ATR -> wider stop -> fewer shares, same dollars at risk.
    risk_budget = equity * cfg.risk_per_trade_pct / 100
    by_risk = math.floor(risk_budget / stop_distance) if stop_distance > 0 else 0
    by_position_cap = math.floor(equity * cfg.max_position_pct / 100 / entry)
    by_buying_power = math.floor(buying_power / entry)
    shares = max(0, min(by_risk, by_position_cap, by_buying_power))

    reason = None
    if shares < 1:
        reason = "position size works out to less than 1 share"
    elif take_profit <= entry + 0.01:
        reason = f"take-profit {take_profit:.2f} is not above entry {entry:.2f}"
    elif stop_loss <= 0 or stop_loss >= entry - 0.01:
        reason = f"stop-loss {stop_loss:.2f} is not a valid price below entry {entry:.2f}"

    return TradePlan(cfg.symbol, entry, take_profit, stop_loss, atr, shares, shares < by_risk, reason)


# --------------------------------------------------------------------------
# Alpaca
# --------------------------------------------------------------------------

class Broker:
    """Thin wrapper around the Alpaca paper trading API."""

    def __init__(self, api_key: str, secret_key: str):
        # paper=True is deliberate: this bot only ever talks to the paper endpoint.
        self.client = TradingClient(api_key, secret_key, paper=True)
        self._keys = (api_key, secret_key)
        self._data = None

    def positions(self) -> dict[str, tuple[float, float]]:
        """Every open position as {symbol: (shares, market value)}."""
        return {p.symbol: (float(p.qty), float(p.market_value)) for p in self.client.get_all_positions()}

    def position_pnl(self, symbol: str) -> float | None:
        """Unrealised profit or loss on an open position, in dollars."""
        for pos in self.client.get_all_positions():
            if pos.symbol == symbol:
                return float(pos.unrealized_pl)
        return None

    def last_price(self, symbol: str) -> float | None:
        """Latest traded price (read-only market data), or None if it cannot be fetched."""
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest
            if self._data is None:
                self._data = StockHistoricalDataClient(*self._keys)
            request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            return float(self._data.get_stock_latest_trade(request)[symbol].price)
        except Exception as err:
            log.warning("Could not read the latest %s price (%s).", symbol, err)
            return None

    def submit_market(self, symbol: str, shares: int, side: OrderSide, client_order_id: str | None = None):
        """Plain market order for today's session (used for the core holding)."""
        return self.client.submit_order(MarketOrderRequest(
            client_order_id=client_order_id, symbol=symbol, qty=shares, side=side, time_in_force=TimeInForce.DAY,
        ))

    def submit_buy_with_stop(self, symbol: str, shares: int, stop_price: float, client_order_id: str | None = None):
        """Market buy with a stop-loss attached. GTC, so the stop stays working on later days."""
        return self.client.submit_order(MarketOrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.OTO,
            stop_loss=StopLossRequest(stop_price=stop_price),
        ))

    def account(self) -> tuple[float, float]:
        acct = self.client.get_account()
        return float(acct.equity), float(acct.buying_power)

    def clock(self):
        return self.client.get_clock()

    def position_qty(self, symbol: str) -> float:
        for pos in self.client.get_all_positions():
            if pos.symbol == symbol:
                return float(pos.qty)
        return 0.0

    def open_orders(self, symbol: str) -> list:
        return self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))

    def cancel_all(self, symbol: str) -> None:
        """Cancel every open order in the symbol and wait until they have cleared."""
        for order in self.open_orders(symbol):
            try:
                self.client.cancel_order_by_id(order.id)
            except APIError:
                pass  # typically the other leg of a bracket, already cancelled with its sibling
        for _ in range(CANCEL_WAIT_SECONDS):
            if not self.open_orders(symbol):
                return
            time.sleep(1)
        raise RuntimeError(f"{symbol} orders are still open {CANCEL_WAIT_SECONDS}s after cancelling them")

    def close_position(self, symbol: str):
        return self.client.close_position(symbol)

    def submit_bracket_buy(self, plan: TradePlan, cfg: Config, client_order_id: str | None = None):
        common = dict(
            client_order_id=client_order_id,   # Alpaca refuses a second order with the same id
            symbol=plan.symbol,
            qty=plan.shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce(cfg.time_in_force),
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=plan.take_profit),
            stop_loss=StopLossRequest(stop_price=plan.stop_loss),
        )
        if cfg.entry_order == "limit":
            return self.client.submit_order(LimitOrderRequest(limit_price=plan.entry, **common))
        return self.client.submit_order(MarketOrderRequest(**common))


def connect_broker(required: bool) -> Broker | None:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    missing = not api_key or not secret_key or "your_paper" in api_key or "your_paper" in secret_key
    if missing:
        if required:
            log.error("Alpaca credentials are not set. Open the .env file and paste in your paper API key and secret.")
            sys.exit(1)
        log.warning("No Alpaca credentials in .env - continuing without an account connection.")
        return None

    broker = Broker(api_key, secret_key)
    attempts = CONNECT_ATTEMPTS if required else 1
    for attempt in range(1, attempts + 1):
        try:
            equity, buying_power = broker.account()
            break
        except Exception as err:
            if isinstance(err, APIError) and err.status_code in (401, 403):
                log.error("Alpaca rejected the API key in .env (%s). Check the paper key and secret.", err)
                sys.exit(1)
            if attempt == attempts:
                raise
            log.warning("Could not reach Alpaca (%s) - attempt %d of %d, retrying in %d seconds.",
                        err, attempt, attempts, CONNECT_RETRY_SECONDS)
            time.sleep(CONNECT_RETRY_SECONDS)
    log.info("Connected to Alpaca paper account: equity $%s, buying power $%s",
             f"{equity:,.2f}", f"{buying_power:,.2f}")
    return broker


# --------------------------------------------------------------------------
# Bot logic
# --------------------------------------------------------------------------

def last_bar_boundary(now: pd.Timestamp, bar_length: pd.Timedelta) -> pd.Timestamp:
    """Most recent bar close at or before `now` (bars are anchored to the 9:30 open)."""
    anchor = now.normalize() + pd.Timedelta(hours=9, minutes=30)
    bars_elapsed = math.floor((now - anchor) / bar_length)
    return anchor + bars_elapsed * bar_length


DISCORD_PREFIXES = ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")


def notify(text: str) -> bool:
    """Post a message to Discord when the bot trades. Returns True if it was sent.

    Set DISCORD_WEBHOOK_URL in .env to switch this on; add DISCORD_USER_ID (your numeric Discord id)
    to be @-mentioned. A failed message is only logged - it must never stop or delay a trade.
    """
    url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        return False
    if not url.startswith(DISCORD_PREFIXES):
        log.warning("DISCORD_WEBHOOK_URL in .env is not a Discord webhook address - no message sent.")
        return False
    user = os.getenv("DISCORD_USER_ID", "").strip()
    mention = user.isdigit()
    payload = {
        "content": (f"<@{user}> " if mention else "") + text[:1800],
        "allowed_mentions": {"users": [user]} if mention else {"parse": []},
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "trading-bot/1.0"},
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
        return True
    except Exception as err:
        log.warning("Discord message could not be sent (%s).", type(err).__name__)
        return False


def keep_awake(on: bool) -> None:
    """Ask Windows not to go to sleep while the bot needs to act (no effect on other systems).

    A scheduled task can wake the PC, but Windows puts it back to sleep after a couple of idle
    minutes unless a program asks otherwise - which would freeze the bot mid-session.
    """
    if sys.platform != "win32":
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def acquire_single_instance_lock():
    """Hold bot.lock for the life of the process. Returns None if another live bot already holds it."""
    handle = open(BASE_DIR / "bot.lock", "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def sleep_until(when: pd.Timestamp) -> None:
    while True:
        remaining = (when - pd.Timestamp.now(tz=NY)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 60))


def describe_plan(plan: TradePlan, cfg: Config) -> str:
    price = f"limit {plan.entry:.2f}" if cfg.entry_order == "limit" else f"at market (~{plan.entry:.2f})"
    return (
        f"BUY {plan.shares} {plan.symbol} {price} | "
        f"take-profit {plan.take_profit:.2f} | stop-loss {plan.stop_loss:.2f} | "
        f"ATR {plan.atr:.2f} | risk ${plan.risk_dollars:,.0f} for reward ${plan.reward_dollars:,.0f}"
        + (" | size capped by position/buying-power limit" if plan.capped else "")
    )


def manage_position(cfg: Config, broker: Broker, clock, dry_run: bool) -> bool:
    """Look after working orders and any open position. Returns True if no new entry should follow."""
    now = pd.Timestamp(clock.timestamp)
    orders = broker.open_orders(cfg.symbol)
    qty = broker.position_qty(cfg.symbol)

    # 1. A buy order that has sat unfilled for a whole bar: the setup has passed.
    #    That is any buy sent before the latest bar close: it has had its one bar to fill.
    latest_close = last_bar_boundary(now.tz_convert(NY), cfg.bar_length)
    if any(o.side == OrderSide.BUY and pd.Timestamp(o.submitted_at) < latest_close for o in orders):
        if dry_run:
            log.info("DRY RUN - would cancel the entry order that did not fill within one bar.")
            return True
        broker.cancel_all(cfg.symbol)
        log.info("Entry order did not fill within one bar - cancelled.")
        orders = []
        qty = broker.position_qty(cfg.symbol)
        if qty > 0:
            # Partly filled: cancelling the entry also cancelled its take-profit and stop-loss.
            order = broker.close_position(cfg.symbol)
            log.info("Entry was only partly filled - selling the %g shares bought (order %s).", qty, order.id)
            notify(f"SELL {qty:g} {cfg.symbol} at market: the entry order only partly filled, so the part bought is being sold.")
            return True

    # 2. Final bar of the session: go flat instead of carrying the position through the overnight gap.
    minutes_left = (pd.Timestamp(clock.next_close) - now).total_seconds() / 60
    if cfg.flatten_before_close and minutes_left <= cfg.bar_length.total_seconds() / 60:
        if qty == 0 and not orders:
            log.info("Final bar of the session - no new entries.")
        elif dry_run:
            log.info("DRY RUN - would cancel %d open order(s) and sell %g shares before the close.", len(orders), qty)
        else:
            broker.cancel_all(cfg.symbol)
            qty = broker.position_qty(cfg.symbol)   # re-read: an entry may have filled just before the cancel
            if qty > 0:
                order = broker.close_position(cfg.symbol)
                log.info("END OF DAY: selling %g %s before the close (order %s).", qty, cfg.symbol, order.id)
                notify(f"SELL {qty:g} {cfg.symbol} at market: end of day, nothing is held overnight.")
        return True

    if qty > 0:
        if any(o.side == OrderSide.SELL for o in orders):
            log.info("Holding %g %s; take-profit and stop-loss are working.", qty, cfg.symbol)
        else:
            log.warning("Holding %g %s with NO take-profit or stop-loss order attached.", qty, cfg.symbol)
    return False


def check_entry(cfg: Config, broker: Broker | None, market_open: bool | None, dry_run: bool,
                fallback_equity: float, state: dict) -> None:
    bars = fetch_fresh_bars(cfg) if market_open else fetch_bars(cfg)
    if len(bars) <= max(cfg.ema_period, cfg.bb_period, cfg.atr_period):
        log.warning("Only %d bars available - not enough to calculate the indicators.", len(bars))
        return

    df = add_signals(add_indicators(bars, cfg))
    bar = df.iloc[-1]
    bar_time = df.index[-1]

    def yes_no(flag) -> str:
        return "yes" if bool(flag) else "no"

    log.info(
        "%s bar %s | close %.2f | EMA%d %.2f | BB lower %.2f mid %.2f | ATR %.2f | "
        "uptrend=%s was-below-band=%s back-inside=%s",
        cfg.symbol, bar_time.strftime("%Y-%m-%d %H:%M"), bar["Close"], cfg.ema_period, bar["ema"],
        bar["bb_lower"], bar["bb_mid"], bar["atr"],
        yes_no(bar["uptrend"]), yes_no(bar["was_below"]), yes_no(bar["back_inside"]),
    )

    if not bar["signal"]:
        log.info("No setup on this bar.")
        return
    if state.get("last_signal_bar") == bar_time:
        return  # already handled this bar

    equity, buying_power = broker.account() if broker else (fallback_equity, fallback_equity)
    plan = build_plan(bar, equity, buying_power, cfg)
    log.info("SETUP FOUND: %s", describe_plan(plan, cfg))

    blockers = []
    if plan.reject_reason:
        blockers.append(plan.reject_reason)
    if market_open is False:
        blockers.append("market is closed")
    age = pd.Timestamp.now(tz=NY) - (bar_time + cfg.bar_length)
    if age > MAX_SIGNAL_AGE:
        blockers.append(f"signal bar closed {int(age.total_seconds() // 60)} minutes ago (too old to chase)")
    if broker:
        qty = broker.position_qty(cfg.symbol)
        open_orders = broker.open_orders(cfg.symbol)
        if qty > 0:
            blockers.append(f"already holding {qty:g} shares")
        elif open_orders:
            blockers.append(f"{len(open_orders)} open order(s) already working")

    if blockers:
        state["last_signal_bar"] = bar_time
        log.info("Not trading: %s.", "; ".join(blockers))
        return
    if dry_run:
        state["last_signal_bar"] = bar_time
        log.info("DRY RUN - order not sent.")
        return

    # One id per signal bar: if a network error hides whether the order went through, the
    # retry cannot place a second one. Only a definite answer marks the bar as handled.
    client_order_id = f"{cfg.symbol}-{bar_time.strftime('%Y%m%d-%H%M')}-{cfg.entry_order}"
    try:
        order = broker.submit_bracket_buy(plan, cfg, client_order_id)
    except APIError as err:
        state["last_signal_bar"] = bar_time
        log.error("ORDER REJECTED by Alpaca: %s", err)
        notify(f"ORDER REJECTED by Alpaca for {cfg.symbol}: {err}")
        return
    state["last_signal_bar"] = bar_time
    log.info("ORDER SUBMITTED: id %s, status %s", order.id, order.status)
    notify("ORDER SENT: " + describe_plan(plan, cfg))


def run_check(cfg: Config, broker: Broker | None, clock, dry_run: bool,
              fallback_equity: float, state: dict) -> None:
    """One pass: look after any open order or position, then look for a new entry."""
    market_open = clock.is_open if clock else None
    if market_open and manage_position(cfg, broker, clock, dry_run):
        return
    check_entry(cfg, broker, market_open, dry_run, fallback_equity, state)


def run_loop(cfg: Config, broker: Broker | None, dry_run: bool, fallback_equity: float) -> None:
    state: dict = {}
    failures = 0
    log.info("Bot running on %s %s bars%s. Press Ctrl+C to stop.",
             cfg.symbol, cfg.interval, " (DRY RUN)" if dry_run else "")
    while True:
        try:
            clock = broker.clock() if broker else None
            if clock is not None and not clock.is_open:
                wake = pd.Timestamp(clock.next_open).tz_convert(NY) + cfg.bar_length + POST_CLOSE_DELAY
                if wake - pd.Timestamp.now(tz=NY) > STAY_AWAKE_BEFORE_OPEN:
                    # A long wait (overnight, weekend): let the PC sleep, then look again nearer the open.
                    keep_awake(False)
                    log.info("Market is closed. Next check after %s ET.", wake.strftime("%Y-%m-%d %H:%M"))
                    sleep_until(wake - STAY_AWAKE_BEFORE_OPEN)
                    continue
                keep_awake(True)
                log.info("Market is closed. Sleeping until %s ET.", wake.strftime("%Y-%m-%d %H:%M"))
                sleep_until(wake)
                continue
            keep_awake(True)
            run_check(cfg, broker, clock, dry_run, fallback_equity, state)
            failures = 0
        except Exception:
            # Network blips and API hiccups should not kill a long-running bot, and a
            # quick retry matters most for the end-of-day exit.
            failures += 1
            if failures <= CHECK_RETRIES:
                log.exception("Check failed - retrying in %d seconds.", RETRY_SECONDS)
                time.sleep(RETRY_SECONDS)
                continue
            log.exception("Check failed %d times - waiting for the next bar.", failures)
            failures = 0

        now = pd.Timestamp.now(tz=NY)
        sleep_until(last_bar_boundary(now, cfg.bar_length) + cfg.bar_length + POST_CLOSE_DELAY)


def backscan(cfg: Config, equity: float) -> None:
    """Print every setup in the fetched history, sized for the given equity."""
    df = add_signals(add_indicators(fetch_bars(cfg), cfg))
    signals = df[df["signal"]]
    log.info("%d bars from %s to %s, %d setup(s) found. Sized for $%s equity.",
             len(df), df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"),
             len(signals), f"{equity:,.0f}")
    for bar_time, bar in signals.iterrows():
        plan = build_plan(bar, equity, equity, cfg)
        note = f"  [skipped: {plan.reject_reason}]" if plan.reject_reason else ""
        log.info("%s  %s%s", bar_time.strftime("%Y-%m-%d %H:%M"), describe_plan(plan, cfg), note)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Trade:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    stop: float
    target: float
    shares: int
    pnl: float
    r: float          # profit as a multiple of the planned risk (1.0 = made what was risked)
    reason: str       # target / stop / eod (sold before the close) / open (still open when the data ended)
    bars: int
    overnight: bool


def mark_last_bars(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Flag the final bar of each session - the bar in which the live bot sells out."""
    out = df.copy()
    day = out.index.normalize()
    bar_end = pd.Series(out.index + cfg.bar_length, index=out.index)
    session_close = bar_end.groupby(day).transform("max")
    # The newest session may be unfinished: assume it runs to the normal 16:00 close.
    final = day == day[-1]
    session_close[final] = max(session_close.iloc[-1], day[-1] + pd.Timedelta(hours=16))
    out["last_bar"] = bar_end >= session_close
    return out


def simulate(df: pd.DataFrame, cfg: Config, start_equity: float) -> list[Trade]:
    """Replay the bot bar by bar. `df` needs indicators, signals and the last-bar flag.

    Mirrors the live bot: it decides when a bar closes, holds one position at a
    time, and assumes the stop was hit first whenever one bar touches both the
    stop and the target. Market orders (market entries, stops, end-of-day sales)
    pay `slippage_pct`; limit orders fill at their price only if the market
    trades through it.
    """
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    mid, atr = df["bb_mid"].to_numpy(dtype=float), df["atr"].to_numpy(dtype=float)
    signal = df["signal"].to_numpy(dtype=bool)
    last_bar = df["last_bar"].to_numpy(dtype=bool)
    day = df.index.normalize().asi8
    times = df.index
    slip = cfg.slippage_pct / 100
    n = len(df)

    trades: list[Trade] = []
    equity = start_equity
    pos: dict | None = None

    def close_position(i: int, price: float, reason: str) -> None:
        nonlocal pos, equity
        pnl = pos["shares"] * (price - pos["entry"])
        equity += pnl
        trades.append(Trade(
            signal_time=times[pos["entry_idx"] - 1], entry_time=times[pos["entry_idx"]], exit_time=times[i],
            entry=pos["entry"], exit=price, stop=pos["stop"], target=pos["target"], shares=pos["shares"],
            pnl=pnl, r=(price - pos["entry"]) / pos["risk"], reason=reason,
            bars=i - pos["entry_idx"] + 1, overnight=bool(day[i] != day[pos["entry_idx"]]),
        ))
        pos = None

    for i in range(n):
        if pos is not None:
            stop, target = pos["stop"], pos["target"]
            if pos["filled_inside_bar"] == i:
                # A limit entry filled part-way through this bar, so the bar's high may have
                # come before the fill: only count the target if the bar closed above it.
                if l[i] <= stop:
                    close_position(i, stop * (1 - slip), "stop")
                elif c[i] >= target:
                    close_position(i, target, "target")
            elif o[i] <= stop:                                    # gapped through the stop
                close_position(i, o[i] * (1 - slip), "stop")
            elif o[i] >= target:                                  # gapped above the target
                close_position(i, o[i], "target")
            elif cfg.flatten_before_close and last_bar[i]:
                close_position(i, o[i] * (1 - slip), "eod")
            elif l[i] <= stop:
                close_position(i, stop * (1 - slip), "stop")
            elif h[i] >= target:
                close_position(i, target, "target")

        if pos is not None or not signal[i]:
            continue
        if i + 1 >= n or day[i + 1] != day[i]:
            continue  # the signal bar ended the session; the market is shut
        if cfg.flatten_before_close and last_bar[i + 1]:
            continue  # the live bot takes no new entries in the final bar

        plan = build_plan({"Close": c[i], "atr": atr[i], "bb_mid": mid[i]}, equity, equity, cfg)
        if plan.reject_reason:
            continue
        next_open = o[i + 1]
        filled_inside_bar = -1
        if cfg.entry_order == "limit":
            if next_open <= plan.entry:                           # already at or below the limit: fills at once
                if next_open <= plan.stop_loss + 0.01:
                    continue
                entry = min(plan.entry, next_open * (1 + slip))
            elif l[i + 1] < plan.entry - 0.005:                   # traded down through the limit
                entry = plan.entry
                filled_inside_bar = i + 1
            else:
                continue                                          # never filled; the bot cancels it
        else:
            if not plan.stop_loss + 0.01 < next_open < plan.take_profit - 0.01:
                continue  # Alpaca rejects a bracket whose stop/target is on the wrong side of the market
            entry = next_open * (1 + slip)
        pos = {
            "entry_idx": i + 1, "entry": entry, "stop": plan.stop_loss, "target": plan.take_profit,
            "shares": plan.shares, "risk": plan.entry - plan.stop_loss, "filled_inside_bar": filled_inside_bar,
        }

    if pos is not None:
        close_position(n - 1, c[-1], "open")
    return trades


def summarize(trades: list[Trade], start_equity: float) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = [t.pnl for t in trades]
    gains = sum(p for p in pnl if p > 0)
    losses = -sum(p for p in pnl if p < 0)
    equity, peak, max_drawdown = start_equity, start_equity, 0.0
    for p in pnl:
        equity += p
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    return {
        "trades": len(trades),
        "win_rate": sum(p > 0 for p in pnl) / len(trades) * 100,
        "pnl": sum(pnl),
        "return_pct": sum(pnl) / start_equity * 100,
        "avg_r": sum(t.r for t in trades) / len(trades),
        "total_r": sum(t.r for t in trades),
        "profit_factor": gains / losses if losses else float("inf"),
        "max_drawdown_pct": max_drawdown,
        "worst_pnl": min(pnl),
        "worst_r": min(t.r for t in trades),
        "overnight": sum(t.overnight for t in trades),
    }


def backtest(cfg: Config, equity: float) -> None:
    """Replay the strategy over the fetched history and print every trade."""
    df = mark_last_bars(add_signals(add_indicators(fetch_bars(cfg), cfg)), cfg)
    trades = simulate(df, cfg, equity)
    log.info("Backtest: %s, %d bars from %s to %s, starting equity $%s, %s entries, %s.",
             cfg.symbol, len(df), df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"),
             f"{equity:,.0f}", cfg.entry_order,
             "flat before the close" if cfg.flatten_before_close else "positions held overnight")
    for t in trades:
        log.info("%s -> %s  %3d sh  in %.2f  out %.2f  %-6s  %+8.2f  %+.2fR%s",
                 t.entry_time.strftime("%Y-%m-%d %H:%M"), t.exit_time.strftime("%m-%d %H:%M"),
                 t.shares, t.entry, t.exit, t.reason, t.pnl, t.r, "  (held overnight)" if t.overnight else "")
    s = summarize(trades, equity)
    if not s["trades"]:
        log.info("No trades.")
        return
    log.info("Trades %d | win rate %.0f%% | P&L $%+.2f (%+.2f%%) | avg %+.2fR | profit factor %.2f | "
             "max drawdown %.2f%% | worst trade $%+.2f (%+.2fR) | held overnight %d",
             s["trades"], s["win_rate"], s["pnl"], s["return_pct"], s["avg_r"], s["profit_factor"],
             s["max_drawdown_pct"], s["worst_pnl"], s["worst_r"], s["overnight"])


# --------------------------------------------------------------------------
# Daily strategy: buy the dip in an uptrend, hold while the trend lasts
# --------------------------------------------------------------------------
#
# On completed daily bars, for every symbol in [daily] symbols:
#   BUY  when the close is above the long average (uptrend), yesterday closed below the lower
#        Bollinger Band (a dip) and today closed back above it (the dip is recovering).
#   SELL when the fast EMA crosses below the slow EMA or the close falls under the long average
#        (the trend has ended), or when the stop-loss (a multiple of ATR below the entry) is hit.
# Orders go in just after the next open. The bot runs once a day and then exits; the stop-loss
# orders stay working at Alpaca in the meantime.

STATE_FILE = BASE_DIR / "daily_state.json"
DAILY_OPEN_DELAY = pd.Timedelta(minutes=2)      # let the opening auction finish before trading
MAX_DAILY_DATA_AGE_DAYS = 5                     # weekend + holiday; older than this means the data is stale


def fetch_daily_bars(cfg: Config) -> dict[str, pd.DataFrame]:
    """Completed daily bars (adjusted for splits and dividends) for every configured symbol."""
    symbols = list(dict.fromkeys(cfg.daily_symbols + ((cfg.core_symbol,) if cfg.core_pct > 0 else ())))
    raw = yf.download(symbols, period=cfg.daily_history, interval="1d", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no daily data")
    now = pd.Timestamp.now(tz=NY)
    today = now.normalize().tz_localize(None)
    session_over = now >= now.normalize() + pd.Timedelta(hours=16, minutes=10)
    frames = {}
    for sym in symbols:
        try:
            df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
        except KeyError:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_convert(NY).tz_localize(None)
        df.index = df.index.normalize()
        df = df[df.index <= today] if session_over else df[df.index < today]   # today's bar is still forming
        if not df.empty:
            frames[sym] = df
    return frames


def add_daily_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]
    out["sma"] = close.rolling(cfg.trend_sma).mean()
    out["ema_fast"] = close.ewm(span=cfg.fast_ema, adjust=False, min_periods=cfg.fast_ema).mean()
    out["ema_slow"] = close.ewm(span=cfg.slow_ema, adjust=False, min_periods=cfg.slow_ema).mean()
    out["bb_lower"] = close.rolling(cfg.bb_period).mean() - cfg.bb_std * close.rolling(cfg.bb_period).std(ddof=0)
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = true_range.ewm(alpha=1 / cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()

    uptrend = close > out["sma"]
    dipped = prev_close < out["bb_lower"].shift(1)
    recovered = close > out["bb_lower"]
    out["entry"] = uptrend & dipped & recovered
    crossed_down = (out["ema_slow"] > out["ema_fast"]) & (out["ema_slow"].shift(1) <= out["ema_fast"].shift(1))
    out["exit"] = crossed_down | (close < out["sma"])
    return out


def trend_ranks(closes: dict[str, pd.Series], cfg: Config) -> dict[str, float]:
    """Today's ranking of the list by recent trend: {symbol: position from 0 (strongest) to 1 (weakest)}.

    Empty when ranking is off (rank_top_pct = 100). Symbols without enough history are left out.
    """
    if cfg.rank_top_pct >= 100:
        return {}
    look = cfg.rank_lookback
    scores = {s: float(c.iloc[-1] / c.iloc[-1 - look] - 1) for s, c in closes.items() if len(c) > look}
    order = sorted(scores, key=scores.get, reverse=True)
    return {s: (i + 1) / len(order) for i, s in enumerate(order)}


def bar_setup(bar) -> str:
    """The market conditions at a buy signal, as the memory files describe them."""
    return adaptive_filter.setup_key(float(bar["Close"]), float(bar["sma"]), float(bar["ema_fast"]), float(bar["ema_slow"]))


def load_memory(cfg: Config) -> adaptive_filter.Memory:
    """Read both memory files fresh (they are small) and return them ready to be asked about a signal."""
    memory.ensure()
    return adaptive_filter.Memory(memory.read_ledger(), memory.parse_warnings(memory.read_learnings()),
                                  cfg.memory_min_losses, cfg.memory_cooldown_days)


def refresh_learnings(cfg: Config) -> list[dict]:
    """Rewrite data/learnings.md from the real outcomes in the ledger. Returns the warnings now in force."""
    rows = memory.read_ledger()
    lessons = adaptive_filter.derive_lessons(rows, cfg.memory_min_losses)
    extra = ""
    if cfg.memory_cooldown_days > 0:
        today = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
        cooling = adaptive_filter.cooling_off(rows, cfg.memory_cooldown_days, today)
        extra = (f"\n## Lesson: wait after a loss\n\nA symbol that has just lost money tends to be moving sideways, and "
                 f"buying its next dip straight away did worse in the replays than waiting. After a losing trade the bot "
                 f"waits {cfg.memory_cooldown_days} days before buying that symbol again.\n\n"
                 f"Cooling off as of {today}: "
                 + (", ".join(f"{s} (lost ${-p:,.0f}, closed {d})" for s, d, p in cooling) if cooling else "none") + ".\n")
    memory.write_learnings(lessons, extra)
    return lessons


def daily_shares(equity: float, room: float, price: float, atr: float, cfg: Config) -> int:
    """Shares so that a stop-out loses daily_risk_pct of equity, within the position cap and without borrowing."""
    stop_distance = cfg.daily_stop_atr * atr
    if not (stop_distance > 0 and price > 0):
        return 0
    return max(0, math.floor(min(equity * cfg.daily_risk_pct / 100 / stop_distance,
                                 equity * cfg.daily_max_position_pct / 100 / price,
                                 room / price)))


def load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state.setdefault("last_checked", {})    # symbol -> date of the newest bar already checked for a sell signal
    state.setdefault("handled_entry", {})   # symbol -> date of the newest buy signal already acted on
    state.setdefault("open", {})            # symbol -> details of the strategy position, for the memory ledger
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def rebalance_core(cfg: Config, broker: Broker | None, equity: float, core_value: float, free_cash: float,
                   frame: pd.DataFrame | None, latest_key: str, dry_run: bool) -> float:
    """Keep the core holding near core_pct of equity. Returns the dollar change in the holding (0 if no trade).

    The core is simply held: no stop-loss and no trend exit. It is only traded when it has drifted
    more than core_band_pct of equity away from its target, so it trades rarely.
    """
    sym = cfg.core_symbol
    target = equity * cfg.core_pct / 100
    gap = target - core_value
    if abs(gap) <= equity * cfg.core_band_pct / 100:
        log.info("Core %s: holding $%s against a target of $%s - close enough, no trade.",
                 sym, f"{core_value:,.0f}", f"{target:,.0f}")
        return 0.0
    price = (broker.last_price(sym) if broker else None) or (float(frame["Close"].iloc[-1]) if frame is not None else 0.0)
    if price <= 0:
        log.warning("Core %s: no price available - not traded today.", sym)
        return 0.0
    if broker and broker.open_orders(sym):
        log.info("Core %s: an order is already working - not traded today.", sym)
        return 0.0
    buying = gap > 0
    shares = math.floor((min(gap, free_cash) if buying else -gap) / price)
    if shares < 1:
        log.info("Core %s: $%s below target but no cash free to add.", sym, f"{gap:,.0f}")
        return 0.0
    text = (f"{'BUY' if buying else 'SELL'} {shares} {sym} at market (~{price:.2f}, ${shares * price:,.0f}) "
            f"to bring the core holding {'up' if buying else 'down'} to {cfg.core_pct:g}% of the account")
    if dry_run:
        log.info("DRY RUN - would %s", text)
        return shares * price if buying else -shares * price
    try:
        order = broker.submit_market(sym, shares, OrderSide.BUY if buying else OrderSide.SELL, f"{sym}-{latest_key}-core")
    except APIError as err:
        log.error("Core %s: ORDER REJECTED by Alpaca: %s", sym, err)
        notify(f"ORDER REJECTED by Alpaca for the core holding {sym}: {err}")
        return 0.0
    log.info("CORE: %s (order %s, status %s)", text, order.id, order.status)
    notify("ORDER SENT: " + text)
    return shares * price if buying else -shares * price


def run_daily(cfg: Config, broker: Broker | None, dry_run: bool, fallback_equity: float) -> None:
    """One daily pass: wait for the open, sell what has lost its trend, buy fresh dip recoveries, then finish."""
    clock = broker.clock() if broker else None
    if clock is not None and not clock.is_open:
        opens = pd.Timestamp(clock.next_open).tz_convert(NY)
        if opens - pd.Timestamp.now(tz=NY) > STAY_AWAKE_BEFORE_OPEN:
            log.info("Market is closed until %s ET%s.", opens.strftime("%Y-%m-%d %H:%M"),
                     " - showing what the latest bars say" if dry_run else " - nothing to do today")
            if not dry_run:
                return
        elif not dry_run:
            keep_awake(True)
            log.info("Waiting for the open at %s ET.", opens.strftime("%H:%M"))
            sleep_until(opens + DAILY_OPEN_DELAY)

    frames = fetch_daily_bars(cfg)
    missing = [s for s in cfg.daily_symbols if s not in frames]
    if missing:
        log.warning("No daily data for: %s.", ", ".join(missing))
    if not frames:
        raise RuntimeError("no daily data for any symbol")
    latest = max(df.index[-1] for df in frames.values())
    age_days = (pd.Timestamp.now(tz=NY).normalize().tz_localize(None) - latest).days
    log.info("Daily strategy on %d symbols; latest completed bar %s.", len(frames), latest.strftime("%Y-%m-%d"))
    if age_days > MAX_DAILY_DATA_AGE_DAYS:
        raise RuntimeError(f"the newest daily bar is {age_days} days old - refusing to trade on stale data")

    signals, too_short = {}, []
    need = max(cfg.trend_sma, cfg.slow_ema, cfg.bb_period, cfg.atr_period) + 1
    for sym, df in frames.items():
        if len(df) < need:
            too_short.append(f"{sym} ({len(df)} bars)")
        elif df.index[-1] != latest:
            log.warning("%s: newest bar is %s, not %s - skipped today.", sym,
                        df.index[-1].strftime("%Y-%m-%d"), latest.strftime("%Y-%m-%d"))
        else:
            signals[sym] = add_daily_signals(df, cfg)
    if too_short:
        log.info("Not enough history yet (needs %d daily bars): %s.", need, ", ".join(too_short))

    core = cfg.core_symbol if cfg.core_pct > 0 else None
    core_frame = frames.get(core) if core else None
    signals.pop(core, None)                 # the core holding is kept, not traded on signals

    state = load_state()
    latest_key = latest.strftime("%Y-%m-%d")
    positions = broker.positions() if broker else {}

    # 0. Anything held at the end of the last run that is gone now was closed by its stop-loss in between.
    sold, bought = [], []
    if broker and not dry_run:
        stopped = [s for s in state.get("held", []) if s not in positions and s != core]
        if stopped:
            log.info("Closed since the last run (stop-loss hit): %s.", ", ".join(stopped))
            notify("STOP-LOSS HIT since the last run - no longer holding: " + ", ".join(stopped))
            rows = []
            for sym in stopped:
                info = state["open"].pop(sym, None)
                if info:                      # the fill price is not known here, so the loss is taken from the stop price
                    rows.append(memory.row(latest_key, sym, "SELL", info["stop"], info["shares"],
                                           f"stop-loss hit (estimated at the stop price); setup={info['setup']}; "
                                           f"entry={info['entry_date']}", "paper", "loss",
                                           (info["stop"] - info["entry"]) * info["shares"]))
            if rows:
                memory.append_rows(rows)
                refresh_learnings(cfg)
            save_state(state)

    # 1. Sell positions whose trend has ended on any bar since the last check.
    for sym in sorted(signals):
        qty = positions.get(sym, (0.0, 0.0))[0]
        if qty <= 0:
            continue
        df = signals[sym]
        since = state["last_checked"].get(sym)
        new_bars = df[df.index > pd.Timestamp(since)] if since else df.iloc[-1:]
        hits = new_bars[new_bars["exit"]]
        if hits.empty:
            if not dry_run:
                state["last_checked"][sym] = latest_key
            if not any(o.side == OrderSide.SELL for o in broker.open_orders(sym)):
                log.warning("Holding %g %s with NO stop-loss order attached.", qty, sym)
            continue
        bar = hits.iloc[-1]
        why = "close fell below the long average" if bar["Close"] < bar["sma"] else "fast EMA crossed below slow EMA"
        if dry_run:
            log.info("DRY RUN - would SELL %g %s: %s on %s.", qty, sym, why, hits.index[-1].strftime("%Y-%m-%d"))
            continue
        # Memory is read before a sell too, but an exit is never blocked: closing a position only reduces risk.
        info = state["open"].get(sym)
        if cfg.memory_enabled and info:
            verdict = load_memory(cfg).check(sym, info["setup"], latest_key)
            log.info("MEMORY before selling %s: %s. Exits are never blocked.", sym, verdict.reason)
        pnl = getattr(broker, "position_pnl", lambda s: None)(sym)
        broker.cancel_all(sym)
        order = broker.close_position(sym)
        state["last_checked"][sym] = latest_key
        state["open"].pop(sym, None)
        save_state(state)
        log.info("SELL %g %s: %s on %s (order %s).", qty, sym, why, hits.index[-1].strftime("%Y-%m-%d"), order.id)
        notify(f"ORDER SENT: SELL {qty:g} {sym} at market - the trend has ended ({why}).")
        sold.append(sym)
        if info:
            outcome = "" if pnl is None else "win" if pnl > 0 else "loss"
            memory.append_rows([memory.row(latest_key, sym, "SELL", float(bar["Close"]), qty,
                                           f"trend ended ({why}); setup={info['setup']}; entry={info['entry_date']}",
                                           "paper", outcome, pnl)])
            refresh_learnings(cfg)

    # 2. Buy fresh dip recoveries, sized by volatility, never borrowing.
    if broker:
        equity, _ = broker.account()
        positions = broker.positions()
        invested = sum(max(value, 0.0) for _, value in positions.values())
    else:
        equity, invested = fallback_equity, 0.0
    free_cash = equity - invested
    core_value = max(positions.get(core, (0.0, 0.0))[1], 0.0) if core else 0.0
    log.info("Equity $%s, invested $%s in %d position(s).", f"{equity:,.2f}", f"{invested:,.2f}", len(positions))

    # 2a. The core holding takes its fixed share of the account first.
    if core:
        change = rebalance_core(cfg, broker, equity, core_value, free_cash, core_frame, latest_key, dry_run)
        free_cash -= max(change, 0.0)       # cash from trimming the core is not spent the same day

    # 2b. The strategy works with the rest of the account.
    strategy_equity = equity * (1 - cfg.core_pct / 100)
    room = min(strategy_equity - (invested - core_value), free_cash)
    equity = strategy_equity                # risk and position limits below are measured against the strategy's share
    setups = [s for s in sorted(signals) if bool(signals[s]["entry"].iloc[-1])]
    log.info("Strategy share of the account $%s, free to invest $%s. Buy signals today: %s.",
             f"{strategy_equity:,.2f}", f"{max(room, 0):,.2f}", ", ".join(setups) or "none")

    # Today's picks: rank the whole list by recent trend and only buy dips in the strongest part of it.
    ranks = trend_ranks({s: df["Close"] for s, df in signals.items()}, cfg)
    if ranks:
        cut = cfg.rank_top_pct / 100
        ordered = sorted(ranks, key=ranks.get)
        log.info("Trend ranking (%d-day return, %d symbols). Strongest today: %s.",
                 cfg.rank_lookback, len(ordered), ", ".join(ordered[:10]))
        for sym in setups:
            if ranks.get(sym, 2.0) > cut:
                where = f"ranks {ordered.index(sym) + 1} of {len(ordered)}" if sym in ranks else "has too little history to rank"
                log.info("%s: buy signal, but it %s by trend - outside today's top %g%%, not traded.",
                         sym, where, cfg.rank_top_pct)
        setups = sorted((s for s in setups if ranks.get(s, 2.0) <= cut), key=lambda s: (ranks[s], s))
    mem = load_memory(cfg) if cfg.memory_enabled else None

    for sym in setups:
        bar = signals[sym].iloc[-1]
        if positions.get(sym, (0.0, 0.0))[0] > 0:
            log.info("%s: buy signal, but already holding it.", sym)
            continue
        if state["handled_entry"].get(sym) == latest_key:
            log.info("%s: buy signal already acted on.", sym)
            continue
        if broker and broker.open_orders(sym):
            log.info("%s: buy signal, but an order is already working.", sym)
            continue
        price = (broker.last_price(sym) if broker else None) or float(bar["Close"])
        atr = float(bar["atr"])
        shares = daily_shares(equity, room, price, atr, cfg)
        stop = round(price - cfg.daily_stop_atr * atr, 2)
        if shares < 1 or stop <= 0:
            log.info("%s: buy signal, but no room for a position (free to invest: $%s).", sym, f"{room:,.2f}")
            continue
        text = (f"BUY {shares} {sym} at market (~{price:.2f}, ${shares * price:,.0f}) | stop-loss {stop:.2f} | "
                f"ATR {atr:.2f} | risk ${shares * (price - stop):,.0f}")
        # Memory: has this symbol lost on this setup before, does learnings.md warn, is it a known bad pattern?
        setup = bar_setup(bar)
        if mem is not None:
            verdict = mem.check(sym, setup, latest_key)
            if not verdict.allow:
                log.info("MEMORY: skipping %s - %s.", sym, verdict.reason)
                if not dry_run:
                    state["handled_entry"][sym] = latest_key
                    save_state(state)
                    memory.append_rows([memory.row(latest_key, sym, "SKIP", price, shares,
                                                   f"memory warning: {verdict.reason}; setup={setup}", "paper", "skipped")])
                    notify(f"SKIPPED {sym}: memory warns against this setup ({verdict.reason}).")
                continue
            log.info("MEMORY: %s - clear to buy.", verdict.reason)
        if dry_run:
            log.info("DRY RUN - would %s", text)
            room -= shares * price
            continue
        try:
            order = broker.submit_buy_with_stop(sym, shares, stop, f"{sym}-{latest_key}-daily")
        except APIError as err:
            state["handled_entry"][sym] = latest_key
            save_state(state)
            log.error("%s: ORDER REJECTED by Alpaca: %s", sym, err)
            notify(f"ORDER REJECTED by Alpaca for {sym}: {err}")
            continue
        room -= shares * price
        state["handled_entry"][sym] = latest_key
        state["last_checked"][sym] = latest_key
        state["open"][sym] = {"setup": setup, "entry_date": latest_key, "shares": shares, "entry": price, "stop": stop}
        save_state(state)
        log.info("%s (order %s, status %s)", text, order.id, order.status)
        notify("ORDER SENT: " + text)
        bought.append(sym)
        memory.append_rows([memory.row(latest_key, sym, "BUY", price, shares,
                                       f"dip recovery in an uptrend; setup={setup}", "paper", "open")])

    if not dry_run:
        if broker:
            state["held"] = sorted((set(broker.positions()) | set(bought)) - set(sold))
        save_state(state)
    log.info("Daily run finished.")


def simulate_daily(frames: dict[str, pd.DataFrame], cfg: Config, start_equity: float, start_day=None,
                   entry_filter=None):
    """Replay the daily strategy (and the core holding, if set) in one account. Returns (trades, equity curve).

    Each trade is (symbol, entry date, exit date, shares, entry, exit, profit, reason, setup). Orders fill at
    the open after the signal; stops fill at the stop price, or at the open when the price gaps through it.
    entry_filter(symbol, signal date, setup) -> bool lets the memory replay veto a buy; None = raw strategy.
    """
    slip = cfg.slippage_pct / 100
    core = cfg.core_symbol if cfg.core_pct > 0 and cfg.core_symbol in frames else None
    core_idx = dict(zip(frames[core].index, range(len(frames[core])))) if core else {}
    core_open = frames[core]["Open"].to_numpy(float) if core else None
    core_close = frames[core]["Close"].to_numpy(float) if core else None
    core_shares, core_spent, core_mark = 0, 0.0, 0.0      # shares held, net cash put in, last close
    strategy_share = 1 - (cfg.core_pct / 100 if core else 0.0)
    data, setups, dates, rank = {}, {}, {}, {}
    ranking = cfg.rank_top_pct < 100
    if ranking:   # each day's ranking of the whole list by recent return (0 = strongest), known at that day's close
        closes = pd.DataFrame({s: f["Close"] for s, f in frames.items() if s != core})
        rank_table = (closes / closes.shift(cfg.rank_lookback) - 1).rank(axis=1, pct=True, ascending=False)
    for sym, df in frames.items():
        if sym == core:
            continue
        f = add_daily_signals(df, cfg)
        data[sym] = (dict(zip(f.index, range(len(f)))), f["Open"].to_numpy(float), f["Low"].to_numpy(float),
                     f["Close"].to_numpy(float), f["atr"].to_numpy(float),
                     f["entry"].to_numpy(bool), f["exit"].to_numpy(bool))
        dates[sym] = f.index
        setups[sym] = [adaptive_filter.setup_key(c_, s_, a_, b_) for c_, s_, a_, b_ in
                       zip(f["Close"].to_numpy(float), f["sma"].to_numpy(float),
                           f["ema_fast"].to_numpy(float), f["ema_slow"].to_numpy(float))]
        if ranking:
            rank[sym] = rank_table[sym].reindex(f.index).to_numpy(float)
    days = sorted(d for d in set(core_idx).union(*(d[0].keys() for d in data.values()))
                  if start_day is None or d >= start_day)
    realized, held, trades, curve = 0.0, {}, [], []

    def account_value():
        return (start_equity + realized + sum(p["shares"] * (marks[s] - p["entry"]) for s, p in held.items())
                + core_shares * core_mark - core_spent)
    marks = {}                                   # last known close per held symbol

    def close(sym, day, price, reason):
        nonlocal realized
        p = held.pop(sym)
        pnl = p["shares"] * (price - p["entry"])
        realized += pnl
        trades.append((sym, p["day"], day, p["shares"], p["entry"], price, pnl, reason, p["setup"]))

    for day in days:
        equity = account_value()
        # exits at the open: gaps through the stop, and sells decided on yesterday's close
        for sym in sorted(held):
            idx, o, l, c, atr, ent, ext = data[sym]
            i = idx.get(day)
            if i is None:
                continue
            p = held[sym]
            if o[i] <= p["stop"]:
                close(sym, day, o[i] * (1 - slip), "stop")
            elif p["sell_next"]:
                close(sym, day, o[i] * (1 - slip), "trend ended")
        # entries at the open, for buy signals on the previous bar
        strategy_invested = sum(p["shares"] * p["entry"] for p in held.values())
        # the core holding: top up or trim at the open when it has drifted outside its band
        ci = core_idx.get(day)
        if ci is not None:
            core_value = core_shares * core_mark
            gap = equity * cfg.core_pct / 100 - core_value
            if abs(gap) > equity * cfg.core_band_pct / 100:
                if gap > 0:
                    price = core_open[ci] * (1 + slip)
                    n = max(0, math.floor(min(gap, equity - strategy_invested - core_value) / price))
                    core_shares += n
                    core_spent += n * price
                else:
                    price = core_open[ci] * (1 - slip)
                    n = min(core_shares, math.floor(-gap / price))
                    core_shares -= n
                    core_spent -= n * price
            if core_mark == 0.0:
                core_mark = core_open[ci]
        strategy_equity = equity * strategy_share
        room = strategy_equity - strategy_invested
        candidates = []
        for sym in sorted(data):
            idx, o, l, c, atr, ent, ext = data[sym]
            i = idx.get(day)
            if i is None or i == 0 or sym in held or not ent[i - 1] or np.isnan(atr[i - 1]):
                continue
            position = rank[sym][i - 1] if ranking else 0.0
            if ranking and not position <= cfg.rank_top_pct / 100:
                continue                         # outside the day's strongest part of the list (or not rankable yet)
            candidates.append((position, sym, i))
        for position, sym, i in sorted(candidates):   # strongest first when cash is short
            idx, o, l, c, atr, ent, ext = data[sym]
            price = o[i] * (1 + slip)
            shares = daily_shares(strategy_equity, room, price, atr[i - 1], cfg)
            if shares < 1:
                continue
            setup = setups[sym][i - 1]
            if entry_filter is not None and not entry_filter(sym, dates[sym][i - 1], setup):
                continue
            held[sym] = {"day": day, "shares": shares, "entry": price, "sell_next": False, "setup": setup,
                         "stop": price - cfg.daily_stop_atr * atr[i - 1]}
            room -= shares * price
        # during the day: stops; at the close: note sells for tomorrow
        for sym in sorted(held):
            idx, o, l, c, atr, ent, ext = data[sym]
            i = idx.get(day)
            if i is None:
                continue
            p = held[sym]
            if l[i] <= p["stop"]:
                close(sym, day, p["stop"] * (1 - slip), "stop")
            else:
                marks[sym] = c[i]
                p["sell_next"] = p["sell_next"] or bool(ext[i])
        if ci is not None:
            core_mark = core_close[ci]
        curve.append(account_value())
    for sym in sorted(held):
        close(sym, days[-1], marks[sym], "open")
    return trades, pd.Series(curve, index=pd.DatetimeIndex(days))


def replay_frames(cfg: Config, history: str):
    """Real daily bars for a replay, and the first day on which every indicator (and the ranking) is ready."""
    frames = fetch_daily_bars(replace(cfg, daily_history=history))
    need = max(cfg.trend_sma, cfg.slow_ema) + 1
    frames = {s: f for s, f in frames.items() if len(f) >= need}
    if not frames:
        raise RuntimeError("no symbol has enough daily history to replay")
    warm_up = max(need, cfg.rank_lookback + 1) if cfg.rank_top_pct < 100 else need
    longest = max(frames.values(), key=len)
    return frames, longest.index[min(warm_up, len(longest)) - 1]


def backtest_daily(cfg: Config, equity: float, history: str) -> None:
    frames, first_signal = replay_frames(cfg, history)
    trades, curve = simulate_daily(frames, cfg, equity, start_day=first_signal)
    if not trades:
        log.info("No trades.")
        return
    years = len(curve) / 252
    if cfg.core_pct > 0:
        log.info("Core holding: %g%% of the account kept in %s; the strategy trades the other %g%%.",
                 cfg.core_pct, cfg.core_symbol, 100 - cfg.core_pct)
    pnl = [t[6] for t in trades]
    log.info("Daily backtest: %d symbols, %s to %s, starting equity $%s, %.2f%% risk per trade, %g ATR stop.",
             len(frames), curve.index[0].strftime("%Y-%m-%d"), curve.index[-1].strftime("%Y-%m-%d"),
             f"{equity:,.0f}", cfg.daily_risk_pct, cfg.daily_stop_atr)
    by_year = curve.groupby(curve.index.year).last()
    start = equity
    for year, value in by_year.items():
        log.info("  %d: %+6.1f%%  (ending equity $%s)", year, (value / start - 1) * 100, f"{value:,.0f}")
        start = value
    for t in trades[-10:]:
        log.info("  %-5s %s -> %s  %4d sh  in %8.2f  out %8.2f  %-11s %+9.2f", t[0], t[1].strftime("%Y-%m-%d"),
                 t[2].strftime("%Y-%m-%d"), t[3], t[4], t[5], t[7], t[6])
    daily_ret = curve.pct_change().dropna()
    log.info("Account $%s -> $%s", f"{equity:,.0f}", f"{curve.iloc[-1]:,.0f}")
    log.info("Trades %d | win rate %.0f%% | strategy trades P&L $%+.2f | %.1f%% a year | worst drop %.1f%% | "
             "worst trade $%+.2f | Sharpe %.2f (last 10 trades shown above)",
             len(trades), sum(p > 0 for p in pnl) / len(trades) * 100, sum(pnl),
             ((curve.iloc[-1] / equity) ** (1 / years) - 1) * 100, ((curve / curve.cummax()) - 1).min() * 100,
             min(pnl), daily_ret.mean() / daily_ret.std() * math.sqrt(252))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean reversion bot for an Alpaca paper account.")
    parser.add_argument("--history", default="10y",
                        help="how much history the daily backtest uses, e.g. 5y or 10y (default 10y)")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.toml", help="path to the config file")
    parser.add_argument("--once", action="store_true", help="run one check and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="find setups and size them, but never send, cancel or close an order")
    parser.add_argument("--backscan", action="store_true", help="list every setup in the fetched history and exit")
    parser.add_argument("--backtest", action="store_true", help="replay the strategy over the fetched history and exit")
    parser.add_argument("--equity", type=float, default=50_000,
                        help="account equity to assume when no Alpaca credentials are set (default 50000)")
    parser.add_argument("--replay-raw", action="store_true",
                        help="replay the daily strategy without memory, record real outcomes in data/ledger.csv")
    parser.add_argument("--replay-memory", action="store_true",
                        help="replay again with the memory filter and compare it against the raw replay")
    parser.add_argument("--memory-reset", action="store_true", help="empty data/ledger.csv and data/learnings.md")
    parser.add_argument("--test-discord", action="store_true",
                        help="send one test message to the Discord webhook in .env and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(BASE_DIR / "bot.log", encoding="utf-8")],
    )

    if args.test_discord:
        load_dotenv(BASE_DIR / ".env")
        if not os.getenv("DISCORD_WEBHOOK_URL", "").strip():
            log.info("DISCORD_WEBHOOK_URL is empty in .env - paste your webhook address there first.")
        elif notify("Test message from your trading bot: Discord notifications are working."):
            log.info("Test message sent - check your Discord channel.")
        else:
            log.info("The test message was NOT sent - check the webhook address in .env.")
        return

    if args.memory_reset or args.replay_raw or args.replay_memory:
        import replay
        try:
            if args.memory_reset:
                replay.reset()
            cfg = load_config(args.config)
            if cfg.mode != "daily" and (args.replay_raw or args.replay_memory):
                log.error('The memory replays work on the daily strategy: set mode = "daily" in config.toml.')
                sys.exit(1)
            if args.replay_raw:
                replay.replay_raw(cfg, args.equity, args.history)
            if args.replay_memory:
                replay.replay_memory(cfg, args.equity, args.history)
        except Exception:
            log.exception("Replay stopped by an unexpected error.")
            sys.exit(1)
        return

    try:
        cfg = load_config(args.config)
        daily = cfg.mode == "daily"
        if daily and args.backscan:
            args.dry_run = True      # in daily mode --backscan just shows today's signals without trading
        live_loop = not (args.dry_run or args.backscan or args.backtest or (args.once and not daily))
        if live_loop:
            lock = acquire_single_instance_lock()   # held until the process ends
            if lock is None:
                log.error("Another copy of the bot is already running - this one is exiting so they cannot "
                          "both trade the same signal.")
                sys.exit(1)
        broker = connect_broker(required=not (args.dry_run or args.backscan or args.backtest))

        if daily and args.backtest:
            backtest_daily(cfg, broker.account()[0] if broker else args.equity, args.history)
        elif daily:
            run_daily(cfg, broker, args.dry_run, args.equity)
        elif args.backtest:
            backtest(cfg, broker.account()[0] if broker else args.equity)
        elif args.backscan:
            backscan(cfg, broker.account()[0] if broker else args.equity)
        elif args.once:
            run_check(cfg, broker, broker.clock() if broker else None, args.dry_run, args.equity, {})
        else:
            run_loop(cfg, broker, args.dry_run, args.equity)
    except KeyboardInterrupt:
        log.info("Stopped.")
    except Exception:
        # Without this a crash only prints to a console nobody sees when the scheduler runs the bot.
        log.exception("Bot stopped by an unexpected error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
