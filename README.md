# Daily Dip-Then-Trend Paper-Trading Bot

An automated trading bot for an [Alpaca](https://alpaca.markets/) **paper** (practice) account, written in Python. Once a day at the market open it ranks a watchlist of 60 ETFs and large US stocks, buys pullbacks in the strongest half while their long-term trend is up, and holds each position until the trend turns. Half the account sits in a plain S&P 500 holding. A two-file memory records every trade and makes the bot skip setups that have a real losing record.

> **Paper trading only.** This is a learning project. It connects to Alpaca's paper endpoint and has never traded real money. Nothing here is financial advice, and the backtest figures below are not a forecast.

## Contents

- [What it does](#what-it-does)
- [Strategy](#strategy)
- [Backtest results](#backtest-results)
- [What the research found](#what-the-research-found)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Memory system](#memory-system)
- [Project layout](#project-layout)
- [Reliability features](#reliability-features)
- [Limitations](#limitations)

## What it does

| Feature | Detail |
| --- | --- |
| Daily picks | Ranks the watchlist by 12-month return and only trades the top 50% |
| Entry | Buys a dip that has started to recover, only while the price is above its 200-day average |
| Exit | Sells when the 20-day EMA crosses below the 50-day EMA, or the price closes below the 200-day average |
| Stop-loss | A stop order placed with every buy, 3 × ATR(14) below the entry price |
| Position sizing | 1% of the strategy's capital at risk per trade, capped at 20% per position, never borrows |
| Core holding | 50% of the account held in SPY, rebalanced only when it drifts more than 5% off target |
| Memory | Trade ledger plus plain-English lessons; skips buys that repeat a losing setup |
| Alerts | Discord message whenever the bot trades |
| Backtesting | Built-in replay over up to 10 years of daily prices, with and without the memory |

## Strategy

Each trading day, two minutes after the open, the bot runs once and exits.

1. **Rank.** Every symbol is ranked by its return over the past 252 trading days. Only the strongest half is eligible for new buys, strongest first when cash is short.
2. **Buy the dip in an uptrend.** A buy signal needs all three:
   - the close is above the 200-day simple moving average
   - the previous close was below the lower Bollinger Band (20 days, 2 standard deviations)
   - the latest close is back above that band
3. **Size by risk.** Shares are the smallest of: 1% of capital divided by the stop distance, 20% of capital divided by the price, and the cash available.
4. **Protect.** Each buy is sent with a good-till-cancelled stop-loss 3 ATRs below entry.
5. **Ride the trend.** A position is sold when the 20-day EMA crosses below the 50-day EMA, or the close falls below the 200-day average.

The core SPY holding is separate: no stop, no trend exit, simply held.

An earlier 15-minute mean-reversion mode (`mode = "intraday"`) is kept in the code for reference. It is not the live mode because it lost money in testing.

## Backtest results

The bot's own replay of the live configuration, November 2017 to November 2026, starting from $50,000, with 0.05% slippage on every fill:

| Run | Return per year | Worst drop | Final account |
| --- | --- | --- | --- |
| Strategy + 50% SPY, memory off | 12.9% | −27.8% | $148,697 |
| Strategy + 50% SPY, memory on | 13.6% | −26.6% | $156,999 |

How the SPY share trades return against risk (2016 to 2026):

| SPY share of account | 0% | 25% | 50% | 75% | 100% |
| --- | --- | --- | --- | --- | --- |
| Return per year | 9.8% | 11.4% | 12.8% | 14.1% | 15.2% |
| Worst drop | −13% | −17% | −22% | −28% | −34% |

**Read these honestly:**

- The strategy does not beat holding the index on return. What it adds is smaller drops.
- The memory's gain of 0.7 points a year is within the noise of a ten-year test.
- The stock list is the Dow Jones membership as of January 2016, to limit survivorship bias, but the ETF list was chosen with hindsight of which funds exist today.
- Prices come from yfinance daily bars. Real fills, dividends and taxes will differ.

## What the research found

The live strategy is what was left after testing and rejecting many alternatives. Rules were designed on one half of the history and checked on the other.

- **Intraday trading showed no edge.** 15-minute mean-reversion rules lost about 4% to 5% a year on stocks and on ten ETFs since 2018.
- **No rule beat buy-and-hold on return.** Short selling, pairs trading, covered-call funds, calendar effects, credit-spread signals and long/short trend following did not help.
- **Trend rules mainly reduce drawdowns.** A 200-day rule on the SPY half cut the worst drop from about −22% to −15% at a cost of about 2 points a year. It was tested and left out.
- **Common "risk rules" did not raise profit.** Tighter per-trade limits, a correlation filter and a 10% circuit breaker changed risk, not return.

## Getting started

### Requirements

- Python 3.11 or newer
- A free [Alpaca paper trading account](https://app.alpaca.markets/signup)
- Optional: a Discord webhook for trade alerts

### Install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Add your keys

Create a file named `.env` in the project folder. It is listed in `.gitignore` and must never be committed.

```ini
ALPACA_API_KEY=your-paper-key
ALPACA_SECRET_KEY=your-paper-secret

# Optional: Discord alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DISCORD_USER_ID=your-numeric-user-id
```

Use **paper** keys. The bot only connects to the paper endpoint.

## Usage

| Command | What it does |
| --- | --- |
| `python bot.py --dry-run` | Shows what the bot would do today without placing orders |
| `python bot.py` | Waits for the open, trades once, then exits |
| `python bot.py --once` | Runs one check immediately and exits |
| `python bot.py --backtest` | Replays the strategy over history (`--history 10y`, `--equity 50000`) |
| `python bot.py --replay-raw` | Baseline replay that ignores memory; writes real outcomes to the ledger |
| `python bot.py --replay-memory` | Walk-forward replay with memory, compared against the baseline |
| `python bot.py --memory-reset` | Empties the ledger and the lessons file |
| `python bot.py --test-discord` | Sends a test alert |
| `python bot.py --backscan` | Lists every setup in the fetched history (intraday mode) |

Start with `--dry-run`.

### Running it every day (Windows)

Create a Task Scheduler task that runs `.venv\Scripts\python.exe bot.py` in the project folder on weekdays, shortly before the US market opens. The bot waits for the open itself, trades, and exits. A lock file stops two copies running at once.

## Configuration

All settings live in [`config.toml`](config.toml) and apply the next time the bot starts.

| Setting | Default | Meaning |
| --- | --- | --- |
| `strategy.mode` | `"daily"` | `"daily"` or the older `"intraday"` mode |
| `daily.symbols` | 60 symbols | Watchlist: 30 ETFs, 29 large US stocks, SKHY |
| `daily.rank_lookback` | `252` | Days of return used for the ranking |
| `daily.rank_top_pct` | `50.0` | Share of the list eligible to buy; `100` turns ranking off |
| `daily.trend_sma` | `200` | Uptrend filter |
| `daily.fast_ema` / `slow_ema` | `20` / `50` | Trend-exit crossover |
| `daily.stop_atr_multiple` | `3.0` | Stop-loss distance in ATRs |
| `daily.risk_per_trade_pct` | `1.0` | Capital lost if the stop is hit |
| `daily.max_position_pct` | `20.0` | Cap on any one position |
| `daily.core_symbol` / `core_pct` | `"SPY"` / `50.0` | Core holding; `0` turns it off |
| `memory.enabled` | `true` | Use the memory before each buy |
| `memory.cooldown_days` | `20` | Wait after a losing trade in a symbol |
| `memory.min_losses` | `4` | Losses on a symbol + setup before it is skipped |

## Memory system

The bot keeps two files in `data/`:

- **`ledger.csv`** records every buy, sell and skip with its real outcome.
  `timestamp,symbol,action,price,quantity,reason,mode,outcome,pnl`
- **`learnings.md`** holds plain-English lessons derived from the ledger.

Before each buy the bot checks both and applies two rules:

1. **Cool-off.** After a losing trade, no new buy in that symbol for 20 days.
2. **Losing setup.** Skip a symbol + setup with at least 4 losses, more losses than wins, and a net loss. A setup is the trend state (fast EMA above or below slow EMA) plus how far the price sits above its 200-day average (under 3%, 3–10%, over 10%).

Exits are never blocked. A warning lifts automatically when the record improves.

Design rules:

- **Real outcomes only.** No seeded losses, no invented prices, no forced failures.
- **No look-ahead.** In the memory replay, each decision uses only trades that had already closed on that date.
- **Honest baseline.** `--replay-raw` always ignores the memory, so the two runs can be compared.

Example lesson:

> **QQQ: dip recovery while the fast EMA is above the slow EMA and the price is 3-10% above its long average.** This setup lost 6 of 6 times on QQQ ($-806 in total; most recent loss 2026-07-30). Skip it on QQQ until the record improves.

## Project layout

```
bot.py                 Main program: data, signals, sizing, orders, backtests, CLI
adaptive_filter.py     Memory rules: setup keys, cool-off, losing-setup check, lessons
memory.py              Reads and writes data/ledger.csv and data/learnings.md
replay.py              Raw and memory replays and their comparison
config.toml            Live settings
config_original.toml   Settings of the original intraday bot
requirements.txt       Python dependencies
data/                  ledger.csv and learnings.md
```

## Reliability features

- **Idempotent orders.** Each signal gets a unique client order ID, so a restart cannot double-buy.
- **Single instance.** A lock file prevents two copies trading at once.
- **Connection retries** at startup, and crash details written to `bot.log`.
- **Keeps the PC awake** while waiting for the open.
- **Stops live at the broker.** Stop-losses are good-till-cancelled orders, so they work when the bot is not running.

## Limitations

- Results come from backtests on free daily data. The bot has only a short record of live paper trading.
- Returns come mostly from market exposure, not from stock-picking skill.
- SKHY has too little history for the 200-day filter and is skipped until it has 201 daily bars.
- Scheduling and the keep-awake feature are written for Windows.
- Paper fills are more forgiving than real markets.

## Built with

Python · [alpaca-py](https://github.com/alpacahq/alpaca-py) · [yfinance](https://github.com/ranaroussi/yfinance) · pandas · python-dotenv

Developed with the help of an AI coding assistant (Claude Code).
