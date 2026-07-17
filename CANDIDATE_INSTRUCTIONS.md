# Quant Exercise — Audit a "Too Good to Be True" Backtest

**Time budget: ~2.5–3 hours.** This is a *statistical* review-and-repair exercise.
The written review is the main deliverable; the corrected backtest demonstrates you can
act on it. You will not be judged on plotting or polish — you will be judged on statistical
rigour, judgement, and how you connect a number to a trading decision.

## Context

A junior colleague built an intraday trading strategy on **90 days of 15-minute German
power prices** and is excited to put it live on Monday. Their backtest is in `src/`
(`backtest.py`) and **runs as-is**. It reports something like:

```
annualized Sharpe    : 6.14
total return         : 9035.9%
hit rate             : 68.4%
```

Their pitch: *"Compute how far the price is from its recent local average (a z-score).
When it's strongly above, momentum is up — go long; strongly below — go short. I searched
for the best entry threshold and the numbers are great. Let's size up."*

Those numbers are, in our honest assessment, **too good to be true.** Your job is to find
out why, decide whether there is anything real here, and say what you would actually do.

## The data

- `data/intraday_prices.csv` — two columns, `timestamp` (15-minute local time) and
  `price` (EUR/MWh). This is the only input. It is synthetic but built to resemble a real
  intraday power series (diurnal shape, weekly pattern, occasional evening spikes).

## Your tasks

### 1. Written review — `REVIEW.md` (primary deliverable)

Audit the methodology and the code. For **each** issue you find:

- **What** it is and **where** (file + function/line).
- **Root cause** — the actual statistical or economic mechanism, not just the symptom.
- **Impact** — how much of the reported performance does it fabricate or misstate? Where
  you can, **quantify** it (e.g. "fixing this alone takes the Sharpe from X to Y").
- **Severity** — and **order your findings by severity.** We care a lot about
  prioritisation: which of these are fatal to the conclusion, and which are secondary?

Think about: information available at decision time (look-ahead / leakage), train vs.
test discipline, parameter selection and multiple comparisons, transaction costs and
turnover, the correct definition of "return" and "risk" for this instrument, and how
performance is annualised and reported.

### 2. Fix it and report the honest number

Produce a corrected backtest (edit `backtest.py` or write a new `backtest_fixed.py`) that
you would trust. Then answer, in `REVIEW.md`:

- **What is the honest, out-of-sample performance** once the strategy is evaluated
  correctly and net of realistic costs?
- **Is there a real edge here or not?** If not, say so and defend it. If you think there
  is a residual effect, quantify how confident you are and what would change your mind.
- **What would you tell the colleague** who wants to go live on Monday?

Keep the fixed strategy conceptually the same (a z-score signal on this price series) —
we are testing whether you can evaluate an idea honestly, not whether you can invent a
better alpha.

## Running it

```bash
cd src
pip install -r ../requirements.txt   # numpy, pandas
python backtest.py
```

You can regenerate nothing — the CSV is fixed input. Read `backtest.py` closely and ask,
for every line, *"could this number have been produced with information we would not have
had at the time, or that we would not keep after costs?"*

## Deliverables

1. `REVIEW.md` — severity-ranked findings, each quantified where possible, plus your
   honest performance number and your go/no-go recommendation.
2. A corrected backtest script.

## What we value

Depth over breadth. One correctly diagnosed, well-quantified flaw with a clean fix and a
clear economic explanation beats a long list of shallow nitpicks. We are especially
interested in your **statistical judgement** (what makes a backtest trustworthy) and your
ability to turn "the Sharpe is 4.7" into a defensible **trading decision.**
