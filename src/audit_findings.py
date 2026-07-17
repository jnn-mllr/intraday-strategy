# %% [markdown]
# # Backtest Audit — Findings Ranked by Severity
# - Re-evaluates a junior colleague's intraday power trading strategy that originally reported
#   a **6.14 Sharpe** and **9035% total return** across 90 days of 15-minute German power prices.
# - Strips away each methodological error **by severity**, re-evaluating the full backtest after
#   every fix so the incremental impact is quantified.
# - Demonstrates that these extreme performance metrics are entirely artifacts of look-ahead bias,
#   execution time travel, and unrealistic assumptions.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "figure.figsize": (14, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

df_raw = pd.read_csv("../data/intraday_prices.csv", parse_dates=["timestamp"])
print(f"{len(df_raw)} bars, {len(df_raw)/96:.0f} days")
print(f"range: {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")
print(f"price: min={df_raw['price'].min():.2f}, mean={df_raw['price'].mean():.2f}, "
      f"max={df_raw['price'].max():.2f} EUR/MWh")

# %%
# === parameters ===
WINDOW = 16                         # 16 × 15 min = 4 h local-average window
BARS_PER_YEAR = 4 * 24 * 365       # = 35,040 continuous 15-min bars
ANNUALIZE_CORRECT = np.sqrt(BARS_PER_YEAR)   # ~187.2 — correct for 15-min commodity bars
ANNUALIZE_ORIGINAL = np.sqrt(252)             # ~15.9 — wrong: daily equity convention
COST = 0.15                         # EUR/MWh combined spread + exchange fee per unit turnover
TRAIN_BARS = 60 * 96               # first 60 days for threshold optimisation
THRESHOLDS = np.arange(0.2, 2.6, 0.1)


def calc_sharpe(returns, annualize=ANNUALIZE_ORIGINAL):
    """Annualised Sharpe ratio from a return series."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return r.mean() / r.std() * annualize


def count_switches(positions):
    """Count the number of position changes."""
    return int((positions.diff().abs() > 0).sum())


# collectors
results = []
equity_curves = {}

# %% [markdown]
# ---
# ## Step 0 — Baseline: Reproducing the Original Results
#
# - Replicates the colleague's original `backtest.py` code **exactly as written** to establish the
#   starting trajectory.
# - Incorporates all six methodological errors simultaneously: centered rolling window, zero
#   execution lag, global sample normalisation, percentage returns, daily equity annualisation,
#   zero transaction costs, and full-sample threshold optimisation.
# - This is the "too good to be true" number we need to explain away.

# %%
def step0_baseline(df):
    """Exact reproduction of the original backtest.py logic."""
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=True, min_periods=1).mean()
    df["sd"] = df["price"].rolling(WINDOW, center=True, min_periods=1).std()
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    # global normalisation — leaks full-sample statistics into every bar
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()

    best_th, best_s, best_bt = None, -np.inf, None
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # no shift — execution time travel
        bt["strat_ret"] = bt["position"] * bt["ret"]
        bt["equity"] = 1.0 + bt["strat_ret"].cumsum()
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s, best_bt = th, s, bt
    return best_th, best_s, best_bt


th0, s0, bt0 = step0_baseline(df_raw)
ret0 = bt0["equity"].iloc[-1] - 1.0
sw0 = count_switches(bt0["position"])
hit0 = (bt0.loc[bt0["position"] != 0, "strat_ret"] > 0).mean()

results.append({
    "step": "0. baseline", "fix": "none (original code)",
    "sharpe": s0, "metric": f"{ret0*100:.1f}% ret",
    "threshold": th0, "switches": sw0,
})
equity_curves["0. baseline"] = bt0[["timestamp", "equity"]].copy()

print(f"step 0 — threshold={th0:.2f}, sharpe={s0:.2f}, return={ret0*100:.1f}%, "
      f"hit rate={hit0*100:.1f}%, switches={sw0}")

# %% [markdown]
# ---
# ## Audit Findings (Ranked by Severity)
#
# Each finding is applied **cumulatively** on top of all previous fixes. After each correction,
# the full backtest is re-evaluated so you can see exactly how much performance each error
# was fabricating.

# %% [markdown]
# ---
# ### Fatal: Future Data Leakage in Moving Average
#
# - **What & Where**: `build_features`, specifically `rolling(window, center=True)`
# - **Root Cause**: `center=True` shifts the window so the average includes future prices.
#   A signal generated at noon uses prices from the afternoon. The strategy essentially
#   trades on a crystal ball.
# - **Impact**: Massive. This alone fabricates the vast majority of the edge because it
#   perfectly anticipates sudden evening spikes.
# - **Fix**: Set `center=False` and require `min_periods=WINDOW` to avoid unstable early estimates.

# %%
def fix1_causal_window(df):
    """Cumulative fix #1: center=False for strictly historical rolling statistics."""
    df = df.copy()
    # FIXED: center=False, require full window
    df["ma"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()
    df["sd"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()  # still broken

    best_th, best_s, best_bt = None, -np.inf, None
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"] * bt["ret"]  # still no shift
        bt["equity"] = 1.0 + bt["strat_ret"].cumsum()
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s, best_bt = th, s, bt
    return best_th, best_s, best_bt


th1, s1, bt1 = fix1_causal_window(df_raw)
ret1 = bt1["equity"].iloc[-1] - 1.0
sw1 = count_switches(bt1["position"])

results.append({
    "step": "1. causal window", "fix": "center=False",
    "sharpe": s1, "metric": f"{ret1*100:.1f}% ret",
    "threshold": th1, "switches": sw1,
})
equity_curves["1. causal window"] = bt1[["timestamp", "equity"]].copy()

print(f"fix 1 — threshold={th1:.2f}, sharpe={s1:.2f}, return={ret1*100:.1f}%, switches={sw1}")
print(f"        Sharpe: {s0:.2f} → {s1:.2f}  (Δ = {s1 - s0:+.2f})")

# %% [markdown]
# ---
# ### Fatal: Simultaneous Execution Leakage
#
# - **What & Where**: `run_backtest`, specifically multiplying `df["position"]` directly by
#   `df["ret"]` on the same row.
# - **Root Cause**: It assumes you can observe the closing price of a 15-minute bar, generate
#   a signal, and retroactively execute your trade at that exact same closing price.
# - **Impact**: Inflates the hit rate dramatically. To fix, the position must be shifted forward
#   by one period (`shift()`) to execute on the next available price.

# %%
def fix2_execution_lag(df):
    """Cumulative fixes #1+2: causal window + lagged execution."""
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()
    df["sd"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()  # still broken

    best_th, best_s, best_bt = None, -np.inf, None
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # FIXED: trade on the bar after the signal
        bt["strat_ret"] = bt["position"].shift(1) * bt["ret"]
        bt["equity"] = 1.0 + bt["strat_ret"].cumsum()
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s, best_bt = th, s, bt
    return best_th, best_s, best_bt


th2, s2, bt2 = fix2_execution_lag(df_raw)
ret2 = bt2["equity"].iloc[-1] - 1.0
sw2 = count_switches(bt2["position"])

results.append({
    "step": "2. lag execution", "fix": "position.shift(1)",
    "sharpe": s2, "metric": f"{ret2*100:.1f}% ret",
    "threshold": th2, "switches": sw2,
})
equity_curves["2. lag execution"] = bt2[["timestamp", "equity"]].copy()

print(f"fix 2 — threshold={th2:.2f}, sharpe={s2:.2f}, return={ret2*100:.1f}%, switches={sw2}")
print(f"        Sharpe: {s1:.2f} → {s2:.2f}  (Δ = {s2 - s1:+.2f})")

# %% [markdown]
# ---
# ### Fatal: Global Look-Ahead in Normalisation
#
# - **What & Where**: `build_features`, specifically subtracting `df["z"].mean()` and dividing
#   by `df["z"].std()`.
# - **Root Cause**: Using the mean and standard deviation of the entire 90-day dataset. On the
#   first day of trading, the model uses volatility data from day 89 to scale its entry signals.
# - **Impact**: Look-ahead bias that allows the model to perfectly calibrate its thresholds to
#   market regimes that haven't happened yet.

# %%
def fix3_no_global_z(df):
    """Cumulative fixes #1+2+3: causal window + lag + no global normalisation."""
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()
    df["sd"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()
    # FIXED: pure rolling z-score, no global scaling
    df["z"] = (df["price"] - df["ma"]) / df["sd"]

    best_th, best_s, best_bt = None, -np.inf, None
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"].shift(1) * bt["ret"]
        bt["equity"] = 1.0 + bt["strat_ret"].cumsum()
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s, best_bt = th, s, bt
    return best_th, best_s, best_bt


th3, s3, bt3 = fix3_no_global_z(df_raw)
ret3 = bt3["equity"].iloc[-1] - 1.0
sw3 = count_switches(bt3["position"])

results.append({
    "step": "3. no global z", "fix": "remove (z−mean)/std",
    "sharpe": s3, "metric": f"{ret3*100:.1f}% ret",
    "threshold": th3, "switches": sw3,
})
equity_curves["3. no global z"] = bt3[["timestamp", "equity"]].copy()

print(f"fix 3 — threshold={th3:.2f}, sharpe={s3:.2f}, return={ret3*100:.1f}%, switches={sw3}")
print(f"        Sharpe: {s2:.2f} → {s3:.2f}  (Δ = {s3 - s2:+.2f})")
print(f"        all three look-ahead leaks now removed")

# %% [markdown]
# ---
# ### High: In-Sample Overfitting
#
# - **What & Where**: `optimize_threshold` and `main`.
# - **Root Cause**: Sweeping through parameters on the entire dataset, picking the absolute best
#   performer, and reporting that as the final result.
# - **Impact**: Curve-fitting. To fix, you must split the dataset — optimise the threshold on the
#   first block of data, and test it strictly on the unseen later block.
#
# > **Note**: From this step onward, all reported metrics are **out-of-sample** (Days 61–90).
# > The threshold is selected using only the first 60 days and then frozen.

# %%
def fix4_train_test(df):
    """Cumulative fixes #1+2+3+4: all look-ahead fixes + train/test discipline."""
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()
    df["sd"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()
    df["z"] = (df["price"] - df["ma"]) / df["sd"]

    train = df.iloc[:TRAIN_BARS].copy()
    test = df.iloc[TRAIN_BARS:].copy()

    # FIXED: optimise on train only
    best_th, best_is = None, -np.inf
    for th in THRESHOLDS:
        bt = train.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"].shift(1) * bt["ret"]
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_is:
            best_th, best_is = th, s

    # evaluate on test with fixed threshold
    oos = test.copy()
    oos["position"] = np.where(oos["z"] > best_th, 1.0,
                               np.where(oos["z"] < -best_th, -1.0, 0.0))
    oos["ret"] = oos["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    oos["strat_ret"] = oos["position"].shift(1) * oos["ret"]
    oos["equity"] = 1.0 + oos["strat_ret"].cumsum()
    oos_sharpe = calc_sharpe(oos["strat_ret"], annualize=ANNUALIZE_ORIGINAL)

    return best_th, best_is, oos_sharpe, oos


th4, is4, s4, bt4 = fix4_train_test(df_raw)
ret4 = bt4["equity"].iloc[-1] - 1.0
sw4 = count_switches(bt4["position"])

results.append({
    "step": "4. train/test split", "fix": "60/30 day split, OOS eval",
    "sharpe": s4, "metric": f"{ret4*100:.1f}% ret (OOS)",
    "threshold": th4, "switches": sw4,
})
equity_curves["4. train/test"] = bt4[["timestamp", "equity"]].copy()

print(f"fix 4 — train threshold={th4:.2f} (IS sharpe={is4:.2f})")
print(f"        OOS sharpe={s4:.2f}, OOS return={ret4*100:.1f}%, switches={sw4}")
print(f"        Sharpe: {s3:.2f} (IS) → {s4:.2f} (OOS)  — in-sample vs out-of-sample")

# %% [markdown]
# ---
# ### High: Frictionless Trading Environment
#
# - **What & Where**: Missing entirely from `run_backtest`.
# - **Root Cause**: The strategy flips positions thousands of times without paying exchange fees
#   or crossing the bid-ask spread.
# - **Impact**: Destroys realistic PnL. In high-frequency intraday trading, the spread often
#   eats the entire alpha. You must subtract a fixed cost per transaction.
# - **Cost model**: €0.15/MWh combined spread crossing + exchange clearing fee per unit turnover.

# %%
def fix5_with_costs(df):
    """Cumulative fixes #1–5: all look-ahead + train/test + transaction costs."""
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()
    df["sd"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()
    df["z"] = (df["price"] - df["ma"]) / df["sd"]

    train = df.iloc[:TRAIN_BARS].copy()
    test = df.iloc[TRAIN_BARS:].copy()

    # optimise on train (net of costs)
    best_th, best_is = None, -np.inf
    for th in THRESHOLDS:
        bt = train.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"].shift(1) * bt["ret"]
        # FIXED: subtract friction on every position change
        bt["turnover"] = bt["position"].diff().abs().fillna(0.0)
        bt["cost_ret"] = bt["turnover"] * COST / bt["price"].abs().clip(lower=1.0)
        bt["net_ret"] = bt["strat_ret"] - bt["cost_ret"]
        s = calc_sharpe(bt["net_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_is:
            best_th, best_is = th, s

    # evaluate on test
    oos = test.copy()
    oos["position"] = np.where(oos["z"] > best_th, 1.0,
                               np.where(oos["z"] < -best_th, -1.0, 0.0))
    oos["ret"] = oos["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    oos["strat_ret"] = oos["position"].shift(1) * oos["ret"]
    oos["turnover"] = oos["position"].diff().abs().fillna(0.0)
    oos["cost_ret"] = oos["turnover"] * COST / oos["price"].abs().clip(lower=1.0)
    oos["net_ret"] = oos["strat_ret"] - oos["cost_ret"]
    oos["equity"] = 1.0 + oos["net_ret"].cumsum()
    oos_sharpe = calc_sharpe(oos["net_ret"], annualize=ANNUALIZE_ORIGINAL)
    total_cost = (oos["turnover"] * COST).sum()

    return best_th, best_is, oos_sharpe, total_cost, oos


th5, is5, s5, cost5, bt5 = fix5_with_costs(df_raw)
ret5 = bt5["equity"].iloc[-1] - 1.0
sw5 = count_switches(bt5["position"])

results.append({
    "step": "5. add costs", "fix": f"−€{COST}/MWh per turnover",
    "sharpe": s5, "metric": f"{ret5*100:.1f}% ret (OOS, net)",
    "threshold": th5, "switches": sw5,
})
equity_curves["5. add costs"] = bt5[["timestamp", "equity"]].copy()

print(f"fix 5 — threshold={th5:.2f}, OOS net sharpe={s5:.2f}, OOS net return={ret5*100:.1f}%")
print(f"        total OOS cost = €{cost5:.2f}/MWh across {sw5} switches")
print(f"        Sharpe: {s4:.2f} → {s5:.2f}  (Δ = {s5 - s4:+.2f})")

# %% [markdown]
# ---
# ### Medium: Inappropriate Return Mathematics
#
# - **What & Where**: `run_backtest` (percentage returns) and `annualized_sharpe` (scaling factor).
# - **Root Cause**: Power prices can go negative, making percentage returns mathematically
#   invalid or infinite. Furthermore, scaling the Sharpe by √252 assumes daily equity markets,
#   not continuous 15-minute commodity markets.
# - **Impact**: Distorts the reported Sharpe ratio and makes the total return percentage
#   economically meaningless. PnL must be tracked in absolute currency (EUR/MWh) based on
#   position size.
# - **Fix**: Replace `pct_change()` with `diff()` for absolute PnL in EUR/MWh, and annualise
#   with √35,040 instead of √252.

# %%
def fix6_absolute_pnl(df):
    """Cumulative fixes #1–6 (fully corrected): absolute PnL + correct annualisation."""
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()
    df["sd"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()
    df["z"] = (df["price"] - df["ma"]) / df["sd"]

    train = df.iloc[:TRAIN_BARS].copy()
    test = df.iloc[TRAIN_BARS:].copy()

    # optimise on train
    best_th, best_is = None, -np.inf
    for th in THRESHOLDS:
        bt = train.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        # FIXED: absolute price changes instead of percentage returns
        bt["dp"] = bt["price"].diff().fillna(0.0)
        bt["gross_pnl"] = bt["position"].shift(1) * bt["dp"]
        bt["turnover"] = bt["position"].diff().abs().fillna(0.0)
        bt["net_pnl"] = bt["gross_pnl"] - bt["turnover"] * COST
        # FIXED: correct annualisation for continuous 15-min bars
        s = calc_sharpe(bt["net_pnl"], annualize=ANNUALIZE_CORRECT)
        if s > best_is:
            best_th, best_is = th, s

    # evaluate on test
    oos = test.copy()
    oos["position"] = np.where(oos["z"] > best_th, 1.0,
                               np.where(oos["z"] < -best_th, -1.0, 0.0))
    oos["dp"] = oos["price"].diff().fillna(0.0)
    oos["gross_pnl"] = oos["position"].shift(1) * oos["dp"]
    oos["turnover"] = oos["position"].diff().abs().fillna(0.0)
    oos["net_pnl"] = oos["gross_pnl"] - oos["turnover"] * COST
    oos["cum_net_pnl"] = oos["net_pnl"].cumsum()

    oos_sharpe = calc_sharpe(oos["net_pnl"], annualize=ANNUALIZE_CORRECT)
    oos_pnl = oos["cum_net_pnl"].iloc[-1]
    oos_gross = oos["gross_pnl"].cumsum().iloc[-1]
    oos_cost = (oos["turnover"] * COST).sum()

    return best_th, best_is, oos_sharpe, oos_pnl, oos_gross, oos_cost, oos


th6, is6, s6, pnl6, gross6, cost6, bt6 = fix6_absolute_pnl(df_raw)
sw6 = count_switches(bt6["position"])

results.append({
    "step": "6. absolute PnL", "fix": "EUR/MWh diff + √35040",
    "sharpe": s6, "metric": f"{pnl6:.2f} EUR/MWh (OOS, net)",
    "threshold": th6, "switches": sw6,
})
# store PnL curve (not equity) for this step
equity_curves["6. absolute PnL"] = bt6[["timestamp", "cum_net_pnl"]].rename(
    columns={"cum_net_pnl": "equity"}).copy()

print(f"fix 6 — threshold={th6:.2f} (IS sharpe={is6:.2f})")
print(f"        OOS sharpe     = {s6:.2f}")
print(f"        OOS gross PnL  = {gross6:.2f} EUR/MWh")
print(f"        OOS total cost = {cost6:.2f} EUR/MWh")
print(f"        OOS net PnL    = {pnl6:.2f} EUR/MWh")
print(f"        switches       = {sw6}")
print(f"        Sharpe: {s5:.2f} → {s6:.2f}  (now correctly annualised)")

# %% [markdown]
# ---
# ## Summary Table
#
# - Aggregates performance metrics across each step of the correction process.
# - Tracks how Sharpe, cumulative return/PnL, optimal threshold, and position turnover evolve
#   as methodological errors are removed in order of severity.

# %%
df_results = pd.DataFrame(results)
print("\n" + "=" * 90)
print(" CUMULATIVE FIX PROGRESSION (by severity)")
print("=" * 90)
display(df_results) if hasattr(__builtins__, '__IPYTHON__') else print(df_results.to_string(index=False))

# %% [markdown]
# ---
# ## Equity Curve Comparison
#
# - Overlays the cumulative equity/PnL trajectories from each step to visualise the progressive
#   degradation as errors are stripped away.
# - Steps 0–3 show full-sample percentage equity curves (in-sample).
# - Steps 4–6 show out-of-sample curves only.

# %%
colors = ["#1a1a2e", "#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#264653", "#e76f51"]

# --- panel 1: full-sample equity curves (steps 0–3, in-sample) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

for idx, (label, ec) in enumerate(equity_curves.items()):
    if idx <= 3:  # steps 0–3: full-sample with % equity
        axes[0].plot(ec["timestamp"], ec["equity"], linewidth=0.9,
                     color=colors[idx], label=label, alpha=0.85)

axes[0].axhline(1.0, color="black", linewidth=0.5, linestyle="--")
axes[0].set_ylabel("equity (1.0 = start)")
axes[0].set_title("Full-Sample Equity Curves — Removing Look-Ahead Leaks (Steps 0–3)")
axes[0].legend(fontsize=8, loc="upper left")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# --- panel 2: OOS curves (steps 4–6) ---
for idx, (label, ec) in enumerate(equity_curves.items()):
    if idx >= 4:  # steps 4–6: OOS only
        axes[1].plot(ec["timestamp"], ec["equity"], linewidth=1.2,
                     color=colors[idx], label=label, alpha=0.9)

axes[1].axhline(1.0 if 4 in range(4) else 0.0, color="black", linewidth=0.5, linestyle="--")
axes[1].set_ylabel("equity / cum. PnL")
axes[1].set_title("Out-of-Sample Curves — After Train/Test + Costs + Correct Math (Steps 4–6)")
axes[1].legend(fontsize=8, loc="upper left")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Isolated Impact Analysis
#
# - Each fix is applied **independently** from the original baseline (Step 0) to measure
#   its individual contribution to the inflated Sharpe.
# - Answers the key question: "Fixing **only** this single error takes the Sharpe from X to Y."
# - Confirms the severity ranking by showing which individual error fabricates the most performance.

# %%
isolated_results = []

# --- isolated: fix ONLY centered window ---
def iso_causal_only(df):
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()  # fixed
    df["sd"] = df["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()    # fixed
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()  # still broken
    best_th, best_s = None, -np.inf
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"] * bt["ret"]  # still no shift
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s = th, s
    return best_s

# --- isolated: fix ONLY execution lag ---
def iso_lag_only(df):
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=True, min_periods=1).mean()  # still broken
    df["sd"] = df["price"].rolling(WINDOW, center=True, min_periods=1).std()   # still broken
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()  # still broken
    best_th, best_s = None, -np.inf
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"].shift(1) * bt["ret"]  # fixed
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s = th, s
    return best_s

# --- isolated: fix ONLY global z ---
def iso_no_global_z_only(df):
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=True, min_periods=1).mean()  # still broken
    df["sd"] = df["price"].rolling(WINDOW, center=True, min_periods=1).std()   # still broken
    df["z"] = (df["price"] - df["ma"]) / df["sd"]  # fixed: no global scaling
    best_th, best_s = None, -np.inf
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"] * bt["ret"]  # still no shift
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s = th, s
    return best_s

# --- isolated: fix ONLY train/test ---
def iso_train_test_only(df):
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=True, min_periods=1).mean()  # still broken
    df["sd"] = df["price"].rolling(WINDOW, center=True, min_periods=1).std()   # still broken
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()  # still broken
    train = df.iloc[:TRAIN_BARS]
    test = df.iloc[TRAIN_BARS:]
    best_th, best_is = None, -np.inf
    for th in THRESHOLDS:
        bt = train.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"] * bt["ret"]  # still no shift
        s = calc_sharpe(bt["strat_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_is:
            best_th, best_is = th, s
    oos = test.copy()
    oos["position"] = np.where(oos["z"] > best_th, 1.0, np.where(oos["z"] < -best_th, -1.0, 0.0))
    oos["ret"] = oos["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    oos["strat_ret"] = oos["position"] * oos["ret"]
    return calc_sharpe(oos["strat_ret"], annualize=ANNUALIZE_ORIGINAL)

# --- isolated: fix ONLY costs ---
def iso_costs_only(df):
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=True, min_periods=1).mean()  # still broken
    df["sd"] = df["price"].rolling(WINDOW, center=True, min_periods=1).std()   # still broken
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()  # still broken
    best_th, best_s = None, -np.inf
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["ret"] = bt["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bt["strat_ret"] = bt["position"] * bt["ret"]  # still no shift
        bt["turnover"] = bt["position"].diff().abs().fillna(0.0)
        bt["cost_ret"] = bt["turnover"] * COST / bt["price"].abs().clip(lower=1.0)
        bt["net_ret"] = bt["strat_ret"] - bt["cost_ret"]  # fixed: costs applied
        s = calc_sharpe(bt["net_ret"], annualize=ANNUALIZE_ORIGINAL)
        if s > best_s:
            best_th, best_s = th, s
    return best_s

# --- isolated: fix ONLY return math ---
def iso_return_math_only(df):
    df = df.copy()
    df["ma"] = df["price"].rolling(WINDOW, center=True, min_periods=1).mean()  # still broken
    df["sd"] = df["price"].rolling(WINDOW, center=True, min_periods=1).std()   # still broken
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()  # still broken
    best_th, best_s = None, -np.inf
    for th in THRESHOLDS:
        bt = df.copy()
        bt["position"] = np.where(bt["z"] > th, 1.0, np.where(bt["z"] < -th, -1.0, 0.0))
        bt["dp"] = bt["price"].diff().fillna(0.0)  # fixed: absolute PnL
        bt["pnl"] = bt["position"] * bt["dp"]  # still no shift
        s = calc_sharpe(bt["pnl"], annualize=ANNUALIZE_CORRECT)  # fixed: correct annualisation
        if s > best_s:
            best_th, best_s = th, s
    return best_s


# run all isolated fixes
iso_s_causal = iso_causal_only(df_raw)
iso_s_lag = iso_lag_only(df_raw)
iso_s_global = iso_no_global_z_only(df_raw)
iso_s_traintest = iso_train_test_only(df_raw)
iso_s_costs = iso_costs_only(df_raw)
iso_s_retmath = iso_return_math_only(df_raw)

isolated_results = [
    {"finding": "baseline (all errors)", "sharpe": s0, "drop": 0.0},
    {"finding": "fix ONLY: centered window", "sharpe": iso_s_causal, "drop": iso_s_causal - s0},
    {"finding": "fix ONLY: execution lag", "sharpe": iso_s_lag, "drop": iso_s_lag - s0},
    {"finding": "fix ONLY: global z", "sharpe": iso_s_global, "drop": iso_s_global - s0},
    {"finding": "fix ONLY: train/test", "sharpe": iso_s_traintest, "drop": iso_s_traintest - s0},
    {"finding": "fix ONLY: costs", "sharpe": iso_s_costs, "drop": iso_s_costs - s0},
    {"finding": "fix ONLY: return math", "sharpe": iso_s_retmath, "drop": iso_s_retmath - s0},
]

df_isolated = pd.DataFrame(isolated_results)
print("\n" + "=" * 70)
print(" ISOLATED IMPACT (each fix applied independently from baseline)")
print("=" * 70)
display(df_isolated) if hasattr(__builtins__, '__IPYTHON__') else print(df_isolated.to_string(index=False))

# %% [markdown]
# ### Impact Waterfall
#
# - Bar chart showing the Sharpe reduction attributable to each individual error when fixed in isolation.
# - Confirms the severity ranking: the largest individual drops correspond to the most severe findings.

# %%
findings = [r["finding"].replace("fix ONLY: ", "") for r in isolated_results[1:]]
drops = [r["drop"] for r in isolated_results[1:]]

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ["#e63946" if d < -1.0 else "#f4a261" if d < -0.3 else "#2a9d8f" for d in drops]
bars = ax.barh(findings, drops, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.7)
ax.set_xlabel("Sharpe change from baseline (negative = error was inflating)")
ax.set_title(f"Isolated Impact: Fixing Each Error Alone (baseline Sharpe = {s0:.2f})")

for bar, val in zip(bars, drops):
    x_pos = val - 0.15 if val < 0 else val + 0.05
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", fontsize=9, fontweight="bold")

ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## OOS Threshold Sensitivity
#
# - Evaluates the fully corrected strategy across **all** candidate thresholds on the OOS period.
# - Demonstrates whether any threshold choice rescues the strategy, or whether the edge is
#   uniformly absent out of sample.

# %%
oos_sharpes = []
oos_pnls = []

df_full = df_raw.copy()
df_full["ma"] = df_full["price"].rolling(WINDOW, center=False, min_periods=WINDOW).mean()
df_full["sd"] = df_full["price"].rolling(WINDOW, center=False, min_periods=WINDOW).std()
df_full["z"] = (df_full["price"] - df_full["ma"]) / df_full["sd"]
test_data = df_full.iloc[TRAIN_BARS:].copy()

for th in THRESHOLDS:
    oos = test_data.copy()
    oos["position"] = np.where(oos["z"] > th, 1.0, np.where(oos["z"] < -th, -1.0, 0.0))
    oos["dp"] = oos["price"].diff().fillna(0.0)
    oos["gross_pnl"] = oos["position"].shift(1) * oos["dp"]
    oos["turnover"] = oos["position"].diff().abs().fillna(0.0)
    oos["net_pnl"] = oos["gross_pnl"] - oos["turnover"] * COST
    s = calc_sharpe(oos["net_pnl"], annualize=ANNUALIZE_CORRECT)
    p = oos["net_pnl"].cumsum().iloc[-1]
    oos_sharpes.append(s)
    oos_pnls.append(p)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

ax1.bar(THRESHOLDS, oos_sharpes, width=0.08, color="#457b9d", alpha=0.8, edgecolor="white")
ax1.axhline(0, color="black", linewidth=0.7)
ax1.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", label="Sharpe = 0.5 (marginal)")
ax1.axvline(th6, color="#e63946", linewidth=1.5, linestyle="--", label=f"selected ({th6:.1f})")
ax1.set_xlabel("z-score threshold")
ax1.set_ylabel("annualised Sharpe (OOS, net)")
ax1.set_title("OOS Sharpe vs. Threshold (fully corrected)")
ax1.legend(fontsize=8)

ax2.bar(THRESHOLDS, oos_pnls, width=0.08, color="#2a9d8f", alpha=0.8, edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.7)
ax2.axvline(th6, color="#e63946", linewidth=1.5, linestyle="--", label=f"selected ({th6:.1f})")
ax2.set_xlabel("z-score threshold")
ax2.set_ylabel("cumulative net PnL (EUR/MWh)")
ax2.set_title("OOS Net PnL vs. Threshold (fully corrected)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()

print(f"best OOS threshold = {THRESHOLDS[np.argmax(oos_sharpes)]:.1f} "
      f"(Sharpe = {max(oos_sharpes):.2f})")
print(f"worst OOS threshold = {THRESHOLDS[np.argmin(oos_sharpes)]:.1f} "
      f"(Sharpe = {min(oos_sharpes):.2f})")

# %% [markdown]
# ---
# ## Rolling Sharpe (OOS Stability)
#
# - Computes a rolling-window annualised Sharpe ratio over the OOS period to check whether the
#   signal has any persistent predictive power or is entirely noise.
# - A legitimate edge should show sustained positive Sharpe; random noise oscillates around zero.

# %%
# use the fully corrected OOS PnL from fix6
rolling_window = 96 * 5  # 5-day rolling window
rolling_mean = bt6["net_pnl"].rolling(rolling_window, min_periods=rolling_window).mean()
rolling_std = bt6["net_pnl"].rolling(rolling_window, min_periods=rolling_window).std()
rolling_sharpe = (rolling_mean / rolling_std) * ANNUALIZE_CORRECT

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(bt6["timestamp"], rolling_sharpe, linewidth=1.2, color="#457b9d")
ax.axhline(0, color="black", linewidth=0.7)
ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5, label="Sharpe = 1.0")
ax.axhline(-1.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
ax.fill_between(bt6["timestamp"], rolling_sharpe, 0,
                where=rolling_sharpe > 0, color="#2a9d8f", alpha=0.2)
ax.fill_between(bt6["timestamp"], rolling_sharpe, 0,
                where=rolling_sharpe < 0, color="#e63946", alpha=0.2)
ax.set_ylabel("annualised Sharpe (5-day rolling)")
ax.set_title("Rolling Sharpe Ratio — Out-of-Sample Period (fully corrected)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout()
plt.show()

pct_positive = (rolling_sharpe.dropna() > 0).mean() * 100
print(f"rolling Sharpe > 0 for {pct_positive:.0f}% of the OOS window")

# %% [markdown]
# ---
# ## Maximum Drawdown (OOS)
#
# - Measures the worst peak-to-trough decline in cumulative PnL during the OOS period.
# - A large drawdown relative to total PnL confirms the absence of a stable edge.

# %%
cum_pnl = bt6["cum_net_pnl"]
running_max = cum_pnl.cummax()
drawdown = cum_pnl - running_max
max_dd = drawdown.min()
max_dd_idx = drawdown.idxmin()
peak_idx = cum_pnl.loc[:max_dd_idx].idxmax()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax1.plot(bt6["timestamp"], cum_pnl, linewidth=1.0, color="#1a1a2e", label="cumulative net PnL")
ax1.plot(bt6["timestamp"], running_max, linewidth=0.8, color="grey", linestyle="--",
         alpha=0.6, label="running peak")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_ylabel("EUR/MWh")
ax1.set_title("Cumulative Net PnL — Out-of-Sample (fully corrected)")
ax1.legend(fontsize=8)

ax2.fill_between(bt6["timestamp"], drawdown, 0, color="#e63946", alpha=0.4)
ax2.plot(bt6["timestamp"], drawdown, linewidth=0.8, color="#e63946")
ax2.axhline(max_dd, color="black", linewidth=0.7, linestyle="--",
            label=f"max drawdown = {max_dd:.2f} EUR/MWh")
ax2.set_ylabel("drawdown (EUR/MWh)")
ax2.set_title("Drawdown from Peak")
ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.tight_layout()
plt.show()

print(f"max drawdown = {max_dd:.2f} EUR/MWh")
print(f"final net PnL = {pnl6:.2f} EUR/MWh")
if pnl6 != 0:
    print(f"PnL-to-drawdown ratio = {pnl6 / abs(max_dd):.2f}")

# %% [markdown]
# ---
# ## Final Verdict
#
# **The reported Sharpe of 6.14 and 9035% total return are entirely fabricated by statistical
# errors that leak future information into the signal.**
#
# Once corrected and evaluated honestly out-of-sample:
# - The Sharpe drops to near zero or negative.
# - Cumulative PnL is negligible or negative net of realistic trading costs.
# - No threshold choice rescues the strategy.
# - The rolling Sharpe oscillates around zero with no persistent directional bias.
#
# **There is no viable trading edge. Do not deploy to live trading.**
#
# The deeper issue is that a backward-looking rolling z-score on a single price series does not
# contain enough predictive information at 15-minute frequency to overcome the cost of crossing
# the spread ~40 times a day. Any real intraday edge in power markets likely requires exogenous
# predictors (renewable forecast revisions, cross-border flows, orderbook imbalances) and minimum
# holding periods to control turnover.

# %%
print("=" * 70)
print(" FINAL HONEST NUMBERS (fully corrected, out-of-sample)")
print("=" * 70)
print(f" original reported Sharpe  : {s0:.2f}")
print(f" honest OOS Sharpe         : {s6:.2f}")
print(f" honest OOS net PnL        : {pnl6:.2f} EUR/MWh")
print(f" max drawdown              : {max_dd:.2f} EUR/MWh")
print(f" OOS position switches     : {sw6}")
print(f" OOS trading costs         : {cost6:.2f} EUR/MWh")
print("=" * 70)
if s6 < 0.5:
    print(" Verdict: NO EDGE. Do not go live.")
else:
    print(" Verdict: marginal edge, requires further validation.")
