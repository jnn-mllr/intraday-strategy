"""
Intraday power trading strategy — fully audited and corrected backtest.

This script implements the fixes for all economic, statistical, 
and logical flaws in the original model ranked by severity:

Fatal mistakes:
Step 1: Look-ahead bias #1 (`center=False` on rolling windows, `min_periods=window`)
Step 2: Look-ahead bias #2 (execution time travel -> `position.shift(1)`)
Step 3: In-sample overfitting (strict out-of-sample split: Days 1-60 IS, Days 61-90 OOS), alternatively Walk-Forward Validation

Big mistakes:
Step 4: No transaction costs (turnover costs subtracted on all trades)
Step 5: Selection bias (picking the optimal theta across 24 trials inflates performance)
Step 6: Buy high, sell low (evaluate mean-reversion direction instead)
Step 7: Wrong annualization (15-minute intervals: sqrt(35040))
Step 8: Absolute cash PnL (`diff()` in EUR/MWh instead of `pct_change()` on zero/negative prices)

Smaller mistakes:
Step 9: Data leakage (local rolling z-score without global mean/std re-standardization)
Step 10: Intraday dynamics (Time-of-day interval normalization)
Step 11: Additive vs compound equity math (exact cumulative PnL without percentage distortion)
Step 12: In-place mutation (strictly uses `.copy()` to avoid input data corruption)
"""
import numpy as np
import pandas as pd

DATA = "../data/intraday_prices.csv"

# constant parameters for the analysis
WINDOW = 16  # 16 × 15 min = 4 h local-average window/ 2*center=True
YEARLY_OBS = 4 * 24 * 365
SCALING_ENERGY = np.sqrt(YEARLY_OBS)
SCALING_STOCKS = np.sqrt(252) # convention as in equity markets
COST = 0.2 # EUR/MWh transaction cost
TRAIN_SPLIT = 60 * 96
THRESHOLDS = np.arange(0.2, 2.6, 0.1)


def load_prices():
    return pd.read_csv(DATA, parse_dates=["timestamp"])


def build_features(df, window=WINDOW, normalize_by_time_of_day=True):
    # Step 12: In-place mutation
    df = df.copy()
    
    if normalize_by_time_of_day:
        # Step 10: Intraday dynamics (time-of-day interval normalization across 14 days)
        df["time_group"] = df["timestamp"].dt.time
        df["ma"] = df.groupby("time_group")["price"].transform(lambda x: x.rolling(14, min_periods=7).mean())
        df["sd"] = df.groupby("time_group")["price"].transform(lambda x: x.rolling(14, min_periods=7).std())
    else:
        # Step 1: Look-ahead bias #1 (rolling window)
        df["ma"] = df["price"].rolling(window, center=False, min_periods=window).mean()
        df["sd"] = df["price"].rolling(window, center=False, min_periods=window).std()
    
    # Step 9: Data leakage (local z-score without global standardization)
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    return df


def make_positions(df, threshold):
    # Step 12: In-place mutation protection
    df = df.copy()
    
    # Step 6: Buy high, sell low / selection bias (short when strongly above average, long when below)
    df["position"] = np.where(df["z"] > threshold, -1.0,
                              np.where(df["z"] < -threshold, 1.0, 0.0))
    return df


def run_backtest(df, cost=COST):
    df = df.copy()
    
    # Step 8: Absolute cash PnL (physical returns in EUR/MWh instead of pct_change)
    df["price_diff"] = df["price"].diff().fillna(0.0)
    
    # Step 2: Look-ahead bias #2 (lag position by 1 step to prevent time travel)
    df["gross_pnl"] = df["position"].shift(1) * df["price_diff"]
    
    # Step 4: No transaction costs (subtract realistic turnover friction)
    df["turnover"] = df["position"].diff().abs().fillna(0.0)
    df["net_pnl"] = df["gross_pnl"] - (df["turnover"] * cost)
    
    # Step 11: Additive vs compound equity math (exact cumulative PnL)
    df["cum_net_pnl"] = df["net_pnl"].cumsum()
    return df


def calc_sharpe(returns):
    # Step 7: Wrong annualization (24/7 15-min intervals: sqrt(35040))
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return r.mean() / r.std() * SCALING_ENERGY


def optimize_threshold(df_train):
    best_th, best_sharpe = None, -np.inf
    for th in THRESHOLDS:
        bt = run_backtest(make_positions(df_train, th))
        s = calc_sharpe(bt["net_pnl"])
        if s > best_sharpe:
            best_th, best_sharpe = th, s
    return best_th, best_sharpe


def main():
    df = load_prices()
    df = build_features(df, normalize_by_time_of_day=True)

    # Step 3: In-sample overfitting (enforce 60 days train / 30 days test out-of-sample split)
    train = df.iloc[:TRAIN_SPLIT].copy()
    test = df.iloc[TRAIN_SPLIT:].copy()

    theta, best_is_sharpe = optimize_threshold(train)
    bt = run_backtest(make_positions(test, theta))

    total_pnl = bt["cum_net_pnl"].iloc[-1]
    # Entry fees: include all data points where capital or turnover costs are active
    active = bt.loc[(bt["position"].shift(1) != 0) | (bt["turnover"] > 0), "net_pnl"]
    hit_rate = (active > 0).mean() if len(active) > 0 else 0.0
    switches = int((bt["position"].diff().abs() > 0).sum())
    oos_sharpe = calc_sharpe(bt["net_pnl"])

    print("=" * 48)
    print("Intraday mean-reversion strategy (corrected logic) — backtest report")
    print("=" * 48)
    print(f"bars                      : {len(bt)}")
    print(f"best threshold            : {theta:.2f}")
    print(f"annualized Sharpe         : {oos_sharpe:.2f}")
    print(f"total PnL (EUR/MWh)       : {total_pnl:+.2f}")
    print(f"hit rate                  : {hit_rate * 100:.1f}%")
    print(f"position changes          : {switches}")
    
    if oos_sharpe <= 0.5:
        print("Recommendation: reject go-live and do not deploy the strategy.")
    else:
        print("Recommendation: paper trading before going live.")


if __name__ == "__main__":
    main()
