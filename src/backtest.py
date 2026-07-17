"""
Intraday power trading strategy — momentum on a z-score signal.

A junior colleague wrote this over a weekend and is very excited: it reports a
strong Sharpe and a large cumulative return on 90 days of 15-minute German
intraday prices, so they want to size it up and put it live on Monday.

The strategy idea (their words):
  "Compute how far the current price is from its local average (a z-score).
   When the price is strongly above its recent average, momentum is up, so go
   long; when it's strongly below, go short. Ride the move. I searched for the
   best entry threshold and the numbers are great."

Your job is described in ../CANDIDATE_INSTRUCTIONS.md. The script runs as-is:

    pip install -r ../requirements.txt
    python backtest.py
"""
import numpy as np
import pandas as pd

DATA = "../data/intraday_prices.csv"
WINDOW = 16  # 16 x 15min = 4h local-average window


def load_prices():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    return df


def build_features(df, window=WINDOW):
    # How far is the price from its local average, in standard deviations?
    df["ma"] = df["price"].rolling(window, center=True, min_periods=1).mean()
    df["sd"] = df["price"].rolling(window, center=True, min_periods=1).std()
    df["z"] = (df["price"] - df["ma"]) / df["sd"]
    # Put the signal on a clean unit scale so the threshold is comparable across regimes.
    df["z"] = (df["z"] - df["z"].mean()) / df["z"].std()
    return df


def make_positions(df, threshold):
    # Momentum: long when the price is strongly above its local average, short when below.
    df = df.copy()
    df["position"] = np.where(
        df["z"] > threshold, 1.0,
        np.where(df["z"] < -threshold, -1.0, 0.0),
    )
    return df


def run_backtest(df):
    df = df.copy()
    df["ret"] = df["price"].pct_change()
    df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["strat_ret"] = df["position"] * df["ret"]
    df["equity"] = 1.0 + df["strat_ret"].cumsum()
    return df


def annualized_sharpe(df):
    r = df["strat_ret"].dropna()
    if r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(252)


def optimize_threshold(df):
    # Try a grid of entry thresholds and keep the best one.
    best_th, best_sharpe = None, -np.inf
    for th in np.arange(0.2, 2.6, 0.1):
        bt = run_backtest(make_positions(df, th))
        s = annualized_sharpe(bt)
        if s > best_sharpe:
            best_th, best_sharpe = th, s
    return best_th, best_sharpe


def main():
    df = load_prices()
    df = build_features(df)

    threshold, best_sharpe = optimize_threshold(df)
    bt = run_backtest(make_positions(df, threshold))

    total_return = bt["equity"].iloc[-1] - 1.0
    active = bt.loc[bt["position"] != 0, "strat_ret"]
    hit_rate = (active > 0).mean()
    switches = int((bt["position"].diff().abs() > 0).sum())

    print("=" * 48)
    print(" Intraday momentum strategy — backtest report")
    print("=" * 48)
    print(f" bars                 : {len(bt)}")
    print(f" best threshold       : {threshold:.2f}")
    print(f" annualized Sharpe    : {best_sharpe:.2f}")
    print(f" total return         : {total_return * 100:.1f}%")
    print(f" hit rate             : {hit_rate * 100:.1f}%")
    print(f" position changes     : {switches}")
    print("=" * 48)
    print(" Looks like a strong, high-Sharpe intraday edge.")
    print(" Recommendation: size up and go live.")


if __name__ == "__main__":
    main()
