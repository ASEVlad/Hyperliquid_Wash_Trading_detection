import json
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_handler import CoinDataStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = Path(os.path.join(BASE_DIR, "..", "..", "analysis"))


# ---- GENERAL ANALYSIS ----

# ---- TEXT analysis ----


def global_snapshot(df: pd.DataFrame, out_dir=None, fname="global_snapshot.json") -> dict:
    out = {}
    out["pairs"] = len(df)
    out["unique_wallets"] = df["wallet_id"].nunique()
    out["span"] = (str(df["t1"].min()), str(df["t2"].max()))
    out["direction_counts"] = df["direction"].value_counts().to_dict()

    # time gap distribution
    q_dt = df["dt_s"].quantile([0.25,0.5,0.75,0.9,0.99]).round(2).to_dict()
    out["delta_seconds_quantiles"] = q_dt
    out["<=60s_share"] = float((df["dt_s"] <= 60).mean())
    out["<=300s_share"] = float((df["dt_s"] <= 300).mean())

    # size matching tightness
    out["size_err_pct_quantiles"] = df["size_err_pct"].quantile([0.5,0.9,0.99]).round(3).to_dict()
    out["same_price_share"] = float(df["same_price"].mean())

    # concentration
    top10 = df["wallet_id"].value_counts().nlargest(10)
    out["top10_share"] = float(top10.sum() / len(df)) if len(df) else 0.0

    # save to disk if requested
    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / fname, "w") as f:
            json.dump(out, f, indent=2)

    return out


# ---- PLOT analysis ----


def plot_pairs_by_hour(df, out_dir=None, show=False):
    counts = df.groupby("hour").size().reindex(range(24), fill_value=0)
    plt.figure(figsize=(8,4))
    counts.plot(kind="bar")
    plt.title("Wash pairs by hour (UTC)")
    plt.xlabel("Hour of day")
    plt.ylabel("Pairs")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / "pairs_by_hour.png", dpi=150)
    if show: plt.show()
    plt.close()

def plot_pairs_by_dow(df, out_dir=None, show=False):
    counts = df.groupby("dow").size().reindex(range(7), fill_value=0)
    plt.figure(figsize=(7,4))
    counts.plot(kind="bar")
    plt.title("Wash pairs by day of week (0=Mon)")
    plt.xlabel("Day of week")
    plt.ylabel("Pairs")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / "pairs_by_dow.png", dpi=150)
    if show: plt.show()
    plt.close()

def plot_delta_seconds_hist(df, out_dir=None, show=False):
    x = df["dt_s"].dropna().to_numpy()
    plt.figure(figsize=(8,4))
    # wide bins but capped for readability
    bins = min(200, max(20, int(np.sqrt(max(len(x),1)))))
    plt.hist(x, bins=bins)
    plt.title("Round-trip time (seconds)")
    plt.xlabel("Δt (seconds)")
    plt.ylabel("Count")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / "delta_seconds_hist.png", dpi=150)
    if show: plt.show()
    plt.close()

def plot_size_error_hist(df, out_dir=None, show=False):
    x = df["size_err_pct"].dropna().to_numpy()
    plt.figure(figsize=(8,4))
    bins = min(200, max(20, int(np.sqrt(max(len(x),1)))))
    plt.hist(x, bins=bins)
    plt.title("Size matching error (%)")
    plt.xlabel("|size2/size1 - 1| × 100%")
    plt.ylabel("Count")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / "size_error_pct_hist.png", dpi=150)
    if show: plt.show()
    plt.close()

def plot_price_change_bps_hist(df, out_dir=None, show=False):
    x = df["price_change_bps"].dropna().to_numpy()
    plt.figure(figsize=(8,4))
    bins = min(200, max(20, int(np.sqrt(max(len(x),1)))))
    plt.hist(x, bins=bins)
    plt.title("Price change per pair (bps)")
    plt.xlabel("Δprice / price1 × 10,000")
    plt.ylabel("Count")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / "price_change_bps_hist.png", dpi=150)
    if show: plt.show()
    plt.close()

def plot_size_vs_dt_scatter(df, sample=200_000, out_dir=None, show=False):
    # For very large sets, downsample for plotting speed
    d = df[["dt_s","size1","size2"]].dropna()
    if len(d) > sample:
        d = d.sample(sample, random_state=42)
    # use min(size1, size2) as conservative "wash size"
    s = np.minimum(d["size1"].to_numpy(), d["size2"].to_numpy())
    x = d["dt_s"].to_numpy()
    plt.figure(figsize=(7.5,5))
    plt.scatter(x, s, s=None, alpha=0.3, edgecolors="none")
    plt.title("Wash size vs. round-trip time")
    plt.xlabel("Δt (seconds)")
    plt.ylabel("min(size1, size2)")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / "size_vs_dt_scatter.png", dpi=150)
    if show: plt.show()
    plt.close()

def plot_cumulative_pairs(df, out_dir=None, show=False):
    if "t1" not in df: return
    ts = (df.sort_values("t1")
            .assign(n=1)
            .set_index("t1")["n"]
            .resample("1h").sum().fillna(0).cumsum())
    plt.figure(figsize=(8,4))
    plt.plot(ts.index, ts.values)
    plt.title("Cumulative wash pairs over time (hourly res.)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative pairs")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / "cumulative_pairs.png", dpi=150)
    if show: plt.show()
    plt.close()

def plot_top_wallets(df, metric="n_pairs", top=20, out_dir=None, show=False):
    # metric: "n_pairs" or "wash_volume"
    wash_vol = np.minimum(df["size1"], df["size2"]).astype("float32")
    if metric == "wash_volume":
        s = df.assign(wash_vol=wash_vol).groupby("wallet_id")["wash_vol"].sum()
        title = f"Top {top} wallets by wash volume"
        ylabel = "Sum min(size1,size2)"
        fname = f"top_wallets_by_volume_top{top}.png"
    else:
        s = df.groupby("wallet_id").size()
        title = f"Top {top} wallets by number of pairs"
        ylabel = "Pairs"
        fname = f"top_wallets_by_pairs_top{top}.png"

    s = s.sort_values(ascending=False).head(top)
    plt.figure(figsize=(10,5))
    s.plot(kind="bar")
    plt.title(title)
    plt.xlabel("wallet_id")
    plt.ylabel(ylabel)
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / fname, dpi=150)
    if show: plt.show()
    plt.close()

def plot_same_price_share_top_wallets(df, top=20, out_dir=None, show=False):
    s = (df.groupby("wallet_id")["same_price"].mean()
           .sort_values(ascending=False)
           .head(top))
    plt.figure(figsize=(10,5))
    s.plot(kind="bar")
    plt.title(f"Top {top} wallets by same-price share")
    plt.xlabel("wallet_id")
    plt.ylabel("Share of pairs with identical price")
    plt.tight_layout()
    if out_dir: plt.savefig(Path(out_dir) / f"top_wallets_same_price_share_top{top}.png", dpi=150)
    if show: plt.show()
    plt.close()

# --------- one-shot driver ----------
def make_dfwash_plots(df, out_dir="plots_dfwash", show=False, top_wallets=20):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # guard: nothing to plot
    if df.empty:
        return

    plot_pairs_by_hour(df, out_dir, show)
    plot_pairs_by_dow(df, out_dir, show)

    plot_delta_seconds_hist(df, out_dir, show)
    plot_size_error_hist(df, out_dir, show)
    plot_price_change_bps_hist(df, out_dir, show)
    plot_size_vs_dt_scatter(df, out_dir=out_dir, show=show)
    plot_cumulative_pairs(df, out_dir, show)
    plot_top_wallets(df, metric="n_pairs", top=top_wallets, out_dir=out_dir, show=show)
    plot_top_wallets(df, metric="wash_volume", top=top_wallets, out_dir=out_dir, show=show)
    plot_same_price_share_top_wallets(df, top=top_wallets, out_dir=out_dir, show=show)



# PER WALLET analysis


# ---- TEXT analysis

def per_wallet_leaderboard(
    df: pd.DataFrame,
    top: int = 50,
    out_dir: str | Path | None = None,
    fname_csv: str = "per_wallet_leaderboard.csv",
    fname_parquet: str | None = None
) -> pd.DataFrame:
    """
    Build per-wallet leaderboard and (optionally) save it to disk.

    Parameters
    ----------
    df : DataFrame
        dfwash-style DataFrame with columns:
        wallet_id, size1, size2, t1, t2, date, dt_s, size_err_pct,
        same_price, abs_price_change_bps, direction
    top : int
        Number of rows to return (sorted by wash_volume then n_pairs).
    out_dir : str | Path | None
        If provided, saves CSV (and optional Parquet) to this directory.
    fname_csv : str
        Output CSV filename (used when out_dir is not None).
    fname_parquet : str | None
        If provided and out_dir is not None, also saves a Parquet file.

    Returns
    -------
    DataFrame
        Leaderboard (top N) with aggregated metrics per wallet.
    """
    # conservative wash volume: min(size1, size2)
    wash_vol = np.minimum(df["size1"], df["size2"]).astype("float32")
    grp = df.assign(wash_vol=wash_vol).groupby("wallet_id", sort=False)

    agg = grp.agg(
        n_pairs=("wallet_id", "size"),
        first_time=("t1", "min"),
        last_time=("t2", "max"),
        active_days=("date", lambda s: s.nunique()),
        wash_volume=("wash_vol", "sum"),
        median_dt_s=("dt_s", "median"),
        q90_dt_s=("dt_s", lambda s: s.quantile(0.9)),
        median_size_err_pct=("size_err_pct", "median"),
        same_price_share=("same_price", "mean"),
        mean_price_bps=("price_change_pct", "mean"),
    ).reset_index()

    # Direction balance: |buy->sell - sell->buy| / (buy->sell + sell->buy)
    # Make it robust to case/categorical:
    dir_series = df["direction"].astype("string").str.lower()
    dir_counts = pd.pivot_table(
        df.assign(direction=dir_series),
        index="wallet_id",
        columns="direction",
        values="t1",
        aggfunc="size",
        fill_value=0,
    )

    # Ensure both columns exist even if absent
    for col in ("buy->sell", "sell->buy"):
        if col not in dir_counts.columns:
            dir_counts[col] = 0

    denom = (dir_counts["buy->sell"] + dir_counts["sell->buy"]).clip(lower=1)
    dir_counts["dir_balance"] = (dir_counts["buy->sell"] - dir_counts["sell->buy"]).abs() / denom

    out = agg.merge(dir_counts[["dir_balance"]].reset_index(), on="wallet_id", how="left")

    # Sort & keep top
    out = out.sort_values(["wash_volume", "n_pairs"], ascending=[False, False]).head(top)

    # Light dtypes
    out["n_pairs"] = out["n_pairs"].astype("uint32")
    out["wash_volume"] = out["wash_volume"].astype("float32")
    out["median_dt_s"] = out["median_dt_s"].astype("float32")
    out["q90_dt_s"] = out["q90_dt_s"].astype("float32")
    out["median_size_err_pct"] = out["median_size_err_pct"].astype("float32")
    out["same_price_share"] = out["same_price_share"].astype("float32")
    out["mean_price_bps"] = out["mean_price_bps"].astype("float32")

    # Optional: save
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # CSV
        out.to_csv(out_path / fname_csv, index=False)
        # Parquet (optional)
        if fname_parquet:
            out.to_parquet(out_path / fname_parquet, index=False)

    return out


# ---- PLOT Analysis ----


def _ensure_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if not np.issubdtype(d["t1"].dtype, np.datetime64):
        d["t1"] = pd.to_datetime(d["t1"], errors="coerce")
    d["wash_vol"] = np.minimum(d["size1"].astype("float64"), d["size2"].astype("float64"))
    d["mid_price"] = ((d["price1"].astype("float64") + d["price2"].astype("float64")) / 2.0)
    return d

def _bar_from_series(agg: pd.Series, title: str, ylabel: str,
                     out_dir=None, filename="plot.png", show=False, max_xticks=20):
    """
    Render a bar chart from a time-indexed Series agg.
    Uses categorical positions with thinned tick labels for readability.
    """
    x = np.arange(len(agg))
    y = agg.values

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)

    # Thin xticks to at most max_xticks
    if len(x) > 0:
        step = max(1, len(x) // max_xticks)
        ax.set_xticks(x[::step])
        # format dates nicely if datetime index, else just str()
        if isinstance(agg.index, pd.DatetimeIndex):
            labels = [ts.strftime("%Y-%m-%d") for ts in agg.index[::step]]
        else:
            labels = [str(v) for v in agg.index[::step]]
        ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / filename, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_wash_volume_by_date_bar(df: pd.DataFrame, freq: str = "D", out_dir=None, show=False):
    """
    Bar chart: sum of wash volume (min(size1,size2)) per date (or 'W'/'M').
    Also writes a CSV next to the PNG if out_dir is provided.
    """
    d = _ensure_derived_cols(df)
    s = pd.Series(d["wash_vol"].values, index=d["t1"])
    agg = s.resample(freq).sum().rename("wash_volume")

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        agg.to_csv(Path(out_dir) / f"wash_volume_by_date_{freq}.csv")

    _bar_from_series(
        agg,
        title=f"Wash volume by date (freq={freq})",
        ylabel="Sum of min(size1, size2)",
        out_dir=out_dir,
        filename=f"wash_volume_by_date_{freq}.png",
        show=show,
    )

def plot_wash_notional_by_date_bar(df: pd.DataFrame, freq: str = "D", out_dir=None, show=False):
    """
    Bar chart: sum of wash notional per date (min(size1,size2) × mid price).
    Also writes a CSV next to the PNG if out_dir is provided.
    """
    d = _ensure_derived_cols(df)
    notional = (d["wash_vol"] * d["mid_price"]).rename("wash_notional")
    s = pd.Series(notional.values, index=d["t1"])
    agg = s.resample(freq).sum()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        agg.to_csv(Path(out_dir) / f"wash_notional_by_date_{freq}.csv")

    _bar_from_series(
        agg,
        title=f"Wash notional (volume × price) by date (freq={freq})",
        ylabel="Sum of min(size) × mid_price",
        out_dir=out_dir,
        filename=f"wash_notional_by_date_{freq}.png",
        show=show,
    )



def wash_trade_summary(df_trades: pd.DataFrame, df_wash: pd.DataFrame, out_dir=None):
    """
    Compare df_trades with df_wash and return summary stats.

    Returns a DataFrame with:
      - total trades count
      - total volume (price * size)
      - percentage of trades/volume
      - mean price and size
    """
    # Compute total volume for all trades
    df_trades['trade_volume'] = df_trades['price'] * df_trades['size']

    # Flatten df_wash into individual trades
    wash_buyers = df_wash[['t1', 'price1', 'size1', 'side1', 'pair_id']].rename(
        columns={'t1':'time', 'price1':'price', 'size1':'size', 'side1':'side'}
    )
    wash_sellers = df_wash[['t2', 'price2', 'size2', 'side2', 'pair_id']].rename(
        columns={'t2':'time', 'price2':'price', 'size2':'size', 'side2':'side'}
    )

    df_wash_trades = pd.concat([wash_buyers, wash_sellers], ignore_index=True)
    df_wash_trades['trade_volume'] = df_wash_trades['price'] * df_wash_trades['size']

    # Total stats
    total_trades = len(df_trades)
    total_volume = df_trades['trade_volume'].sum()

    # Wash trades stats
    wash_trades_count = len(df_wash_trades)
    wash_volume = df_wash_trades['trade_volume'].sum()

    # Non-wash trades stats
    non_wash_count = total_trades - wash_trades_count
    non_wash_volume = total_volume - wash_volume

    # Average price and size
    avg_price_wash = df_wash_trades['price'].mean()
    avg_size_wash = df_wash_trades['size'].mean()
    avg_price_non = df_trades[~df_trades.index.isin(df_wash_trades.index)]['price'].mean()
    avg_size_non = df_trades[~df_trades.index.isin(df_wash_trades.index)]['size'].mean()

    # Build summary DataFrame
    summary = pd.DataFrame({
        'Category': ['Wash Trades', 'Non-Wash Trades', 'Total'],
        'Count': [wash_trades_count, non_wash_count, total_trades],
        'Count %': [wash_trades_count/total_trades*100,
                    non_wash_count/total_trades*100, 100],
        'Volume': [wash_volume, non_wash_volume, total_volume],
        'Volume %': [wash_volume/total_volume*100,
                     non_wash_volume/total_volume*100, 100],
        'Avg Price': [avg_price_wash, avg_price_non, df_trades['price'].mean()],
        'Avg Size': [avg_size_wash, avg_size_non, df_trades['size'].mean()]
    })

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        summary.to_csv(Path(out_dir) / f"wash_trade_summary.csv")

    return summary



def wash_trading_pairs_analysis(df_wash: pd.DataFrame, token: str, detector_type: str):
    out_dir_path = os.path.join(ANALYSIS_DIR, token, detector_type)
    df_wash.to_csv(os.path.join(out_dir_path, "plots_dfwash_general", "df_wash_general.csv"), index=False)

    make_dfwash_plots(df_wash, out_dir=os.path.join(out_dir_path, "plots_dfwash_general"), show=False, top_wallets=25)
    global_snapshot(df_wash, out_dir=os.path.join(out_dir_path, "plots_dfwash_general"), fname=f"global_snapshot_{token}.json")

    per_wallet_leaderboard(df_wash, top=50, out_dir=os.path.join(out_dir_path, "plots_dfwash_general"))
    plot_wash_volume_by_date_bar(df_wash, freq="D", out_dir=os.path.join(out_dir_path, "plots_dfwash_general"), show=False)
    plot_wash_notional_by_date_bar(df_wash, freq="D", out_dir=os.path.join(out_dir_path, "plots_dfwash_general"), show=False)

    store = CoinDataStore(token, engine="fastparquet")
    df_trades = store.load_all()
    wash_trade_summary(df_trades, df_wash, out_dir=os.path.join(out_dir_path, "plots_dfwash_general"))
