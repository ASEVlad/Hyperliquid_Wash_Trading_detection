import numpy as np
import pandas as pd

from src.data_handler import CoinDataStore


# --------------------------
# 1) Helpers
# --------------------------

def _to_long_per_wallet(df_day: pd.DataFrame) -> pd.DataFrame:
    if df_day.empty:
        return pd.DataFrame(columns=["wallet_id","side","time","price","size"])

    df = df_day.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    sells = df[["seller", "time", "price", "size"]].rename(columns={"seller": "wallet_id"}).copy()
    sells["side"] = "sell"

    buys = df[["buyer", "time", "price", "size"]].rename(columns={"buyer": "wallet_id"}).copy()
    buys["side"] = "buy"

    long = pd.concat([sells, buys], ignore_index=True)

    # Tidy types
    long["wallet_id"] = long["wallet_id"].astype("uint64", copy=False)
    long["price"] = long["price"].astype("float32", copy=False)
    long["size"] = long["size"].astype("float32", copy=False)

    # Order by time
    long = long.sort_values("time").reset_index(drop=True)
    return long


def _aggregate_microfills_long(
    long_df: pd.DataFrame,
    bin_freq: str = "50ms",
    round_mode: str = "ceil",   # "ceil" | "floor" | "round"
) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    df = long_df.copy()
    # Choose binning variant
    if round_mode == "ceil":
        df["time_bin"] = df["time"].dt.ceil(bin_freq)
    elif round_mode == "floor":
        df["time_bin"] = df["time"].dt.floor(bin_freq)
    else:
        df["time_bin"] = df["time"].dt.round(bin_freq)

    # Remove nonpositive sizes to avoid bad VWAP math
    df = df[df["size"] > 0].copy()
    df["notional"] = df["price"] * df["size"]

    g = (
        df.groupby(["wallet_id", "side", "time_bin"], as_index=False)
          .agg(size=("size", "sum"), notional=("notional", "sum"))
    )
    g["price"] = (g["notional"] / g["size"]).astype("float32")
    g = g.drop(columns=["notional"])
    g = g.rename(columns={"time_bin": "time"})

    # Keep dtypes tidy & sort for asof
    g["time"] = pd.to_datetime(g["time"], errors="coerce")
    g = g.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    g["wallet_id"] = g["wallet_id"].astype("uint64", copy=False)
    g["size"] = g["size"].astype("float32", copy=False)
    return g


def _one_direction_pair(open_df: pd.DataFrame, close_df: pd.DataFrame,
                        time_diff_s: int, price_diff_pct: float, size_diff_pct: float,
                        direction_label: str) -> pd.DataFrame:
    if open_df.empty or close_df.empty:
        return pd.DataFrame()

    right = close_df[["wallet_id","time","price","size","row_id"]].rename(
        columns={"time":"close_time","price":"close_price","size":"close_size","row_id":"close_row_id"}
    )
    left = open_df.rename(
        columns={"time":"open_time","price":"open_price","size":"open_size","row_id":"open_row_id"}
    )

    # Guard against zeros to avoid division by zero in size % diff
    left = left[left["open_size"] > 0].copy()
    right = right[right["close_size"] > 0].copy()

    # **Important**: merge_asof requires GLOBAL sort by time keys
    left = left.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    right = right.sort_values("close_time", kind="mergesort").reset_index(drop=True)

    pairs = pd.merge_asof(
        left,
        right,
        left_on="open_time",
        right_on="close_time",
        by="wallet_id",
        direction="forward",
        tolerance=pd.Timedelta(seconds=time_diff_s),
        allow_exact_matches=True,
    ).dropna(subset=["close_time"])

    if pairs.empty:
        return pairs

    # Compute filters
    pairs["duration_s"] = (pairs["close_time"] - pairs["open_time"]).dt.total_seconds()
    pairs["price_change_pct"] = (pairs["close_price"] - pairs["open_price"]).abs() / pairs["open_price"]
    pairs["size_change_pct"]  = (pairs["close_size"]  - pairs["open_size"]).abs()  / pairs["open_size"]

    # Apply thresholds
    pairs = pairs[
        (pairs["duration_s"] <= time_diff_s)
        & (pairs["price_change_pct"] <= price_diff_pct)
        & (pairs["size_change_pct"] <= size_diff_pct)
    ]
    if pairs.empty:
        return pairs

    # Label sides/direction — the open side is given by open_df; close side is the opposite
    pairs = pairs.assign(
        pairing_direction=direction_label,
        open_side=pairs["side"],  # from 'left' (open_df)
        close_side=np.where(pairs["side"] == "buy", "sell", "buy"),
    )

    # Keep only relevant columns before conflict resolution
    keep_cols = [
        "wallet_id", "day",
        "open_row_id", "close_row_id",
        "open_time", "close_time",
        "open_side", "close_side",
        "open_price", "close_price", "price_change_pct",
        "open_size", "close_size", "size_change_pct",
        "duration_s", "pairing_direction",
    ]
    pairs = pairs[keep_cols].copy()

    # 1–to–1 greedy pruning: prefer smallest Δt, then smallest Δp
    pairs = pairs.sort_values(
        ["wallet_id", "duration_s", "price_change_pct", "open_time"],
        kind="mergesort"
    )
    pairs = pairs.drop_duplicates(subset=["wallet_id", "close_row_id"], keep="first")
    pairs = pairs.drop_duplicates(subset=["wallet_id", "open_row_id"],  keep="first")
    return pairs


# --------------------------
# 2) Main detector
# --------------------------

def detect_wash_trades_nearest(
    store: CoinDataStore,
    time_diff_s: int,
    price_diff_pct: float,
    size_diff_pct: float,
    bin_freq: str = "50ms",
    round_mode: str = "ceil",
) -> pd.DataFrame:
    """
    Wash-trading detector for schema: price, size, time, seller, buyer.

    Pipeline (per day):
      A) Expand to per-wallet events (seller->sell, buyer->buy).
      B) Aggregate micro-fills in 50 ms bins per (wallet_id, side) using VWAP price.
      C) Pair nearest-forward opposite-side events within time_diff_s (per wallet).
      D) Keep only pairs with price & size within tolerances; enforce 1–to–1.

    Returns one row per flagged pair with timings, sides, prices, sizes, and directions.
    """
    all_events = []

    for df_day in store.iter_days():
        if df_day.empty:
            continue

        # A) Long view per wallet (buy/sell events)
        long_df = _to_long_per_wallet(df_day)
        if long_df.empty:
            continue

        # B) Aggregate micro-fills
        agg = _aggregate_microfills_long(long_df, bin_freq=bin_freq, round_mode=round_mode)
        if agg.empty:
            continue

        df = agg.sort_values("time").reset_index(drop=True).copy()
        df["row_id"] = np.arange(len(df), dtype=np.int64)  # stable within day
        df["day"] = df["time"].dt.date

        buys  = df[df["side"] == "buy"].copy()
        sells = df[df["side"] == "sell"].copy()

        # C) Two directions
        d1 = _one_direction_pair(buys,  sells, time_diff_s, price_diff_pct, size_diff_pct, "buy_to_sell")
        d2 = _one_direction_pair(sells, buys,  time_diff_s, price_diff_pct, size_diff_pct, "sell_to_buy")

        day_events = pd.concat([d1, d2], ignore_index=True)
        if not day_events.empty:
            # Instead of relying on drop_duplicates in each direction, apply global pruning:
            day_events = _prune_pairs_greedy_no_reuse(
                day_events,
                sort_cols=("wallet_id", "duration_s", "price_change_pct"),
                ascending=(True, True, True)
            )
            all_events.append(day_events)

    if not all_events:
        return pd.DataFrame(
            columns=[
                "wallet_id","day",
                "open_time","close_time","duration_s",
                "open_side","close_side",
                "open_price","close_price","price_change_pct",
                "open_size","close_size","size_change_pct",
                "pairing_direction",
                "pair_id",
            ]
        )

    events = pd.concat(all_events, ignore_index=True)
    events = events.sort_values(["day","wallet_id","open_time","close_time"]).reset_index(drop=True)

    # Assign a simple pair_id (one row == one pair)
    events["pair_id"] = np.arange(len(events), dtype=np.int64)

    # Final tidy order
    events = events[
        [
            "wallet_id","day",
            "open_time","close_time","duration_s",
            "open_side","close_side",
            "open_price","close_price","price_change_pct",
            "open_size","close_size","size_change_pct",
            "pairing_direction",
            "pair_id",
        ]
    ]
    return events


def _prune_pairs_greedy_no_reuse(candidates: pd.DataFrame,
                                 sort_cols=("wallet_id", "duration_s", "price_change_pct"),
                                 ascending=(True, True, True)) -> pd.DataFrame:
    """
    Greedy 1-to-1 pruning across all candidate pairs:
      - candidates: concatenated d1 + d2 with open_row_id and close_row_id present
      - sort_cols: ordering to prefer better matches first (default: smaller duration, smaller price diff)
    Returns pruned DataFrame where no open_row_id or close_row_id is reused.
    """
    if candidates.empty:
        return candidates

    # Ensure deterministic order
    if isinstance(sort_cols, str):
        sort_cols = (sort_cols,)
    if isinstance(ascending, bool):
        ascending = tuple([ascending] * len(sort_cols))

    cand = candidates.sort_values(list(sort_cols), ascending=list(ascending), kind="mergesort").copy()

    used_id = set()

    keep_idx = []
    for i, row in cand.iterrows():
        o = int(row["open_row_id"])
        c = int(row["close_row_id"])
        if (o in used_id) or (c in used_id):
            continue
        used_id.add(o)
        used_id.add(c)
        keep_idx.append(i)

    pruned = cand.loc[keep_idx].reset_index(drop=True)
    return pruned


def detected_to_dfwash_full(df_detected: pd.DataFrame) -> pd.DataFrame:
    # core dfwash schema
    mapped = pd.DataFrame({
        "wallet_id": df_detected["wallet_id"],
        "price1": df_detected["open_price"],
        "size1": df_detected["open_size"],
        "price2": df_detected["close_price"],
        "size2": df_detected["close_size"],
        "side1": df_detected["open_side"],
        "side2": df_detected["close_side"],
        "t1": pd.to_datetime(df_detected["open_time"], errors="coerce"),
        "t2": pd.to_datetime(df_detected["close_time"], errors="coerce"),
        "delta_seconds": df_detected["duration_s"],
    }, index=df_detected.index)

    # size ratio
    mapped["size_ratio"] = mapped["size2"] / mapped["size1"].replace(0, np.nan)

    # enforce lightweight dtypes
    cast = {
        "wallet_id": "uint32",
        "price1": "float32", "size1": "float32",
        "price2": "float32", "size2": "float32",
        "delta_seconds": "float32", "size_ratio": "float32",
    }
    for c, dt in cast.items():
        if c in mapped.columns:
            try:
                mapped[c] = mapped[c].astype(dt, copy=False)
            except Exception:
                pass

    # derived cols
    mapped["direction"] = (mapped["side1"].astype("string") + "->" + mapped["side2"].astype("string")).astype("category")
    mapped["dt_s"] = mapped["delta_seconds"].astype("float32")
    mapped["size_err_pct"] = (mapped["size_ratio"] - 1.0).abs().astype("float32") * 100.0
    base = mapped["price1"].replace(0, np.nan)
    mapped["price_change_pct"] = ((mapped["price2"] - mapped["price1"]) / base).astype("float32")
    mapped["price_change_bps"] = (mapped["price_change_pct"] * 1e4).astype("float32")
    mapped["same_price"] = (mapped["price2"] - mapped["price1"]).abs() <= 1e-8
    mapped["date"] = mapped["t1"].dt.date
    mapped["hour"] = mapped["t1"].dt.hour.astype("int16")
    mapped["dow"] = mapped["t1"].dt.dayofweek.astype("int8")

    # combine original df_detected + mapped features
    df_full = pd.concat([df_detected.reset_index(drop=True), mapped.reset_index(drop=True)], axis=1)
    # remove duplicated column names across the whole frame
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]

    return df_full
