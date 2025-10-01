import numpy as np
import pandas as pd

def _aggregate_microfills(
    df_day: pd.DataFrame,
    bin_freq: str = "50ms",
    round_mode: str = "ceil",   # "ceil" | "floor" | "round"
) -> pd.DataFrame:
    """
    Collapse micro-fills into a single print per (wallet_id, is_ask, time_bin).

    - time_bin: time rounded to `bin_freq` (ceil/floor/round)
    - size: summed within the bin
    - price: volume-weighted average (VWAP) within the bin

    Returns a DataFrame with the same columns as input (price, size, time, is_ask, wallet_id),
    but 'time' is replaced by the binned timestamp and rows are aggregated.
    """
    if df_day.empty:
        return df_day

    df = df_day.copy()
    # Ensure proper dtypes
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    # Choose the binning function
    if round_mode == "ceil":
        df["time_bin"] = df["time"].dt.ceil(bin_freq)
    elif round_mode == "floor":
        df["time_bin"] = df["time"].dt.floor(bin_freq)
    else:
        df["time_bin"] = df["time"].dt.round(bin_freq)

    # Guard against nonpositive sizes in VWAP
    df = df[df["size"] > 0].copy()

    # Precompute notional for VWAP
    df["notional"] = df["price"] * df["size"]

    g = (
        df.groupby(["wallet_id", "is_ask", "time_bin"], as_index=False)
          .agg(size=("size", "sum"), notional=("notional", "sum"))
    )
    # VWAP = sum(price*size)/sum(size)
    g["price"] = g["notional"] / g["size"]
    g = g.drop(columns=["notional"])

    # Rename back to your schema
    g = g.rename(columns={"time_bin": "time"})

    # Enforce your dtypes (optional, tidy)
    g["price"] = g["price"].astype("float32")
    g["size"] = g["size"].astype("float32")
    g["is_ask"] = g["is_ask"].astype("bool")
    g["wallet_id"] = g["wallet_id"].astype("uint32")

    # Sort by time (needed later for merge_asof)
    g = g.sort_values("time").reset_index(drop=True)
    return g


def detect_wash_trades_nearest(
    store: "CoinDataStore",
    time_diff_s: int,
    price_diff_pct: float,
    size_diff_pct: float,          # tolerance for size match, e.g. 0.02 for ±2%
    bin_freq: str = "50ms",        # micro-fill aggregation window
    round_mode: str = "ceil",      # how to align bins: "ceil" matches your note
) -> pd.DataFrame:
    """
    Simple wash trading detection with micro-fill aggregation + nearest-neighbor pairing.

    Steps (per day):
      1) Aggregate micro-fills into 50ms bins per (wallet_id, is_ask).
         - size = sum
         - price = VWAP
      2) For each wallet, pair each open trade with the nearest *forward* opposite-side trade
         within `time_diff_s` using `pd.merge_asof`.
      3) Keep pairs where:
           - abs(price_close - price_open)/price_open  <= price_diff_pct
           - abs(size_close  - size_open )/size_open   <= size_diff_pct
      4) Resolve conflicts to enforce 1–to–1 matches (greedy by smallest Δt, then Δp).

    Returns one row per flagged pair.
    """
    def one_direction_pair(open_df: pd.DataFrame, close_df: pd.DataFrame, direction_label: str) -> pd.DataFrame:
        if open_df.empty or close_df.empty:
            return pd.DataFrame()

        right = close_df[["wallet_id", "time", "price", "size", "row_id"]].rename(
            columns={
                "time": "close_time",
                "price": "close_price",
                "size": "close_size",
                "row_id": "close_row_id",
            }
        )
        left = open_df.rename(
            columns={
                "time": "open_time",
                "price": "open_price",
                "size": "open_size",
                "row_id": "open_row_id",
            }
        )

        # Remove zero sizes to avoid divisions in size % diff
        left = left[left["open_size"] > 0].copy()
        right = right[right["close_size"] > 0].copy()

        # **Important**: global sort by the time keys for merge_asof
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

        # Filters
        pairs["duration_s"] = (pairs["close_time"] - pairs["open_time"]).dt.total_seconds()
        pairs["price_change_pct"] = (pairs["close_price"] - pairs["open_price"]).abs() / pairs["open_price"]
        pairs["size_change_pct"]  = (pairs["close_size"]  - pairs["open_size"]).abs()  / pairs["open_size"]

        pairs = pairs[
            (pairs["duration_s"] <= time_diff_s)
            & (pairs["price_change_pct"] <= price_diff_pct)
            & (pairs["size_change_pct"]  <= size_diff_pct)
        ]
        if pairs.empty:
            return pairs

        pairs = pairs.assign(
            pairing_direction=direction_label,
            open_side=pairs["side"],  # side from open leg
            close_side=np.where(pairs["side"] == "buy", "sell", "buy"),
        )

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

        # 1–to–1 greedy pruning
        pairs = pairs.sort_values(
            ["wallet_id", "duration_s", "price_change_pct", "open_time"],
            kind="mergesort"
        )
        pairs = pairs.drop_duplicates(subset=["wallet_id", "close_row_id"], keep="first")
        pairs = pairs.drop_duplicates(subset=["wallet_id", "open_row_id"],  keep="first")
        return pairs

    all_events = []
    for df_day in store.iter_days():
        print(len(all_events))
        if df_day.empty:
            continue

        # 1) Aggregate micro-fills first
        agg = _aggregate_microfills(df_day, bin_freq=bin_freq, round_mode=round_mode)
        if agg.empty:
            continue

        # Enrich for pairing
        df = agg.copy()
        df = df.sort_values("time").reset_index(drop=True)
        df["side"] = np.where(df["is_ask"], "sell", "buy")
        df["row_id"] = np.arange(len(df), dtype=np.int64)  # stable within the day
        df["day"] = df["time"].dt.date

        buys  = df[df["side"] == "buy"].copy()
        sells = df[df["side"] == "sell"].copy()

        d1 = one_direction_pair(buys,  sells, "buy_to_sell")
        d2 = one_direction_pair(sells, buys,  "sell_to_buy")

        day_events = pd.concat([d1, d2], ignore_index=True)
        if not day_events.empty:
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
                "open_trade_id","close_trade_id",
            ]
        )

    events = pd.concat(all_events, ignore_index=True)
    events = events.sort_values(["day", "wallet_id", "open_time", "close_time"]).reset_index(drop=True)
    events["open_trade_id"]  = np.arange(len(events), dtype=np.int64)
    events["close_trade_id"] = events["open_trade_id"]
    events = events[
        [
            "wallet_id","day",
            "open_time","close_time","duration_s",
            "open_side","close_side",
            "open_price","close_price","price_change_pct",
            "open_size","close_size","size_change_pct",
            "pairing_direction",
            "open_trade_id","close_trade_id",
        ]
    ]
    return events