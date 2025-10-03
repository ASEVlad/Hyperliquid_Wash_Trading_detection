from pathlib import Path
from typing import Dict, Tuple, List
import os
import json
import pandas as pd
from loguru import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
HL_NODE_TRADES_PATH = os.path.abspath(os.path.join(HOME_DIR, "hl-node-trades"))

# --- Config/paths ---
DATA_DIR = Path(os.path.join(HOME_DIR, "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

WALLETS_CSV = DATA_DIR / "wallet_db.csv"


# --- Wallet DB helpers ---
def load_wallet_db(csv_path: Path = WALLETS_CSV) -> Tuple[Dict[str, int], int]:
    """
    Load wallets from CSV into a dict {wallet: wallet_id}, return dict and next_id.
    If file doesn't exist, start fresh at 1.
    """
    mapping: Dict[str, int] = {}
    next_id = 1
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"wallet_id": "uint32", "wallet": "string"})
        if not df.empty:
            for wid, wal in zip(df["wallet_id"].astype("uint32"), df["wallet"].astype("string")):
                mapping[str(wal)] = int(wid)
            next_id = int(df["wallet_id"].max()) + 1
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["wallet_id", "wallet"]).to_csv(csv_path, index=False)
    return mapping, next_id


def append_wallet(csv_path: Path, wallet: str, wallet_id: int) -> None:
    """Append a single wallet row to the CSV."""
    pd.DataFrame([{"wallet_id": wallet_id, "wallet": wallet}]).to_csv(
        csv_path, mode="a", header=False, index=False
    )


def get_wallet_id(wallet: str, mapping: Dict[str, int], next_id_ref: List[int], csv_path: Path) -> int:
    """
    Return wallet_id for wallet, creating a new id if needed.
    next_id_ref is a single-item list to allow in-place increment.
    """
    w = str(wallet)
    wid = mapping.get(w)
    if wid is not None:
        return wid
    wid = next_id_ref[0]
    mapping[w] = wid
    next_id_ref[0] += 1
    append_wallet(csv_path, w, wid)
    return wid


def retrieve_data(file_path: Path, wallet_map: Dict[str, int], next_id_ref: List[int],
                  wallets_csv: Path = WALLETS_CSV) -> pd.DataFrame:
    """
    Read a newline-delimited JSON file of trades and produce a normalized DataFrame
    for later partitioned saving.
    Output columns: coin, price, size, time, is_ask, wallet_id
    """
    records = []
    with open(file_path) as f:
        append = records.append
        for line in f:
            if not line.strip():
                continue
            trade = json.loads(line)

            side_info_list = trade.get("side_info")
            buyer_wallet = side_info_list[0].get("user")
            seller_wallet = side_info_list[1].get("user")
            buyer_wallet_id = get_wallet_id(buyer_wallet, wallet_map, next_id_ref, wallets_csv)
            seller_wallet_id = get_wallet_id(seller_wallet, wallet_map, next_id_ref, wallets_csv)

            px = trade.get("px")
            sz = trade.get("sz")

            append(
                {
                    "coin": trade.get("coin"),
                    "price": float(px),
                    "size": float(sz),
                    "time": trade.get("time"),
                    "seller": seller_wallet_id,
                    "buyer": buyer_wallet_id,
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    # Types & cleaning
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    # enforce dtypes
    df["price"] = df["price"].astype("float32")
    df["size"] = df["size"].astype("float32")
    df["seller"] = df["seller"].astype("uint64")
    df["buyer"] = df["buyer"].astype("uint64")

    return df[["coin", "price", "size", "time", "seller", "buyer"]]


def _target_path_for(coin: str, dt: pd.Timestamp) -> Path:
    return DATA_DIR / str(coin) / f"{dt.date()}.parquet"


def _write_daily_parquet(target: Path, df_day: pd.DataFrame) -> None:
    """
    Write/merge the daily file. If target exists, read, concat, de-dup, sort, write.
    We de-dup on [time, wallet_id, price, size, is_ask] as a reasonable row identity.
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    # Keep only required columns & types
    cols = ["price", "size", "time", "seller", "buyer"]
    df_day = df_day[cols].copy()

    if target.exists():
        try:
            old = pd.read_parquet(target, engine="pyarrow")
            # Cast to same dtypes to avoid upcasting surprises
            old["price"] = old["price"].astype("float32")
            old["size"] = old["size"].astype("float32")
            old["time"] = pd.to_datetime(old["time"], errors="coerce")
            old["seller"] = old["seller"].astype("uint64")
            old["buyer"] = old["buyer"].astype("uint64")
            df_day = pd.concat([old, df_day], ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to read existing parquet {target}: {e}. Overwriting.")

    df_day = df_day.dropna(subset=["time"]).drop_duplicates(
        subset=["time", "seller", "buyer", "price", "size"], keep="last"
    )
    df_day = df_day.sort_values("time")
    df_day.to_parquet(target, index=False, engine="pyarrow", compression="snappy")


def save_partitioned(df: pd.DataFrame) -> None:
    """
    Save rows to data/<coin>/<YYYY-MM-DD>.parquet, merging per-day files if present.
    """
    if df.empty:
        logger.warning("No data to save.")
        return

    # Add date for grouping
    df = df.copy()
    df["date"] = df["time"].dt.date

    # Group by coin/date
    for (coin, day), g in df.groupby(["coin", "date"], sort=False):
        if pd.isna(coin) or coin == "":
            logger.warning("Skipping rows with empty coin.")
            continue
        target = DATA_DIR / str(coin) / f"{day}.parquet"
        _write_daily_parquet(target, g)

    logger.info("Data has been saved successfully.")



def main():
    logger.add("logfile.log", rotation="2 MB", level="INFO")
    old_data_folders = os.listdir(os.path.join(HOME_DIR, "hl-node-trades"))

    wallet_map, next_id = load_wallet_db()
    next_id_ref = [next_id]  # mutable holder

    for i, date in enumerate(old_data_folders):
        hour_file_names = os.listdir(os.path.join(HOME_DIR, "hl-node-trades", date))

        for file_name in hour_file_names:
            file_full_path = os.path.join(HOME_DIR, "hl-node-trades", date, file_name)

            logger.info(f"{file_full_path} is processing")
            df = retrieve_data(Path(file_full_path), wallet_map, next_id_ref, WALLETS_CSV)
            save_partitioned(df)

        logger.info(f"Processed {i} out of {len(old_data_folders)}")


if __name__ == "__main__":
    main()