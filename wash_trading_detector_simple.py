import os
from pathlib import Path

from loguru import logger

from src.data_handler import CoinDataStore
from src.simple_wash_detector_utils import detect_wash_trades_nearest, detected_to_dfwash_full
from src.wash_trading_pairs_analyser import wash_trading_pairs_analysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = Path(os.path.join(BASE_DIR, "..", "data"))

def main():
    logger.add("logfile.log", rotation="2 MB", level="INFO")

    token_list = os.listdir(DATA_DIR)
    for token in token_list:
        store = CoinDataStore(token, engine="fastparquet")
        print(store.list_days())
        time_diff_s = 10 * 60  # 10 minutes difference
        price_diff_pct = 0.01
        size_diff_pct = 0.01
        df_detected = detect_wash_trades_nearest(store, time_diff_s, price_diff_pct, size_diff_pct)
        dfwash = detected_to_dfwash_full(df_detected)
        wash_trading_pairs_analysis(dfwash, token, "simple_detector")


if __name__ == "__main__":
    main()
