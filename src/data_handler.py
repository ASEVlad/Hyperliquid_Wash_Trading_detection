import os
import pandas as pd
import datetime as dt
from pathlib import Path
from loguru import logger
from typing import Iterable, Iterator, Optional, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Config/paths
DATA_DIR = Path(os.path.join(HOME_DIR, "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _fix_pyarrow_period_double_registration():
    # Only relevant when using pyarrow; harmless otherwise.
    try:
        import pyarrow as pa
        try:
            pa.unregister_extension_type("pandas.period")
        except Exception:
            pass
    except Exception:
        pass

_SCHEMA = {
    "price": "float32",
    "size": "float32",
    "time": "datetime64[ns]",
    "seller": "uint64",
    "buyer": "uint64",
}

class CoinDataStore:
    def __init__(self, coin: str, base_dir: Path | str = DATA_DIR, engine: str = "fastparquet"):
        """
        engine: "fastparquet" (default) or "pyarrow"
        """
        self.coin = str(coin)
        self.base_dir = Path(base_dir)
        self.coin_dir = self.base_dir / self.coin
        self.engine = engine

    # ---------- Paths & listing ----------
    def day_path(self, day: dt.date | str) -> Path:
        if isinstance(day, str):
            day = dt.date.fromisoformat(day)  # 'YYYY-MM-DD'
        return self.coin_dir / f"{day}.parquet"

    def list_days(self) -> List[dt.date]:
        if not self.coin_dir.exists():
            return []
        days = []
        for name in os.listdir(self.coin_dir):
            if not name.endswith(".parquet"):
                continue
            try:
                d = dt.date.fromisoformat(name[:-8])  # strip '.parquet'
                days.append(d)
            except ValueError:
                logger.warning(f"Skipping unexpected file in {self.coin_dir}: {name}")
        days.sort()
        return days

    # ---------- Core reader ----------
    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """
        Try fastparquet first (if selected), then pyarrow as a fallback.
        """
        # Engine preference
        engines = [self.engine]
        if self.engine == "fastparquet":
            engines += ["pyarrow"]
        else:
            engines += ["fastparquet"]

        last_err = None
        for eng in engines:
            try:
                if eng == "pyarrow":
                    _fix_pyarrow_period_double_registration()
                return pd.read_parquet(path, engine=eng)
            except Exception as e:
                last_err = e
                logger.debug(f"read_parquet failed with engine={eng} for {path.name}: {e}")
                continue
        # If all engines failed, raise the last error
        raise last_err if last_err else RuntimeError(f"Failed to read {path}")

    def _read_one_day(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=list(_SCHEMA.keys())).astype(_SCHEMA)

        df = self._read_parquet(path)

        # Ensure required columns exist
        for col in _SCHEMA.keys():
            if col not in df.columns:
                df[col] = pd.Series(dtype=_SCHEMA[col])

        # Time parsing
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # Dtype normalization (be robust to NaNs in bool/uint)
        # floats
        for col in ("price", "size"):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            except Exception as e:
                logger.warning(f"Column {col} cast to float32 failed in {path.name}: {e}")

        # uint64 (need fillna first)
        try:
            df["seller"] = pd.to_numeric(df["seller"], errors="coerce").fillna(0).astype("uint64")
        except Exception as e:
            logger.warning(f"Column seller cast to uint64 failed in {path.name}: {e}")
            df["seller"] = pd.to_numeric(df["seller"], errors="coerce").fillna(0).astype("uint64")

        # uint64 (need fillna first)
        try:
            df["buyer"] = pd.to_numeric(df["buyer"], errors="coerce").fillna(0).astype("uint64")
        except Exception as e:
            logger.warning(f"Column buyer cast to uint64 failed in {path.name}: {e}")
            df["buyer"] = pd.to_numeric(df["buyer"], errors="coerce").fillna(0).astype("uint64")

        # Column order & drop bad times
        df = df[["price", "size", "time", "seller", "buyer"]]
        df = df.dropna(subset=["time"])
        return df

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.astype(_SCHEMA)

        df = df.drop_duplicates(subset=["time", "price", "size", "buyer", "seller"], keep="last")
        df = df.sort_values("time")

        # Reassert dtypes (idempotent)
        df["price"] = df["price"].astype("float32")
        df["size"] = df["size"].astype("float32")
        df["buyer"] = df["buyer"].astype("uint64")
        df["seller"] = df["seller"].astype("uint64")
        return df

    # ---------- APIs ----------
    def load_all(self) -> pd.DataFrame:
        days = self.list_days()
        if not days:
            return pd.DataFrame(columns=list(_SCHEMA.keys())).astype(_SCHEMA)
        parts = [self._read_one_day(self.day_path(d)) for d in days]
        return self._finalize(pd.concat(parts, ignore_index=True)) if parts else self._finalize(pd.DataFrame())

    def load_between(
        self,
        start: dt.date | str | None = None,
        end: dt.date | str | None = None,
        inclusive: str = "both",
    ) -> pd.DataFrame:

        avail = self.list_days()
        if not avail:
            return pd.DataFrame(columns=list(_SCHEMA.keys())).astype(_SCHEMA)

        if isinstance(start, str): start = dt.date.fromisoformat(start)
        if isinstance(end, str):   end = dt.date.fromisoformat(end)

        lo = start if start else avail[0]
        hi = end if end else avail[-1]

        def in_range(d: dt.date) -> bool:
            if inclusive in ("both", "left"):
                left_ok = d >= lo
            else:
                left_ok = d > lo
            if inclusive in ("both", "right"):
                right_ok = d <= hi
            else:
                right_ok = d < hi
            return left_ok and right_ok

        selected = [d for d in avail if in_range(d)]
        if not selected:
            return pd.DataFrame(columns=list(_SCHEMA.keys())).astype(_SCHEMA)

        parts = [self._read_one_day(self.day_path(d)) for d in selected]
        return self._finalize(pd.concat(parts, ignore_index=True))

    def load_days(self, days: Iterable[dt.date | str]) -> pd.DataFrame:
        paths = []
        for d in days:
            if isinstance(d, str):
                d = dt.date.fromisoformat(d)
            paths.append(self.day_path(d))
        parts = [self._read_one_day(p) for p in paths]
        return self._finalize(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())

    def iter_days(self, start: Optional[dt.date | str] = None, end: Optional[dt.date | str] = None) -> Iterator[pd.DataFrame]:
        avail = self.list_days()
        if not avail:
            return
        if isinstance(start, str): start = dt.date.fromisoformat(start)
        if isinstance(end, str):   end = dt.date.fromisoformat(end)
        lo = start if start else avail[0]
        hi = end if end else avail[-1]
        for d in avail:
            if d < lo or d > hi:
                continue
            yield self._finalize(self._read_one_day(self.day_path(d)))
