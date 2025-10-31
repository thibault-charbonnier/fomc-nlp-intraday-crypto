import io
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from src.tools.logging_config import get_logger

logger = get_logger(__name__)


class MarketDataLoader:
    """
    Market data loader for crypto assets (e.g. BTCUSDT).
    Responsible for returning 1-minute spot data over a given time range.

    Notes:
        - Data is pulled from Binance public data ("Binance Vision"):
          https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-YYYY-MM-DD.zip
          The ZIP contains one CSV for that trading day.
          Columns are standard Binance kline format:
              [ open_time,
                open, high, low, close, volume,
                close_time,
                quote_asset_volume,
                number_of_trades,
                taker_buy_base_volume,
                taker_buy_quote_volume,
                ignore ]
          open_time and close_time are in ms (historical) or µs (2025+) UTC.

        - We cache the full concatenated minute data for each symbol locally
          into <data_dir>/<SYMBOL>_1m.parquet

        - When load() is called for a given [start_ts, end_ts], we:
            1) read the local cache if present
            2) fetch remotely the required days (if cache is missing or incomplete)
            3) merge, de-duplicate, sort, update cache
            4) return the requested slice
    """

    def __init__(self, data_dir: str) -> None:
        """
        Params:
            data_dir : str
                Base directory for local cached market data.
        """
        self.data_dir = Path(data_dir).resolve()

    # -------------------- Public method --------------------
    def load(self,
             symbol: str,
             start_ts: datetime,
             end_ts: datetime,
             force_fetch: bool = False) -> pd.DataFrame:
        """
        Load 1-minute spot market data for a given symbol over [start_ts, end_ts].

        Params:
            symbol : str
                Asset symbol, e.g. "BTCUSDT".
            start_ts : datetime
                Start timestamp (UTC).
            end_ts : datetime
                End timestamp (UTC).
            force_fetch : bool (default False)
                If True, ignore the local cache and always re-fetch remote
                data for the requested range.

        Returns:
            pd.DataFrame
                Minute-level OHLCV data indexed by timestamp (UTC).
        """
        logger.info(
            "Loading market data for %s between %s and %s (force_fetch=%s)",
            symbol, start_ts, end_ts, force_fetch
        )

        # Step 1: try loading local cache
        df_full = self._load_local_cache(symbol)
        if df_full is not None:
            logger.debug("Local cache loaded for %s with %d rows", symbol, len(df_full))
        else:
            logger.debug("No local cache for %s", symbol)

        # Step 2: fetch missing range (or full range if force_fetch)
        df_remote = self._fetch_remote(symbol, start_ts, end_ts, force_fetch=force_fetch)

        # Step 3: merge cache + fetched
        if df_full is None:
            df_merged = df_remote
        else:
            if df_remote is not None and not df_remote.empty:
                df_merged = pd.concat([df_full, df_remote]).sort_index().drop_duplicates(subset=None, keep="last")
            else:
                df_merged = df_full

        # Step 4: write updated cache locally
        if df_merged is not None and not df_merged.empty:
            self._write_local_cache(symbol, df_merged)

        # Step 5: return requested slice
        if df_merged is None or df_merged.empty:
            logger.error("No data available for %s in requested range", symbol)
            return pd.DataFrame()

        df_slice = self._subset_range(df_merged, start_ts, end_ts)
        return df_slice

    # -------------------- Private utilities --------------------
    def _load_local_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Attempt to load cached historical 1-minute data for the given symbol.

        Params:
            symbol : str
                Asset symbol.

        Returns:
            pd.DataFrame | None
                Cached data if available, otherwise None.
                Expected index: UTC DateTimeIndex named "timestamp".
        """
        path = self.data_dir / f"{symbol}_1m.parquet"

        if not path.exists():
            logger.warning("No local cache found for %s at %s", symbol, path)
            return None

        logger.debug("Loading local cache for %s from %s", symbol, path)
        df = pd.read_parquet(path)

        # Ensure timestamp index and sorting
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")

        df = df.sort_index()

        # Basic sanity check: must have "close"
        if "close" not in df.columns:
            logger.error("Cached data for %s is missing 'close' column", symbol)
            return None

        return df

    def _write_local_cache(self,
                           symbol: str,
                           df: pd.DataFrame) -> None:
        """
        Write (or overwrite) the local cache for a given symbol.

        Params:
            symbol : str
                Asset symbol.
            df : pd.DataFrame
                Historical data to cache, indexed by timestamp (UTC).
        """
        path = self.data_dir / f"{symbol}_1m.parquet"
        logger.debug("Writing local cache for %s to %s (%d rows)", symbol, path, len(df))

        df_to_save = df.copy()
        if df_to_save.index.name is None:
            df_to_save.index.name = "timestamp"

        df_to_save.reset_index().to_parquet(path, index=False)

    def _fetch_remote(self,
                      symbol: str,
                      start_ts: datetime,
                      end_ts: datetime,
                      force_fetch: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch 1-minute spot kline data for [start_ts, end_ts] from Binance Vision.

        Params:
            symbol : str
                Asset symbol, e.g. "BTCUSDT".
            start_ts : datetime
                Start timestamp (UTC).
            end_ts : datetime
                End timestamp (UTC).
            force_fetch : bool
                If True, we will re-query remote even if we already have a cache.

        Returns:
            pd.DataFrame | None
                Full minute-level data covering all requested days,
                indexed by UTC timestamp.
                Columns:
                    ["open","high","low","close","volume","n_trades"]
                Returns None if nothing could be downloaded.

        Notes:
            We pull per-day ZIPs from:
                https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/{symbol}-1m-YYYY-MM-DD.zip
            Example:
                .../BTCUSDT-1m-2025-08-27.zip
            Each ZIP contains a single CSV with all 1m candles of that UTC day.
        """
        day_list = pd.date_range(
            start=start_ts.date(),
            end=end_ts.date(),
            freq="D",
            tz="UTC"
        )

        if len(day_list) == 0:
            logger.warning("Empty day_list for _fetch_remote on %s", symbol)
            return None

        all_days: List[pd.DataFrame] = []

        for day in day_list:
            day_str = day.strftime("%Y-%m-%d")
            logger.debug("Fetching remote data for %s on %s", symbol, day_str)

            df_day = self._download_daily_klines(symbol, day_str)

            if df_day is None or df_day.empty:
                logger.warning("No data returned for %s on %s", symbol, day_str)
                continue

            all_days.append(df_day)

        if len(all_days) == 0:
            logger.error("Remote fetch returned no data for %s in requested range", symbol)
            return None

        df_all = pd.concat(all_days).sort_index().drop_duplicates(subset=None, keep="last")

        return df_all

    def _download_daily_klines(self,
                               symbol: str,
                               day_str: str) -> Optional[pd.DataFrame]:
        """
        Download and parse the daily ZIP for a given symbol and UTC day.

        Params:
            symbol : str
                Asset symbol, e.g. "BTCUSDT".
            day_str : str
                UTC day formatted "YYYY-MM-DD".

        Returns:
            pd.DataFrame | None
                Minute-level OHLCV for that day, indexed by timestamp (UTC).
                Columns:
                    ["open","high","low","close","volume","n_trades"]
                Returns None if the file is not found or cannot be parsed.

        Remote file pattern:
            https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/{symbol}-1m-YYYY-MM-DD.zip

        The CSV inside the ZIP has rows of the form:
            open_time,
            open,high,low,close,volume,
            close_time,
            quote_asset_volume,
            number_of_trades,
            taker_buy_base_volume,
            taker_buy_quote_volume,
            ignore
        :contentReference[oaicite:1]{index=1}
        """
        base_url = (
            "https://data.binance.vision/data/spot/daily/klines/"
            f"{symbol}/1m/{symbol}-1m-{day_str}.zip"
        )

        try:
            r = requests.get(base_url, timeout=10)
        except Exception as e:
            logger.exception("Request error while fetching %s %s: %s", symbol, day_str, e)
            return None

        if r.status_code != 200:
            logger.warning("HTTP %s for %s on %s", r.status_code, symbol, day_str)
            return None

        # Unzip in memory
        try:
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            # We assume there is exactly one CSV inside, named like {symbol}-1m-YYYY-MM-DD.csv
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if len(csv_names) == 0:
                logger.error("No CSV found in ZIP for %s on %s", symbol, day_str)
                return None

            with zf.open(csv_names[0]) as f:
                df_raw = pd.read_csv(
                    f,
                    header=None,
                    dtype=str
                )
        except Exception as e:
            logger.exception("Error extracting/parsing ZIP for %s on %s: %s", symbol, day_str, e)
            return None

        # Assign expected columns according to Binance public data format. :contentReference[oaicite:2]{index=2}
        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "n_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        if df_raw.shape[1] != len(cols):
            logger.error("Unexpected column count for %s on %s (got %d)", symbol, day_str, df_raw.shape[1])
            return None
        df_raw.columns = cols

        # Convert timestamps.
        # Spot data timestamps are in ms historically, and in µs for 2025+.
        # We detect unit by magnitude. :contentReference[oaicite:3]{index=3}
        ts_raw = df_raw["open_time"].astype("int64")
        unit = "us" if ts_raw.iloc[0] > 10**14 else "ms"

        df_raw["timestamp"] = pd.to_datetime(ts_raw, unit=unit, utc=True)

        # Keep only useful columns (OHLCV + n_trades)
        df_day = pd.DataFrame({
            "open":   pd.to_numeric(df_raw["open"], errors="coerce"),
            "high":   pd.to_numeric(df_raw["high"], errors="coerce"),
            "low":    pd.to_numeric(df_raw["low"], errors="coerce"),
            "close":  pd.to_numeric(df_raw["close"], errors="coerce"),
            "volume": pd.to_numeric(df_raw["volume"], errors="coerce"),
            "n_trades": pd.to_numeric(df_raw["n_trades"], errors="coerce"),
            "timestamp": df_raw["timestamp"],
        })

        # Index on timestamp, sort
        df_day = (
            df_day
            .set_index("timestamp")
            .sort_index()
        )

        return df_day

    @staticmethod
    def _subset_range(df: pd.DataFrame,
                      start_ts: datetime,
                      end_ts: datetime) -> pd.DataFrame:
        """
        Restrict a DataFrame to the interval [start_ts, end_ts].

        Params:
            df : pd.DataFrame
                Full historical data, indexed by timestamp (UTC).
            start_ts : datetime
                Start timestamp (UTC).
            end_ts : datetime
                End timestamp (UTC).

        Returns:
            pd.DataFrame
                Sliced DataFrame with timestamps between start_ts and end_ts
                (inclusive).
        """
        if df.index.tz is None:
            logger.warning("DataFrame index is not timezone-aware. Expected UTC.")

        mask = (df.index >= start_ts) & (df.index <= end_ts)
        return df.loc[mask].copy()
