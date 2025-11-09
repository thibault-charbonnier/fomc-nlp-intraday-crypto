import io
import time
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from src.tools.logging_config import get_logger

logger = get_logger(__name__)


class MarketDataLoader:
    """
    Market data loader for crypto assets (e.g. BTCUSDT).
    Responsible for returning 1-minute spot data over a given time range.

    Notes:
        - Primary source: Binance Vision (per-day zipped CSVs)
        - Fallback: native exchange APIs with deep 1m history:
            1) Bitstamp /v2/ohlc/{pair}
            2) Coinbase Exchange /products/{product_id}/candles
        - Output schema is kept identical across sources:
            index = UTC minute ("timestamp")
            columns = ["open","high","low","close","volume","n_trades"]
              (n_trades is NaN for fallbacks that don't provide it)
    """

    def __init__(self, data_dir: str="data/_cache/market_data") -> None:
        self.data_dir = Path(data_dir).resolve()

    # -------------------- Public method --------------------
    def load(self,
             symbol: str,
             start_ts: datetime,
             end_ts: datetime,
             force_fetch: bool = False,
             save_day_cache: bool = True) -> pd.DataFrame:
        """
        Load 1-minute spot market data for a given symbol over [start_ts, end_ts].
        """
        logger.debug(
            "Loading market data for %s between %s and %s (force_fetch=%s)",
            symbol, start_ts, end_ts, force_fetch
        )
        start_ts = pd.to_datetime(start_ts, utc=True)
        end_ts = pd.to_datetime(end_ts, utc=True)

        df_full = self._load_local_cache(symbol)
        if df_full is not None:
            logger.debug("Local cache loaded for %s with %d rows", symbol, len(df_full))
        else:
            logger.debug("No local cache for %s", symbol)

        df_remote = self._fetch_remote(symbol, start_ts, end_ts, force_fetch=force_fetch)

        if df_full is None:
            df_merged = df_remote
        else:
            if df_remote is not None and not df_remote.empty:
                df_merged = (
                    pd.concat([df_full, df_remote])
                    .sort_index()
                    .drop_duplicates(subset=None, keep="last")
                )
            else:
                df_merged = df_full

        if df_merged is not None and not df_merged.empty:
            self._write_local_cache(symbol, df_merged)

            if save_day_cache:
                try:
                    for day, df_day in df_merged.groupby(df_merged.index.date):
                        day_str = pd.to_datetime(day).strftime("%Y-%m-%d")
                        self._write_day_cache(symbol, day_str, df_day)
                except Exception:
                    logger.exception("Failed to write per-day cache for %s", symbol)

        if df_merged is None or df_merged.empty:
            logger.error("No data available for %s in requested range", symbol)
            return pd.DataFrame()

        df_slice = self._subset_range(df_merged, start_ts, end_ts)
        return df_slice

    # -------------------- Per-day cache helpers --------------------
    def _day_cache_path(self, symbol: str, day_str: str) -> Path:
        base = self.data_dir / symbol
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{day_str}.parquet"

    def _load_day_cache(self, symbol: str, day_str: str) -> Optional[pd.DataFrame]:
        path = self._day_cache_path(symbol, day_str)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.exception("Failed to read day cache %s", path)
            return None

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")

        df = df.sort_index()

        if "close" not in df.columns:
            logger.error("Day cache %s missing 'close' column", path)
            return None

        return df

    def _write_day_cache(self, symbol: str, day_str: str, df: pd.DataFrame) -> None:
        path = self._day_cache_path(symbol, day_str)
        try:
            df_to_save = df.copy()
            if df_to_save.index.name is None:
                df_to_save.index.name = "timestamp"
            df_to_save.reset_index().to_parquet(path, index=False)
        except Exception:
            logger.exception("Failed to write day cache %s", path)

    # -------------------- Private utilities --------------------
    def _load_local_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self.data_dir / f"{symbol}_1m.parquet"

        if not path.exists():
            logger.warning("No local cache found for %s at %s", symbol, path)
            return None

        logger.debug("Loading local cache for %s from %s", symbol, path)
        df = pd.read_parquet(path)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")

        df = df.sort_index()

        if "close" not in df.columns:
            logger.error("Cached data for %s is missing 'close' column", symbol)
            return None

        return df

    def _write_local_cache(self,
                           symbol: str,
                           df: pd.DataFrame) -> None:
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
        If Binance returns nothing, try native-exchange fallback (Bitstamp → Coinbase).
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
            logger.error("Binance returned no data for %s in requested range. Trying fallback (Bitstamp/Coinbase)...", symbol)
            df_fb = self._fetch_remote_fallback_cryptowatch(symbol, start_ts, end_ts)
            if df_fb is not None and not df_fb.empty:
                return df_fb
            logger.error("Fallback also empty for %s", symbol)
            return None

        df_all = pd.concat(all_days).sort_index().drop_duplicates(subset=None, keep="last")
        return df_all

    def _download_daily_klines(self,
                               symbol: str,
                               day_str: str) -> Optional[pd.DataFrame]:
        """
        Download and parse the daily ZIP for a given symbol and UTC day from Binance Vision.
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

        # Convert timestamps (ms historically, µs for 2025+).
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

        df_day = df_day.set_index("timestamp").sort_index()
        return df_day

    @staticmethod
    def _subset_range(df: pd.DataFrame,
                      start_ts: datetime,
                      end_ts: datetime) -> pd.DataFrame:
        """
        Restrict a DataFrame to the interval [start_ts, end_ts] (inclusive).
        """
        if df.index.tz is None:
            logger.warning("DataFrame index is not timezone-aware. Expected UTC.")
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        return df.loc[mask].copy()

    # -------------------- Fallback: native exchanges (Bitstamp → Coinbase) --------------------
    def _fetch_remote_fallback_cryptowatch(self,
                                           symbol: str,
                                           start_ts: datetime,
                                           end_ts: datetime) -> Optional[pd.DataFrame]:
        """
        Replacement for the old 'Cryptowatch' fallback (now disabled).
        Tries Bitstamp first, then Coinbase Exchange.
        """
        bitstamp_pair, coinbase_product = self._map_to_usd_pairs(symbol)

        # 1) Bitstamp
        df_bs = self._download_minute_ohlc_bitstamp(bitstamp_pair, start_ts, end_ts)
        if df_bs is not None and not df_bs.empty:
            logger.info("Fallback successful via Bitstamp (%s)", bitstamp_pair)
            return df_bs

        # 2) Coinbase
        df_cb = self._download_minute_ohlc_coinbase(coinbase_product, start_ts, end_ts)
        if df_cb is not None and not df_cb.empty:
            logger.info("Fallback successful via Coinbase (%s)", coinbase_product)
            return df_cb

        return None

    def _download_minute_ohlc_bitstamp(self,
                                       pair: str,
                                       start_ts: datetime,
                                       end_ts: datetime) -> Optional[pd.DataFrame]:
        """
        Bitstamp: GET https://www.bitstamp.net/api/v2/ohlc/{pair}/?step=60&limit=1000&start=...&end=...
        Iterates 1000-minute windows to cover [start_ts, end_ts].
        """
        url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
        start_ts = pd.to_datetime(start_ts, utc=True)
        end_ts   = pd.to_datetime(end_ts,   utc=True)

        parts: List[pd.DataFrame] = []
        cur_start = int(start_ts.timestamp())
        stop      = int(end_ts.timestamp())

        while cur_start <= stop:
            # 1000 candles * 60s = ~16h40 per request
            cur_end = min(cur_start + 1000*60, stop)
            params = {"step": 60, "limit": 1000, "start": cur_start, "end": cur_end}
            try:
                r = requests.get(url, params=params, timeout=15)
            except Exception as e:
                logger.warning("Bitstamp request error on %s: %s", pair, e)
                break
            if r.status_code != 200:
                logger.warning("Bitstamp HTTP %s on %s (%s→%s)", r.status_code, pair, cur_start, cur_end)
                cur_start = cur_end + 60
                continue

            try:
                data = r.json().get("data", {}).get("ohlc", [])
            except Exception:
                data = []

            if not data:
                cur_start = cur_end + 60
                continue

            df = pd.DataFrame(data)
            # fields: timestamp, open, high, low, close, volume (strings)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
            for c in ("open","high","low","close","volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
            if not df.empty:
                parts.append(df[["timestamp","open","high","low","close","volume"]])

            # advance 1 minute after the last returned bar
            cur_start = int(df["timestamp"].max().timestamp()) + 60

            # simple rate-limit politeness
            time.sleep(0.12)

        if not parts:
            return None

        out = (pd.concat(parts)
                 .drop_duplicates(subset=["timestamp"])
                 .set_index("timestamp")
                 .sort_index())
        out["n_trades"] = np.nan
        return out

    def _download_minute_ohlc_coinbase(self,
                                       product_id: str,
                                       start_ts: datetime,
                                       end_ts: datetime) -> Optional[pd.DataFrame]:
        """
        Coinbase Exchange: GET /products/{product_id}/candles?granularity=60&start=...&end=...
        Max 300 candles per request → iterate 300-minute windows.
        Response rows: [time, low, high, open, close, volume] (newest→oldest).
        """
        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        start_ts = pd.to_datetime(start_ts, utc=True)
        end_ts   = pd.to_datetime(end_ts,   utc=True)

        parts: List[pd.DataFrame] = []
        step_sec = 300 * 60
        cur_start = int(start_ts.timestamp())
        stop      = int(end_ts.timestamp())

        while cur_start <= stop:
            cur_end = min(cur_start + step_sec, stop)
            params = {
                "granularity": 60,
                "start": pd.to_datetime(cur_start, unit="s", utc=True).isoformat().replace("+00:00","Z"),
                "end":   pd.to_datetime(cur_end,   unit="s", utc=True).isoformat().replace("+00:00","Z"),
            }
            headers = {"User-Agent": "mdloader/1.0"}
            try:
                r = requests.get(url, params=params, timeout=15, headers=headers)
            except Exception as e:
                logger.warning("Coinbase request error on %s: %s", product_id, e)
                break
            if r.status_code != 200:
                logger.warning("Coinbase HTTP %s on %s (%s→%s)", r.status_code, product_id, cur_start, cur_end)
                cur_start = cur_end + 60
                continue

            try:
                arr = r.json()  # list of lists (newest -> oldest)
            except Exception:
                arr = []

            if not arr:
                cur_start = cur_end + 60
                continue

            a = np.array(arr, dtype=float)
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(a[:,0].astype(int), unit="s", utc=True),
                "open":  a[:,3],
                "high":  a[:,2],
                "low":   a[:,1],
                "close": a[:,4],
                "volume":a[:,5],
            })
            df = df.sort_values("timestamp")
            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
            if not df.empty:
                parts.append(df)

            cur_start = cur_end + 60
            time.sleep(0.12)

        if not parts:
            return None

        out = (pd.concat(parts)
                 .drop_duplicates(subset=["timestamp"])
                 .set_index("timestamp")
                 .sort_index())
        out["n_trades"] = np.nan
        return out

    # -------------------- Helpers --------------------
    def _map_to_usd_pairs(self, symbol: str) -> Tuple[str, str]:
        """
        Map 'BTCUSDT' -> ('btcusd' for Bitstamp, 'BTC-USD' for Coinbase).
        Stablecoin quotes mapped to USD to maximize history depth.
        """
        s = symbol.upper()
        for q in ("USDT","USDC","BUSD","FDUSD","USD"):
            if s.endswith(q):
                base = s[:-len(q)]
                break
        else:
            base = s[:-3]
        base = base.upper()
        bitstamp_pair = f"{base.lower()}usd"
        coinbase_prod = f"{base}-USD"
        return bitstamp_pair, coinbase_prod
