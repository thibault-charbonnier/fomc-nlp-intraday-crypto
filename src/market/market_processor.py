import re
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import List
from src.tools.logging_config import get_logger
from src.market.market_loader import MarketDataLoader
from src.models.fomc_event import FOMCEvent

logger = get_logger(__name__)


class MarketProcessor:
    """
    Pipeline to process market data around FOMC events and compute reactions.

    Steps:
        1. For each FOMC event and each symbol, load market data around the event time.
        2. Compute cumulative returns over specified horizons after the event.
        3. Enrich each FOMCEvent with computed reactions.
    """

    def __init__(self,
                 symbols: List[str],
                 windows: List[int],
                 pre_margin: int = 10,
                 post_margin: int = 0) -> None:
        """
        Params:
            symbols : List[str]
                List of symbols to process, e.g. ["BTCUSDT","ETHUSDT"].
            windows : List[int]
                Horizons in minutes after t_statement for which we will compute
                cumulative returns. Example: [1,2,5,10,30].
            pre_margin : int (default 10)
                Number of minutes to load before t_statement when pulling market data,
                for safety/robustness.
            post_margin : int (default 0)
                Extra minutes to load beyond the largest horizon.
        """
        self.symbols = symbols
        self.windows = windows
        self.pre_margin = pre_margin
        self.post_margin = post_margin
        self.loader = MarketDataLoader("data/_cache/market_data")

    def process_events(self,
                       events: List[FOMCEvent],
                       force_fetch: bool = False,
                       save_computation: bool = True) -> List[FOMCEvent]:
        """
        Process a list of FOMC events to compute market reactions.
        Adds results into each event's .reactions attribute.

        Params:
            events : List[FOMCEvent]
                List of FOMC events to process.
            force_fetch : bool (default False)
                If True, bypass local cache when loading market data.
            save_computation : bool (default True)
                If True, save computed reactions to CSV files in cache.
        """
        logger.info("Computing market reactions to FOMC events...")

        for event in events:
            event.reactions = {}

            for symbol in self.symbols:
                logger.info(
                    "Computing reactions for FOMC event on %s for crypto %s...",
                    event.meeting_date, symbol
                )

                reaction = self._compute_reaction(
                    event=event,
                    symbol=symbol,
                    loader=self.loader,
                    force_fetch=force_fetch
                )
                event.reactions[symbol] = reaction

        if save_computation:
            self._save_events(events)

        return events
    
    def _compute_reaction(self,
                          event: FOMCEvent,
                          symbol: str,
                          loader: MarketDataLoader,
                          force_fetch: bool = False) -> pd.Series:
        """
        Compute cumulative returns (in bps) for a single (event, symbol)
        over all configured windows.

        Params:
            event : FOMCEvent
                FOMC event containing timestamps and scores.
            symbol : str
                Asset symbol, e.g. "BTCUSDT".
            loader : MarketDataLoader
                Loader to access raw market data.
            force_fetch : bool
                If True, request loader to bypass local cache.

        Returns:
            pd.Series
                Index: horizon in minutes (int).
                Values: cumulative return in basis points over [0 -> H].
        """
        if isinstance(event.meeting_date, (pd.Timestamp, datetime)):
            t0 = pd.to_datetime(event.meeting_date, utc=True)
        else:
            t0 = pd.to_datetime(event.meeting_date,
                                format="%d-%m-%Y",
                                utc=True)

        if isinstance(event.t_statement, (pd.Timestamp, datetime)):
            ts_time = pd.to_datetime(event.t_statement)
            t0 = t0.replace(
                hour=ts_time.hour,
                minute=ts_time.minute,
                second=ts_time.second,
                microsecond=0
            )
        else:
            h, m, s = str(event.t_statement).split(":")
            t0 = t0.replace(
                hour=int(h),
                minute=int(m),
                second=int(s),
                microsecond=0
            )

        max_h = max(self.windows) if self.windows else 0
        start_ts = t0 - timedelta(minutes=self.pre_margin)
        end_ts   = t0 + timedelta(minutes=max_h + self.post_margin)

        day_str = t0.strftime("%Y-%m-%d")
        px = None

        if not force_fetch:
            try:
                day_cache = loader._load_day_cache(symbol, day_str)
            except Exception:
                day_cache = None

            if day_cache is not None and not day_cache.empty:
                if "close" in day_cache.columns:
                    px = day_cache.loc[start_ts:end_ts].copy()

        if px is None or px.empty:
            px = loader.load(
                symbol=symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                force_fetch=force_fetch,
                save_day_cache=True
            )

        if px is None or px.empty:
            logger.error("No market data available for %s around %s", symbol, t0)
            return pd.Series({h: np.nan for h in self.windows}, name=symbol)

        px = px.sort_index()
        px = px.resample("1min").ffill()

        def _cumret_bps(h: int) -> float:
            t_start = t0
            t_end   = t0 + timedelta(minutes=h)

            if t_start not in px.index or t_end not in px.index:
                return float("nan")

            p0 = px.loc[t_start, "close"]
            p1 = px.loc[t_end,   "close"]

            return float(np.log(p1 / p0) * 10000)

        data = {h: _cumret_bps(h) for h in self.windows}
        s = pd.Series(data, name=symbol)

        return s

    def _save_events(self,
                    events: List[FOMCEvent],
                    filepath: str = "data/_cache/full_sentiment/full_sentiment_cache.csv") -> None:
        """
        Save processed events with reactions to a CSV file.

        Params:
            events : List[FOMCEvent]
                List of processed FOMC events.
            filepath : str
                Path to save the CSV file.
        """
        rows = []
        for e in events:
            base = e.to_dict() if hasattr(e, "to_dict") else vars(e).copy()
            reactions = base.pop("reactions", None)
            if reactions is None:
                reactions = getattr(e, "reactions", {})

            row = dict(base)

            for symbol, series in (reactions or {}).items():
                for i in range(len(self.windows)):
                    col = f"{symbol}_{int(self.windows[i])}m"
                    val = series[i]
                    row[col] = float(val) if pd.notna(val) else np.nan

            rows.append(row)

        pd.DataFrame(rows).to_csv(filepath, index=False)

