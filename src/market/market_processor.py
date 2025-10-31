import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List
from src.tools.logging_config import get_logger
from src.market.market_loader import MarketDataLoader
from models.fomc_event import FOMCEvent

logger = get_logger(__name__)


class MarketProcessor:
    """
    Market processor for computing per-event returns.

    The processor:
        - Iterates over a list of FOMCEvent objects.
        - For each event and each symbol, loads minute data around t_statement.
        - Computes cumulative returns in basis points over specified horizons
          (1 min, 2 min, 5 min, 10 min, 30 min, ...).
        - Stores these results back into the FOMCEvent.reactions[symbol]
          as a pandas Series.
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

    def compute_reaction(self,
                         event: FOMCEvent,
                         symbol: str,
                         loader: MarketDataLoader,
                         force_fetch: bool = False) -> pd.Series:
        """
        Compute cumulative returns (in bps) for a single (event, symbol)
        over all configured windows.

        The anchor time is event.t_statement. For each horizon H (in minutes),
        we measure log-return from t_statement to t_statement + H.

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
        t0 = event.t_statement

        max_h = max(self.windows) if len(self.windows) > 0 else 0
        start_ts = t0 - timedelta(minutes=self.pre_margin)
        end_ts   = t0 + timedelta(minutes=max_h + self.post_margin)

        logger.debug("Loading %s data for event %s between %s and %s",
                     symbol, event.meeting_date, start_ts, end_ts)

        px = loader.load(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            force_fetch=force_fetch
        )
        px = px.sort_index()
        if "close" not in px.columns:
            logger.error("No 'close' column found for %s data.", symbol)
            raise KeyError("Expected 'close' column in market data.")

        def _cumret_bps(h: int) -> float:
            t_start = t0
            t_end   = t0 + timedelta(minutes=h)

            idx_start = px.index.get_loc(t_start, method="nearest")
            idx_end   = px.index.get_loc(t_end,   method="nearest")

            p0 = px.iloc[idx_start]["close"]
            p1 = px.iloc[idx_end]["close"]

            return float(np.log(p1 / p0) * 10000)

        data = {h: _cumret_bps(h) for h in self.windows}
        s = pd.Series(data, name=symbol)

        return s
