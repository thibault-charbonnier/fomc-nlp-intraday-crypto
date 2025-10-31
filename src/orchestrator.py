import pandas as pd
from datetime import datetime
from typing import List, Tuple
from tqdm.auto import tqdm
from src.tools.logging_config import get_logger
from src.sentiment.nlp_pipeline import FOMCPipeline
from src.market.market_loader import MarketDataLoader
from src.market.market_processor import MarketProcessor
from src.models.fomc_event import FOMCEvent

logger = get_logger(__name__)


class Orchestrator:
    """
    High-level pipeline orchestrator.

    This class coordinates:
        - NLP scoring of FOMC communications (statement / press conf)
        - Market data extraction and reaction computation
        - Aggregation into a final, analysis-ready DataFrame
    """

    def __init__(self,
                 use_sentiment_cache: bool,
                 nlp_config_path: str,
                 data_dir: str,
                 symbols: List[str],
                 windows: List[int],
                 pre_margin: int = 10,
                 post_margin: int = 0,
                 force_fetch: bool = False) -> None:
        """
        Params:
            use_sentiment_cache: bool
                Whether to use cached sentiment results if available
            nlp_config_path : str
                Path to the YAML configuration file for the FOMCPipeline
                (same type of config file you already use to locate PDFs,
                regex, model name, etc.).
            data_dir : str
                Base directory where market data (1m candles) will be cached.
            symbols : List[str]
                List of market symbols to process (e.g. ["BTCUSDT","ETHUSDT"]).
            windows : List[int]
                Horizons in minutes after t_statement for which we compute
                cumulative returns (e.g. [1,2,5,10,30]).
            pre_margin : int (default 10)
                Minutes of data to load before t_statement for safety.
            post_margin : int (default 0)
                Extra minutes beyond max(window) to load.
            force_fetch : bool (default False)
                If True, always pull from remote source (Binance Vision)
                even if cached locally.
        """
        self._use_sentiment_cache = use_sentiment_cache
        self._nlp_config_path = nlp_config_path
        self._data_dir = data_dir
        self._symbols = symbols
        self._windows = windows
        self._pre_margin = pre_margin
        self._post_margin = post_margin
        self._force_fetch = force_fetch

        self._fomc_pipeline = FOMCPipeline(config_path=self._nlp_config_path)
        self._loader = MarketDataLoader(data_dir=self._data_dir)
        self._processor = MarketProcessor(
            symbols=self._symbols,
            windows=self._windows,
            pre_margin=self._pre_margin,
            post_margin=self._post_margin
        )

    # -------------------- Public method --------------------
    def run(self) -> Tuple[List[FOMCEvent], pd.DataFrame]:
        """
        Run the full pipeline:
            1. Run the NLP pipeline to build base FOMCEvent objects
               (scores, timestamps, etc.).
            2. For each event, for each symbol, compute market reactions
               over all configured horizons.
            3. Attach these reactions back to each FOMCEvent.
            4. Flatten all events into an analysis-ready DataFrame.

        Returns:
            Tuple[List[FOMCEvent], pd.DataFrame]
                - enriched_events : List[FOMCEvent]
                    Same objects returned by the NLP pipeline but updated with
                    event.reactions[symbol] = pandas.Series of returns by horizon.
                - df_flat : pd.DataFrame
                    Long-format table gathering all events / symbols / horizons,
                    ready to be fed to the analysis functions.
        """
        logger.info("Starting main orchestrator run...")

        logger.info("Running NLP pipeline...")
        events = self._fomc_pipeline.run(use_cache=self._use_sentiment_cache)
        logger.info("NLP pipeline returned %d events", len(events))

        logger.info("Computing market reactions for %d symbols", len(self._symbols))

        for ev in tqdm(events, desc="Processing FOMC events"):
            logger.debug("Processing market data for event on %s", ev.meeting_date)

            for symbol in self._symbols:
                s = self._processor.compute_reaction(
                    event=ev,
                    symbol=symbol,
                    loader=self._loader,
                    force_fetch=self._force_fetch
                )

                # Attach to event
                ev.reactions[symbol] = s

        # Step 3. Flatten for analysis
        logger.info("Flattening enriched events into analysis DataFrame...")
        df_flat = self._flatten_events(events)
        logger.info("Flattened DataFrame shape: %s rows, %s cols", df_flat.shape[0], df_flat.shape[1])

        return events, df_flat

    # -------------------- Private utilities --------------------
    @staticmethod
    def _flatten_events(events: List[FOMCEvent]) -> pd.DataFrame:
        """
        Convert a list of enriched FOMCEvent objects into a long-format DataFrame.

        For each event and each symbol, we take the stored pandas Series of
        cumulative returns by horizon and expand it into rows:
            meeting_date, symbol, horizon_min, ret_bps,
            score_stmt, score_qa, delta_score

        Params:
            events : List[FOMCEvent]
                Enriched events (i.e. after market reactions have been computed).

        Returns:
            pd.DataFrame
                Long-format table, one row per (event, symbol, horizon).
        """
        rows = []

        for ev in events:
            # Defensive cast for readability in downstream analysis
            # meeting_date_str: str (ISO, yyyy-mm-dd)
            if isinstance(ev.meeting_date, datetime):
                meeting_date_str = ev.meeting_date.date().isoformat()
            else:
                # if it's already a date
                meeting_date_str = str(ev.meeting_date)

            for symbol, series_ret in ev.reactions.items():
                # series_ret: index = horizon in minutes (int),
                #             value = cumulative log-return in bps
                for horizon_min, ret_bps in series_ret.items():
                    rows.append({
                        "meeting_date": meeting_date_str,
                        "symbol": symbol,
                        "horizon_min": int(horizon_min),
                        "ret_bps": float(ret_bps),
                        "score_stmt": float(ev.score_stmt),
                        "score_qa": float(ev.score_qa),
                        "delta_score": float(ev.delta_score),
                        "t_statement": ev.t_statement,
                        "t_pressconf": ev.t_pressconf,
                    })

        if len(rows) == 0:
            return pd.DataFrame(columns=[
                "meeting_date",
                "symbol",
                "horizon_min",
                "ret_bps",
                "score_stmt",
                "score_qa",
                "delta_score",
                "t_statement",
                "t_pressconf",
            ])

        df = pd.DataFrame(rows)

        # Basic cleaning / sorting for convenience
        df = df.sort_values(
            by=["meeting_date", "symbol", "horizon_min"]
        ).reset_index(drop=True)

        return df
