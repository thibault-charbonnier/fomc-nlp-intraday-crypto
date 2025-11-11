import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
from ..sentiment.sentiment_scorer import SentimentScorer
from ..ingestor.statement_ingestor import StatementIngestor
from ..ingestor.pressconf_ingestor import PressConfIngestor
from ..models.transcript import Transcript
from ..tools.logging_config import get_logger
from ..models.fomc_event import FOMCEvent

logger = get_logger(__name__)


class NLPPipeline:
    """
    Main pipeline for FOMC sentiment analysis.

    This variant does NOT read a config YAML. All relevant parameters are
    provided to the constructor. The statement/pressconf directory path is
    fixed by default to the project's cache location (data/_cache/fomc_data)
    unless explicitly overridden.
    """

    def __init__(
        self,
        data_path: str | Path = "data/_cache/fomc_data",
        model: str = "gtfintechlab/FOMC-RoBERTa",
        max_tokens: int = 510,
        stride: int = 96,
        max_workers: int = 4,
        fomc_time: str = "14:00:00",
        qa_time: str = "14:30:00",
        sentiment_cache: str = "data/_cache/raw_sentiment/raw_sentiment_cache.csv",
        date_fmt: str = "%Y%m%d",
    ) -> None:
        """
        Params:
            data_path: str | Path (default "data/_cache/fomc_data")
                Path to the directory containing FOMC statement and press conference PDFs.
            model: str (default "gtfintechlab/FOMC-RoBERTa")
                Name of the NLP model to use for sentiment analysis.
            max_tokens: int (default 510)
                Maximum number of tokens per chunk for NLP processing.
            stride: int (default 96)
                Stride size for chunking text for NLP processing.
            max_workers: int (default 4)
                Maximum number of parallel workers for processing.
            output_path: str | None (default None)
                Path to save the output results. If None, results are not saved to a file.
            fomc_time: str (default "14:00:00")
                Time of the FOMC statement release.
            qa_time: str (default "14:30:00")
                Time of the FOMC press conference Q&A session.
            sentiment_cache: str (default "sentiment_cache.csv")
                Path to the sentiment cache file.
            date_fmt: str (default "%Y%m%d")
                Date format used in file names.
        """
        self.data_path = Path(data_path)

        # model / tokenization options
        self.model = model
        self.max_tokens = int(max_tokens)
        self.stride = int(stride)
        self.max_workers = int(max_workers)

        self.fomc_time = fomc_time
        self.qa_time = qa_time
        self.sentiment_cache = sentiment_cache
        self.date_fmt = date_fmt

        # pipeline components
        self.ing_stmt = StatementIngestor(
            model=self.model, max_tokens=self.max_tokens, stride=self.stride
        )
        self.ing_pc = PressConfIngestor(
            model=self.model, max_tokens=self.max_tokens, stride=self.stride
        )

        self.scorer = SentimentScorer(model=self.model)
        self.pairs: List[Tuple[str, Path, Path]] = []

    # -------------------- Public method --------------------
    def run(self, 
            use_cache: bool=True,
            save_cache: bool=True,
            save_cache_name: str="raw_sentiment_cache.csv") -> list[FOMCEvent]:
        """
        Orchestrator of the full pipeline :

        1. If use_cache is True:
            - Check if the cache file exists
            - If it does, load the results from the cache
            - If it doesn't, proceed with the normal processing

        2. Normal processing:
            - Get the transcripts for each date
            - Execution by date
            - Output gathering

        Params:
            use_cache: bool (default False)
                Whether to use cached results if available
            save_cache: bool (default False)
                Whether to save the results to cache
            cache_name: str (default f"sentiment_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                Name of the cache file

        Returns:
            list[FOMCEvent]
                List of FOMCEvent objects containing the results
        """
        logger.info("Pipeline run started")

        res: List[FOMCEvent] = []

        if use_cache:
            res = self._get_cache()
            if res:
                logger.info("Using cached results, skipping processing.")
                return res
            else:
                logger.info("No cached results found, proceeding with full processing.")

        self._get_files()
        logger.info("Found %d date pairs to process", len(self.pairs))
        
        for date_str, stmt_path, pc_path in self.pairs:
            try:
                event_res = self._process_date(date_str, stmt_path, pc_path)
                res.append(event_res)
            except Exception as e:
                logger.exception("Error processing date %s: %s", date_str, e)

        if save_cache:
            self._to_cache(res, save_cache_name)

        return res
    
    # --------------------  Private utilities --------------------
    def _process_date(self, date_str: str, stmt_path: Path, pc_path: Path) -> Dict:
        """
        Process the data for a specific date as follows:
            - Ingest the 2 text files
            - NLP scoring
            - Results formatting

        Params:
            date_str: str
                Date of the files
            stmt_path: Path
                Path of the statement file
            pc_path: Path
                Path of the press conferecne file

        Returns:
            Dict
                Formatted results
        """
        logger.info("Processing date %s", date_str)
        dt = datetime.strptime(date_str, self.date_fmt)

        logger.debug("\tIngesting statement")
        t_stmt: Transcript = self.ing_stmt.ingest(pdf_path=str(stmt_path), meeting_date=dt)
        logger.debug("\tIngesting press conference")
        t_pc: Transcript = self.ing_pc.ingest(pdf_path=str(pc_path), meeting_date=dt)

        logger.info("\tScoring statement")
        s_stmt = self.scorer.analyse(t_stmt)
        logger.info("\tScoring press conference")
        s_pc = self.scorer.analyse(t_pc)
        logger.info("\tDate processed: Statement score %.2f, PressConf score %.2f", s_stmt, s_pc)

        return FOMCEvent(
            meeting_date=dt.strftime("%d-%m-%Y"),
            t_statement=self.fomc_time,
            t_pressconf=self.qa_time,
            score_stmt=s_stmt,
            score_qa=s_pc,
            delta_score=s_pc - s_stmt,
        )

    def _get_files(self) -> List[Tuple[str, Path, Path]]:
        """
        Gets all the file pairs : Statement / Q&A with the corresponding date.

        Returns:
            List[Tuple[str, Path, Path]]
                List of file path pairs and their date
        """
        logger.info("Gathering statement and press conference files...")
        pairs: List[Tuple[str, Path, Path]] = []

        pattern_stmt = re.compile(r"^statement\.(pdf|html?)$", re.IGNORECASE)
        pattern_pc   = re.compile(r"^pressconf\.(pdf|html?)$", re.IGNORECASE)

        for meeting_dir in sorted(self.data_path.iterdir()):
            if not meeting_dir.is_dir():
                continue

            date_str = meeting_dir.name

            stmt_path = next((p.resolve() for p in meeting_dir.iterdir() if pattern_stmt.match(p.name)), None)
            pc_path   = next((p.resolve() for p in meeting_dir.iterdir() if pattern_pc.match(p.name)), None)

            if stmt_path and pc_path:
                pairs.append((date_str, stmt_path, pc_path))
            else:
                logger.debug(
                    "Skipping %s (statement=%s, pressconf=%s)",
                    date_str, bool(stmt_path), bool(pc_path)
                )
                
        self.pairs = pairs
        logger.info("Found %d valid meetings with both statement & pressconf", len(pairs))

        return pairs

    def _get_cache(self) -> List[FOMCEvent]:
        """
        Load cached sentiment scores if available.

        Returns:
            List[FOMCEvent]
                List of cached sentiment events
        """
        logger.info("Using cached results if available")
        try:
            df_cache = pd.read_csv(self.sentiment_cache)
            logger.info("Loaded cached results with %d entries", len(df_cache))
            res = [FOMCEvent.from_dict(row.to_dict()) for _, row in df_cache.iterrows()]

            return res

        except FileNotFoundError:
            return []

    def _to_cache(self, events: List[FOMCEvent], cache_name: str) -> None:
        """
        Save the sentiment events to the cache.

        Params:
            events: List[FOMCEvent]
                List of sentiment events to cache
            cache_name: str
                Name of the cache file
        """
        logger.info("Saving results to cache")
        df = pd.DataFrame([e.to_dict() for e in events])
        df.to_csv(f"data/_cache/raw_sentiment/{cache_name}", index=False)
