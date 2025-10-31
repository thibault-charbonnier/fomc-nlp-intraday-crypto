import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import yaml
import pandas as pd
from src.sentiment.sentiment_scorer import SentimentScorer
from src.ingestor.statement_ingestor import StatementIngestor
from src.ingestor.pressconf_ingestor import PressConfIngestor
from models.transcript import Transcript
from src.tools.logging_config import get_logger
from models.fomc_event import FOMCEvent

logger = get_logger(__name__)


class FOMCPipeline:
    """
    Main pipeline for FOMC sentiment analysis.
    """

    def __init__(self, config_path: str):
        """
        Params:
            config_path: str
                Path of the yaml configuration file
        """
        self._read_config(config_path)

        self.ing_stmt = StatementIngestor(
            model=self.model, max_tokens=self.max_tokens, stride=self.stride
        )
        self.ing_pc = PressConfIngestor(
            model=self.model, max_tokens=self.max_tokens, stride=self.stride
        )

        self.scorer = SentimentScorer(model=self.model)

        # Prepare the file paths and the date for possible multi-threading 
        self.pairs: List[Tuple[str, Path, Path]] = []

    def _read_config(self, config_path: str) -> None:
        """
        Read the yaml configuration file.

        Params:
            config_path: str
                Path of the yaml configuration file
        """
        logger.info("Reading configuration file...")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.statement_dir = Path(self.cfg.get("statement_dir", "."))
        self.pressconf_dir = Path(self.cfg.get("pressconf_dir", "."))
        self.re_stmt = re.compile(self.cfg.get("statement_regex", r"(?i)monetary(\d{8})a1\.pdf$"))
        self.re_pc   = re.compile(self.cfg.get("pressconf_regex", r"(?i)FOMCpresconf(\d{8})\.pdf$"))
        self.date_fmt = self.cfg.get("date_fmt", "%Y%m%d")
        self.model = self.cfg.get("model")
        self.max_tokens  = int(self.cfg.get("max_tokens", 510))
        self.stride      = int(self.cfg.get("stride", 96))
        self.max_workers = int(self.cfg.get("max_workers", 4))
        self.output_path  = self.cfg.get("output_path", None)
        self.fomc_time = self.cfg.get("fomc_time", "14:00:00")
        self.qa_time   = self.cfg.get("qa_time", "14:30:00")
        self.sentiment_cache = self.cfg.get("sentiment_cache", "sentiment_cache.csv")

    # -------------------- Public method --------------------
    def run(self, use_cache: bool) -> pd.DataFrame:
        """
        Orchestrator of the full pipeline :
            - Get the transcripts for each date
            - Execution by date
            - Output gathering

        Params:
            use_cache: bool
                Whether to use cached results if available
        """
        logger.info("Pipeline run started")

        res: List[FOMCEvent] = []

        if use_cache:
            logger.info("Using cached results if available")
            try:
                df_cache = pd.read_csv(self.sentiment_cache)
                logger.info("Loaded cached results with %d entries", len(df_cache))
                res = [
                    FOMCEvent(
                        meeting_date=row["meeting_date"],
                        t_statement=datetime.strptime(row["t_statement"], "%Y-%m-%d %H:%M:%S"),
                        t_pressconf=datetime.strptime(row["t_pressconf"], "%Y-%m-%d %H:%M:%S"),
                        score_stmt=row["score_stmt"],
                        score_qa=row["score_qa"],
                        delta_score=row["delta_score"],
                    )
                    for _, row in df_cache.iterrows()
                ]

                return res
            
            except FileNotFoundError:
                logger.info("No cache file found, proceeding with full processing")

        self._get_files()
        logger.info("Found %d date pairs to process", len(self.pairs))
        
        for date_str, stmt_path, pc_path in self.pairs:
            try:
                event_res = self._process_date(date_str, stmt_path, pc_path)
                res.append(event_res)
            except Exception as e:
                logger.exception("Error processing date %s: %s", date_str, e)

        return res
    # --------------------  Private utilities --------------------
    def _get_files(self) -> List[Tuple[str, Path, Path]]:
        """
        Gets all the file pairs : Statement / Q&A with the corresponding date.

        Returns:
            List[Tuple[str, Path, Path]]
                List of file path pairs and their date
        """
        logger.info("Gathering statement and press conference files...")
        stmt_files = {}
        for p in self.statement_dir.glob("**/*.pdf"):
            m = self.re_stmt.search(p.name)
            if m:
                stmt_files[m.group(1)] = p.resolve()

        pc_files = {}
        for p in self.pressconf_dir.glob("**/*.pdf"):
            m = self.re_pc.search(p.name)
            if m:
                pc_files[m.group(1)] = p.resolve()

        common = sorted(set(stmt_files.keys()) & set(pc_files.keys()))
        self.pairs = [(d, stmt_files[d], pc_files[d]) for d in common]
        logger.info("Statement files: %d, PressConf files: %d", len(stmt_files), len(pc_files))

        return self.pairs

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

        logger.info("\tIngesting statement")
        t_stmt: Transcript = self.ing_stmt.ingest(pdf_path=str(stmt_path), meeting_date=dt)
        logger.info("\tIngesting press conference")
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
