from src.sentiment.nlp_pipeline import NLPPipeline
from src.tools.logging_config import get_logger
from src.market.market_loader import MarketDataLoader
from src.market.market_processor import MarketProcessor
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.models.fomc_event import FOMCEvent
from src.ingestor.fomc_data import FOMCDownloader

# Token HF : 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

if __name__ == "__main__":

    logger = get_logger("main")

    logger.info("Starting FOMC historical data sync...")

    dl = FOMCDownloader(
        start_year=2016,
        end_year=2025,
        sleep_sec=0.5,
    )
    dl.sync()

    logger.info("Loading sentiment data in cache...")
    df_sentiment = pd.read_csv("data/sentiment/sentiment_cache_20251101_115727.csv", parse_dates=['meeting_date'])

    processor = MarketProcessor(symbols=["BTCUSDT"],
                                windows=[1,2,5,10,30])
    events = processor.process_events(events=[FOMCEvent.from_dict(row.to_dict()) for _, row in df_sentiment.iterrows()],)

    logger.info("Saving sentiment data with reactions to CSV...")
    df = pd.DataFrame([e.to_dict() for e in events])
    df.to_csv(f"data/sentiment/sentiment_with_reactions.csv", index=False)

    # logger.info("Starting FOMC pipeline")
    # try:
    #     pipeline = NLPPipeline("config.yaml")
    #     pipeline.run(use_cache=True)
    #     logger.info("Pipeline finished successfully.")
    # except Exception as e:
    #     logger.exception("Pipeline failed: %s", e)
