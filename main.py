# from src.sentiment.nlp_pipeline import FOMCPipeline
from src.tools.logging_config import get_logger
from src.market.market_loader import MarketDataLoader
from datetime import datetime

# Token HF : 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

if __name__ == "__main__":
    logger = get_logger("main")
    # logger.info("Starting FOMC pipeline")
    # try:
    #     pipeline = FOMCPipeline("config.yaml")
    #     pipeline.run()
    #     logger.info("Pipeline finished successfully.")
    # except Exception as e:
    #     logger.exception("Pipeline failed: %s", e)
    logger.info("Starting Market Data Loader test")
    loader = MarketDataLoader(data_dir="data/market_data")
    loader.load(symbol="BTCUSDT",
                start_ts=datetime.strptime("2023-01-01 13:55:00", "%Y-%m-%d %H:%M:%S"),
                end_ts=datetime.strptime("2023-01-02 15:00:00", "%Y-%m-%d %H:%M:%S"))
