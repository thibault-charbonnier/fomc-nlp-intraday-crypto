import logging, sys
from typing import List, Any
import pandas as pd
from src.models.fomc_event import FOMCEvent
from src.tools.logging_config import get_logger
import random
import ast
import json

def setup_notebook_logger(level: int = logging.INFO):
    """
    Setup the right configuration for Jupyter notebook logging.

    Params:
        level: int
            Logging level to set (default: logging.INFO).
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)


    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)

    for name, logger_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.handlers = []
            logger_obj.propagate = True
            logger_obj.setLevel(level)

def load_cached_sentiments(cache_path: str = "data/_cache/raw_sentiment/raw_sentiment_cache.csv") -> List[FOMCEvent]:
    """
    Load cached sentiment data from a CSV file.

    Params:
        cache_path: str
            Path to the cached sentiment CSV file.
    
    Returns:
        List[FOMCEvent]
            List of cached sentiment events
    """
    df = pd.read_csv(cache_path)
    return [FOMCEvent.from_dict(row.to_dict()) for _, row in df.iterrows()]

def print_event_reactions() -> None:
    """
    Print a few exemples of event reactions for visual inspection from pre_cached sentiments with reactions.
    """
    events = load_cached_sentiments("data/_cache/full_sentiment/full_sentiment_cache.csv")
    n = min(3, len(events))
    idxs = random.sample(range(len(events)), k=n)
    for idx in idxs:
        event = events[idx]
        print(f"Meeting date on {event.meeting_date}:")
        print(f"NLP scoring for statement file : {event.score_stmt}")
        print(f"NLP scoring for press conference transcript : {event.score_qa}")
        print(event.reactions)
        # for symbol, reactions in event.reactions.items():
        #     print(f"  Reactions for {symbol}:")
        #     print(reactions)
            # for window, ret in reactions.items():
            #     print(f"    +{window} min: {ret:.2f} bps")
        print("-" * 40)
    