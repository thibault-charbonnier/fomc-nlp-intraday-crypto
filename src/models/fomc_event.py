import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class FOMCEvent:
    """
    Represents a single FOMC event and its associated metadata.

    Params:
        meeting_date : datetime
            Date of the FOMC meeting.
        t_statement : datetime
            Timestamp (UTC) of the official FOMC statement release.
        t_pressconf : datetime
            Timestamp (UTC) of the start of the press conference / Q&A.
        score_stmt : float
            Hawkish/dovish score inferred from the statement.
        score_qa : float
            Hawkish/dovish score inferred from the Q&A / press conference.
        delta_score : float
            score_qa - score_stmt.
        reactions : Dict[str, pd.Series]
            Market reactions by asset. The key is the asset symbol
            (e.g. "BTCUSDT"), and the value is a pandas Series:
                - index: list of windows in minutes after t_statement (e.g. [1,2,5,10,30])
                - values: cumulative returns (in bps) for that horizon.
            This dictionary is populated later by the market processing step.
    """
    meeting_date: datetime
    t_statement: datetime
    t_pressconf: datetime
    score_stmt: float
    score_qa: float
    delta_score: float
    reactions: Dict[str, pd.Series] = field(default_factory=dict)
