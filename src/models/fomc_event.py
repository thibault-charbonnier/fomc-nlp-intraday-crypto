import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class FOMCEvent:
    """
    Represents a single FOMC event and its associated metadata.

    Params:
        meeting_date : datetime
            Date of the FOMC meeting.
        t_statement : str
            Time of the official FOMC statement release.
        t_pressconf : str
            Time of the press conference / Q&A start.
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
    meeting_date: Optional[datetime]
    t_statement: Optional[str]
    t_pressconf: Optional[str]
    score_stmt: Optional[float]
    score_qa: Optional[float]
    delta_score: Optional[float]
    reactions: Optional[Dict[str, pd.Series]] = field(default_factory=dict)

    def from_dict(data: Dict[str, any]) -> "FOMCEvent":
        """
        Create an FOMCEvent instance from a dictionary.

        Params:
            data : Dict[str, any]
                Dictionary containing the FOMC event data.
        
        Returns:
            FOMCEvent
                An instance of FOMCEvent populated with the provided data.
        """
        return FOMCEvent(
            meeting_date=data.get("meeting_date"),
            t_statement=data.get("t_statement"),
            t_pressconf=data.get("t_pressconf"),
            score_stmt=data.get("score_stmt"),
            score_qa=data.get("score_qa"),
            delta_score=data.get("delta_score"),
            reactions=data.get("reactions", {})
        )
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert the FOMCEvent to a dictionary representation.

        Returns:
            Dict[str, any]
                Dictionary representation of the FOMCEvent
        """
        return {
            "meeting_date": self.meeting_date,
            "t_statement": self.t_statement,
            "t_pressconf": self.t_pressconf,
            "score_stmt": self.score_stmt,
            "score_qa": self.score_qa,
            "delta_score": self.delta_score,
            "reactions": {k: v.tolist() for k, v in self.reactions.items()},
        }