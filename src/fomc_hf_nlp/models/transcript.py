from datetime import datetime
from typing import List
from dataclasses import dataclass

@dataclass
class Transcript:
    """
    Represents a processed transcript.

    Params:
        token_chunks : List[List[int]]
            List of token id chunks already truncated to model window.
            Each element is a list[int] of token ids.
        meeting_date : datetime
            Date of the meeting.
    """
    token_chunks: List[List[int]]
    meeting_date: datetime
