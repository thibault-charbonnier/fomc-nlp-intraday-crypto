from dataclasses import dataclass

@dataclass
class FOMCMeeting:
    """
    Lightweight container for one FOMC meeting reference.

    Params:
        date_str : str
            Meeting date as YYYYMMDD (Fed-style timestamp).
        url : str
            URL of the 'fomcpresconfYYYYMMDD.htm' page (press conf page).
    """
    date_str: str
    url: str