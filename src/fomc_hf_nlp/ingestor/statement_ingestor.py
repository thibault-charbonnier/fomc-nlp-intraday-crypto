import re
from .abstract_ingestor import BasePdfIngestor


class StatementIngestor(BasePdfIngestor):
    """
    Ingestor pour FOMC Statements.

    Specific rules for cleaning the text :
        - Remove the line 'For release at ...' at the top
        - Remove the date line (e.g. 'September 17, 2025') just after
        - Remove '(more)', 'Attachment', 'For media inquiries...'
        - Cut at '-0-' (end of statement)
    """

    def _clean_text(self, text: str) -> str:
        """
        Implement specific cleaning rules for FOMC Statements.

        Params:
            text : str
                Raw extracted text from the PDF.
        
        Returns:
            str
                Cleaned text.
        """
        t = self._basic_clean(text)
        t = re.sub(r'(?im)^\s*For release at[^\n]*\n', '', t, count=1)
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        t = re.sub(rf'(?m)^\s*{month}\s+\d{{1,2}},\s+\d{{4}}\s*\n', '', t, count=1)

        t = re.sub(r'(?im)^\s*\(more\)\s*$', '', t)
        t = re.sub(r'(?im)^\s*Attachment\s*$', '', t)
        t = re.sub(r'(?im)^\s*For media inquiries.*$', '', t)

        m = re.search(r'(?m)^\s*-\s*0\s*-\s*$', t)
        if m:
            t = t[:m.start()].rstrip()

        t = re.sub(r'\n{3,}', '\n\n', t).strip()
        return t
