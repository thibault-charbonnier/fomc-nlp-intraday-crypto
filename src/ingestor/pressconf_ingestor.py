import re
from .abstract_ingestor import BasePdfIngestor

class PressConfIngestor(BasePdfIngestor):
    """
    Ingestor for FOMC Press Conferences.

    Specific rules for cleaning the text :
        - Remove common headers/footers in transcripts
        - Light diarization into segments (speaker, text)
        - Keep only segments from the Chair (CHAIR..., CHAIRMAN...) to avoid biais in journalist questions
        - Concatenate properly (double line break between blocks)
    """

    _SPEAKER_RE = re.compile(r'(?m)^(?P<name>[A-Z][A-Z\s\.\-\'()]{2,})(?:\:|\.)\s*')

    def _clean_text(self, text: str) -> str:
        """
        Implement specific cleaning rules for FOMC Q&A Conference transcripts.

        Params:
            text : str
                Raw extracted text from the PDF.
        
        Returns:
            str
                Cleaned text.
        """
        t = self._basic_clean(text)

        t = re.sub(r'(?m)^\s*Transcript of .*?Press Conference.*$', '', t)
        t = re.sub(r'(?m)^\s*Press Conference.*$', '', t)
        t = re.sub(r'(?m)^\s*Page\s+\d+\s+of\s+\d+\s*$', '', t)
        t = re.sub(r'(?m)^\s*FINAL\s*$', '', t)

        segments = []
        cur_name = None
        cur_buf = []
        for line in t.splitlines():
            m = self._SPEAKER_RE.match(line)
            if m:
                if cur_name is not None:
                    segments.append((cur_name, "\n".join(cur_buf).strip()))
                cur_name = m.group("name").strip().replace("  ", " ")
                cur_buf = [line[m.end():].strip()]
            else:
                if cur_name is None:
                    cur_name = "CHAIR"
                cur_buf.append(line.strip())

        if cur_name is not None:
            segments.append((cur_name, "\n".join(cur_buf).strip()))

        def is_chair(name: str) -> bool:
            n = re.sub(r"[^A-Z ]", "", name.upper()).strip()
            return (n == "CHAIR" or n.startswith("CHAIR ") or n.startswith("CHAIRMAN "))

        chair_blocks = [seg_text for spk, seg_text in segments if is_chair(spk) and seg_text]

        text_concatenated = "\n\n".join(chair_blocks).strip()
        return text_concatenated
