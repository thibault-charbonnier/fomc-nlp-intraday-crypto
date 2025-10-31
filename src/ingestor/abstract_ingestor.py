import re
from typing import List
from datetime import datetime
from pdfminer.high_level import extract_text
from src.tools.chunker import Chunker
from models.transcript import Transcript
from abc import ABC


class BasePdfIngestor(ABC):
    """
    Abstract base class for PDF ingestors to process central bank transcripts.
    Provides common utilities and defines the ingestor interface.
    """

    def __init__(self,
                 model: str,
                 max_tokens: int = 510,
                 stride: int = 96) -> None:
        """
        Params:
            model : str
                LLM Model used in the pipeline.
            max_tokens : int (default 510)
                Default maximum number of tokens per chunk.
            stride : int (default 96)
                Default stride (overlap) between chunks.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.stride = stride

    # -------------------- Public method --------------------
    def ingest(self,
               pdf_path: str,
               meeting_date: datetime) -> Transcript:
        """
        Ingest a PDF file, process the ingestor pipeline and return a Transcript object.

        Params:
            pdf_path : str
                Path to the PDF file to ingest.
            meeting_date : datetime
                Date of the meeting.
        
        Returns:
            Transcript
                The processed transcript object.
        """
        raw_text = self._pdf_to_text(pdf_path)
        cleaned_text = self._clean_text(raw_text)
        chunks = self._chunk_text(cleaned_text)

        return Transcript(
            token_chunks=chunks,
            meeting_date=meeting_date,
        )

    # -------------------- Abstract methods (possible override) ------------------
    def _clean_text(self, text: str) -> str:
        """
        Clean the extracted text from the PDF.

        Params:
            text : str
                Raw extracted text.
        
        Returns:
            str
                Cleaned text.
        """
        text = self._basic_clean(text)
        return text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping windows based on model token counts.

        Params:
            text : str
                Text to chunk.

        Returns:
            List[str]
                List of chunked text segments.
        """
        chunker = Chunker(model=self.model, special_reserve=2, default_window=self.max_tokens, stride=self.stride)
        return chunker.chunk(text)
    
    # -------------------- Common utilities --------------------
    @staticmethod
    def _pdf_to_text(path: str) -> str:
        """
        Extract text from a PDF file located.

        Params:
            path : str
                Path to the PDF file.

        Returns:
            str
                Extracted text from the PDF.
        """
        return extract_text(path)

    @staticmethod
    def _basic_clean(text: str) -> str:
        t = text.replace("\r", "\n")
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        t = re.sub(r"(?m)^\s*-+\s*\d+\s*-+\s*$", "", t)
        return t.strip()
