from typing import List
from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer, AutoConfig


class Chunker:
    """
    Chunker class to split text into overlapping token-based windows.
    """

    def __init__(self,
                 model: str,
                 special_reserve: int = 2,
                 default_window: int = 512,
                 stride: int = 96) -> None:
        """
        Params:
            model: str
                LLM Model used in the pipeline.
            special_reserve: int (default 2)
                Number of special tokens to reserve per chunk.
            default_window: int (default 512)
                Default window size if model config cannot be read.
            stride: int (default 96)
                Number of overlapping tokens between chunks.
        """
        hf_logging.set_verbosity_error()

        self.name = model
        self.default_window = default_window
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.window_total = self._infer_model_window_total()
        self.window_content = max(8, self.window_total - special_reserve)

    def _infer_model_window_total(self) -> int:
        """
        Try to infer the model's max token window from its config.
        First checks common config attributes, then falls back to tokenizer attribute.
        Otherwise returns default_window.

        Returns:
            int
                Max token window size of the model or default_window if not found.
        """
        try:
            cfg = AutoConfig.from_pretrained(self.name)
            for attr in ("max_position_embeddings", "n_positions", "seq_length", "model_max_length"):
                v = getattr(cfg, attr, None)
                if v and int(v) > 0:
                    return int(v)
        except Exception:
            pass

        try:
            ml = int(getattr(self.tokenizer, "model_max_length", self.default_window))
            if ml > 100_000:
                ml = self.default_window
            return ml
        except Exception:
            return self.default_window

    def chunk(self, text: str) -> List[List[int]]:
        """
        Chunk text into overlapping windows based on token counts :

        Params:
            text: str
                Text to chunk.

        Returns:
            List[List[int]]
                List of token-id chunks (each chunk is <= self.window_content tokens).
                No decoding back to text.
        """
        max_tokens = self.window_content
        stride = self.stride

        if not (0 <= stride < max_tokens):
            raise ValueError("Stride must be non-negative and less than max_tokens.")

        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return []

        out: List[List[int]] = []
        i = 0
        step = max_tokens - stride

        while i < len(ids):
            window_ids = ids[i : i + max_tokens]
            out.append(window_ids)

            if i + max_tokens >= len(ids):
                break

            i += step

        return out