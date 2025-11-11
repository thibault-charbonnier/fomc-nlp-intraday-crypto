import torch
import torch.nn.functional as F
from typing import Any, List
from numpy import mean
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging
from time import time
from ..tools.logging_config import get_logger

logger = get_logger(__name__)


class SentimentScorer:
    """
    NLP Sentiment Scorer Pipeline.
    """

    def __init__(self, model: str) -> None:
        """
        Params:
            model : str
                LLM Model used in the pipeline.
        """
        hf_logging.set_verbosity_error()

        self.model_name = model
        self._load_pretrained_model()

        ml = getattr(self.tokenizer, "model_max_length", 512)
        if ml is None or ml > 100_000:
            ml = 512
        self.max_len = int(ml)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def _load_pretrained_model(self) -> None:
        """
        Connect to the selected pretrained model via Hugging Face,
        load the model and the associated tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    # -------------------- Public method --------------------
    def analyse(self, transcript, batch_size: int = 8) -> float:
        """
        Analyse the sentiment of a transcript, chunk by chunk, based on token id chunks.
        Uses mini-batching to speed up large transcripts (press conferences).

        Convention:
            0   = very dovish
            5   = neutral
            10  = very hawkish

        Params: 
            transcript :
                Transcript object with token_chunks: List[List[int]]
            batch_size : int (default 32)
                Number of chunks to score per forward pass.

        Returns:
            float
                Mean dovish score (/10) across all chunks.
        """
        chunks = transcript.token_chunks
        if not chunks:
            return 5.0

        scores: List[float] = []
        
        timer = time()
        for start in range(0, len(chunks), batch_size):
            batch_tokens = chunks[start : start + batch_size]

            encoded_batch = self._encode_batch(batch_tokens)

            with torch.no_grad():
                logits = self.model(**encoded_batch).logits

            probs = F.softmax(logits, dim=-1)
            p_dovish = probs[:, 0]
            p_hawkish = probs[:, 1]
            batch_scores = 5.0 + 5.0 * (p_hawkish - p_dovish)

            scores.extend(batch_scores.cpu().tolist())

        elapsed = time() - timer
        logger.info("\t\tNumber of chunks : %d | computed in %.2f seconds", len(transcript.token_chunks), elapsed)
        
        return mean(scores)

    # -------------------- Private utilities --------------------
    def _encode_single_ids(self, token_ids: List[int]) -> dict:
        """
        Prépare un seul chunk (liste d'IDs) pour le modèle :
            - ajoute les tokens spéciaux
            - tronque à self.max_len
        Ne retourne PAS encore de tenseurs alignés batchés/paddés.
        """
        prepared = self.tokenizer.prepare_for_model(
            token_ids,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None
        )
        return prepared

    def _encode_batch(self, batch_token_ids: List[List[int]]) -> dict:
        """
        Encode et pad un batch de chunks tokenisés pour pouvoir
        les passer en une seule fois au modèle.

        Params:
            batch_token_ids : List[List[int]]
                Liste de chunks (chaque chunk = liste d'IDs).

        Returns:
            dict[str, torch.Tensor]
                input_ids, attention_mask, etc. tous en shape (B, L)
                et déjà déplacés sur self.device.
        """
        prepared_list = [self._encode_single_ids(toks) for toks in batch_token_ids]

        padded = self.tokenizer.pad(
            prepared_list,
            padding=True,
            return_tensors="pt"
        )

        for k, v in padded.items():
            padded[k] = v.to(self.device)

        return padded
