"""Frozen, zero-shot CLIP concept-bottleneck extractor (Section 4.1).

The concept space the verifier operates in. For each image we compute the cosine similarity
between the (frozen) CLIP image embedding and the (frozen) CLIP text embedding of each a-priori
concept prompt, giving a continuous (N, K) concept vector.

HARD CONSTRAINTS (no oracle):
  - CLIP is FROZEN and used zero-shot. NEVER finetuned on Waterbirds labels.
  - The concept bank is a-priori (fixed in the config before seeing results).
  - Any standardization stats are fit on the TRAIN split ONLY (no leakage).

torch / open_clip / PIL are imported lazily so the rest of the repo imports without them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConceptStandardizer:
    """Per-concept standardization with TRAIN-only statistics."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.mean[None, :]) / (self.std[None, :] + 1e-8)

    @staticmethod
    def fit(train_scores: np.ndarray) -> "ConceptStandardizer":
        return ConceptStandardizer(mean=train_scores.mean(0), std=train_scores.std(0))


class CLIPConceptExtractor:
    """Frozen CLIP -> (N, K) concept-score vectors over an a-priori prompt bank."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        concept_bank: list[str],
        device: str = "cuda",
        temperature_softmax: bool = False,
        temperature: float = 0.01,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.concept_bank = list(concept_bank)
        self.device = device
        self.temperature_softmax = temperature_softmax
        self.temperature = temperature
        self._model = None
        self._preprocess = None
        self._text_emb = None  # (K, d) normalized
        self.standardizer: Optional[ConceptStandardizer] = None

    # ---- frozen model + text bank ----
    def load(self):
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)  # FROZEN
        tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.inference_mode():
            toks = tokenizer(self.concept_bank).to(self.device)
            temb = model.encode_text(toks)
            temb = temb / temb.norm(dim=-1, keepdim=True)
        self._model = model
        self._preprocess = preprocess
        self._text_emb = temb  # (K, d)
        return self

    @property
    def n_concepts(self) -> int:
        return len(self.concept_bank)

    # ---- image -> raw cosine concept scores ----
    def encode_paths(self, paths: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Return raw (N, K) cosine concept scores for a list of image paths."""
        import torch
        from PIL import Image

        if self._model is None:
            self.load()
        scores = []
        rng = range(0, len(paths), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm

                rng = tqdm(rng, desc="CLIP concepts")
            except ImportError:
                pass
        with torch.inference_mode():
            for i in rng:
                batch_paths = paths[i : i + batch_size]
                imgs = torch.stack(
                    [self._preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
                ).to(self.device)
                iemb = self._model.encode_image(imgs)
                iemb = iemb / iemb.norm(dim=-1, keepdim=True)
                sc = iemb @ self._text_emb.T  # (b, K) cosine
                if self.temperature_softmax:
                    sc = torch.softmax(sc / self.temperature, dim=1)
                scores.append(sc.float().cpu().numpy())
        return np.concatenate(scores, axis=0)

    # ---- image -> normalized global CLIP features (the feature space, NOT the concept cosines) ----
    def encode_image_features(self, paths: list[str], batch_size: int = 64,
                              show_progress: bool = True) -> np.ndarray:
        """Return L2-normalized (N, d) global CLIP image embeddings for a list of image paths.

        This is the FEATURE space used by E1/E3/E4 (a logistic head is fit on top of these), as
        opposed to ``encode_paths`` which returns cosine similarities to the text concept bank. The
        model is loaded lazily and frozen (same backbone as the concept path)."""
        import torch
        from PIL import Image

        if self._model is None:
            self.load()
        feats = []
        rng = range(0, len(paths), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm

                rng = tqdm(rng, desc="CLIP features")
            except ImportError:
                pass
        with torch.inference_mode():
            for i in rng:
                batch_paths = paths[i : i + batch_size]
                imgs = torch.stack(
                    [self._preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
                ).to(self.device)
                iemb = self._model.encode_image(imgs)
                iemb = iemb / iemb.norm(dim=-1, keepdim=True)
                feats.append(iemb.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    # ---- standardization (TRAIN-only fit) ----
    def fit_standardizer(self, train_scores: np.ndarray) -> "CLIPConceptExtractor":
        self.standardizer = ConceptStandardizer.fit(train_scores)
        return self

    def apply_standardizer(self, scores: np.ndarray, standardize: bool) -> np.ndarray:
        if standardize and self.standardizer is not None:
            return self.standardizer.transform(scores)
        return scores
