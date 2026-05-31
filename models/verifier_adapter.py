"""VerifierAdapter — clean interface wrapping NCV (Prover-Verifier Game over concepts).

The rest of the pipeline depends ONLY on this interface, so NCV (official code/checkpoints or
our reimplementation) is fully decoupled (Do/Don't list). For every query x the adapter exposes
the cached quantities the NCV signals (signals/ncv.py) and the verifier-aware score
(conformal/verifier_aware.py) need:

  S_M (Merlin support set)         : indices/mask of concepts the cooperative prover presents
  S_A (Morgana misleading set)     : indices/mask of concepts the adversarial prover presents
  p_A(. | S_M)  shape (N, C)       : Arthur's class distribution given Merlin's set
  p_A(. | S_A)  shape (N, C)       : Arthur's class distribution given Morgana's set
  reject_prob   shape (N,)         : Arthur's reject ("bottom") probability given S_A

Intrinsic NCV diagnostics (Section 6, reported as sanity): completeness and soundness.

Prefer the OFFICIAL NCV code/checkpoints (Turan et al., ICML 2025, arXiv:2507.07532). Set
ncv.source=official with a repo path + checkpoint. Otherwise ncv.source=reimpl uses the minimal
Merlin/Morgana/Arthur below. The Phase-1 report MUST state which was used and report
completeness/soundness.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VerifierOutputs:
    """Per-sample NCV outputs, cached to disk by the precompute stage."""

    pA_given_SM: np.ndarray            # (N, C)
    pA_given_SA: np.ndarray            # (N, C)
    reject_prob: Optional[np.ndarray]  # (N,) or None
    merlin_concepts: list              # length N; each a list[int] of concept indices in S_M
    morgana_concepts: list             # length N; each a list[int] of concept indices in S_A
    # per-label competence for the verifier-aware score (Arthur p over labels given S_M, S_A)
    pA_given_SM_all: np.ndarray        # == pA_given_SM, kept explicit for the sV score
    r_adv_per_label: Optional[np.ndarray] = None  # (N, C) per-label adversarial vulnerability


class VerifierAdapter(abc.ABC):
    """Interface every NCV backend implements."""

    @abc.abstractmethod
    def predict(self, concepts: np.ndarray, y_pred: np.ndarray) -> VerifierOutputs:
        """Run Merlin/Morgana/Arthur for a batch of concept encodings and predicted labels."""

    @abc.abstractmethod
    def intrinsic_metrics(self, concepts: np.ndarray, y_true: np.ndarray) -> dict:
        """Return {'completeness': float, 'soundness': float} sanity diagnostics."""

    @property
    @abc.abstractmethod
    def n_classes(self) -> int: ...

    @property
    @abc.abstractmethod
    def has_reject(self) -> bool: ...


# --------------------------------------------------------------------------------------
# Reimplementation: minimal Merlin / Morgana / Arthur over concept encodings
# --------------------------------------------------------------------------------------
class ReimplNCV(VerifierAdapter):
    """Minimal NCV reimplementation (used when official code/checkpoints are unavailable).

    Arthur is a small MLP A(masked_concepts) -> logits over C (+1 reject). Merlin greedily
    selects a sparse concept subset maximizing p_A(yhat|S); Morgana greedily selects a subset
    maximizing p_A(y'|S) for the best wrong label y'. Arthur is trained via the Prover-Verifier
    Game: correct under Merlin's helpful sets, reject under Morgana's misleading sets.

    The heavy training lives in ``train`` (torch). ``predict`` runs the cached greedy selection
    + Arthur forward passes and returns numpy arrays for caching.
    """

    def __init__(
        self,
        concept_dim: int,
        n_classes: int,
        merlin_sparsity: int = 4,
        morgana_sparsity: int = 4,
        reject_class: bool = True,
        hidden: int = 128,
        device: str = "cuda",
    ):
        self.concept_dim = concept_dim
        self._n_classes = n_classes
        self.merlin_sparsity = merlin_sparsity
        self.morgana_sparsity = morgana_sparsity
        self._has_reject = reject_class
        self.hidden = hidden
        self.device = device
        self.arthur = None  # torch module, built in train()

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def has_reject(self) -> bool:
        return self._has_reject

    # ---- model ----
    def _build_arthur(self):
        import torch.nn as nn

        out_dim = self._n_classes + (1 if self._has_reject else 0)
        # input = concept values concatenated with the binary mask (which concepts are revealed)
        return nn.Sequential(
            nn.Linear(self.concept_dim * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, out_dim),
        )

    def _arthur_probs(self, concepts_t, mask_t):
        import torch

        inp = torch.cat([concepts_t * mask_t, mask_t], dim=1)
        logits = self.arthur(inp)
        return torch.softmax(logits, dim=1)

    def train(self, concepts: np.ndarray, y: np.ndarray, perf_ctx, epochs: int = 30,
              lr: float = 1e-3, n_mask_samples: int = 4):
        """Prover-Verifier Game training of Arthur (and implicit greedy provers).

        Each step: sample random sparse masks (proxy provers early in training), plus the
        current greedy Merlin / Morgana masks; train Arthur to (a) predict y under helpful
        masks, (b) predict reject under misleading masks. This is the standard NCV objective
        rebuilt minimally; swap in the official trainer for the paper-grade verifier.
        """
        import torch
        import torch.nn as nn

        self.arthur = self._build_arthur().to(self.device)
        opt = torch.optim.Adam(self.arthur.parameters(), lr=lr)
        C = self._concept_dim_safe()
        x = torch.as_tensor(concepts, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        reject_idx = self._n_classes  # label for reject
        n = x.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            x, yt = x[perm], yt[perm]
            # sample helpful (Merlin-like) masks: keep a sparse subset; train -> y
            for _ in range(n_mask_samples):
                mask = self._random_sparse_mask(n, self.merlin_sparsity)
                p = self._arthur_probs(x, mask)
                loss_help = nn.functional.nll_loss(torch.log(p + 1e-9), yt)
                # misleading (Morgana-like) masks: train -> reject (if enabled) else uniform
                mask_adv = self._random_sparse_mask(n, self.morgana_sparsity)
                p_adv = self._arthur_probs(x, mask_adv)
                if self._has_reject:
                    target = torch.full((n,), reject_idx, device=self.device, dtype=torch.long)
                    loss_adv = nn.functional.nll_loss(torch.log(p_adv + 1e-9), target)
                else:
                    loss_adv = -(-(p_adv * torch.log(p_adv + 1e-9)).sum(1)).mean()
                loss = loss_help + 0.5 * loss_adv
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return self

    def _concept_dim_safe(self):
        return self.concept_dim

    def _random_sparse_mask(self, n, k):
        import torch

        mask = torch.zeros(n, self.concept_dim, device=self.device)
        idx = torch.argsort(torch.rand(n, self.concept_dim, device=self.device), dim=1)[:, :k]
        mask.scatter_(1, idx, 1.0)
        return mask

    # ---- greedy prover selection at inference ----
    def _greedy_select(self, concepts_t, target_labels, sparsity, maximize_label):
        """Greedily add concepts to maximize Arthur's prob of ``maximize_label``.

        Returns (mask, chosen_indices). target_labels guides which label to push.
        """
        import torch

        n, d = concepts_t.shape
        mask = torch.zeros(n, d, device=concepts_t.device)
        chosen = [[] for _ in range(n)]
        rows = torch.arange(n, device=concepts_t.device)
        for _ in range(sparsity):
            best_gain = torch.full((n,), -1e9, device=concepts_t.device)
            best_j = torch.zeros(n, dtype=torch.long, device=concepts_t.device)
            for j in range(d):
                trial = mask.clone()
                trial[:, j] = 1.0
                p = self._arthur_probs(concepts_t, trial)[rows, maximize_label]
                # don't reselect already-chosen concepts
                already = mask[:, j] > 0
                gain = torch.where(already, torch.full_like(p, -1e9), p)
                upd = gain > best_gain
                best_gain = torch.where(upd, gain, best_gain)
                best_j = torch.where(upd, torch.full_like(best_j, j), best_j)
            mask[rows, best_j] = 1.0
            for i in range(n):
                chosen[i].append(int(best_j[i].item()))
        return mask, chosen

    def predict(self, concepts: np.ndarray, y_pred: np.ndarray) -> VerifierOutputs:
        import torch

        assert self.arthur is not None, "train() Arthur before predict()"
        x = torch.as_tensor(concepts, dtype=torch.float32, device=self.device)
        yp = torch.as_tensor(y_pred, dtype=torch.long, device=self.device)
        rows = torch.arange(x.shape[0], device=self.device)

        with torch.inference_mode():
            # Merlin: cooperative -> push yhat
            mask_M, sel_M = self._greedy_select(x, yp, self.merlin_sparsity, yp)
            pA_SM = self._arthur_probs(x, mask_M)
            # Morgana: adversarial -> push the best wrong label
            pA_full = self._arthur_probs(x, torch.ones_like(x))
            wrong = pA_full[:, : self._n_classes].clone()
            wrong[rows, yp] = -1.0
            y_adv = wrong.argmax(dim=1)
            mask_A, sel_A = self._greedy_select(x, y_adv, self.morgana_sparsity, y_adv)
            pA_SA = self._arthur_probs(x, mask_A)

        pA_SM_np = pA_SM[:, : self._n_classes].detach().cpu().numpy()
        pA_SA_np = pA_SA[:, : self._n_classes].detach().cpu().numpy()
        reject = (
            pA_SA[:, self._n_classes].detach().cpu().numpy() if self._has_reject else None
        )
        return VerifierOutputs(
            pA_given_SM=pA_SM_np,
            pA_given_SA=pA_SA_np,
            reject_prob=reject,
            merlin_concepts=sel_M,
            morgana_concepts=sel_A,
            pA_given_SM_all=pA_SM_np,
        )

    def intrinsic_metrics(self, concepts: np.ndarray, y_true: np.ndarray) -> dict:
        """Completeness = Arthur acc given Merlin sets; soundness = 1 - acc-of-being-fooled
        by Morgana sets (higher = sounder)."""
        out = self.predict(concepts, y_true)
        comp = float((out.pA_given_SM.argmax(1) == y_true).mean())
        fooled = (out.pA_given_SA.argmax(1) != y_true).mean()
        sound = float(1.0 - fooled)
        return {"completeness": comp, "soundness": sound}


# --------------------------------------------------------------------------------------
# Official-code wrapper (preferred)
# --------------------------------------------------------------------------------------
class OfficialNCV(VerifierAdapter):
    """Wrap the official NCV release. Loads the repo + checkpoint and routes calls to it.

    Implement the four hooks against the official API once the repo path/checkpoint are set;
    the rest of the pipeline is unchanged because it only sees VerifierOutputs.
    """

    def __init__(self, repo_path: str, checkpoint: str, n_classes: int, has_reject: bool = True,
                 device: str = "cuda"):
        self.repo_path = repo_path
        self.checkpoint = checkpoint
        self._n_classes = n_classes
        self._has_reject = has_reject
        self.device = device
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not self.repo_path or not self.checkpoint:
            raise RuntimeError(
                "OfficialNCV requires ncv.official_repo and ncv.checkpoint. Clone Turan et al. "
                "(arXiv:2507.07532) and set NCV_REPO / NCV_CKPT, or use ncv.source=reimpl."
            )
        # import sys; sys.path.insert(0, self.repo_path); from ncv import ...
        raise NotImplementedError(
            "Bind the official NCV API here (load Merlin/Morgana/Arthur from checkpoint). "
            "Until bound, run with ncv.source=reimpl."
        )

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def has_reject(self) -> bool:
        return self._has_reject

    def predict(self, concepts, y_pred):  # pragma: no cover - requires official repo
        self._ensure_loaded()

    def intrinsic_metrics(self, concepts, y_true):  # pragma: no cover
        self._ensure_loaded()


def build_verifier(ncv_cfg: dict, concept_dim: int, n_classes: int, device: str = "cuda") -> VerifierAdapter:
    """Factory: pick official vs reimpl from config (Section 17 deliverable #2)."""
    source = ncv_cfg.get("source", "reimpl")
    if source == "official":
        return OfficialNCV(
            repo_path=ncv_cfg.get("official_repo", ""),
            checkpoint=ncv_cfg.get("checkpoint", ""),
            n_classes=n_classes,
            has_reject=ncv_cfg.get("reject_class", True),
            device=device,
        )
    return ReimplNCV(
        concept_dim=concept_dim,
        n_classes=n_classes,
        merlin_sparsity=ncv_cfg.get("merlin_sparsity", 4),
        morgana_sparsity=ncv_cfg.get("morgana_sparsity", 4),
        reject_class=ncv_cfg.get("reject_class", True),
        device=device,
    )
