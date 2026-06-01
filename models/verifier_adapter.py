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
    """Faithful minimal Prover-Verifier Game over CONTINUOUS concept vectors (task Section 3).

    Arthur is a small MLP ``A([concepts (.) mask | mask]) -> logits`` over C classes (+1 reject
    ``bottom``). The provers select sparse concept subsets *using the current Arthur* (greedy):

      - **Merlin** (cooperative): picks a subset S of size <= merlin_sparsity MAXIMIZING
        ``p_A(y_target | S)`` (helpful evidence for the target label).
      - **Morgana** (adversarial): picks a subset S MAXIMIZING ``p_A(y' | S)`` for the best WRONG
        label y' (misleading evidence).

    Training is the actual alternating game (NOT random masks): every ``prover_refresh`` epochs
    the greedy Merlin/Morgana selections are recomputed from the current Arthur, then Arthur is
    updated to (a) be CORRECT under Merlin's helpful set (toward the TRUE label) and (b) REJECT
    under Morgana's misleading set. With ``morgana_enabled=False`` the adversarial branch is
    dropped entirely (Arthur is only taught to be correct under Merlin) — the required ablation,
    in which case ``predict`` returns ``S_A = S_M`` and zero reject so ``V_sound`` carries no
    adversarial information (callers fix ``beta=1`` so ``V_full == V_comp``).

    Concept inputs are standardized with TRAIN-only statistics fit in ``train`` and reapplied in
    ``predict`` (no leakage). The reference design is ZIB-IOL/merlin-arthur-classifiers
    (Waeldchen et al.) / Turan et al. (arXiv:2507.07532).
    """

    def __init__(
        self,
        concept_dim: int,
        n_classes: int,
        merlin_sparsity: int = 6,
        morgana_sparsity: int = 6,
        reject_class: bool = True,
        hidden: int = 128,
        device: str = "cuda",
        morgana_enabled: bool = True,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        prover_refresh: int = 1,
        morgana_weight: float = 1.0,
        n_train_max: Optional[int] = 4000,
        standardize: bool = True,
    ):
        self.concept_dim = concept_dim
        self._n_classes = n_classes
        self.merlin_sparsity = merlin_sparsity
        self.morgana_sparsity = morgana_sparsity
        # a reject head is only meaningful when Morgana is in play
        self._has_reject = reject_class and morgana_enabled
        self.hidden = hidden
        self.device = device
        self.morgana_enabled = morgana_enabled
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.prover_refresh = max(1, prover_refresh)
        self.morgana_weight = morgana_weight
        self.n_train_max = n_train_max
        self.standardize = standardize
        self.arthur = None  # torch module, built in train()
        self._mean = None    # (D,) TRAIN concept mean (standardizer)
        self._std = None     # (D,) TRAIN concept std

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

    def _standardize_np(self, concepts: np.ndarray) -> np.ndarray:
        if not self.standardize or self._mean is None:
            return np.asarray(concepts, dtype=np.float32)
        return ((np.asarray(concepts, dtype=np.float32) - self._mean[None, :])
                / (self._std[None, :] + 1e-8)).astype(np.float32)

    def _arthur_logits(self, concepts_t, mask_t):
        import torch

        inp = torch.cat([concepts_t * mask_t, mask_t], dim=1)
        return self.arthur(inp)

    def _arthur_probs(self, concepts_t, mask_t):
        import torch

        return torch.softmax(self._arthur_logits(concepts_t, mask_t), dim=1)

    # ---- greedy prover selection (used in BOTH training and inference) ----
    def _greedy_select(self, concepts_t, sparsity, maximize_label):
        """Greedily grow a sparse mask to MAXIMIZE Arthur's prob of ``maximize_label``.

        Vectorized over samples; loops over the ``d`` candidate concepts per greedy step. Runs
        under inference_mode (no grad) — provers select against a frozen snapshot of Arthur.
        Returns (mask (n,d) float, chosen list[list[int]]).
        """
        import torch

        n, d = concepts_t.shape
        mask = torch.zeros(n, d, device=concepts_t.device)
        chosen = [[] for _ in range(n)]
        rows = torch.arange(n, device=concepts_t.device)
        for _ in range(int(sparsity)):
            best_gain = torch.full((n,), -1e9, device=concepts_t.device)
            best_j = torch.zeros(n, dtype=torch.long, device=concepts_t.device)
            for j in range(d):
                trial = mask.clone()
                trial[:, j] = 1.0
                p = self._arthur_probs(concepts_t, trial)[rows, maximize_label]
                already = mask[:, j] > 0  # don't reselect a chosen concept
                gain = torch.where(already, torch.full_like(p, -1e9), p)
                upd = gain > best_gain
                best_gain = torch.where(upd, gain, best_gain)
                best_j = torch.where(upd, torch.full_like(best_j, j), best_j)
            mask[rows, best_j] = 1.0
            for i in range(n):
                chosen[i].append(int(best_j[i].item()))
        return mask, chosen

    def _best_wrong_label(self, concepts_t, y_avoid):
        """Best wrong label for each sample = argmax_{y != y_avoid} p_A(y | full concept set)."""
        import torch

        rows = torch.arange(concepts_t.shape[0], device=concepts_t.device)
        with torch.inference_mode():
            pf = self._arthur_probs(concepts_t, torch.ones_like(concepts_t))[:, : self._n_classes]
        wrong = pf.clone()
        wrong[rows, y_avoid] = -1.0
        return wrong.argmax(dim=1)

    def train(self, concepts: np.ndarray, y: np.ndarray, perf_ctx=None, epochs: Optional[int] = None,
              lr: Optional[float] = None, seed: int = 0):
        """Alternating Prover-Verifier-Game training of Arthur (Section 3).

        Loop: (1) freeze Arthur, recompute greedy Merlin sets toward the TRUE label and (if
        enabled) Morgana sets toward the best wrong label; (2) SGD Arthur to predict y under
        Merlin's sets and reject under Morgana's. ``perf_ctx`` is accepted for call-site symmetry
        with the rest of the pipeline but unused (this is a tiny CPU/GPU MLP).
        """
        import torch
        import torch.nn as nn

        epochs = int(epochs if epochs is not None else self.epochs)
        lr = float(lr if lr is not None else self.lr)

        concepts = np.asarray(concepts, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # TRAIN-only standardizer (fit here, reapplied in predict — no leakage).
        if self.standardize:
            self._mean = concepts.mean(axis=0)
            self._std = concepts.std(axis=0)

        # optional class-stratified subsample to bound the (one-time) greedy training cost
        if self.n_train_max is not None and concepts.shape[0] > self.n_train_max:
            rng = np.random.default_rng(seed)
            classes = np.unique(y)
            per = max(1, self.n_train_max // len(classes))
            keep = np.concatenate([
                rng.permutation(np.where(y == c)[0])[:per] for c in classes
            ])
            rng.shuffle(keep)
            concepts, y = concepts[keep], y[keep]

        cstd = self._standardize_np(concepts)
        x = torch.as_tensor(cstd, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        n = x.shape[0]
        reject_idx = self._n_classes
        rows = torch.arange(n, device=self.device)

        self.arthur = self._build_arthur().to(self.device)
        opt = torch.optim.Adam(self.arthur.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        mask_M = None
        mask_A = None
        for ep in range(epochs):
            if ep % self.prover_refresh == 0:
                self.arthur.eval()
                with torch.inference_mode():
                    # Merlin: cooperative, push the TRUE label
                    mask_M, _ = self._greedy_select(x, self.merlin_sparsity, yt)
                    if self.morgana_enabled:
                        y_adv = self._best_wrong_label(x, yt)
                        mask_A, _ = self._greedy_select(x, self.morgana_sparsity, y_adv)
            self.arthur.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                logits_M = self._arthur_logits(x[idx], mask_M[idx])
                loss = crit(logits_M, yt[idx])
                if self.morgana_enabled:
                    logits_A = self._arthur_logits(x[idx], mask_A[idx])
                    tgt = torch.full((idx.shape[0],), reject_idx, device=self.device,
                                     dtype=torch.long)
                    loss = loss + self.morgana_weight * crit(logits_A, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.arthur.eval()
        return self

    def predict(self, concepts: np.ndarray, y_pred: np.ndarray) -> VerifierOutputs:
        import torch

        assert self.arthur is not None, "train() Arthur before predict()"
        x = torch.as_tensor(self._standardize_np(concepts), dtype=torch.float32, device=self.device)
        yp = torch.as_tensor(np.asarray(y_pred, dtype=np.int64), dtype=torch.long, device=self.device)
        rows = torch.arange(x.shape[0], device=self.device)

        with torch.inference_mode():
            # Merlin: cooperative -> push f's predicted label yhat
            mask_M, sel_M = self._greedy_select(x, self.merlin_sparsity, yp)
            pA_SM = self._arthur_probs(x, mask_M)
            if self.morgana_enabled:
                # Morgana: adversarial -> push the best wrong label (!= yhat)
                y_adv = self._best_wrong_label(x, yp)
                mask_A, sel_A = self._greedy_select(x, self.morgana_sparsity, y_adv)
                pA_SA = self._arthur_probs(x, mask_A)
            else:
                # ablation: no adversary -> S_A = S_M, no reject signal
                mask_A, sel_A, pA_SA = mask_M, sel_M, pA_SM

        pA_SM_np = pA_SM[:, : self._n_classes].detach().cpu().numpy()
        pA_SA_np = pA_SA[:, : self._n_classes].detach().cpu().numpy()
        if self._has_reject and self.morgana_enabled:
            reject = pA_SA[:, self._n_classes].detach().cpu().numpy()
        else:
            reject = np.zeros(pA_SA_np.shape[0], dtype=np.float32)
        return VerifierOutputs(
            pA_given_SM=pA_SM_np,
            pA_given_SA=pA_SA_np,
            reject_prob=reject,
            merlin_concepts=sel_M,
            morgana_concepts=sel_A,
            pA_given_SM_all=pA_SM_np,
        )

    def intrinsic_metrics(self, concepts: np.ndarray, y_true: np.ndarray) -> dict:
        """Sanity diagnostics. completeness / merlin_acc = Arthur acc when Merlin pushes the TRUE
        label; morgana_acc = Arthur acc under Morgana's misleading set (lower = the game bites);
        soundness = 1 - rate Arthur is FOOLED into a wrong class under Morgana."""
        y_true = np.asarray(y_true, dtype=np.int64)
        out = self.predict(concepts, y_true)
        merlin_acc = float((out.pA_given_SM.argmax(1) == y_true).mean())
        morgana_acc = float((out.pA_given_SA.argmax(1) == y_true).mean())
        fooled = float((out.pA_given_SA.argmax(1) != y_true).mean())
        return {
            "completeness": merlin_acc,
            "soundness": float(1.0 - fooled),
            "merlin_acc": merlin_acc,
            "morgana_acc": morgana_acc,
            "morgana_enabled": bool(self.morgana_enabled),
        }


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
        merlin_sparsity=ncv_cfg.get("merlin_sparsity", 6),
        morgana_sparsity=ncv_cfg.get("morgana_sparsity", 6),
        reject_class=ncv_cfg.get("reject_class", True),
        hidden=ncv_cfg.get("hidden", 128),
        device=device,
        morgana_enabled=str(ncv_cfg.get("morgana", "on")).lower() in ("on", "true", "1"),
        epochs=ncv_cfg.get("epochs", 30),
        lr=ncv_cfg.get("lr", 1e-3),
        batch_size=ncv_cfg.get("batch_size", 256),
        prover_refresh=ncv_cfg.get("prover_refresh", 1),
        morgana_weight=ncv_cfg.get("morgana_weight", 1.0),
        n_train_max=ncv_cfg.get("n_train_max", 4000),
        standardize=ncv_cfg.get("standardize", True),
    )
