"""RLCD: Gaussian-logit reports trained with a real score-function gradient."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from futurix_decision.datasets import DecisionExample
from futurix_decision.evaluation.metrics import compute_metrics
from futurix_decision.models import DecisionBatch, DecisionModel
from futurix_decision.rewards import LogScore, ProperScoringRule, RankedProbabilityScore, SphericalScore
from futurix_decision.trainers.losses import decision_sft_loss
from futurix_decision.trainers.sft import DatasetSource, SFTConfig, SFTTrainer

BaselineMethod = Literal["loo", "group"]


def group_advantages(rewards: torch.Tensor, *, baseline: BaselineMethod = "loo") -> torch.Tensor:
    """First axis is G; default leave-one-out avoids self-baseline bias."""
    if rewards.ndim != 2 or rewards.shape[0] < 2:
        raise ValueError("rewards must have shape [group_size >= 2, batch]")
    if baseline == "loo":
        reference = (rewards.sum(dim=0, keepdim=True) - rewards) / (rewards.shape[0] - 1)
    elif baseline == "group":
        reference = rewards.mean(dim=0, keepdim=True)
    else:
        raise ValueError("baseline must be loo or group")
    return (rewards - reference).detach()


@dataclass
class RLCDStep:
    loss: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    probabilities: torch.Tensor


def rlcd_policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_size: int = 8,
    noise_sigma: float = 0.25,
    rewards: Sequence[ProperScoringRule] | None = None,
    kinds: Sequence[str] | None = None,
    available: torch.Tensor | None = None,
    baseline: BaselineMethod = "loo",
) -> RLCDStep:
    """One Gaussian sample operation on small logits, not G transformer passes.

    ``Normal.sample`` is deliberately non-reparameterized: the score-function
    gradient is through log_pi(report | original_logits), not through reward.
    """
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must have shape [batch, candidates]")
    if group_size < 2 or not math.isfinite(noise_sigma) or noise_sigma <= 0:
        raise ValueError("group_size must be >= 2 and noise_sigma positive")
    mask = available if available is not None else torch.ones_like(logits, dtype=torch.bool)
    if mask.shape != logits.shape or mask.dtype != torch.bool or (mask.sum(dim=-1) < 2).any():
        raise ValueError("available must mark at least two candidates in each row")
    if kinds is not None and len(kinds) != logits.shape[0]:
        raise ValueError("kinds must match the batch size")
    # Padding positions are *not* Normal policy actions. Sampling on the small
    # padded tensor is safe because their log probabilities are then excluded.
    mean = logits.float().masked_fill(~mask, 0.0)
    policy = Normal(mean, noise_sigma)
    reports = policy.sample((group_size,))
    p = torch.softmax(reports.masked_fill(~mask.unsqueeze(0), -1e9), dim=-1)
    rules = list(rewards) if rewards is not None else [LogScore(), SphericalScore(weight=0.5)]
    if not rules:
        raise ValueError("provide at least one scoring rule")
    reward_values = torch.zeros_like(p[..., 0])
    for rule in rules:
        value = rule.weight * rule.reward(p, targets.unsqueeze(0))
        if isinstance(rule, RankedProbabilityScore) and kinds is not None:
            ordinal_rows = torch.tensor(
                [kind == "score" for kind in kinds], device=logits.device
            )
            value = value * ordinal_rows.unsqueeze(0)
        reward_values = reward_values + value
    if rewards is None and kinds is not None:
        ordinal = torch.tensor([kind == "score" for kind in kinds], device=logits.device)
        if ordinal.any():
            reward_values[:, ordinal] += RankedProbabilityScore().reward(
                p[:, ordinal], targets.unsqueeze(0)[:, ordinal]
            )
    advantages = group_advantages(reward_values, baseline=baseline)
    log_pi = policy.log_prob(reports).masked_fill(~mask.unsqueeze(0), 0.0).sum(dim=-1)
    rl_loss = -(advantages * log_pi).mean()
    return RLCDStep(rl_loss, reward_values.detach(), advantages, p.detach())


@dataclass(frozen=True)
class RLCDConfig(SFTConfig):
    group_size: int = 8
    noise_sigma: float = 0.25
    sft_weight: float = 0.2
    kl_beta: float = 0.01
    baseline: BaselineMethod = "loo"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.group_size < 2 or not math.isfinite(self.noise_sigma) or self.noise_sigma <= 0:
            raise ValueError("invalid RLCD group_size or noise_sigma")
        if any(not math.isfinite(x) or x < 0 for x in (self.sft_weight, self.kl_beta)):
            raise ValueError("RLCD anchor weights must be finite and non-negative")
        if self.baseline not in ("loo", "group"):
            raise ValueError("baseline must be loo or group")


class RLCDTrainer(SFTTrainer):
    def __init__(
        self,
        model: DecisionModel,
        *,
        config: RLCDConfig | None = None,
        group_size: int = 8,
        noise_sigma: float = 0.25,
        rewards: Sequence[ProperScoringRule] | None = None,
        sft_weight: float = 0.2,
        kl_beta: float = 0.01,
        reference_model: DecisionModel | None = None,
    ) -> None:
        resolved = config or RLCDConfig(
            group_size=group_size, noise_sigma=noise_sigma, sft_weight=sft_weight, kl_beta=kl_beta
        )
        super().__init__(model, config=resolved)
        self.config: RLCDConfig
        self.rewards = rewards
        self.reference_model = reference_model

    def train(self, dataset: DatasetSource) -> list[dict[str, float | int]]:
        if self.reference_model is self.model or (
            self.reference_model is not None
            and self.reference_model.backbone is self.model.backbone
        ):
            raise ValueError("reference_model must be a separate frozen checkpoint")
        if self.config.kl_beta and self.reference_model is None:
            self.reference_model = copy.deepcopy(self.model)
        if self.reference_model is not None:
            if (
                self.reference_model.decision_tokens,
                self.reference_model.seed,
                self.reference_model.head_type,
                self.reference_model.head_config,
            ) != (
                self.model.decision_tokens,
                self.model.seed,
                self.model.head_type,
                self.model.head_config,
            ):
                raise ValueError("reference must have identical decision readout and seed")
            self.reference_model.backbone.requires_grad_(False)
            self.reference_model.backbone.eval()
            if self.reference_model.decision_head is not None:
                self.reference_model.decision_head.requires_grad_(False)
                self.reference_model.decision_head.eval()
        return super().train(dataset)

    def compute_loss(
        self, batch: DecisionBatch, examples: Sequence[DecisionExample]
    ) -> torch.Tensor:
        lengths = [len(row.options) for row in examples]
        width = max(lengths)
        means = torch.zeros(len(examples), width, device=batch.logits.device)
        targets = torch.zeros_like(means)
        available = torch.zeros_like(means, dtype=torch.bool)
        for i, (row, slots) in enumerate(zip(examples, batch.slots)):
            means[i, : lengths[i]] = batch.logits[i, slots].float()
            targets[i, : lengths[i]] = torch.tensor(
                row.target_distribution(), device=batch.logits.device
            )
            available[i, : lengths[i]] = True
        step = rlcd_policy_loss(
            means,
            targets,
            group_size=self.config.group_size,
            noise_sigma=self.config.noise_sigma,
            rewards=self.rewards,
            kinds=[row.kind for row in examples],
            available=available,
            baseline=self.config.baseline,
        )
        sft = (
            decision_sft_loss(batch, examples)
            if self.config.sft_weight
            else step.loss.new_zeros(())
        )
        kl = step.loss.new_zeros(())
        if self.config.kl_beta:
            assert self.reference_model is not None
            reference = self.reference_model
            reference_device = next(reference.backbone.parameters()).device
            if reference_device != batch.logits.device:
                if getattr(reference.backbone, "is_loaded_in_4bit", False):
                    raise ValueError("4-bit reference must already be on the training GPU")
                reference.backbone.to(batch.logits.device)
            if reference.decision_head is not None:
                reference.decision_head.to(batch.logits.device)
            with torch.no_grad():
                ref = reference.forward_decisions(
                    examples, assignment_seed=self._assignment_seed
                )
            for i, slots in enumerate(batch.slots):
                if ref.slots[i] != slots:
                    raise ValueError("reference candidate assignments do not match")
                log_p = F.log_softmax(batch.logits[i, slots].float(), dim=-1)
                log_q = F.log_softmax(ref.logits[i, slots].float(), dim=-1)
                kl = kl + (log_p.exp() * (log_p - log_q)).sum()
            kl = kl / len(examples)
        loss = step.loss + self.config.sft_weight * sft + self.config.kl_beta * kl
        with torch.no_grad():
            p = torch.softmax(means.detach().masked_fill(~available, -1e9), dim=-1)
            metrics = compute_metrics(
                [p[i, :length].cpu() for i, length in enumerate(lengths)],
                [targets[i, :length].cpu() for i, length in enumerate(lengths)],
            )
            self._step_metrics = {
                "total_loss": float(loss.detach()),
                "rl_loss": float(step.loss.detach()),
                "sft_loss": float(sft.detach()),
                "kl_loss": float(kl.detach()),
                "mean_reward": float(step.rewards.mean()),
                "reward_std": float(step.rewards.std(unbiased=False)),
                "entropy": float(-(p * p.clamp_min(1e-12).log()).sum(dim=-1).mean()),
                "accuracy": metrics["accuracy"],
                "nll": metrics["nll"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "mean_confidence": metrics["mean_confidence"],
            }
        return loss
