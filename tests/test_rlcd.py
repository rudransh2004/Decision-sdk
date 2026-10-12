import copy
import importlib.util
import math
import unittest
from unittest.mock import patch

import torch

from futurix_decision.rewards import (
    BrierScore,
    LogScore,
    RankedProbabilityScore,
    SphericalScore,
)
from futurix_decision.trainers.losses import decision_sft_loss
from futurix_decision.trainers.rlcd import (
    RLCDConfig,
    RLCDTrainer,
    group_advantages,
    rlcd_policy_loss,
)


class ScoringTests(unittest.TestCase):
    def test_proper_scores_hard_and_soft(self):
        p = torch.tensor([0.8, 0.2])
        hard = torch.tensor([1.0, 0.0])
        soft = torch.tensor([0.7, 0.3])
        self.assertAlmostEqual(LogScore().reward(p, hard).item(), math.log(0.8), places=6)
        self.assertAlmostEqual(
            LogScore().reward(p, soft).item(),
            0.7 * math.log(0.8) + 0.3 * math.log(0.2),
            places=6,
        )
        self.assertAlmostEqual(BrierScore().reward(p, hard).item(), -0.08, places=6)
        self.assertAlmostEqual(
            SphericalScore().reward(p, hard).item(), 0.8 / math.sqrt(0.68), places=6
        )
        ranked = RankedProbabilityScore().reward(
            torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
        )
        self.assertEqual(ranked.item(), -2.0)

    def test_baselines(self):
        rewards = torch.tensor([[1.0], [2.0], [4.0]])
        self.assertTrue(
            torch.allclose(group_advantages(rewards), torch.tensor([[-2.0], [-0.5], [2.5]]))
        )
        self.assertTrue(
            torch.allclose(
                group_advantages(rewards, baseline="group").sum(),
                torch.tensor(0.0),
                atol=1e-6,
            )
        )

    def test_score_function_gradient_and_mask(self):
        torch.manual_seed(17)
        logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        outcome = rlcd_policy_loss(
            logits,
            torch.tensor([[1.0, 0.0, 0.0]]),
            group_size=1024,
            noise_sigma=0.25,
            available=mask,
        )
        outcome.loss.backward()
        self.assertLess(logits.grad[0, 0].item(), 0)  # GD increases A's logit
        self.assertEqual(logits.grad[0, 2].item(), 0)
        self.assertEqual(outcome.rewards.shape, (1024, 1))
        self.assertFalse(outcome.advantages.requires_grad)

    def test_rlcd_hard_and_soft_calibration(self):
        torch.manual_seed(19)
        for target, interval in (
            (torch.tensor([[1.0, 0.0]]), (0.80, 1.0)),
            (torch.tensor([[0.7, 0.3]]), (0.60, 0.80)),
        ):
            mean = torch.nn.Parameter(torch.zeros(1, 2))
            optimizer = torch.optim.Adam([mean], lr=0.05)
            for _ in range(220):
                optimizer.zero_grad()
                loss = rlcd_policy_loss(mean, target, group_size=64, noise_sigma=0.25).loss
                loss.backward()
                optimizer.step()
            probability = mean.detach().softmax(dim=-1)[0, 0].item()
            self.assertGreater(probability, interval[0])
            self.assertLess(probability, interval[1])

    def test_rlcd_combines_policy_sft_and_reference_kl(self):
        from test_sft import example, tiny_model

        model = tiny_model()
        reference = copy.deepcopy(model)
        row = example([0.7, 0.3])
        with torch.no_grad():
            model.backbone.get_output_embeddings().weight[model._token_ids[0]].add_(0.25)
        trainer = RLCDTrainer(
            model,
            config=RLCDConfig(group_size=32, kl_beta=0.5, sft_weight=0.2),
            reference_model=reference,
        )
        trainer._assignment_seed = 13
        batch = model.forward_decisions([row], assignment_seed=13)
        selected = batch.logits[0, batch.slots[0]]
        torch.manual_seed(9)
        expected_policy = rlcd_policy_loss(
            selected.unsqueeze(0),
            torch.tensor([[0.7, 0.3]]),
            group_size=32,
        ).loss
        torch.manual_seed(9)
        total = trainer.compute_loss(batch, [row])
        self.assertAlmostEqual(
            total.item(),
            expected_policy.item()
            + 0.2 * decision_sft_loss(batch, [row]).item()
            + 0.5 * trainer._step_metrics["kl_loss"],
            places=5,
        )
        self.assertGreaterEqual(trainer._step_metrics["kl_loss"], 0.0)


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class RLCDTrainingTests(unittest.TestCase):
    def test_reference_cannot_be_current_policy(self):
        from test_sft import example, tiny_model

        model = tiny_model()
        trainer = RLCDTrainer(model, reference_model=model)
        with self.assertRaisesRegex(ValueError, "separate frozen"):
            trainer.train([example()])

    def test_one_transformer_forward_for_all_reports(self):
        from test_sft import example, tiny_model

        model = tiny_model()
        trainer = RLCDTrainer(
            model,
            config=RLCDConfig(
                group_size=16,
                batch_size=2,
                epochs=1,
                kl_beta=0.0,
                sft_weight=0.2,
                learning_rate=0.002,
            ),
        )
        with patch.object(model.backbone, "forward", wraps=model.backbone.forward) as forward:
            history = trainer.train([example("approve", "a"), example([0.7, 0.3], "b")])
            self.assertEqual(forward.call_count, 1)
        self.assertTrue(math.isfinite(history[0]["rl_loss"]))
        self.assertIn("entropy", history[0])

    def test_default_reference_kl_is_active(self):
        from test_sft import example, tiny_model

        model = tiny_model()
        trainer = RLCDTrainer(
            model,
            config=RLCDConfig(group_size=8, batch_size=2, epochs=1, learning_rate=0.002),
        )
        history = trainer.train([example("approve", "a"), example("reject", "b")])
        self.assertIsNotNone(trainer.reference_model)
        self.assertGreaterEqual(history[0]["kl_loss"], 0)
        self.assertTrue(math.isfinite(history[0]["loss"]))


if __name__ == "__main__":
    unittest.main()
