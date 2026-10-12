import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from futurix_decision.evaluation import compute_metrics, evaluate, fit_temperature_from_logits
from futurix_decision.models import DecisionBatch


class MetricTests(unittest.TestCase):
    def test_hard_and_soft_metrics_and_selective_accuracy(self):
        p = [[0.8, 0.2], [0.6, 0.4]]
        y = [[1.0, 0.0], [0.3, 0.7]]
        result = compute_metrics(p, y, bins=5)
        self.assertAlmostEqual(result["accuracy"], 0.65)
        expected_nll = (-math.log(0.8) - 0.3 * math.log(0.6) - 0.7 * math.log(0.4)) / 2
        self.assertAlmostEqual(result["nll"], expected_nll, places=5)
        self.assertAlmostEqual(result["brier"], (0.08 + 0.18) / 2, places=5)
        self.assertAlmostEqual(result["ece"], (0.2 + 0.3) / 2, places=5)
        self.assertAlmostEqual(result["mce"], 0.3, places=5)
        self.assertEqual(result["coverage"][0.8], 0.5)
        self.assertEqual(result["selective_accuracy"][0.8], 1.0)
        self.assertIsNone(result["selective_accuracy"][0.9])

    def test_temperature_optimizes_soft_label_nll(self):
        raw = torch.tensor([math.log(0.95 / 0.05), 0.0])
        target = torch.tensor([0.7, 0.3])
        temp = fit_temperature_from_logits([raw] * 20, [target] * 20)
        self.assertGreater(temp, 1.0)
        self.assertAlmostEqual(torch.softmax(raw / temp, dim=-1)[0].item(), 0.7, delta=0.02)

    def test_calibration_ce_plus_brier_objective(self):
        from futurix_decision.trainers.calibration import CalibrationConfig, CalibrationTrainer
        from test_sft import example, tiny_model

        trainer = CalibrationTrainer(
            tiny_model(), config=CalibrationConfig(ce_weight=1.0, brier_weight=0.5)
        )
        logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
        target = torch.tensor([0.7, 0.3])
        loss = trainer.compute_loss(
            DecisionBatch(logits=logits, slots=[[1, 0]]),
            [example([0.7, 0.3])],
        )
        selected = logits[0, [1, 0]]
        expected = -(target * F.log_softmax(selected, dim=-1)).sum()
        expected += 0.5 * (F.softmax(selected, dim=-1) - target).square().sum()
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        self.assertTrue(torch.any(logits.grad != 0))


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "train extra not installed")
class EvaluatorTrainerTests(unittest.TestCase):
    def test_vllm_decision_results_use_identical_metric_and_journal_path(self):
        from test_sft import example, tiny_model

        rows = [example([1.0, 0.0], "a"), example([0.0, 1.0], "b")]
        calls = []

        def batch_decide(batch):
            calls.append(len(batch))
            return [
                SimpleNamespace(probabilities=dict(zip(row.options, (0.7, 0.3)))) for row in batch
            ]

        engine = SimpleNamespace(
            prompt_format="futurix",
            temperature=1.0,
            seed=0,
            decision_tokens=16,
            tokenizer=tiny_model().tokenizer,
            batch_decide=batch_decide,
        )
        with tempfile.TemporaryDirectory() as temp:
            result = evaluate(
                engine,
                rows,
                batch_size=2,
                max_batch_tokens=1,
                predictions_output=Path(temp) / "vllm.jsonl",
                run_id="vllm",
            )
            self.assertEqual(calls, [1, 1])
            self.assertEqual(result["count"], 2)
            self.assertAlmostEqual(result["accuracy"], 0.5)

    def test_token_budget_splits_long_padded_batches_without_changing_metrics(self):
        from test_sft import example, tiny_model

        model = tiny_model()
        rows = [example([0.7, 0.3], "short"), example([0.2, 0.8], "long" * 40)]
        expected = evaluate(model, rows, batch_size=2)
        with patch.object(model.backbone, "forward", wraps=model.backbone.forward) as forward:
            actual = evaluate(model, rows, batch_size=2, max_batch_tokens=1)
            self.assertEqual(forward.call_count, 2)
        self.assertAlmostEqual(actual["accuracy"], expected["accuracy"], places=6)
        self.assertAlmostEqual(actual["nll"], expected["nll"], places=6)

    def test_evaluation_journal_resumes_without_recomputing_rows(self):
        from test_sft import example, tiny_model

        model = tiny_model()
        rows = [
            example([0.7, 0.3], "one"),
            example([0.2, 0.8], "two"),
            example([0.9, 0.1], "three"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            journal = Path(temp) / "predictions.jsonl"
            first = evaluate(
                model, rows[:2], predictions_output=journal, run_id="same", batch_size=2
            )
            self.assertEqual(first["count"], 2)
            with journal.open("a", encoding="utf-8") as stream:
                stream.write('{"id":"incomplete')
            with patch.object(model.backbone, "forward", wraps=model.backbone.forward) as forward:
                resumed = evaluate(
                    model,
                    rows,
                    predictions_output=journal,
                    run_id="same",
                    resume=True,
                    batch_size=2,
                )
                self.assertEqual(forward.call_count, 1)
            self.assertEqual(resumed["count"], 3)
            self.assertAlmostEqual(resumed["accuracy"], evaluate(model, rows)["accuracy"], places=6)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                evaluate(model, rows, predictions_output=journal, run_id="different", resume=True)

    def test_batched_evaluation_and_calibration_save(self):
        from futurix_decision import DecisionModel
        from futurix_decision.trainers.calibration import CalibrationConfig, CalibrationTrainer
        from test_sft import example, tiny_model

        model = tiny_model()
        dataset = [example([0.7, 0.3], "a"), example([0.2, 0.8], "b")]
        with patch.object(model.backbone, "forward", wraps=model.backbone.forward) as forward:
            result = evaluate(model, dataset, batch_size=2)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(result["count"], 2)
        self.assertGreaterEqual(result["ece"], 0)
        trainer = CalibrationTrainer(
            model, config=CalibrationConfig(batch_size=2, epochs=1, brier_weight=0.2)
        )
        history = trainer.train(dataset, calibration_dataset=dataset)
        self.assertTrue(history)
        self.assertGreater(model.temperature, 0)
        with tempfile.TemporaryDirectory() as temp:
            trainer.save_pretrained(temp)
            restored = DecisionModel.from_pretrained(temp)
            self.assertAlmostEqual(model.temperature, restored.temperature)


if __name__ == "__main__":
    unittest.main()
