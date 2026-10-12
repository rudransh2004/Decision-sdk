import json
import tempfile
import unittest
from pathlib import Path

from futurix_decision.utils.logging import TrainingLogger


class LoggingTests(unittest.TestCase):
    def test_jsonl_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "logs" / "rlcd.jsonl"
            logger = TrainingLogger(jsonl=str(path))
            logger.log(
                {
                    "step": 1, "loss": 0.3, "rl_loss": 0.1, "sft_loss": 0.2,
                    "kl_loss": 0.0, "mean_reward": -0.1, "reward_std": 0.05,
                    "entropy": 0.7, "accuracy": 0.6, "nll": 0.8,
                    "brier": 0.2, "ece": 0.1, "mean_confidence": 0.65,
                }
            )
            logger.close()
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["step"], 1)
            self.assertEqual(record["ece"], 0.1)


if __name__ == "__main__":
    unittest.main()
