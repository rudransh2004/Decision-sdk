import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from futurix_decision.datasets import (
    DecisionExample,
    DecisionInput,
    load_decisions,
    load_open_decisions,
    normalize_decision,
)
from futurix_decision.datasets.loader import OPEN_DECISIONS_DATASET

ROW = {
    "id": "procurement-001",
    "kind": "choice",
    "state": {"amount": 4_800_000, "gst_certificate": "expired"},
    "question": "What should the workflow do next?",
    "options": ["approve", "request_document", "reject", "escalate"],
    "target": [0.02, 0.78, 0.05, 0.15],
}


class DatasetTests(unittest.TestCase):
    def test_soft_and_hard_targets(self):
        example = DecisionExample.model_validate(ROW)
        self.assertEqual(example.target_distribution(), ROW["target"])
        hard = DecisionExample.model_validate({**ROW, "target": "reject"})
        self.assertEqual(hard.target_distribution(), [0.0, 0.0, 1.0, 0.0])
        indexed = DecisionExample.model_validate({**ROW, "target": 1})
        self.assertEqual(indexed.target_distribution(), [0.0, 1.0, 0.0, 0.0])

    def test_noul_and_ordinal_levels(self):
        noul = normalize_decision(
            {"state": "ready", "query": "Proceed?", "type": "binary", "label": True}
        )
        self.assertEqual(noul.options, ["false", "true"])
        self.assertEqual(noul.target_distribution(), [0.0, 1.0])
        score = normalize_decision(
            {"state": "ready", "question": "Rate", "kind": "score", "target": 5}
        )
        self.assertEqual(score.options, ["0", "1", "2", "3", "4", "5"])
        self.assertEqual(score.target_distribution(), [0, 0, 0, 0, 0, 1.0])

    def test_external_columns_and_dataset_extras(self):
        example = normalize_decision(
            {
                "id": "v3-1",
                "kind": "noul",
                "state": "Is invoice valid?",
                "question": "Should it pass?",
                "options": ["false", "true"],
                "target": [0.11, 0.89],
                "source": "yuri_v3",
                "family": "technical",
            }
        )
        self.assertEqual(example.target_distribution(), [0.11, 0.89])
        choice = normalize_decision(
            {
                "input": {"ok": True},
                "query": "Next?",
                "choices": ["approve", "reject"],
                "probabilities": {"approve": 0.2, "reject": 0.8},
            }
        )
        self.assertEqual(choice.target_distribution(), [0.2, 0.8])

    def test_invalid_schema_is_not_silently_repaired(self):
        invalid = [
            {**ROW, "options": ["approve", "approve"]},
            {**ROW, "target": [0.0, 0.5, 0.5]},
            {**ROW, "target": [0.02, -0.1, 0.9, 0.18]},
            {**ROW, "target": "unknown"},
            {**ROW, "options": ["approve"]},
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                DecisionExample.model_validate(row)
        with self.assertRaises(ValueError):
            DecisionInput(
                kind="noul", state="ready", question="Go?", options=["true", "false"]
            )
        with self.assertRaises(ValueError):
            DecisionInput(state={"broken": float("nan")}, question="Go?", options=["a", "b"])
        with self.assertRaises(ValueError):
            DecisionInput(state={"broken": object()}, question="Go?", options=["a", "b"])

    def test_openjev_four_decimal_rounding_only(self):
        rounded = {
            **ROW, "options": ["0", "3", "4"],
            "target": [0.3333, 0.3333, 0.3333], "source": "openjev_v2",
        }
        example = normalize_decision(rounded)
        self.assertAlmostEqual(sum(example.target_distribution()), 1.0)
        self.assertAlmostEqual(
            sum(normalize_decision({**rounded, "source": "yuri_v3"}).target_distribution()), 1.0
        )
        with self.assertRaises(ValueError):
            normalize_decision({**rounded, "source": "other"})
        with self.assertRaises(ValueError):
            normalize_decision({**rounded, "target": [0.2, 0.2, 0.2]})

    def test_local_json_jsonl_and_streaming(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "decisions.json").write_text(json.dumps([ROW]), encoding="utf-8")
            (path / "decisions.jsonl").write_text(
                "\n" + json.dumps(ROW) + "\n", encoding="utf-8"
            )
            self.assertEqual(load_decisions(path / "decisions.json")[0].id, ROW["id"])
            self.assertEqual(load_decisions(path / "decisions.jsonl")[0].id, ROW["id"])
            self.assertEqual(
                next(load_decisions(path / "decisions.jsonl", streaming=True)).id,
                ROW["id"],
            )
            self.assertEqual(load_decisions(ROW)[0].id, ROW["id"])
            self.assertEqual(load_decisions([ROW])[0].id, ROW["id"])

    def test_open_decisions_uses_real_dataset_id_and_streams(self):
        with patch("futurix_decision.datasets.loader.load_decisions", return_value=iter([])) as load:
            load_open_decisions()
            load.assert_called_once_with(OPEN_DECISIONS_DATASET, split="train", streaming=True)

    def test_hub_evaluation_can_pin_dataset_revision(self):
        row = {"state": "context", "question": "Choose", "options": ["a", "b"], "target": "a"}
        with patch("datasets.load_dataset", return_value=[row]) as hub:
            result = load_decisions(
                "SargeDev/jev-distill-corpus-v3", split="test_set_30k",
                revision="a" * 40,
            )
            self.assertEqual(result[0].options, ["a", "b"])
            hub.assert_called_once_with(
                "SargeDev/jev-distill-corpus-v3", split="test_set_30k",
                streaming=False, revision="a" * 40,
            )


if __name__ == "__main__":
    unittest.main()
