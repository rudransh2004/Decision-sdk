"""Labeled System One records and common HF/JSONL decision rows."""

import json
import tempfile
import unittest
from pathlib import Path

from futurix_decision import DecisionTrainer, SFTConfig, evaluate, load_decisions
from futurix_decision.evaluation.evaluator import iter_examples
from futurix_decision.trainers.sft import _canonical_rows
from test_sft import tiny_model


GROUP = {
    "id": "ticket-7",
    "state": [{"role": "user", "content": "Invoice expired"}],
    "questions": {
        "action": {
            "type": "choice", "instructions": "Next action?",
            "criteria": {"approve": "Complete", "reject": "Stop", "escalate": "Ask manager"},
        },
        "valid": {
            "type": "noul", "instructions": "Is it valid?",
            "criteria": {"false": "Expired certificate", "true": "Current certificate"},
        },
        "rating": {
            "type": "score", "instructions": "Rate compliance",
            "criteria": ["low", "medium", "high"],
        },
    },
    "targets": {
        "action": {"probabilities": {"approve": 0.1, "reject": 0.8, "escalate": 0.1}},
        "valid": False,
        "rating": 2,
    },
}


class DatasetInterchangeTests(unittest.TestCase):
    def test_systemone_preserves_order_descriptions_and_grouping(self):
        rows = load_decisions(GROUP)
        self.assertEqual([row.question_id for row in rows], ["action", "valid", "rating"])
        self.assertEqual([row.id for row in rows], [
            "ticket-7:action", "ticket-7:valid", "ticket-7:rating",
        ])
        self.assertTrue(all(row.group_id == "ticket-7" for row in rows))
        self.assertEqual(rows[0].options, ["approve", "reject", "escalate"])
        self.assertEqual(rows[0].candidate_descriptions[1], "Stop")
        self.assertEqual(rows[0].target_distribution(), [0.1, 0.8, 0.1])
        self.assertEqual(rows[1].target_distribution(), [1.0, 0.0])
        self.assertEqual(rows[1].candidate_descriptions, [
            "Expired certificate", "Current certificate",
        ])
        self.assertEqual(rows[2].candidate_descriptions, ["low", "medium", "high"])
        self.assertEqual(rows[2].target_distribution(), [0.0, 0.0, 1.0])

    def test_jsonl_in_memory_and_hf_rows_share_one_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            hf_row = {
                "input": "A refund request", "query": "Next?", "choices": ["approve", "reject"],
                "label": 1,
            }
            path.write_text(json.dumps(GROUP) + "\n" + json.dumps(hf_row) + "\n")
            disk = list(load_decisions(path, streaming=True))
        memory = list(_canonical_rows([GROUP, hf_row]))
        evaluation = list(iter_examples([GROUP, hf_row]))
        self.assertEqual(len(disk), len(memory))
        self.assertEqual([row.question_id for row in disk], [
            row.question_id for row in evaluation
        ])
        self.assertEqual(disk[-1].target_distribution(), [0.0, 1.0])
        model = tiny_model()
        self.assertEqual(evaluate(model, [GROUP, hf_row], batch_size=2)["count"], 4)

    def test_systemone_rejects_missing_or_ambiguous_targets(self):
        invalid = [
            {**GROUP, "targets": {"action": "approve"}},
            {**GROUP, "targets": {**GROUP["targets"], "extra": "foo"}},
            {**GROUP, "targets": {**GROUP["targets"], "rating": 1.3}},
            {**GROUP, "questions": {**GROUP["questions"], "new": {"type": "choice"}}},
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                load_decisions(row)
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            load_decisions({**GROUP, "state": [{"broken": float("nan")}]})

    def test_grouped_questions_keep_structured_native_contract(self):
        native = {
            "type": "choice",
            "instructions": {"goal": "Route with care", "deadline": "today"},
            "criteria": {
                "billing": {"team": "Finance", "sla": 2},
                "support": {"team": "Service", "sla": 4},
            },
        }
        row = {
            "state": {"ticket": 17}, "questions": {"route": native},
            "targets": {"route": "billing"},
        }
        example = load_decisions(row)[0]
        self.assertEqual(example.native_question, native)
        self.assertEqual(example.options, ["billing", "support"])
        self.assertEqual(example.candidate_descriptions[0], '{"sla": 2, "team": "Finance"}')
        from futurix_decision.models.native import _question
        self.assertEqual(_question(example), native)

    def test_image_references_survive_for_supported_native_models(self):
        row = {**GROUP, "images": ["./invoice.png"]}
        self.assertTrue(all(item.images == ["./invoice.png"] for item in load_decisions(row)))
        with self.assertRaisesRegex(ValueError, "video and generic media"):
            load_decisions({**GROUP, "videos": ["./clip.mp4"]})
        with self.assertRaises(ValueError):
            load_decisions({**GROUP, "images": [""]})

    def test_grouped_rows_train_the_existing_futurix_head(self):
        trainer = DecisionTrainer(
            model=tiny_model(), dataset=[GROUP],
            sft_config=SFTConfig(epochs=1, batch_size=1, max_examples=3),
        )
        self.assertEqual(len(trainer.train()), 3)
        with self.assertRaisesRegex(ValueError, "text-only"):
            DecisionTrainer(
                model=tiny_model(), dataset=[{**GROUP, "images": ["./invoice.png"]}],
                sft_config=SFTConfig(epochs=1, batch_size=1),
            ).train()


if __name__ == "__main__":
    unittest.main()
