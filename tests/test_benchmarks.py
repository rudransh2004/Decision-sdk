import json
import unittest

from futurix_decision.datasets import normalize_decision
from futurix_decision.datasets.benchmarks import normalize_decisionbench
from futurix_decision.evaluation import evaluate
from test_bosun import bosun_tiny_model


class BenchmarkTests(unittest.TestCase):
    def test_binary_true_first_retains_native_candidate_order(self):
        row = {
            "row_id": "retrieval-1",
            "primitive": "binary_classification",
            "instruction": "Is it relevant?",
            "state_json": '{"query":"test"}',
            "candidates_json": json.dumps(
                [
                    {"id": "true", "label": "Relevant", "description": "Answers the query"},
                    {"id": "false", "label": "Not relevant", "description": None},
                ]
            ),
            "gold_candidate_id": "false",
            "gold_probabilities": [0.0, 1.0],
            "domain": "web",
            "family": "retrieval_verification",
        }
        decision = normalize_decision(row)
        self.assertEqual(decision.options, ["true", "false"])
        self.assertEqual(decision.candidate_labels, ["Relevant", "Not relevant"])
        self.assertEqual(decision.target_distribution(), [0.0, 1.0])
        self.assertEqual(decision.family, "retrieval_verification")
        report = evaluate(bosun_tiny_model(), [decision])
        self.assertEqual(report["by_kind"]["noul"]["count"], 1)
        self.assertEqual(report["by_family"]["retrieval_verification"]["count"], 1)
        self.assertEqual(report["by_domain"]["web"]["count"], 1)

    def test_binary_arbitrary_ids_keep_bosun_noul_type(self):
        row = {
            "row_id": "tool-safety-gate-000000",
            "primitive": "binary_classification",
            "instruction": "Decide whether to approve.",
            "state_json": '{"call":"delete"}',
            "candidates_json": json.dumps(
                [
                    {"id": "c000", "label": "Reject", "description": None},
                    {"id": "c001", "label": "Approve", "description": None},
                ]
            ),
            "gold_candidate_id": "c000",
            "gold_probabilities": [1.0, 0.0],
        }
        decision = normalize_decision(row)
        self.assertEqual(decision.kind, "noul")
        self.assertEqual(decision.options, ["c000", "c001"])
        self.assertIn('"type":"noul"', bosun_tiny_model()._prompt_and_slots(decision, None)[0])

    def test_ordered_scoring_preserves_ids_and_ordinal_values(self):
        row = {
            "row_id": "score-1",
            "primitive": "ordinal_scoring",
            "instruction": "Rate it",
            "state_json": '"state"',
            "candidates_json": json.dumps(
                [
                    {"id": "c1", "label": "High", "ordinal_value": 1.0, "description": None},
                    {"id": "c0", "label": "Low", "ordinal_value": 0.0, "description": None},
                ]
            ),
            "gold_candidate_id": "c1",
            "gold_probabilities": [1.0, 0.0],
        }
        decision = normalize_decisionbench(row)
        self.assertEqual(decision.options, ["c1", "c0"])
        self.assertEqual(decision.candidate_ids, ["c1", "c0"])
        self.assertEqual(decision.target_distribution(), [1.0, 0.0])
        self.assertEqual(evaluate(bosun_tiny_model(), [decision])["by_kind"]["score"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
