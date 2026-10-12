"""Pinned readouts for the Decider, JPT, and Eikos letter families on tiny local models.

The expected values were recorded before the families moved onto the shared
``LabelReadoutModel`` base. A refactor that changes a prompt, an option order,
a seeded permutation, or a temperature changes these numbers.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from futurix_decision import DecisionInput, DecisionModel
from test_decider import make_native
from test_eikos import make_eikos
from test_jpt import make_jpt

INF = float("-inf")
INPUTS = [
    DecisionInput(
        state={"ticket": "refund", "rows": list(range(9))},
        question="Next?",
        options=["approve", "reject", "rate"],
        candidate_labels=["Approve", "Reject", "Rate"],
        candidate_descriptions=["approve one", "", "rate two"],
    ),
    DecisionInput(state="review", question="Next?", options=["approve", "reject", "one", "two"]),
    DecisionInput(
        kind="noul",
        state="review",
        question="Valid?",
        options=["false", "true"],
        candidate_descriptions=["reject", "approve"],
    ),
    DecisionInput(
        kind="noul",
        state="review",
        question="Valid?",
        options=["true", "false"],
        candidate_ids=["true", "false"],
        preserve_order=True,
    ),
    DecisionInput(
        kind="score",
        state="review",
        question="Rate",
        options=["0", "1", "2"],
        candidate_descriptions=["one", "two", "three"],
    ),
]
SEEDS = (None, 7, 11)


def readouts(model):
    """Logits per assignment seed, then inference probabilities, as plain floats."""
    model.backbone.eval()
    with torch.no_grad():
        logits = {
            seed: model.forward_decisions(INPUTS, assignment_seed=seed).logits.tolist()
            for seed in SEEDS
        }
    probabilities = [
        [result.probabilities[option] for option in row.options]
        for row, result in zip(INPUTS, model.batch_decide(INPUTS), strict=True)
    ]
    return logits, probabilities


def build(family, root: Path):
    source = root / "native"
    source.mkdir()
    if family == "decider":
        make_native(source)
        return DecisionModel.from_decider(source, max_sequence_length=512)
    if family == "eikos":
        make_eikos(source)
        return DecisionModel.from_eikos(source, max_sequence_length=512)
    return make_jpt(source)


class PinnedLetterReadoutTests(unittest.TestCase):
    def check(self, family, expected_logits, expected_probabilities):
        with tempfile.TemporaryDirectory() as temp:
            logits, probabilities = readouts(build(family, Path(temp)))
        for seed in expected_logits:
            torch.testing.assert_close(
                torch.tensor(logits[seed]),
                torch.tensor(expected_logits[seed]),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda text, seed=seed: f"{family} assignment_seed={seed}: {text}",
            )
        for row, expected in zip(probabilities, expected_probabilities, strict=True):
            torch.testing.assert_close(
                torch.tensor(row), torch.tensor(expected), rtol=1e-4, atol=1e-6
            )

    def test_decider(self):
        self.check("decider", DECIDER_LOGITS, DECIDER_PROBABILITIES)

    def test_jpt(self):
        self.check("jpt", JPT_LOGITS, JPT_PROBABILITIES)

    def test_eikos(self):
        self.check("eikos", EIKOS_LOGITS, EIKOS_PROBABILITIES)


# Recorded values. Change them only for an intended contract change.
DECIDER_LOGITS = {
    None: [
        [0.08043122, 0.09152388, -0.04118405, INF],
        [0.07334791, 0.08124228, -0.04589369, -0.1390357],
        [0.05230901, 0.05847468, INF, INF],
        [0.0619899, 0.0513541, INF, INF],
        [-0.6882052, -0.6887507, -0.6888834, INF],
    ],
    7: [
        [0.08047193, -0.04119643, 0.09146392, INF],
        [-0.0461775, -0.1395451, 0.08131003, 0.07351512],
        [0.05855851, 0.05211005, INF, INF],
        [0.05140677, 0.06207035, INF, INF],
        [-0.6882052, -0.6887507, -0.6888834, INF],
    ],
    11: [
        [0.09149978, -0.04119584, 0.08040035, INF],
        [0.08082727, -0.1395893, 0.07345722, -0.04598754],
        [0.05855851, 0.05211005, INF, INF],
        [0.0619899, 0.0513541, INF, INF],
        [-0.6882052, -0.6887507, -0.6888834, INF],
    ],
}
DECIDER_PROBABILITIES = [
    [0.3452272, 0.349078, 0.3056949],
    [0.2699658, 0.2721054, 0.2396198, 0.218309],
    [0.4984586, 0.5015414],
    [0.5026589, 0.497341],
    [0.3334693, 0.3332874, 0.3332433],
]
JPT_LOGITS = {
    None: [
        [-0.01320741, 0.07735346, -0.04303917, INF],
        [-0.01332549, 0.07865295, -0.04206258, -6.901698e-05],
        [0.07923828, -0.01260851, INF, INF],
        [-0.01218623, 0.08021969, INF, INF],
        [-0.01209825, 0.08038843, -0.04105712, INF],
    ],
    7: [
        [-0.01326246, -0.04291257, 0.07742677, INF],
        [-0.04190293, 6.923291e-06, 0.0787688, -0.01340582],
        [0.07923828, -0.01260851, INF, INF],
        [0.08027572, -0.01217638, INF, INF],
        [-0.04114251, -0.01207194, 0.08043429, INF],
    ],
    11: [
        [0.07765294, -0.04277515, -0.01313191, INF],
        [0.07872554, -5.281522e-05, -0.01331302, -0.04203128],
        [0.07923828, -0.01260851, INF, INF],
        [-0.01218623, 0.08021969, INF, INF],
        [0.08046283, -0.01205321, -0.04107755, INF],
    ],
}
JPT_PROBABILITIES = [
    [0.3264811, 0.3563047, 0.3172142],
    [0.2451964, 0.2679611, 0.2384885, 0.2483541],
    [0.5221493, 0.4778507],
    [0.477716, 0.5222839],
    [0.3261777, 0.3566358, 0.3171864],
]
# Eikos permutes choice and noul under a seed; its positional score scale stays in order.
EIKOS_LOGITS = {
    None: [
        [0.1253583, 0.1037952, -0.03176053, INF],
        [0.1247472, 0.1037103, -0.03036545, -0.107718],
        [0.1069747, 0.124801, INF, INF],
        [0.1250303, 0.1078181, INF, INF],
        [0.1254903, 0.1056631, -0.03061498, INF],
    ],
    7: [
        [0.1253902, -0.03169285, 0.1037292, INF],
        [-0.02985991, -0.1077066, 0.1040488, 0.124862],
        [0.1247497, 0.1069114, INF, INF],
        [0.1077233, 0.1250299, INF, INF],
        [0.1254903, 0.1056631, -0.03061498, INF],
    ],
    11: [
        [0.1038686, -0.03159346, 0.1253517, INF],
        [0.1038232, -0.107774, 0.1248425, -0.03006401],
        [0.1247497, 0.1069114, INF, INF],
        [0.1250303, 0.1078181, INF, INF],
        [0.1254903, 0.1056631, -0.03061498, INF],
    ],
}
EIKOS_PROBABILITIES = [
    [0.3529491, 0.3454198, 0.3016311],
    [0.275628, 0.2698902, 0.2360254, 0.2184565],
    [0.4955435, 0.5044565],
    [0.504303, 0.4956971],
    [0.3526295, 0.3457068, 0.3016637],
]


if __name__ == "__main__":
    unittest.main()
