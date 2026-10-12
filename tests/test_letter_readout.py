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
        [0.03410782, 0.07414082, 0.04168131, -0.07355111],
        [-0.01207026, 0.01959683, INF, INF],
        [0.1232093, 0.0127697, INF, INF],
        [-0.7273787, -0.728117, -0.7280481, INF],
    ],
    7: [
        [0.08047193, -0.04119643, 0.09146392, INF],
        [0.04082557, -0.07343735, 0.07403185, 0.03442999],
        [0.02003791, -0.01176477, INF, INF],
        [0.01285106, 0.1233311, INF, INF],
        [-0.7273787, -0.728117, -0.7280481, INF],
    ],
    11: [
        [0.09149978, -0.04119584, 0.08040035, INF],
        [0.07379974, -0.07347221, 0.03416878, 0.04137857],
        [0.02003791, -0.01176477, INF, INF],
        [0.1232093, 0.0127697, INF, INF],
        [-0.7273787, -0.728117, -0.7280481, INF],
    ],
}
DECIDER_PROBABILITIES = [
    [0.3452272, 0.349078, 0.3056949],
    [0.2533967, 0.2637467, 0.2553231, 0.2275334],
    [0.4920838, 0.5079161],
    [0.5275819, 0.4724181],
    [0.3334898, 0.3332436, 0.3332666],
]
JPT_LOGITS = {
    None: [
        [-0.01320741, 0.07735346, -0.04303917, INF],
        [0.07797963, -0.02899402, -0.02087804, -0.08778731],
        [-0.07118063, 0.08362817, INF, INF],
        [0.0839373, -0.06986246, INF, INF],
        [0.002703882, 0.2758429, 0.07303166, INF],
    ],
    7: [
        [-0.01326246, -0.04291257, 0.07742677, INF],
        [-0.02084115, -0.08790261, -0.02907478, 0.07804172],
        [-0.07118063, 0.08362817, INF, INF],
        [-0.06952789, 0.08405729, INF, INF],
        [0.07311447, 0.002690734, 0.2757595, INF],
    ],
    11: [
        [0.07765294, -0.04277515, -0.01313191, INF],
        [-0.02915887, -0.08794243, 0.07807145, -0.02094669],
        [-0.07118063, 0.08362817, INF, INF],
        [0.0839373, -0.06986246, INF, INF],
        [0.2759095, 0.00266034, 0.07305123, INF],
    ],
}
JPT_PROBABILITIES = [
    [0.3264811, 0.3563047, 0.3172142],
    [0.2729986, 0.2462163, 0.2481527, 0.2326325],
    [0.462712, 0.537288],
    [0.5370458, 0.4629542],
    [0.2965677, 0.3860332, 0.317399],
]
# Eikos permutes choice and noul under a seed; its positional score scale stays in order.
EIKOS_LOGITS = {
    None: [
        [0.1253583, 0.1037952, -0.03176053, INF],
        [0.08396138, 0.09963644, -0.07495917, -0.1575558],
        [0.09892965, 0.1550923, INF, INF],
        [0.1551316, 0.0995146, INF, INF],
        [0.1186088, 0.1123042, -0.1437565, INF],
    ],
    7: [
        [0.1253902, -0.03169285, 0.1037292, INF],
        [-0.07518192, -0.1574645, 0.09951869, 0.08384471],
        [0.1549516, 0.09889916, INF, INF],
        [0.09956911, 0.1551589, INF, INF],
        [0.1186088, 0.1123042, -0.1437565, INF],
    ],
    11: [
        [0.1038686, -0.03159346, 0.1253517, INF],
        [0.09976925, -0.1574536, 0.08379163, -0.07488264],
        [0.1549516, 0.09889916, INF, INF],
        [0.1551316, 0.0995146, INF, INF],
        [0.1186088, 0.1123042, -0.1437565, INF],
    ],
}
EIKOS_PROBABILITIES = [
    [0.3529491, 0.3454198, 0.3016311],
    [0.2736503, 0.2779737, 0.2334413, 0.2149347],
    [0.485963, 0.514037],
    [0.5139006, 0.4860993],
    [0.3619326, 0.359658, 0.2784095],
]


if __name__ == "__main__":
    unittest.main()
