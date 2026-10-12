"""Pinned token, Bosun, pointer, and letter-code readouts on tiny local models.

The expected values were recorded before ``DecisionModel`` was split into
readout strategies. A change to a prompt, a slot assignment, token
registration, row initialization, or a head forward changes these numbers.
The legacy fixtures under ``tests/fixtures/legacy-0.1`` hold the
``decision_config.json`` (and head weights) that the pre-split code saved.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from futurix_decision import DecisionInput, DecisionModel, load_model
from futurix_decision.models.custom_head import HEAD_WEIGHTS
from test_bosun import bosun_tiny_model
from test_custom_head import make_model
from test_decision_model import tiny_checkpoint
from test_letter_code_conversion import raw_checkpoint
from test_letter_readout import build as build_letter_family
from test_letter_readout import readouts as letter_readouts

INF = float("-inf")
FIXTURES = Path(__file__).parent / "fixtures" / "legacy-0.1"
INPUTS = [
    DecisionInput(
        state={"ticket": "refund"},
        question="Next?",
        options=["approve", "reject", "escalate"],
        candidate_labels=["Approve", "Reject", "Escalate"],
        candidate_descriptions=["approve one", "", "send up"],
    ),
    DecisionInput(state="review", question="Next?", options=["approve", "reject"]),
    DecisionInput(kind="noul", state="review", question="Valid?", options=["false", "true"]),
    DecisionInput(
        kind="score",
        state="review",
        question="Rate",
        options=["0", "1", "2"],
        candidate_descriptions=["low", "mid", "high"],
    ),
]
SEEDS = (None, 7)


def readouts(model):
    """Assigned candidate-order logits per seed, then inference probabilities."""
    model.backbone.eval()
    if model.decision_head is not None:
        model.decision_head.eval()
    with torch.no_grad():
        logits = {}
        for seed in SEEDS:
            batch = model.forward_decisions(INPUTS, assignment_seed=seed)
            logits[seed] = [batch.assigned_logits(row).tolist() for row in range(len(INPUTS))]
    probabilities = [
        [result.probabilities[option] for option in row.options]
        for row, result in zip(INPUTS, model.batch_decide(INPUTS), strict=True)
    ]
    return logits, probabilities


def span_model():
    base = make_model()
    torch.manual_seed(29)
    return DecisionModel(
        base.backbone,
        base.tokenizer,
        head_type="pointer",
        head_prompt_version="span-v1",
        decision_tokens=16,
        max_sequence_length=256,
    )


def abstaining_bosun():
    base = bosun_tiny_model()
    return DecisionModel(
        base.backbone,
        base.tokenizer,
        decision_tokens=256,
        max_sequence_length=256,
        prompt_format="bosun-v3",
        learn_abstain=True,
    )


def letter_code_model(root: Path):
    base = root / "base"
    base.mkdir()
    raw_checkpoint(base)
    return load_model(base, convert=True, strategy="letter_code", decision_tokens=8)


BUILDERS = {
    "tokens": lambda root: tiny_checkpoint(),
    "bosun": lambda root: bosun_tiny_model(),
    "bosun-abstain": lambda root: abstaining_bosun(),
    "pointer-marker": lambda root: make_model(),
    "pointer-span": lambda root: span_model(),
    "letter-code": letter_code_model,
}


def assert_readouts(actual, expected, label):
    logits, probabilities = actual
    expected_logits, expected_probabilities = expected
    for seed, rows in expected_logits.items():
        for index, row in enumerate(rows):
            torch.testing.assert_close(
                torch.tensor(logits[seed][index]),
                torch.tensor(row),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda text, seed=seed: f"{label} assignment_seed={seed}: {text}",
            )
    for row, values in zip(probabilities, expected_probabilities, strict=True):
        torch.testing.assert_close(torch.tensor(row), torch.tensor(values), rtol=1e-4, atol=1e-6)


class PinnedReadoutTests(unittest.TestCase):
    def test_pinned_readouts(self):
        for name, expected in PINS.items():
            with self.subTest(readout=name), tempfile.TemporaryDirectory() as temp:
                assert_readouts(readouts(BUILDERS[name](Path(temp))), expected, name)

    def test_pointer_abstain_head_is_rejected_at_forward(self):
        base = make_model()
        model = DecisionModel(
            base.backbone,
            base.tokenizer,
            head_type="pointer",
            learn_abstain=True,
            decision_tokens=16,
            max_sequence_length=256,
        )
        with self.assertRaisesRegex(ValueError, "one logit per supplied option"):
            model.forward_decisions(INPUTS[:1])


class LegacyCheckpointTests(unittest.TestCase):
    """Checkpoints whose decision metadata the pre-split code wrote still load the same."""

    def reload(self, model, case: str, root: Path):
        path = root / case
        model.backbone.save_pretrained(path)
        model.tokenizer.save_pretrained(path)
        for item in (FIXTURES / case).iterdir():
            shutil.copy(item, path / item.name)
        return DecisionModel.from_pretrained(path)

    def test_futurix_readouts_reload_from_legacy_metadata(self):
        for name in ("tokens", "bosun", "pointer-marker", "pointer-span", "letter-code"):
            with self.subTest(readout=name), tempfile.TemporaryDirectory() as temp:
                model = BUILDERS[name](Path(temp))
                restored = self.reload(model, name, Path(temp))
                self.assertIs(type(restored), DecisionModel)
                assert_readouts(readouts(restored), PINS[name], name)

    def test_decider_reloads_with_its_type_temperatures(self):
        with tempfile.TemporaryDirectory() as temp:
            model = build_letter_family("decider", Path(temp))
            expected = letter_readouts(model)
            restored = self.reload(model, "decider", Path(temp))
            self.assertEqual(type(restored).__name__, "DeciderDecisionModel")
            actual = letter_readouts(restored)
        for seed, rows in expected[0].items():
            torch.testing.assert_close(torch.tensor(actual[0][seed]), torch.tensor(rows))
        for row, values in zip(actual[1], expected[1], strict=True):
            torch.testing.assert_close(torch.tensor(row), torch.tensor(values))

    def test_fixtures_hold_metadata_and_head_weights_only(self):
        # The fixtures carry metadata and head weights only; the tiny backbones
        # come from the seeded builders above.
        files = {item.name for case in FIXTURES.iterdir() for item in case.iterdir()}
        self.assertLessEqual(files, {"decision_config.json", HEAD_WEIGHTS})


# Recorded values. Change them only for an intended contract change.
PINS = {
    "tokens": (
        {
            None: [
                [-0.2191808, -0.02912955, 0.08430357],
                [0.1232546, 0.03746371],
                [-0.02601211, 0.03664925],
                [0.08801907, 0.1585255, -0.06437696],
            ],
            7: [
                [0.03308875, -0.0286766, 0.02224171],
                [-0.0280017, 0.1261919],
                [0.03641086, -0.165997],
                [-0.2250208, 0.08694691, 0.1119309],
            ],
        },
        [
            [0.2805929, 0.3393242, 0.3800828],
            [0.5214346, 0.4785655],
            [0.4843398, 0.5156602],
            [0.3410991, 0.3660169, 0.2928841],
        ],
    ),
    "bosun": (
        {
            None: [
                [0.08189026, -0.06302445, 0.1440065],
                [0.1452093, -0.06268336],
                [0.1452093, -0.06268336],
                [0.08219463, -0.06335004, 0.1446909],
            ],
            7: [
                [0.08189026, -0.06302445, 0.1440065],
                [0.1452093, -0.06268336],
                [0.1452093, -0.06268336],
                [0.08219463, -0.06335004, 0.1446909],
            ],
        },
        [
            [0.3413922, 0.2953371, 0.3632707],
            [0.5517868, 0.4482132],
            [0.5517868, 0.4482132],
            [0.3414086, 0.2951652, 0.3634262],
        ],
    ),
    "bosun-abstain": (
        {
            None: [
                [0.1633813, -0.072857, -0.01409437],
                [-0.01310465, -0.07229672],
                [-0.01310465, -0.07229672],
                [0.1635662, -0.0731116, -0.01369575],
            ],
            7: [
                [0.1633813, -0.072857, -0.01409437],
                [-0.01310465, -0.07229672],
                [-0.01310465, -0.07229672],
                [0.1635662, -0.0731116, -0.01369575],
            ],
        },
        [
            [0.3806661, 0.3005711, 0.3187627],
            [0.5147937, 0.4852063],
            [0.5147937, 0.4852063],
            [0.3806905, 0.3004583, 0.3188512],
        ],
    ),
    "pointer-marker": (
        {
            None: [
                [-0.1163997, 0.003344323, 0.1561487],
                [0.3800181, 0.4918284],
                [0.4781006, 0.3747421],
                [0.1815087, 0.1306503, 0.0208749],
            ],
            7: [
                [-0.1163997, 0.003344323, 0.1561487],
                [0.4846835, 0.3792357],
                [0.3741575, 0.478057],
                [0.1815087, 0.0208749, 0.1306503],
            ],
        },
        [
            [0.2906541, 0.3276277, 0.3817181],
            [0.4720765, 0.5279235],
            [0.5258166, 0.4741833],
            [0.3568857, 0.3391889, 0.3039253],
        ],
    ),
    "pointer-span": (
        {
            None: [
                [0.1446823, 0.2427762, 0.08757291],
                [0.4708724, 0.3427657],
                [0.4652868, 0.3390097],
                [-0.1989443, -0.1105545, 0.119188],
            ],
            7: [
                [0.1446823, 0.2427762, 0.08757291],
                [0.4731454, 0.3456565],
                [0.4711878, 0.3378934],
                [-0.1989443, -0.1105545, 0.119188],
            ],
        },
        [
            [0.3281316, 0.361951, 0.3099173],
            [0.531983, 0.468017],
            [0.5315274, 0.4684726],
            [0.2884361, 0.3150916, 0.3964722],
        ],
    ),
    "letter-code": (
        {
            None: [
                [-0.05855216, -0.05085766, 0.1148686],
                [-0.0567766, -0.05311113],
                [-0.05660435, -0.05206931],
                [-0.0569668, -0.05133426, 0.1135041],
            ],
            7: [
                [-0.05855216, -0.05085766, 0.1148686],
                [-0.0567766, -0.05311113],
                [-0.05660435, -0.05206931],
                [-0.0569668, -0.05133426, 0.1135041],
            ],
        },
        [
            [0.3127844, 0.3152004, 0.3720153],
            [0.4990836, 0.5009164],
            [0.4988662, 0.5011337],
            [0.3133312, 0.315101, 0.3715678],
        ],
    ),
}
