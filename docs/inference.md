# Run decisions

Every backend answers decisions through the same methods: `choice`, `noul`,
`score`, `decide`, `batch_decide`, and `close`. The `DecisionBackend` protocol
names them. A local `DecisionModel`, a `NativeDecisionModel`, a `VLLMEngine`,
a `DecisionClient`, and a `Calibrated` wrapper all implement it, so code that
takes a `DecisionBackend` works with any of them.

Use a decision-trained checkpoint, not a generic language model, for
meaningful outputs.

<!-- test -->
```python
from futurix_decision import load_model

model = load_model("./my-decision-model")
result = model.choice(
    state={"gst_certificate": "expired"},
    question="What should the workflow do next?",
    choices=["approve", "request_document", "reject", "escalate"],
    confidence_threshold=0.8,
)
print(result.model_dump())  # kind, decision, confidence, probabilities, abstained, latency_ms

binary = model.noul(state="The certificate is current.", question="Can it be approved?")
ordinal = model.score(state="Claim details", question="Rate reliability, 0 to 5.")
```

## Reading a result

`result.kind` fixes the type of `result.decision`. `noul` returns a bool,
`score` an int level, and `choice` the option string. This holds for every
backend, and whether or not the row sets `preserve_order`.

Any kind can abstain below a confidence threshold: `decision=None` and
`abstained=True`. The full probability distribution is still returned. A
`preserve_order` benchmark row whose candidate IDs are not `true`/`false` or
integer levels also has `decision=None`; read its winner from `probabilities`.

`latency_ms` is the wall time per row, so a batch of 8 that takes 80 ms
reports 10 ms for each row. `DecisionResult` is a frozen pydantic model: use
`result.model_dump()` for a dict.

`temperature` defaults to 1.0. Fit it only on held-out examples. A
`DecisionModel` keeps it in `model.config.temperature` and saves it in
`decision_config.json`. See
[calibration](datasets-and-training.md#calibration-and-proper-scores).

## Batches

`batch_decide` takes `DecisionInput` rows and runs on every backend. On a
`DecisionModel`, a batch is one padded forward pass. Pass `max_batch_tokens` to
run rows in order through chunks whose padded size stays within a token
budget, so peak memory stays flat however many rows you pass. `max_batch_tokens`
is not part of `DecisionBackend`, because an HTTP client has no tokenizer to
budget with. A `DecisionModel` also has `batch_choice` for plain dict rows.

<!-- test -->
```python
from futurix_decision import DecisionInput, load_model

model = load_model("./my-decision-model")
rows = [
    DecisionInput(state="First case", question="Next?", options=["approve", "reject"]),
    DecisionInput(state="Second case", question="Next?", options=["request_document", "escalate"]),
]
results = model.batch_decide(rows, max_batch_tokens=4096)
many = model.batch_choice(
    [
        {"state": "First case", "choices": ["approve", "reject"]},
        {"state": "Second case", "choices": ["request_document", "escalate"]},
    ]
)
```

A row's result does not depend on the other rows in its batch. Every readout
numbers positions from each row's first real token, so a batched row matches
the same row decided alone.

## Decision slots

The token readout, which `DecisionModel.from_pretrained` gives a raw decoder
checkpoint by default, reserves 256 `<|decision_NNN|>` special vocabulary
tokens. Pointer and letter-code readouts do not. For every request, candidates receive distinct decision slots; the
prompt includes each slot and its candidate text. A *stable*, seed-based
assignment is used at inference. `forward_decisions(examples,
assignment_seed=...)` returns differentiable masked logits plus assignments;
change `assignment_seed` during training to randomize candidate slots. Only
assigned logits participate in softmax. Inputs exceeding
`max_sequence_length` raise an error instead of silently losing choices.

`model.save_pretrained(path)` saves the underlying
Hugging Face model, tokenizer, and `decision_config.json`; loading it restores
the slot count, seed, maximum sequence length, and temperature. A raw decoder
checkpoint can also be loaded: this adds randomly initialized decision tokens
and requires decision training before inference is meaningful.
