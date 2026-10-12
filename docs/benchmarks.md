# Benchmarks and run logs

This page records the runs behind the SDK's claims: what was run, with which
pins, and what came out. Dates are when a run happened. These are results from
specific runs, not guarantees. The SDK ships no pretrained Futurix model, and
no benchmark numbers are included in the package.

## Released-checkpoint smoke runs (2026-10-11)

These runs check that a path works end to end. They are not benchmark results.
The command to repeat them is in
[datasets and training](datasets-and-training.md#fine-tune-a-released-model).

- Bosun v3.1 0.6B and 1.7B: GPU load, all four inference paths, four SFT
  optimizer steps each, full export, and reload. On the two-option billing
  probe, probabilities went from 0.699 to 0.798 and 0.585 to 0.642.
- Kev-9B: GPU load, four inference paths, two native pointer-head/LoRA
  steps with the exact released and base revisions, and reload. The billing
  probability went from 0.984 to 1.000 on the training-like probe; this is
  overfitting evidence, not measured improvement.
- Laya and Laya Multilingual: GPU load, native fine-tuning, and reload
  with four and ten examples respectively. The ten-example run had just
  one calibration row: its temperature was not fitted.
- GLiNER2.5-Decide: CPU load, all four inference paths, one native training
  step on two hard-labeled rows, export under `output/final`, and reload.
  The billing probability moved from 0.99230 to 0.99273. The GPU path
  could not run on this host because CUDA NVRTC builtins were missing.
- Clef-Flash (9B) and Clef (27B): GPU load, all four inference paths,
  eight native head-only SFT steps on choice/noul/score records, complete
  publisher-format backbone/head exports and fresh GPU reloads. Billing
  probe probability changed from 0.9777 to 0.9897 (Flash) and from 0.9935
  to 0.9977 (Clef). Full-backbone training is exposed by the config but
  was not run on these large releases.
- Perplexity Decider (27B): GPU load, all four inference paths, eight
  native readout-only steps on choice/noul/score records, complete
  noncausal-backbone/readout export and fresh GPU reload. Full-backbone
  training is exposed by the config but was not run. These toy results
  do not demonstrate generalization or benchmark improvements.

## Bosun 1.7B against a raw Qwen base (2026-10-10)

Bosun 3.1 is an existing decision checkpoint, not a plain Qwen base. For how
the SDK loads it, see [model support](model-support.md#bosun). This run
compares continuing Bosun with training the raw Qwen base on the same data.

### Commands

With GPU 1 free, benchmark the untouched model and run the three SDK stages:

```bash
futurix-decision evaluate --model Hanno-Labs/bosun-v3.1-1.7b \
  --model-format bosun --dataset SargeDev/jev-distill-corpus-v3 \
  --split test_set_30k --dataset-revision 75111a13a8455032f0435050bfc7e0c9026ecfe8 \
  --gpu 1 \
  --metrics-output /home/ubuntu/futurix-runs/bosun-1.7b/jev_baseline.json
futurix-decision train --config configs/bosun_1_7b.yaml --gpu 1 \
  --max-examples 32768
# A killed or interrupted run can resume after the last completed stage:
futurix-decision train --config configs/bosun_1_7b.yaml --gpu 1 \
  --max-examples 32768 --resume
futurix-decision evaluate \
  --model /home/ubuntu/futurix-runs/bosun-1.7b/export \
  --dataset SargeDev/jev-distill-corpus-v3 --split test_set_30k \
  --dataset-revision 75111a13a8455032f0435050bfc7e0c9026ecfe8 --gpu 1
```

`--max-examples` takes a deterministic hash-based sample across the whole
training split, never a source-ordered prefix. Omit it to train on all
655,806 rows; the limit applies separately to SFT, calibration training, and
RLCD. The separate 14,111-row validation split tracks each stage; 13,766
calibration rows fit the temperature; 29,955 `test_set_30k` rows are for final
comparison, not checkpoint selection. Four-decimal rounding in the
`openjev_v2`/`yuri_v3` streams is corrected only within the source's maximum
rounding error. Stage checkpoints and validation results are saved under
`checkpoint_dir`; `--resume` verifies the dataset and stage prefix.

The raw-base experiment uses
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) at Bosun's pinned
base revision with new decision rows but the same Bosun prompt and benchmark
contract:

```bash
futurix-decision train --config configs/qwen_base_1_7b.yaml --gpu 1 \
  --max-examples 32768
```

The SDK explicitly initializes new decision-token input/output rows even when
their IDs fall inside a base model's unused padded vocabulary (as they do for
Qwen3); resizing the embedding matrix alone does not initialize such
rows. This checkpoint has no meaningful decision probabilities until trained.

### Pins

- Bosun `Hanno-Labs/bosun-v3.1-1.7b`: `1d8dc82a20e4a32ed60927a47272d6efff48eed2`.
- Raw base `Qwen/Qwen3-1.7B`: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Jev corpus: `75111a13a8455032f0435050bfc7e0c9026ecfe8` (`train`, `validation`, `calibration`, and `test_set_30k`).
- DecisionBench: `071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb` (`eval`).
- Prompt: `bosun-decision-prompt-v3-stable-slots`, ChatML with 255 candidate slots
  and slot 255 reserved for null; seed 42 selects training rows, seed 0
  assigns inference slots. Training uses only GPU 1.

### DecisionBench evaluation

For an independent hard-label benchmark, evaluate either export (or Bosun's
untouched source) using `--dataset Hanno-Labs/decision-bench --split eval
--max-sequence-length 32768`. DecisionBench has different labels and task
families from the Jev corpus; its reported accuracy must not be compared
numerically with Jev's expected correctness under soft teacher targets. Check
DecisionBench's terms before redistributing its data. No SDK or training
procedure can guarantee Bosun-level accuracy for arbitrary base models.
For high-throughput DecisionBench evaluation, first export the untouched
Bosun source with `futurix-decision export --model
Hanno-Labs/bosun-v3.1-1.7b --model-format bosun --revision
1d8dc82a20e4a32ed60927a47272d6efff48eed2 --output ./bosun-baseline`,
then run `futurix-decision evaluate --backend vllm --model ./bosun-baseline
--dataset Hanno-Labs/decision-bench --dataset-revision
071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb --split eval --gpu 1
--max-sequence-length 32768`. Use the same backend for every compared
checkpoint; floating-point kernel differences can slightly change probabilities.
For a long benchmark, pass `--predictions-output ./predictions.jsonl` and
`--metrics-output ./metrics.json`. Re-run the same command with
`--resume-evaluation` after an interruption: the SDK verifies row IDs,
options, targets, model identity, and temperature before skipping saved rows.
The CLI also caps padded inference batches at `--max-batch-tokens 16384` by
default; adjust this budget without changing the prompts or invalidating
saved predictions if long DecisionBench examples pressure GPU memory.

### Results

Both runs completed SFT, calibration, RLCD, held-out temperature fitting, and
validation-based selection of their RLCD checkpoints. Each stage used 32,768
deterministically sampled Jev training examples. The prompts, data revisions,
holdouts, evaluation code, and model size match; the configs use different
learning rates because one continues trained decision rows and the other
starts from newly initialized rows.

- Jev `test_set_30k` (29,955 soft-target rows, expected correctness):
  untouched Bosun 42.16%; Bosun SFT 60.25%, calibration 61.06%, RLCD 61.14%;
  raw Qwen SFT 59.61%, calibration 61.32%, RLCD 61.56%.
- DecisionBench `eval` (23,900 hard-label rows, accuracy, all via vLLM):
  untouched Bosun 85.06%; Bosun SFT 84.15%, calibration 84.11%, RLCD 84.07%;
  raw Qwen SFT 49.21%, calibration 50.36%, RLCD 50.91%.

The raw base slightly surpasses Bosun continuation on the Jev holdout but
falls 33.16 percentage points short on DecisionBench after RLCD. Matching a
distillation corpus does not establish Bosun-level cross-domain ability.
Do not train on DecisionBench `eval` to close this gap: obtain separate,
permitted training data for the target domains and retain an untouched test.
Full NLL, Brier, ECE, selective accuracy, coverage, and grouped metrics are
saved as JSON under `/home/ubuntu/futurix-runs/bosun-1.7b/` and
`/home/ubuntu/futurix-runs/qwen-base-1.7b/`. For stage exports use
`checkpoints/decision_sft`, `checkpoints/calibration`, or `checkpoints/rlcd`;
`export` is the validation-selected deployable checkpoint.

## What the tests cover

```bash
python -m pytest -q
FUTURIX_TEST_VLLM=1 python -m pytest -q tests/test_serving.py
FUTURIX_TEST_SERVER=1 python -m pytest -q tests/test_serving.py
```

Unit tests use tiny random checkpoints to check objectives, calibration, RLCD
convergence toward soft targets, export and reload, the API, and batching. The
two environment variables turn on the opt-in vLLM and server smoke tests; the
vLLM one needs free GPU memory. The `extras` CI jobs install each extra alone
and import what it provides, and the vLLM extras only check that their pins
resolve.

Not covered: a pretrained Futurix model, full-corpus training, a 1.7B
checkpoint in CI, multi-GPU runs, and published-weight parity for the vLLM
adapters.
