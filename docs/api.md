# API reference

Everything here is exported from `futurix_decision`, which loads torch-dependent
names on first use. The clients and errors import without torch.

## Load and run models

::: futurix_decision.load_model

::: futurix_decision.DecisionBackend

::: futurix_decision.DecisionModel

::: futurix_decision.DecisionModelConfig

::: futurix_decision.NativeDecisionModel

::: futurix_decision.NativeCapabilities

::: futurix_decision.Calibrated

::: futurix_decision.PointerDecisionHead

::: futurix_decision.register_decision_head

## Decision data and results

::: futurix_decision.DecisionInput

::: futurix_decision.DecisionExample

::: futurix_decision.DecisionResult

::: futurix_decision.load_decisions

::: futurix_decision.load_open_decisions

## Training

::: futurix_decision.DecisionTrainer

::: futurix_decision.SFTConfig

::: futurix_decision.SFTTrainer

::: futurix_decision.CalibrationConfig

::: futurix_decision.CalibrationTrainer

::: futurix_decision.RLCDConfig

::: futurix_decision.RLCDTrainer

::: futurix_decision.NativeSFTConfig

## Evaluation and calibration

::: futurix_decision.evaluate

::: futurix_decision.fit_temperature

## Model families

::: futurix_decision.FamilyAdapter

::: futurix_decision.register_family

## HTTP clients

These live in `futurix_decision.client` and need only the base install.

::: futurix_decision.DecisionClient

::: futurix_decision.AsyncDecisionClient

## Errors

Defined in `futurix_decision.errors`.

::: futurix_decision.errors.FuturixError

::: futurix_decision.errors.ConfigError

::: futurix_decision.errors.CheckpointError

::: futurix_decision.errors.MissingDependencyError

::: futurix_decision.errors.BackendError

::: futurix_decision.errors.InvalidOutputError
