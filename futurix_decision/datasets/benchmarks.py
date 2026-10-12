"""Canonical DecisionBench rows without losing candidate IDs or descriptions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from futurix_decision.datasets.schema import DecisionExample

PRIMITIVES = {
    "binary_classification": "noul",
    "candidate_selection": "choice",
    "ordinal_scoring": "score",
}


def normalize_decisionbench(row: Mapping[str, Any], *, compact: bool = False) -> DecisionExample:
    """Preserve the standard benchmark input; compact input is an explicit opt-in."""
    kind = PRIMITIVES[row["primitive"]]
    candidate_key = "compact_candidates_json" if compact else "candidates_json"
    state_key = "compact_state_json" if compact else "state_json"
    instruction_key = "compact_instruction" if compact else "instruction"
    candidates = json.loads(row[candidate_key])
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 255:
        raise ValueError("DecisionBench candidates must be a list of 2 to 255 rows")
    state = json.loads(row[state_key])
    if not isinstance(state, (str, dict)):
        raise ValueError("DecisionBench state must be a JSON string or object")
    ids = [str(candidate["id"]) for candidate in candidates]
    labels = [str(candidate["label"]) for candidate in candidates]
    descriptions = [
        "" if candidate.get("description") is None else str(candidate["description"])
        for candidate in candidates
    ]
    probabilities = row.get("gold_probabilities")
    if probabilities is None:
        probabilities = [float(candidate_id == row["gold_candidate_id"]) for candidate_id in ids]
    if len(probabilities) != len(candidates):
        raise ValueError("DecisionBench gold probabilities must align with the candidates")
    if kind == "noul":
        if len(candidates) != 2:
            raise ValueError("DecisionBench binary classification requires exactly two candidates")
        # Preserve the benchmark's presented order, even when true comes first.
        # Reordering would alter Bosun's row-ID-seeded prompt and its readout.
        options = ids
    else:
        # DecisionBench numeric choices need not be integral or sorted. Their
        # IDs and labels, not cast integer values, define the native contract.
        options = ids
    return DecisionExample(
        id=str(row["row_id"]),
        kind=kind,
        state=state,
        question=str(row[instruction_key]),
        options=options,
        candidate_ids=ids,
        candidate_labels=labels,
        candidate_descriptions=descriptions,
        preserve_order=True,
        target=[float(value) for value in probabilities],
        family=row.get("family"),
        domain=row.get("domain"),
        source="Hanno-Labs/decision-bench",
    )
