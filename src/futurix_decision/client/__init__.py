"""HTTP clients for the ``futurix-decision serve`` API. They do not need torch."""

from futurix_decision.client.decision import AsyncDecisionClient, DecisionClient

__all__ = ["AsyncDecisionClient", "DecisionClient"]
