"""Tests for the HeuristicProvider offline LLM."""

from __future__ import annotations

from raghub.llm import GenerationRequest, HeuristicProvider


def test_heuristic_returns_context_first_sentence():
    provider = HeuristicProvider()
    answer = provider.generate(
        GenerationRequest(
            question="revenue",
            context=["Revenue grew 12 percent in Q3.", "Other text."],
        )
    )
    assert "Revenue" in answer


def test_heuristic_returns_message_when_no_context():
    provider = HeuristicProvider()
    answer = provider.generate(GenerationRequest(question="anything", context=[]))
    assert "no context" in answer.lower() or "API key" in answer


def test_heuristic_default_model_name():
    provider = HeuristicProvider()
    assert provider.model_name == "heuristic"


def test_heuristic_picks_question_relevant_sentence():
    provider = HeuristicProvider()
    answer = provider.generate(
        GenerationRequest(
            question="blue sky",
            context=["Apples are red.", "The sky is blue today.", "Cars are fast."],
        )
    )
    assert "blue" in answer.lower()
