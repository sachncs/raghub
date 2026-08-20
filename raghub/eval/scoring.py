"""String-overlap scoring helpers shared across benchmark adapters."""

from __future__ import annotations

from raghub.runtime import capture


class Scoring:
    """Tiny string-overlap helpers shared by adapters."""

    @staticmethod
    def jaccard(predicted: str, expected: str) -> float:
        """Token-overlap (Jaccard) score.

        Args:
            predicted: Model output.
            expected: Ground truth.

        Returns:
            A score in ``[0, 1]``.

        """
        pred_tokens = set(predicted.lower().split())
        exp_tokens = set(expected.lower().split())
        if not exp_tokens:
            return 1.0 if not pred_tokens else 0.0
        union = pred_tokens | exp_tokens
        if not union:
            return 0.0
        return len(pred_tokens & exp_tokens) / len(union)

    @staticmethod
    def first_number(text: str) -> str:
        """Return the first whitespace-delimited token that parses as a number.

        Args:
            text: Arbitrary text.

        Returns:
            The first numeric token as a string. Empty if none.

        """
        for token in text.replace(",", "").split():
            parsed, _ = capture(float, token)
            if isinstance(parsed, float):
                return token
        return ""


__all__ = ["Scoring"]
