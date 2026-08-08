"""Ranked-list fusion: RRF and weighted linear combination.

Fusion merges ranked result lists (one per retrieval channel) into a
single ordering. The two supported strategies are reciprocal rank fusion
(``"rrf"``) and max-normalised linear scoring (``"linear"``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from raghub.models import Hit


class Fusion:
    """Fuse ranked result lists using RRF or weighted linear scoring."""

    def __init__(self, method: str = "rrf", *, k: int = 60) -> None:
        """Initialise fusion with ``method`` and RRF damping constant.

        Args:
            method: ``"rrf"`` (default) or ``"linear"``.
            k: RRF damping constant.

        Raises:
            ValueError: When ``method`` is unknown or ``k < 1``.

        """
        if method not in {"rrf", "linear"}:
            raise ValueError(f"Unknown fusion method: {method!r}")
        if k < 1:
            raise ValueError("rrf k must be >= 1")
        self.method = method
        self.k = k

    def fuse(self, lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse lists of result dictionaries keyed by ``chunk_id``."""
        if self.method == "rrf":
            scores: dict[str, float] = {}
            records: dict[str, dict[str, Any]] = {}
            for ranking in lists:
                for rank, item in enumerate(ranking, start=1):
                    chunk_id = item["chunk_id"]
                    if not isinstance(chunk_id, str):
                        raise ValueError(f"chunk_id must be str, got {type(chunk_id).__name__}")
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.k + rank)
                    records.setdefault(chunk_id, item)
            return [
                records[key] | {"score": score}
                for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]
        linear_scores: dict[str, float] = {}
        records_linear: dict[str, dict[str, Any]] = {}
        for ranking in lists:
            maximum = max((float(item.get("score", 0)) for scored_record in ranking), default=0.0) or 1.0
            for scored_record in ranking:
                key = item["chunk_id"]
                if not isinstance(key, str):
                    raise ValueError(f"chunk_id must be str, got {type(key).__name__}")
                linear_scores[key] = (
                    linear_scores.get(key, 0.0) + float(item.get("score", 0)) / maximum
                )
                records_linear.setdefault(key, item)
        return [
            records_linear[key] | {"score": score}
            for key, score in sorted(linear_scores.items(), key=lambda x: x[1], reverse=True)
        ]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse ordered chunk-id rankings with reciprocal rank fusion.

    Args:
        rankings: Each inner list is a ranked channel of chunk ids.
        k: RRF damping constant.

    Returns:
        A list of ``(chunk_id, score)`` tuples sorted by descending
        fused score.

    Raises:
        ValueError: When ``k < 1`` or any chunk id is not a ``str``.

    """
    rows: list[list[dict[str, Any]]] = []
    for ranking in rankings:
        rows.append([{"chunk_id": scored_record} for scored_record in ranking])
    fused = Fusion(k=k).fuse(rows)
    return [(row["chunk_id"], float(row["score"])) for row in fused]


def linear_combine(
    channel_scores: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Combine max-normalised channel scores.

    Args:
        channel_scores: ``{channel_name: {chunk_id: score}}`` mapping.
        weights: Optional per-channel multiplier AFTER per-channel
            max normalisation.

    Returns:
        A list of ``(chunk_id, fused_score)`` tuples sorted by descending score.

    """
    weight_map = dict(weights or {})
    scores: dict[str, float] = {}
    for channel_name, channel in channel_scores.items():
        if not channel:
            continue
        max_score = max(channel.values())
        if max_score <= 0:
            max_score = 1.0
        weight = weight_map.get(channel_name, 1.0)
        for chunk_id, raw in channel.items():
            normalised = float(raw) / max_score
            scores[chunk_id] = scores.get(chunk_id, 0.0) + normalised * weight
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def merge_rrf(per_window: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Reciprocal-Rank-Fusion merge across ranked windows."""
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranked in per_window:
        for rank, hit in enumerate(ranked, start=1):
            cid = hit.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in order:
                order[cid] = len(order)
    return sorted(
        {hit.chunk_id: hit for window in per_window for hit in window}.values(),
        key=lambda h: (-scores.get(h.chunk_id, 0.0), order.get(h.chunk_id, 0)),
    )


__all__ = [
    "Fusion",
    "linear_combine",
    "merge_rrf",
    "reciprocal_rank_fusion",
]
