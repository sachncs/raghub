"""Ranked-list fusion: RRF and weighted linear combination.

Fusion merges ranked result lists (one per retrieval channel) into a
single ordering. The two supported strategies are reciprocal rank fusion
(``"rrf"``) and max-normalised linear scoring (``"linear"``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from raghub.models import Hit
from raghub.registry import Registry


class Fusion(Registry):
    """Polymorphic base for ranked-list fusion strategies.

    Concrete strategies register themselves via ``@Fusion.register``
    and implement :meth:`fuse`. Use :meth:`Fusion.get` to look up a
    strategy by name; the helpers below (``reciprocal_rank_fusion``,
    ``linear_combine``, ``merge_rrf``) are thin wrappers preserved for
    backward compatibility.
    """

    name: str = "fusion"

    def fuse(self, lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse lists of result dictionaries keyed by ``chunk_id``."""
        raise NotImplementedError


@Fusion.register("rrf")
class ReciprocalRankFusion(Fusion):
    """Reciprocal-rank-fusion strategy."""

    name = "rrf"

    def __init__(self, *, k: int = 60) -> None:
        """Initialise with the RRF damping constant."""
        if k < 1:
            raise ValueError("rrf k must be >= 1")
        self.k = k

    def fuse(self, lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse lists of result dictionaries keyed by ``chunk_id``."""
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


@Fusion.register("linear")
class LinearFusion(Fusion):
    """Max-normalised linear scoring strategy."""

    name = "linear"

    def fuse(self, lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse lists of result dictionaries keyed by ``chunk_id``."""
        linear_scores: dict[str, float] = {}
        records_linear: dict[str, dict[str, Any]] = {}
        for ranking in lists:
            maximum = max(
                (float(scored_record.get("score", 0)) for scored_record in ranking),
                default=0.0,
            ) or 1.0
            for scored_record in ranking:
                key = scored_record["chunk_id"]
                if not isinstance(key, str):
                    raise ValueError(f"chunk_id must be str, got {type(key).__name__}")
                linear_scores[key] = (
                    linear_scores.get(key, 0.0) + float(scored_record.get("score", 0)) / maximum
                )
                records_linear.setdefault(key, scored_record)
        return [
            records_linear[key] | {"score": score}
            for key, score in sorted(linear_scores.items(), key=lambda x: x[1], reverse=True)
        ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60
) -> list[tuple[str, float]]:
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
    fused = ReciprocalRankFusion(k=k).fuse(rows)
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
    rows: list[list[dict[str, Any]]] = []
    weight_map = dict(weights or {})
    for channel_name, channel in channel_scores.items():
        if not channel:
            continue
        max_score = max(channel.values())
        if max_score <= 0:
            max_score = 1.0
        weight = weight_map.get(channel_name, 1.0)
        rows.append(
            [
                {"chunk_id": cid, "score": float(raw) / max_score * weight}
                for cid, raw in channel.items()
            ]
        )
    fused = LinearFusion().fuse(rows)
    return [(row["chunk_id"], float(row["score"])) for row in fused]


def merge_rrf(per_window: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Reciprocal-Rank-Fusion merge across ranked :class:`Hit` windows."""
    rows: list[list[dict[str, Any]]] = []
    for window in per_window:
        rows.append([{"chunk_id": hit.chunk_id} for hit in window])
    fused = ReciprocalRankFusion(k=rrf_k).fuse(rows)
    fused_ids = {row["chunk_id"]: float(row["score"]) for row in fused}
    unique = {hit.chunk_id: hit for window in per_window for hit in window}
    return sorted(
        unique.values(),
        key=lambda h: -fused_ids.get(h.chunk_id, 0.0),
    )


__all__ = [
    "Fusion",
    "LinearFusion",
    "ReciprocalRankFusion",
    "linear_combine",
    "merge_rrf",
    "reciprocal_rank_fusion",
]
