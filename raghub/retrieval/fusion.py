"""Hybrid-retrieval fusion helpers (Phase 3.2).

Two strategies are exported:

* :func:`rrf` — Reciprocal Rank Fusion, the literature default
  (Cormack et al., 2009). Channel-agnostic; only the per-channel
  ranks matter.
* :func:`linear_combine` — Weighted sum of per-channel scores after
  max-normalisation. The legacy :class:`RetrievalPipeline.retrieve_hybrid`
  uses this; kept here so the new RRF path can be opted in without
  breaking the linear path's tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def rrf(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    Each ranking is an ordered sequence of chunk ids (best first).
    For every chunk id, the fused score is the sum of
    ``1 / (k + rank)`` across the rankings in which it appears.
    The standard literature constant is ``k = 60``; smaller ``k``
    weights the top ranks more aggressively.

    Args:
        rankings: One sequence per channel (e.g. dense, sparse,
            ColBERT). Chunks present in only some channels still
            contribute from the channels that did see them.
        k: Rank damping constant. Must be ``>= 1``; the default of
            ``60`` matches the original paper.

    Returns:
        Pairs of ``(chunk_id, fused_score)`` sorted by descending
        score. Empty input yields an empty list.

    Raises:
        ValueError: When ``k < 1`` or any ranking contains a
            non-string entry.
    """
    if k < 1:
        raise ValueError("rrf k must be >= 1")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, str):
                raise ValueError(
                    f"rrf expects string chunk ids, got {type(item).__name__}"
                )
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def linear_combine(
    channel_scores: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Weighted linear combination of per-channel max-normalised scores.

    Args:
        channel_scores: Channel name → ``{chunk_id: raw_score}``.
            Empty channels are tolerated — they contribute nothing.
        weights: Optional per-channel weights. Missing channels
            default to ``1.0``. The absolute scale does not matter
            because every channel is max-normalised to ``[0, 1]``
            before fusion.

    Returns:
        Pairs of ``(chunk_id, fused_score)`` sorted by descending
        score. Empty input yields an empty list.
    """
    if not channel_scores:
        return []
    weights = dict(weights or {})
    aggregated: dict[str, float] = {}
    for channel, by_id in channel_scores.items():
        if not by_id:
            continue
        max_score = max(by_id.values())
        if max_score <= 0:
            continue
        weight = float(weights.get(channel, 1.0))
        for chunk_id, raw in by_id.items():
            aggregated[chunk_id] = aggregated.get(chunk_id, 0.0) + (raw / max_score) * weight
    return sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)


__all__ = ["linear_combine", "rrf"]