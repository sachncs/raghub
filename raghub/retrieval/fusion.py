"""Ranking fusion strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class RankFusion:
    """Fuse ranked result lists using RRF or weighted linear scoring.

    Attributes:
        method: ``"rrf"`` (default) or ``"linear"``.
        k: RRF damping constant. ``60`` matches the literature.
    """

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

    def fuse(self, lists: list[list[dict]]) -> list[dict]:
        """Fuse lists of result dictionaries keyed by ``chunk_id``.

        Args:
            lists: Each inner list is a ranked channel. Every item must
                be a dict carrying ``chunk_id`` (str).

        Returns:
            A list of dicts sorted by descending fused score. Each
            dict carries the original payload plus a ``score`` key.
        """
        if self.method == "rrf":
            scores: dict[str, float] = {}
            records: dict[str, dict] = {}
            for ranking in lists:
                for rank, item in enumerate(ranking, start=1):
                    chunk_id = item["chunk_id"]
                    if not isinstance(chunk_id, str):
                        raise ValueError(
                            f"chunk_id must be str, got {type(chunk_id).__name__}"
                        )
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.k + rank)
                    records.setdefault(chunk_id, item)
            return [
                records[key] | {"score": score}
                for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]
        scores: dict[str, float] = {}
        records: dict[str, dict] = {}
        for ranking in lists:
            maximum = max((float(item.get("score", 0)) for item in ranking), default=0.0) or 1.0
            for item in ranking:
                key = item["chunk_id"]
                if not isinstance(key, str):
                    raise ValueError(
                        f"chunk_id must be str, got {type(key).__name__}"
                    )
                scores[key] = scores.get(key, 0.0) + float(item.get("score", 0)) / maximum
                records.setdefault(key, item)
        return [
            records[key] | {"score": score}
            for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]


def rrf(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
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
    rows: list[list[dict]] = []
    for ranking in rankings:
        rows.append([{"chunk_id": item} for item in ranking])
    fused = RankFusion(k=k).fuse(rows)
    return [(row["chunk_id"], float(row["score"])) for row in fused]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Backward-compatible alias for :func:`rrf`."""
    return rrf(rankings, k=k)


def linear_combine(
    channel_scores: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Combine max-normalised channel scores.

    Args:
        channel_scores: ``{channel_name: {chunk_id: score}}`` mapping.
        weights: Optional per-channel multiplier applied AFTER
            per-channel max normalisation.

    Returns:
        A list of ``(chunk_id, fused_score)`` tuples sorted by
        descending score.
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


