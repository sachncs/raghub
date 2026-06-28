from __future__ import annotations

from typing import Any

from raghub.models import ChunkRecord


class Chunk:
    def __init__(self, record: ChunkRecord) -> None:
        self.record = record

    @property
    def chunk_id(self) -> str:
        return self.record.chunk_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("record",):
            super().__setattr__(name, value)
        else:
            setattr(self.record, name, value)

    def update(self, **kwargs: Any) -> Chunk:
        for key, value in kwargs.items():
            setattr(self.record, key, value)
        return self
