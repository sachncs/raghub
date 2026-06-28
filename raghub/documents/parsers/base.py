from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSection:
    section_index: int
    source_location: str
    text: str
    metadata: dict


class FileParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        ...
