"""Abstract base class for dataset providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from evals.core.types import EvalRecord


class DatasetProvider(ABC):
    """Base class for all evaluation dataset providers."""

    dataset_id: str
    dataset_name: str

    @abstractmethod
    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Load the dataset (possibly downloading from HuggingFace)."""

    @abstractmethod
    def iter_records(self) -> Iterable[EvalRecord]:
        """Iterate over loaded records."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of loaded records."""


__all__ = ["DatasetProvider"]
