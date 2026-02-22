"""Core data types for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class EvalRecord:
    """A single evaluation sample."""

    record_id: str
    problem: str
    reference: str
    category: str  # "chat" | "reasoning" | "rag" | "agentic"
    subject: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    """Result of evaluating a single sample."""

    record_id: str
    model_answer: str
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    scoring_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunConfig:
    """Configuration for an evaluation run."""

    benchmark: str
    backend: str
    model: str
    max_samples: Optional[int] = None
    max_workers: int = 4
    temperature: float = 0.0
    max_tokens: int = 2048
    judge_model: str = "gpt-4o"
    engine_key: Optional[str] = None
    agent_name: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    seed: int = 42
    dataset_split: Optional[str] = None


@dataclass(slots=True)
class RunSummary:
    """Summary statistics for a completed evaluation run."""

    benchmark: str
    category: str
    backend: str
    model: str
    total_samples: int
    scored_samples: int
    correct: int
    accuracy: float
    errors: int
    mean_latency_seconds: float
    total_cost_usd: float
    per_subject: Dict[str, Dict[str, float]] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0


__all__ = ["EvalRecord", "EvalResult", "RunConfig", "RunSummary"]
