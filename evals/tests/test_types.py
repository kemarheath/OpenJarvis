"""Tests for core data types."""

from __future__ import annotations

from evals.core.types import EvalRecord, EvalResult, RunConfig, RunSummary


class TestEvalRecord:
    def test_creation(self):
        r = EvalRecord(
            record_id="r1", problem="What?", reference="42",
            category="reasoning",
        )
        assert r.record_id == "r1"
        assert r.problem == "What?"
        assert r.reference == "42"
        assert r.category == "reasoning"
        assert r.subject == ""
        assert r.metadata == {}

    def test_with_subject_and_metadata(self):
        r = EvalRecord(
            record_id="r2", problem="Q", reference="A",
            category="chat", subject="greet",
            metadata={"key": "val"},
        )
        assert r.subject == "greet"
        assert r.metadata == {"key": "val"}


class TestEvalResult:
    def test_defaults(self):
        r = EvalResult(record_id="r1", model_answer="42")
        assert r.is_correct is None
        assert r.score is None
        assert r.latency_seconds == 0.0
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.cost_usd == 0.0
        assert r.error is None
        assert r.scoring_metadata == {}

    def test_full(self):
        r = EvalResult(
            record_id="r1", model_answer="42", is_correct=True,
            score=1.0, latency_seconds=1.5, prompt_tokens=100,
            completion_tokens=50, cost_usd=0.01,
            scoring_metadata={"match": "exact"},
        )
        assert r.is_correct is True
        assert r.score == 1.0
        assert r.cost_usd == 0.01


class TestRunConfig:
    def test_defaults(self):
        c = RunConfig(benchmark="supergpqa", backend="jarvis-direct", model="qwen3:8b")
        assert c.max_samples is None
        assert c.max_workers == 4
        assert c.temperature == 0.0
        assert c.max_tokens == 2048
        assert c.judge_model == "gpt-4o"
        assert c.seed == 42
        assert c.tools == []

    def test_with_agent(self):
        c = RunConfig(
            benchmark="gaia", backend="jarvis-agent", model="gpt-4o",
            engine_key="cloud", agent_name="orchestrator",
            tools=["calculator", "think"],
        )
        assert c.agent_name == "orchestrator"
        assert c.tools == ["calculator", "think"]


class TestRunSummary:
    def test_creation(self):
        s = RunSummary(
            benchmark="supergpqa", category="reasoning",
            backend="jarvis-direct", model="qwen3:8b",
            total_samples=100, scored_samples=95, correct=47,
            accuracy=0.495, errors=5, mean_latency_seconds=2.1,
            total_cost_usd=0.0,
            per_subject={"math": {"accuracy": 0.5}},
        )
        assert s.accuracy == 0.495
        assert s.per_subject["math"]["accuracy"] == 0.5
        assert s.started_at == 0.0
