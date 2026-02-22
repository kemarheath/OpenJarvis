"""Tests for the EvalRunner."""

from __future__ import annotations

import json

from evals.core.runner import EvalRunner
from evals.core.types import EvalRecord, RunConfig
from evals.tests.conftest import MockBackend, MockDataset, MockScorer


class TestEvalRunner:
    def _make_records(self, n=5):
        return [
            EvalRecord(
                record_id=f"r{i}",
                problem=f"Question {i}",
                reference=f"Answer {i}",
                category="reasoning",
                subject="math" if i % 2 == 0 else "science",
            )
            for i in range(n)
        ]

    def test_basic_run(self, tmp_path):
        records = self._make_records(5)
        output_path = tmp_path / "results.jsonl"

        config = RunConfig(
            benchmark="test",
            backend="mock",
            model="test-model",
            max_workers=1,
            output_path=str(output_path),
        )

        dataset = MockDataset(records)
        backend = MockBackend()
        scorer = MockScorer(result=True)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 5
        assert summary.scored_samples == 5
        assert summary.correct == 5
        assert summary.accuracy == 1.0
        assert summary.errors == 0
        assert summary.benchmark == "test"
        assert summary.model == "test-model"

    def test_with_errors(self, tmp_path):
        records = self._make_records(3)
        output_path = tmp_path / "results.jsonl"

        config = RunConfig(
            benchmark="test",
            backend="mock",
            model="m",
            max_workers=1,
            output_path=str(output_path),
        )

        # Backend that raises on second call
        class FailingBackend(MockBackend):
            def __init__(self):
                super().__init__()
                self._fail_count = 0

            def generate_full(self, prompt, **kw):
                self._fail_count += 1
                if self._fail_count == 2:
                    raise RuntimeError("test error")
                return super().generate_full(prompt, **kw)

        dataset = MockDataset(records)
        backend = FailingBackend()
        scorer = MockScorer(result=True)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 3
        assert summary.errors == 1

    def test_per_subject_breakdown(self, tmp_path):
        records = self._make_records(4)
        output_path = tmp_path / "results.jsonl"

        config = RunConfig(
            benchmark="test",
            backend="mock",
            model="m",
            max_workers=1,
            output_path=str(output_path),
        )

        dataset = MockDataset(records)
        backend = MockBackend()
        scorer = MockScorer(result=True)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert "math" in summary.per_subject
        assert "science" in summary.per_subject
        assert summary.per_subject["math"]["accuracy"] == 1.0

    def test_jsonl_output(self, tmp_path):
        records = self._make_records(3)
        output_path = tmp_path / "results.jsonl"

        config = RunConfig(
            benchmark="test",
            backend="mock",
            model="m",
            max_workers=1,
            output_path=str(output_path),
        )

        dataset = MockDataset(records)
        backend = MockBackend()
        scorer = MockScorer(result=True)

        runner = EvalRunner(config, dataset, backend, scorer)
        runner.run()

        # Verify JSONL
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert "record_id" in first
        assert "model_answer" in first
        assert "is_correct" in first

        # Verify summary JSON
        summary_path = output_path.with_suffix(".summary.json")
        assert summary_path.exists()
        summary_data = json.loads(summary_path.read_text())
        assert summary_data["total_samples"] == 3

    def test_parallel_workers(self, tmp_path):
        records = self._make_records(10)
        output_path = tmp_path / "results.jsonl"

        config = RunConfig(
            benchmark="test",
            backend="mock",
            model="m",
            max_workers=4,
            output_path=str(output_path),
        )

        dataset = MockDataset(records)
        backend = MockBackend()
        scorer = MockScorer(result=True)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 10
        assert summary.correct == 10

    def test_mixed_scoring(self, tmp_path):
        records = self._make_records(4)
        output_path = tmp_path / "results.jsonl"

        config = RunConfig(
            benchmark="test",
            backend="mock",
            model="m",
            max_workers=1,
            output_path=str(output_path),
        )

        # Scorer that alternates correct/incorrect
        class AlternatingScorer(MockScorer):
            def __init__(self):
                super().__init__()
                self._count = 0

            def score(self, record, model_answer):
                self._count += 1
                return (self._count % 2 == 0), {"count": self._count}

        dataset = MockDataset(records)
        backend = MockBackend()
        scorer = AlternatingScorer()

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.scored_samples == 4
        assert summary.correct == 2
        assert summary.accuracy == 0.5
