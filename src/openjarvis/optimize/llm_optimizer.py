"""LLM-based optimizer for OpenJarvis configuration tuning.

Uses a cloud LLM to propose optimal OpenJarvis configs, inspired by DSPy's
GEPA approach: textual feedback from execution traces rather than just scalar
rewards guides the optimizer toward better configurations.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from openjarvis.core.types import Trace
from openjarvis.evals.core.backend import InferenceBackend
from openjarvis.evals.core.types import RunSummary
from openjarvis.optimize.types import SearchSpace, TrialConfig, TrialResult


class LLMOptimizer:
    """Uses a cloud LLM to propose optimal OpenJarvis configs.

    Inspired by DSPy's GEPA: uses textual feedback from execution
    traces rather than just scalar rewards.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        optimizer_model: str = "claude-sonnet-4-6",
        optimizer_backend: Optional[InferenceBackend] = None,
    ) -> None:
        self.search_space = search_space
        self.optimizer_model = optimizer_model
        self.optimizer_backend = optimizer_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_initial(self) -> TrialConfig:
        """Propose a reasonable starting config from the search space."""
        if self.optimizer_backend is None:
            raise ValueError(
                "optimizer_backend is required to propose configurations"
            )

        prompt = self._build_initial_prompt()
        response = self.optimizer_backend.generate(
            prompt,
            model=self.optimizer_model,
            system="You are an expert AI systems optimizer.",
            temperature=0.7,
            max_tokens=2048,
        )
        return self._parse_config_response(response)

    def propose_next(
        self,
        history: List[TrialResult],
        traces: Optional[List[Trace]] = None,
    ) -> TrialConfig:
        """Ask the LLM to propose the next config to evaluate."""
        if self.optimizer_backend is None:
            raise ValueError(
                "optimizer_backend is required to propose configurations"
            )

        prompt = self._build_propose_prompt(history, traces)
        response = self.optimizer_backend.generate(
            prompt,
            model=self.optimizer_model,
            system="You are an expert AI systems optimizer.",
            temperature=0.7,
            max_tokens=2048,
        )
        return self._parse_config_response(response)

    def analyze_trial(
        self,
        trial: TrialConfig,
        summary: RunSummary,
        traces: Optional[List[Trace]] = None,
    ) -> str:
        """Ask the LLM to analyze a completed trial. Returns textual analysis."""
        if self.optimizer_backend is None:
            raise ValueError(
                "optimizer_backend is required to analyze trials"
            )

        prompt = self._build_analyze_prompt(trial, summary, traces)
        response = self.optimizer_backend.generate(
            prompt,
            model=self.optimizer_model,
            system="You are an expert AI systems analyst.",
            temperature=0.3,
            max_tokens=2048,
        )
        return response.strip()

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_initial_prompt(self) -> str:
        """Construct the prompt for the initial config proposal."""
        lines: List[str] = []
        lines.append(
            "You are optimizing an OpenJarvis AI system configuration."
        )
        lines.append("")
        lines.append(self.search_space.to_prompt_description())
        lines.append("## Objective")
        lines.append(
            "Maximize accuracy while minimizing latency and cost."
        )
        lines.append("")
        lines.append("## Your Task")
        lines.append(
            "Propose an initial configuration that is a reasonable starting "
            "point for optimization. Choose sensible defaults that balance "
            "accuracy, latency, and cost."
        )
        lines.append("")
        lines.append(
            "Return a JSON object inside a ```json code block with:"
        )
        lines.append(
            '1. "params": dict of config params (dotted keys matching '
            "the search space)"
        )
        lines.append(
            '2. "reasoning": string explaining why this is a good '
            "starting configuration"
        )
        return "\n".join(lines)

    def _build_propose_prompt(
        self,
        history: List[TrialResult],
        traces: Optional[List[Trace]] = None,
    ) -> str:
        """Construct the full prompt for propose_next."""
        lines: List[str] = []
        lines.append(
            "You are optimizing an OpenJarvis AI system configuration."
        )
        lines.append("")
        lines.append(self.search_space.to_prompt_description())

        lines.append("## Optimization History")
        if history:
            lines.append(self._format_history(history))
        else:
            lines.append("No trials have been run yet.")
        lines.append("")

        if traces:
            lines.append("## Recent Execution Traces")
            lines.append(self._format_traces(traces))
            lines.append("")

        lines.append("## Objective")
        lines.append(
            "Maximize accuracy while minimizing latency and cost."
        )
        lines.append("")
        lines.append("## Your Task")
        lines.append(
            "Propose the next configuration to evaluate. Learn from "
            "previous trials to improve results."
        )
        lines.append("")
        lines.append(
            "Return a JSON object inside a ```json code block with:"
        )
        lines.append(
            '1. "params": dict of config params (dotted keys matching '
            "the search space)"
        )
        lines.append(
            '2. "reasoning": string explaining why this config should '
            "improve results"
        )
        return "\n".join(lines)

    def _build_analyze_prompt(
        self,
        trial: TrialConfig,
        summary: RunSummary,
        traces: Optional[List[Trace]] = None,
    ) -> str:
        """Construct the prompt for analyze_trial."""
        lines: List[str] = []
        lines.append("Analyze this OpenJarvis evaluation result.")
        lines.append("")

        lines.append("## Configuration")
        for key, value in sorted(trial.params.items()):
            lines.append(f"- {key}: {value}")
        if trial.reasoning:
            lines.append(f"\nOptimizer reasoning: {trial.reasoning}")
        lines.append("")

        lines.append("## Results")
        lines.append(f"- accuracy: {summary.accuracy:.4f}")
        lines.append(
            f"- mean_latency_seconds: {summary.mean_latency_seconds:.4f}"
        )
        lines.append(f"- total_cost_usd: {summary.total_cost_usd:.4f}")
        lines.append(f"- total_samples: {summary.total_samples}")
        lines.append(f"- scored_samples: {summary.scored_samples}")
        lines.append(f"- correct: {summary.correct}")
        lines.append(f"- errors: {summary.errors}")
        if summary.per_subject:
            lines.append("\n### Per-Subject Breakdown")
            for subject, metrics in sorted(summary.per_subject.items()):
                metrics_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in sorted(metrics.items())
                )
                lines.append(f"- {subject}: {metrics_str}")
        lines.append("")

        if traces:
            lines.append("## Sample Traces")
            lines.append(self._format_traces(traces))
            lines.append("")

        lines.append(
            "Provide a detailed textual analysis of what worked, what "
            "failed, and what changes would likely improve results."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_history(self, history: List[TrialResult]) -> str:
        """Render trial history as structured text for the LLM prompt."""
        lines: List[str] = []
        for i, result in enumerate(history, 1):
            lines.append(f"### Trial {i} (id={result.trial_id})")
            lines.append(f"Params: {json.dumps(result.config.params)}")
            lines.append(f"Accuracy: {result.accuracy:.4f}")
            lines.append(
                f"Latency: {result.mean_latency_seconds:.4f}s"
            )
            lines.append(f"Cost: ${result.total_cost_usd:.4f}")
            if result.analysis:
                lines.append(f"Analysis: {result.analysis}")
            if result.failure_modes:
                lines.append(
                    f"Failure modes: {', '.join(result.failure_modes)}"
                )
            lines.append("")
        return "\n".join(lines)

    def _format_traces(self, traces: List[Trace]) -> str:
        """Render traces as structured text for the LLM prompt.

        Limits to the last 10 traces and truncates long outputs to keep
        the prompt manageable.
        """
        max_traces = 10
        max_result_len = 500
        max_steps_per_trace = 10

        recent = traces[-max_traces:]
        lines: List[str] = []

        for trace in recent:
            lines.append(
                f"### Trace {trace.trace_id} "
                f"(agent={trace.agent}, model={trace.model})"
            )
            lines.append(f"Query: {trace.query}")
            if trace.outcome:
                lines.append(f"Outcome: {trace.outcome}")
            if trace.feedback is not None:
                lines.append(f"Feedback: {trace.feedback}")
            lines.append(
                f"Latency: {trace.total_latency_seconds:.3f}s, "
                f"Tokens: {trace.total_tokens}"
            )

            # Show steps (limited)
            steps = trace.steps[:max_steps_per_trace]
            if steps:
                lines.append("Steps:")
                for step in steps:
                    step_input = json.dumps(step.input)
                    step_output = json.dumps(step.output)
                    if len(step_input) > max_result_len:
                        step_input = step_input[:max_result_len] + "..."
                    if len(step_output) > max_result_len:
                        step_output = step_output[:max_result_len] + "..."
                    lines.append(
                        f"  - {step.step_type.value}: "
                        f"input={step_input}, "
                        f"output={step_output} "
                        f"({step.duration_seconds:.3f}s)"
                    )
                if len(trace.steps) > max_steps_per_trace:
                    lines.append(
                        f"  ... ({len(trace.steps) - max_steps_per_trace} "
                        "more steps)"
                    )

            result_text = trace.result
            if len(result_text) > max_result_len:
                result_text = result_text[:max_result_len] + "..."
            lines.append(f"Result: {result_text}")
            lines.append("")

        return "\n".join(lines)

    def _parse_config_response(self, response: str) -> TrialConfig:
        """Extract a TrialConfig from an LLM response.

        Looks for a ```json ... ``` block first, then falls back to
        finding a raw JSON object in the response text.
        """
        trial_id = uuid.uuid4().hex[:12]

        # Try to extract from a ```json code block
        json_block_match = re.search(
            r"```json\s*\n?(.*?)\n?\s*```", response, re.DOTALL
        )
        if json_block_match:
            raw_json = json_block_match.group(1).strip()
            try:
                data = json.loads(raw_json)
                return self._config_from_dict(data, trial_id)
            except json.JSONDecodeError:
                pass

        # Try to extract from a generic ``` code block
        code_block_match = re.search(
            r"```\s*\n?(.*?)\n?\s*```", response, re.DOTALL
        )
        if code_block_match:
            raw_json = code_block_match.group(1).strip()
            try:
                data = json.loads(raw_json)
                return self._config_from_dict(data, trial_id)
            except json.JSONDecodeError:
                pass

        # Try to find a raw JSON object in the response by scanning
        # for each '{' and attempting to parse from that position.
        decoder = json.JSONDecoder()
        for m in re.finditer(r"\{", response):
            try:
                data, _ = decoder.raw_decode(response, m.start())
                if isinstance(data, dict):
                    return self._config_from_dict(data, trial_id)
            except json.JSONDecodeError:
                continue

        # Last resort: return empty config
        return TrialConfig(
            trial_id=trial_id,
            params={},
            reasoning="Failed to parse LLM response.",
        )

    def _config_from_dict(
        self, data: Dict[str, Any], trial_id: str
    ) -> TrialConfig:
        """Build a TrialConfig from a parsed JSON dict."""
        params = data.get("params", {})
        reasoning = data.get("reasoning", "")
        return TrialConfig(
            trial_id=trial_id,
            params=params,
            reasoning=reasoning,
        )


__all__ = ["LLMOptimizer"]
