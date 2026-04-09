"""DiagnosisRunner: orchestrates phase 1 of the distillation loop.

Builds diagnostic tools, runs the TeacherAgent, parses failure clusters
from the teacher's output, and persists artifacts.

See spec §5.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openjarvis.learning.distillation.diagnose.teacher_agent import (
    TeacherAgent,
)
from openjarvis.learning.distillation.diagnose.tools import (
    build_diagnostic_tools,
)
from openjarvis.learning.distillation.diagnose.types import ToolCallRecord
from openjarvis.learning.distillation.models import FailureCluster

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a meta-engineer analyzing the performance of a local AI assistant \
called OpenJarvis. Your job is to diagnose why the local student model fails \
on certain tasks and identify root cause patterns.

You have access to diagnostic tools that let you:
- Browse and search the student's trace history
- Read the student's current configuration, prompts, and tools
- Re-run the student on benchmark tasks
- Run yourself on the same tasks for comparison
- Compare student vs teacher outputs

Your analysis should:
1. Identify 2-5 failure clusters — groups of related failures with shared root causes.
2. For each cluster, actually run the student and yourself on at least 3 sample tasks \
to populate student_failure_rate and teacher_success_rate with real data.
3. Describe the skill gap for each cluster.

After your analysis, output the failure clusters as a JSON array inside a \
```json code fence. Each cluster object must have these fields:
- id (string)
- description (string)
- sample_trace_ids (list of strings)
- student_failure_rate (float 0-1)
- teacher_success_rate (float 0-1)
- skill_gap (string)

Budget: max ~{max_turns} tool calls, max ${max_cost_usd:.2f} USD.
"""


@dataclass
class DiagnosisResult:
    """The output of a diagnosis run."""

    diagnosis_md: str
    clusters: list[FailureCluster] = field(default_factory=list)
    cost_usd: float = 0.0
    tool_call_records: list[ToolCallRecord] = field(default_factory=list)


class DiagnosisRunner:
    """Orchestrates phase 1 of the distillation loop.

    Parameters
    ----------
    teacher_engine :
        The CloudEngine (or mock) for teacher inference.
    teacher_model :
        Frontier model id (e.g. "claude-opus-4-6").
    trace_store :
        TraceStore for reading student traces.
    benchmark_samples :
        List of PersonalBenchmarkSample objects.
    student_runner :
        Callable to re-execute the student on a task.
    judge :
        TraceJudge for comparing outputs.
    session_dir :
        Path where session artifacts are written.
    session_id :
        Current session id.
    config :
        Dict with config_path and openjarvis_home.
    max_turns :
        Max teacher tool calls (default 30).
    max_cost_usd :
        Max teacher API cost (default 5.0).
    """

    def __init__(
        self,
        *,
        teacher_engine: Any,
        teacher_model: str,
        trace_store: Any,
        benchmark_samples: list,
        student_runner: Any,
        judge: Any,
        session_dir: Path,
        session_id: str,
        config: dict[str, Any],
        max_turns: int = 30,
        max_cost_usd: float = 5.0,
    ) -> None:
        self._teacher_engine = teacher_engine
        self._teacher_model = teacher_model
        self._trace_store = trace_store
        self._benchmark_samples = benchmark_samples
        self._student_runner = student_runner
        self._judge = judge
        self._session_dir = Path(session_dir)
        self._session_id = session_id
        self._config = config
        self._max_turns = max_turns
        self._max_cost_usd = max_cost_usd

    def run(self) -> DiagnosisResult:
        """Execute the diagnosis phase.

        Returns
        -------
        DiagnosisResult
            Contains the diagnosis markdown, parsed clusters, cost, and
            tool call records.
        """
        # Ensure session directory exists
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Build diagnostic tools
        tools = build_diagnostic_tools(
            trace_store=self._trace_store,
            config=self._config,
            benchmark_samples=self._benchmark_samples,
            student_runner=self._student_runner,
            teacher_engine=self._teacher_engine,
            teacher_model=self._teacher_model,
            judge=self._judge,
            session_id=self._session_id,
        )

        # Build system prompt with budget hints
        system_prompt = _SYSTEM_PROMPT.format(
            max_turns=self._max_turns,
            max_cost_usd=self._max_cost_usd,
        )

        # Run the teacher
        agent = TeacherAgent(
            engine=self._teacher_engine,
            model=self._teacher_model,
            tools=tools,
            max_turns=self._max_turns,
            max_cost_usd=self._max_cost_usd,
        )
        agent_result = agent.run(
            "Analyze the student's recent trace history, identify failure patterns, "
            "and produce a structured diagnosis with failure clusters.",
            system_prompt=system_prompt,
        )

        # Persist diagnosis.md
        diagnosis_path = self._session_dir / "diagnosis.md"
        diagnosis_path.write_text(agent_result.content, encoding="utf-8")

        # Persist teacher traces JSONL
        traces_dir = self._session_dir / "teacher_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = traces_dir / "diagnose.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in agent_result.tool_call_records:
                f.write(json.dumps(record.to_jsonl_dict()) + "\n")

        # Parse failure clusters from the diagnosis content
        clusters = _parse_clusters(agent_result.content)

        return DiagnosisResult(
            diagnosis_md=agent_result.content,
            clusters=clusters,
            cost_usd=agent_result.total_cost_usd,
            tool_call_records=agent_result.tool_call_records,
        )


def _parse_clusters(content: str) -> list[FailureCluster]:
    """Extract failure clusters from teacher diagnosis output.

    Looks for a JSON array inside a ```json code fence. Falls back to
    searching for any JSON array in the content. Returns an empty list
    if no valid clusters are found.
    """
    # Try to find JSON in a code fence first
    fence_match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
    if fence_match:
        try:
            return _parse_cluster_list(fence_match.group(1))
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to parse clusters from JSON code fence")

    # Fallback: try to find any JSON array in the content
    for match in re.finditer(r"\[[\s\S]*?\]", content):
        try:
            return _parse_cluster_list(match.group(0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    logger.warning("No failure clusters found in diagnosis output")
    return []


def _parse_cluster_list(json_str: str) -> list[FailureCluster]:
    """Parse a JSON string into a list of FailureCluster."""
    data = json.loads(json_str)
    if not isinstance(data, list):
        return []
    clusters = []
    for item in data:
        clusters.append(
            FailureCluster(
                id=item["id"],
                description=item["description"],
                sample_trace_ids=item.get("sample_trace_ids", []),
                student_failure_rate=item["student_failure_rate"],
                teacher_success_rate=item["teacher_success_rate"],
                skill_gap=item["skill_gap"],
            )
        )
    return clusters
