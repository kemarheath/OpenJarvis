"""Tests for openjarvis.learning.distillation.models module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEditPillar:
    """Tests for EditPillar enum."""

    def test_has_four_pillars(self) -> None:
        from openjarvis.learning.distillation.models import EditPillar

        assert EditPillar.INTELLIGENCE.value == "intelligence"
        assert EditPillar.AGENT.value == "agent"
        assert EditPillar.TOOLS.value == "tools"
        assert EditPillar.ENGINE.value == "engine"

    def test_is_string_enum(self) -> None:
        from openjarvis.learning.distillation.models import EditPillar

        assert isinstance(EditPillar.AGENT, str)
        assert EditPillar("agent") is EditPillar.AGENT


class TestEditRiskTier:
    """Tests for EditRiskTier enum."""

    def test_has_three_tiers(self) -> None:
        from openjarvis.learning.distillation.models import EditRiskTier

        assert EditRiskTier.AUTO.value == "auto"
        assert EditRiskTier.REVIEW.value == "review"
        assert EditRiskTier.MANUAL.value == "manual"


class TestEditOp:
    """Tests for EditOp enum — must contain all v1 ops plus v2 placeholders."""

    def test_intelligence_ops(self) -> None:
        from openjarvis.learning.distillation.models import EditOp

        assert EditOp.SET_MODEL_FOR_QUERY_CLASS.value == "set_model_for_query_class"
        assert EditOp.SET_MODEL_PARAM.value == "set_model_param"

    def test_agent_ops(self) -> None:
        from openjarvis.learning.distillation.models import EditOp

        assert EditOp.PATCH_SYSTEM_PROMPT.value == "patch_system_prompt"
        assert EditOp.REPLACE_SYSTEM_PROMPT.value == "replace_system_prompt"
        assert EditOp.SET_AGENT_CLASS.value == "set_agent_class"
        assert EditOp.SET_AGENT_PARAM.value == "set_agent_param"
        assert EditOp.EDIT_FEW_SHOT_EXEMPLARS.value == "edit_few_shot_exemplars"

    def test_tools_ops(self) -> None:
        from openjarvis.learning.distillation.models import EditOp

        assert EditOp.ADD_TOOL_TO_AGENT.value == "add_tool_to_agent"
        assert EditOp.REMOVE_TOOL_FROM_AGENT.value == "remove_tool_from_agent"
        assert EditOp.EDIT_TOOL_DESCRIPTION.value == "edit_tool_description"

    def test_v2_placeholder_ops(self) -> None:
        from openjarvis.learning.distillation.models import EditOp

        assert EditOp.LORA_FINETUNE.value == "lora_finetune"


class TestTriggerKind:
    """Tests for TriggerKind enum."""

    def test_four_trigger_kinds(self) -> None:
        from openjarvis.learning.distillation.models import TriggerKind

        assert TriggerKind.SCHEDULED.value == "scheduled"
        assert TriggerKind.CLUSTER.value == "cluster"
        assert TriggerKind.USER_FLAG.value == "user_flag"
        assert TriggerKind.ON_DEMAND.value == "on_demand"


class TestAutonomyMode:
    """Tests for AutonomyMode enum."""

    def test_three_modes(self) -> None:
        from openjarvis.learning.distillation.models import AutonomyMode

        assert AutonomyMode.AUTO.value == "auto"
        assert AutonomyMode.TIERED.value == "tiered"
        assert AutonomyMode.MANUAL.value == "manual"


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_all_statuses(self) -> None:
        from openjarvis.learning.distillation.models import SessionStatus

        assert SessionStatus.INITIATED.value == "initiated"
        assert SessionStatus.DIAGNOSING.value == "diagnosing"
        assert SessionStatus.PLANNING.value == "planning"
        assert SessionStatus.EXECUTING.value == "executing"
        assert SessionStatus.AWAITING_REVIEW.value == "awaiting_review"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.FAILED.value == "failed"
        assert SessionStatus.ROLLED_BACK.value == "rolled_back"
