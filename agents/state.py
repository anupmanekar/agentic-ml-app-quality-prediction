"""
LangGraph shared state schema.
Every node reads from and writes to this state.
"""
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Drift detection outputs
    drift_detected: bool
    drift_report: dict[str, Any]          # PSI/KS scores per feature

    # Root cause analysis
    root_cause_summary: str               # LLM-generated narrative

    # Experiment design
    experiment_strategy: dict[str, Any]   # Proposed params / feature changes
    experiment_iteration: int             # Tracks retry count

    # MLflow run tracking
    champion_run_id: str                  # Current production model run ID
    challenger_run_id: str                # New experiment run ID
    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]

    # Evaluation outcome
    evaluation_passed: bool
    evaluation_summary: str

    # Human approval
    human_decision: str                   # "approve" | "reject" | "pending"
    human_notes: str

    # Final output
    promotion_completed: bool
    final_report: str

    # LangGraph message thread (for LLM nodes)
    messages: Annotated[list, add_messages]
