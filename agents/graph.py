"""
LangGraph graph definition.
Wires all nodes and edges together.
"""
from langgraph.graph import StateGraph, END
from agents.state import AgentState

# Node imports (implemented in Phase 3)
from agents.nodes.drift_monitor import drift_monitor
from agents.nodes.root_cause_analyst import root_cause_analyst
from agents.nodes.experiment_designer import experiment_designer
from agents.nodes.experiment_runner import experiment_runner
from agents.nodes.model_evaluator import model_evaluator
from agents.nodes.human_approval import human_approval
from agents.nodes.registry_manager import registry_manager
from agents.nodes.report_writer import report_writer

from configs.agent_config import MAX_EXPERIMENT_ITERATIONS


def route_after_drift(state: AgentState) -> str:
    if state["drift_detected"]:
        return "root_cause_analyst"
    return "report_writer"


def route_after_evaluation(state: AgentState) -> str:
    if state["evaluation_passed"]:
        return "human_approval"
    if state["experiment_iteration"] >= MAX_EXPERIMENT_ITERATIONS:
        return "report_writer"   # escalate, stop looping
    return "experiment_designer"  # retry loop


def route_after_human(state: AgentState) -> str:
    if state["human_decision"] == "approve":
        return "registry_manager"
    return "report_writer"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("drift_monitor", drift_monitor)
    graph.add_node("root_cause_analyst", root_cause_analyst)
    graph.add_node("experiment_designer", experiment_designer)
    graph.add_node("experiment_runner", experiment_runner)
    graph.add_node("model_evaluator", model_evaluator)
    graph.add_node("human_approval", human_approval)
    graph.add_node("registry_manager", registry_manager)
    graph.add_node("report_writer", report_writer)

    # Entry point
    graph.set_entry_point("drift_monitor")

    # Edges
    graph.add_conditional_edges("drift_monitor", route_after_drift)
    graph.add_edge("root_cause_analyst", "experiment_designer")
    graph.add_edge("experiment_designer", "experiment_runner")
    graph.add_edge("experiment_runner", "model_evaluator")
    graph.add_conditional_edges("model_evaluator", route_after_evaluation)
    graph.add_conditional_edges("human_approval", route_after_human)
    graph.add_edge("registry_manager", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()
