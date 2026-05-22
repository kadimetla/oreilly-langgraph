"""Fan-out / fan-in parallel graphs using LangGraph's Send API.

This module builds two graphs:
  1. `app`        -- coordinator -> parallel workers (SerpAPI) -> aggregator
  2. `search_app` -- minimal Send-based graph for timing (search only, no LLM)

Graph logic lives here (not in the notebook) because LangGraph runs Send nodes
in a thread pool, and the SerpAPI client corrupts Jupyter's stdout.

Usage from the notebook:
    from parallel_graphs import build_graphs, safe_invoke
    app, search_app = build_graphs(llm)
    result = safe_invoke(app, {"topic": "..."})
"""

import ast
import operator
import os
import sys
from typing import Annotated, TypedDict

from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

serp_search = load_tools(["serpapi"])[0]


# -- State -------------------------------------------------------------------

class ParallelState(TypedDict):
    topic: str
    tasks: list[str]
    results: Annotated[list[str], operator.add]
    summary: str


# -- stdout protection -------------------------------------------------------

def safe_invoke(graph, inputs, **kwargs):
    """Invoke a graph while protecting stdout from SerpAPI corruption.

    The google-search-results library closes the process-level stdout fd
    when called from threads. We dup/restore fd 1 around the call.
    """
    stdout_fd = os.dup(1)
    try:
        old_stdout = sys.stdout
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            os.dup2(devnull.fileno(), 1)
            result = graph.invoke(inputs, **kwargs)
        os.dup2(stdout_fd, 1)
        sys.stdout = old_stdout
        return result
    finally:
        os.close(stdout_fd)


# -- Node factories ----------------------------------------------------------

def _make_nodes(llm):
    """Create node functions closed over the provided LLM."""

    def coordinator_node(state: ParallelState) -> dict:
        response = llm.invoke(
            "Break the following topic into exactly 4 independent research subtasks. "
            "Each subtask should be a short search query. "
            "Return ONLY a Python list of strings, nothing else.\n\n"
            f"Topic: {state['topic']}"
        )
        tasks = ast.literal_eval(response.content.strip())
        return {"tasks": tasks}

    def worker_node(state: dict) -> dict:
        search_result = serp_search.invoke(state["task"])
        return {"results": [f"**{state['task']}**\n{search_result}"]}

    def aggregator_node(state: ParallelState) -> dict:
        joined = "\n\n".join(state["results"])
        response = llm.invoke(
            f"You received the following research results:\n\n{joined}\n\n"
            "Write a brief 3-4 sentence summary synthesizing these findings "
            f"about: {state['topic']}"
        )
        return {"summary": response.content}

    return coordinator_node, worker_node, aggregator_node


def _route_tasks(state: ParallelState) -> list[Send]:
    return [Send("worker", {"task": t}) for t in state["tasks"]]


def _fan_out_searches(state: dict) -> list[Send]:
    return [Send("search_worker", {"task": t}) for t in state["tasks"]]


def _search_worker(state: dict) -> dict:
    serp_search.invoke(state["task"])
    return {"results": [f"Done: {state['task']}"]}


# -- Public API --------------------------------------------------------------

def build_graphs(llm):
    """Return (app, search_app) ready to invoke."""

    coordinator_node, worker_node, aggregator_node = _make_nodes(llm)

    # Full graph: coordinator -> workers -> aggregator
    graph = StateGraph(ParallelState)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("worker", worker_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges("coordinator", _route_tasks, ["worker"])
    graph.add_edge("worker", "aggregator")
    graph.add_edge("aggregator", END)
    app = graph.compile()

    # Timing graph: search-only workers (no LLM)
    search_graph = StateGraph(ParallelState)
    search_graph.add_node("search_worker", _search_worker)
    search_graph.add_conditional_edges(START, _fan_out_searches, ["search_worker"])
    search_graph.add_edge("search_worker", END)
    search_app = search_graph.compile()

    return app, search_app
