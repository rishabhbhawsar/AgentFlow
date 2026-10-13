# Shared LangGraph state: every node reads/writes this TypedDict.
# Interview angle: keep graph-internal state as TypedDict (cheap, reducer-
# friendly) and reserve Pydantic models for API request/response at the edge.
# Trade-off vs a single big Pydantic model: Annotated reducers like
# `add_messages` are first-class in LangGraph; Pydantic needs extra glue.

"""LangGraph state schema for the multi-agent research flow."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Router outputs — keep the set small so conditional edges stay readable.
RouteName = Literal["retrieve", "web_search", "answer"]


class AgentState(TypedDict):
    """Working memory for one research request through the graph.

    Required keys are always present after the entry node. Optional keys
    use NotRequired so early nodes can omit them without fake defaults.
    """

    # Chat transcript; `add_messages` appends/merges instead of replacing.
    messages: Annotated[list[BaseMessage], add_messages]
    # Normalized user question used by retrieve / web_search / answer nodes.
    query: str
    # Set by the router; drives conditional edges. None before routing runs.
    route: NotRequired[RouteName | None]
    # Chunk texts (and later citations) gathered for the answer node.
    context: NotRequired[list[str]]
    # Final natural-language answer; None until the answer node finishes.
    answer: NotRequired[str | None]


__all__ = ["AgentState", "RouteName"]
