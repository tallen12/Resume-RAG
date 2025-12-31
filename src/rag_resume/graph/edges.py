from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Protocol, final

from rag_resume.graph.types import (
    GraphStateType,
    GraphStateType_contra,
    GraphStepsType,
    GraphStepsType_co,
)


class CommonGraphSteps(Enum):
    """Common Node Types for agent graphs.

    START: The starting node of the graph.
    END: Ending node of the graph.
    """

    START = auto()
    END = auto()


class DynamicGraphCallable(Protocol[GraphStepsType_co, GraphStateType_contra]):
    """A callable that can dynamically determine the next step of the graph depending on state."""

    def __call__(self, state: GraphStateType_contra) -> GraphStepsType_co | CommonGraphSteps:
        """Make callable."""
        ...


@final
@dataclass
class GraphEdge[GraphStepsType]:
    """Static graph edge that always moves to the given step regardless of state."""

    start: GraphStepsType | CommonGraphSteps
    end: GraphStepsType | CommonGraphSteps


@final
@dataclass
class DynamicGraphEdge(Generic[GraphStepsType, GraphStateType]):
    """Dynamic graph edge which will determine the next step based on state."""

    start: GraphStepsType | CommonGraphSteps
    end: DynamicGraphCallable[GraphStepsType, GraphStateType]


GraphEdgeLike = GraphEdge[GraphStepsType] | DynamicGraphEdge[GraphStepsType, GraphStateType]
