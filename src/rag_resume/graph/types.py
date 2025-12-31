from collections.abc import Sequence
from enum import Enum
from typing import Protocol, TypeVar

from pydantic import BaseModel

from rag_resume._types import DataclassLike, TypedDictLike

GraphStateTypes = DataclassLike | TypedDictLike | BaseModel

GraphStepsType = TypeVar("GraphStepsType", bound=Enum)
GraphStepsType_contra = TypeVar("GraphStepsType_contra", bound=Enum, contravariant=True)
GraphStepsType_co = TypeVar("GraphStepsType_co", bound=Enum, covariant=True)

GraphStateType = TypeVar("GraphStateType", bound=GraphStateTypes)
GraphStateType_contra = TypeVar("GraphStateType_contra", bound=GraphStateTypes, contravariant=True)
GraphStateType_co = TypeVar("GraphStateType_co", bound=GraphStateTypes, covariant=True)


class AgentGraph(Protocol[GraphStepsType_co, GraphStateType]):
    """Protocol a concrete graph to run operations on."""

    def invoke(self, initial_state: GraphStateType) -> GraphStateType:
        """Invoke the graph with an initial state and return the final state.

        Args:
            initial_state (GraphStateType): The initial state of the graph.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed.
        """
        ...

    def batch(self, initial_states: Sequence[GraphStateType]) -> Sequence[GraphStateType]:
        """Invoke the graph with a sequence of initial states and return the final state for each.

        Args:
            initial_states (Sequence[GraphStateType]): A sequence of initial states to run through the graph.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed
                for each initial state.
        """
        ...


class AsyncAgentGraph(Protocol[GraphStepsType_co, GraphStateType]):
    """Protocol a concrete graph to run operations on."""

    async def async_invoke(self, initial_state: GraphStateType) -> GraphStateType:
        """Invoke the graph with an initial state and return the final state.

        Args:
            initial_state (GraphStateType): The initial state of the graph.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed.
        """
        ...

    async def async_batch(self, initial_states: Sequence[GraphStateType]) -> Sequence[GraphStateType]:
        """Invoke the graph with a sequence of initial states and return the final state for each.

        Args:
            initial_states (Sequence[GraphStateType]): A sequence of initial states to run through the graph.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed
                for each initial state.
        """
        ...
