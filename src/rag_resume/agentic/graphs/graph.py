from collections.abc import Sequence
from typing import Protocol

from rag_resume.agentic.graphs.edges import GraphEdgeLike
from rag_resume.agentic.graphs.types import GraphStateType, GraphStepsType


class GraphAction(Protocol[GraphStateType]):
    """Represents a synchronous action that can be performed on a graph state.

    Args:
        state (GraphStateType): The current state of the graph. This state is modified
            by the action and returned as the new state.

    Returns:
        GraphStateType: The updated state of the graph after executing the action.
    """

    def __call__(self, state: GraphStateType) -> GraphStateType:
        """Call the GraphAction."""
        ...


class AsyncGraphAction(Protocol[GraphStateType]):
    """Represents an asynchronous action that can be performed on a graph state.

    Args:
        state (GraphStateType): The current state of the graph. This state is modified
            by the action and returned as the new state.

    Returns:
        GraphStateType: The updated state of the graph after executing the action.
    """

    async def __call__(self, state: GraphStateType) -> GraphStateType:
        """Call the AsyncGraphAction."""
        ...


class GraphProtocol(Protocol[GraphStepsType, GraphStateType]):
    """Protocol for graphs using a global state and typed names for steps.

    This protocol defines the interface for graphs that use a global state and typed names
    for steps. It is intended to be a wrapper compatible with LangGraph API but could allow
    for other backends in the future.

    Attributes:
        GraphStepsType (type[GraphStepsType]): The type of the steps in the graph. These represent the different
            stages or operations in the graph workflow.
        GraphStateType (type[GraphStateType]): The type of the graph state that this graph operates on. The state
            contains the data that is passed through the graph steps.
        graph_edges (Sequence[GraphEdgeLike[GraphStepsType, GraphStateType]]): A sequence of graph edges that define the connections between steps
            in the graph. Each edge represents a relationship between steps.
    """

    steps_type: type[GraphStepsType]
    state_type: type[GraphStateType]
    graph_edges: Sequence[GraphEdgeLike[GraphStepsType, GraphStateType]]

    def implementation_for(
        self,
        step: GraphStepsType,
    ) -> GraphAction[GraphStateType] | AsyncGraphAction[GraphStateType]:
        """Return the action to take for a given step.

        This method should return either a synchronous or asynchronous graph action that can be executed to perform
        the specified step. The returned action will be called with the current state of the graph and should return
        the updated state after executing the step.

        Args:
            step (GraphStepsType): The step for which to retrieve the implementation.

        Returns:
            GraphAction[GraphStateType] | AsyncGraphAction[GraphStateType]: A synchronous or asynchronous
                action that can be executed to perform the specified step.
        """
        ...
