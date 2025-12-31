from collections.abc import Sequence
from typing import Protocol

from rag_resume.agentic.graphs.edges import GraphEdgeLike
from rag_resume.agentic.graphs.types import GraphStateType, GraphStepsType


class GraphAction(Protocol[GraphStateType]):
    """Represents a synchronous action that can be performed on a graph state."""

    def __call__(self, state: GraphStateType) -> GraphStateType:
        """Call the GraphAction."""
        ...


class AsyncGraphAction(Protocol[GraphStateType]):
    """Represents an asynchronous action that can be performed on a graph state."""

    async def __call__(self, state: GraphStateType) -> GraphStateType:
        """Call the AsyncGraphAction."""
        ...


class GraphProtocol(Protocol[GraphStepsType, GraphStateType]):
    """Protocol for graphs using a global state and typed names for steps.

    Intended to be a wrapper compatible with LangGraph API but could allow for other backends in the future.
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
