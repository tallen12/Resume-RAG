from collections.abc import Sequence
from typing import Protocol

from rag_resume.graph.edges import PipelineEdgeLike
from rag_resume.graph.types import GraphStateType, GraphStepsType


class GraphAction(Protocol[GraphStateType]):
    """Represents a synchronous action that can be performed on a pipeline state."""

    def __call__(self, state: GraphStateType) -> GraphStateType:
        """Call the PipelineAction."""
        ...


class AsyncGraphAction(Protocol[GraphStateType]):
    """Represents an asynchronous action that can be performed on a pipeline state."""

    async def __call__(self, state: GraphStateType) -> GraphStateType:
        """Call the AsyncPipelineAction."""
        ...


class GraphProtocol(Protocol[GraphStepsType, GraphStateType]):
    """Protocol for pipeline graphs using a global state and typed names for steps.

    Intended to be a wrapper compatible with LangGraph API but could allow for other backends in the future.
    """

    steps_type: type[GraphStepsType]
    state_type: type[GraphStateType]
    graph_edges: Sequence[PipelineEdgeLike[GraphStepsType, GraphStateType]]

    def implementation_for(
        self,
        step: GraphStepsType,
    ) -> GraphAction[GraphStateType] | AsyncGraphAction[GraphStateType]:
        """Return the action to take for a given step.

        This method should return either a synchronous or asynchronous pipeline action that can be executed to perform
        the specified step. The returned action will be called with the current state of the pipeline and should return
        the updated state after executing the step.

        Args:
            step (PipelineStepsType): The step for which to retrieve the implementation.

        Returns:
            PipelineAction[PipelineStateType] | AsyncPipelineAction[PipelineStateType]: A synchronous or asynchronous
                action that can be executed to perform the specified step.

        """
        ...
