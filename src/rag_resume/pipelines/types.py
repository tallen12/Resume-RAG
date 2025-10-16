from typing import Protocol

from rag_resume._types import PipelineStateType, PipelineStepsType
from rag_resume.graph.edges import PipelineEdgeLike


class PipelineAction(Protocol[PipelineStateType]):
    """Represents a synchronous action that can be performed on a pipeline state."""

    def __call__(self, state: PipelineStateType) -> PipelineStateType:
        """Call the PipelineAction."""
        ...


class AsyncPipelineAction(Protocol[PipelineStateType]):
    """Represents an asynchronous action that can be performed on a pipeline state."""

    async def __call__(self, state: PipelineStateType) -> PipelineStateType:
        """Call the AsyncPipelineAction."""
        ...


class PipelineProtocol(Protocol[PipelineStepsType, PipelineStateType]):
    """Protocol for pipeline graphs using a global state and typed names for steps.

    Intended to be a wrapper compatible with LangGraph API but could allow for other backends in the future.
    """

    steps_type: type[PipelineStepsType]
    state_type: type[PipelineStateType]
    graph_edges: list[PipelineEdgeLike[PipelineStepsType]]

    def implementation_for(
        self,
        step: PipelineStepsType,
    ) -> PipelineAction[PipelineStateType] | AsyncPipelineAction[PipelineStateType]:
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
