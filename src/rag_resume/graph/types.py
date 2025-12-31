from collections.abc import Sequence
from typing import Protocol

from rag_resume.pipelines.types import PipelineStateType, PipelineStepsType_co


class PipelineGraph(Protocol[PipelineStepsType_co, PipelineStateType]):
    """Protocol a concrete graph to run operations on."""

    def invoke(self, initial_state: PipelineStateType) -> PipelineStateType:
        """Invoke the pipeline with an initial state and return the final state.

        Args:
            initial_state (PipelineStateType): The initial state of the pipeline.

        Returns:
            PipelineStateType: The final state of the pipeline after all steps have been executed.
        """
        ...

    def batch(self, initial_states: Sequence[PipelineStateType]) -> Sequence[PipelineStateType]:
        """Invoke the pipeline with a sequence of initial states and return the final state for each.

        Args:
            initial_states (Sequence[PipelineStateType]): A sequence of initial states to run through the pipeline.

        Returns:
            PipelineStateType: The final state of the pipeline after all steps have been executed
                for each initial state.
        """
        ...


class AsyncPipelineGraph(Protocol[PipelineStepsType_co, PipelineStateType]):
    """Protocol a concrete graph to run operations on."""

    async def async_invoke(self, initial_state: PipelineStateType) -> PipelineStateType:
        """Invoke the pipeline with an initial state and return the final state.

        Args:
            initial_state (PipelineStateType): The initial state of the pipeline.

        Returns:
            PipelineStateType: The final state of the pipeline after all steps have been executed.
        """
        ...

    async def async_batch(self, initial_states: Sequence[PipelineStateType]) -> Sequence[PipelineStateType]:
        """Invoke the pipeline with a sequence of initial states and return the final state for each.

        Args:
            initial_states (Sequence[PipelineStateType]): A sequence of initial states to run through the pipeline.

        Returns:
            PipelineStateType: The final state of the pipeline after all steps have been executed
                for each initial state.
        """
        ...
