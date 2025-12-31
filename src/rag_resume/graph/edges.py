from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Protocol, final

from rag_resume.pipelines.types import (
    PipelineStateType,
    PipelineStateType_contra,
    PipelineStepsType,
    PipelineStepsType_co,
)


class PipelineEdgeLike(Protocol[PipelineStepsType_co]):
    """Represents a directed edge in the pipeline graph.

    This protocol defines the structure of an edge that connects two steps
    in the pipeline. The `PipelineStepsType_co` type parameter specifies the
    type of the steps involved in the edge.
    """


class CommonGraphStates(Enum):
    """Common Node Types for pipelines.

    START: The starting node of the pipeline.
    END: Ending node of the pipeline.
    """

    START = auto()
    END = auto()


class DynamicPipelineCallable(Protocol[PipelineStepsType_co, PipelineStateType_contra]):
    """A callable that can dynamically determine the next state of the pipeline depending on state."""

    def __call__(self, state: PipelineStateType_contra) -> PipelineStepsType_co | CommonGraphStates:
        """Make callable."""
        ...


@final
@dataclass
class PipelineEdge(Generic[PipelineStepsType]):
    """Static pipeline edge will always move to the given step regardless of state."""

    start: PipelineStepsType | CommonGraphStates
    end: PipelineStepsType | CommonGraphStates


@final
@dataclass
class DynamicPipelineEdge(Generic[PipelineStepsType, PipelineStateType]):
    """Dynamic pipeline edge which will determine the next step based on state."""

    start: PipelineStepsType | CommonGraphStates
    end: DynamicPipelineCallable[PipelineStepsType, PipelineStateType]
