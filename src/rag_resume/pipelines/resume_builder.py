from __future__ import annotations

import dataclasses
import typing
from enum import Enum, auto
from typing import final

from attr import dataclass

from rag_resume.graph.edges import CommonGraphStates, PipelineEdge
from rag_resume.pipelines.types import AsyncPipelineAction, PipelineAction, PipelineProtocol

if typing.TYPE_CHECKING:
    from rag_resume.llms.chat import ChatLLMProtocol


class ResumeBuilderSteps(Enum):
    """Resume builder pipeline steps."""

    LOOKUP_EXPRIENCE = auto()
    GENERATE_BULLET_POINTS = auto()


@dataclass
class ResumeBuilderState:
    """Resume builder state."""

    description: str
    exprience: list[str]
    bullet_points: list[str]


@final
class ResumeBuilderPipeline(PipelineProtocol[ResumeBuilderSteps, ResumeBuilderState]):
    """Pipeline implementation for ResumeBuilder task."""

    steps_type = ResumeBuilderSteps
    state_type = ResumeBuilderState

    def __init__(self, chat_llm: ChatLLMProtocol) -> None:
        """Initialize ResumeBuilderPipeline."""
        self.chat_llm = chat_llm
        self.graph_edges = [
            PipelineEdge(
                CommonGraphStates.START,
                ResumeBuilderSteps.LOOKUP_EXPRIENCE,
            ),
            PipelineEdge(
                ResumeBuilderSteps.LOOKUP_EXPRIENCE,
                ResumeBuilderSteps.GENERATE_BULLET_POINTS,
            ),
            PipelineEdge(ResumeBuilderSteps.LOOKUP_EXPRIENCE, CommonGraphStates.END),
        ]

    def lookup(self, state: ResumeBuilderState) -> ResumeBuilderState:
        """Lookup experience based on the query.

        Args:
            state (ResumeBuilderState): The current state of the pipeline, containing a query and an answer.

        Returns:
            ResumeBuilderState: The updated state of the pipeline after looking up experience.
        """
        return dataclasses.replace(state, description="Lookup experience")

    def generate(self, state: ResumeBuilderState) -> ResumeBuilderState:
        """Generate bullet points based on the query and answer.

        Args:
            state (ResumeBuilderState): The current state of the pipeline, containing a query and an answer.

        Returns:
            ResumeBuilderState: The updated state of the pipeline after generating bullet points.
        """
        return dataclasses.replace(state, description="Lookup experience")

    def implementation_for(
        self, step: ResumeBuilderSteps
    ) -> PipelineAction[ResumeBuilderState] | AsyncPipelineAction[ResumeBuilderState]:
        """Implementation for each step in the pipeline.

        Args:
            step (ResumeBuilderSteps): The current step in the pipeline.

        Returns:
            PipelineAction[ResumeBuilderState] | AsyncPipelineAction[ResumeBuilderState]: The action to be taken
                for the given step.
        """
        match step:
            case ResumeBuilderSteps.LOOKUP_EXPRIENCE:
                return self.lookup
            case ResumeBuilderSteps.GENERATE_BULLET_POINTS:
                return self.generate
            case _:
                typing.assert_never(step)
