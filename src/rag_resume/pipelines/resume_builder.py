from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass
from enum import Enum, auto
from typing import final, override

from pydantic import BaseModel

from rag_resume.graph.edges import CommonGraphStates, PipelineEdge
from rag_resume.llms.chat import ChatMessage, ChatRole
from rag_resume.pipelines.types import AsyncPipelineAction, PipelineAction, PipelineProtocol

if typing.TYPE_CHECKING:
    import uuid

    from rag_resume.llms.chat import ChatLLMProtocol
    from rag_resume.llms.embedding import VectorStoreProtocol


class ResumeBuilderSteps(Enum):
    """Resume builder pipeline steps."""

    LOOKUP_EXPRIENCE = auto()
    GENERATE_BULLET_POINTS = auto()


@dataclass
class ResumeBuilderState:
    """Resume builder state."""

    description: str
    exprience: list[str] | None = None
    bullet_points: list[str] | None = None


class ResumeBuilderVectorMetadata(BaseModel):
    """Metadata for resume builder vector store."""

    user_name: str | None = None
    user_id: uuid.UUID | None = None


@final
class ResumeBuilderPipeline(PipelineProtocol[ResumeBuilderSteps, ResumeBuilderState]):
    """Pipeline implementation for ResumeBuilder task."""

    steps_type = ResumeBuilderSteps
    state_type = ResumeBuilderState

    def __init__(
        self, chat_llm: ChatLLMProtocol, vector_store: VectorStoreProtocol[ResumeBuilderVectorMetadata]
    ) -> None:
        """Initialize ResumeBuilderPipeline."""
        self.chat_llm = chat_llm
        self.vector_store = vector_store
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
        exprience_docs = self.vector_store.lookup(query=state.description, filter_func=lambda _: True, top_k=4)
        return dataclasses.replace(state, exprience=[doc.content for doc in exprience_docs])

    def generate(self, state: ResumeBuilderState) -> ResumeBuilderState:
        """Generate bullet points based on the query and answer.

        Args:
            state (ResumeBuilderState): The current state of the pipeline, containing a query and an answer.

        Returns:
            ResumeBuilderState: The updated state of the pipeline after generating bullet points.
        """
        prompt = {
            "prompt": "Generate bullet points for the following exprience that best match this description",
            "exprience": state.exprience,
        }
        response = self.chat_llm.chat(messages=[ChatMessage(ChatRole.USER, content=json.dumps(prompt))])
        return dataclasses.replace(state, bullet_points=response.content)

    @override
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
