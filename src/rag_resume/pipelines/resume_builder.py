from __future__ import annotations

import dataclasses
import json
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import final, override
from uuid import UUID

from pydantic import BaseModel
from seriacade.implementations.pydantic import PydanticJsonCodec

from rag_resume.agentic.graphs.edges import CommonGraphSteps, GraphEdge, GraphEdgeLike
from rag_resume.agentic.graphs.graph import AsyncGraphAction, GraphAction, GraphProtocol
from rag_resume.agentic.llms.chat import ChatMessage, ChatRole

if typing.TYPE_CHECKING:
    from rag_resume.agentic.llms.chat import ChatLLMProtocol
    from rag_resume.agentic.llms.embedding import VectorStoreProtocol


class ResumeBuilderSteps(Enum):
    """Enum representing the steps in the resume builder pipeline."""

    LOOKUP_EXPERIENCE = auto()
    GENERATE_BULLET_POINTS = auto()


@dataclass
class ResumeBuilderState:
    """Data class representing the state of the resume builder pipeline."""

    description: str
    experience: list[str] | None = None
    bullet_points: list[str] | None = None


class ResumeBuilderVectorMetadata(BaseModel):
    """Pydantic model for metadata used in the resume builder vector store."""

    user_name: str | None = None
    user_id: UUID | None = None


class ResumeBuilderStructuredOutput(BaseModel):
    """Pydantic model for structured output from the resume builder pipeline."""

    bullet_points: list[str]


@final
class ResumeBuilderPipeline(GraphProtocol[ResumeBuilderSteps, ResumeBuilderState]):
    """Implementation of the resume builder pipeline using graph data structures."""

    steps_type = ResumeBuilderSteps
    state_type = ResumeBuilderState

    graph_edges: Sequence[GraphEdgeLike[ResumeBuilderSteps, ResumeBuilderState]] = (
        GraphEdge(
            CommonGraphSteps.START,
            ResumeBuilderSteps.LOOKUP_EXPERIENCE,
        ),
        GraphEdge(
            ResumeBuilderSteps.LOOKUP_EXPERIENCE,
            ResumeBuilderSteps.GENERATE_BULLET_POINTS,
        ),
        GraphEdge(ResumeBuilderSteps.LOOKUP_EXPERIENCE, CommonGraphSteps.END),
    )

    def __init__(
        self, chat_llm: ChatLLMProtocol, vector_store: VectorStoreProtocol[ResumeBuilderVectorMetadata]
    ) -> None:
        """Initialize the ResumeBuilderPipeline with a chat language model and a vector store.

        Args:
            chat_llm (ChatLLMProtocol): The chat language model to use for generating responses.
            vector_store (VectorStoreProtocol[ResumeBuilderVectorMetadata]): The vector store for embeddings.
        """
        self.chat_llm = chat_llm
        self.vector_store = vector_store

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
            "prompt": "Generate bullet points for the following experience that best match this description",
            "exprience": state.experience,
        }
        response: ChatMessage = self.chat_llm.with_structured_output(
            structured_output=PydanticJsonCodec(model_type=ResumeBuilderStructuredOutput)
        ).chat(messages=[ChatMessage(role=ChatRole.USER, content=json.dumps(prompt))])
        return dataclasses.replace(state, bullet_points=response.content)

    @override
    def implementation_for(
        self, step: ResumeBuilderSteps
    ) -> GraphAction[ResumeBuilderState] | AsyncGraphAction[ResumeBuilderState]:
        """Implementation for each step in the pipeline.

        Args:
            step (ResumeBuilderSteps): The current step in the pipeline.

        Returns:
            PipelineAction[ResumeBuilderState] | AsyncPipelineAction[ResumeBuilderState]: The action to be taken
                for the given step.
        """
        match step:
            case ResumeBuilderSteps.LOOKUP_EXPERIENCE:
                return self.lookup
            case ResumeBuilderSteps.GENERATE_BULLET_POINTS:
                return self.generate
            case _:
                typing.assert_never(step)
