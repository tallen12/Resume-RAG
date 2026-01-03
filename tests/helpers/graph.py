from collections.abc import Sequence
from typing import Generic, final

from rag_resume.agentic.graphs.edges import GraphEdgeLike
from rag_resume.agentic.graphs.graph import AsyncGraphAction, GraphAction
from rag_resume.agentic.graphs.types import GraphStateType, GraphStepsType


@final
class ParameterizedTestGraph(Generic[GraphStepsType, GraphStateType]):
    def __init__(
        self,
        edges: Sequence[GraphEdgeLike[GraphStepsType, GraphStateType]],
        implementation: dict[GraphStepsType, GraphAction[GraphStateType] | AsyncGraphAction[GraphStateType]],
        steps_type: type[GraphStepsType],
        state_type: type[GraphStateType],
    ) -> None:
        self.graph_edges = edges
        self.implementation = implementation
        self.state_type = state_type
        self.steps_type = steps_type

    def implementation_for(
        self, step: GraphStepsType
    ) -> GraphAction[GraphStateType] | AsyncGraphAction[GraphStateType]:
        return self.implementation[step]
