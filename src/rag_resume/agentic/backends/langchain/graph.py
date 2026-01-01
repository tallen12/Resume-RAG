from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, final, override

from langgraph.func import END, START  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import StateGraph  # pyright: ignore[reportMissingTypeStubs]

from rag_resume.agentic.graphs.edges import CommonGraphSteps, DynamicGraphCallable, DynamicGraphEdge, GraphEdge
from rag_resume.agentic.graphs.graph import GraphProtocol
from rag_resume.agentic.graphs.types import AgentGraph, AsyncAgentGraph, GraphStateType, GraphStepsType

if TYPE_CHECKING:
    from seriacade.json.types import JsonType


def _wrap_dynamic_call_return(
    edge: DynamicGraphCallable[GraphStepsType, GraphStateType],
    node_name_overrides: dict[CommonGraphSteps | GraphStepsType, str],
) -> Callable[[GraphStateType], str]:
    """Wraps DynamicGraphCallable to override the node names."""

    def wrapped(state: GraphStateType) -> str:
        next_state = edge(state)
        return node_name_overrides.get(next_state, next_state.name)

    return wrapped


def _build_lang_graph(
    impl: GraphProtocol[GraphStepsType, GraphStateType],
    node_name_overrides: dict[CommonGraphSteps | GraphStepsType, str] | None = None,
) -> StateGraph[GraphStateType, None, GraphStateType, GraphStateType]:
    """Builds a state graph for the given graph implementation.

    Args:
        impl (GraphProtocol[GraphStepsType, GraphStateType]): The implementation of a GraphProtocol
            that will generate this graph builder.
        node_name_overrides (dict[CommonGraphSteps | GraphStepsType, str] | None, optional): A dictionary mapping
            node name enums to custom strings. Defaults to None, only setting the common node names.

    Returns:
        StateGraph: The built state graph.
    """
    builder = StateGraph(impl.state_type)
    node_name_overrides = node_name_overrides or {CommonGraphSteps.START: START, CommonGraphSteps.END: END}
    for step in impl.steps_type:
        builder.add_node(node_name_overrides.get(step, step.name), impl.implementation_for(step))  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
    for edge in impl.graph_edges:
        match edge:
            case GraphEdge(start=start, end=end):
                _ = builder.add_edge(node_name_overrides.get(start, start.name), node_name_overrides.get(end, end.name))
            case DynamicGraphEdge(start=start, end=end):
                _ = builder.add_conditional_edges(
                    node_name_overrides.get(start, start.name), _wrap_dynamic_call_return(end, node_name_overrides)
                )
    return builder


@final
class LangchainGraph(AgentGraph[GraphStepsType, GraphStateType], AsyncAgentGraph[GraphStepsType, GraphStateType]):
    """Graph implementation using LangGraph."""

    def __init__(self, impl: GraphProtocol[GraphStepsType, GraphStateType]) -> None:
        """Initializes a new instance of the LangchainGraph class.

        Args:
            impl (GraphProtocol[GraphStepsType, GraphStateType]): The graph implementation to use.
        """
        self.impl = impl
        self.graph = _build_lang_graph(self.impl).compile()  # pyright: ignore[reportUnknownMemberType]

    def _to_output_type(self, **kwargs) -> GraphStateType:  # noqa: ANN003  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return self.impl.state_type(**kwargs)

    @override
    def invoke(self, initial_state: GraphStateType) -> GraphStateType:
        """Invokes the graph with a single initial state and returns the final state.

        Args:
            initial_state (GraphStateType): The initial state of the graph.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed.
        """
        return self._to_output_type(**self.graph.invoke(initial_state))  # pyright: ignore[reportUnknownMemberType, reportAny]

    @override
    def batch(self, initial_states: Sequence[GraphStateType]) -> Sequence[GraphStateType]:
        """Invokes the graph with multiple initial states and returns the final states for each.

        Args:
            initial_states (Sequence[GraphStateType]): A sequence of initial states for the graph.

        Returns:
            Sequence[GraphStateType]: A sequence of final states for each initial state after all
                steps have been executed.
        """
        return [self._to_output_type(**result) for result in self.graph.batch(list(initial_states))]  # pyright: ignore[reportUnknownMemberType, reportAny]

    @override
    async def async_invoke(self, initial_state: GraphStateType) -> GraphStateType:
        """Asynchronously invokes the graph with a single initial state and returns the final state.

        Args:
            initial_state (GraphStateType): The initial state of the graph.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed.
        """
        result: dict[str, JsonType] = await self.graph.ainvoke(initial_state)  # pyright: ignore[reportUnknownMemberType]
        return self._to_output_type(**result)  # pyright: ignore[reportUnknownMemberType]

    @override
    async def async_batch(self, initial_states: Sequence[GraphStateType]) -> Sequence[GraphStateType]:
        """Asynchronously invokes the graph with multiple initial states and returns the final states for each.

        Args:
            initial_states (Sequence[GraphStateType]): A sequence of initial states for the graph.

        Returns:
            Sequence[GraphStateType]: A sequence of final states for each initial state after all steps
                have been executed.
        """
        results = await self.graph.abatch(list(initial_states))
        return [self._to_output_type(**result) for result in results]  # pyright: ignore[reportUnknownMemberType, reportAny]
