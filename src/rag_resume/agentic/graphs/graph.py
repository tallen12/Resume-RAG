from collections.abc import Sequence
from typing import Protocol

from rag_resume.agentic.graphs.edges import GraphEdgeLike
from rag_resume.agentic.graphs.types import (
    GraphStateType,
    GraphStateType_contra,
    GraphStateUpdateType_co,
    GraphStepsType,
    GraphStepsType_co,
)


class GraphAction(Protocol[GraphStateType_contra, GraphStateUpdateType_co]):
    """Represents a synchronous action that can be performed on a graph state.

    Args:
        state (GraphStateType): The current state of the graph. This state is modified
            by the action and returned as the new state.

    Returns:
        GraphStateType: The updated state of the graph after executing the action.
    """

    def __call__(self, state: GraphStateType_contra) -> GraphStateUpdateType_co:
        """Call the GraphAction."""
        ...


class AsyncGraphAction(Protocol[GraphStateType_contra, GraphStateUpdateType_co]):
    """Represents an asynchronous action that can be performed on a graph state.

    Args:
        state (GraphStateType): The current state of the graph. This state is modified
            by the action and returned as the new state.

    Returns:
        GraphStateType: The updated state of the graph after executing the action.
    """

    async def __call__(self, state: GraphStateType_contra) -> GraphStateUpdateType_co:
        """Call the AsyncGraphAction."""
        ...


class GraphProtocol(Protocol[GraphStepsType, GraphStateType, GraphStateUpdateType_co]):
    """Protocol for graphs using a global state and typed names for steps.

    This protocol defines the interface for graphs that use a global state and typed names
    for steps. It is intended to be a wrapper compatible with LangGraph API but could allow
    for other backends in the future.

    Attributes:
        GraphStepsType (type[GraphStepsType]): The type of the steps in the graph. These represent the different
            stages or operations in the graph workflow.
        GraphStateType (type[GraphStateType]): The type of the graph state that this graph operates on. The state
            contains the data that is passed through the graph steps.
        graph_edges (Sequence[GraphEdgeLike[GraphStepsType, GraphStateType]]): A sequence of graph edges that define the
             connections between steps in the graph. Each edge represents a relationship between steps.
    """

    steps_type: type[GraphStepsType]
    state_type: type[GraphStateType]
    graph_edges: Sequence[GraphEdgeLike[GraphStepsType, GraphStateType]]

    def implementation_for(
        self,
        step: GraphStepsType,
    ) -> (
        GraphAction[GraphStateType, GraphStateUpdateType_co] | AsyncGraphAction[GraphStateType, GraphStateUpdateType_co]
    ):
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


class AgentGraph(Protocol[GraphStepsType_co, GraphStateType, GraphStateUpdateType_co]):
    """Protocol defining the interface for a synchronous agent graph.

    This protocol defines the operations that a concrete agent graph must implement
    to run workflows in a synchronous manner.

    Attributes:
        GraphStepsType_co: Covariant type variable representing the step type in the graph.
        GraphStateType: Type variable representing the state type in the graph.
    """

    def __init__(self, impl: GraphProtocol[GraphStepsType_co, GraphStateType, GraphStateUpdateType_co]) -> None: ...

    def invoke(self, initial_state: GraphStateType) -> GraphStateType:
        """Invoke the graph with an initial state and return the final state.

        Args:
            initial_state (GraphStateType): The initial state of the graph. This should
                be an instance of the GraphStateType type variable, which can be a
                DataclassLike, TypedDictLike, or BaseModel.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed.
                This is an instance of the GraphStateType type variable.
        """
        ...

    def batch(self, initial_states: Sequence[GraphStateType]) -> Sequence[GraphStateType]:
        """Invoke the graph with a sequence of initial states and return the final state for each.

        Args:
            initial_states (Sequence[GraphStateType]): A sequence of initial states to run
                through the graph. Each element in the sequence should be an instance of
                the GraphStateType type variable.

        Returns:
            Sequence[GraphStateType]: A sequence of final states, where each element
                corresponds to the final state of the graph after all steps have been
                executed for each initial state in the input sequence.
        """
        ...


class AsyncAgentGraph(Protocol[GraphStepsType_co, GraphStateType, GraphStateUpdateType_co]):
    """Protocol defining the interface for an asynchronous agent graph.

    This protocol defines the operations that a concrete agent graph must implement
    to run workflows in an asynchronous manner.

    Attributes:
        GraphStepsType_co: Covariant type variable representing the step type in the graph.
        GraphStateType: Type variable representing the state type in the graph.
    """

    def __init__(self, impl: GraphProtocol[GraphStepsType_co, GraphStateType, GraphStateUpdateType_co]) -> None: ...

    async def async_invoke(self, initial_state: GraphStateType) -> GraphStateType:
        """Invoke the graph with an initial state and return the final state.

        Args:
            initial_state (GraphStateType): The initial state of the graph. This should
                be an instance of the GraphStateType type variable, which can be a
                DataclassLike, TypedDictLike, or BaseModel.

        Returns:
            GraphStateType: The final state of the graph after all steps have been executed.
                This is an instance of the GraphStateType type variable.
        """
        ...

    async def async_batch(self, initial_states: Sequence[GraphStateType]) -> Sequence[GraphStateType]:
        """Invoke the graph with a sequence of initial states and return the final state for each.

        Args:
            initial_states (Sequence[GraphStateType]): A sequence of initial states to run
                through the graph. Each element in the sequence should be an instance of
                the GraphStateType type variable.

        Returns:
            Sequence[GraphStateType]: A sequence of final states, where each element
                corresponds to the final state of the graph after all steps have been
                executed for each initial state in the input sequence.
        """
        ...
