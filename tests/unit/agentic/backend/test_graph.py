import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Protocol

import pytest
from hamcrest import assert_that, contains_inanyorder, equal_to
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from rag_resume.agentic.backends.langchain.graph import LangchainGraph
from rag_resume.agentic.graphs.edges import CommonGraphSteps, DynamicGraphEdge, GraphEdge
from rag_resume.agentic.graphs.graph import AgentGraph, AsyncAgentGraph
from rag_resume.changeset import ChangeSet, NoChange, ReducerChange, apply_changeset
from tests.helpers.graph import ParameterizedTestGraph


class SimpleTestGraphStep(Enum):
    One = 1
    Two = 2
    Three = 3


@dataclass
class SimpleTestGraphState:
    visits: Annotated[int, apply_changeset]
    should_end: Annotated[bool, apply_changeset] = False


@dataclass
class SimpleTestGraphUpdate:
    visits: ChangeSet[int] = field(default_factory=NoChange)
    should_end: ChangeSet[bool] = field(default_factory=NoChange)


class AssertionCall(Protocol):
    def __call__(self, batch_size: int) -> None: ...


@dataclass
class GraphTestCase:
    graph: ParameterizedTestGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate]
    start_state: SimpleTestGraphState
    expected_end_state: SimpleTestGraphState
    assertions: AssertionCall


def iterate_state(_: SimpleTestGraphState) -> SimpleTestGraphUpdate:
    return SimpleTestGraphUpdate(visits=ReducerChange(lambda current: current + 1))


# Create some simple test cases to parameterize over the backends
def simple_graph_test_case(
    start: SimpleTestGraphState,
    mocker: MockerFixture,
) -> GraphTestCase:
    """Simple graph that iterates through nodes sequentially."""
    step_one = mocker.Mock(side_effect=iterate_state)
    step_two = mocker.Mock(side_effect=iterate_state)
    step_three = mocker.Mock(side_effect=iterate_state)

    def assertions(batch_size: int = 1) -> None:
        assert_that(step_one.mock_calls, contains_inanyorder(*[mocker.call(start)] * batch_size))
        assert_that(
            step_two.mock_calls,
            contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits + 1))] * batch_size),
        )
        assert_that(
            step_three.mock_calls,
            contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits + 2))] * batch_size),
        )

    graph: ParameterizedTestGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate] = (
        ParameterizedTestGraph(
            edges=[
                GraphEdge(CommonGraphSteps.START, SimpleTestGraphStep.One),
                GraphEdge(SimpleTestGraphStep.One, SimpleTestGraphStep.Two),
                GraphEdge(SimpleTestGraphStep.Two, SimpleTestGraphStep.Three),
                GraphEdge(SimpleTestGraphStep.Three, CommonGraphSteps.END),
            ],
            implementation={
                SimpleTestGraphStep.One: step_one,
                SimpleTestGraphStep.Two: step_two,
                SimpleTestGraphStep.Three: step_three,
            },
            steps_type=SimpleTestGraphStep,
            state_type=SimpleTestGraphState,
        )
    )
    return GraphTestCase(graph, start, dataclasses.replace(start, visits=start.visits + 3), assertions)


def dynamic_simple_graph_test_case(start: SimpleTestGraphState, mocker: MockerFixture) -> GraphTestCase:
    """Create a graph that optionally ends immediately if the initial state has should_end"""
    step_one = mocker.Mock(side_effect=iterate_state)
    step_two = mocker.Mock(side_effect=iterate_state)
    step_three = mocker.Mock(side_effect=iterate_state)

    end = dataclasses.replace(start, visits=start.visits + 3) if not start.should_end else start

    def assertions(batch_size: int = 1) -> None:
        if not start.should_end:
            assert_that(step_one.mock_calls, contains_inanyorder(*[mocker.call(start)] * batch_size))
            assert_that(
                step_two.mock_calls,
                contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits + 1))] * batch_size),
            )
            assert_that(
                step_three.mock_calls,
                contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits + 2))] * batch_size),
            )
        else:
            step_one.assert_not_called()
            step_two.assert_not_called()
            step_three.assert_not_called()

    graph: ParameterizedTestGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate] = (
        ParameterizedTestGraph(
            edges=[
                DynamicGraphEdge(
                    CommonGraphSteps.START,
                    lambda state: CommonGraphSteps.END if state.should_end else SimpleTestGraphStep.One,
                ),
                GraphEdge(SimpleTestGraphStep.One, SimpleTestGraphStep.Two),
                GraphEdge(SimpleTestGraphStep.Two, SimpleTestGraphStep.Three),
                GraphEdge(SimpleTestGraphStep.Three, CommonGraphSteps.END),
            ],
            implementation={
                SimpleTestGraphStep.One: step_one,
                SimpleTestGraphStep.Two: step_two,
                SimpleTestGraphStep.Three: step_three,
            },
            steps_type=SimpleTestGraphStep,
            state_type=SimpleTestGraphState,
        )
    )
    return GraphTestCase(graph, start, end, assertions)


def looped_graph_case(start: SimpleTestGraphState, mocker: MockerFixture) -> GraphTestCase:
    """Create a graph that loops back to the start and assert states progress correctly"""
    step_one = mocker.Mock(side_effect=iterate_state)
    step_two = mocker.Mock(side_effect=iterate_state)
    step_three = mocker.Mock(side_effect=iterate_state)

    graph: ParameterizedTestGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate] = (
        ParameterizedTestGraph(
            edges=[
                GraphEdge(CommonGraphSteps.START, SimpleTestGraphStep.One),
                DynamicGraphEdge(
                    SimpleTestGraphStep.One,
                    lambda state: CommonGraphSteps.END if state.visits > start.visits + 3 else SimpleTestGraphStep.Two,
                ),
                GraphEdge(SimpleTestGraphStep.Two, SimpleTestGraphStep.Three),
                GraphEdge(SimpleTestGraphStep.Three, SimpleTestGraphStep.One),
            ],
            implementation={
                SimpleTestGraphStep.One: step_one,
                SimpleTestGraphStep.Two: step_two,
                SimpleTestGraphStep.Three: step_three,
            },
            steps_type=SimpleTestGraphStep,
            state_type=SimpleTestGraphState,
        )
    )

    end = dataclasses.replace(start, visits=start.visits + 4)

    def assertions(batch_size: int = 1) -> None:
        assert_that(
            step_one.mock_calls,
            contains_inanyorder(
                *[mocker.call(start), mocker.call(dataclasses.replace(start, visits=start.visits + 3))] * batch_size
            ),
        )
        assert_that(
            step_two.mock_calls,
            contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits + 1))] * batch_size),
        )
        assert_that(
            step_three.mock_calls,
            contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits + 2))] * batch_size),
        )

    return GraphTestCase(graph, start, end, assertions)


def parallel_graph_case(start: SimpleTestGraphState, mocker: MockerFixture) -> GraphTestCase:
    step_one = mocker.Mock(side_effect=iterate_state)
    step_two = mocker.Mock(side_effect=iterate_state)
    step_three = mocker.Mock(side_effect=iterate_state)
    graph: ParameterizedTestGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate] = (
        ParameterizedTestGraph(
            edges=[
                GraphEdge(CommonGraphSteps.START, SimpleTestGraphStep.One),
                GraphEdge(CommonGraphSteps.START, SimpleTestGraphStep.Two),
                GraphEdge(SimpleTestGraphStep.Two, SimpleTestGraphStep.Three),
                GraphEdge(SimpleTestGraphStep.Three, CommonGraphSteps.END),
            ],
            implementation={
                SimpleTestGraphStep.One: step_one,
                SimpleTestGraphStep.Two: step_two,
                SimpleTestGraphStep.Three: step_three,
            },
            steps_type=SimpleTestGraphStep,
            state_type=SimpleTestGraphState,
        )
    )

    end = dataclasses.replace(start, visits=start.visits + 3)

    def assertions(batch_size: int = 1) -> None:
        assert_that(
            step_one.mock_calls,
            contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits))] * batch_size),
        )
        assert_that(
            step_two.mock_calls,
            contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits))] * batch_size),
        )
        assert_that(
            step_three.mock_calls,
            contains_inanyorder(*[mocker.call(dataclasses.replace(start, visits=start.visits + 2))] * batch_size),
        )

    return GraphTestCase(graph, start, end, assertions)


TEST_CASES = (simple_graph_test_case, dynamic_simple_graph_test_case, looped_graph_case, parallel_graph_case)


@given(state=st.tuples(st.integers(), st.booleans()))
@pytest.mark.parametrize(
    ("graph_test_case_generator", "graph_backend"), itertools.product(TEST_CASES, (LangchainGraph,))
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_graph_invoke_for_implementation(
    state: tuple[int, bool],
    graph_test_case_generator: Callable[[SimpleTestGraphState, MockerFixture], GraphTestCase],
    graph_backend: type[AgentGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate]],
    mocker: MockerFixture,
) -> None:
    """Property based testing AgentGraph.invoke against some basic graphs to ensure the backends work correctly"""
    graph_test_case = graph_test_case_generator(SimpleTestGraphState(state[0], state[1]), mocker)
    graph: AgentGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate] = graph_backend(
        graph_test_case.graph
    )
    result = graph.invoke(graph_test_case.start_state)
    assert_that(result, equal_to(graph_test_case.expected_end_state))
    graph_test_case.assertions(batch_size=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("graph_test_case_generator", "graph_backend"), itertools.product(TEST_CASES, (LangchainGraph,))
)
@given(state=st.tuples(st.integers(), st.booleans()))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_graph_async_invoke_for_implementation(
    state: tuple[int, bool],
    graph_test_case_generator: Callable[[SimpleTestGraphState, MockerFixture], GraphTestCase],
    graph_backend: type[AsyncAgentGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate]],
    mocker: MockerFixture,
) -> None:
    """Property based testing AsyncAgentGraph.async_invoke against some basic graphs to ensure the backends work
    correctly."""
    graph_test_case = graph_test_case_generator(SimpleTestGraphState(state[0], state[1]), mocker)
    graph = graph_backend(graph_test_case.graph)
    result = await graph.async_invoke(graph_test_case.start_state)
    assert_that(result, equal_to(graph_test_case.expected_end_state))
    graph_test_case.assertions(batch_size=1)


@pytest.mark.parametrize(
    ("graph_test_case_generator", "graph_backend"), itertools.product(TEST_CASES, (LangchainGraph,))
)
@given(state=st.tuples(st.integers(), st.booleans()))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_graph_batch_for_implementation(
    state: tuple[int, bool],
    graph_test_case_generator: Callable[[SimpleTestGraphState, MockerFixture], GraphTestCase],
    graph_backend: type[AgentGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate]],
    mocker: MockerFixture,
) -> None:
    """Property based testing AgentGraph.batch against some basic graphs to ensure the backends work correctly"""
    graph_test_case = graph_test_case_generator(SimpleTestGraphState(state[0], state[1]), mocker)
    graph = graph_backend(graph_test_case.graph)
    result = graph.batch([graph_test_case.start_state] * 5)
    assert_that(result, equal_to([graph_test_case.expected_end_state] * 5))
    graph_test_case.assertions(batch_size=5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("graph_test_case_generator", "graph_backend"), itertools.product(TEST_CASES, (LangchainGraph,))
)
@given(state=st.tuples(st.integers(), st.booleans()))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_graph_async_batch_for_implementation(
    state: tuple[int, bool],
    graph_test_case_generator: Callable[[SimpleTestGraphState, MockerFixture], GraphTestCase],
    graph_backend: type[AsyncAgentGraph[SimpleTestGraphStep, SimpleTestGraphState, SimpleTestGraphUpdate]],
    mocker: MockerFixture,
) -> None:
    """Property based testing AsyncAgentGraph.async_batch against some basic graphs to ensure the backends work
    correctly,"""
    graph_test_case = graph_test_case_generator(SimpleTestGraphState(state[0], state[1]), mocker)
    graph = graph_backend(graph_test_case.graph)
    result = await graph.async_batch([graph_test_case.start_state] * 5)
    assert_that(result, equal_to([graph_test_case.expected_end_state] * 5))
    graph_test_case.assertions(batch_size=5)
