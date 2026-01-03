import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import pytest
from hamcrest import assert_that, equal_to
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from rag_resume.agentic.backends.langchain.graph import LangchainGraph
from rag_resume.agentic.graphs.edges import CommonGraphSteps, DynamicGraphEdge, GraphEdge
from rag_resume.agentic.graphs.graph import AgentGraph
from tests.helpers.graph import ParameterizedTestGraph


class SimpleTestGraphStep(Enum):
    One = 1
    Two = 2
    Three = 3


@dataclass
class SimpleTestGraphState:
    visits: int
    should_end: bool = False


@dataclass
class GraphTestCase:
    graph: ParameterizedTestGraph[SimpleTestGraphStep, SimpleTestGraphState]
    start_state: SimpleTestGraphState
    expected_end_state: SimpleTestGraphState
    assertions: Callable[[], None]


def iterate_state(state: SimpleTestGraphState) -> SimpleTestGraphState:
    return dataclasses.replace(state, visits=state.visits + 1)


def simple_graph_test_case(
    start: SimpleTestGraphState,
    mocker: MockerFixture,
) -> GraphTestCase:
    step_one = mocker.Mock(side_effect=iterate_state)
    step_two = mocker.Mock(side_effect=iterate_state)
    step_three = mocker.Mock(side_effect=iterate_state)

    def assertions() -> None:
        step_one.assert_called_once_with(start)
        step_two.assert_called_once_with(dataclasses.replace(start, visits=start.visits + 1))
        step_three.assert_called_once_with(dataclasses.replace(start, visits=start.visits + 2))

    graph = ParameterizedTestGraph(
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
    return GraphTestCase(graph, start, dataclasses.replace(start, visits=start.visits + 3), assertions)


def dynamic_simple_graph_test_case(start: SimpleTestGraphState, mocker: MockerFixture) -> GraphTestCase:
    step_one = mocker.Mock(side_effect=iterate_state)
    step_two = mocker.Mock(side_effect=iterate_state)
    step_three = mocker.Mock(side_effect=iterate_state)

    end = dataclasses.replace(start, visits=start.visits + 3) if not start.should_end else start

    def assertions() -> None:
        if not start.should_end:
            step_one.assert_called_once_with(start)
            step_two.assert_called_once_with(dataclasses.replace(start, visits=start.visits + 1))
            step_three.assert_called_once_with(dataclasses.replace(start, visits=start.visits + 2))
        else:
            step_one.assert_not_called()
            step_two.assert_not_called()
            step_three.assert_not_called()

    graph = ParameterizedTestGraph(
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
    return GraphTestCase(graph, start, end, assertions)


def looped_graph_case(start: SimpleTestGraphState, mocker: MockerFixture) -> GraphTestCase:
    step_one = mocker.Mock(side_effect=iterate_state)
    step_two = mocker.Mock(side_effect=iterate_state)
    step_three = mocker.Mock(side_effect=iterate_state)

    graph = ParameterizedTestGraph(
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

    end = dataclasses.replace(start, visits=start.visits + 4)

    def assertions() -> None:
        step_one.assert_has_calls(
            calls=[mocker.call(start), mocker.call(dataclasses.replace(start, visits=start.visits + 3))]  # pyright: ignore[reportArgumentType]
        )
        step_two.assert_called_once_with(dataclasses.replace(start, visits=start.visits + 1))
        step_three.assert_called_once_with(dataclasses.replace(start, visits=start.visits + 2))

    return GraphTestCase(graph, start, end, assertions)


TEST_CASES = (simple_graph_test_case, dynamic_simple_graph_test_case, looped_graph_case)


@given(state=st.tuples(st.integers(), st.booleans()))
@pytest.mark.parametrize(
    ("graph_test_case_generator", "graph_backend"), itertools.product(TEST_CASES, (LangchainGraph,))
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_graph_invoke_for_implementation(
    state: tuple[int, bool],
    graph_test_case_generator: Callable[[SimpleTestGraphState, MockerFixture], GraphTestCase],
    graph_backend: type[AgentGraph[SimpleTestGraphStep, SimpleTestGraphState]],
    mocker: MockerFixture,
) -> None:
    """Property based testing against some basic graphs to ensure the backends generate the graphs correctly"""
    graph_test_case = graph_test_case_generator(SimpleTestGraphState(state[0], state[1]), mocker)
    graph = graph_backend(graph_test_case.graph)
    result = graph.invoke(graph_test_case.start_state)
    assert_that(result, equal_to(graph_test_case.expected_end_state))
    graph_test_case.assertions()
