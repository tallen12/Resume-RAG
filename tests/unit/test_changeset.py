import pytest
from hamcrest import assert_that, equal_to
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from rag_resume.changeset import ChangeSet, NoChange, OverwriteChange, ReducerChange, apply_changeset


@given(current=st.integers() | st.booleans() | st.text())
def test_no_change_apply(current: bool | str | float) -> None:
    """Test NoChange applies no change to current value."""
    change = NoChange()
    assert_that(change.apply(current), equal_to(current))


@given(
    data=st.tuples(st.integers(), st.integers())
    | st.tuples(st.booleans(), st.booleans())
    | st.tuples(st.text(), st.text())
)
def test_overwrite_change_apply(data: tuple[int, int] | tuple[bool, bool] | tuple[str, str]) -> None:
    """Test OverwriteChange applies new value over the current value."""
    current, new = data
    change = OverwriteChange(new)
    assert_that(change.apply(current), equal_to(new))


@given(
    data=st.tuples(st.integers(), st.integers())
    | st.tuples(st.booleans(), st.booleans())
    | st.tuples(st.text(), st.text())
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_reducer_change_apply(
    data: tuple[int, int] | tuple[bool, bool] | tuple[str, str], mocker: MockerFixture
) -> None:
    """Test ReducerChange applies new value based on the provided function."""
    current, reducer_value = data
    mock_reducer = mocker.Mock(return_value=reducer_value)
    change = ReducerChange(reducer=mock_reducer)
    assert_that(
        change.apply(current),  # pyright: ignore[reportAny]
        equal_to(reducer_value),
    )
    mock_reducer.assert_called_once_with(current)


@pytest.mark.parametrize("applied_change", [NoChange, OverwriteChange, ReducerChange])
def test_apply_changeset_for_change(
    applied_change: type[ChangeSet[int | bool | str]],
    mocker: MockerFixture,
) -> None:
    """Test apply_changeset for correctly provided changeset values."""
    # The various implementations of ChangeSet are tested above, so only test if it was called correctly
    current = mocker.Mock()
    mock_change = mocker.create_autospec(applied_change)
    _ = apply_changeset(current, mock_change)
    mock_change.apply.assert_called_once_with(current)  # pyright: ignore[reportAny]


def test_apply_changeset_for_arbitrary_type_default_change(mocker: MockerFixture) -> None:
    """Test the default implementation of apply_changeset with a non changeset value.

    Current default is to overwrite the value and not raise an exception.
    """
    current = mocker.Mock()
    new = mocker.Mock()
    result = apply_changeset(current, new)
    assert_that(result, equal_to(new))


@pytest.mark.parametrize("applied_change", [NoChange, OverwriteChange, ReducerChange])
def test_apply_changeset_for_arbitrary_type_default_change_with_provided_changeset(
    applied_change: type[ChangeSet[int | bool | str]],
    mocker: MockerFixture,
) -> None:
    """Test apply_changeset with a non changeset value and a specified default behavior."""
    current = mocker.Mock()
    new = mocker.Mock()
    mock_change = mocker.create_autospec(applied_change)
    mock_change_factory = mocker.Mock(return_value=mock_change)
    _ = apply_changeset(current, new, default_change=mock_change_factory)
    mock_change_factory.assert_called_once_with(new)
    mock_change.apply.assert_called_once_with(current)  # pyright: ignore[reportAny]


@pytest.mark.parametrize("applied_change", [NoChange, OverwriteChange, ReducerChange])
def test_apply_changeset_for_arbitrary_type_raise_exception(
    applied_change: type[ChangeSet[int | bool | str]],
    mocker: MockerFixture,
) -> None:
    """Test apply_changeset with a non changeset value and flag set to raise exceptions."""
    current = mocker.Mock()
    new = mocker.Mock()
    mock_change = mocker.create_autospec(applied_change)
    mock_change_factory = mocker.Mock(return_value=mock_change)
    with pytest.raises(
        ValueError, match=r"New value was expected to be a ChangeSet type\(NoChange, OverwriteChange, ReducerChange\)"
    ):
        _ = apply_changeset(current, new, raise_exception_on_unrecognized=True, default_change=mock_change_factory)
    mock_change_factory.assert_not_called()
    mock_change.apply.assert_not_called()  # pyright: ignore[reportAny]
