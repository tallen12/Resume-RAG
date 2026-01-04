from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class NoChange:
    """NOP change."""

    def apply[T](self, current: T) -> T:
        return current


@dataclass
class OverwriteChange[T]:
    """Overwrite current value for this change."""

    new: T

    def apply(self, _: T) -> T:
        return self.new


@dataclass
class ReducerChange[T]:
    """Apply a custom reducer for this change."""

    reducer: Callable[[T], T]

    def apply(self, current: T) -> T:
        return self.reducer(current)


type ChangeSet[T] = NoChange | OverwriteChange[T] | ReducerChange[T]


def apply_changeset[T](
    current: T,
    new: ChangeSet[T] | T,
    *,
    default_change: Callable[[T], ChangeSet[T]] = lambda current: OverwriteChange(current),
    raise_exception_on_unrecognized: bool = False,
) -> T:
    """Helper to apply a ChangeSet to the current value.

    Useful for applying it as a langchain reducer for example. In addition it handles accepting non changeset inputs by
    either applying a default change, or raising an exception.

    Args:
        current (T): The chat language model to use for generating responses.
        new (ChangeSet[T] | T): The new value either a ChangeSet or a value of T.
        default_change (type[NoChange | OverwriteChange[T]]): The default change to apply on non-ChangeSet inputs,
            will be ignored if set to raise an exception using raise_exception_on_unrecognized.
            Defaults to OverwriteChange.
        raise_exception_on_unrecognized(bool): raise an exception on non-ChangeSet inputs.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    match new:
        case NoChange() | OverwriteChange() | ReducerChange():
            # Technically typing this as ChangeSet[T] | T breaks the typechecking a bit
            # This was done to allow handling of non ChangeSet new which is useful for using this as
            # a langchain reducer.
            return new.apply(current)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        case _:
            pass
    if raise_exception_on_unrecognized:
        msg = "New value was expected to be a ChangeSet type(NoChange, OverwriteChange, ReducerChange)"
        raise ValueError(msg)
    default_change_impl = default_change(new)
    return default_change_impl.apply(current)
