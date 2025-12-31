from typing import Any, ClassVar, Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class DataclassLike(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]  # pyright: ignore[reportExplicitAny]


@runtime_checkable
class TypedDictLike(Protocol):
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]
