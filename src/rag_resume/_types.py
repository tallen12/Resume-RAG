from enum import Enum
from typing import Any, ClassVar, Protocol, TypeVar

from pydantic import BaseModel
from typing_extensions import runtime_checkable


@runtime_checkable
class DataclassLike(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]  # pyright: ignore[reportExplicitAny]


@runtime_checkable
class TypedDictLike(Protocol):
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]


PipelineStateTypes = DataclassLike | TypedDictLike | BaseModel

PipelineStepsType = TypeVar("PipelineStepsType", bound=Enum)
PipelineStepsType_contra = TypeVar("PipelineStepsType_contra", bound=Enum, contravariant=True)
PipelineStepsType_co = TypeVar("PipelineStepsType_co", bound=Enum, covariant=True)

PipelineStateType = TypeVar("PipelineStateType", bound=PipelineStateTypes)
PipelineStateType_contra = TypeVar("PipelineStateType_contra", bound=PipelineStateTypes, contravariant=True)
PipelineStateType_co = TypeVar("PipelineStateType_co", bound=PipelineStateTypes, covariant=True)
