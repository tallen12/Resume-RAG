from enum import Enum
from typing import Any, ClassVar, Protocol, TypeVar


class DataclassLike(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


class TypedDictLike(Protocol):
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]


SerializableTypes = DataclassLike | TypedDictLike

PipelineStateTypes = TypedDictLike | DataclassLike

PipelineStepsType = TypeVar("PipelineStepsType", bound=Enum)
PipelineStepsType_contra = TypeVar("PipelineStepsType_contra", bound=Enum, contravariant=True)
PipelineStepsType_co = TypeVar("PipelineStepsType_co", bound=Enum, covariant=True)

PipelineStateType = TypeVar("PipelineStateType", bound=PipelineStateTypes)
PipelineStateType_contra = TypeVar("PipelineStateType_contra", bound=PipelineStateTypes, contravariant=True)
PipelineStateType_co = TypeVar("PipelineStateType_co", bound=PipelineStateTypes, covariant=True)
