from enum import Enum
from typing import TypeVar

from pydantic import BaseModel

from rag_resume._types import DataclassLike, TypedDictLike

PipelineStateTypes = DataclassLike | TypedDictLike | BaseModel

PipelineStepsType = TypeVar("PipelineStepsType", bound=Enum)
PipelineStepsType_contra = TypeVar("PipelineStepsType_contra", bound=Enum, contravariant=True)
PipelineStepsType_co = TypeVar("PipelineStepsType_co", bound=Enum, covariant=True)

PipelineStateType = TypeVar("PipelineStateType", bound=PipelineStateTypes)
PipelineStateType_contra = TypeVar("PipelineStateType_contra", bound=PipelineStateTypes, contravariant=True)
PipelineStateType_co = TypeVar("PipelineStateType_co", bound=PipelineStateTypes, covariant=True)
