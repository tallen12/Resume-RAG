from enum import Enum
from typing import TypeVar

from pydantic import BaseModel

from rag_resume._types import DataclassLike, TypedDictLike

GraphStateTypes = DataclassLike | TypedDictLike | BaseModel

GraphStepsType = TypeVar("GraphStepsType", bound=Enum)
GraphStepsType_contra = TypeVar("GraphStepsType_contra", bound=Enum, contravariant=True)
GraphStepsType_co = TypeVar("GraphStepsType_co", bound=Enum, covariant=True)

GraphStateType = TypeVar("GraphStateType", bound=GraphStateTypes)
GraphStateType_contra = TypeVar("GraphStateType_contra", bound=GraphStateTypes, contravariant=True)
GraphStateType_co = TypeVar("GraphStateType_co", bound=GraphStateTypes, covariant=True)
