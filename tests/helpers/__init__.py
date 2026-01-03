from typing import TypeVar

from rag_resume._types import DataclassLike

T = TypeVar("T", bound=DataclassLike)


class ChangeSet[T]:
    def __init__(self, given_type: type[T]):
        self.given_type = given_type

    def change_set(self):
        pass
