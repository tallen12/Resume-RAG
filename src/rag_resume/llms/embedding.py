import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from rag_resume.json import JsonCodecProtocol

MetadataType = TypeVar("MetadataType")


class EmbeddingModelProtocol(Protocol):
    """Protocol for an embedding model."""

    def embed(self, text: str | list[str]) -> Sequence[Sequence[float]]:
        """Embed the given text to vectors.

        Args:
            text (str | list[str]): a string or sequence of strings to embed.

        Returns:
            Sequence[Sequence[float]]: a sequence of embedding vectors.
        """
        ...


@dataclass
class Document(Generic[MetadataType]):
    """Represents a document in a vector store.

    Metadata is provided a generic to allow for custom datatypes.

    """

    id: uuid.UUID
    content: str
    metadata: MetadataType


class VectorStoreProtocol(Protocol[MetadataType]):
    """Protocol for implementing a vector store.

    Requires an embedding model and a codec to encode metadata.

    """

    embedding_model: EmbeddingModelProtocol
    metadata_codec: JsonCodecProtocol[MetadataType]

    def add(self, text: Sequence[str]) -> Sequence[uuid.UUID]:
        """Add text to the vector store without metadata.

        Args:
            text (Sequence[str]): Text to store in the vector store.

        Returns:
            Sequence[uuid.UUID]: UUIDs for the text inserted into the vector store
        """
        ...

    def add_with_metadata(self, text: Sequence[str], metadata: list[MetadataType]) -> Sequence[uuid.UUID]:
        """Add text with metadata.

        Args:
            text (Sequence[str]): Text to store in the vector store
            metadata (list[MetadataType]): Metadata to store in the vector store, must be serializable to json.

        Returns:
            Sequence[uuid.UUID]: UUIDs for the text inserted into the vector store
        """
        ...

    def lookup(
        self, query: str, filter_func: Callable[[Document[MetadataType]], bool], top_k: int
    ) -> Sequence[Document[MetadataType]]:
        """Lookup entries in the vector store by query string based on cosine similarity.

        Args:
            query (str): query string to lookup in the vector store.
            filter_func (Callable[[Document[MetadataType]], bool]): function to filter
                documents returned by lookup. Can be implemented differently (ie. sql queries, mapping over results) \
                depending on implementation.
            top_k (int): Return the top k most relevant documents based on similarity score.

        Returns:
            Sequence[Document[MetadataType]]: Sequence of Documents that are most similar to query string.
        """
        ...
