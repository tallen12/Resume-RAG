import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from seriacade.json.interfaces import JsonCodecProtocol

MetadataType = TypeVar("MetadataType")


class EmbeddingModelProtocol(Protocol):
    """Protocol for an embedding model.

    This protocol defines the interface for an embedding model, which is used to
    convert text into numerical vectors. It is used in vector stores to
    represent and query documents based on their embeddings.
    """

    def embed(self, text: str | list[str]) -> Sequence[Sequence[float]]:
        """Embed the given text to vectors.

        Args:
            text (str | list[str]): A string or sequence of strings to embed.
                This can be a single string or a list of strings.

        Returns:
            Sequence[Sequence[float]]: A sequence of embedding vectors.
                Each vector is a list of floats representing the embedded text.
        """
        ...


@dataclass
class Document(Generic[MetadataType]):
    """Represents a document in a vector store.

    This class represents a document that is stored in a vector store. It contains
    the content of the document, a unique identifier, and metadata that can be
    used for filtering or additional information.

    """

    id: uuid.UUID
    content: str
    metadata: MetadataType


class VectorStoreProtocol(Protocol[MetadataType]):
    """Protocol for implementing a vector store.

    This protocol defines the interface for a vector store, which is used to
    store and query documents based on their embeddings. It requires an
    embedding model and a codec to encode metadata.
    """

    embedding_model: EmbeddingModelProtocol
    metadata_codec: JsonCodecProtocol[MetadataType]

    def add(self, text: Sequence[str]) -> Sequence[uuid.UUID]:
        """Add text to the vector store without metadata.

        Args:
            text (Sequence[str]): Text to store in the vector store.
                This can be a sequence of strings.

        Returns:
            Sequence[uuid.UUID]: UUIDs for the text inserted into the vector store.
                Each UUID corresponds to a piece of text added.
        """
        ...

    def add_with_metadata(self, text: Sequence[str], metadata: list[MetadataType]) -> Sequence[uuid.UUID]:
        """Add text with metadata.

        Args:
            text (Sequence[str]): Text to store in the vector store.
                This can be a sequence of strings.

            metadata (list[MetadataType]): Metadata to store in the vector store.
                This must be a list of serializable objects that can be encoded
                to JSON. Each metadata item corresponds to a piece of text.

        Returns:
            Sequence[uuid.UUID]: UUIDs for the text inserted into the vector store.
                Each UUID corresponds to a piece of text added.
        """
        ...

    # TODO: Specify what sort of similarity metric is being used here.
    def lookup(
        self, query: str, filter_func: Callable[[Document[MetadataType]], bool], top_k: int
    ) -> Sequence[Document[MetadataType]]:
        """Lookup entries in the vector store by query string based on cosine similarity.

        Args:
            query (str): Query string to lookup in the vector store.
                This is the text used to find similar documents.

            filter_func (Callable[[Document[MetadataType]], bool]):
                Function to filter documents returned by lookup. This function
                takes a document and returns a boolean indicating whether the
                document should be included in the results. This can be used to
                filter based on metadata or other criteria.
                The underlying implementation of how this function is applied
                depends on the implementation.

            top_k (int): Number of most relevant documents to return based on
                similarity score. This is the maximum number of documents to
                return in the results.

        Returns:
            Sequence[Document[MetadataType]]: Sequence of Documents that are most
                similar to the query string. Each document is returned with its
                metadata and embedding vector.
        """
        ...
