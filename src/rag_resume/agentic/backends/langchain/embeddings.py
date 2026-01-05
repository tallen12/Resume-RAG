import uuid
from abc import ABC
from collections.abc import Callable, Sequence
from functools import partial
from typing import assert_never, final, override

from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from seriacade.json.interfaces import JsonCodecProtocol

from rag_resume.agentic.llms.embedding import Document, EmbeddingModelProtocol, MetadataType, VectorStoreProtocol
from rag_resume.json import enforce_dict_type


class LangchainEmbeddingModel(EmbeddingModelProtocol):
    """Wrapper for embedding models using LangChain.

    This class provides a wrapper around LangChain's embedding models, allowing
    for seamless integration with other components of the project.

    Attributes:
        embedding_model (Embeddings): The LangChain embedding model to be wrapped.
    """

    def __init__(self, embedding_model: Embeddings) -> None:
        """Initialize the LangchainEmbeddingModel.

        Args:
            embedding_model (Embeddings): The LangChain embedding model to be wrapped.
        """
        self.embedding_model: Embeddings = embedding_model

    @override
    def embed(self, text: str | list[str]) -> Sequence[Sequence[float]]:
        """Embed the given text using the wrapped LangChain embedding model.

        Args:
            text (str | list[str]): The text to be embedded. Can be a single string
                or a list of strings.

        Returns:
            Sequence[Sequence[float]]: A sequence of sequences of floats representing
                the embeddings of the input text.
        """
        match text:
            case str():
                embedding_value = [text]
            case list():
                embedding_value = text
            case _:
                assert_never(text)
        return self.embedding_model.embed_documents(embedding_value)


class LangchainVectorStore(ABC, VectorStoreProtocol[MetadataType]):
    """Abstract Base Class for Langchain based vector stores.

    This class serves as an abstract base class for vector stores that are
    built on top of LangChain's vector store functionality.

    Attributes:
        vector_store (VectorStore): The LangChain vector store instance.
    """

    vector_store: VectorStore

    def _filter_adapter(
        self,
        document: LangchainDocument,
        filter_func: Callable[[Document[MetadataType]], bool],
    ) -> bool:
        """Adapt the filter function for use with LangChain documents.

        Args:
            document (LangchainDocument): The LangChain document to be filtered.
            filter_func (Callable[[Document[MetadataType]], bool]): The filter function
                to be applied to the document.

        Returns:
            bool: The result of applying the filter function to the document.
        """
        return filter_func(
            Document(
                id=uuid.UUID(document.id),
                content=document.page_content,
                metadata=self.metadata_codec.convert_from_json(document.metadata),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
        )

    @override
    def add(self, text: Sequence[str]) -> Sequence[uuid.UUID]:
        """Add the given text to the vector store.

        Args:
            text (Sequence[str]): The text to be added to the vector store.

        Returns:
            Sequence[uuid.UUID]: A sequence of UUIDs representing the IDs of the
                added documents.
        """
        return [uuid.UUID(uuid_str) for uuid_str in self.vector_store.add_texts(text)]  # pyright: ignore[reportUnknownMemberType]

    @override
    def add_with_metadata(self, text: Sequence[str], metadata: list[MetadataType]) -> Sequence[uuid.UUID]:
        """Add the given text with metadata to the vector store.

        Args:
            text (Sequence[str]): The text to be added to the vector store.
            metadata (list[MetadataType]): The metadata associated with the text.

        Returns:
            Sequence[uuid.UUID]: A sequence of UUIDs representing the IDs of the
                added documents.
        """
        return [
            uuid.UUID(uuid_str)
            for uuid_str in self.vector_store.add_texts(  # pyright: ignore[reportUnknownMemberType]
                text,
                metadatas=[enforce_dict_type(self.metadata_codec.convert_to_json(metadatum)) for metadatum in metadata],
            )
        ]

    @override
    def lookup(
        self, query: str, filter_func: Callable[[Document[MetadataType]], bool], top_k: int
    ) -> Sequence[Document[MetadataType]]:
        """Look up documents in the vector store based on a query.

        Args:
            query (str): The query to search for in the vector store.
            filter_func (Callable[[Document[MetadataType]], bool]): The filter function
                to be applied to the documents.
            top_k (int): The maximum number of documents to return.

        Returns:
            Sequence[Document[MetadataType]]: A sequence of documents that match
                the query and pass the filter function.
        """
        filter_pipeline = partial(self._filter_adapter, filter_func=filter_func)
        return [
            Document(
                id=uuid.UUID(doc.id),
                content=doc.page_content,
                metadata=self.metadata_codec.convert_from_json(doc.metadata),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
            for doc in self.vector_store.similarity_search(query, k=top_k, filter=filter_pipeline)
        ]


@final
class LangchainInMemoryVectorStore(LangchainVectorStore[MetadataType]):
    """InMemory vector store from langchain.

    This class provides an in-memory vector store implementation using
    LangChain's vector store functionality.

    Attributes:
        embedding_model (LangchainEmbeddingModel): The embedding model used
            to generate embeddings for the documents.
        metadata_codec (JsonCodecProtocol[MetadataType]): The codec used to
            encode and decode metadata.
        vector_store (InMemoryVectorStore): The LangChain in-memory vector store
            instance.
    """

    def __init__(
        self, embedding_model: LangchainEmbeddingModel, metadata_codec: JsonCodecProtocol[MetadataType]
    ) -> None:
        """Initialize the LangchainInMemoryVectorStore.

        Args:
            embedding_model (LangchainEmbeddingModel): The embedding model used
                to generate embeddings for the documents.
            metadata_codec (JsonCodecProtocol[MetadataType]): The codec used to
                encode and decode metadata.
        """
        self.embedding_model = embedding_model
        self.metadata_codec = metadata_codec
        self.vector_store = InMemoryVectorStore(self.embedding_model.embedding_model)
