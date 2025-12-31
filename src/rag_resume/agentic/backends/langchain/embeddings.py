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
    """Wrapper for embedding models using LangChain."""

    def __init__(self, embedding_model: Embeddings) -> None:
        self.embedding_model: Embeddings = embedding_model

    @override
    def embed(self, text: str | list[str]) -> Sequence[Sequence[float]]:
        match text:
            case str():
                embedding_value = [text]
            case list():
                embedding_value = text
            case _:
                assert_never(text)
        return self.embedding_model.embed_documents(embedding_value)


class LangchainVectorStore(ABC, VectorStoreProtocol[MetadataType]):
    """ABC for Langchain based vector stores."""

    vector_store: VectorStore

    def _filter_adapter(
        self,
        document: LangchainDocument,
        filter_func: Callable[[Document[MetadataType]], bool],
    ) -> bool:
        """Helper to convert filter function to interface with langchain."""
        return filter_func(
            Document(
                id=uuid.UUID(document.id),
                content=document.page_content,
                metadata=self.metadata_codec.convert_from_json(document.metadata),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
        )

    @override
    def add(self, text: Sequence[str]) -> Sequence[uuid.UUID]:
        return [uuid.UUID(uuid_str) for uuid_str in self.vector_store.add_texts(text)]  # pyright: ignore[reportUnknownMemberType]

    @override
    def add_with_metadata(self, text: Sequence[str], metadata: list[MetadataType]) -> Sequence[uuid.UUID]:
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
    """InMemory vector store from langchain."""

    def __init__(
        self, embedding_model: LangchainEmbeddingModel, metadata_codec: JsonCodecProtocol[MetadataType]
    ) -> None:
        self.embedding_model = embedding_model
        self.metadata_codec = metadata_codec
        self.vector_store = InMemoryVectorStore(self.embedding_model.embedding_model)
