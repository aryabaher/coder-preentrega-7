"""RAG híbrido (BM25 + vectorial) sobre las políticas internas de TechCorp."""

import hashlib
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from errors import ClaveAPIError, IndiceIncompatibleError

RUTA_DATOS = Path(__file__).with_name("data")
INDEX_NAME = "techcorp-rag-hibrido"
NAMESPACE = "politicas-internas"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
K_DEFAULT = 5
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

_VACIAS = {"a", "al", "con", "de", "del", "el", "en", "la", "las", "lo", "los", "por", "que", "un", "una", "y"}


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalnum() else " " for c in sin_tildes.lower()).split())


class EmbeddingsLocales(Embeddings):
    """Bolsa de palabras del mismo ancho que text-embedding-3-small (1536). Sin red, para los tests."""

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(texto) for texto in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, texto: str) -> list[float]:
        vector = [0.0] * self.dimension
        for palabra in _normalizar(texto).split():
            if palabra in _VACIAS or len(palabra) < 2:
                continue
            digest = hashlib.md5(palabra[:6].encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        norma = sum(x * x for x in vector) ** 0.5
        return [x / norma for x in vector] if norma else vector


def nombre_indice() -> str:
    """INDEX_NAME del entorno. Si no está, el mismo default del índice de políticas."""
    return (os.getenv("INDEX_NAME") or INDEX_NAME).strip() or INDEX_NAME


class RAGSystem:
    """BM25 + vectorial. `k` llega a los dos retrievers en cada consulta."""

    def __init__(self, retriever: EnsembleRetriever, k: int = K_DEFAULT, origen: str = "local") -> None:
        self.retriever = retriever
        self.k = k
        self.origen = origen

    def obtener_top_k(self, query: str) -> List[Dict[str, str]]:
        self._propagar_k()
        docs = self.retriever.invoke(query)[: self.k]
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "?"),
                "categoria": d.metadata.get("categoria", "?"),
            }
            for d in docs
        ]

    def _propagar_k(self) -> None:
        for pieza in getattr(self.retriever, "retrievers", []):
            if hasattr(pieza, "k"):
                pieza.k = self.k
            kwargs = getattr(pieza, "search_kwargs", None)
            if isinstance(kwargs, dict):
                kwargs["k"] = self.k


def cargar_chunks(
    ruta: Path = RUTA_DATOS, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Parte los .txt de data/. chunk_size y chunk_overlap llegan al splitter."""
    loader = DirectoryLoader(str(ruta), glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    chunks = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(loader.load())
    for i, chunk in enumerate(chunks):
        nombre = os.path.basename(str(chunk.metadata.get("source", "")))
        chunk.metadata["source"] = nombre
        chunk.metadata["fuente"] = nombre
        chunk.metadata["categoria"] = nombre.rsplit(".", 1)[0].replace("_", " ")
        chunk.metadata["chunk_id"] = i
    return chunks


def construir_rag_local(
    k: int = K_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embeddings: Embeddings | None = None,
) -> RAGSystem:
    """BM25 + memoria, sin red. `k` llega a los dos retrievers y a RAGSystem."""
    chunks = cargar_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever_bm25 = BM25Retriever.from_documents(chunks)
    retriever_bm25.k = k
    vectorstore = InMemoryVectorStore(embeddings or EmbeddingsLocales(dimension=EMBEDDING_DIM))
    vectorstore.add_documents(chunks)
    retriever_vectorial = vectorstore.as_retriever(search_kwargs={"k": k})
    retriever_hibrido = EnsembleRetriever(retrievers=[retriever_bm25, retriever_vectorial], weights=[0.5, 0.5])
    return RAGSystem(retriever_hibrido, k=k, origen="local (no hay PINECONE_API_KEY): BM25 + memoria")


def verificar_indice(desc: object, dimension: int = EMBEDDING_DIM) -> None:
    """Si el índice ya existe, exige la misma dimensión y métrica cosine. No lo recrea."""
    existente = getattr(desc, "dimension", None)
    if existente is None and isinstance(desc, dict):
        existente = desc.get("dimension")
    if existente is not None and int(existente) != dimension:
        raise IndiceIncompatibleError(
            f"Mismatch de dimensiones: el índice {nombre_indice()} es {existente}D "
            f"y {EMBEDDING_MODEL} usa {dimension}D. No se reingesta."
        )
    metric = getattr(desc, "metric", None)
    if metric is None and isinstance(desc, dict):
        metric = desc.get("metric")
    if metric and str(metric).lower() != "cosine":
        raise IndiceIncompatibleError(
            f"Mismatch de métrica: el índice {nombre_indice()} es {metric} y {EMBEDDING_MODEL} usa cosine."
        )


def vectores_en_namespace(stats: Any, namespace: str = NAMESPACE) -> int:
    namespaces = stats["namespaces"] if isinstance(stats, dict) else stats.namespaces
    espacio = namespaces.get(namespace, {}) if hasattr(namespaces, "get") else {}
    if isinstance(espacio, dict):
        return int(espacio.get("vector_count", 0))
    return int(getattr(espacio, "vector_count", 0))


def conectar_pinecone(k: int = K_DEFAULT) -> RAGSystem:
    """Conecta al índice de políticas. Si el namespace ya tiene vectores, no reingesta."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    if not (os.getenv("PINECONE_API_KEY") or "").strip():
        raise ClaveAPIError(
            "401/key: falta PINECONE_API_KEY en las variables de entorno. "
            "Cargala en .env. No se reintenta."
        )
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise ClaveAPIError(
            "401/key: falta OPENAI_API_KEY en las variables de entorno. "
            f"El índice {nombre_indice()} usa {EMBEDDING_MODEL}. No se reintenta."
        )
    indice = nombre_indice()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    nombres = [i["name"] if isinstance(i, dict) else i.name for i in pc.list_indexes()]
    if indice not in nombres:
        pc.create_index(
            name=indice,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        verificar_indice(pc.describe_index(indice), EMBEDDING_DIM)
    vectorstore = PineconeVectorStore(index_name=indice, embedding=embeddings, namespace=NAMESPACE)
    cantidad = vectores_en_namespace(pc.Index(indice).describe_index_stats(), NAMESPACE)
    if cantidad == 0:
        vectorstore.add_documents(cargar_chunks())
        detalle = "namespace vacío, se ingesta"
    else:
        detalle = f"vectores={cantidad}, no se reingesta"
    chunks = cargar_chunks()
    retriever_bm25 = BM25Retriever.from_documents(chunks)
    retriever_bm25.k = k
    retriever_vectorial = vectorstore.as_retriever(search_kwargs={"k": k, "namespace": NAMESPACE})
    retriever_hibrido = EnsembleRetriever(retrievers=[retriever_bm25, retriever_vectorial], weights=[0.5, 0.5])
    origen = f"Pinecone índice={indice} namespace={NAMESPACE} {detalle}"
    return RAGSystem(retriever_hibrido, k=k, origen=origen)


def construir_rag(k: int = K_DEFAULT) -> RAGSystem:
    """Pinecone si hay PINECONE_API_KEY; si no, el índice local híbrido."""
    if os.getenv("PINECONE_API_KEY"):
        return conectar_pinecone(k)
    return construir_rag_local(k)


rag_system = construir_rag()
