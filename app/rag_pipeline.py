import os
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb
import numpy as np


class EmbeddingManager:
    """OpenAI embeddings helper."""

    def __init__(
        self, model: str = "text-embedding-3-large", dimensions: int | None = None
    ):
        self.model_name = model
        self.dimensions = dimensions
        self.model = (
            OpenAIEmbeddings(model=self.model_name, dimensions=self.dimensions)
            if dimensions
            else OpenAIEmbeddings(model=self.model_name)
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.embed_documents(texts)
        return np.array(vectors)


class VectorStore:
    """Simple ChromaDB wrapper for persistence and similarity search."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )

    def add(self, documents: List[Any], embeddings: np.ndarray) -> None:
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        contents: List[str] = []
        emb_list: List[List[float]] = []
        for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{idx}")
            meta = dict(doc.metadata)
            meta["content_length"] = len(doc.page_content)
            metadatas.append(meta)
            contents.append(doc.page_content)
            emb_list.append(emb.tolist())
        self.collection.add(
            ids=ids, embeddings=emb_list, metadatas=metadatas, documents=contents
        )

    def similarity_search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=k
        )


def load_pdfs(data_dir: str) -> List[Any]:
    docs: List[Any] = []
    for pdf_path in Path(data_dir).rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_file"] = pdf_path.name
            d.metadata["file_type"] = "pdf"
            docs.append(d)
    return docs


def split_documents(
    documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed_texts([query])[0]
        results = self.vector_store.similarity_search(np.array(q_emb), k=top_k)
        retrieved: List[Dict[str, Any]] = []
        if results.get("documents") and results["documents"][0]:
            for i, (doc, meta, dist, _id) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    results["ids"][0],
                )
            ):
                retrieved.append(
                    {
                        "id": _id,
                        "content": doc,
                        "metadata": meta,
                        "similarity": 1 - dist,
                        "rank": i + 1,
                    }
                )
        return retrieved


class RAGPipeline:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large",
    ):
        self.embedder = EmbeddingManager(model=embedding_model)
        self.vector = VectorStore()
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.retriever = RAGRetriever(self.vector, self.embedder)

    def index_data(self, data_dir: str) -> Dict[str, Any]:
        docs = load_pdfs(data_dir)
        splits = split_documents(docs)
        texts = [d.page_content for d in splits]
        emb = self.embedder.embed_texts(texts)
        self.vector.add(splits, emb)
        return {
            "added": len(splits),
            "collection_count": self.vector.collection.count(),
        }

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join([r["content"] for r in results]) if results else ""
        if not context:
            return {"answer": "No relevant context found.", "sources": []}
        prompt = f"Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        resp = self.llm.invoke([prompt])
        return {"answer": resp.content, "sources": results}
