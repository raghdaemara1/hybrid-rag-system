import os
import re
from typing import List, Optional

import numpy as np

from .models import Document

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


class VectorStore:
    def __init__(
        self,
        embedding_backend: str = "local",
        embedding_model: str = "models/embedding-001",
        embedding_dim: int = 256,
        use_faiss: bool = True,
    ):
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        self.index = None
        self.documents: List[Document] = []
        self.embeddings = None
        self._gemini = None

    def _configure_gemini(self) -> None:
        if self._gemini is not None:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to use Gemini embeddings.")
        import google.generativeai as genai  # local import to avoid warnings if unused

        genai.configure(api_key=api_key)
        self._gemini = genai

    def _embed_text_gemini(self, text: str, task_type: str) -> np.ndarray:
        self._configure_gemini()
        response = self._gemini.embed_content(
            model=self.embedding_model,
            content=text,
            task_type=task_type,
        )
        return np.array(response["embedding"], dtype="float32")

    def _embed_text_local(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype="float32")
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for token in tokens:
            idx = hash(token) % self.embedding_dim
            vec[idx] += 1.0
        return vec

    def _embed_text(self, text: str, task_type: str) -> np.ndarray:
        if self.embedding_backend == "gemini":
            return self._embed_text_gemini(text, task_type)
        return self._embed_text_local(text)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        embeddings = [
            self._normalize(self._embed_text(doc.content, "retrieval_document"))
            for doc in documents
        ]
        matrix = np.vstack(embeddings).astype("float32")

        if self.use_faiss:
            dim = matrix.shape[1]
            if self.index is None:
                self.index = faiss.IndexFlatIP(dim)
            self.index.add(matrix)
        else:
            if self.embeddings is None:
                self.embeddings = matrix
            else:
                self.embeddings = np.vstack([self.embeddings, matrix])

        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.documents:
            return []

        query_vec = self._normalize(self._embed_text(query, "retrieval_query"))
        k = min(top_k, len(self.documents))

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(np.array([query_vec]), k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                doc = self.documents[idx]
                doc.score = float(score)
                doc.source = "vector"
                results.append(doc)
            return results

        if self.embeddings is None:
            return []

        scores = self.embeddings @ query_vec
        best_idx = np.argsort(-scores)[:k]
        results = []
        for idx in best_idx:
            doc = self.documents[idx]
            doc.score = float(scores[idx])
            doc.source = "vector"
            results.append(doc)
        return results
