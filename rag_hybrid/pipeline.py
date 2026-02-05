import os
from typing import Iterable, List, Optional

from .fusion import reciprocal_rank_fusion
from .graph_store import GraphStore
from .models import Document, SearchStrategy
from .vector_store import VectorStore


class HybridRAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        strategy: SearchStrategy = SearchStrategy.PARALLEL,
        generation_model: str = "gemini-1.5-flash",
        max_context_chars: int = 2000,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.strategy = strategy
        self.generation_model = generation_model
        self.max_context_chars = max_context_chars
        self._gemini = None
        self._groq = None

    def _configure_gemini(self) -> None:
        if self._gemini is not None:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to generate answers.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._gemini = genai

    def _configure_groq(self) -> None:
        if self._groq is not None:
            return
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required to generate answers with Groq.")
        from groq import Groq

        self._groq = Groq(api_key=api_key)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if self.strategy == SearchStrategy.VECTOR_ONLY:
            return self.vector_store.search(query, top_k)
        if self.strategy == SearchStrategy.GRAPH_ONLY:
            entities = self._extract_entities(query)
            graph_results: List[Document] = []
            for entity in entities:
                graph_results.extend(self.graph_store.search_by_entity(entity))
            return self._deduplicate_and_rank(graph_results, top_k)
        if self.strategy == SearchStrategy.SEQUENTIAL:
            return self._sequential_retrieval(query, top_k)
        return self._parallel_retrieval(query, top_k)

    def _parallel_retrieval(self, query: str, top_k: int) -> List[Document]:
        vector_results = self.vector_store.search(query, max(1, top_k // 2))
        entities = self._extract_entities(query)
        graph_results: List[Document] = []
        for entity in entities:
            graph_results.extend(self.graph_store.search_by_entity(entity, max_depth=2))
        return reciprocal_rank_fusion(vector_results, graph_results, top_k=top_k)

    def _sequential_retrieval(self, query: str, top_k: int) -> List[Document]:
        vector_results = self.vector_store.search(query, max(1, top_k // 2))
        expanded: List[Document] = list(vector_results)
        for doc in vector_results:
            entities = self._extract_entities(doc.content)
            for entity in entities[:2]:
                expanded.extend(self.graph_store.search_by_entity(entity, max_depth=1))
        return self._deduplicate_and_rank(expanded, top_k)

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        words = [w.strip(".,!?") for w in text.lower().split()]
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "are",
            "with",
            "as",
            "by",
            "from",
        }
        return [w for w in words if w and w not in stop_words and len(w) > 3][:6]

    @staticmethod
    def _deduplicate_and_rank(documents: Iterable[Document], top_k: int) -> List[Document]:
        best = {}
        for doc in documents:
            current = best.get(doc.id)
            if current is None or doc.score > current.score:
                best[doc.id] = doc
        ranked = sorted(best.values(), key=lambda d: d.score, reverse=True)
        return ranked[:top_k]

    def generate_response(self, query: str, documents: List[Document]) -> str:
        context = "\n\n".join(
            [f"[Source {i + 1}] {doc.content}" for i, doc in enumerate(documents)]
        )
        truncated_context = context[: self.max_context_chars]
        prompt = (
            "Answer using only the context. Cite sources like [Source 1].\n\n"
            f"Context:\n{truncated_context}\n\n"
            f"Question: {query}\n"
        )

        if os.getenv("GROQ_API_KEY"):
            self._configure_groq()
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            response = self._groq.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()

        self._configure_gemini()
        model = self._gemini.GenerativeModel(self.generation_model)
        response = model.generate_content(prompt)
        return response.text.strip()
