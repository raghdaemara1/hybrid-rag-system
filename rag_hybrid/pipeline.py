import os
from typing import Iterable, List
from .fusion import reciprocal_rank_fusion
from .graph_store import GraphStore
from .models import Document, SearchStrategy
from .vector_store import VectorStore
# Try to load spaCy for real named entity recognition.
# Falls back to keyword extraction if spaCy / the model is not installed.
try:
    import spacy as _spacy
    _nlp = _spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _nlp = None
    _SPACY_AVAILABLE = False
class HybridRAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        strategy: SearchStrategy = SearchStrategy.PARALLEL,
        generation_model: str = "llama-3.1-8b-instant",
        max_context_chars: int = 2000,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.strategy = strategy
        self.generation_model = generation_model
        self.max_context_chars = max_context_chars
        self._gemini = None
        self._groq = None
    # ── LLM configuration ────────────────────────────────────────────────────
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
    # ── Retrieval ─────────────────────────────────────────────────────────────
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
    # ── Entity extraction ─────────────────────────────────────────────────────
    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """
        Extract named entities from text using spaCy (en_core_web_sm).
        Falls back to keyword extraction (stopword filtering) when spaCy is
        not installed. Install spaCy support with:
            pip install spacy && python -m spacy download en_core_web_sm
        """
        if _SPACY_AVAILABLE and _nlp is not None:
            doc = _nlp(text)
            # Collect unique lowercased entity strings; prefer named entities
            entities = list({ent.text.lower() for ent in doc.ents if len(ent.text) > 2})
            # If spaCy found nothing (very short/technical text), fall through
            # to keyword extraction below
            if entities:
                return entities[:6]
        # Keyword fallback — used when spaCy is unavailable or returns nothing
        _STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "is", "are", "with", "as", "by", "from", "it",
            "its", "this", "that", "was", "be", "been", "have", "has",
        }
        words = [w.strip(".,!?;:\\"'") for w in text.lower().split()]
        return [w for w in words if w and w not in _STOP_WORDS and len(w) > 3][:6]
    @staticmethod
    def _deduplicate_and_rank(documents: Iterable[Document], top_k: int) -> List[Document]:
        best = {}
        for doc in documents:
            current = best.get(doc.id)
            if current is None or doc.score > current.score:
                best[doc.id] = doc
        ranked = sorted(best.values(), key=lambda d: d.score, reverse=True)
        return ranked[:top_k]
    # ── Generation ────────────────────────────────────────────────────────────
    def generate_response(self, query: str, documents: List[Document]) -> str:
        context = "\\n\\n".join(
            [f"[Source {i + 1}] {doc.content}" for i, doc in enumerate(documents)]
        )
        truncated_context = context[: self.max_context_chars]
        prompt = (
            "Answer using only the context. Cite sources like [Source 1].\\n\\n"
            f"Context:\\n{truncated_context}\\n\\n"
            f"Question: {query}\\n"
        )
        # Route to the correct LLM based on generation_model.
        # Groq is chosen when the model name is NOT a Gemini model name.
        use_groq = (
            not self.generation_model.startswith("gemini")
            and os.getenv("GROQ_API_KEY")
        )
        if use_groq:
            self._configure_groq()
            response = self._groq.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        # Gemini path
        self._configure_gemini()
        model = self._gemini.GenerativeModel(self.generation_model)
        response = model.generate_content(prompt)
        return response.text.strip()



