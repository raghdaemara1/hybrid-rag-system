import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_hybrid.data_loader import load_documents_jsonl
from rag_hybrid.graph_store import GraphStore
from rag_hybrid.models import SearchStrategy
from rag_hybrid.pipeline import HybridRAGPipeline
from rag_hybrid.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG demo (Groq/Gemini + vector + graph)")
    parser.add_argument("--question", type=str, required=False, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=4, help="Number of results to retrieve")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=int(os.getenv("MAX_CONTEXT_CHARS", "2000")),
        help="Max characters passed into the generation prompt",
    )
    args = parser.parse_args()

    load_dotenv()
    if not (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise ValueError("Set GROQ_API_KEY or GEMINI_API_KEY in your environment or .env file.")

    data_path = Path(__file__).parent / "data" / "documents.jsonl"
    documents = load_documents_jsonl(str(data_path))

    embedding_backend = os.getenv("RAG_EMBEDDINGS", "local").lower()
    vector_store = VectorStore(embedding_backend=embedding_backend)
    vector_store.add_documents(documents)

    graph_store = GraphStore()
    graph_store.add_documents(documents)

    pipeline = HybridRAGPipeline(
        vector_store=vector_store,
        graph_store=graph_store,
        strategy=SearchStrategy.PARALLEL,
        max_context_chars=args.max_context_chars,
    )

    question = args.question or input("Question: ").strip()
    if not question:
        raise ValueError("A question is required.")

    results = pipeline.retrieve(question, top_k=args.top_k)

    print("\nRetrieved sources:")
    for i, doc in enumerate(results, 1):
        preview = doc.content[:120].replace("\n", " ")
        print(f"{i}. ({doc.source}, score={doc.score:.4f}) {preview}...")

    answer = pipeline.generate_response(question, results)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
