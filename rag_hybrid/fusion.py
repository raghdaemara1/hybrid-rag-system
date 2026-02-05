from typing import Dict, List

from .models import Document


def reciprocal_rank_fusion(
    vector_results: List[Document], graph_results: List[Document], top_k: int = 5, k: int = 60
) -> List[Document]:
    scores: Dict[str, float] = {}

    for rank, doc in enumerate(vector_results, 1):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(graph_results, 1):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)

    all_docs = {doc.id: doc for doc in vector_results + graph_results}
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results: List[Document] = []
    for doc_id, score in sorted_ids[:top_k]:
        doc = all_docs[doc_id]
        doc.score = score
        doc.source = "hybrid"
        results.append(doc)
    return results
