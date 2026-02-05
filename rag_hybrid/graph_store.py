from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List

import networkx as nx

from .models import Document


class GraphStore:
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.entity_to_docs: Dict[str, set] = defaultdict(set)
        self.doc_by_id: Dict[str, Document] = {}

    def add_documents(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self.doc_by_id[doc.id] = doc
            entities = [e.lower() for e in doc.metadata.get("entities", [])]
            for entity in entities:
                self.graph.add_node(entity)
                self.entity_to_docs[entity].add(doc.id)
            for a, b in combinations(sorted(set(entities)), 2):
                if self.graph.has_edge(a, b):
                    self.graph[a][b]["weight"] += 1
                else:
                    self.graph.add_edge(a, b, weight=1)

    def search_by_entity(self, entity: str, max_depth: int = 2) -> List[Document]:
        entity = entity.lower()
        if entity not in self.graph:
            return []

        related = nx.single_source_shortest_path_length(
            self.graph, entity, cutoff=max_depth
        )
        scores: Dict[str, float] = defaultdict(float)
        for related_entity in related.keys():
            for doc_id in self.entity_to_docs.get(related_entity, []):
                scores[doc_id] += 1.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results: List[Document] = []
        for doc_id, score in ranked:
            doc = self.doc_by_id[doc_id]
            doc.score = float(score)
            doc.source = "graph"
            results.append(doc)
        return results
