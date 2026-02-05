from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SearchStrategy(Enum):
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
