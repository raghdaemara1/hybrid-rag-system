import json
from typing import List

from .models import Document


def load_documents_jsonl(path: str) -> List[Document]:
    documents: List[Document] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            documents.append(
                Document(
                    id=item["id"],
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                )
            )
    return documents
