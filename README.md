# Hybrid RAG System — Retrieval-Augmented Generation

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![FAISS](https://img.shields.io/badge/vector_index-FAISS-orange.svg)
![NetworkX](https://img.shields.io/badge/graph-NetworkX-green.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini-purple.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Feed it a JSONL document corpus; ask it any question; it runs semantic vector search and entity-graph traversal in parallel, fuses both ranked lists with Reciprocal Rank Fusion, and returns a cited, grounded answer from Groq or Gemini — that dual-path retrieval plus fusion is what makes it "hybrid".**

---

## What Does This App Do?

| Feature | What You Can Do |
|---|---|
| Hybrid retrieval | Run vector search and entity-graph search simultaneously and merge results |
| Flexible embedding backends | Choose `local` (sentence-transformers, no API key) or `gemini` (Google API) |
| FAISS-accelerated vector index | Index large corpora with FAISS `IndexFlatIP`; falls back to NumPy dot-product if FAISS is absent |
| Entity co-occurrence graph | Automatically build a weighted NetworkX graph from `entities` metadata; traverse up to configurable depth |
| Reciprocal Rank Fusion | Combine vector and graph ranked lists with the RRF formula `1/(k+rank)` where k=60 |
| Four search strategies | Switch between `VECTOR_ONLY`, `GRAPH_ONLY`, `PARALLEL` (default), or `SEQUENTIAL` at runtime |
| Multi-provider generation | Generate answers with Groq (Llama-3.1-8b-instant by default) or any Gemini model |
| Source citations | Every generated answer is instructed to cite `[Source N]` inline against the retrieved context |
| Configurable context window | `MAX_CONTEXT_CHARS` caps the prompt context; prevents token-limit overruns |
| Zero-boilerplate data format | Load any corpus that follows the `{id, content, metadata: {entities}}` JSONL schema |

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/raghdaemara1/hybrid-rag-system.git
cd hybrid-rag-system

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model (required for named-entity extraction)
python -m spacy download en_core_web_sm

# 5. Configure credentials
cp .env.example .env
# Open .env and set GROQ_API_KEY and/or GEMINI_API_KEY

# 6. Run the demo
python run_demo.py --question "What is hybrid RAG?"

# Optional flags
python run_demo.py --question "How does fusion work?" --top-k 6 --max-context-chars 3000
```

Expected terminal output:

```
Retrieved sources:
1. (hybrid, score=0.0328) Hybrid RAG blends vector search with graph search for better coverage....
2. (hybrid, score=0.0323) Reciprocal Rank Fusion merges ranked lists by rank....
...

Answer:

Hybrid RAG combines vector search and graph traversal... [Source 1] [Source 4]
```

---

## System Architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         INDEXING PHASE                              │
  │                                                                     │
  │  documents.jsonl                                                    │
  │       │                                                             │
  │       ▼                                                             │
  │  load_documents_jsonl()  →  List[Document]                         │
  │       │                                                             │
  │       ├──────────────────────────────────────────────────────────┐  │
  │       │                                                          │  │
  │       ▼                                                          ▼  │
  │  VectorStore.add_documents()                  GraphStore.add_documents() │
  │  ┌───────────────────────┐                    ┌──────────────────────┐ │
  │  │ _embed_text()         │                    │ entity co-occurrence │ │
  │  │  local: all-MiniLM-L6 │                    │ graph (NetworkX)     │ │
  │  │  gemini: embed-001    │                    │ entity_to_docs dict  │ │
  │  │ FAISS IndexFlatIP or  │                    │ doc_by_id dict       │ │
  │  │ NumPy matrix          │                    └──────────────────────┘ │
  │  └───────────────────────┘                                          │  │
  └─────────────────────────────────────────────────────────────────────┘
                                                                         
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         QUERY PHASE (PARALLEL strategy)             │
  │                                                                     │
  │  User query string                                                  │
  │       │                                                             │
  │       ├──────────────────────────┬──────────────────────────────┐  │
  │       │                          │                              │  │
  │       ▼                          ▼                              │  │
  │  VectorStore.search()      _extract_entities()                  │  │
  │  ┌──────────────────┐      (spaCy NER / keyword fallback)       │  │
  │  │ embed query      │            │                              │  │
  │  │ FAISS search or  │            ▼                              │  │
  │  │ NumPy dot-product│      GraphStore.search_by_entity()        │  │
  │  │ → List[Document] │      ┌────────────────────────┐           │  │
  │  │   source=vector  │      │ nx shortest-path BFS   │           │  │
  │  └──────────────────┘      │ depth ≤ max_depth (2)  │           │  │
  │       │                    │ → List[Document]        │           │  │
  │       │                    │   source=graph          │           │  │
  │       │                    └────────────────────────┘           │  │
  │       │                          │                              │  │
  │       └──────────────┬───────────┘                              │  │
  │                      ▼                                          │  │
  │            reciprocal_rank_fusion()                             │  │
  │            ┌───────────────────────────────┐                   │  │
  │            │ score = Σ 1/(60 + rank)        │                   │  │
  │            │ merge by doc.id                │                   │  │
  │            │ → top-k List[Document]         │                   │  │
  │            │   source=hybrid                │                   │  │
  │            └───────────────────────────────┘                   │  │
  │                      │                                          │  │
  │                      ▼                                          │  │
  │          HybridRAGPipeline.generate_response()                  │  │
  │          ┌─────────────────────────────────────┐                │  │
  │          │ build prompt with [Source N] tags   │                │  │
  │          │ truncate to MAX_CONTEXT_CHARS        │                │  │
  │          │ Groq chat completions  OR            │                │  │
  │          │ Gemini GenerativeModel               │                │  │
  │          └─────────────────────────────────────┘                │  │
  │                      │                                          │  │
  │                      ▼                                          │  │
  │               Cited answer string                               │  │
  └─────────────────────────────────────────────────────────────────┘
```

---

## The Pipelines

### Pipeline 1 — Document Indexing

Triggered once at startup by `run_demo.py`. Reads the JSONL corpus, builds the vector index, and populates the entity graph so both stores are ready before any query is served.

```
data/documents.jsonl (file on disk)
    │
    ▼
1. load_documents_jsonl(path) in data_loader.py
   — Reads each non-empty line, json.loads() it
   — Constructs Document(id, content, metadata, score=0.0, source="unknown")
   — Returns List[Document]
    │
    ├──────────────────────────────────────────────────┐
    │                                                  │
    ▼                                                  ▼
2. VectorStore.add_documents(documents)           3. GraphStore.add_documents(documents)
   in vector_store.py                                in graph_store.py
   — For each Document: _embed_text(content,          — For each Document: reads doc.metadata["entities"]
     task_type="retrieval_document")                  — Adds entity nodes to nx.Graph()
   — Local path: SentenceTransformer.encode()         — For every pair of entities in a doc,
     all-MiniLM-L6-v2, 384-dim, unit-normalised         increments edge weight (co-occurrence count)
   — Gemini path: genai.embed_content()               — Populates entity_to_docs[entity] → set of doc IDs
     models/embedding-001                             — Populates doc_by_id[id] → Document
   — _normalize() ensures L2 unit vectors
   — FAISS path: faiss.IndexFlatIP(dim).add(matrix)
   — NumPy fallback: np.vstack() into self.embeddings
    │                                                  │
    ▼                                                  ▼
   VectorStore ready                             GraphStore ready
   (FAISS inner-product index OR NumPy matrix)   (NetworkX undirected weighted graph +
                                                  two lookup dicts)
```

---

### Pipeline 2 — Hybrid Retrieval (PARALLEL strategy, the default)

Triggered by `pipeline.retrieve(query, top_k)`. Both retrieval paths run on the same query, then fuse.

```
query: str  (user question)
    │
    ▼
HybridRAGPipeline.retrieve(query, top_k) in pipeline.py
   — Checks self.strategy == SearchStrategy.PARALLEL
   — Calls _parallel_retrieval(query, top_k)
    │
    ├─────────────────────────────────┬────────────────────────────────────┐
    │   VECTOR PATH                   │   GRAPH PATH                       │
    ▼                                 ▼                                    │
1. VectorStore.search(query,      2. _extract_entities(query)              │
   top_k=max(1, top_k//2))           in pipeline.py                       │
   in vector_store.py                — spaCy NER via en_core_web_sm:       │
   — _embed_text(query,               doc.ents → lowercased strings        │
     "retrieval_query")               up to 6 entities                     │
   — Local: SentenceTransformer       — Keyword fallback: split, strip     │
     .encode(), normalised            punctuation, filter stopwords,       │
   — Gemini: embed_content()          keep words len>3, up to 6            │
   — _normalize() on query vec        Returns List[str]                    │
   — FAISS: index.search()                  │                              │
     → (scores, indices)                    ▼                              │
   — NumPy: embeddings @ query_vec   3. GraphStore.search_by_entity(       │
     argsort descending               entity, max_depth=2)                 │
   — Sets doc.score = cosine sim      in graph_store.py                    │
   — Sets doc.source = "vector"       — entity.lower()                     │
   — Returns List[Document]           — nx.single_source_shortest_path_    │
         │                              length(graph, entity, cutoff=2)    │
         │                            — For each reachable entity:         │
         │                              scores[doc_id] += 1.0              │
         │                            — Sets doc.score, doc.source="graph" │
         │                            — Returns List[Document]             │
         │                                  │                              │
         └─────────────────┬────────────────┘                              │
                           │                                               │
                           ▼                                               │
4. reciprocal_rank_fusion(vector_results, graph_results, top_k, k=60)
   in fusion.py
   — Iterates vector_results: scores[doc.id] += 1.0 / (60 + rank)
   — Iterates graph_results:  scores[doc.id] += 1.0 / (60 + rank)
   — Merges all_docs dict (doc.id → Document)
   — sorted(scores.items(), key=score, reverse=True)[:top_k]
   — Sets doc.score = fused RRF score
   — Sets doc.source = "hybrid"
   — Returns List[Document] (length = top_k)
```

---

### Pipeline 3 — Sequential Retrieval (SEQUENTIAL strategy)

Triggered when `strategy=SearchStrategy.SEQUENTIAL`. Uses vector hits to seed graph expansion.

```
query: str
    │
    ▼
1. VectorStore.search(query, top_k=max(1, top_k//2))
   — Returns top vector hits as List[Document]
    │
    ▼
2. For each doc in vector_results:
      _extract_entities(doc.content)    ← entities from DOC text, not query
      For each entity[:2]:
          GraphStore.search_by_entity(entity, max_depth=1)
          → extend expanded: List[Document]
    │
    ▼
3. _deduplicate_and_rank(expanded, top_k) in pipeline.py
   — dict keyed by doc.id, keeps highest score per id
   — sorted(best.values(), key=score, reverse=True)[:top_k]
   — Returns List[Document]
```

---

### Pipeline 4 — Answer Generation

Triggered by `pipeline.generate_response(query, documents)`. Takes the ranked documents and calls the LLM.

```
query: str  +  documents: List[Document]
    │
    ▼
1. Build context string in generate_response() in pipeline.py
   — "\n\n".join(f"[Source {i+1}] {doc.content}" for each doc)
   — context[:self.max_context_chars]   (truncated to MAX_CONTEXT_CHARS)
    │
    ▼
2. Assemble prompt string
   — "Answer using only the context. Cite sources like [Source 1].\n\n
      Context:\n{truncated_context}\n\nQuestion: {query}\n"
    │
    ├──────────────────────────────┬─────────────────────────────────────┐
    │  GROQ PATH                   │  GEMINI PATH                        │
    ▼                              ▼                                     │
3a. _configure_groq()          3b. _configure_gemini()                  │
    Groq(api_key)                  genai.configure(api_key)              │
    chat.completions.create(       GenerativeModel(generation_model)     │
      model=generation_model,      .generate_content(prompt)             │
      messages=[{role,content}],   response.text.strip()                 │
      temperature=0.2)                                                   │
    response.choices[0].message                                          │
    .content.strip()                                                     │
         │                              │                                │
         └──────────────────────────────┘                                │
                           │                                             │
                           ▼                                             │
                     answer: str  (cited answer returned to caller)      │
```

---

## Core Component Structure

```
hybrid-rag-system/
│
├── rag_hybrid/                        # Main package
│   ├── __init__.py                    # Re-exports Document, SearchStrategy, HybridRAGPipeline
│   │
│   ├── models.py                      # Data contracts
│   │   ├── Document (dataclass)       # id, content, metadata, score, source; .to_dict()
│   │   └── SearchStrategy (Enum)      # VECTOR_ONLY | GRAPH_ONLY | PARALLEL | SEQUENTIAL
│   │
│   ├── data_loader.py                 # I/O layer
│   │   └── load_documents_jsonl()     # JSONL → List[Document]
│   │
│   ├── vector_store.py                # Semantic retrieval
│   │   └── VectorStore
│   │       ├── add_documents()        # Embed + index (FAISS or NumPy)
│   │       ├── search()              # Query → top-k Documents by cosine sim
│   │       ├── _embed_text()         # Routes to local or gemini backend
│   │       ├── _embed_text_local()   # sentence-transformers all-MiniLM-L6-v2
│   │       ├── _embed_text_gemini()  # genai.embed_content models/embedding-001
│   │       └── _normalize()          # L2 normalization
│   │
│   ├── graph_store.py                 # Structural retrieval
│   │   └── GraphStore
│   │       ├── add_documents()       # Build nx.Graph from entity metadata
│   │       └── search_by_entity()    # BFS up to max_depth → scored Documents
│   │
│   ├── fusion.py                      # Fusion algorithm
│   │   └── reciprocal_rank_fusion()  # RRF merge of two ranked lists
│   │
│   └── pipeline.py                    # Orchestration
│       └── HybridRAGPipeline
│           ├── retrieve()             # Dispatches to strategy-specific method
│           ├── _parallel_retrieval()  # Vector + graph → RRF
│           ├── _sequential_retrieval()# Vector → graph expansion → deduplicate
│           ├── generate_response()    # Context build + LLM call
│           ├── _extract_entities()    # spaCy NER with keyword fallback
│           └── _deduplicate_and_rank()# Dedup by id, keep best score
│
├── data/
│   └── documents.jsonl                # Sample corpus: 5 docs, entity metadata
│
├── run_demo.py                        # CLI entry point; wires all components
├── requirements.txt                   # Pinned runtime dependencies
├── .env.example                       # Credential + config template
└── README.md                          # This file
```

---

## Hybrid Retrieval — How the Fusion Works

The key differentiator is that neither vector scores nor graph scores are used directly in the final ranking. Instead, both systems contribute through their **rank position**, not their raw score value, via Reciprocal Rank Fusion.

```
VECTOR RETRIEVAL SIDE
─────────────────────
query_vec = _normalize(_embed_text(query, "retrieval_query"))
         ↓
  If FAISS available:
    scores, indices = faiss.IndexFlatIP.search([query_vec], k)
    score = inner product (= cosine sim for unit vectors)
  Else:
    scores = embeddings_matrix @ query_vec   (NumPy dot product)
    best_idx = np.argsort(-scores)[:k]
         ↓
  vector_results = [Document(score=cosine_sim, source="vector"), ...]
  Ranked: highest cosine similarity first (rank 1 = best match)


GRAPH RETRIEVAL SIDE
────────────────────
entities = _extract_entities(query)          (spaCy or keyword fallback)
         ↓
  For each entity:
    related = nx.single_source_shortest_path_length(graph, entity, cutoff=2)
    For each reachable node:
        for each doc_id in entity_to_docs[node]:
            scores[doc_id] += 1.0            (simple count accumulation)
         ↓
  graph_results = [Document(score=entity_hit_count, source="graph"), ...]
  Ranked: most entity connections first (rank 1 = most related)


RECIPROCAL RANK FUSION  (fusion.py)
────────────────────────────────────
  k = 60   (constant that dampens the influence of top positions)

  For each document at rank r in vector_results:
      rrf_score[doc.id] += 1.0 / (60 + r)
        e.g. rank 1 → +0.01639
             rank 2 → +0.01613
             rank 5 → +0.01538

  For each document at rank r in graph_results:
      rrf_score[doc.id] += 1.0 / (60 + r)

  If a document appears in BOTH lists it accumulates from both:
      e.g. vector rank 1 + graph rank 1 → 0.01639 + 0.01639 = 0.03279
      vs.  vector only rank 1            → 0.01639

  Final ranking = sorted(rrf_score, descending)[:top_k]
  doc.score  ← fused RRF score
  doc.source ← "hybrid"


FINAL SCORE EXAMPLE (5 docs, top_k=4)
───────────────────────────────────────
  doc1: vector rank 1 + graph rank 1 → 0.0328  ← wins both
  doc4: vector rank 2 + graph rank 2 → 0.0323  ← second in both
  doc3: graph only rank 3            → 0.0154
  doc2: vector only rank 3           → 0.0154
```

The constant k=60 is the standard RRF parameter: it prevents the top rank from dominating overwhelmingly, so a document ranked 2nd in both lists typically beats one ranked 1st in only one list.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Embeddings (local) | sentence-transformers `all-MiniLM-L6-v2` | 384-dim unit vectors, no API key, fast CPU inference |
| Embeddings (cloud) | Google Gemini `models/embedding-001` | Higher quality for production; requires `GEMINI_API_KEY` |
| Vector index | FAISS `IndexFlatIP` | Exact inner-product search; GPU-portable; drops to NumPy fallback automatically |
| NumPy fallback | NumPy dot product + argsort | Zero extra dependencies when FAISS unavailable |
| Graph store | NetworkX undirected weighted `Graph` | BFS / shortest-path traversal; edge weight = entity co-occurrence count |
| Fusion | Custom `reciprocal_rank_fusion()` | RRF is rank-based (no score calibration needed), well-studied, one hyperparameter |
| NER | spaCy `en_core_web_sm` | Fast on-device NER; keyword fallback means spaCy is optional |
| LLM — default | Groq `llama-3.1-8b-instant` | Sub-second generation on Groq inference hardware |
| LLM — alternative | Google Gemini `GenerativeModel` | Larger context window option; same API surface |
| Data serialisation | JSONL | Streaming-friendly, one document per line, easy to extend |
| Config management | python-dotenv | 12-factor app pattern; `.env.example` ships with repo |
| CLI | argparse | Zero-dependency, standard library |

---

## How to Run It

### Prerequisites

- Python 3.8 or higher
- pip
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com)) OR a Google Gemini API key

### Install

```bash
git clone https://github.com/raghdaemara1/hybrid-rag-system.git
cd hybrid-rag-system

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```
GEMINI_API_KEY=your-gemini-key-here        # optional
GROQ_API_KEY=your-groq-key-here            # needed for default LLM path
RAG_EMBEDDINGS=local                        # or "gemini"
GROQ_MODEL=llama-3.1-8b-instant            # any model available on Groq
MAX_CONTEXT_CHARS=2000                      # prompt context budget
```

### Run the Demo

```bash
# Basic — uses local embeddings, Groq LLM
python run_demo.py --question "What is hybrid RAG?"

# Custom top-k and context
python run_demo.py --question "How does fusion work?" --top-k 6 --max-context-chars 3000

# Interactive prompt if no --question flag given
python run_demo.py
```

### Expected Output

```
Retrieved sources:
1. (hybrid, score=0.0328) Hybrid RAG blends vector search with graph search for better coverage....
2. (hybrid, score=0.0323) Reciprocal Rank Fusion merges ranked lists by rank....
3. (hybrid, score=0.0161) Vector search uses embeddings to find similar text....
4. (hybrid, score=0.0154) Graph search expands context using linked entities....

Answer:

Hybrid RAG is a retrieval-augmented generation approach that combines both
vector search and graph-based retrieval. [Source 1] The results are merged
using Reciprocal Rank Fusion, which ranks documents by their position across
both retrieval lists. [Source 2]
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (if using Groq) | — | API key for Groq chat completions (LLM generation) |
| `GEMINI_API_KEY` | Yes (if using Gemini) | — | Google Gemini API key for embeddings and/or generation |
| `RAG_EMBEDDINGS` | No | `local` | Embedding backend: `local` (sentence-transformers) or `gemini` |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model ID passed to `chat.completions.create()` |
| `MAX_CONTEXT_CHARS` | No | `2000` | Maximum characters of retrieved context inserted into the generation prompt |

At least one of `GROQ_API_KEY` or `GEMINI_API_KEY` must be set; the system raises `ValueError` at startup otherwise. The routing logic in `generate_response()` prefers Groq when `GROQ_API_KEY` is set and the model name does not start with `"gemini"`.

---

## Key Design Decisions

**Why Reciprocal Rank Fusion instead of weighted score combination?**
Vector cosine similarity values (0–1) and graph entity hit counts (integers) live on completely different scales. Normalising and weighting them requires domain tuning for every new corpus. RRF operates on rank positions, which are always comparable, and its single constant k=60 is robust across tasks with no per-corpus calibration.

**Why FAISS `IndexFlatIP` instead of approximate nearest-neighbour (HNSW, IVF)?**
The sample corpus is five documents, and real use cases at this scale fit comfortably in an exact flat index. `IndexFlatIP` guarantees no missed results and adds FAISS as the only difference from NumPy. Upgrading to `IndexHNSWFlat` requires changing one constructor line.

**Why lazy-load LLM and embedding clients?**
Both `VectorStore` and `HybridRAGPipeline` defer client construction (`_groq`, `_gemini`, `_st_model` all start as `None`) until first use. This means the system starts in under a second even if a model download or API handshake is slow, and running vector-only mode never touches the Gemini SDK.

**Why spaCy with a keyword fallback instead of requiring spaCy?**
spaCy and its `en_core_web_sm` model are a 50 MB download. Making it optional with a graceful stopword-filtering fallback means the system runs in constrained environments (CI, small containers) without modifying any code. Named entities still improve graph recall when spaCy is available.

**Why store documents in JSONL with explicit `entities` metadata instead of extracting entities at index time?**
Entities baked into the corpus are deterministic and reproducible. Index-time extraction with spaCy would produce different graphs depending on the spaCy model version. The JSONL schema allows human-curated or pipeline-generated entity lists without changing any retrieval code.

**Why four `SearchStrategy` modes instead of a single hybrid path?**
Ablation and debugging. `VECTOR_ONLY` and `GRAPH_ONLY` let you measure each retrieval arm in isolation. `SEQUENTIAL` provides a cascade (vector seeds graph expansion) as an alternative fusion without RRF. `PARALLEL` + RRF is the production default.

**Why cap context with `MAX_CONTEXT_CHARS` instead of token counting?**
Token counts are model-specific and require the tokeniser. Character limits are model-agnostic, require no extra dependency, and are a conservative upper bound (one token ≥ one character for most English text). For tighter control, replace the slice with a tiktoken counter.

---

## Production Readiness

| Current (demo) | Production recommendation |
|---|---|
| 5-document JSONL corpus | Replace `load_documents_jsonl()` with a database-backed loader or streaming ingest pipeline |
| FAISS flat in-memory index | Persist with `faiss.write_index()` / `faiss.read_index()`; migrate to IVF or HNSW for >100k docs |
| NetworkX in-memory graph | Export to Neo4j or Amazon Neptune for multi-hop queries and persistent graph storage |
| Single-process, synchronous | Wrap `retrieve()` and `generate_response()` in async handlers (FastAPI + asyncio) |
| No re-ranking | Add a cross-encoder re-ranking step between fusion output and `generate_response()` |
| Manual entity metadata | Replace static JSONL entities with a spaCy/Flair NER pipeline at ingest time |
| `temperature=0.2` hard-coded | Expose as an env var or per-request parameter for tunable factuality |
| No observability | Add structured logging, OpenTelemetry spans around retrieval and generation calls |
| Character-based context truncation | Swap to tokeniser-based truncation (tiktoken) for accurate context window management |

---

## Project File Map

| File | What it does |
|---|---|
| `run_demo.py` | CLI entry point: parses args, loads `.env`, instantiates `VectorStore` + `GraphStore` + `HybridRAGPipeline`, calls `retrieve()` then `generate_response()`, prints results |
| `rag_hybrid/__init__.py` | Package surface: re-exports `Document`, `SearchStrategy`, `HybridRAGPipeline` |
| `rag_hybrid/models.py` | Defines `Document` dataclass (id, content, metadata, score, source) and `SearchStrategy` enum (four values) |
| `rag_hybrid/data_loader.py` | `load_documents_jsonl()` — reads JSONL file line-by-line into `List[Document]` |
| `rag_hybrid/vector_store.py` | `VectorStore` — embeds documents and queries (local or Gemini), indexes with FAISS or NumPy, returns cosine-ranked `List[Document]` |
| `rag_hybrid/graph_store.py` | `GraphStore` — builds entity co-occurrence graph with NetworkX, performs BFS traversal to find related documents |
| `rag_hybrid/fusion.py` | `reciprocal_rank_fusion()` — pure function that merges two ranked `List[Document]` into one using the RRF formula |
| `rag_hybrid/pipeline.py` | `HybridRAGPipeline` — orchestrates retrieval strategy dispatch, entity extraction, answer generation via Groq or Gemini |
| `data/documents.jsonl` | Sample five-document corpus with entity metadata for demo use |
| `requirements.txt` | Pinned runtime dependencies: groq, google-generativeai, faiss-cpu, networkx, sentence-transformers, spacy, numpy, python-dotenv |
| `.env.example` | Template for `GROQ_API_KEY`, `GEMINI_API_KEY`, `RAG_EMBEDDINGS`, `GROQ_MODEL`, `MAX_CONTEXT_CHARS` |

---

*Built with sentence-transformers · FAISS · NetworkX · spaCy · Groq · Google Gemini · python-dotenv*
