# Hybrid RAG System: Vector + Graph Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready hybrid Retrieval-Augmented Generation (RAG) system that combines **vector search** and **graph-based retrieval** with intelligent fusion for enhanced question-answering capabilities.

## Overview

This project demonstrates advanced RAG techniques by merging two complementary retrieval methods:

- **Vector Retrieval**: Semantic search using embeddings (local or Gemini API)
- **Graph Retrieval**: Entity relationship mapping with NetworkX for context-aware results
- **Reciprocal Rank Fusion**: Intelligent result merging for optimal answer quality

The system supports multiple LLM providers (Groq, Gemini) and offers flexible embedding options, making it adaptable for various use cases from research to production deployments.

## Key Features

- **Hybrid Architecture**: Combines semantic and structural information retrieval
- **Flexible LLM Support**: Compatible with Groq and Google Gemini APIs
- **Embedding Options**: Local embeddings (no API needed) or Gemini embeddings
- **Entity Graph**: Automatic entity extraction and co-occurrence graph construction
- **Smart Fusion**: Reciprocal rank fusion for optimal result ranking
- **Citation Support**: Responses include source citations for transparency
- **Environment-based Config**: Secure API key management with `.env` files
- **Lightweight & Fast**: Minimal dependencies, optimized for quick experimentation

## Architecture

```
User Query
    ↓
┌───────────────────────────────────┐
│   Dual Retrieval Strategy         │
├───────────────┬───────────────────┤
│ Vector Search │  Graph Traversal  │
│  (Semantic)   │   (Relational)    │
└───────┬───────┴────────┬──────────┘
        │                │
        └────────┬────────┘
                 ↓
      Reciprocal Rank Fusion
                 ↓
           LLM Generation
           (with citations)
                 ↓
            Response
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hybrid-rag-system.git # TODO: Replace with your repo URL
   cd hybrid-rag-system
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your API keys
   # Required: GROQ_API_KEY (for LLM generation)
   # Optional: GEMINI_API_KEY (for Gemini embeddings/generation)
   ```

## Usage

### Basic Usage

Run the demo with a sample question:

```bash
python run_demo.py --question "What is hybrid RAG?"
```

### Advanced Options

```bash
# Use Gemini embeddings instead of local
export RAG_EMBEDDINGS=gemini
python run_demo.py --question "Your question here"

# Use Gemini for generation (instead of default Groq)
python run_demo.py --question "Your question here" --model gemini
```

### Example Output

```
Question: What is hybrid RAG?

Retrieved Documents:
- [Vector] Document about RAG architectures (score: 0.89)
- [Graph] Related concepts: retrieval, generation, LLM (score: 0.76)

Answer:
Hybrid RAG combines multiple retrieval strategies to improve answer quality...
[Citation: doc_001, doc_003]
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes* | - | Groq API key for LLM generation |
| `GEMINI_API_KEY` | No | - | Google Gemini API key (optional) |
| `RAG_EMBEDDINGS` | No | `local` | Embedding mode: `local` or `gemini` |

*Required if using Groq for generation (default)

### Embedding Modes

- **Local** (`RAG_EMBEDDINGS=local`): Uses sentence-transformers, no API calls needed
- **Gemini** (`RAG_EMBEDDINGS=gemini`): Uses Google's embedding API, requires `GEMINI_API_KEY`

## Project Structure

```
rag_hybrid/
├── data/
│   └── documents.jsonl          # Sample documents for indexing
├── rag_hybrid/
│   ├── __init__.py
│   ├── retrieval.py             # Vector & graph retrieval logic
│   ├── fusion.py                # Reciprocal rank fusion
│   ├── graph_builder.py         # Entity graph construction
│   └── llm_client.py            # LLM provider integrations
├── run_demo.py                  # Main demo script
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## Technical Stack

- **LLM Providers**: Groq, Google Gemini
- **Embeddings**: sentence-transformers, Google Gemini
- **Graph Library**: NetworkX
- **Vector Search**: NumPy-based cosine similarity
- **NLP**: spaCy for entity extraction
- **Environment**: python-dotenv

## How It Works

1. **Document Indexing**: Sample documents are loaded and indexed with both vector embeddings and entity graph
2. **Query Processing**: User query is embedded and entities are extracted
3. **Dual Retrieval**:
   - Vector search finds semantically similar documents
   - Graph traversal finds contextually related documents
4. **Fusion**: Results are merged using reciprocal rank fusion
5. **Generation**: Top results are sent to LLM with the query for answer generation
6. **Citation**: Response includes references to source documents

## Use Cases

- **Knowledge Management**: Internal documentation search and Q&A
- **Research Assistant**: Academic paper analysis and question answering
- **Customer Support**: Automated response systems with source citations
- **Educational Tools**: Interactive learning platforms
- **Content Discovery**: Enhanced search for content platforms

## Limitations & Future Work

- Currently uses simplified entity extraction (can be enhanced with custom NER)
- Graph construction is based on co-occurrence (can incorporate more sophisticated relationships)
- Sample dataset is small (designed for demonstration purposes)
- Future: Add re-ranking, query expansion, and multi-hop reasoning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## API Key Security

**Important**: Never commit your `.env` file or expose API keys in code. This project uses environment variables for secure credential management.

## Acknowledgments

- Built with [Groq](https://groq.com/) and [Google Gemini](https://ai.google.dev/) APIs
- Inspired by state-of-the-art RAG research and best practices
- Uses open-source libraries: NetworkX, sentence-transformers, spaCy

## Contact & Portfolio

**Author**: [Your Name] <!-- TODO: Replace with your actual name -->
**GitHub**: [@yourusername](https://github.com/yourusername) <!-- TODO: Replace with your GitHub username -->
**LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile) <!-- TODO: Replace with your LinkedIn profile -->
**Portfolio**: [yourwebsite.com](https://yourwebsite.com) <!-- TODO: Replace with your portfolio URL -->

---

⭐ If you found this project helpful, please consider giving it a star!
