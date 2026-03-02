# Modular RAG MCP Server

A modular RAG (Retrieval-Augmented Generation) system with MCP (Model Context Protocol) integration.

## Features

- Hybrid retrieval: BM25 + Dense Embedding with RRF fusion
- Pluggable components: LLM / Embedding / VectorStore / Reranker / Evaluator
- MCP Server via Stdio Transport
- Full observability: structured tracing + Streamlit dashboard
- Configuration-driven: switch components via `config/settings.yaml`

## Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Start MCP Server
python main.py
```

## Project Structure

See `DEV_SPEC.md` for detailed architecture and module design.

## License

MIT
