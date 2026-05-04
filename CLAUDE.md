# Text_Mining_Project — RAG/NLP Research

## Purpose
Research project evaluating RAG pipelines for text mining tasks. Covers data
scraping, chunking, embedding, retrieval, and automated evaluation using
BERTScore, ROUGE, and a custom ablation harness.

## Architecture
- `agent/rag_agent.py` — main RAG agent (ChromaDB retrieval + Ollama generation)
- `indexer/chunk_and_embed.py` — chunks documents and stores embeddings in ChromaDB
- `scraper/scraper.py` + `scraper/cleaner.py` — raw data ingestion and cleaning
- `eval/run_evaluation.py` — runs full evaluation pipeline against ground truth
- `eval/ablation.py` — ablation experiments (chunking strategies, retrieval params)
- `eval/eval_dataset.json` — ground-truth QA pairs
- `eval/ablation_results.json` — stored ablation outputs
- `notebooks/` — Jupyter notebooks for analysis and plotting
- `report/` — figures and analysis outputs

## Environment
- Vector DB: ChromaDB (port 8000), started via `docker compose`
- LLM: Ollama (port 11434), GPU-enabled via nvidia docker runtime
- Python: `requirements.txt` (langchain 0.2.16, chromadb 0.5.15, bert-score 0.3.13)
- Data dirs `data/raw/`, `data/clean/`, `data/chroma/` are gitignored runtime state

## Common Commands
```bash
docker compose up -d                 # start ChromaDB + Ollama
python agent/rag_agent.py           # run the RAG agent
python indexer/chunk_and_embed.py   # chunk and embed documents
python eval/run_evaluation.py       # run full evaluation
python eval/ablation.py             # run ablation experiments
```

## Git
- Standalone repo: `git log`, `git commit`, `git push` all work from this directory.
- Do NOT run git commands from the parent `Personal_Projects_Agent/` directory.
