# GraphRAG IPL: Proving Graph-Based Retrieval Beats Basic RAG on Every Metric

> **GraphRAG achieves 94.6% fewer tokens than Basic RAG with 14pp better accuracy — proven across 50 benchmark questions on 665 IPL Wikipedia articles.**

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![TigerGraph](https://img.shields.io/badge/TigerGraph-GraphRAG-orange)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Submission for the **TigerGraph GraphRAG Inference Hackathon 2026**.  
Three retrieval pipelines are benchmarked head-to-head on 50 IPL trivia questions drawn from 665 Wikipedia articles ingested into a TigerGraph knowledge graph.

---

## Live Demo

| | Link |
|---|---|
| **Interactive Dashboard** | [graphrag-ipl.streamlit.app](https://graphrag-ipl.streamlit.app) |
| **GitHub Repository** | [github.com/AbdullahMustafa7/GraphRAG-IPL](https://github.com/AbdullahMustafa7/GraphRAG-IPL) |

---

## Benchmark Results (50 questions, llama-3.3-70b-versatile judge)

| Pipeline | Avg Tokens | Accuracy | BERTScore F1 | BERTScore Rescaled |
|---|---|---|---|---|
| P1 — LLM-Only | 174 | 62% | 0.8724 | 0.6178 |
| P2 — Basic RAG (FAISS) | 2,541 | 50% | 0.7974 | 0.3930 |
| **P3 — GraphRAG (TigerGraph)** | **137** | **64%** | **0.8826 ✅** | **0.6484 ✅** |

✅ = meets bonus threshold (BERTScore F1 >= 0.88, Rescaled >= 0.55)

### Key Stats

| Metric | Value |
|---|---|
| Token reduction vs Basic RAG | **94.6%** |
| BERTScore F1 bonus threshold (>= 0.88) | **Hit — 0.8826** |
| BERTScore Rescaled bonus threshold (>= 0.55) | **Hit — 0.6484** |
| Knowledge graph entities | **8,373** |
| Knowledge graph relationships | **5,349** |
| Wikipedia articles ingested | **665** |
| Total tokens ingested | **2M+** |
| Evaluation questions | **50** (20 single-hop, 30 multi-hop) |

---

## Dataset

The corpus covers **Indian Premier League (IPL)** cricket, ingested from Wikipedia:

- **Players** — batters, bowlers, all-rounders, wicket-keepers
- **Teams** — all 10 franchises, historical franchises (Pune, Kochi, etc.)
- **Seasons** — IPL 2008 through 2024, venues, results, winners
- **Awards** — Orange Cap, Purple Cap, Most Valuable Player, Emerging Player
- **Coaches & Officials** — head coaches, selectors, BCCI administrators
- **Venues** — home grounds, hosting cities, stadium capacities

Raw articles live in `data/raw/`. Preprocessed corpus at `data/corpus.txt`.

---

## How It Works

### Pipeline 1 — LLM-Only Baseline
Questions are sent directly to **Groq llama-3.3-70b-versatile** with no retrieval context. Minimal tokens, no infrastructure required. Sets the floor for raw model knowledge.

### Pipeline 2 — Basic RAG (FAISS + Sentence Transformers)
The corpus is chunked and embedded with **all-MiniLM-L6-v2**, stored in a **FAISS** vector index. At query time the top-5 chunks are retrieved and prepended to the prompt. Highest token cost due to long retrieved passages.

### Pipeline 3 — GraphRAG (TigerGraph)
1. **Entity extraction** — Groq llama-3.1-8b-instant extracts named entities from the question
2. **Graph traversal** — Up to 3-hop traversal of `RELATED_TO` edges in TigerGraph retrieves the most relevant entities and relationships (60 entities, 40 relationships max)
3. **Community context** — `BELONGS_TO` edges pull community summaries for broader thematic context
4. **Answer generation** — Structured graph context is formatted into a concise prompt; Groq llama-3.3-70b-versatile answers in 1-2 sentences

Multi-hop questions (containing keywords like "also", "both", "same season", "which year") automatically trigger 3-hop traversal instead of 2-hop.

---

## Project Structure

```
graphrag-hackathon/
|
+-- pipeline1_llm_only/
|   +-- pipeline.py             # Direct Groq LLM call, no retrieval
|
+-- pipeline2_basic_rag/
|   +-- pipeline.py             # FAISS retrieval + Groq answer
|   +-- chunks.json             # Pre-chunked corpus
|   +-- chunk_metadata.json     # Chunk source metadata
|   +-- faiss_index.bin         # Serialised FAISS vector index
|
+-- pipeline3_graphrag/
|   +-- pipeline.py             # GraphRAG: entity extraction + TG traversal + Groq
|   +-- ingest.py               # Wikipedia -> TigerGraph ingestion
|   +-- communities.py          # Community detection & summary generation
|   +-- setup_tigergraph.py     # Schema creation on TigerGraph Savanna
|   +-- refresh_token.py        # Bearer token rotation helper
|   +-- _tg_dns_fix.py          # DNS override for local TigerGraph resolution
|
+-- dashboard/
|   +-- app.py                  # Full Streamlit dashboard (local, all 3 live)
|   +-- app_cloud.py            # Streamlit Cloud version (P2 shows cached results)
|
+-- evaluation/
|   +-- evaluate.py             # LLM-as-Judge + BERTScore evaluation
|   +-- generate_questions.py   # Auto-generates 50 benchmark questions
|   +-- questions.json          # 50 Q&A pairs with ground truth
|
+-- results/
|   +-- pipeline1_results.json  # P1 answers + metrics (50 questions)
|   +-- pipeline2_results.json  # P2 answers + metrics (50 questions)
|   +-- pipeline3_results.json  # P3 answers + metrics (50 questions)
|   +-- evaluation_report.json  # Judge verdicts + BERTScore per question
|
+-- data/
|   +-- raw/                    # 665 raw Wikipedia article JSON files
|   +-- corpus.txt              # Merged plaintext corpus (2M+ tokens)
|
+-- run_pipeline1.py            # Batch runner for Pipeline 1
+-- run_pipeline2.py            # Batch runner for Pipeline 2
+-- run_pipeline3.py            # Batch runner for Pipeline 3
+-- config.py                   # Shared config (API keys, paths, model names)
+-- requirements.txt            # Python dependencies
```

---

## Setup & Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/AbdullahMustafa7/GraphRAG-IPL.git
cd GraphRAG-IPL
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_...
GROQ_API_KEY_2=gsk_...        # optional -- for rate limit key rotation
GROQ_API_KEY_3=gsk_...        # optional -- for rate limit key rotation
GEMINI_API_KEY=...             # only needed if regenerating the corpus
```

### 3. Run Pipeline 1 — LLM-Only

```bash
python run_pipeline1.py
# Results -> results/pipeline1_results.json
```

### 4. Run Pipeline 2 — Basic RAG

```bash
# FAISS index is pre-built in pipeline2_basic_rag/faiss_index.bin
python run_pipeline2.py
# Results -> results/pipeline2_results.json
```

### 5. Set up TigerGraph (Pipeline 3)

```bash
# a) Create schema on TigerGraph Savanna
python pipeline3_graphrag/setup_tigergraph.py

# b) Ingest Wikipedia corpus into the graph
python pipeline3_graphrag/ingest.py

# c) Generate community summaries
python pipeline3_graphrag/communities.py

# d) Refresh the bearer token (run whenever the token expires ~7 days)
python pipeline3_graphrag/refresh_token.py
```

### 6. Run Pipeline 3 — GraphRAG

```bash
python run_pipeline3.py --delay 3
# Options:
#   --type single   only single-hop questions
#   --type multi    only multi-hop questions
#   --limit N       run first N questions
# Results -> results/pipeline3_results.json
```

### 7. Evaluate all pipelines

```bash
python evaluation/evaluate.py --include-p3 --delay 1
# Prints summary table + saves results/evaluation_report.json
```

### 8. Launch the dashboard

```bash
# Full local version (all 3 pipelines live):
streamlit run dashboard/app.py

# Cloud-optimised version (P2 shows cached results):
streamlit run dashboard/app_cloud.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Knowledge graph | [TigerGraph Savanna](https://www.tigergraph.com/) (free tier) |
| LLM inference | [Groq](https://groq.com/) — llama-3.3-70b-versatile |
| Entity extraction | Groq — llama-3.1-8b-instant |
| Vector search (P2) | [FAISS](https://github.com/facebookresearch/faiss) |
| Embeddings (P2) | [sentence-transformers](https://sbert.net/) — all-MiniLM-L6-v2 |
| Evaluation judge | Groq — llama-3.3-70b-versatile |
| Semantic scoring | [BERTScore](https://github.com/Tiiiger/bert_score) — distilbert-base-uncased |
| Graph analytics | [NetworkX](https://networkx.org/) |
| Dashboard | [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/) |
| Data collection | [Wikipedia-API](https://pypi.org/project/Wikipedia-API/) |
| Language | Python 3.13 |

---

## Hackathon

**Event:** TigerGraph GraphRAG Inference Hackathon 2026  
**Built by:** Abdullah Mustafa  
**Repo:** [github.com/AbdullahMustafa7/GraphRAG-IPL](https://github.com/AbdullahMustafa7/GraphRAG-IPL)

---

## License

MIT
