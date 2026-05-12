# 🏏 GraphRAG IPL Hackathon

> **81.8% token reduction** using TigerGraph GraphRAG on the IPL Wikipedia dataset

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![TigerGraph](https://img.shields.io/badge/TigerGraph-GraphRAG-orange?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-brightgreen?logo=groq)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit&logoColor=white)
![Hackathon](https://img.shields.io/badge/TigerGraph-GraphRAG%20Hackathon%202026-purple)

Submission for the **TigerGraph GraphRAG Inference Hackathon 2026**.  
Three retrieval pipelines are benchmarked head-to-head on 50 IPL trivia questions drawn from 665 Wikipedia articles.

---

## 🏆 Key Result

| Pipeline | Avg Tokens | Avg Cost / Query | Avg Latency |
|---|---|---|---|
| P1 — LLM-Only (Gemini 2.5 Flash) | ~500 | ~$0.000038 | ~2,100 ms |
| P2 — Basic RAG (FAISS + Groq) | **2,541** | $0.000194 | 1,875 ms |
| **P3 — GraphRAG (TigerGraph + Groq)** | **464** | $0.000277 | ~1,200 ms |

> **GraphRAG uses 81.8% fewer tokens than Basic RAG** (464 vs 2,541 avg tokens per query), because graph traversal delivers only the precise subgraph relevant to the question instead of bulk-retrieving the top-K embedding neighbours.

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │           665 IPL Wikipedia Articles          │
                        └──────────────┬──────────────────────────────┘
                                       │ data/collect_dataset.py
                   ┌───────────────────┼───────────────────────┐
                   ▼                   ▼                        ▼
         ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
         │  Pipeline 1  │    │   Pipeline 2      │    │    Pipeline 3        │
         │  LLM-Only    │    │   Basic RAG       │    │    GraphRAG          │
         │              │    │                  │    │                      │
         │ Gemini 2.5   │    │ all-MiniLM-L6-v2 │    │ Groq (entity        │
         │ Flash        │    │ → FAISS index    │    │ extraction)          │
         │              │    │                  │    │    ↓                  │
         │ No retrieval │    │ Top-5 chunk      │    │ TigerGraph           │
         │              │    │ retrieval        │    │ 2-hop traversal      │
         │              │    │    ↓             │    │    ↓                  │
         │              │    │ Groq llama-70b   │    │ Community context    │
         │              │    │ generation       │    │    ↓                  │
         │              │    │                  │    │ Groq llama-70b       │
         └──────┬───────┘    └───────┬──────────┘    └──────────┬───────────┘
                │                   │                            │
                └───────────────────┴────────────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │ Streamlit        │
                              │ Comparison       │
                              │ Dashboard        │
                              │ localhost:8502   │
                              └─────────────────┘
```

### Pipeline Deep Dive

**Pipeline 1 — LLM-Only** (`pipeline1_llm_only/`)  
Sends the question directly to Gemini 2.5 Flash with a cricket expert system prompt. No retrieval, no context. Pure parametric knowledge. Establishes the baseline.

**Pipeline 2 — Basic RAG** (`pipeline2_basic_rag/`)  
1. Chunks all 665 articles into 512-token windows with 50-token overlap  
2. Embeds chunks with `all-MiniLM-L6-v2` → FAISS flat inner-product index  
3. Retrieves top-5 chunks by cosine similarity  
4. Sends the 5-chunk context block (~2,500 tokens) to Groq llama-3.3-70b-versatile  

**Pipeline 3 — GraphRAG** (`pipeline3_graphrag/`)  
1. Extracts entity names from the question using Groq llama-3.1-8b-instant (~50 tokens)  
2. Looks up entities in TigerGraph as seed vertices  
3. 2-hop `RELATED_TO` graph traversal (capped at 30 entities, 20 relationships)  
4. Fetches community summaries via `BELONGS_TO` → `Community` edges  
5. Builds a focused structured prompt (~464 tokens) and calls Groq llama-3.3-70b-versatile  
6. Falls back to direct Groq call if TigerGraph is unreachable (graceful degradation)

---

## 📁 Project Structure

```
graphrag-hackathon/
│
├── data/
│   ├── collect_dataset.py          # Scrapes 665 IPL Wikipedia articles
│   ├── raw/                        # Article .txt files (gitignored — too large)
│   └── corpus.txt                  # Merged corpus (gitignored — too large)
│
├── pipeline1_llm_only/
│   └── pipeline.py                 # Gemini 2.5 Flash, no retrieval
│
├── pipeline2_basic_rag/
│   ├── pipeline.py                 # Chunking + FAISS + Groq
│   ├── chunks.json                 # Generated on first run (gitignored)
│   ├── chunk_metadata.json         # Generated on first run (gitignored)
│   └── faiss_index.bin             # Generated on first run (gitignored — large)
│
├── pipeline3_graphrag/
│   ├── pipeline.py                 # Entity extraction + TigerGraph + Groq
│   ├── ingest.py                   # Bulk article → entity/relationship ingestion
│   ├── communities.py              # Community detection & summarisation
│   ├── setup_tigergraph.py         # Schema creation on TigerGraph Cloud
│   ├── refresh_token.py            # Rotate TigerGraph bearer token
│   └── tg_token.txt                # Bearer token (gitignored — credentials)
│
├── dashboard/
│   └── app.py                      # Streamlit comparison dashboard
│
├── evaluation/
│   └── generate_questions.py       # Generates 50 IPL eval questions (JSON)
│
├── results/
│   ├── pipeline2_results.json      # P2 batch run output
│   ├── pipeline3_results.json      # P3 batch run output
│   ├── ingest_progress.json        # TigerGraph ingestion checkpoint
│   └── ingest_failures.json        # Failed ingestion records for retry
│
├── config.py                       # API keys, model names, paths, pricing
├── run_pipeline1.py                # Batch runner — Pipeline 1
├── run_pipeline2.py                # Batch runner — Pipeline 2
├── run_pipeline3.py                # Batch runner — Pipeline 3
├── .env.example                    # Environment variable template
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/AbdullahMustafa7/GraphRAG-IPL.git
cd GraphRAG-IPL
pip install -r requirements.txt
```

Or install manually:

```bash
pip install google-genai groq faiss-cpu sentence-transformers tiktoken \
            pyTigerGraph python-dotenv requests streamlit plotly pandas tqdm
```

### 2. Configure API keys

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

> **TigerGraph:** The cloud instance credentials are embedded in `pipeline3_graphrag/pipeline.py`. Fetch your bearer token once with:
> ```bash
> python pipeline3_graphrag/refresh_token.py
> ```

### 3. Collect the dataset

```bash
python data/collect_dataset.py
```

Downloads ~665 IPL Wikipedia articles into `data/raw/` and merges them into `data/corpus.txt`. Takes ~10 minutes (polite rate-limiting at 0.5s per article).

### 4. Generate evaluation questions

```bash
python evaluation/generate_questions.py
```

Produces `evaluation/questions.json` — 50 questions split between single-hop and multi-hop.

---

## 🚀 Running the Pipelines

### Pipeline 1 — LLM-Only (Gemini)

```bash
# Full 50-question batch
python run_pipeline1.py

# Quick smoke-test (5 questions)
python run_pipeline1.py --limit 5

# Single question
python pipeline1_llm_only/pipeline.py "Who won the IPL 2023 title?"
```

### Pipeline 2 — Basic RAG (FAISS + Groq)

```bash
# Build FAISS index only (runs once, cached to disk)
python run_pipeline2.py --build-only

# Full 50-question batch
python run_pipeline2.py

# Smoke-test; slower delay if rate-limited
python run_pipeline2.py --limit 5 --delay 6

# Single question
python pipeline2_basic_rag/pipeline.py "Who won the IPL 2023 title?"
```

### Pipeline 3 — GraphRAG (TigerGraph + Groq)

```bash
# Step A: Ingest articles into TigerGraph (resumable)
python pipeline3_graphrag/ingest.py

# Step B: Build community summaries (run after >500 articles ingested)
python pipeline3_graphrag/communities.py

# Step C: Full 50-question batch
python run_pipeline3.py

# Smoke-test
python run_pipeline3.py --limit 5 --delay 5
```

---

## 📊 Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8502**

Features:
- Live query box — runs all three pipelines simultaneously
- Side-by-side answers with token count, latency, and cost
- 🏆 Best badge on the lowest-token pipeline
- Plotly bar charts (tokens + cost per pipeline)
- Results history loaded from `results/*.json`
- **"GraphRAG uses 81.8% fewer tokens than Basic RAG"** highlight stat
- Yellow `⚠️` warning (not a red error) when TigerGraph is temporarily unavailable — fallback answer still shown

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Graph database | [TigerGraph Cloud](https://tgcloud.io) — `MyDatabase` graph |
| LLM (P1) | Google Gemini 2.5 Flash via `google-genai` |
| LLM (P2 & P3) | Groq `llama-3.3-70b-versatile` (generation) + `llama-3.1-8b-instant` (entity extraction) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector index | FAISS `IndexFlatIP` (exact cosine similarity) |
| Tokenizer | `tiktoken` — `cl100k_base` encoding |
| Dashboard | Streamlit 1.57 + Plotly |
| Dataset | Wikipedia via `wikipedia-api` — 665 IPL articles |

---

## 📈 Results Detail

### Token efficiency (measured on completed queries)

```
Pipeline 2 — Basic RAG
  Avg prompt tokens    : 2,524.1
  Avg completion tokens:    16.8
  Avg total tokens     : 2,540.9
  Avg latency          : 1,874.6 ms
  Total cost (22 ok)   : $0.004276

Pipeline 3 — GraphRAG
  Avg total tokens     :   463.5   ← 81.8% fewer than P2
  Avg cost / query     : $0.000277
  Graph entities found : ~14 per query (2-hop traversal)
  Fallback rate        : varies with ingestion completeness
```

### Why GraphRAG uses fewer tokens

Basic RAG retrieves the top-5 chunks regardless of relevance — each chunk is 512 tokens, so every prompt is padded with ~2,500 tokens of context whether it's needed or not.

GraphRAG uses entity-centric traversal: the question is parsed for entity names (e.g. `ms_dhoni`, `chennai_super_kings`), and only the 2-hop neighbourhood of those entities is included. A typical subgraph has 10–30 entities and 10–20 relationships — compact, structured, and on-topic.

---

## 🏅 Hackathon

**Event:** TigerGraph GraphRAG Inference Hackathon 2026  
**Track:** Open — Best use of TigerGraph for GraphRAG inference  
**Repository:** https://github.com/AbdullahMustafa7/GraphRAG-IPL  

---

## 📄 License

MIT
