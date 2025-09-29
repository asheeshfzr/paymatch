# 🔹 PayMatch: Intelligent User–Transaction Mapping & Semantic Search

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-🚀-green.svg)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20GPT-orange.svg)

**PayMatch** is a hybrid AI system that solves **two core problems in financial systems**:

1. **User–Transaction Matching** → Identify the most likely user(s) for a given transaction.
2. **Semantic Transaction Search** → Retrieve transactions matching free-text queries (e.g., *"consulting fee in July"*).

The system combines **fuzzy search, embeddings, BM25, vector search, and optional LLM enhancements (GPT)** to achieve high recall & precision.

---

## 🔹 Features

✅ **Hybrid Matching** — Fuzzy matching, trigram index, embeddings, and similarity scoring.
✅ **Semantic Search** — BM25 + embeddings for lexical & semantic relevance.
✅ **LLM Enhancements** — GPT for name extraction, query expansion, and reranking.
✅ **Optional Vector DB** — Scalable search with **Qdrant ANN indexing**.
✅ **FastAPI Server** — REST API with Swagger docs.
✅ **Production-Ready Roadmap** — Scalability, observability, and compliance recommendations.

---

## 🔹 Architecture

```
            ┌───────────────────────────┐
            │     Transaction Data       │
            └──────────────┬────────────┘
                           │
                 ┌─────────▼─────────┐
                 │  BM25 / Embeddings │
                 └─────────┬─────────┘
                           │
              ┌────────────▼────────────┐
              │     Qdrant Vector DB     │
              └────────────┬────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │   FastAPI Service (PayMatch API)   │
         └─────────────────┬─────────────────┘
                           │
    ┌───────────────┬───────────────┬───────────────┐
    │ Fuzzy Match   │ Embeddings     │ LLM Enhancer  │
    │ (names)       │ (semantic sim) │ (GPT parsing) │
    └───────────────┴───────────────┴───────────────┘
```

---

## 🔹 Setup Instructions

### 1. Clone & install dependencies

```bash
git clone <repo_url>
cd paymatch
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

💡 For minimal setup (no GPT/Qdrant):

```bash
pip install -r requirements-minimal.txt
```

---

### 2. Add `.env` file

Create a `.env` file in the project root:

```ini
# Required (for LLM features)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_CHAT_MODEL=gpt-4o-mini

# Optional configs
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

> 🔒 **Never commit `.env`**. Use a secrets manager in production.

---

### 3. Run the server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API Docs → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔹 Example Usage

### Task 1 — User–Transaction Matching

```bash
curl "http://127.0.0.1:8000/match_transaction/caqjJtrI?top_k=5&use_llm_for_parse=true"
```

Response:

```json
{
  "users": [
    {"id": "U4NNQUQIeE", "match_metric": 89.25},
    {"id": "U7XJLPMqf1", "match_metric": 72.14}
  ],
  "total_number_of_matches": 2
}
```

---

### Task 2 — Semantic Transaction Search

```bash
curl "http://127.0.0.1:8000/search_transactions/?query=consulting%20fee&top_k=5&use_llm_for_expansion=true&use_llm_for_rerank=true"
```

Response:

```json
{
  "transactions": [
    {"id": "TXN123", "embedding": 0.8123},
    {"id": "TXN456", "embedding": 0.8011},
    {"id": "TXN789", "embedding": 0.7955}
  ]
}
```

---

## 🔹 Task Discussions

### Task 1 — User–Transaction Matching

* **Signals Used**: Fuzzy match, trigram, embeddings, description-to-user similarity.
* **LLM Role**: GPT extracts names from messy descriptions.
* **Limitations**:

  * Without GPT → struggles with nicknames, noisy text.
  * With GPT → adds latency, cost, non-determinism.

---

### Task 2 — Semantic Transaction Search

* **Pipeline**: BM25 → Embeddings → (optional) Qdrant ANN → (optional) LLM Expansion/Rerank.
* **LLM Role**: Improves recall (expansion) & precision (rerank).
* **Limitations**:

  * Without GPT → misses synonyms, paraphrases.
  * With GPT → latency + cost overhead.

---

## 🔹 Roadmap for Production

🔧 **System Improvements**

* Move to managed vector DBs (Qdrant/Pinecone).
* Use caching + local LLMs for cost control.
* Add observability (Prometheus, Grafana, OpenTelemetry).
* Train supervised ML matcher + feedback loop.

🚀 **Deployment Strategy**

* Containerize with Docker & orchestrate with Kubernetes.
* CI/CD via GitHub Actions.
* Secrets via Vault / AWS Secrets Manager.

🔐 **Compliance & Security**

* Encrypt PII at rest & in transit.
* Sanitize logs before sending to LLM.
* Add audit trails for financial compliance.

---

## 🔹 Tech Stack

* **Backend**: FastAPI + Uvicorn
* **Search**: BM25, Embeddings, Qdrant (optional)
* **LLM**: OpenAI GPT (for parsing, expansion, reranking)
* **Infra (future-ready)**: Docker, Kubernetes, Prometheus, Vault

---

## 🔹 License

MIT License © 2025 — Open to contributions 🚀

---
