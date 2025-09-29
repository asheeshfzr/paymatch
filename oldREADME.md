# 🔹 Project Overview

PayMatch: Intelligent User–Transaction Mapping & Semantic Search
This project implements two main tasks:

1. **Task 1 — Match Users to Transactions**
   Given a transaction ID, return the most likely users who match the payment.

   * Hybrid approach: fuzzy matching, trigram index, embeddings, and description-to-user similarity.
   * Enhanced by an **LLM (OpenAI GPT)** for better name extraction from noisy descriptions.

2. **Task 2 — Semantic Transaction Search**
   Given a free-text query, return the most semantically similar transactions.

   * Uses **BM25** for fast candidate retrieval, **embeddings** for semantic similarity, and a **Qdrant vector DB** for large-scale ANN search.
   * Enhanced by an **LLM (OpenAI GPT)** for query expansion (improves recall) and reranking (improves precision).

---

## 🔹 Setup Instructions

### 1. Clone & install dependencies

```bash
git clone <repo_url>
cd project
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

> 💡 For a lightweight setup, you can install only `requirements-minimal.txt`:
>
> ```bash
> pip install -r requirements-minimal.txt
> ```

---

### 2. Add `.env` file

Create a file called **`.env`** in the project root (same level as `requirements.txt`):

```ini
# Required if you want LLM features
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_CHAT_MODEL=gpt-4o-mini

# Optional configs
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

> 🔒 **Do not commit `.env`** to git. Keep it local or use a secrets manager in production.

---

### 3. Run the server

Start the FastAPI app:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

You can now access interactive API docs at:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔹 Example Usage

### Task 1: Match Users to Transaction

**Request (with LLM parsing):**

```bash
curl "http://127.0.0.1:8000/match_transaction/caqjJtrI?top_k=5&min_score=30&use_llm_for_parse=true"
```

**Response:**

```json
{
  "users": [
    {"id": "U4NNQUQIeE", "match_metric": 89.25},
    {"id": "U7XJLPMqf1", "match_metric": 72.14}
  ],
  "total_number_of_matches": 2
}
```

* `use_llm_for_parse=true` → Uses GPT to extract names from transaction description.
* If omitted, it falls back to heuristics.

---

### Task 2: Search Transactions

**Baseline search:**

```bash
curl "http://127.0.0.1:8000/search_transactions/?query=consulting%20fee&top_k=5"
```

**With LLM enhancements (expansion + rerank):**

```bash
curl "http://127.0.0.1:8000/search_transactions/?query=consulting%20fee&top_k=5&use_llm_for_expansion=true&use_llm_for_rerank=true"
```

**Response:**

```json
{
  "transactions": [
    {"id": "TXN123", "embedding": 0.8123},
    {"id": "TXN456", "embedding": 0.8011},
    {"id": "TXN789", "embedding": 0.7955}
  ],
  "total_number_of_tokens_used": 9
}
```

* `use_llm_for_expansion=true` → GPT generates query paraphrases for better recall.
* `use_llm_for_rerank=true` → GPT reranks the top candidates for better precision.

---

🔹 Task Discussions
Task 1: User–Transaction Matching
Solution Approach:
	• Combines multiple signals to make the match robust:
		○ Fuzzy Matching → Handles typos and minor variations in names.
		○ Trigram Index → Speeds up approximate string comparisons.
		○ Embeddings → Captures semantic similarity between transaction descriptions and user names.
		○ Description-to-User Similarity → Embedding-based scoring of the transaction text against all user profiles.
	• LLM Enhancement: When enabled, GPT parses transaction descriptions to extract possible user names (e.g., from “Payment received from J. Doe for invoice #123”).
Limitations:
	• Without LLM:
		○ Nicknames and abbreviations may not match well (e.g., “Jon” vs “Jonathan”).
		○ Very noisy transaction descriptions can defeat fuzzy/embedding matching.
	• With LLM:
		○ Latency: Adds ~200–1000ms due to API calls.
		○ Cost: Increases per-request cost depending on token usage.
		○ Non-determinism: Same query may yield slightly different parsing results.
	• Hybrid fallback ensures the system is never fully dependent on GPT, but reliability may still vary on edge cases.

Task 2: Semantic Transaction Search
Solution Approach:
	• BM25: Lexical search to retrieve top candidate transactions quickly.
	• Embeddings: Re-rank those candidates by semantic closeness.
	• Qdrant (optional): Enables large-scale Approximate Nearest Neighbor (ANN) search for efficiency.
	• LLM Enhancements:
		○ Query Expansion → GPT generates paraphrases (“consulting fee” → “advisory services payment”), boosting recall.
		○ Reranking → GPT refines the ranking, improving precision for subtle matches.
Limitations:
	• Without LLM:
		○ Misses queries phrased differently from transaction text (synonyms, paraphrases).
		○ BM25 alone struggles with out-of-vocabulary words.
	• With LLM:
		○ Latency + Cost: Multiple GPT calls (expansion + rerank) make it heavier than baseline search.
		○ Non-determinism: Expanded queries may include irrelevant paraphrases.
		○ Scaling: Running LLM-powered reranking on thousands of results can be expensive.
	• Production systems should use LLM selectively, cache expansions, and integrate observability for token usage.


🔹 Task 3: Taking This Proof of Concept to Production
If given additional resources, here’s how this system could be hardened and scaled for real-world production use:
🔧 System Improvements
	• Scalability:
		○ Move from BM25 + local embeddings to fully managed vector DB (Qdrant, Pinecone, Weaviate) for fast, distributed semantic search.
		○ Introduce sharding & replication for handling millions of users and transactions.
	• LLM Usage Optimization:
		○ Apply LLM selectively — only on low-confidence matches or ambiguous queries, reducing cost and latency.
		○ Cache common query expansions and parsed transaction names to avoid repeated GPT calls.
		○ Experiment with smaller local LLMs (e.g., Llama 3, Mistral) for cost control, while keeping GPT for high-value tasks.
	• Observability & Monitoring:
		○ Integrate with Prometheus + Grafana to track:
			§ Matching accuracy (precision/recall)
			§ Token consumption and latency from LLM calls
			§ System throughput & error rates
		○ Add tracing (OpenTelemetry) for request pipelines.
	• Robustness:
		○ Train a supervised ML model on historical transaction–user matches to replace or augment heuristics.
		○ Add active learning loop: collect feedback when predictions are wrong and use it to retrain models.
		○ Build a fallback strategy (e.g., embeddings-only) if LLM services are unavailable.
	• Deployment & Ops:
		○ Containerize with Docker and orchestrate with Kubernetes for scaling across environments.
		○ Use CI/CD pipelines (GitHub Actions, GitLab CI) to automate tests, builds, and deployments.
		○ Secure credentials with Vault / AWS Secrets Manager instead of .env files.
	• Compliance & Security:
		○ Encrypt sensitive transaction/user data at rest and in transit.
		○ Ensure logs are sanitized (no PII leakage to LLM prompts).
		○ Add audit logs for regulatory compliance (especially if used in finance).
🚀 Roadmap for Production
	1. MVP (Current POC) → FastAPI app with hybrid matching & search, optional GPT calls.
	2. Pilot Deployment → Add Qdrant + monitoring, use LLM selectively with caching.
	3. Full Production → ML-trained matcher, containerized deployment on Kubernetes, scalable vector DB, full observability, security hardening.


