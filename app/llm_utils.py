# app/llm_utils.py
"""
Optional LLM helpers using OpenAI.
Uses the new OpenAI v1.x client (from openai import OpenAI).
These functions are defensive: they only attempt to call OpenAI if OPENAI_API_KEY is present.
If any error occurs, they return safe fallbacks.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_ENABLED = bool(OPENAI_API_KEY)
if LLM_ENABLED:
    logger.info("LLM support enabled (OPENAI_API_KEY present)")
else:
    logger.info("LLM support disabled (no OPENAI_API_KEY)")

client = None
if LLM_ENABLED:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None
        LLM_ENABLED = False
        logger.warning("Failed to initialize OpenAI client; disabling LLM features")


def _safe_chat_completion(prompt: str, max_tokens: int = 128, temperature: float = 0.0) -> str:
    """
    Ask OpenAI ChatCompletion. Returns the assistant text or empty string on any failure.
    """
    if not LLM_ENABLED or client is None:
        logger.info("Skipping LLM call: LLM not enabled or client unavailable")
        return ""
    try:
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        logger.info("Calling OpenAI ChatCompletion (model=%s)", model)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if resp and resp.choices:
            text = resp.choices[0].message.content.strip()
            logger.info("LLM call succeeded")
            return text
        logger.info("LLM call returned no choices; treating as empty response")
        return ""
    except Exception:
        logger.exception("LLM call failed with exception")
        return ""


def llm_extract_name(description: str) -> str:
    """
    Extract payer name from a transaction description using an LLM.
    Returns empty string on failure (so caller can fallback to heuristics).
    """
    if not LLM_ENABLED or not description:
        if not LLM_ENABLED:
            logger.info("llm_extract_name skipped: LLM disabled")
        return ""

    prompt = (
        "Extract the payer's full name from the following bank transaction description. "
        "Return the name only, without extra explanation or punctuation.\n\n"
        f"Description:\n\"{description}\"\n\nName:"
    )
    out = _safe_chat_completion(prompt, max_tokens=32, temperature=0.0)
    if not out:
        logger.info("llm_extract_name received empty result from LLM")
        return ""
    out = out.strip().splitlines()[0].strip()

    import re
    out = re.sub(r'[^A-Za-z\s]', ' ', out)
    out = re.sub(r'\s+', ' ', out).strip()
    logger.info("llm_extract_name extracted name via LLM: '%s'", out)
    return out


def llm_expand_query(query: str) -> List[str]:
    """
    Use LLM to produce paraphrases/variations of a search query.
    Returns a list with the original query first, and up to 3 paraphrases after it.
    Falls back to [query] on failure.
    """
    if not LLM_ENABLED or not query:
        if not LLM_ENABLED:
            logger.info("llm_expand_query skipped: LLM disabled")
        return [query]
    prompt = (
        "Generate up to 3 short paraphrases of the following financial/search query. "
        "Each paraphrase should be on its own line. Do NOT include any numbering or extra text.\n\n"
        f"Query: \"{query}\"\n\nParaphrases:"
    )
    out = _safe_chat_completion(prompt, max_tokens=120, temperature=0.2)
    if not out:
        logger.info("llm_expand_query received empty result from LLM; using original query")
        return [query]
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    results = [query]
    for l in lines:
        if l and l not in results:
            results.append(l)
            if len(results) >= 4:
                break
    logger.info("llm_expand_query produced %d queries via LLM", len(results))
    return results


def llm_rerank(query: str, candidates: List[str]) -> List[int]:
    """
    Ask the LLM to rerank candidate texts (best-to-worst) for semantic similarity to query.
    Returns the indices of candidates in the new order. On failure returns the original order [0..n-1].
    """
    if not LLM_ENABLED or not candidates:
        if not LLM_ENABLED:
            logger.info("llm_rerank skipped: LLM disabled")
        return list(range(len(candidates)))

    joined = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    prompt = (
        "Rank the following candidate transaction descriptions by how semantically similar they are to the query.\n"
        "Return only the ordered list of indices (1-based), separated by commas (e.g. 3,1,2).\n\n"
        f"Query: \"{query}\"\n\nCandidates:\n{joined}\n\nRank:"
    )
    out = _safe_chat_completion(prompt, max_tokens=256, temperature=0.0)
    if not out:
        logger.info("llm_rerank received empty result from LLM; using original order")
        return list(range(len(candidates)))

    import re
    nums = re.findall(r'\d+', out)
    order = []
    for n in nums:
        try:
            idx = int(n) - 1
            if 0 <= idx < len(candidates) and idx not in order:
                order.append(idx)
        except Exception:
            continue

    if not order:
        logger.info("llm_rerank parsed no indices; using original order")
        return list(range(len(candidates)))

    for i in range(len(candidates)):
        if i not in order:
            order.append(i)

    logger.info("llm_rerank produced an LLM-guided order for %d candidates", len(candidates))
    return order
