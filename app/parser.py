# app/parser.py
"""
Parser for extracting candidate names from transaction descriptions.
Supports optional LLM extraction when OPENAI_API_KEY is present.
"""

import re
import logging
from typing import List
from app.llm_utils import llm_extract_name, LLM_ENABLED as LLM_AVAILABLE

logger = logging.getLogger(__name__)


def extract_candidate_names(description: str, max_candidates: int = 3, use_llm: bool = False) -> List[str]:
    """
    Extract candidate payer names from a transaction description with comprehensive error handling.
    If use_llm=True and an OPENAI_API_KEY is configured, the function will try LLM extraction first.
    Otherwise, it falls back to heuristics (regex + capitalized sequences + fallback).
    """
    try:
        if not description:
            return []

        if max_candidates <= 0:
            max_candidates = 3

        desc = str(description).strip()
        if not desc:
            return []

        # Try LLM extraction if requested and available
        if use_llm and LLM_AVAILABLE:
            logger.info("Parser: use_llm requested and available; attempting LLM name extraction")
            try:
                name = llm_extract_name(desc)
                if name:
                    return [name]
            except Exception as e:
                # on any failure, fall back to heuristics
                logger.exception(f"Parser: LLM extraction raised; falling back to heuristics: {e}")
        elif use_llm and not LLM_AVAILABLE:
            logger.info("Parser: use_llm requested but LLM not available; skipping LLM")
        else:
            logger.info("Parser: use_llm not requested; using heuristics")

        # Heuristic patterns
        patterns = [
            r'from\s+(.+?)(?:\s+for\b|\s+ref\b|\s+invoice\b|\s*$)',
            r'payment from\s+(.+?)(?:\s+for\b|\s+ref\b|\s+invoice\b|\s*$)',
            r'to\s+(.+?)(?:\s+for\b|\s+ref\b|\s+invoice\b|\s*$)'
        ]
        candidates: List[str] = []
        
        try:
            for pat in patterns:
                try:
                    m = re.search(pat, desc, flags=re.IGNORECASE)
                    if m:
                        name_raw = m.group(1).strip()
                        name_clean = re.sub(r'[^A-Za-z\s]', ' ', name_raw)
                        name_clean = re.sub(r'\s+', ' ', name_clean).strip()
                        if name_clean:
                            candidates.append(name_clean)
                            if len(candidates) >= max_candidates:
                                return candidates
                except Exception as e:
                    logger.warning(f"Parser: Pattern matching failed for pattern '{pat}': {e}")
                    continue

            # Capitalized sequences (2-4 words)
            try:
                cap_seq = re.findall(r'(?:\b[A-Z][a-z]+\b(?:\s+|$)){2,4}', desc)
                for seq in cap_seq:
                    seq_clean = re.sub(r'[^A-Za-z\s]', ' ', seq).strip()
                    if seq_clean and seq_clean not in candidates:
                        candidates.append(seq_clean)
                        if len(candidates) >= max_candidates:
                            return candidates
            except Exception as e:
                logger.warning(f"Parser: Capitalized sequence extraction failed: {e}")

            # Fallback: first two words
            try:
                words = re.sub(r'[^A-Za-z\s]', ' ', desc).split()
                if len(words) >= 2:
                    fallback = " ".join(words[:2])
                    if fallback not in candidates:
                        candidates.append(fallback)
            except Exception as e:
                logger.warning(f"Parser: Fallback word extraction failed: {e}")

            return candidates
            
        except Exception as e:
            logger.error(f"Parser: Heuristic extraction failed: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Parser: extract_candidate_names failed: {e}")
        return []
