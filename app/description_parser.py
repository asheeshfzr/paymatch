# app/description_parser.py
import re
import unicodedata
from typing import List

def _clean_text(s: str) -> str:
    if not s:
        return ""
    # normalize unicode and replace non-letters with spaces
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # replace punctuation / digits with space (we assume latin alphabet names)
    s = re.sub(r'[^A-Za-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_candidate_names(description: str, max_candidates: int = 3) -> List[str]:
    """
    Return list of candidate payer names found in description ordered by heuristic confidence.
    Heuristics:
      1) capture after 'from' up to 'for'/'ref'/'invoice' (case-insensitive)
      2) capture after 'to' / 'payment from'
      3) find capitalized word sequences (2-4 words)
      4) fallback: try to take first 2 capitalized tokens
    """
    if not description:
        return []

    desc = description.strip()
    # heuristic 1 and 2 using flexible regex (case-insensitive)
    patterns = [
        r'from\s+(.+?)(?:\s+for\b|\s+ref\b|\s+invoice\b|\s*$)',
        r'payment from\s+(.+?)(?:\s+for\b|\s+ref\b|\s+invoice\b|\s*$)',
        r'to\s+(.+?)(?:\s+for\b|\s+ref\b|\s+invoice\b|\s*$)'
    ]
    candidates = []
    for pat in patterns:
        m = re.search(pat, desc, flags=re.IGNORECASE)
        if m:
            name_raw = m.group(1).strip()
            # clean it but preserve spaces
            name_clean = _clean_text(name_raw)
            if name_clean and name_clean.lower() not in (c.lower() for c in candidates):
                candidates.append(name_clean)
                if len(candidates) >= max_candidates:
                    return candidates

    # heuristic 3: find capitalized sequences (before cleaning) - good when parser misses
    # we consider sequences of 2-4 words where each starts with uppercase letter
    cap_seq = re.findall(r'(?:\b[A-Z][a-z]+\b(?:\s+|$)){2,4}', desc)
    for seq in cap_seq:
        seq_clean = _clean_text(seq)
        if seq_clean and seq_clean.lower() not in (c.lower() for c in candidates):
            candidates.append(seq_clean)
            if len(candidates) >= max_candidates:
                return candidates

    # fallback: try first two words after cleaning
    cleaned = _clean_text(desc)
    if cleaned:
        tokens = cleaned.split()
        if len(tokens) >= 2:
            fallback = " ".join(tokens[:2])
            if fallback.lower() not in (c.lower() for c in candidates):
                candidates.append(fallback)

    return candidates
