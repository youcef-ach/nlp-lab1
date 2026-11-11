"""NLP Text Preprocessing & Entity Normalization Pipeline.

Academic Coursework: Natural Language Processing (M2 IASD, ESI-SBA)
Dataset: Spooky Author Identification (Kaggle)
"""

import re
import string
from typing import Dict, List, Optional


HOMOGLYPH_MAP: Dict[str, str] = {
    "$": "s",
    "0": "o",
    "1": "l",
    "3": "e",
    "@": "a",
    "!": "i",
}


def normalize_entities(text: str) -> str:
    """Normalize URLs and emails to canonical semantic tokens."""
    if not text:
        return ""
    text = re.sub(r"http\S+|www\.\S+", "<url>", text)
    text = re.sub(r"\S+@\S+", "<email>", text)
    return text


def replace_homoglyphs(text: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    """Replace visual homoglyphs and leetspeak characters with canonical ASCII."""
    mapping = custom_map if custom_map is not None else HOMOGLYPH_MAP
    return "".join(mapping.get(ch, ch) for ch in text)


def clean_and_tokenize_words(text: str) -> List[str]:
    """Normalize text, replace homoglyphs, and split into sanitized alphanumeric tokens."""
    normalized = normalize_entities(text)
    dehomoglyphed = replace_homoglyphs(normalized)
    # Simple regex word boundary tokenization
    tokens = re.findall(r"\b\w+\b|<url>|<email>", dehomoglyphed)
    return tokens


def filter_lexicon(tokens: List[str], lexicon: set) -> List[str]:
    """Filter tokens matching a specific lexicon (e.g. sentiment opinion lexicon)."""
    return [t.lower() for t in tokens if t.lower() in lexicon]
