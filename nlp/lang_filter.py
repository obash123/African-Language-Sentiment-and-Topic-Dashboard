"""Tags free-text comments with one of the five target language codes.

langdetect (a Python port of Google's language-detection library) covers
Hausa, Yoruba, Igbo, and Swahili reasonably well from character n-grams, but
it doesn't have a Nigerian Pidgin class at all — pidgin gets misclassified as
English. We patch that with a keyword/stopword heuristic that runs before
falling back to langdetect.

Codes follow the AfriSenti convention: hau, yor, ibo, pcm, swa.
"""
from __future__ import annotations

import re

from langdetect import DetectorFactory, detect_langs, LangDetectException

DetectorFactory.seed = 0  # deterministic output

TARGET_LANGS = {"hau", "yor", "ibo", "pcm", "swa"}

_LANGDETECT_TO_TARGET = {
    "ha": "hau",
    "yo": "yor",
    "ig": "ibo",
    "sw": "swa",
}

# Nigerian Pidgin markers: words/spellings that rarely appear in standard
# English but are common in naija pidgin comments.
_PCM_MARKERS = re.compile(
    r"\b("
    r"wetin|abeg|dey|una|wahala|sabi|dem|no\s+dey|na\s+so|no\s+be|"
    r"gbege|shey|oga|sha|jare|abi|wey|comot|waka|sef|kudi|palava"
    r")\b",
    re.IGNORECASE,
)

# Diacritic / character cues that boost confidence for the three Nigerian
# tonal languages when langdetect's probability is low or wrong.
_YORUBA_CHARS = re.compile(r"[ẹọṣĐŒŚåẸỌṢ]|(?:\bọ|\bẹ)")
_IGBO_CHARS = re.compile(r"[ịọụṅ]|(?:\bịbọ|\bnwa)")
_HAUSA_CHARS = re.compile(r"[ɓɗƙƴ]")


def _pidgin_score(text: str) -> int:
    return len(_PCM_MARKERS.findall(text))


def detect_language(text: str) -> str | None:
    """Returns one of TARGET_LANGS, or None if the text doesn't look like
    any of them confidently enough to keep."""
    stripped = text.strip()
    if len(stripped) < 3:
        return None

    if _pidgin_score(stripped) >= 1:
        return "pcm"
    if _YORUBA_CHARS.search(stripped):
        return "yor"
    if _IGBO_CHARS.search(stripped):
        return "ibo"
    if _HAUSA_CHARS.search(stripped):
        return "hau"

    try:
        candidates = detect_langs(stripped)
    except LangDetectException:
        return None

    for candidate in candidates:
        mapped = _LANGDETECT_TO_TARGET.get(candidate.lang)
        if mapped and candidate.prob >= 0.5:
            return mapped
    return None


def tag_dataframe(df, text_col: str = "text", lang_col: str = "language"):
    """Adds a language column in place-style (returns a new DataFrame),
    dropping rows that didn't match any target language."""
    df = df.copy()
    df[lang_col] = df[text_col].astype(str).map(detect_language)
    return df[df[lang_col].isin(TARGET_LANGS)].reset_index(drop=True)
