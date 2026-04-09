"""
Word-level profanity detection against a configurable banned-word list.

Public API
----------
load_banned_words(path) -> set[str]
detect_profanity(words, banned, padding) -> list[FlaggedRange]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from .transcriber import WordSegment

logger = logging.getLogger(__name__)


@dataclass
class FlaggedRange:
    """A time range that should be muted or removed."""
    start: float
    end: float
    reason: str
    source: str          # "profanity" | "classifier"
    severity: str = "severe"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Banned word loading
# ---------------------------------------------------------------------------

def load_banned_words(path: str | Path) -> Set[str]:
    """
    Read a banned-word file (one word/phrase per line, ``#`` comments).

    Returns a set of lowercase strings.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Banned-word file not found: %s — profanity detection disabled.", path)
        return set()

    words: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line.lower())

    logger.info("Loaded %d banned words/phrases from %s", len(words), path)
    return words


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_STRIP_RE = re.compile(r"[^a-zA-Z0-9']")


def _normalize(word: str) -> str:
    return _STRIP_RE.sub("", word).lower()


def detect_profanity(
    words: List[WordSegment],
    banned: Set[str],
    padding_seconds: float = 0.3,
) -> List[FlaggedRange]:
    """
    Scan *words* for banned terms and return padded FlaggedRanges.

    Supports single-word and two-word phrase matching.
    """
    if not banned:
        return []

    hits: List[FlaggedRange] = []

    for i, ws in enumerate(words):
        token = _normalize(ws.word)
        if not token:
            continue

        # Single-word match
        if token in banned:
            hits.append(FlaggedRange(
                start=max(0.0, ws.start - padding_seconds),
                end=ws.end + padding_seconds,
                reason=f"Banned word: '{ws.word}'",
                source="profanity",
            ))
            continue

        # Two-word phrase match (e.g., "son of")
        if i + 1 < len(words):
            phrase = f"{token} {_normalize(words[i + 1].word)}"
            if phrase in banned:
                hits.append(FlaggedRange(
                    start=max(0.0, ws.start - padding_seconds),
                    end=words[i + 1].end + padding_seconds,
                    reason=f"Banned phrase: '{ws.word} {words[i + 1].word}'",
                    source="profanity",
                ))

    logger.info("Profanity scan: %d hits found.", len(hits))
    return hits
