"""
AI-powered content classification for transcript chunks.

Works with any OpenAI-compatible API (OpenAI, LM Studio, Ollama, etc.).

Public API
----------
classify_chunks(chunks, config, sensitivity) -> list[ChunkResult]
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests

from .config import ClassificationConfig
from .chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """Classification result for a single transcript chunk."""
    chunk_index: int
    start_time: float
    end_time: float
    contains_explicit_sex: bool = False
    contains_graphic_violence: bool = False
    contains_drug_content: bool = False
    contains_mature_themes: bool = False
    severity: str = "none"      # none | mild | moderate | severe
    confidence: float = 0.0
    reason: str = ""
    word_count: int = 0
    text_preview: str = ""      # first 120 chars for the review report

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_flagged(self) -> bool:
        return any([
            self.contains_explicit_sex,
            self.contains_graphic_violence,
            self.contains_drug_content,
            self.contains_mature_themes,
        ])


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SENSITIVITY_GUIDANCE = {
    "strict": (
        "Flag ALL potentially questionable content.  This includes: metaphorical "
        "or implied violence, romantic tension or innuendo, any alcohol / drug / "
        "tobacco mention even in passing, dark humor, mature vocabulary, and "
        "thematic darkness.  Err heavily on the side of caution."
    ),
    "moderate": (
        "Flag clearly inappropriate content for a child audience.  Passing "
        "references to wine at dinner, mild tension, or cartoon-style action "
        "violence may be acceptable.  Focus on content that would be rated "
        "PG-13 or higher."
    ),
    "minimal": (
        "Flag only content that is explicitly graphic, overtly sexual, or "
        "depicts serious substance abuse.  General action-adventure violence, "
        "mild language, and thematic elements common in young-adult fiction "
        "may pass."
    ),
}

SYSTEM_PROMPT = """\
You are a child-safety content classifier for audiobook text.
Your job is to identify content that is inappropriate for children under 13.

IMPORTANT: Be CONSERVATIVE.  It is far better to flag safe content than to
miss harmful content.  When in doubt, FLAG IT.

## Classification Categories

1. **Explicit Sexual Content** (`contains_explicit_sex`)
   Sexual acts, explicit nudity descriptions, sexual innuendo, seduction
   scenes, or romantic content inappropriate for children.

2. **Graphic Violence** (`contains_graphic_violence`)
   Detailed injury descriptions, gore, torture, graphic death, domestic
   violence, animal cruelty, or war violence with visceral detail.

3. **Drug / Alcohol Content** (`contains_drug_content`)
   Drug use or manufacturing, intoxication scenes, substance abuse,
   detailed alcohol consumption, or smoking depicted positively.

4. **Other Mature Themes** (`contains_mature_themes`)
   Strong profanity, self-harm, suicide, intense psychological horror,
   racial slurs, hate speech, or other content unsuitable for children.

## Severity Scale
- **none** — No concerning content.
- **mild** — Borderline; might concern very protective parents.
- **moderate** — Clearly inappropriate for young children.
- **severe** — Extremely explicit or disturbing content.

## Confidence
Rate your confidence from 0.0 (guessing) to 1.0 (certain).
"""

USER_PROMPT_TEMPLATE = """\
Sensitivity level: {sensitivity}
{sensitivity_guidance}

Respond with ONLY valid JSON — no markdown fencing, no commentary:
{{
  "contains_explicit_sex": <bool>,
  "contains_graphic_violence": <bool>,
  "contains_drug_content": <bool>,
  "contains_mature_themes": <bool>,
  "severity": "<none|mild|moderate|severe>",
  "confidence": <float 0.0–1.0>,
  "reason": "<brief explanation of flags, or 'No concerning content detected'>"
}}

--- BEGIN TRANSCRIPT CHUNK (chunk {chunk_index}) ---
{chunk_text}
--- END TRANSCRIPT CHUNK ---
"""


# ---------------------------------------------------------------------------
# API call with retries
# ---------------------------------------------------------------------------

def _call_api(
    chunk: Chunk,
    config: ClassificationConfig,
    sensitivity: str,
) -> ChunkResult:
    """Send one chunk to the classification API and parse the response."""

    guidance = SENSITIVITY_GUIDANCE.get(sensitivity, SENSITIVITY_GUIDANCE["moderate"])
    user_msg = USER_PROMPT_TEMPLATE.format(
        sensitivity=sensitivity,
        sensitivity_guidance=guidance,
        chunk_index=chunk.index,
        chunk_text=chunk.text,
    )

    headers = {"Content-Type": "application/json"}
    if config.api_key and config.api_key not in ("", "not-needed", "none"):
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }

    url = f"{config.api_base.rstrip('/')}/chat/completions"

    for attempt in range(1, config.retry_attempts + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=config.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"].strip()

            # Strip markdown fencing if the model added it
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(content)

            return ChunkResult(
                chunk_index=chunk.index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                contains_explicit_sex=bool(parsed.get("contains_explicit_sex", False)),
                contains_graphic_violence=bool(parsed.get("contains_graphic_violence", False)),
                contains_drug_content=bool(parsed.get("contains_drug_content", False)),
                contains_mature_themes=bool(parsed.get("contains_mature_themes", False)),
                severity=parsed.get("severity", "none"),
                confidence=float(parsed.get("confidence", 0.0)),
                reason=parsed.get("reason", ""),
                word_count=chunk.word_count,
                text_preview=chunk.text[:120],
            )

        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Chunk %d — attempt %d/%d failed: %s",
                chunk.index, attempt, config.retry_attempts, exc,
            )
            if attempt < config.retry_attempts:
                time.sleep(config.retry_delay * attempt)

    # All retries exhausted — return a conservative "flag everything" result
    logger.error(
        "Chunk %d — all %d attempts failed; marking as flagged (fail-safe).",
        chunk.index, config.retry_attempts,
    )
    return ChunkResult(
        chunk_index=chunk.index,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        contains_mature_themes=True,
        severity="moderate",
        confidence=0.0,
        reason="Classification failed — flagged as precaution.",
        word_count=chunk.word_count,
        text_preview=chunk.text[:120],
    )


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def classify_chunks(
    chunks: List[Chunk],
    config: ClassificationConfig,
    sensitivity: str = "moderate",
) -> List[ChunkResult]:
    """
    Classify all *chunks* concurrently and return sorted ChunkResults.

    Uses ThreadPoolExecutor with *config.max_concurrent* workers.
    """
    if not chunks:
        return []

    logger.info(
        "Classifying %d chunks (model=%s, concurrency=%d) …",
        len(chunks), config.model, config.max_concurrent,
    )

    results: List[ChunkResult] = []

    with ThreadPoolExecutor(max_workers=config.max_concurrent) as pool:
        futures = {
            pool.submit(_call_api, chunk, config, sensitivity): chunk.index
            for chunk in chunks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "FLAGGED" if result.is_flagged else "clean"
            logger.debug(
                "  chunk %d  [%s]  severity=%s  confidence=%.2f",
                result.chunk_index, status, result.severity, result.confidence,
            )

    results.sort(key=lambda r: r.chunk_index)
    flagged_count = sum(1 for r in results if r.is_flagged)
    logger.info("Classification complete: %d/%d chunks flagged.", flagged_count, len(results))
    return results


# ---------------------------------------------------------------------------
# Mock classifier (for dry-run testing)
# ---------------------------------------------------------------------------

def mock_classify_chunk(chunk: Chunk, sensitivity: str = "moderate") -> ChunkResult:
    """
    Rule-based mock classifier for dry-run testing without an API.

    Scans for keyword indicators in the chunk text.
    """
    text_lower = chunk.text.lower()

    sex = any(kw in text_lower for kw in [
        "naked", "undressed", "sexual", "seduced", "moaned", "caressed",
        "thrust", "orgasm", "erotic",
    ])
    violence = any(kw in text_lower for kw in [
        "blood spurted", "entrails", "decapitated", "stabbed repeatedly",
        "skull cracked", "gore", "dismembered", "slaughter",
    ])
    drugs = any(kw in text_lower for kw in [
        "snorted", "injected heroin", "smoked crack", "meth lab",
        "overdose", "cocaine", "drug deal",
    ])
    mature = any(kw in text_lower for kw in [
        "fuck", "shit", "bastard", "bitch", "asshole", "goddamn",
        "suicide", "self-harm", "racial slur",
    ])

    flags = [sex, violence, drugs, mature]
    is_flagged = any(flags)

    if not is_flagged:
        severity = "none"
        confidence = 0.95
    elif sum(flags) >= 3:
        severity = "severe"
        confidence = 0.9
    elif sum(flags) >= 2:
        severity = "moderate"
        confidence = 0.85
    else:
        severity = "mild" if sensitivity == "strict" else "moderate"
        confidence = 0.75

    reasons = []
    if sex:
        reasons.append("sexual content keywords detected")
    if violence:
        reasons.append("graphic violence keywords detected")
    if drugs:
        reasons.append("drug-related keywords detected")
    if mature:
        reasons.append("mature language / themes detected")

    return ChunkResult(
        chunk_index=chunk.index,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        contains_explicit_sex=sex,
        contains_graphic_violence=violence,
        contains_drug_content=drugs,
        contains_mature_themes=mature,
        severity=severity,
        confidence=confidence,
        reason="; ".join(reasons) if reasons else "No concerning content detected",
        word_count=chunk.word_count,
        text_preview=chunk.text[:120],
    )
