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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests

from .config import ClassificationConfig
from .chunker import Chunk
from .transcriber import WordSegment

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
    contains_blasphemy: bool = False
    severity: str = "none"      # none | mild | moderate | severe
    confidence: float = 0.0
    reason: str = ""
    word_count: int = 0
    text_preview: str = ""      # first 120 chars for the review report
    segment_start: Optional[float] = None  # tighter start within chunk
    segment_end: Optional[float] = None    # tighter end within chunk

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_flagged(self) -> bool:
        return any([
            self.contains_explicit_sex,
            self.contains_graphic_violence,
            self.contains_drug_content,
            self.contains_mature_themes,
            self.contains_blasphemy,
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

5. **Blasphemy** (`contains_blasphemy`)
   Religious names, deities, or sacred concepts used as expletives, curses,
   or in a contemptuous or mocking manner.  Do NOT flag respectful religious
   discussion, prayer, scripture quotation, or narrative use of religious
   figures and events.

## Severity Scale
- **none** — No concerning content.
- **mild** — Borderline; might concern very protective parents.
- **moderate** — Clearly inappropriate for young children.
- **severe** — Extremely explicit or disturbing content.

## Confidence
Rate your confidence from 0.0 (guessing) to 1.0 (certain).

## Segment Timestamps
Return `segment_start` and `segment_end` — the tightest possible timestamps
(in seconds, floats) identifying where within the chunk the objectionable
content begins and ends.  These must be within the chunk's start/end span.
Return null for both if the entire chunk is objectionable or if precise
boundaries cannot be determined.
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
  "contains_blasphemy": <bool>,
  "severity": "<none|mild|moderate|severe>",
  "confidence": <float 0.0–1.0>,
  "reason": "<brief explanation of flags, or 'No concerning content detected'>",
  "segment_start": <float or null>,
  "segment_end": <float or null>
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

    system_prompt = SYSTEM_PROMPT
    user_template = USER_PROMPT_TEMPLATE
    if not config.screen_blasphemy:
        # Strip blasphemy from the prompt schema so the model never returns it
        system_prompt = "\n".join(
            line for line in system_prompt.splitlines()
            if "contains_blasphemy" not in line and "Blasphemy" not in line
        )
        user_template = "\n".join(
            line for line in user_template.splitlines()
            if "contains_blasphemy" not in line
        )

    user_msg = user_template.format(
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
            {"role": "system", "content": system_prompt},
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

            # Validate segment_start / segment_end
            seg_start = parsed.get("segment_start")
            seg_end = parsed.get("segment_end")
            if seg_start is not None and seg_end is not None:
                try:
                    seg_start = float(seg_start)
                    seg_end = float(seg_end)
                except (TypeError, ValueError):
                    seg_start = None
                    seg_end = None
                else:
                    if seg_start < chunk.start_time or seg_end > chunk.end_time or seg_start >= seg_end:
                        seg_start = None
                        seg_end = None
            else:
                seg_start = None
                seg_end = None

            return ChunkResult(
                chunk_index=chunk.index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                contains_explicit_sex=bool(parsed.get("contains_explicit_sex", False)),
                contains_graphic_violence=bool(parsed.get("contains_graphic_violence", False)),
                contains_drug_content=bool(parsed.get("contains_drug_content", False)),
                contains_mature_themes=bool(parsed.get("contains_mature_themes", False)),
                contains_blasphemy=bool(parsed.get("contains_blasphemy", False)) if config.screen_blasphemy else False,
                severity=parsed.get("severity", "none"),
                confidence=float(parsed.get("confidence", 0.0)),
                reason=parsed.get("reason", ""),
                word_count=chunk.word_count,
                text_preview=chunk.text[:120],
                segment_start=seg_start,
                segment_end=seg_end,
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
    When bisection is enabled, flagged chunks are recursively bisected to
    isolate the objectionable sub-chunk.
    """
    if not chunks:
        return []

    logger.info(
        "Classifying %d chunks (model=%s, concurrency=%d) …",
        len(chunks), config.model, config.max_concurrent,
    )

    initial_results: List[ChunkResult] = []

    with ThreadPoolExecutor(max_workers=config.max_concurrent) as pool:
        futures = {
            pool.submit(_call_api, chunk, config, sensitivity): chunk.index
            for chunk in chunks
        }
        for future in as_completed(futures):
            result = future.result()
            initial_results.append(result)
            status = "FLAGGED" if result.is_flagged else "clean"
            logger.debug(
                "  chunk %d  [%s]  severity=%s  confidence=%.2f",
                result.chunk_index, status, result.severity, result.confidence,
            )

    initial_results.sort(key=lambda r: r.chunk_index)

    # Bisection drill-down on flagged chunks
    results: List[ChunkResult] = []
    chunks_by_index = {c.index: c for c in chunks}
    for r in initial_results:
        chunk = chunks_by_index.get(r.chunk_index)
        if (
            config.bisect
            and _is_flagged(r)
            and chunk is not None
            and (chunk.end_time - chunk.start_time) > config.bisect_min_seconds
        ):
            logger.info(
                "Bisecting flagged chunk %d [%.2fs–%.2fs] (%.1fs) — drilling down for precise boundaries …",
                chunk.index, chunk.start_time, chunk.end_time, chunk.end_time - chunk.start_time,
            )
            classify_fn = lambda c, _cfg=config, _sens=sensitivity: _call_api(c, _cfg, _sens)
            bisected = _bisect_chunk(chunk, classify_fn, config.bisect_min_seconds, config.bisect_max_depth,
                                     pause_gap_seconds=getattr(config, 'pause_gap_seconds', 1.5))
            logger.info(
                "Bisect chunk %d complete — %d sub-chunk(s) flagged, boundaries: %s",
                chunk.index, len(bisected),
                ", ".join(f"[{b.start_time:.2f}s–{b.end_time:.2f}s]" for b in bisected) or "none",
            )
            results.extend(bisected)
        else:
            results.append(r)

    results.sort(key=lambda r: r.start_time)
    flagged_count = sum(1 for r in results if r.is_flagged)
    logger.info("Classification complete: %d/%d results flagged.", flagged_count, len(results))
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

    # Blasphemy: detect religious names/concepts used as expletives.
    # Narrative context words indicate respectful/story use, not expletives.
    _narrative_ctx = {"said", "replied", "answered", "told", "asked"}
    blasphemy = False
    # Simple phrase matches (unambiguous expletive forms)
    if any(kw in text_lower for kw in [
        "god damn", "goddamn", "holy shit", "holy hell", "holy crap",
        "christ almighty", "jesus f",
        "for christ's sake", "for god's sake",
    ]):
        blasphemy = True
    # "jesus christ" as exclamation — only flag if NOT preceded by narrative context
    if not blasphemy and "jesus christ" in text_lower:
        idx = text_lower.index("jesus christ")
        # Check words immediately before the phrase for narrative context
        before = text_lower[:idx].split()
        if not before or before[-1].rstrip(".,;:!?") not in _narrative_ctx:
            blasphemy = True

    flags = [sex, violence, drugs, mature, blasphemy]
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

    # Blasphemy alone should be at least mild severity with 0.8 confidence
    if blasphemy and not any([sex, violence, drugs, mature]):
        confidence = max(confidence, 0.8)
        if severity == "none":
            severity = "mild"

    reasons = []
    if sex:
        reasons.append("sexual content keywords detected")
    if violence:
        reasons.append("graphic violence keywords detected")
    if drugs:
        reasons.append("drug-related keywords detected")
    if mature:
        reasons.append("mature language / themes detected")
    if blasphemy:
        reasons.append("Blasphemous use of religious name/concept as expletive")

    return ChunkResult(
        chunk_index=chunk.index,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        contains_explicit_sex=sex,
        contains_graphic_violence=violence,
        contains_drug_content=drugs,
        contains_mature_themes=mature,
        contains_blasphemy=blasphemy,
        severity=severity,
        confidence=confidence,
        reason="; ".join(reasons) if reasons else "No concerning content detected",
        word_count=chunk.word_count,
        text_preview=chunk.text[:120],
        segment_start=chunk.start_time if is_flagged else None,
        segment_end=chunk.end_time if is_flagged else None,
    )


# ---------------------------------------------------------------------------
# Bisection helpers
# ---------------------------------------------------------------------------

def _is_flagged(result: ChunkResult, min_confidence: float = 0.3) -> bool:
    """Check if a ChunkResult is flagged (severity != 'none' and confidence meets threshold)."""
    return result.severity != "none" and result.confidence >= min_confidence


def _find_sentence_split(words, pause_gap_seconds=1.5):
    """
    Find the best split point in a word list.
    Prefer a sentence boundary near the midpoint.
    Falls back to midpoint if no sentence boundary found.
    Returns the split index (left = words[:idx], right = words[idx:]).
    """
    mid = len(words) // 2
    # Search outward from midpoint for a sentence boundary
    for offset in range(mid):
        for i in [mid - offset, mid + offset]:
            if 0 < i < len(words):
                w = words[i - 1]
                # Terminal punctuation boundary
                if re.search(r'[.!?]["\']?$', w.word.strip()):
                    return i
                # Long pause boundary
                if i < len(words) and (words[i].start - w.end) >= pause_gap_seconds:
                    return i
    return mid  # fallback to midpoint


def _bisect_chunk(
    chunk: Chunk,
    classify_fn,
    min_seconds: float,
    max_depth: int,
    depth: int = 0,
    pause_gap_seconds: float = 1.5,
) -> List[ChunkResult]:
    """
    Recursively bisect a flagged chunk to find the minimal sub-chunk
    containing the objectionable content.

    Returns a list of ChunkResult for the leaf sub-chunks that are flagged.
    If neither half is flagged, returns the original chunk result (full span)
    as a fallback to avoid losing the detection.
    """
    duration = chunk.end_time - chunk.start_time
    indent = "  " * depth
    logger.debug(
        "%sBisect depth=%d chunk=%d [%.2fs–%.2fs] (%.1fs, %d words)",
        indent, depth, chunk.index, chunk.start_time, chunk.end_time, duration, len(chunk.words),
    )

    if duration <= min_seconds or depth >= max_depth or len(chunk.words) < 2:
        logger.debug("%s  → leaf reached (duration=%.1fs, depth=%d) — classifying.", indent, duration, depth)
        return [classify_fn(chunk)]

    mid = _find_sentence_split(chunk.words, pause_gap_seconds)
    left_words = chunk.words[:mid]
    right_words = chunk.words[mid:]

    left_chunk = Chunk(
        index=chunk.index,
        text=" ".join(w.word for w in left_words),
        word_count=len(left_words),
        start_time=left_words[0].start,
        end_time=left_words[-1].end,
        words=left_words,
    )
    right_chunk = Chunk(
        index=chunk.index,
        text=" ".join(w.word for w in right_words),
        word_count=len(right_words),
        start_time=right_words[0].start,
        end_time=right_words[-1].end,
        words=right_words,
    )

    logger.debug(
        "%s  → split at word %d: left [%.2fs–%.2fs] (%d words), right [%.2fs–%.2fs] (%d words)",
        indent, mid,
        left_chunk.start_time, left_chunk.end_time, left_chunk.word_count,
        right_chunk.start_time, right_chunk.end_time, right_chunk.word_count,
    )

    left_result = classify_fn(left_chunk)
    right_result = classify_fn(right_chunk)

    left_flagged = _is_flagged(left_result)
    right_flagged = _is_flagged(right_result)

    logger.debug(
        "%s  → left=%s (sev=%s conf=%.2f), right=%s (sev=%s conf=%.2f)",
        indent,
        "FLAGGED" if left_flagged else "clean", left_result.severity, left_result.confidence,
        "FLAGGED" if right_flagged else "clean", right_result.severity, right_result.confidence,
    )

    if not left_flagged and not right_flagged:
        logger.info(
            "Bisect chunk=%d depth=%d: both halves clean — falling back to full-span result [%.2fs–%.2fs].",
            chunk.index, depth, chunk.start_time, chunk.end_time,
        )
        return [classify_fn(chunk)]

    results = []
    if left_flagged:
        results.extend(_bisect_chunk(left_chunk, classify_fn, min_seconds, max_depth, depth + 1, pause_gap_seconds))
    if right_flagged:
        results.extend(_bisect_chunk(right_chunk, classify_fn, min_seconds, max_depth, depth + 1, pause_gap_seconds))

    return results
