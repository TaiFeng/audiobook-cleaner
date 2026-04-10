"""
Merge overlapping flagged time ranges into consolidated cut regions.

Public API
----------
merge_ranges(ranges, padding) -> list[FlaggedRange]
build_ranges_from_results(results, threshold, padding) -> list[FlaggedRange]
"""

from __future__ import annotations

import logging
from typing import List

from .config import ThresholdConfig, severity_gte
from .profanity import FlaggedRange
from .classifier import ChunkResult

logger = logging.getLogger(__name__)


def merge_ranges(
    ranges: List[FlaggedRange],
    padding_seconds: float = 0.0,
) -> List[FlaggedRange]:
    """
    Sort ranges by start time, add *padding_seconds*, then merge overlaps.

    Returns a new list of non-overlapping FlaggedRange objects.
    """
    if not ranges:
        return []

    # Apply padding and clamp to zero
    padded = []
    for r in ranges:
        padded.append(FlaggedRange(
            start=max(0.0, r.start - padding_seconds),
            end=r.end + padding_seconds,
            reason=r.reason,
            source=r.source,
            severity=r.severity,
            confidence=r.confidence,
        ))

    padded.sort(key=lambda r: r.start)

    merged: List[FlaggedRange] = [padded[0]]
    for current in padded[1:]:
        prev = merged[-1]
        if current.start <= prev.end:
            # Overlapping — extend and combine reasons
            reasons = set(prev.reason.split(" | ")) | set(current.reason.split(" | "))
            merged[-1] = FlaggedRange(
                start=prev.start,
                end=max(prev.end, current.end),
                reason=" | ".join(sorted(reasons)),
                source=f"{prev.source}+{current.source}" if prev.source != current.source else prev.source,
                severity=max(prev.severity, current.severity, key=lambda s: {"none": 0, "mild": 1, "moderate": 2, "severe": 3}.get(s, 0)),
                confidence=max(prev.confidence, current.confidence),
            )
        else:
            merged.append(current)

    logger.info(
        "Merged %d ranges → %d non-overlapping regions (padding=%.1fs).",
        len(ranges), len(merged), padding_seconds,
    )
    return merged


def build_ranges_from_results(
    results: List[ChunkResult],
    threshold: ThresholdConfig,
    padding_seconds: float = 0.0,
) -> List[FlaggedRange]:
    """
    Convert ChunkResults into FlaggedRanges, applying threshold filters.

    Only chunks whose severity >= threshold.min_severity AND whose
    confidence >= threshold.min_confidence are included.
    """
    ranges: List[FlaggedRange] = []
    for r in results:
        if not r.is_flagged:
            continue
        if r.confidence < threshold.min_confidence:
            logger.debug(
                "Chunk %d skipped — confidence %.2f < threshold %.2f",
                r.chunk_index, r.confidence, threshold.min_confidence,
            )
            continue
        if not severity_gte(r.severity, threshold.min_severity):
            logger.debug(
                "Chunk %d skipped — severity '%s' < threshold '%s'",
                r.chunk_index, r.severity, threshold.min_severity,
            )
            continue

        ranges.append(FlaggedRange(
            start=r.start_time,
            end=r.end_time,
            reason=r.reason,
            source="classifier",
            severity=r.severity,
            confidence=r.confidence,
        ))

    logger.info(
        "%d/%d classified chunks pass threshold (confidence>=%.2f, severity>=%s).",
        len(ranges), len(results), threshold.min_confidence, threshold.min_severity,
    )
    return ranges
