"""
Split a word-level transcript into overlapping chunks for AI classification.

Public API
----------
create_chunks(words, config) -> list[Chunk]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from .config import ChunkingConfig
from .transcriber import WordSegment

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A window of transcript text with timestamp boundaries."""
    index: int
    text: str
    word_count: int
    start_time: float
    end_time: float
    words: List[WordSegment] = field(repr=False, default_factory=list)


def create_chunks(
    words: List[WordSegment],
    config: ChunkingConfig,
) -> List[Chunk]:
    """
    Slide a window over *words* to produce overlapping Chunks.

    Parameters
    ----------
    words : list[WordSegment]
        Flat, time-sorted word list from the transcriber.
    config : ChunkingConfig
        chunk_size, overlap, min/max constraints.

    Returns
    -------
    list[Chunk]
    """
    if not words:
        return []

    step = max(1, config.chunk_size - config.overlap)
    chunks: List[Chunk] = []
    total = len(words)
    idx = 0
    chunk_num = 0

    while idx < total:
        end_idx = min(idx + config.chunk_size, total)
        window = words[idx:end_idx]

        # Skip if the remaining window is too small (unless it's the only chunk)
        if len(window) < config.min_chunk_size and chunks:
            # Merge remainder into previous chunk
            prev = chunks[-1]
            combined_words = prev.words + window
            chunks[-1] = Chunk(
                index=prev.index,
                text=" ".join(w.word for w in combined_words),
                word_count=len(combined_words),
                start_time=combined_words[0].start,
                end_time=combined_words[-1].end,
                words=combined_words,
            )
            break

        chunk = Chunk(
            index=chunk_num,
            text=" ".join(w.word for w in window),
            word_count=len(window),
            start_time=window[0].start,
            end_time=window[-1].end,
            words=window,
        )
        chunks.append(chunk)
        chunk_num += 1
        idx += step

    logger.info(
        "Created %d chunks (target=%d words, overlap=%d) from %d words.",
        len(chunks), config.chunk_size, config.overlap, total,
    )
    return chunks
