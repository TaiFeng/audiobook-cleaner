"""
Split a word-level transcript into overlapping chunks for AI classification.

Public API
----------
create_chunks(words, config) -> list[Chunk]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from .config import ChunkingConfig
from .transcriber import WordSegment

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r'[.!?]["\']?$')


@dataclass
class Chunk:
    """A window of transcript text with timestamp boundaries."""
    index: int = 0
    text: str = ""
    word_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    words: List[WordSegment] = field(repr=False, default_factory=list)


def _create_chunks_fixed(
    words: List[WordSegment],
    config: ChunkingConfig,
) -> List[Chunk]:
    """Original fixed-window chunking strategy."""
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

    return chunks


def _create_chunks_sentence(
    words: List[WordSegment],
    config: ChunkingConfig,
) -> List[Chunk]:
    """Sentence-boundary chunking strategy."""
    chunks: List[Chunk] = []
    buffer: List[WordSegment] = []
    chunk_num = 0

    for i, word in enumerate(words):
        buffer.append(word)

        # Detect sentence boundary
        is_terminal = bool(_SENTENCE_END_RE.search(word.word.strip()))
        has_long_pause = (
            i + 1 < len(words)
            and (words[i + 1].start - word.end) >= config.pause_gap_seconds
        )
        at_max_words = len(buffer) >= config.max_sentence_words

        if is_terminal or has_long_pause or at_max_words:
            chunk = Chunk(
                index=chunk_num,
                text=" ".join(w.word for w in buffer),
                word_count=len(buffer),
                start_time=buffer[0].start,
                end_time=buffer[-1].end,
                words=list(buffer),
            )
            chunks.append(chunk)
            chunk_num += 1
            buffer = []

    # Handle trailing buffer
    if buffer:
        if len(buffer) < 5 and chunks:
            # Merge into previous chunk
            prev = chunks[-1]
            combined_words = prev.words + buffer
            chunks[-1] = Chunk(
                index=prev.index,
                text=" ".join(w.word for w in combined_words),
                word_count=len(combined_words),
                start_time=combined_words[0].start,
                end_time=combined_words[-1].end,
                words=combined_words,
            )
        else:
            chunk = Chunk(
                index=chunk_num,
                text=" ".join(w.word for w in buffer),
                word_count=len(buffer),
                start_time=buffer[0].start,
                end_time=buffer[-1].end,
                words=list(buffer),
            )
            chunks.append(chunk)

    return chunks


def create_chunks(
    words: List[WordSegment],
    config: ChunkingConfig,
) -> List[Chunk]:
    """
    Split *words* into Chunks using the strategy specified in *config.chunk_mode*.

    Parameters
    ----------
    words : list[WordSegment]
        Flat, time-sorted word list from the transcriber.
    config : ChunkingConfig
        chunk_mode selects strategy: "sentence" or "fixed".

    Returns
    -------
    list[Chunk]
    """
    if not words:
        return []

    if config.chunk_mode == "sentence":
        chunks = _create_chunks_sentence(words, config)
    else:
        chunks = _create_chunks_fixed(words, config)

    logger.info(
        "Created %d chunks (mode=%s) from %d words.",
        len(chunks), config.chunk_mode, len(words),
    )
    return chunks
