"""
Audio transcription via WhisperX with word-level timestamps.

Public API
----------
transcribe(audio_path, config) -> list[WordSegment]
    Returns a flat list of words with start/end times.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from .config import TranscriptionConfig

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    """A single word with its aligned timestamp."""
    word: str
    start: float
    end: float
    score: float = 1.0          # alignment confidence (0-1)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_to_wav(audio_path: Path, tmp_dir: Path) -> Path:
    """Convert any audio format to 16 kHz mono WAV for WhisperX."""
    wav_path = tmp_dir / "input.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    logger.info("Converting audio to WAV: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return wav_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str | Path,
    config: TranscriptionConfig,
    cache_path: str | Path | None = None,
) -> List[WordSegment]:
    """
    Transcribe *audio_path* and return word-level segments.

    If *cache_path* is given and already exists, the cached transcript
    is loaded instead of re-running WhisperX.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # --- Check cache ---
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached transcript from %s", cache_path)
            return _load_transcript(cache_path)

    # --- Lazy-import whisperx (heavy dependency) ---
    try:
        import whisperx
        import torch
    except ImportError:
        raise ImportError(
            "WhisperX is required for transcription.  Install with:\n"
            "  pip install whisperx\n"
            "See README.md for full setup instructions."
        )

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU (will be slower).")
        device = "cpu"

    compute_type = config.compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "float32"

    # --- Transcribe ---
    logger.info(
        "Loading WhisperX model=%s  device=%s  compute=%s",
        config.model, device, compute_type,
    )
    model = whisperx.load_model(
        config.model,
        device=device,
        compute_type=compute_type,
        language=config.language,
    )

    # WhisperX may need WAV; convert if not already.
    suffix = audio_path.suffix.lower()
    if suffix in (".wav",):
        load_path = audio_path
        tmp_dir_obj = None
    else:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        load_path = _convert_to_wav(audio_path, Path(tmp_dir_obj.name))

    logger.info("Transcribing %s …", audio_path.name)
    audio = whisperx.load_audio(str(load_path))
    result = model.transcribe(audio, batch_size=config.batch_size)

    # --- Align (word-level timestamps) ---
    logger.info("Aligning words …")
    align_model, align_meta = whisperx.load_align_model(
        language_code=config.language, device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        device=device,
        return_char_alignments=False,
    )

    # Clean up temp files
    if tmp_dir_obj:
        tmp_dir_obj.cleanup()

    # --- Flatten into WordSegment list ---
    words: List[WordSegment] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            # WhisperX may omit timestamps for some tokens
            if "start" not in w or "end" not in w:
                continue
            words.append(WordSegment(
                word=w["word"].strip(),
                start=round(w["start"], 3),
                end=round(w["end"], 3),
                score=round(w.get("score", 1.0), 4),
            ))

    logger.info("Transcription complete: %d words extracted.", len(words))

    # --- Cache ---
    if cache_path:
        _save_transcript(words, cache_path)
        logger.info("Transcript cached to %s", cache_path)

    return words


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_transcript(words: List[WordSegment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([w.to_dict() for w in words], fh, indent=2)


def _load_transcript(path: Path) -> List[WordSegment]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [WordSegment(**d) for d in data]


def save_transcript(words: List[WordSegment], path: str | Path) -> None:
    """Public wrapper to persist a word list."""
    _save_transcript(words, Path(path))


def load_transcript(path: str | Path) -> List[WordSegment]:
    """Public wrapper to load a persisted word list."""
    return _load_transcript(Path(path))
