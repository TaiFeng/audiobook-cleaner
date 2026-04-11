"""
Configuration management.

Loads settings from a YAML file and exposes them as typed dataclasses.
CLI flags can override any value after loading.
"""

from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}


def severity_gte(a: str, b: str) -> bool:
    """Return True if severity *a* is greater than or equal to *b*."""
    return SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0)


# ---------------------------------------------------------------------------
# Dataclass sections
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionConfig:
    model: str = "large-v2"
    device: str = "cuda"          # "cuda" or "cpu"
    compute_type: str = "float16" # "float16", "int8", "float32"
    language: str = "en"
    batch_size: int = 16


@dataclass
class ChunkingConfig:
    chunk_size: int = 800         # target words per chunk
    overlap: int = 200            # overlap words between adjacent chunks
    min_chunk_size: int = 300
    max_chunk_size: int = 1500
    chunk_mode: str = "sentence"        # "sentence" (recommended) or "fixed"
    max_sentence_words: int = 120       # max words per sentence-chunk before forcing a split
    pause_gap_seconds: float = 1.5      # treat a gap >= this as a sentence boundary


@dataclass
class ProfanityConfig:
    enabled: bool = True
    banned_words_file: str = "banned_words.txt"
    padding_seconds: float = 0.0  # silence padding each side of a hit (0.0 = word-precise)


@dataclass
class ClassificationConfig:
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_concurrent: int = 4
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 2.0
    bisect: bool = True                 # drill down into flagged chunks to find precise boundaries
    bisect_min_seconds: float = 5.0     # stop bisecting when sub-chunk is shorter than this
    bisect_max_depth: int = 6           # max recursion depth (safety cap)
    screen_blasphemy: bool = True       # flag religious names/concepts used as expletives or curses


@dataclass
class ThresholdConfig:
    min_confidence: float = 0.5
    min_severity: str = "moderate"


@dataclass
class OutputConfig:
    mode: str = "mute"            # "mute" or "remove"
    padding_seconds: float = 0.0  # padding added around each flagged range (0.0 = word-precise)
    format: Optional[str] = None  # None = same as input; or "mp3" / "m4b"
    report_format: str = "both"   # "csv", "json", or "both"
    # Encoding overrides — None means auto-detect from the source file
    bitrate: Optional[str] = None        # e.g. "64k"  — overrides probed bitrate
    sample_rate: Optional[int] = None    # e.g. 44100  — overrides probed sample rate
    channels: Optional[int] = None       # e.g. 1      — overrides probed channel count


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, ThresholdConfig] = {
    "strict":   ThresholdConfig(min_confidence=0.3, min_severity="mild"),
    "moderate": ThresholdConfig(min_confidence=0.5, min_severity="moderate"),
    "minimal":  ThresholdConfig(min_confidence=0.7, min_severity="severe"),
}


@dataclass
class AppConfig:
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    profanity: ProfanityConfig = field(default_factory=ProfanityConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    sensitivity: str = "moderate"
    thresholds: Dict[str, ThresholdConfig] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- helpers ----------------------------------------------------------

    @property
    def active_threshold(self) -> ThresholdConfig:
        return self.thresholds.get(self.sensitivity, DEFAULT_THRESHOLDS["moderate"])

    # -- factory ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Build an AppConfig from a YAML file, falling back to defaults."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found — using defaults.", path)
            return cls()

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        cfg = cls()
        # Transcription
        if "transcription" in raw:
            for k, v in raw["transcription"].items():
                if hasattr(cfg.transcription, k):
                    setattr(cfg.transcription, k, v)
        # Chunking
        if "chunking" in raw:
            for k, v in raw["chunking"].items():
                if hasattr(cfg.chunking, k):
                    setattr(cfg.chunking, k, v)
        # Profanity
        if "profanity" in raw:
            for k, v in raw["profanity"].items():
                if hasattr(cfg.profanity, k):
                    setattr(cfg.profanity, k, v)
        # Classification
        if "classification" in raw:
            for k, v in raw["classification"].items():
                if hasattr(cfg.classification, k):
                    setattr(cfg.classification, k, v)
        # Output
        if "output" in raw:
            for k, v in raw["output"].items():
                if hasattr(cfg.output, k):
                    setattr(cfg.output, k, v)
        # Sensitivity
        if "sensitivity" in raw:
            cfg.sensitivity = raw["sensitivity"]
        # Thresholds overrides
        if "thresholds" in raw:
            for level, vals in raw["thresholds"].items():
                cfg.thresholds[level] = ThresholdConfig(**vals)

        logger.info("Loaded config from %s  (sensitivity=%s)", path, cfg.sensitivity)
        return cfg
