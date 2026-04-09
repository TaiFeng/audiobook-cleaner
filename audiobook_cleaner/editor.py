"""
Audio editing via FFmpeg — mute or remove flagged time ranges.

Public API
----------
apply_edits(input_path, output_path, ranges, mode, output_format)
write_edl(ranges, path)
"""

from __future__ import annotations

import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

from .profanity import FlaggedRange

logger = logging.getLogger(__name__)


def _check_ffmpeg() -> str:
    """Return the path to ffmpeg, raising if not found."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise EnvironmentError(
            "ffmpeg not found on PATH.  Install from https://ffmpeg.org/download.html\n"
            "Windows: download a release build, extract, and add the bin/ folder to PATH."
        )
    return ff


def _get_duration(path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# Mute mode
# ---------------------------------------------------------------------------

def _build_mute_filter(ranges: List[FlaggedRange]) -> str:
    """
    Build an FFmpeg audio filter that silences each flagged range.

    Uses chained volume filters with time-based enable expressions.
    """
    if not ranges:
        return "anull"

    parts = []
    for r in ranges:
        # volume=0 during the flagged window
        parts.append(
            f"volume=enable='between(t,{r.start:.3f},{r.end:.3f})':volume=0"
        )
    return ",".join(parts)


def _apply_mute(
    input_path: Path,
    output_path: Path,
    ranges: List[FlaggedRange],
    extra_args: List[str],
) -> None:
    """Mute flagged ranges by setting volume to zero."""
    af = _build_mute_filter(ranges)
    cmd = [
        _check_ffmpeg(), "-y", "-i", str(input_path),
        "-af", af,
        *extra_args,
        str(output_path),
    ]
    logger.info("Running mute edit (%d ranges) …", len(ranges))
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Remove mode
# ---------------------------------------------------------------------------

def _build_remove_filter(
    ranges: List[FlaggedRange],
    total_duration: float,
) -> str:
    """
    Build an FFmpeg filter_complex that trims out flagged ranges and
    concatenates the remaining segments.
    """
    # Compute "keep" intervals: the gaps between flagged ranges
    keeps: list[tuple[float, float]] = []
    cursor = 0.0
    for r in sorted(ranges, key=lambda r: r.start):
        if r.start > cursor:
            keeps.append((cursor, r.start))
        cursor = max(cursor, r.end)
    if cursor < total_duration:
        keeps.append((cursor, total_duration))

    if not keeps:
        # Everything is flagged — return silence
        return "anull"

    filter_parts = []
    labels = []
    for i, (s, e) in enumerate(keeps):
        label = f"a{i}"
        filter_parts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS[{label}]")
        labels.append(f"[{label}]")

    concat = "".join(labels) + f"concat=n={len(keeps)}:v=0:a=1[out]"
    filter_parts.append(concat)
    return ";".join(filter_parts)


def _apply_remove(
    input_path: Path,
    output_path: Path,
    ranges: List[FlaggedRange],
    extra_args: List[str],
) -> None:
    """Remove flagged ranges entirely by trimming and concatenating."""
    duration = _get_duration(input_path)
    fc = _build_remove_filter(ranges, duration)

    cmd = [
        _check_ffmpeg(), "-y", "-i", str(input_path),
        "-filter_complex", fc,
        "-map", "[out]",
        *extra_args,
        str(output_path),
    ]
    logger.info("Running remove edit (%d ranges, duration=%.1fs) …", len(ranges), duration)
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_edits(
    input_path: str | Path,
    output_path: str | Path,
    ranges: List[FlaggedRange],
    mode: str = "mute",
    output_format: Optional[str] = None,
) -> Path:
    """
    Apply audio edits and write the cleaned file.

    Parameters
    ----------
    input_path : path to source audiobook
    output_path : path for the cleaned output
    ranges : merged FlaggedRange list
    mode : "mute" or "remove"
    output_format : override output codec (e.g. "mp3", "m4b"); None = copy codec

    Returns
    -------
    Path to the output file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not ranges:
        logger.info("No ranges to edit — copying original file.")
        shutil.copy2(input_path, output_path)
        return output_path

    # Build codec / format arguments
    extra: List[str] = []
    suffix = output_path.suffix.lower()
    if output_format:
        suffix = f".{output_format.lstrip('.')}"
    if suffix in (".mp3",):
        extra += ["-c:a", "libmp3lame", "-q:a", "2"]
    elif suffix in (".m4b", ".m4a", ".aac"):
        extra += ["-c:a", "aac", "-b:a", "128k"]
    # else: let ffmpeg choose defaults

    if mode == "mute":
        _apply_mute(input_path, output_path, ranges, extra)
    elif mode == "remove":
        _apply_remove(input_path, output_path, ranges, extra)
    else:
        raise ValueError(f"Unknown edit mode: {mode!r} (expected 'mute' or 'remove')")

    logger.info("Cleaned audiobook written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Edit Decision List
# ---------------------------------------------------------------------------

def write_edl(
    ranges: List[FlaggedRange],
    path: str | Path,
    mode: str = "mute",
) -> Path:
    """
    Write an Edit Decision List (JSON) describing all modifications.

    The EDL can be reloaded later to re-apply edits or adjust them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, r in enumerate(ranges, 1):
        h_s, m_s = divmod(int(r.start), 3600), None  # compute below
        entries.append({
            "edit_number": i,
            "action": mode,
            "start": r.start,
            "end": r.end,
            "start_formatted": _fmt_time(r.start),
            "end_formatted": _fmt_time(r.end),
            "duration": round(r.end - r.start, 3),
            "reason": r.reason,
            "source": r.source,
            "severity": r.severity,
            "confidence": r.confidence,
        })

    edl = {
        "version": "1.0",
        "mode": mode,
        "total_edits": len(entries),
        "total_edited_seconds": round(sum(e["duration"] for e in entries), 2),
        "edits": entries,
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(edl, fh, indent=2)

    logger.info("EDL written to %s (%d edits).", path, len(entries))
    return path


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
