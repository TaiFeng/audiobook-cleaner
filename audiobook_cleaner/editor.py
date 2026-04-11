"""
Audio editing via FFmpeg — mute or remove flagged time ranges.

Public API
----------
probe_audio(input_path) -> dict
    Read codec, bitrate, sample rate, and channel count from a source file.
apply_edits(input_path, output_path, ranges, mode, output_format,
            bitrate, sample_rate, channels)
    Apply edits; encoding parameters default to probed source values.
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


def get_audio_duration(path: str | Path) -> float:
    """Public wrapper — return the duration of *path* in seconds."""
    return _get_duration(Path(path))


def probe_audio(path: Path) -> dict:
    """
    Read audio stream properties from *path* using ffprobe.

    Combines stream-level entries (codec, sample_rate, channels) with the
    format-level bit_rate, which is more reliable for VBR-encoded files
    because it is computed from file size / duration rather than from
    in-stream metadata that may be absent or set to zero.

    Returns a dict with string values for the keys:
      codec_name, bit_rate (bps), sample_rate (Hz), channels
    Returns an empty dict on failure so callers can fall back gracefully.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,bit_rate,sample_rate,channels:format=bit_rate",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        info: dict = {}
        streams = data.get("streams", [])
        if streams:
            info.update(streams[0])

        # Many VBR MP3 files report bit_rate="0" at the stream level.
        # The format-level value (file_size * 8 / duration) is always accurate.
        fmt_br = (data.get("format") or {}).get("bit_rate")
        stream_br = info.get("bit_rate", "0")
        if fmt_br and (not stream_br or stream_br == "0"):
            info["bit_rate"] = fmt_br

        logger.info(
            "Source audio probed: codec=%s  bitrate=%s bps  "
            "sample_rate=%s Hz  channels=%s",
            info.get("codec_name", "?"),
            info.get("bit_rate",  "?"),
            info.get("sample_rate", "?"),
            info.get("channels",  "?"),
        )
        return info

    except Exception as exc:
        logger.warning(
            "ffprobe failed for %s: %s — using codec defaults for output.",
            path.name, exc,
        )
        return {}


def _build_codec_args(
    probe: dict,
    output_suffix: str,
    bitrate_override: Optional[str] = None,
    sample_rate_override: Optional[int] = None,
    channels_override: Optional[int] = None,
) -> List[str]:
    """
    Build FFmpeg audio codec arguments that mirror the source file's encoding.

    Priority for each parameter:
      CLI / config override  →  probed value  →  per-codec fallback default

    Parameters
    ----------
    probe            : dict returned by probe_audio()
    output_suffix    : file extension including dot, e.g. ".mp3"
    bitrate_override : e.g. "64k" — bypasses auto-detection
    sample_rate_override : e.g. 44100 — bypasses auto-detection
    channels_override    : e.g. 1 (mono) — bypasses auto-detection
    """
    args: List[str] = []

    # --- Sample rate ---
    sr = sample_rate_override
    if sr is None and probe.get("sample_rate"):
        sr = int(probe["sample_rate"])
    if sr:
        args += ["-ar", str(sr)]

    # --- Channels ---
    ch = channels_override
    if ch is None and probe.get("channels"):
        ch = int(probe["channels"])
    if ch:
        args += ["-ac", str(ch)]

    # --- Bitrate — convert bps string → kbps string ---
    if bitrate_override:
        # Accept both "64" and "64k"
        br_str = bitrate_override if bitrate_override.endswith("k") else f"{bitrate_override}k"
    elif probe.get("bit_rate") and probe["bit_rate"] not in ("0", ""):
        br_kbps = max(1, int(float(probe["bit_rate"])) // 1000)
        br_str = f"{br_kbps}k"
    else:
        br_str = None  # fall through to per-codec default below

    # --- Codec selection + bitrate application ---
    if output_suffix == ".mp3":
        args += ["-c:a", "libmp3lame"]
        if br_str:
            args += ["-b:a", br_str]      # CBR matching source
        else:
            args += ["-q:a", "4"]         # VBR ~128 kbps fallback
    elif output_suffix in (".m4b", ".m4a", ".aac"):
        args += ["-c:a", "aac"]
        args += ["-b:a", br_str or "64k"]
    else:
        if br_str:
            args += ["-b:a", br_str]      # best-effort for other containers

    logger.info(
        "Output encoding: codec=%s  bitrate=%s  sample_rate=%s Hz  channels=%s",
        output_suffix.lstrip("."),
        br_str or "default",
        sr or "default",
        ch or "default",
    )
    return args


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
    bitrate: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> Path:
    """
    Apply audio edits and write the cleaned file.

    The output is re-encoded to match the source file's bitrate, sample rate,
    and channel count by default.  Use the override parameters to force
    specific values when auto-detection is not sufficient.

    Parameters
    ----------
    input_path    : path to source audiobook
    output_path   : path for the cleaned output
    ranges        : merged FlaggedRange list
    mode          : "mute" or "remove"
    output_format : force a container/codec (e.g. "mp3", "m4b"); None = same as input
    bitrate       : e.g. "64k" — overrides auto-detected bitrate
    sample_rate   : e.g. 44100 — overrides auto-detected sample rate
    channels      : e.g. 1 (mono) — overrides auto-detected channel count

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

    # Determine output container suffix
    suffix = output_path.suffix.lower()
    if output_format:
        suffix = f".{output_format.lstrip('.')}"

    # Probe source file and build codec arguments that match it
    probe = probe_audio(input_path)
    extra = _build_codec_args(probe, suffix, bitrate, sample_rate, channels)

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
