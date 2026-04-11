"""
Generate human-readable review reports (CSV + JSON) before audio is modified.

Public API
----------
generate_report(results, merged_ranges, output_dir, config)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List

from .classifier import ChunkResult
from .profanity import FlaggedRange
from .config import OutputConfig

logger = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: List[ChunkResult],
    merged_ranges: List[FlaggedRange],
    output_dir: str | Path,
    config: OutputConfig,
    profanity_hits: List[FlaggedRange] | None = None,
) -> dict:
    """
    Write review report files and return a summary dict.

    Outputs (depending on config.report_format):
      - chunk_results.{csv,json}   — per-chunk classification detail
      - flagged_ranges.{csv,json}  — merged cut regions
      - profanity_hits.{csv,json}  — word-level profanity hits
      - summary.json               — aggregate statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.report_format  # "csv", "json", or "both"

    written_files: List[str] = []

    # --- Chunk results ---
    flagged_results = [r for r in results if r.is_flagged]
    if fmt in ("json", "both"):
        p = output_dir / "chunk_results.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=2)
        written_files.append(str(p))

    if fmt in ("csv", "both"):
        p = output_dir / "chunk_results.csv"
        _write_chunk_csv(results, p)
        written_files.append(str(p))

    # --- Merged flagged ranges ---
    if fmt in ("json", "both"):
        p = output_dir / "flagged_ranges.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(
                [{"start": r.start, "end": r.end, "start_fmt": _fmt_time(r.start),
                  "end_fmt": _fmt_time(r.end), "duration": round(r.end - r.start, 3),
                  "reason": r.reason, "source": r.source,
                  "severity": r.severity, "confidence": r.confidence}
                 for r in merged_ranges],
                fh, indent=2,
            )
        written_files.append(str(p))

    if fmt in ("csv", "both"):
        p = output_dir / "flagged_ranges.csv"
        _write_range_csv(merged_ranges, p)
        written_files.append(str(p))

    # --- Profanity hits ---
    if profanity_hits:
        if fmt in ("json", "both"):
            p = output_dir / "profanity_hits.json"
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(
                    [{"start": h.start, "end": h.end, "reason": h.reason} for h in profanity_hits],
                    fh, indent=2,
                )
            written_files.append(str(p))

    # --- Summary ---
    total_muted = sum(r.end - r.start for r in merged_ranges)
    summary = {
        "total_chunks": len(results),
        "flagged_chunks": len(flagged_results),
        "merged_ranges": len(merged_ranges),
        "total_flagged_seconds": round(total_muted, 2),
        "total_flagged_formatted": _fmt_time(total_muted),
        "profanity_hits": len(profanity_hits) if profanity_hits else 0,
        "categories": {
            "explicit_sex": sum(1 for r in results if r.contains_explicit_sex),
            "graphic_violence": sum(1 for r in results if r.contains_graphic_violence),
            "drug_content": sum(1 for r in results if r.contains_drug_content),
            "mature_themes": sum(1 for r in results if r.contains_mature_themes),
            "blasphemy": sum(1 for r in results if r.contains_blasphemy),
        },
        "report_files": written_files,
    }
    p = output_dir / "summary.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Report written to %s  (%d files).", output_dir, len(written_files) + 1)
    return summary


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_chunk_csv(results: List[ChunkResult], path: Path) -> None:
    fields = [
        "chunk_index", "start_time", "end_time", "word_count",
        "contains_explicit_sex", "contains_graphic_violence",
        "contains_drug_content", "contains_mature_themes",
        "contains_blasphemy",
        "severity", "confidence", "reason", "text_preview",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def _write_range_csv(ranges: List[FlaggedRange], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["start", "end", "start_fmt", "end_fmt", "duration",
                         "reason", "source", "severity", "confidence"])
        for r in ranges:
            writer.writerow([
                r.start, r.end, _fmt_time(r.start), _fmt_time(r.end),
                round(r.end - r.start, 3), r.reason, r.source,
                r.severity, r.confidence,
            ])
