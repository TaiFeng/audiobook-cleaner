"""
Dry-run integration test — validates the full pipeline logic using
a mock transcript.  No audio files, no API calls required.

Run with:
    python -m pytest tests/test_dry_run.py -v
  or:
    python main.py dry-run
"""

import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiobook_cleaner.config import AppConfig, severity_gte
from audiobook_cleaner.transcriber import WordSegment
from audiobook_cleaner.profanity import load_banned_words, detect_profanity
from audiobook_cleaner.chunker import create_chunks
from audiobook_cleaner.classifier import mock_classify_chunk
from audiobook_cleaner.merger import merge_ranges, build_ranges_from_results
from audiobook_cleaner.reporter import generate_report
from audiobook_cleaner.editor import write_edl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_words(text: str, start: float = 0.0) -> list[WordSegment]:
    """Generate mock WordSegments from plain text."""
    words = []
    t = start
    for token in text.split():
        dur = len(token) * 0.04 + 0.08
        words.append(WordSegment(word=token, start=round(t, 3), end=round(t + dur, 3)))
        t += dur + 0.04
    return words


# ---------------------------------------------------------------------------
# Test scenes
# ---------------------------------------------------------------------------

CLEAN_TEXT = (
    "Emily walked through the sunlit meadow carrying a basket of wildflowers. "
    "The birds sang cheerfully in the tall oak trees and a gentle breeze "
    "rustled the leaves. She was on her way to visit her grandmother who "
    "lived in a cozy cottage at the edge of the forest. It was a perfect "
    "spring day full of warmth and promise."
)

VIOLENT_TEXT = (
    "The creature lunged forward and its claws tore through the knight's "
    "armor. Blood spurted from deep gashes across his chest. His skull "
    "cracked against the stone wall and his entrails spilled across the "
    "floor. The soldiers watched the slaughter in horror as gore painted "
    "the walls crimson."
)

DRUG_TEXT = (
    "Someone laid out lines of cocaine on the table. Jake injected heroin "
    "into his arm and collapsed against the wall. The smell of crack smoke "
    "filled the room as the drug deal spiraled out of control and Marcus "
    "feared an overdose."
)

PROFANITY_TEXT = (
    "The damn car broke down again and Tom said what the fuck is going on. "
    "He kicked the tire and called it a piece of shit. This was a hell of "
    "a day and he felt like an asshole for not checking the engine earlier."
)

CLEAN_ENDING = (
    "Emily reached the cottage and knocked on the door. Her grandmother "
    "greeted her with a warm smile and a plate of fresh cookies. They sat "
    "together by the fireplace and read stories until the stars came out."
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chunker_produces_overlapping_chunks():
    words = _make_words(CLEAN_TEXT + " " + VIOLENT_TEXT + " " + CLEAN_ENDING)
    config = AppConfig()
    config.chunking.chunk_size = 40
    config.chunking.overlap = 10
    config.chunking.min_chunk_size = 10  # low minimum so small remnants stay separate
    chunks = create_chunks(words, config.chunking)

    assert len(chunks) >= 2, "Should produce multiple chunks"
    # Verify overlap: last words of chunk N should appear in chunk N+1
    for i in range(len(chunks) - 1):
        assert chunks[i].end_time <= chunks[i + 1].end_time
        # Start of next chunk should be before end of current
        assert chunks[i + 1].start_time < chunks[i].end_time


def test_profanity_detection():
    words = _make_words(PROFANITY_TEXT)
    banned = {"fuck", "shit", "asshole", "damn", "hell"}
    hits = detect_profanity(words, banned, padding_seconds=0.2)

    found_words = " ".join(h.reason for h in hits).lower()
    assert "fuck" in found_words
    assert "shit" in found_words
    assert "asshole" in found_words
    assert len(hits) >= 3


def test_mock_classifier_flags_violence():
    words = _make_words(VIOLENT_TEXT, start=120.0)
    config = AppConfig()
    config.chunking.chunk_size = 200  # single chunk
    chunks = create_chunks(words, config.chunking)
    assert len(chunks) == 1

    result = mock_classify_chunk(chunks[0])
    assert result.contains_graphic_violence is True
    assert result.severity in ("moderate", "severe")
    assert result.is_flagged


def test_mock_classifier_flags_drugs():
    words = _make_words(DRUG_TEXT, start=180.0)
    config = AppConfig()
    config.chunking.chunk_size = 200
    chunks = create_chunks(words, config.chunking)

    result = mock_classify_chunk(chunks[0])
    assert result.contains_drug_content is True
    assert result.is_flagged


def test_mock_classifier_passes_clean():
    words = _make_words(CLEAN_TEXT, start=0.0)
    config = AppConfig()
    config.chunking.chunk_size = 200
    chunks = create_chunks(words, config.chunking)

    result = mock_classify_chunk(chunks[0])
    assert result.is_flagged is False
    assert result.severity == "none"


def test_merger_combines_overlapping_ranges():
    from audiobook_cleaner.profanity import FlaggedRange

    ranges = [
        FlaggedRange(start=10.0, end=20.0, reason="A", source="profanity"),
        FlaggedRange(start=15.0, end=25.0, reason="B", source="classifier"),
        FlaggedRange(start=50.0, end=55.0, reason="C", source="profanity"),
    ]
    merged = merge_ranges(ranges, padding_seconds=0.5)

    # First two ranges overlap (10-20 and 15-25) → merge into one
    assert len(merged) == 2
    # After 0.5s padding: 9.5–25.5 and 49.5–55.5
    assert merged[0].start <= 10.0
    assert merged[0].end >= 25.0
    assert merged[1].start <= 50.0


def test_severity_comparison():
    assert severity_gte("severe", "moderate")
    assert severity_gte("moderate", "moderate")
    assert not severity_gte("mild", "moderate")
    assert severity_gte("mild", "none")


def test_full_dry_run_pipeline(tmp_path):
    """End-to-end dry run producing reports and EDL."""
    config = AppConfig()
    config.chunking.chunk_size = 50
    config.chunking.overlap = 10
    config.sensitivity = "moderate"

    # Build combined transcript
    all_words = (
        _make_words(CLEAN_TEXT, 0.0)
        + _make_words(VIOLENT_TEXT, 60.0)
        + _make_words(DRUG_TEXT, 120.0)
        + _make_words(PROFANITY_TEXT, 180.0)
        + _make_words(CLEAN_ENDING, 240.0)
    )

    # Profanity
    banned = {"fuck", "shit", "asshole", "damn"}
    profanity_hits = detect_profanity(all_words, banned, padding_seconds=0.3)

    # Chunk + classify (mock)
    chunks = create_chunks(all_words, config.chunking)
    results = [mock_classify_chunk(c, config.sensitivity) for c in chunks]

    # Merge
    from audiobook_cleaner.merger import build_ranges_from_results
    classifier_ranges = build_ranges_from_results(
        results, config.active_threshold, padding_seconds=0.0,
    )
    all_ranges = profanity_hits + classifier_ranges
    merged = merge_ranges(all_ranges, padding_seconds=config.output.padding_seconds)

    # Report
    report_dir = tmp_path / "report"
    summary = generate_report(results, merged, report_dir, config.output, profanity_hits)

    # EDL
    edl_path = tmp_path / "edl.json"
    write_edl(merged, edl_path, config.output.mode)

    # Assertions
    assert summary["total_chunks"] == len(chunks)
    assert summary["flagged_chunks"] > 0
    assert summary["merged_ranges"] > 0
    assert summary["profanity_hits"] > 0
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "chunk_results.csv").exists()
    assert (report_dir / "flagged_ranges.json").exists()
    assert edl_path.exists()

    with open(edl_path) as f:
        edl = json.load(f)
    assert edl["total_edits"] > 0

    print(f"\n  Chunks: {summary['total_chunks']}")
    print(f"  Flagged: {summary['flagged_chunks']}")
    print(f"  Profanity hits: {summary['profanity_hits']}")
    print(f"  Merged ranges: {summary['merged_ranges']}")
    print(f"  Total flagged time: {summary['total_flagged_formatted']}")
    print(f"  EDL edits: {edl['total_edits']}")
    print("  PASS ✓")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, os
    tmp = Path(tempfile.mkdtemp())
    print("Running dry-run tests …\n")
    test_chunker_produces_overlapping_chunks()
    print("  ✓ chunker overlap")
    test_profanity_detection()
    print("  ✓ profanity detection")
    test_mock_classifier_flags_violence()
    print("  ✓ classifier flags violence")
    test_mock_classifier_flags_drugs()
    print("  ✓ classifier flags drugs")
    test_mock_classifier_passes_clean()
    print("  ✓ classifier passes clean content")
    test_merger_combines_overlapping_ranges()
    print("  ✓ merger combines overlapping ranges")
    test_severity_comparison()
    print("  ✓ severity comparison")
    test_full_dry_run_pipeline(tmp)
    print("\nAll tests passed.")
