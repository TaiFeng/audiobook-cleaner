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
    config.chunking.chunk_mode = "fixed"
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
    hits = detect_profanity(words, banned, padding_seconds=0.0)

    found_words = " ".join(h.reason for h in hits).lower()
    assert "fuck" in found_words
    assert "shit" in found_words
    assert "asshole" in found_words
    assert len(hits) >= 3


def test_mock_classifier_flags_violence():
    words = _make_words(VIOLENT_TEXT, start=120.0)
    config = AppConfig()
    config.chunking.chunk_mode = "fixed"
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
    config.chunking.chunk_mode = "fixed"
    config.chunking.chunk_size = 200
    chunks = create_chunks(words, config.chunking)

    result = mock_classify_chunk(chunks[0])
    assert result.contains_drug_content is True
    assert result.is_flagged


def test_mock_classifier_passes_clean():
    words = _make_words(CLEAN_TEXT, start=0.0)
    config = AppConfig()
    config.chunking.chunk_mode = "fixed"
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
    config.chunking.chunk_mode = "fixed"
    config.chunking.chunk_size = 50
    config.chunking.overlap = 10
    config.chunking.min_chunk_size = 10
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
    profanity_hits = detect_profanity(all_words, banned, padding_seconds=0.0)

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


# ===========================================================================
# Word-level precision tests — zero-padding pipeline
# ===========================================================================
# These tests use hand-crafted WordSegment lists with explicit, non-round
# timestamps so that any arithmetic offset introduced by padding would
# produce values NOT present in the word-timestamp sets, causing failures.
# All four tests assert exact float equality (==) against source timestamps.
# ===========================================================================

def test_word_precise_single_profanity_hit():
    """
    A single banned word in a transcript of clean words.

    With padding_seconds=0.0 the FlaggedRange must start and end at exactly
    the banned word's WhisperX-aligned timestamps.  Adjacent clean words
    must fall entirely outside the range — no bleed in either direction.
    """
    # Craft non-round timestamps so arithmetic drift would be immediately
    # visible (e.g. 0.800 - 0.001 = 0.799, which is NOT in the word set).
    words = [
        WordSegment("The",     0.000, 0.200),
        WordSegment("weather", 0.240, 0.580),
        WordSegment("was",     0.620, 0.760),   # ← preceding clean word
        WordSegment("fucking", 0.800, 0.980),   # ← BANNED
        WordSegment("nice",    1.020, 1.200),   # ← following clean word
        WordSegment("today",   1.240, 1.500),
    ]
    banned_word   = words[3]
    preceding     = words[2]
    following     = words[4]

    hits = detect_profanity(words, {"fucking"}, padding_seconds=0.0)

    assert len(hits) == 1, f"Expected 1 hit, got {len(hits)}"
    hit = hits[0]

    # --- exact timestamp alignment ---
    assert hit.start == banned_word.start, (
        f"hit.start {hit.start} != word.start {banned_word.start} — "
        f"non-zero padding applied to start boundary"
    )
    assert hit.end == banned_word.end, (
        f"hit.end {hit.end} != word.end {banned_word.end} — "
        f"non-zero padding applied to end boundary"
    )

    # --- no backward bleed ---
    assert hit.start > preceding.end, (
        f"Cut starts at {hit.start} but preceding word ends at {preceding.end} — "
        f"backward bleed detected (gap should be {hit.start - preceding.end:.3f}s)"
    )

    # --- no forward bleed ---
    assert hit.end < following.start, (
        f"Cut ends at {hit.end} but following word starts at {following.start} — "
        f"forward bleed detected (gap should be {following.start - hit.end:.3f}s)"
    )

    # --- merge with padding=0.0 must not alter the range ---
    merged = merge_ranges(hits, padding_seconds=0.0)
    assert len(merged) == 1
    assert merged[0].start == hit.start, "merge_ranges expanded start with zero padding"
    assert merged[0].end   == hit.end,   "merge_ranges expanded end with zero padding"


def test_two_word_phrase_merge_without_bleed():
    """
    A two-word banned phrase followed by the second word also appearing in
    the single-word banned set.  After merging, the consolidated range must
    span exactly phrase[0].start → phrase[-1].end with no bleed beyond.
    """
    words = [
        WordSegment("She",    0.000, 0.150),   # clean
        WordSegment("said",   0.190, 0.380),   # ← preceding clean word
        WordSegment("holy",   0.420, 0.590),   # ← start of banned phrase
        WordSegment("shit",   0.630, 0.790),   # ← end of banned phrase; also single-banned
        WordSegment("really", 0.830, 1.100),   # ← following clean word
        WordSegment("loudly", 1.140, 1.380),   # clean
    ]
    phrase_first  = words[2]   # "holy"
    phrase_last   = words[3]   # "shit"
    preceding     = words[1]   # "said"
    following     = words[4]   # "really"

    # Phrase match AND single-word match will both fire; merge must handle it.
    hits = detect_profanity(words, {"holy shit", "shit"}, padding_seconds=0.0)
    assert len(hits) >= 1, "Expected at least one profanity hit"

    merged = merge_ranges(hits, padding_seconds=0.0)
    assert len(merged) == 1, (
        f"Expected 1 merged range for overlapping phrase/single hits, got {len(merged)}: {merged}"
    )
    r = merged[0]

    # --- exact phrase boundary alignment ---
    assert r.start == phrase_first.start, (
        f"Merged start {r.start} != phrase first word start {phrase_first.start}"
    )
    assert r.end == phrase_last.end, (
        f"Merged end {r.end} != phrase last word end {phrase_last.end}"
    )

    # --- no bleed into adjacent clean words ---
    assert r.start > preceding.end, (
        f"Cut bleeds into preceding word 'said': "
        f"cut.start={r.start}, said.end={preceding.end}"
    )
    assert r.end < following.start, (
        f"Cut bleeds into following word 'really': "
        f"cut.end={r.end}, really.start={following.start}"
    )


def test_chunk_start_end_times_are_word_timestamps():
    """
    Chunker must set chunk.start_time and chunk.end_time to exactly the
    first and last word's .start / .end fields — no rounding or offsets.

    Uses irrational-looking timestamps to expose any float arithmetic drift.
    """
    # Timestamps chosen to be non-round and mutually distinct
    words = [
        WordSegment("Once",  1.123, 1.334),
        WordSegment("upon",  1.389, 1.521),
        WordSegment("a",     1.576, 1.612),
        WordSegment("time",  1.667, 1.903),
        WordSegment("there", 1.958, 2.201),
        WordSegment("was",   2.256, 2.399),
        WordSegment("a",     2.454, 2.490),
        WordSegment("great", 2.545, 2.789),
        WordSegment("king",  2.844, 3.012),
    ]

    config = AppConfig()
    config.chunking.chunk_mode = "fixed"
    config.chunking.chunk_size   = 4
    config.chunking.overlap      = 0
    config.chunking.min_chunk_size = 1
    chunks = create_chunks(words, config.chunking)

    assert len(chunks) >= 2, "Need multiple chunks to test boundaries"

    for chunk in chunks:
        first_word = chunk.words[0]
        last_word  = chunk.words[-1]

        assert chunk.start_time == first_word.start, (
            f"Chunk {chunk.index} start_time={chunk.start_time} != "
            f"first word '{first_word.word}'.start={first_word.start} — "
            f"chunker introduced a timestamp offset"
        )
        assert chunk.end_time == last_word.end, (
            f"Chunk {chunk.index} end_time={chunk.end_time} != "
            f"last word '{last_word.word}'.end={last_word.end} — "
            f"chunker introduced a timestamp offset"
        )


def test_zero_padding_pipeline_boundaries_are_word_aligned():
    """
    Full pipeline precision test (profanity + mock classifier + merge).

    The violent section is isolated in its own chunk so the mock classifier
    can flag it cleanly.  After the pipeline runs with padding_seconds=0.0:

    1. Every merged range .start is in {word.start for word in all_words}.
    2. Every merged range .end   is in {word.end   for word in all_words}.
    3. Clean words in a separate time region do not overlap any cut range.

    If any padding > 0 were applied, condition 1 or 2 would fail because the
    offset value would not appear in the word-timestamp sets.
    """
    # --- Violent chunk (chunk 0): 8 words containing mock-classifier triggers ---
    violent = [
        WordSegment("blood",      0.000, 0.210),
        WordSegment("spurted",    0.250, 0.530),
        WordSegment("from",       0.570, 0.710),
        WordSegment("the",        0.750, 0.830),
        WordSegment("wound",      0.870, 1.050),
        WordSegment("and",        1.090, 1.180),
        WordSegment("gore",       1.220, 1.410),
        WordSegment("splattered", 1.450, 1.780),
    ]

    # --- Clean chunk (chunk 1): 8 words, well separated in time ---
    clean = [
        WordSegment("Later",     3.000, 3.210),
        WordSegment("that",      3.250, 3.390),
        WordSegment("evening",   3.430, 3.710),
        WordSegment("the",       3.750, 3.830),
        WordSegment("heroes",    3.870, 4.110),
        WordSegment("gathered",  4.150, 4.440),
        WordSegment("to",        4.480, 4.540),
        WordSegment("celebrate", 4.580, 4.930),
    ]

    all_words = violent + clean

    # Build lookup sets — any padding arithmetic would produce values absent here
    all_starts = {w.start for w in all_words}
    all_ends   = {w.end   for w in all_words}

    # --- Run pipeline (all padding=0.0) ---
    config = AppConfig()
    config.chunking.chunk_mode = "fixed"
    config.chunking.chunk_size    = 8   # each section is exactly one chunk
    config.chunking.overlap       = 0
    config.chunking.min_chunk_size = 1
    config.sensitivity = "moderate"

    profanity_hits = detect_profanity(all_words, set(), padding_seconds=0.0)

    chunks  = create_chunks(all_words, config.chunking)
    results = [mock_classify_chunk(c, config.sensitivity) for c in chunks]

    classifier_ranges = build_ranges_from_results(
        results, config.active_threshold, padding_seconds=0.0,
    )
    merged = merge_ranges(profanity_hits + classifier_ranges, padding_seconds=0.0)

    assert merged, (
        "Pipeline produced no flagged ranges — mock classifier did not flag the "
        "violent section (check keywords: 'blood spurted', 'gore')"
    )

    # --- Assertion 1 & 2: every boundary is an exact word timestamp ---
    for r in merged:
        assert r.start in all_starts, (
            f"Range start {r.start} is not in the word-start set.  "
            f"Non-zero padding must have been applied (nearest word starts: "
            f"{sorted(s for s in all_starts if abs(s - r.start) < 0.5)})"
        )
        assert r.end in all_ends, (
            f"Range end {r.end} is not in the word-end set.  "
            f"Non-zero padding must have been applied (nearest word ends: "
            f"{sorted(e for e in all_ends if abs(e - r.end) < 0.5)})"
        )

    # --- Assertion 3: clean words do not overlap any cut ---
    # The violent section ends at 1.780; clean words start at 3.000 — there
    # is a 1.22-second gap that zero-padding must never bridge.
    for word in clean:
        for r in merged:
            overlaps = word.start < r.end and word.end > r.start
            assert not overlaps, (
                f"Clean word '{word.word}' [{word.start}–{word.end}] overlaps "
                f"cut range [{r.start}–{r.end}].  "
                f"The cut is bleeding into content that must be preserved."
            )

    # --- Diagnostic print (visible with pytest -s) ---
    print(f"\n  Violent section:  [{violent[0].start}–{violent[-1].end}]")
    print(f"  Clean section:    [{clean[0].start}–{clean[-1].end}]")
    for i, r in enumerate(merged):
        print(f"  Cut {i}: [{r.start}–{r.end}]  source={r.source}")
    print(f"  All boundaries word-aligned: ✓")
    print(f"  No bleed into clean section: ✓")


# ---------------------------------------------------------------------------
# New tests for mute_then_remove feature
# ---------------------------------------------------------------------------

def test_flagged_range_action_defaults_to_mute():
    from audiobook_cleaner.profanity import FlaggedRange
    r = FlaggedRange(start=1.0, end=2.0, reason="test", source="profanity", severity="moderate", confidence=1.0)
    assert r.action == "mute"


def test_classifier_ranges_have_remove_action():
    from audiobook_cleaner.transcriber import WordSegment
    from audiobook_cleaner.chunker import create_chunks
    from audiobook_cleaner.classifier import mock_classify_chunk
    from audiobook_cleaner.merger import build_ranges_from_results
    from audiobook_cleaner.config import AppConfig
    words = [
        WordSegment(word="blood", start=0.0, end=0.5, score=1.0),
        WordSegment(word="spurted", start=0.5, end=1.0, score=1.0),
        WordSegment(word="everywhere", start=1.0, end=1.5, score=1.0),
    ]
    config = AppConfig()
    config.chunking.chunk_mode = "fixed"
    config.chunking.chunk_size = 200  # single chunk
    chunks = create_chunks(words, config.chunking)
    results = [mock_classify_chunk(c) for c in chunks]
    flagged = build_ranges_from_results(results, config.active_threshold)
    for r in flagged:
        assert r.action == "remove"


def test_mute_then_remove_edl_has_mixed_actions():
    """write_edl with mixed-action ranges produces per-entry action fields and no top-level mode."""
    import tempfile, json, os
    from audiobook_cleaner.profanity import FlaggedRange
    from audiobook_cleaner.editor import write_edl
    ranges = [
        FlaggedRange(start=1.0, end=2.0, reason="banned", source="profanity", severity="moderate", confidence=1.0, action="mute"),
        FlaggedRange(start=10.0, end=20.0, reason="violence", source="classifier", severity="severe", confidence=0.9, action="remove"),
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        write_edl(ranges, tmp)
        with open(tmp) as f:
            edl = json.load(f)
        assert "mode" not in edl
        assert edl["edits"][0]["action"] == "mute"
        assert edl["edits"][1]["action"] == "remove"
        assert "total_muted_seconds" in edl
        assert "total_removed_seconds" in edl
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Sentence chunking tests
# ---------------------------------------------------------------------------

def test_sentence_chunker_splits_on_punctuation():
    """Sentence chunker splits on terminal punctuation."""
    from audiobook_cleaner.transcriber import WordSegment
    from audiobook_cleaner.chunker import create_chunks
    from audiobook_cleaner.config import ChunkingConfig
    words = [
        WordSegment("Hello", 0.0, 0.5, 1.0),
        WordSegment("world.", 0.5, 1.0, 1.0),   # boundary here
        WordSegment("How", 1.0, 1.3, 1.0),
        WordSegment("are", 1.3, 1.6, 1.0),
        WordSegment("you?", 1.6, 2.0, 1.0),      # boundary here
    ]
    config = ChunkingConfig(chunk_mode="sentence")
    chunks = create_chunks(words, config)
    assert len(chunks) == 2
    assert chunks[0].words[0].word == "Hello"
    assert chunks[1].words[0].word == "How"


def test_sentence_chunker_splits_on_pause():
    """Sentence chunker splits on long pause gap."""
    from audiobook_cleaner.transcriber import WordSegment
    from audiobook_cleaner.chunker import create_chunks
    from audiobook_cleaner.config import ChunkingConfig
    words = [
        WordSegment("One", 0.0, 0.5, 1.0),
        WordSegment("two", 0.5, 1.0, 1.0),
        # 2-second gap here
        WordSegment("three", 3.0, 3.5, 1.0),
        WordSegment("four", 3.5, 4.0, 1.0),
        WordSegment("five", 4.0, 4.5, 1.0),
        WordSegment("six", 4.5, 5.0, 1.0),
        WordSegment("seven.", 5.0, 5.5, 1.0),
    ]
    config = ChunkingConfig(chunk_mode="sentence", pause_gap_seconds=1.5)
    chunks = create_chunks(words, config)
    assert len(chunks) == 2
    assert chunks[0].words[-1].word == "two"
    assert chunks[1].words[0].word == "three"


def test_bisection_narrows_flagged_range():
    """Bisection isolates flagged content to a sub-chunk."""
    from audiobook_cleaner.transcriber import WordSegment
    from audiobook_cleaner.chunker import Chunk
    from audiobook_cleaner.classifier import mock_classify_chunk, _bisect_chunk, _is_flagged

    # Build a chunk where only the second half contains violent words
    clean_words = [WordSegment(f"word{i}", float(i), float(i)+0.9, 1.0) for i in range(10)]
    violent_words = [
        WordSegment("blood", 10.0, 10.9, 1.0),
        WordSegment("spurted", 11.0, 11.9, 1.0),
        WordSegment("everywhere", 12.0, 12.9, 1.0),
    ]
    all_words = clean_words + violent_words
    chunk = Chunk(start_time=0.0, end_time=12.9, text=" ".join(w.word for w in all_words), words=all_words)

    results = _bisect_chunk(chunk, mock_classify_chunk, min_seconds=2.0, max_depth=6)
    flagged = [r for r in results if _is_flagged(r)]
    assert len(flagged) >= 1
    # The flagged result should cover only the violent portion, not the full chunk
    for r in flagged:
        assert r.end_time <= 13.0
        # Should not span the full original chunk
        assert (r.end_time - r.start_time) < (chunk.end_time - chunk.start_time)


def test_mock_classifier_flags_blasphemy():
    """Mock classifier detects blasphemous expletive use."""
    from audiobook_cleaner.transcriber import WordSegment
    from audiobook_cleaner.chunker import Chunk
    from audiobook_cleaner.classifier import mock_classify_chunk
    words = [
        WordSegment("Christ", 0.0, 0.5, 1.0),
        WordSegment("almighty,", 0.5, 1.0, 1.0),
        WordSegment("what", 1.0, 1.3, 1.0),
        WordSegment("happened?", 1.3, 1.8, 1.0),
    ]
    chunk = Chunk(start_time=0.0, end_time=1.8, text="Christ almighty, what happened?", words=words)
    result = mock_classify_chunk(chunk)
    assert result.contains_blasphemy is True
    assert result.severity != "none"

def test_mock_classifier_does_not_flag_religious_narrative():
    """Mock classifier does not flag respectful religious narrative."""
    from audiobook_cleaner.transcriber import WordSegment
    from audiobook_cleaner.chunker import Chunk
    from audiobook_cleaner.classifier import mock_classify_chunk
    words = [
        WordSegment("Jesus", 0.0, 0.5, 1.0),
        WordSegment("said", 0.5, 0.8, 1.0),
        WordSegment("unto", 0.8, 1.1, 1.0),
        WordSegment("them,", 1.1, 1.5, 1.0),
        WordSegment("follow", 1.5, 1.8, 1.0),
        WordSegment("me.", 1.8, 2.1, 1.0),
    ]
    chunk = Chunk(start_time=0.0, end_time=2.1, text="Jesus said unto them, follow me.", words=words)
    result = mock_classify_chunk(chunk)
    assert result.contains_blasphemy is False


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
    print("  ✓ full dry-run pipeline")
    test_word_precise_single_profanity_hit()
    print("  ✓ word-precise single profanity hit")
    test_two_word_phrase_merge_without_bleed()
    print("  ✓ two-word phrase merge without bleed")
    test_chunk_start_end_times_are_word_timestamps()
    print("  ✓ chunk boundaries are word timestamps")
    test_zero_padding_pipeline_boundaries_are_word_aligned()
    print("  ✓ zero-padding pipeline: all boundaries word-aligned, no bleed")
    test_flagged_range_action_defaults_to_mute()
    print("  ✓ FlaggedRange action defaults to mute")
    test_classifier_ranges_have_remove_action()
    print("  ✓ classifier ranges have remove action")
    test_mute_then_remove_edl_has_mixed_actions()
    print("  ✓ mute_then_remove EDL has mixed actions")
    test_sentence_chunker_splits_on_punctuation()
    print("  ✓ sentence chunker splits on punctuation")
    test_sentence_chunker_splits_on_pause()
    print("  ✓ sentence chunker splits on pause")
    test_bisection_narrows_flagged_range()
    print("  ✓ bisection narrows flagged range")
    test_mock_classifier_flags_blasphemy()
    print("  ✓ mock classifier flags blasphemy")
    test_mock_classifier_does_not_flag_religious_narrative()
    print("  ✓ mock classifier does not flag religious narrative")
    print("\nAll tests passed.")
