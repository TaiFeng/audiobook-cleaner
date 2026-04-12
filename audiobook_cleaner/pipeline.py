"""
End-to-end pipeline orchestrator.

Each public method corresponds to a CLI subcommand and calls the
appropriate modules in sequence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .transcriber import transcribe, save_transcript, load_transcript, WordSegment
from .profanity import load_banned_words, detect_profanity, FlaggedRange
from .chunker import create_chunks, Chunk
from .classifier import classify_chunks, mock_classify_chunk, ChunkResult, _is_flagged, _bisect_chunk
from .merger import merge_ranges, build_ranges_from_results
from .reporter import generate_report
from .editor import apply_edits, write_edl, load_edl, get_audio_duration

logger = logging.getLogger(__name__)


def _clip_ranges_to_file(merged, file_start: float, file_end: float):
    """Intersect absolute-time ranges with [file_start, file_end] and translate to file-local timestamps."""
    local_ranges = []
    for r in merged:
        if r.end <= file_start or r.start >= file_end:
            continue
        local_ranges.append(FlaggedRange(
            start=max(r.start, file_start) - file_start,
            end=min(r.end, file_end) - file_start,
            reason=r.reason,
            source=r.source,
            severity=r.severity,
            confidence=r.confidence,
            action=r.action,
        ))
    return local_ranges


def _batch_summary(files, failures):
    """Log a summary of batch processing results."""
    completed = len(files) - len(failures)
    logger.info(f"Batch complete: {completed}/{len(files)} files processed successfully.")
    if failures:
        for f, err in failures:
            logger.warning(f"  FAILED: {f} — {err}")


class Pipeline:
    """High-level orchestrator wired to an AppConfig."""

    def __init__(self, config: AppConfig):
        self.config = config

    # -----------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------

    def run_full(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        report_only: bool = False,
    ) -> None:
        """Run transcription → analysis → report → (optional) edit."""
        inp = Path(input_path)
        stem = inp.stem
        work_dir = inp.parent / f"{stem}_cleaned"
        work_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Transcribe ---
        transcript_path = work_dir / "transcript.json"
        logger.info("=== Step 1/6: Transcription ===")
        words = transcribe(
            inp, self.config.transcription, cache_path=transcript_path,
        )

        # --- Step 2: Profanity detection ---
        logger.info("=== Step 2/6: Profanity detection ===")
        profanity_hits = self._detect_profanity(words)

        # --- Step 3: Chunking ---
        logger.info("=== Step 3/6: Chunking ===")
        chunks = create_chunks(words, self.config.chunking)

        # --- Step 4: AI classification ---
        logger.info("=== Step 4/6: AI classification ===")
        results = classify_chunks(
            chunks, self.config.classification, self.config.sensitivity,
        )

        # --- Step 5: Merge & report ---
        logger.info("=== Step 5/6: Merge & report ===")
        merged = self._merge_all(profanity_hits, results)

        report_dir = work_dir / "report"
        summary = generate_report(
            results, merged, report_dir, self.config.output,
            profanity_hits=profanity_hits,
        )
        edl_path = write_edl(merged, work_dir / "edl.json", self.config.output.mode)

        self._print_summary(summary)

        if report_only:
            logger.info("Report-only mode — audio not modified.")
            logger.info("Review the report at: %s", report_dir)
            return

        # --- Step 6: Edit audio ---
        logger.info("=== Step 6/6: Audio editing ===")
        if not output_path:
            suffix = inp.suffix
            output_path = str(inp.parent / f"{stem}_clean{suffix}")

        apply_edits(
            inp, output_path, merged,
            mode=self.config.output.mode,
            output_format=self.config.output.format,
            bitrate=self.config.output.bitrate,
            sample_rate=self.config.output.sample_rate,
            channels=self.config.output.channels,
        )
        logger.info("Done!  Cleaned file: %s", output_path)

    # -----------------------------------------------------------------
    # Transcribe only
    # -----------------------------------------------------------------

    def run_transcribe(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> None:
        inp = Path(input_path)
        if not output_path:
            output_path = str(inp.with_suffix(".transcript.json"))
        words = transcribe(inp, self.config.transcription, cache_path=output_path)
        logger.info("Transcript saved: %s  (%d words)", output_path, len(words))

    # -----------------------------------------------------------------
    # Analyze existing transcript
    # -----------------------------------------------------------------

    def run_analyze(
        self,
        transcript_path: str,
        output_dir: Optional[str] = None,
    ) -> None:
        words = load_transcript(transcript_path)
        logger.info("Loaded transcript: %d words", len(words))

        profanity_hits = self._detect_profanity(words)
        chunks = create_chunks(words, self.config.chunking)
        results = classify_chunks(
            chunks, self.config.classification, self.config.sensitivity,
        )
        merged = self._merge_all(profanity_hits, results)

        if not output_dir:
            output_dir = str(Path(transcript_path).parent / "report")
        summary = generate_report(
            results, merged, output_dir, self.config.output,
            profanity_hits=profanity_hits,
        )
        edl_path = write_edl(
            merged,
            Path(output_dir) / "edl.json",
            self.config.output.mode,
        )
        self._print_summary(summary)

    # -----------------------------------------------------------------
    # Clean from EDL
    # -----------------------------------------------------------------

    def run_clean(
        self,
        input_path: str,
        edl_path: str,
        output_path: Optional[str] = None,
    ) -> None:
        inp = Path(input_path)
        ranges = load_edl(edl_path)

        # Auto-detect mode from EDL action fields so `clean` always applies
        # the correct passes without requiring -m to be specified.
        # An explicit -m flag on the CLI overrides this via config.output.mode.
        actions = {r.action for r in ranges}
        explicit_mode = self.config.output.mode  # set by -m if user passed it
        if explicit_mode in ("remove", "mute_then_remove"):
            # User explicitly chose a mode — honour it
            mode = explicit_mode
        elif "mute" in actions and "remove" in actions:
            mode = "mute_then_remove"
        elif "remove" in actions:
            mode = "remove"
        else:
            mode = "mute"
        logger.info("EDL contains %d mute + %d remove ranges → auto-selected mode '%s'.",
                    sum(1 for r in ranges if r.action == "mute"),
                    sum(1 for r in ranges if r.action == "remove"),
                    mode)

        if not output_path:
            suffix = inp.suffix
            output_path = str(inp.parent / f"{inp.stem}_clean{suffix}")

        apply_edits(
            inp, output_path, ranges,
            mode=mode,
            bitrate=self.config.output.bitrate,
            sample_rate=self.config.output.sample_rate,
            channels=self.config.output.channels,
        )
        logger.info("Cleaned file: %s", output_path)

    # -----------------------------------------------------------------
    # Dry run (mock data, no audio, no API)
    # -----------------------------------------------------------------

    def run_dry_run(self) -> None:
        """
        Run the full pipeline on a mock transcript to validate logic.

        No audio file, no API calls — uses keyword-based mock classifier.
        """
        logger.info("=" * 60)
        logger.info("DRY RUN — testing pipeline with mock data")
        logger.info("=" * 60)

        words = _generate_mock_transcript()
        logger.info("Mock transcript: %d words", len(words))

        # Profanity
        profanity_hits = self._detect_profanity(words)

        # Chunk
        chunks = create_chunks(words, self.config.chunking)

        # Mock classification (no API)
        logger.info("Running mock classification on %d chunks …", len(chunks))
        initial_results = [
            mock_classify_chunk(c, self.config.sensitivity) for c in chunks
        ]

        # Bisection drill-down on flagged mock results
        results = []
        cls_config = self.config.classification
        for r, chunk in zip(initial_results, chunks):
            if (
                cls_config.bisect
                and _is_flagged(r)
                and (chunk.end_time - chunk.start_time) > cls_config.bisect_min_seconds
            ):
                classify_fn = lambda c: mock_classify_chunk(c, self.config.sensitivity)
                bisected = _bisect_chunk(chunk, classify_fn, cls_config.bisect_min_seconds, cls_config.bisect_max_depth)
                results.extend(bisected)
            else:
                results.append(r)

        flagged = [r for r in results if r.is_flagged]
        logger.info("Mock classification: %d/%d results flagged.", len(flagged), len(results))

        # Merge
        merged = self._merge_all(profanity_hits, results)

        # Report
        report_dir = Path("dry_run_output/report")
        summary = generate_report(
            results, merged, report_dir, self.config.output,
            profanity_hits=profanity_hits,
        )
        write_edl(merged, Path("dry_run_output/edl.json"), self.config.output.mode)

        self._print_summary(summary)
        logger.info("Dry-run output written to: dry_run_output/")
        logger.info("DRY RUN COMPLETE — all pipeline stages validated.")

    # -----------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------

    def _batch_output_path(self, input_path: Path, output_dir):
        """Return the output file path for a cleaned file."""
        output_dir = Path(output_dir) if output_dir else input_path.parent
        return output_dir / f"{input_path.stem}_clean{input_path.suffix}"

    def _batch_work_dir(self, input_path: Path, output_dir):
        """Return the work directory for a file's intermediates."""
        output_dir = Path(output_dir) if output_dir else input_path.parent
        return output_dir / f"{input_path.stem}_cleaned"

    def _run_batch_independent(self, files, output_dir, report_only: bool):
        """Process each file independently through the full pipeline."""
        failures = []
        for f in files:
            f = Path(f)
            out_path = self._batch_output_path(f, output_dir)
            work_dir = self._batch_work_dir(f, output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.run_full(
                    input_path=str(f),
                    output_path=str(out_path),
                    report_only=report_only,
                )
            except Exception as e:
                failures.append((str(f), str(e)))
        _batch_summary(files, failures)

    def _run_batch_join(self, files, output_dir, report_only: bool):
        """Transcribe all files, combine transcripts with time offsets, classify once, then apply edits per file."""
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Transcribe each file, accumulate combined word list with offsets
        all_words = []
        file_spans = []  # (path, start_offset, end_offset)
        offset = 0.0
        failures = []

        for f in files:
            f = Path(f)
            work_dir = self._batch_work_dir(f, output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            duration = get_audio_duration(str(f))
            file_start = offset
            file_end = offset + duration

            try:
                transcript_path = work_dir / "transcript.json"
                words = transcribe(f, self.config.transcription, cache_path=transcript_path)
                # Shift word timestamps by offset
                shifted = []
                for w in words:
                    shifted.append(WordSegment(
                        word=w.word,
                        start=w.start + offset,
                        end=w.end + offset,
                        score=w.score,
                    ))
                all_words.extend(shifted)
            except Exception as e:
                logger.warning(f"Transcription failed for {f}: {e}")
                failures.append((str(f), str(e)))

            file_spans.append((f, file_start, file_end))
            offset = file_end

        if not all_words:
            logger.error("No words transcribed across any file. Aborting join mode.")
            return

        # Step 2: Run profanity detection + chunking + classification on combined transcript
        profanity_hits = self._detect_profanity(all_words)
        chunks = create_chunks(all_words, self.config.chunking)
        results = classify_chunks(
            chunks, self.config.classification, self.config.sensitivity,
        )
        merged = self._merge_all(profanity_hits, results)

        # Step 3: Write combined report
        join_report_dir = Path(output_dir) / "_batch_join_report" if output_dir else Path(files[0]).parent / "_batch_join_report"
        join_report_dir.mkdir(parents=True, exist_ok=True)
        generate_report(
            results, merged, join_report_dir, self.config.output,
            profanity_hits=profanity_hits,
        )

        # Write combined EDL
        combined_edl_path = join_report_dir / "combined_edl.json"
        write_edl(merged, str(combined_edl_path), self.config.output.mode)

        if report_only:
            logger.info("report-only mode: skipping audio edits.")
            _batch_summary(files, failures)
            return

        # Step 4: Per-file: clip merged ranges to file span and apply edits
        for (f, file_start, file_end) in file_spans:
            local_ranges = _clip_ranges_to_file(merged, file_start, file_end)
            out_path = self._batch_output_path(f, output_dir)
            work_dir = self._batch_work_dir(f, output_dir)

            edl_path = work_dir / "edl.json"
            write_edl(local_ranges, str(edl_path), self.config.output.mode)

            if local_ranges:
                try:
                    apply_edits(
                        str(f), str(out_path), local_ranges,
                        mode=self.config.output.mode,
                        output_format=self.config.output.format,
                        bitrate=self.config.output.bitrate,
                        sample_rate=self.config.output.sample_rate,
                        channels=self.config.output.channels,
                    )
                except Exception as e:
                    logger.warning(f"Edit failed for {f}: {e}")
                    failures.append((str(f), str(e)))
            else:
                import shutil
                shutil.copy2(str(f), str(out_path))
                logger.info(f"No cuts for {f.name} — copied to output unchanged.")

        _batch_summary(files, failures)

    def run_batch(self, input_files, output_dir=None, join: bool = False, report_only: bool = False):
        """
        Process multiple audiobook files in batch.

        Args:
            input_files: List of file paths (str or Path) to process.
            output_dir:  Directory for cleaned files and work dirs. If None, output lives next to each source.
            join:        If True, combine transcripts across all files before classifying (catches cross-boundary content).
            report_only: If True, run the full pipeline but skip writing audio output.
        """
        files = [Path(f) for f in input_files]
        if not files:
            logger.warning("run_batch called with no files.")
            return

        logger.info(f"Batch processing {len(files)} file(s). join={join}, report_only={report_only}")

        if join:
            self._run_batch_join(files, output_dir, report_only)
        else:
            self._run_batch_independent(files, output_dir, report_only)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _detect_profanity(self, words):
        if not self.config.profanity.enabled:
            return []
        banned = load_banned_words(self.config.profanity.banned_words_file)
        return detect_profanity(words, banned, self.config.profanity.padding_seconds)

    def _merge_all(self, profanity_hits, results):
        classifier_ranges = build_ranges_from_results(
            results,
            self.config.active_threshold,
            padding_seconds=0.0,  # padding applied in merge step
        )
        all_ranges = profanity_hits + classifier_ranges
        return merge_ranges(all_ranges, padding_seconds=self.config.output.padding_seconds)

    @staticmethod
    def _print_summary(summary: dict) -> None:
        logger.info("-" * 50)
        logger.info("SUMMARY")
        logger.info("  Total chunks analyzed : %d", summary["total_chunks"])
        logger.info("  Chunks flagged        : %d", summary["flagged_chunks"])
        logger.info("  Profanity hits        : %d", summary["profanity_hits"])
        logger.info("  Merged edit ranges    : %d", summary["merged_ranges"])
        logger.info("  Total flagged time    : %s  (%.1fs)",
                     summary["total_flagged_formatted"],
                     summary["total_flagged_seconds"])
        cats = summary["categories"]
        logger.info("  Categories:")
        logger.info("    Explicit sex        : %d chunks", cats["explicit_sex"])
        logger.info("    Graphic violence    : %d chunks", cats["graphic_violence"])
        logger.info("    Drug content        : %d chunks", cats["drug_content"])
        logger.info("    Mature themes       : %d chunks", cats["mature_themes"])
        logger.info("-" * 50)


# ---------------------------------------------------------------------------
# Mock transcript generator (for dry-run)
# ---------------------------------------------------------------------------

def _generate_mock_transcript():
    """Build a realistic mock word list spanning ~6 'minutes' of audio."""
    scenes = [
        # Scene 1: Clean intro (0-60s)
        (
            "The morning sun cast golden light across the meadow as Emily walked "
            "along the winding path toward the old stone bridge. Birds sang in "
            "the oak trees overhead and the gentle breeze carried the scent of "
            "wildflowers. She smiled thinking about the adventure that lay ahead. "
            "Her grandfather had told her stories about the enchanted forest beyond "
            "the bridge and today she would finally see it for herself. The path "
            "narrowed as she approached the tree line and she could hear the soft "
            "babbling of the creek below. Everything felt peaceful and full of "
            "promise on this beautiful spring day."
        ),
        # Scene 2: Mild tension (60-120s) — should pass moderate filter
        (
            "The forest grew darker as Emily ventured deeper. Strange shadows "
            "danced between the ancient trunks and an owl hooted somewhere above. "
            "She clutched her lantern tighter and reminded herself that grandpa "
            "said the forest was safe as long as she stayed on the path. A twig "
            "snapped behind her and she spun around but saw nothing. Her heart "
            "pounded but she took a deep breath and kept walking. The trees "
            "seemed to whisper secrets to each other in a language she could "
            "almost understand."
        ),
        # Scene 3: Graphic violence (120-180s) — should be flagged
        (
            "Without warning the creature lunged and its claws tore through the "
            "knight's armor like paper. Blood spurted from the deep gashes across "
            "his chest and he screamed in agony. The beast slammed him against "
            "the wall and his skull cracked with a sickening crunch. His entrails "
            "spilled onto the cold stone floor as the creature dismembered him "
            "piece by piece. The other soldiers watched in horror as their "
            "captain was slaughtered before their eyes. The gore splattered "
            "across the walls painting them crimson."
        ),
        # Scene 4: Drug content (180-240s) — should be flagged
        (
            "Marcus retreated to the back room where the others were already "
            "gathered around the table. Someone had laid out lines of cocaine "
            "on a mirror and the smell of crack smoke hung heavy in the air. "
            "He watched as Jake injected heroin into his arm and slumped back "
            "against the wall with glazed eyes. The drug deal had gone wrong "
            "and they needed to figure out their next move before the cops "
            "showed up. Marcus grabbed a bottle of whiskey and took a long "
            "pull trying to steady his nerves."
        ),
        # Scene 5: Clean resolution (240-300s)
        (
            "Emily emerged from the forest into a sunlit clearing filled with "
            "the most beautiful flowers she had ever seen. A crystal clear "
            "stream wound through the center and butterflies danced in the "
            "warm air. She sat down on a mossy rock and opened the old leather "
            "journal her grandfather had given her. Inside she found a map "
            "showing hidden treasures throughout the enchanted land. She "
            "carefully traced the paths with her finger planning her next "
            "expedition. This was going to be the best summer ever."
        ),
        # Scene 6: Profanity test (300-360s)
        (
            "The damn car broke down again right in the middle of nowhere. "
            "Tom kicked the tire and said shit not this again. What "
            "a hell of a day this had turned out to be. He muttered "
            "what the fuck and paced around the car. At least the weather "
            "was nice and the walk back to town would only take about an hour. "
            "He grabbed his backpack and started walking along the dusty road "
            "humming a tune to pass the time."
        ),
    ]

    words = []
    time_cursor = 0.0
    for scene_text in scenes:
        for token in scene_text.split():
            duration = len(token) * 0.04 + 0.08
            words.append(WordSegment(
                word=token,
                start=round(time_cursor, 3),
                end=round(time_cursor + duration, 3),
            ))
            time_cursor += duration + 0.04  # inter-word gap
        time_cursor += 1.5  # scene gap

    return words
