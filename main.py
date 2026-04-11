#!/usr/bin/env python3
"""
Audiobook Cleaner — CLI entry point.

Usage examples:
    python main.py run -i book.mp3                         # full pipeline
    python main.py run -i book.mp3 --report-only           # analyze without editing
    python main.py run -i book.mp3 -s strict -m remove     # strict filter, cut mode
    python main.py transcribe -i book.mp3                   # transcribe only
    python main.py analyze -t transcript.json               # classify existing transcript
    python main.py clean -i book.mp3 --edl edl.json         # re-apply saved edits
    python main.py dry-run                                  # test with mock data

Batch usage:
    python main.py batch chapter01.mp3 chapter02.mp3 --output-dir cleaned/
    python main.py batch --input-dir ./chapters --pattern "*.mp3" --output-dir cleaned/
    python main.py batch --input-dir ./chapters --join --output-dir cleaned/
    python main.py batch --input-dir ./chapters --report-only
"""

import argparse
import sys
import logging
from pathlib import Path

from audiobook_cleaner.config import AppConfig
from audiobook_cleaner.pipeline import Pipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiobook-cleaner",
        description="Family-friendly audiobook cleaner — detect and remove explicit content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ---- run (full pipeline) ----
    p_run = sub.add_parser("run", help="Full pipeline: transcribe → analyze → edit")
    p_run.add_argument("-i", "--input", required=True, help="Input audiobook file (MP3, M4B, …)")
    p_run.add_argument("-o", "--output", help="Output path (default: <input>_clean.<ext>)")
    p_run.add_argument("-c", "--config", default="config.yaml", help="YAML config file")
    p_run.add_argument("-s", "--sensitivity", choices=["strict", "moderate", "minimal"])
    p_run.add_argument("-m", "--mode", choices=["mute", "remove", "mute_then_remove"],
                       default="mute_then_remove",
                       help="Edit mode: mute (silence), remove (cut), or mute_then_remove (silence profanity, cut classifier hits)")
    p_run.add_argument("--report-only", action="store_true", help="Generate report; skip audio edit")
    p_run.add_argument("-v", "--verbose", action="store_true")

    # ---- transcribe ----
    p_tr = sub.add_parser("transcribe", help="Transcribe audiobook to word-level JSON")
    p_tr.add_argument("-i", "--input", required=True)
    p_tr.add_argument("-o", "--output", help="Transcript output path")
    p_tr.add_argument("-c", "--config", default="config.yaml")
    p_tr.add_argument("-v", "--verbose", action="store_true")

    # ---- analyze ----
    p_an = sub.add_parser("analyze", help="Classify an existing transcript JSON")
    p_an.add_argument("-t", "--transcript", required=True, help="Transcript JSON from 'transcribe'")
    p_an.add_argument("-o", "--output", help="Report output directory")
    p_an.add_argument("-c", "--config", default="config.yaml")
    p_an.add_argument("-s", "--sensitivity", choices=["strict", "moderate", "minimal"])
    p_an.add_argument("-v", "--verbose", action="store_true")

    # ---- clean ----
    p_cl = sub.add_parser("clean", help="Apply an EDL to audio (no re-analysis)")
    p_cl.add_argument("-i", "--input", required=True)
    p_cl.add_argument("--edl", required=True, help="Edit Decision List JSON")
    p_cl.add_argument("-o", "--output")
    p_cl.add_argument("-m", "--mode", choices=["mute", "remove", "mute_then_remove"],
                       help="Edit mode: mute (silence), remove (cut), or mute_then_remove (two-pass)")
    p_cl.add_argument("-c", "--config", default="config.yaml")
    p_cl.add_argument("-v", "--verbose", action="store_true")

    # ---- dry-run ----
    p_dr = sub.add_parser("dry-run", help="Validate pipeline with mock data (no audio, no API)")
    p_dr.add_argument("-c", "--config", default="config.yaml")
    p_dr.add_argument("-s", "--sensitivity", choices=["strict", "moderate", "minimal"])
    p_dr.add_argument("-v", "--verbose", action="store_true")

    # ---- batch ----
    p_ba = sub.add_parser("batch", help="Process multiple audiobook files in batch")
    p_ba.add_argument("files", nargs="*", help="Audio files to process")
    p_ba.add_argument("-i", "--input-dir", help="Directory to glob files from")
    p_ba.add_argument("--pattern", default="*.mp3,*.m4b",
                       help="Comma-separated glob patterns (used with --input-dir)")
    p_ba.add_argument("-o", "--output-dir", help="Output directory")
    p_ba.add_argument("--join", action="store_true",
                       help="Combine transcripts across all files before classifying")
    p_ba.add_argument("--report-only", action="store_true",
                       help="Generate report without writing audio output")
    p_ba.add_argument("-s", "--sensitivity", choices=["strict", "moderate", "minimal"],
                       default="moderate")
    p_ba.add_argument("-m", "--mode", choices=["mute", "remove", "mute_then_remove"],
                       default="mute_then_remove",
                       help="Edit mode: mute (silence), remove (cut), or mute_then_remove (two-pass)")
    p_ba.add_argument("-c", "--config", default="config.yaml", help="Path to config YAML")
    p_ba.add_argument("-v", "--verbose", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    setup_logging(getattr(args, "verbose", False))

    # Load configuration
    config_path = getattr(args, "config", "config.yaml")
    config = AppConfig.from_yaml(config_path)

    # CLI overrides
    if getattr(args, "sensitivity", None):
        config.sensitivity = args.sensitivity
    if getattr(args, "mode", None):
        config.output.mode = args.mode

    pipeline = Pipeline(config)

    try:
        if args.command == "run":
            pipeline.run_full(
                input_path=args.input,
                output_path=args.output,
                report_only=getattr(args, "report_only", False),
            )
        elif args.command == "transcribe":
            pipeline.run_transcribe(args.input, args.output)
        elif args.command == "analyze":
            pipeline.run_analyze(args.transcript, args.output)
        elif args.command == "clean":
            pipeline.run_clean(args.input, args.edl, args.output)
        elif args.command == "dry-run":
            pipeline.run_dry_run()
        elif args.command == "batch":
            # Collect files from positional args and --input-dir
            files = list(args.files) if args.files else []
            if args.input_dir:
                input_dir = Path(args.input_dir)
                for pat in args.pattern.split(","):
                    pat = pat.strip()
                    files.extend(str(p) for p in sorted(input_dir.glob(pat)))
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for f in files:
                key = str(Path(f).resolve())
                if key not in seen:
                    seen.add(key)
                    deduped.append(f)
            files = deduped

            if not files:
                logging.getLogger(__name__).error("No input files found.")
                sys.exit(1)

            pipeline.run_batch(
                files,
                output_dir=args.output_dir,
                join=args.join,
                report_only=args.report_only,
            )
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logging.getLogger(__name__).error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
