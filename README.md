# Audiobook Cleaner

A local-first, family-friendly audiobook processing tool.  Transcribes audiobooks with word-level timestamps, detects profanity and mature content, and produces a cleaned audio file with explicit scenes muted or removed.

---

## Features

| Capability | Details |
|---|---|
| **Transcription** | WhisperX with word-level alignment (runs locally on GPU or CPU) |
| **Profanity filter** | Configurable banned-word list — instant muting at the word level |
| **AI classification** | Chunk-based detection of sexual content, graphic violence, drug use, and mature themes |
| **Review before edit** | CSV + JSON reports and an Edit Decision List (EDL) you can inspect and adjust |
| **Three edit modes** | *Mute* (silence flagged words), *Remove* (cut flagged ranges), or *Mute-then-Remove* (mute banned words + cut AI-flagged scenes in two passes — default) |
| **Tighter AI cuts** | Classifier returns precise segment timestamps within each chunk, so only the offending lines are cut rather than the entire chunk span |
| **Sensitivity presets** | Strict / Moderate / Minimal — each with tunable confidence and severity thresholds |
| **API-agnostic** | Works with OpenAI, LM Studio, Ollama, or any OpenAI-compatible endpoint |

---

## Project Layout

```
audiobook-cleaner/
├── main.py                         # CLI entry point
├── config.yaml                     # Default settings (edit this)
├── banned_words.txt                # Word-level profanity list
├── requirements.txt
├── audiobook_cleaner/
│   ├── config.py                   # Configuration loading
│   ├── transcriber.py              # WhisperX transcription + alignment
│   ├── profanity.py                # Banned-word scanner
│   ├── chunker.py                  # Overlapping chunk builder
│   ├── classifier.py               # AI content classifier + mock classifier
│   ├── merger.py                   # Merge overlapping flagged ranges
│   ├── reporter.py                 # CSV / JSON report generator
│   ├── editor.py                   # FFmpeg audio editor + EDL writer
│   └── pipeline.py                 # End-to-end orchestrator
└── tests/
    └── test_dry_run.py             # Mock-data integration tests
```

---

## Windows Setup (Step by Step)

### 1. Python 3.10+

Download from <https://www.python.org/downloads/>.  During installation, check **"Add Python to PATH"**.

### 2. FFmpeg

1. Download a **release build** from <https://github.com/BtbN/FFmpeg-Builds/releases> (pick `ffmpeg-master-latest-win64-gpl.zip`).
2. Extract the zip.
3. Add the `bin/` folder to your system PATH:
   - Search "Environment Variables" → Edit `Path` → New → paste the full path to `bin/`.
4. Verify: open a new terminal and run `ffmpeg -version`.

### 3. PyTorch (GPU recommended)

Pick the command for your CUDA version (check with `nvidia-smi`):

```powershell
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (much slower for transcription)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. WhisperX

```powershell
pip install whisperx
```

> WhisperX requires a Hugging Face token for the alignment model the first time you run it.  Get one at <https://huggingface.co/settings/tokens> and set it:
> ```powershell
> set HF_TOKEN=hf_your_token_here
> ```

### 5. Project dependencies

```powershell
cd audiobook-cleaner
pip install -r requirements.txt
```

### 6. API key (for AI classification)

Set your key in `config.yaml` or as an environment variable:

```powershell
set OPENAI_API_KEY=sk-...
```

If using a **local model** (LM Studio, Ollama), update the `classification` section in `config.yaml` — see comments there.

---

## Quick Start

### Dry run (no audio, no API — tests the pipeline)

```bash
python main.py dry-run -v
```

Or with pytest:

```bash
python -m pytest tests/test_dry_run.py -v
```

### Full pipeline

```bash
python main.py run -i "My Audiobook.mp3"
```

This will:
1. Transcribe the audiobook (cached for re-runs).
2. Scan for banned words.
3. Chunk the transcript and classify each chunk via AI.
4. Merge flagged ranges.
5. Write a report to `My Audiobook_cleaned/report/`.
6. Produce `My Audiobook_clean.mp3`.

### Report-only (inspect before editing)

```bash
python main.py run -i book.m4b --report-only
```

Review the CSV/JSON reports in the output directory, then apply edits:

```bash
python main.py clean -i book.m4b --edl "book_cleaned/edl.json"
```

### Edit modes

Audiobook Cleaner uses a two-pass strategy by default:

1. **Mute pass** — banned words (from `banned_words.txt`) are silenced at word-level precision. File length is unchanged.
2. **Remove pass** — AI-flagged scenes are cut entirely, working in reverse chronological order so earlier timestamps are not shifted by later cuts.

The AI classifier returns tight `segment_start` / `segment_end` timestamps identifying exactly where within a chunk the objectionable content sits, so cuts are as short as possible.

Override the mode with `-m`:

```bash
python main.py run -i book.m4b -m mute             # banned words only, no cuts
python main.py run -i book.m4b -m remove           # AI cuts only, no muting
python main.py run -i book.m4b -m mute_then_remove # both passes (default)
```

The `clean` subcommand defaults to `mute` for backward compatibility with existing EDLs:

```bash
python main.py clean -i book.m4b --edl book_cleaned/edl.json -m mute_then_remove
```

### Override sensitivity and mode

```bash
python main.py run -i book.mp3 -s strict -m remove
```

---

## CLI Reference

| Command | Purpose |
|---|---|
| `run` | Full pipeline: transcribe → analyze → edit |
| `transcribe` | Transcribe only (saves `transcript.json`) |
| `analyze` | Classify an existing transcript (no audio needed) |
| `clean` | Apply a saved EDL to audio (no re-analysis) |
| `dry-run` | Validate pipeline with mock data |
| `batch` | Process multiple files; use `--join` for cross-boundary detection |

Common flags: `-i INPUT`, `-o OUTPUT`, `-c CONFIG`, `-s SENSITIVITY`, `-m MODE` (`mute` / `remove` / `mute_then_remove`), `-v` (verbose), `--report-only`.

---

## Configuration Guide

All settings live in `config.yaml`.  Key sections:

### Sensitivity

| Preset | Confidence threshold | Minimum severity | Behavior |
|---|---|---|---|
| `strict` | 0.3 | mild | Flags anything remotely questionable |
| `moderate` | 0.5 | moderate | Flags clearly inappropriate content |
| `minimal` | 0.7 | severe | Flags only extremely explicit content |

### Chunking

- `chunk_size`: Words per chunk (default 800).  Larger = fewer API calls but less precise time boundaries.
- `overlap`: Words shared between adjacent chunks (default 200).  Ensures content at boundaries is not missed.

### Classification backend

Switch between providers by editing `classification.api_base`:

| Provider | `api_base` | `api_key` | `model` |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | your key | `gpt-4o-mini` |
| LM Studio | `http://localhost:1234/v1` | `not-needed` | your model name |
| Ollama | `http://localhost:11434/v1` | `not-needed` | your model name |

---

## AI Classification Prompt

The system prompt is embedded in `audiobook_cleaner/classifier.py`.  It instructs the model to act as a **conservative child-safety classifier** that evaluates four categories:

1. **Explicit Sexual Content** — sexual acts, nudity, innuendo
2. **Graphic Violence** — gore, torture, graphic death
3. **Drug / Alcohol Content** — substance use, drug deals, intoxication
4. **Other Mature Themes** — strong profanity, self-harm, hate speech

The prompt asks for a JSON response with:
```json
{
  "contains_explicit_sex": false,
  "contains_graphic_violence": true,
  "contains_drug_content": false,
  "contains_mature_themes": false,
  "severity": "moderate",
  "confidence": 0.85,
  "reason": "Detailed description of battlefield injuries and gore.",
  "segment_start": 142.3,
  "segment_end": 198.7
}
```

`segment_start` and `segment_end` identify the tightest timestamp boundaries within the chunk where the objectionable content occurs. The editor uses these for precise cuts instead of cutting the entire chunk span. Both are `null` if the entire chunk is flagged.

Sensitivity guidance is injected per chunk so the model's threshold shifts with your chosen preset.

> **Design principle:** The prompt is deliberately conservative.  It is better to over-flag (and let you un-flag during review) than to miss inappropriate content for a child audience.

---

## Output Files

After a full run, the output directory contains:

| File | Contents |
|---|---|
| `transcript.json` | Word-level transcript with timestamps |
| `report/chunk_results.csv` | Per-chunk classification detail |
| `report/chunk_results.json` | Same, in JSON |
| `report/flagged_ranges.csv` | Merged time ranges to be edited |
| `report/flagged_ranges.json` | Same, in JSON |
| `report/profanity_hits.json` | Word-level profanity detections |
| `report/summary.json` | Aggregate statistics |
| `edl.json` | Edit Decision List — one entry per flagged range with `action: "mute"` or `action: "remove"`. Reload with `clean` command or edit manually before applying. |
| `*_clean.mp3` | Cleaned audiobook file |

---

## Performance Notes (10+ hour audiobooks)

### Transcription (WhisperX)

| Hardware | Model | ~Speed (real-time factor) | 10-hr book |
|---|---|---|---|
| RTX 3080+ (GPU) | large-v2 | ~10-20× real time | 30–60 min |
| RTX 3080+ (GPU) | medium | ~25-40× | 15–25 min |
| Modern CPU (no GPU) | large-v2 | ~0.5–1× | 10–20 hrs |
| Modern CPU (no GPU) | small | ~2–4× | 2.5–5 hrs |

**Recommendations:**
- Use a GPU if at all possible.  Even a modest GTX 1660 is 5–10× faster than CPU.
- If VRAM is tight (< 6 GB), lower `batch_size` to 4 or switch to `compute_type: int8`.
- Transcription results are **cached** — subsequent runs skip this step.

### AI Classification

| Chunks (800-word) | API calls | gpt-4o-mini cost (est.) |
|---|---|---|
| ~150 (10-hr book) | 150 | ~$0.15–0.30 |
| ~300 (20-hr book) | 300 | ~$0.30–0.60 |

- Costs are minimal because `gpt-4o-mini` input tokens are cheap and each chunk is small.
- With `max_concurrent: 4`, classification of 150 chunks takes ~2–5 minutes.
- **Local models** (LM Studio, Ollama) eliminate API cost entirely but may be slower and less accurate.

### Audio Editing (FFmpeg)

- **Mute mode:** ~1–3 minutes for a 10-hr file (single-pass volume filter).
- **Remove mode:** ~3–8 minutes (requires trim + concat re-encode).
- **Mute-then-Remove mode (default):** two sequential FFmpeg passes; total time is roughly the sum of the above. A temporary intermediate file is written and deleted automatically.
- Bottleneck is re-encoding. The tool probes the source file and matches its bitrate, sample rate, and channel count to avoid unnecessary quality loss or size inflation.

### Memory

- WhisperX alignment loads the audio into memory.  A 10-hr file at 16 kHz mono is ~1.1 GB.
- The transcript and chunk data are lightweight (tens of MB).
- FFmpeg streams audio and does not load the full file into Python memory.

### Recommended Workflow for Very Long Books

1. **Transcribe once**, cache the result:
   ```bash
   python main.py transcribe -i book.m4b
   ```
2. **Iterate on classification** (adjust sensitivity, review reports) without re-transcribing:
   ```bash
   python main.py analyze -t book_cleaned/transcript.json -s strict
   ```
3. **Edit the EDL** manually if needed (it is plain JSON).
4. **Apply edits** when satisfied:
   ```bash
   python main.py clean -i book.m4b --edl book_cleaned/edl.json -m mute_then_remove
   ```

---

## Extending the Project

- **Custom categories:** Add new classification fields in `classifier.py` and update the system prompt.
- **Chapter awareness:** Parse M4B chapter metadata and include chapter info in chunks/reports.
- **Web UI:** Wrap the pipeline in a Flask/FastAPI app for a review dashboard.
- **Batch processing:** Use the built-in `batch` subcommand to process a directory of MP3 or M4B files. Use `--join` to combine transcripts across chapter files for cross-boundary detection.
- **Speaker diarization:** WhisperX supports diarization — add speaker labels to improve classification context.

---

## Caveats

- **M4B chapter metadata** is not preserved in the cleaned output.  Re-add chapters with tools like `mp4chaps` if needed.
- **WhisperX alignment** may occasionally miss or mis-time words, especially with heavy accents or background music.  Review the transcript for critical passages.
- **AI classification is not perfect.**  Always review the report before sharing cleaned audio with children.  The tool is designed as an assistant, not a guarantee.

---

## License

This project is provided as-is for personal use.  No warranty.
