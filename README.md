# DVR Clean and Scan

Automates re-encoding and motion-detection scanning of DVR recordings. Useful for footage from cheap DVRs that produce broken or corrupted video files.

## How it works

1. **Clean** — Re-encodes every video in the source folder using `ffmpeg` (H.264) and writes the result to a `cleaned/` subfolder. Already-processed files are skipped. A GPU encoder is used automatically when available (see table below), falling back to CPU (`libx264`).
2. **Scan** — Runs `dvr-scan` on each cleaned file to extract motion-detected segments into an `output/` subfolder.

### GPU encoder priority

| GPU vendor | Encoder | Hardware acceleration |
|------------|---------|----------------------|
| NVIDIA | `h264_nvenc` | CUDA |
| AMD | `h264_amf` | — |
| Intel | `h264_qsv` | Quick Sync |
| *(fallback)* | `libx264` | CPU only |

The script probes `ffmpeg -encoders` at startup and picks the first available encoder from the list above.

Folder layout after a run:

```
<your-folder>/
├── video1.avi
├── video2.mp4
├── cleaned/
│   ├── cl_video1.avi
│   └── cl_video2.mp4
└── output/
    └── <motion segments>
```

## Requirements

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Runtime |
| [ffmpeg](https://ffmpeg.org/) | Video re-encoding |
| [dvr-scan](https://github.com/Breakthrough/DVR-Scan) | Motion detection |

Install dvr-scan via pip:

```bash
pip install dvr-scan[opencv]
```

## Download (pre-built binaries)

Grab the latest binary from the [Releases](../../releases) page — no Python installation required.

| Platform | File |
|----------|------|
| Windows | `dvr_clean_and_scan-windows-x64.exe` |
| macOS | `dvr_clean_and_scan-macos-x64` |

On macOS, mark the binary as executable after downloading:

```bash
chmod +x dvr_clean_and_scan-macos-x64
```

## Usage

### Interactive wizard (no arguments)

Run the tool without arguments to launch the interactive wizard:

```bash
# From source
python dvr_clean_and_scan.py

# From binary (Windows)
dvr_clean_and_scan.exe

# From binary (macOS)
./dvr_clean_and_scan-macos-x64
```

The wizard offers two modes:

- **Simple** — only asks for the folder path; all other settings use defaults.
- **Advanced** — walks through every parameter with its default value and a plain-text explanation of what it does.

### CLI mode (with arguments)

```bash
# Minimal — process a folder with defaults
python dvr_clean_and_scan.py "C:\Recordings\2026-03-29"

# Full control
python dvr_clean_and_scan.py "C:\Recordings\2026-03-29" \
  --threshold 0.65 \
  --downscale 4 \
  --min-length 1s \
  --kernel 5 \
  --crf 26 \
  --dry-run
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `folder` | *(wizard)* | Root folder of DVR recordings. Omit to launch the wizard. |
| `--dry-run` | off | Print ffmpeg/dvr-scan commands without executing them. |
| `--threshold T` | `0.75` | Motion sensitivity (0.0–1.0). Higher = less sensitive. |
| `--downscale N` | `6` | Frame downscale factor (1–16). Higher = faster, less accurate. |
| `--min-length DUR` | `2s` | Minimum motion event duration (e.g. `500ms`, `3s`). |
| `--kernel K` | `-1` | Blur kernel size (odd integer, or `-1` for auto). |
| `--crf Q` | `30` | Re-encode quality (18–51). Lower = better quality, larger file. |

## Building from source

### Prerequisites

```bash
pip install -r requirements-build.txt
```

### Build locally

```bash
# Windows → produces dist/dvr_clean_and_scan.exe
pyinstaller --onefile --name dvr_clean_and_scan --console dvr_clean_and_scan.py

# macOS/Linux → produces dist/dvr_clean_and_scan
pyinstaller --onefile --name dvr_clean_and_scan --console dvr_clean_and_scan.py
```

### CI / GitHub Actions

Every push to `main` and every pull request automatically builds both binaries via the [`build.yml`](.github/workflows/build.yml) workflow. Pushing a version tag (e.g. `v1.2.0`) additionally creates a GitHub Release with both binaries attached.


