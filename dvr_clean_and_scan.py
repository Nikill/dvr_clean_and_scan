import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass — single source of truth for all tunable parameters
# ---------------------------------------------------------------------------

@dataclass
class Config:
    folder: Path
    dry_run: bool = False
    # dvr-scan parameters
    threshold: float = 0.75
    downscale_factor: int = 6
    min_event_length: str = "2s"
    kernel_size: int = -1
    # ffmpeg quality (CRF for libx264, equivalent for GPU encoders)
    crf: int = 30


# ---------------------------------------------------------------------------
# GPU encoder detection
# ---------------------------------------------------------------------------

# GPU encoders in priority order: (ffmpeg encoder name, hw accel flag, quality param name, quality value)
_GPU_ENCODERS = [
    ("h264_nvenc",  ["-hwaccel", "cuda"],   "-cq",  None),   # NVIDIA
    ("h264_amf",   [],                      "-qp_i", None),   # AMD
    ("h264_qsv",   ["-hwaccel", "qsv"],    "-global_quality", None),  # Intel
]


def _detect_encoder(crf: int) -> tuple[str, list[str], str, str]:
    """Return the best available H.264 encoder as (encoder, hw_flags, quality_flag, quality_value).
    Falls back to libx264 if no GPU encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders", "-loglevel", "quiet"],
            capture_output=True, text=True, check=False,
        )
        available = result.stdout
        for encoder, hw_flags, quality_flag, _ in _GPU_ENCODERS:
            if encoder in available:
                log.info("GPU encoder detected: %s", encoder)
                return encoder, hw_flags, quality_flag, str(crf)
    except FileNotFoundError:
        log.error("ffmpeg not found in PATH")
        sys.exit(1)
    log.info("No GPU encoder found, using libx264 (CPU)")
    return "libx264", [], "-crf", str(crf)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], dry_run: bool) -> bool:
    """Run a subprocess command and return True on success."""
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    if dry_run:
        return True
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("Command failed with exit code %d", result.returncode)
        return False
    return True


def clean_videos(cfg: Config, cleaned_folder: Path) -> None:
    """Re-encode source videos into the cleaned folder."""
    cleaned_folder.mkdir(parents=True, exist_ok=True)

    encoder, hw_flags, quality_flag, quality_value = _detect_encoder(cfg.crf)
    preset_flags = ["-preset", "ultrafast"] if encoder == "libx264" else ["-preset", "p1"]

    video_files = [f for f in cfg.folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        log.warning("No video files found in %s", cfg.folder)
        return

    for video_file in sorted(video_files):
        output_path = cleaned_folder / f"cl_{video_file.name}"
        if output_path.exists():
            log.info("%s already cleaned, skipping.", video_file.name)
            continue

        log.info("Cleaning %s -> %s  [encoder: %s]", video_file.name, output_path.name, encoder)
        run_cmd(
            [
                "ffmpeg", *hw_flags, "-i", str(video_file),
                "-c:v", encoder, *preset_flags, quality_flag, quality_value,
                "-loglevel", "warning",
                str(output_path),
            ],
            cfg.dry_run,
        )


def scan_videos(cfg: Config, cleaned_folder: Path, output_folder: Path) -> None:
    """Run dvr-scan on all cleaned video files."""
    output_folder.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in cleaned_folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        log.warning("No cleaned video files found in %s", cleaned_folder)
        return

    for video_file in sorted(video_files):
        log.info("Scanning %s", video_file.name)
        run_cmd(
            [
                "dvr-scan", "-i", str(video_file),
                "-d", str(output_folder),
                "-t", str(cfg.threshold),
                "-df", str(cfg.downscale_factor),
                "-l", cfg.min_event_length,
                "-k", str(cfg.kernel_size),
            ],
            cfg.dry_run,
        )


def run_pipeline(cfg: Config) -> None:
    """Execute the full clean → scan pipeline."""
    if not cfg.folder.is_dir():
        log.error("Folder not found: %s", cfg.folder)
        sys.exit(1)

    cleaned_folder = cfg.folder / "cleaned"
    output_folder = cfg.folder / "output"

    log.info("=== Step 1/2: Cleaning videos (GPU auto-detect) ===")
    clean_videos(cfg, cleaned_folder)

    log.info("=== Step 2/2: Scanning for motion ===")
    scan_videos(cfg, cleaned_folder, output_folder)

    log.info("Done.")


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_DIM    = "\033[2m"


def _h(text: str) -> str:
    """Highlight text."""
    return f"{_BOLD}{_CYAN}{text}{_RESET}"


def _prompt(label: str, default: str, hint: str = "") -> str:
    """Single-line prompt with default value shown."""
    hint_str = f"  {_DIM}{hint}{_RESET}" if hint else ""
    answer = input(f"  {label} [{_GREEN}{default}{_RESET}]{hint_str}: ").strip()
    return answer if answer else default


def _confirm(label: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    answer = input(f"  {label} [{_GREEN}{default_str}{_RESET}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _print_banner() -> None:
    print()
    print(_h("╔══════════════════════════════════════╗"))
    print(_h("║      DVR Clean & Scan  — Wizard      ║"))
    print(_h("╚══════════════════════════════════════╝"))
    print()


def _wizard_simple() -> Config:
    print(f"\n  {_BOLD}Simple mode{_RESET} — runs with recommended defaults.\n")
    raw = _prompt("Recordings folder", str(Path.cwd()))
    folder = Path(raw).resolve()
    dry_run = _confirm("Dry run (preview commands only)?", default=False)
    return Config(folder=folder, dry_run=dry_run)


_PARAM_HINTS = {
    "threshold":         ("Motion threshold",      "0.75",  "float 0.0–1.0 · higher = less sensitive · default 0.75"),
    "downscale_factor":  ("Downscale factor",       "6",     "int 1–16 · higher = faster but less accurate · default 6"),
    "min_event_length":  ("Min event length",       "2s",    "duration string e.g. 1s, 500ms · default 2s"),
    "kernel_size":       ("Blur kernel size",       "-1",    "odd int or -1 for auto (adapts to resolution) · default -1"),
    "crf":               ("Quality / CRF",          "30",    "int 18–51 · lower = better quality & larger file · default 30"),
}


def _wizard_advanced() -> Config:
    print(f"\n  {_BOLD}Advanced mode{_RESET} — configure every parameter.\n")

    raw = _prompt("Recordings folder", str(Path.cwd()))
    folder = Path(raw).resolve()
    dry_run = _confirm("Dry run (preview commands only)?", default=False)

    print(f"\n  {_YELLOW}━━  dvr-scan parameters  ━━{_RESET}")
    threshold        = float(_prompt(*_PARAM_HINTS["threshold"][:2],        _PARAM_HINTS["threshold"][2]))
    downscale_factor = int(_prompt(*_PARAM_HINTS["downscale_factor"][:2],   _PARAM_HINTS["downscale_factor"][2]))
    min_event_length = _prompt(*_PARAM_HINTS["min_event_length"][:2],        _PARAM_HINTS["min_event_length"][2])
    kernel_size      = int(_prompt(*_PARAM_HINTS["kernel_size"][:2],         _PARAM_HINTS["kernel_size"][2]))

    print(f"\n  {_YELLOW}━━  ffmpeg parameters  ━━{_RESET}")
    crf = int(_prompt(*_PARAM_HINTS["crf"][:2], _PARAM_HINTS["crf"][2]))

    return Config(
        folder=folder,
        dry_run=dry_run,
        threshold=threshold,
        downscale_factor=downscale_factor,
        min_event_length=min_event_length,
        kernel_size=kernel_size,
        crf=crf,
    )


def _wizard() -> Config:
    _print_banner()
    print("  No folder supplied — launching interactive wizard.")
    print("  Select mode:\n")
    print(f"    {_GREEN}1{_RESET}  Simple   — recommended defaults, just pick a folder")
    print(f"    {_GREEN}2{_RESET}  Advanced — tune every parameter\n")

    while True:
        choice = input("  Choice [1/2]: ").strip()
        if choice in ("", "1"):
            return _wizard_simple()
        if choice == "2":
            return _wizard_advanced()
        print("  Please enter 1 or 2.")


def _print_summary(cfg: Config) -> None:
    print(f"\n  {_BOLD}Configuration summary:{_RESET}")
    print(f"    Folder           : {cfg.folder}")
    print(f"    Dry run          : {cfg.dry_run}")
    print(f"    Threshold        : {cfg.threshold}")
    print(f"    Downscale factor : {cfg.downscale_factor}")
    print(f"    Min event length : {cfg.min_event_length}")
    print(f"    Kernel size      : {cfg.kernel_size}")
    print(f"    CRF / quality    : {cfg.crf}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dvr_clean_and_scan",
        description=(
            "Re-encode DVR recordings and extract motion-detected segments.\n"
            "Run without arguments to launch the interactive wizard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder", nargs="?", default=None,
        help="Root folder containing the DVR recordings. Omit to launch the wizard.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    # dvr-scan tunables
    parser.add_argument("--threshold",    type=float, default=0.75, metavar="T",
                        help="Motion detection threshold (0.0–1.0). Higher = less sensitive. Default: 0.75")
    parser.add_argument("--downscale",    type=int,   default=6,    metavar="N",
                        help="Downscale factor (1–16). Higher = faster, less accurate. Default: 6")
    parser.add_argument("--min-length",   type=str,   default="2s", metavar="DUR",
                        help="Minimum motion event length, e.g. 1s, 500ms. Default: 2s")
    parser.add_argument("--kernel",       type=int,   default=-1,   metavar="K",
                        help="Blur kernel size (odd int or -1 for auto). Default: -1")
    # ffmpeg tunables
    parser.add_argument("--crf",          type=int,   default=30,   metavar="Q",
                        help="Re-encode quality (18–51). Lower = better quality. Default: 30")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.folder is None:
        # No folder supplied — enter interactive wizard
        try:
            cfg = _wizard()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)
        _print_summary(cfg)
        if not _confirm("Proceed?", default=True):
            print("Aborted.")
            sys.exit(0)
    else:
        folder = Path(args.folder).resolve()
        cfg = Config(
            folder=folder,
            dry_run=args.dry_run,
            threshold=args.threshold,
            downscale_factor=args.downscale,
            min_event_length=args.min_length,
            kernel_size=args.kernel,
            crf=args.crf,
        )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
