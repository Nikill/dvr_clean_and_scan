import argparse
import logging
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# dvr-scan Python API (bundled by PyInstaller; subprocess fallback used if unavailable)
try:
    from dvr_scan.scanner import ScanContext as _ScanContext  # type: ignore
    _DVR_SCAN_PYTHON_API = True
except ImportError:  # pragma: no cover
    _DVR_SCAN_PYTHON_API = False

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
    downscale_factor: int = 8
    min_event_length: str = "2s"
    kernel_size: int = -1
    # ffmpeg quality (CRF for libx264, equivalent for GPU encoders)
    crf: int = 30
    # parallel jobs for clean and scan phases
    workers: int = 1


# ---------------------------------------------------------------------------
# ffmpeg resolution — system binary (GPU-capable) → imageio-ffmpeg fallback (CPU)
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> str:
    """Return the path to an ffmpeg binary.

    Priority:
    1. System ffmpeg on PATH — supports GPU encoders (NVENC, AMF, QSV).
    2. imageio-ffmpeg bundled static binary — CPU only, always available.
    """
    system = shutil.which("ffmpeg")
    if system:
        log.info("Using system ffmpeg: %s", system)
        return system
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        bundled = get_ffmpeg_exe()
        log.info("System ffmpeg not found — using bundled imageio-ffmpeg (CPU only): %s", bundled)
        return bundled
    except ImportError:
        log.error(
            "No ffmpeg found. Install ffmpeg (https://ffmpeg.org/download.html) "
            "or add imageio-ffmpeg to your environment."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# GPU encoder detection
# ---------------------------------------------------------------------------

# GPU encoders in priority order: (ffmpeg encoder name, hw accel flag, quality param name, quality value)
_GPU_ENCODERS = [
    ("h264_nvenc",  ["-hwaccel", "cuda"],   "-cq",  None),   # NVIDIA
    ("h264_amf",   [],                      "-qp_i", None),   # AMD
    ("h264_qsv",   ["-hwaccel", "qsv"],    "-global_quality", None),  # Intel
]


def _detect_encoder(ffmpeg_bin: str, crf: int) -> tuple[str, list[str], str, str]:
    """Return the best available H.264 encoder as (encoder, hw_flags, quality_flag, quality_value).

    Checking the encoder list is not sufficient — GPU encoders (e.g. h264_nvenc) are
    compiled into ffmpeg regardless of the hardware present and will appear in the list
    even on machines that cannot use them.  We therefore run a 1-frame probe through
    each candidate and only accept it when the probe succeeds.
    Falls back to libx264 if no GPU encoder is available.
    """
    result = subprocess.run(
        [ffmpeg_bin, "-encoders", "-loglevel", "quiet"],
        capture_output=True, text=True, check=False,
    )
    available = result.stdout
    for encoder, hw_flags, quality_flag, _ in _GPU_ENCODERS:
        if encoder not in available:
            continue
        # Probe: encode a single synthetic frame to verify the encoder actually works
        # on this machine's hardware before committing to it.
        probe = subprocess.run(
            [
                ffmpeg_bin, *hw_flags,
                # 128x128 @ 25 fps: AMD AMF requires at least 128x128 (64x64 triggers
                # AMF_NOT_SUPPORTED on Radeon iGPUs such as the 780M)
                "-f", "lavfi", "-i", "color=s=128x128:r=25",
                "-frames:v", "1",
                "-c:v", encoder,
                "-loglevel", "quiet",
                "-f", "null", "-",
            ],
            capture_output=True, check=False,
        )
        if probe.returncode == 0:
            log.info("GPU encoder verified: %s", encoder)
            return encoder, hw_flags, quality_flag, str(crf)
        log.debug("GPU encoder %s listed but probe failed (no compatible hardware), skipping.", encoder)
    log.info("No working GPU encoder found, falling back to libx264 (CPU)")
    return "libx264", [], "-crf", str(crf)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], dry_run: bool) -> bool:
    """Run a subprocess command and return True on success."""
    log.debug("Running: %s", " ".join(str(c) for c in cmd))
    if dry_run:
        return True
    try:
        result = subprocess.run(cmd, check=False, capture_output=True)
    except FileNotFoundError:
        log.error("Executable not found: %s — ensure it is installed and on PATH.", cmd[0])
        return False
    if result.returncode != 0:
        log.error("Command failed with exit code %d", result.returncode)
        # Surface the last few stderr lines through the logger so tqdm keeps bars stable
        if result.stderr:
            for line in result.stderr.decode(errors="replace").splitlines()[-5:]:
                line = line.strip()
                if line:
                    log.error("  ffmpeg: %s", line)
        return False
    return True


def clean_videos(cfg: Config, cleaned_folder: Path, ffmpeg_bin: str) -> None:
    """Re-encode source videos into the cleaned folder."""
    cleaned_folder.mkdir(parents=True, exist_ok=True)

    # Check for video files *before* spawning the encoder-detection subprocess.
    video_files = [f for f in cfg.folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        log.warning("No video files found in %s", cfg.folder)
        return

    encoder, hw_flags, quality_flag, quality_value = _detect_encoder(ffmpeg_bin, cfg.crf)
    _preset_map = {
        "libx264":    "ultrafast",
        "h264_nvenc": "p1",       # NVENC fastest preset
        "h264_amf":   "speed",    # AMF fastest preset
        "h264_qsv":   "veryfast", # QSV fastest preset
    }
    preset_flags = ["-preset", _preset_map.get(encoder, "ultrafast")]

    def _process(video_file: Path) -> None:
        output_path = cleaned_folder / f"cl_{video_file.name}"
        if output_path.exists():
            log.info("%s already cleaned, skipping.", video_file.name)
            return
        log.info("Cleaning %s -> %s  [encoder: %s]", video_file.name, output_path.name, encoder)
        run_cmd(
            [
                ffmpeg_bin, *hw_flags, "-i", str(video_file),
                "-c:v", encoder, *preset_flags, quality_flag, quality_value,
                "-loglevel", "warning",
                str(output_path),
            ],
            cfg.dry_run,
        )

    with logging_redirect_tqdm():
        bar = tqdm(total=len(video_files), desc="[1/2] Cleaning", unit="file",
                   dynamic_ncols=True, leave=True)
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_process, f): f for f in sorted(video_files)}
            for future in as_completed(futures):
                fname = futures[future].name
                exc = future.exception()
                if exc:
                    bar.write(f"[ERROR] {fname}: {exc}")
                bar.set_postfix_str(fname, refresh=False)
                bar.update(1)
        bar.close()


def _parse_min_length(duration_str: str, fps: float = 25.0) -> int:
    """Convert a CLI duration string ('2s', '500ms') to a frame count at the given fps."""
    s = duration_str.strip().lower()
    if s.endswith("ms"):
        return max(1, round(float(s[:-2]) / 1000.0 * fps))
    if s.endswith("s"):
        return max(1, round(float(s[:-1]) * fps))
    return max(1, int(s))  # already a frame count


def scan_videos(cfg: Config, cleaned_folder: Path, output_folder: Path) -> None:
    """Run dvr-scan motion detection on all cleaned video files.

    Uses the dvr-scan Python API when available (works inside a PyInstaller bundle).
    Falls back to the 'dvr-scan' CLI subprocess when the Python API is unavailable.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in cleaned_folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        log.warning("No cleaned video files found in %s", cleaned_folder)
        return

    min_event_frames = _parse_min_length(cfg.min_event_length)

    if not _DVR_SCAN_PYTHON_API:
        if not shutil.which("dvr-scan"):
            log.error(
                "dvr-scan is not available: Python API import failed and 'dvr-scan' "
                "executable was not found on PATH. "
                "Install it with: pip install dvr-scan[opencv]"
            )
            return

    def _scan_one(video_file: Path) -> None:
        log.info("Scanning %s", video_file.name)
        if cfg.dry_run:
            log.info(
                "[dry-run] dvr-scan -i %s -d %s -t %s -df %s -l %s -k %s",
                video_file, output_folder,
                cfg.threshold, cfg.downscale_factor, cfg.min_event_length, cfg.kernel_size,
            )
            return
        if _DVR_SCAN_PYTHON_API:
            _scan_with_api(video_file, output_folder, cfg, min_event_frames)
        else:
            log.warning("dvr-scan Python API not available — using subprocess fallback.")
            run_cmd(
                [
                    "dvr-scan", "-i", str(video_file),
                    "-d", str(output_folder),
                    "-t", str(cfg.threshold),
                    "-df", str(cfg.downscale_factor),
                    "-l", cfg.min_event_length,
                    "-k", str(cfg.kernel_size),
                ],
                dry_run=False,
            )

    with logging_redirect_tqdm():
        bar = tqdm(total=len(video_files), desc="[2/2] Scanning", unit="file",
                   dynamic_ncols=True, leave=True)

        def _scan_one_tracked(video_file: Path) -> None:
            bar.set_postfix_str(f"→ {video_file.name}", refresh=True)
            _scan_one(video_file)

        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_scan_one_tracked, f): f for f in sorted(video_files)}
            for future in as_completed(futures):
                fname = futures[future].name
                exc = future.exception()
                if exc:
                    bar.write(f"[ERROR] {fname}: {exc}")
                bar.update(1)
                bar.set_postfix_str(f"done: {fname}", refresh=True)
        bar.close()


def _scan_with_api(video_file: Path, output_folder: Path, cfg: Config, min_event_frames: int) -> None:
    """Invoke dvr-scan via its Python API (PyInstaller-compatible)."""
    try:
        scanner = _ScanContext(  # type: ignore[name-defined]
            input_videos=[str(video_file)],
            output_dir=str(output_folder),
        )
        scanner.scan_motion(
            threshold=cfg.threshold,
            min_event_len=min_event_frames,
            downscale_factor=cfg.downscale_factor,
            kernel_size=cfg.kernel_size,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        log.error("dvr-scan Python API error for %s: %s", video_file.name, exc)
        log.error("Tip: check that dvr-scan[opencv] is installed and matches the expected API.")


def run_pipeline(cfg: Config) -> None:
    """Execute the full clean → scan pipeline."""
    if not cfg.folder.is_dir():
        log.error("Folder not found: %s", cfg.folder)
        sys.exit(1)

    ffmpeg_bin = _resolve_ffmpeg()

    cleaned_folder = cfg.folder / "cleaned"
    output_folder = cfg.folder / "output"

    tqdm.write("\n=== Step 1/2: Cleaning videos (GPU auto-detect) ===")
    clean_videos(cfg, cleaned_folder, ffmpeg_bin)

    tqdm.write("\n=== Step 2/2: Scanning for motion ===")
    scan_videos(cfg, cleaned_folder, output_folder)

    tqdm.write("\nDone.")


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


_RED = "\033[31m"


def _prompt(label: str, default: str, hint: str = "") -> str:
    """Single-line prompt with default value shown."""
    hint_str = f"  {_DIM}{hint}{_RESET}" if hint else ""
    answer = input(f"  {label} [{_GREEN}{default}{_RESET}]{hint_str}: ").strip()
    return answer if answer else default


def _prompt_float(label: str, default: float, lo: float, hi: float) -> float:
    """Prompt for a float in [lo, hi], re-asking on bad input."""
    default_str = str(default)
    while True:
        raw = input(f"  {label} [{_GREEN}{default_str}{_RESET}]  ({lo}–{hi}): ").strip()
        val_str = raw if raw else default_str
        try:
            val = float(val_str)
            if lo <= val <= hi:
                return val
            print(f"  {_RED}✗  Must be between {lo} and {hi}.{_RESET}")
        except ValueError:
            print(f"  {_RED}✗  Please enter a number (e.g. 0.75).{_RESET}")


def _prompt_int(label: str, default: int, lo: int, hi: int) -> int:
    """Prompt for an integer in [lo, hi], re-asking on bad input."""
    default_str = str(default)
    while True:
        raw = input(f"  {label} [{_GREEN}{default_str}{_RESET}]  ({lo}–{hi}): ").strip()
        val_str = raw if raw else default_str
        try:
            val = int(val_str)
            if lo <= val <= hi:
                return val
            print(f"  {_RED}✗  Must be between {lo} and {hi}.{_RESET}")
        except ValueError:
            print(f"  {_RED}✗  Please enter a whole number.{_RESET}")


def _prompt_kernel() -> int:
    """Prompt for kernel size: -1 (auto) or a positive odd integer."""
    while True:
        raw = input(f"  Blur kernel size [{_GREEN}-1{_RESET}]  (-1 for auto, or odd int ≥ 3): ").strip()
        val_str = raw if raw else "-1"
        try:
            val = int(val_str)
            if val == -1:
                return val
            if val >= 3 and val % 2 == 1:
                return val
            if val % 2 == 0:
                print(f"  {_RED}✗  Must be odd (e.g. 3, 5, 7) or -1 for auto.{_RESET}")
            else:
                print(f"  {_RED}✗  Must be -1 (auto) or an odd integer ≥ 3.{_RESET}")
        except ValueError:
            print(f"  {_RED}✗  Please enter -1 or an odd whole number (e.g. 5).{_RESET}")


def _prompt_duration(label: str, default: str) -> str:
    """Prompt for a duration string like '2s' or '500ms', re-asking on bad input."""
    import re
    _DURATION_RE = re.compile(r"^\d+(\.\d+)?(ms|s)$")
    while True:
        raw = input(f"  {label} [{_GREEN}{default}{_RESET}]  (e.g. 2s, 500ms): ").strip()
        val = raw if raw else default
        if _DURATION_RE.match(val):
            return val
        print(f"  {_RED}✗  Enter a duration like '2s' or '500ms'.{_RESET}")


def _prompt_folder(default: str) -> Path:
    """Prompt for a folder path, re-asking until it exists."""
    while True:
        raw = input(f"  Recordings folder [{_GREEN}{default}{_RESET}]: ").strip()
        p = Path(raw).resolve() if raw else Path(default).resolve()
        if p.is_dir():
            return p
        print(f"  {_RED}✗  Folder not found: {p}{_RESET}")
        print(f"  {_DIM}   Create it first, or enter a different path.{_RESET}")


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
    folder = _prompt_folder(str(Path.cwd()))
    dry_run = _confirm("Dry run (preview commands only)?", default=False)
    return Config(folder=folder, dry_run=dry_run)


_PARAM_HINTS = {
    "threshold":         ("Motion threshold",      "0.75",  "float 0.0–1.0"),
    "downscale_factor":  ("Downscale factor",      "8",     "int 1–16"),
    "min_event_length":  ("Min event length",      "2s",    "e.g. 1s, 500ms"),
    "kernel_size":       ("Blur kernel size",      "-1",    "odd int or -1 for auto"),
    "crf":               ("Quality / CRF",         "30",    "int 18–51"),
    "workers":           ("Parallel workers",      "1",     "int 1–N"),
}

_D = _DIM  # shorthand

def _explain(lines: list[str]) -> None:
    """Print dim explanation lines before a prompt."""
    for line in lines:
        print(f"    {_D}{line}{_RESET}")


def _wizard_advanced() -> Config:
    print(f"\n  {_BOLD}Advanced mode{_RESET} — configure every parameter.\n")

    folder = _prompt_folder(str(Path.cwd()))
    dry_run = _confirm("Dry run (preview commands only)?", default=False)

    # ── dvr-scan parameters ──────────────────────────────────────────────
    print(f"\n  {_YELLOW}━━  Motion detection (dvr-scan)  ━━{_RESET}\n")

    _explain([
        "How sensitive the detector is to pixel changes between frames.",
        "↑ higher  →  less sensitive, fewer false positives (shadows, compression noise)",
        "↓ lower   →  more sensitive, catches subtle motion but more noise",
        "Typical range: 0.5 (sensitive) – 0.9 (strict)  |  recommended: 0.75",
    ])
    threshold = _prompt_float("Motion threshold", 0.75, 0.0, 1.0)

    print()
    _explain([
        "Divides frame resolution before analysis  (e.g. 8 → 1080p becomes ~135p).",
        "↑ higher  →  much faster scan, uses less CPU, slightly less precise",
        "↓ lower   →  more accurate detection on small/distant motion",
        "Best speed knob: 6–10 for most DVR footage  |  recommended: 8",
    ])
    downscale_factor = _prompt_int("Downscale factor", 8, 1, 16)

    print()
    _explain([
        "Minimum duration a motion event must last to be kept.",
        "↑ longer  →  skips short bursts (compression artefacts, insects, leaves)",
        "↓ shorter →  catches brief events like a quick hand gesture",
        "Recommended: 2s for most cameras  |  try 1s for fast-moving scenes",
    ])
    min_event_length = _prompt_duration("Min event length", "2s")

    print()
    _explain([
        "Size of the Gaussian blur kernel applied before frame comparison.",
        "-1 (auto)  →  kernel scales automatically with the downscaled resolution",
        "Larger odd values (e.g. 7, 9)  →  smoother diff, ignores fine grain/noise",
        "Recommended: -1 (auto) works well in most cases",
    ])
    kernel_size = _prompt_kernel()

    # ── ffmpeg parameters ────────────────────────────────────────────────
    print(f"\n  {_YELLOW}━━  Video re-encoding (ffmpeg)  ━━{_RESET}\n")

    _explain([
        "CRF (Constant Rate Factor) controls output quality vs. file size.",
        "↓ lower   →  better quality, larger file  (18 = near-lossless)",
        "↑ higher  →  smaller file, more compression artefacts",
        "For surveillance footage 28–32 is usually fine  |  recommended: 30",
    ])
    crf = _prompt_int("Quality / CRF", 30, 18, 51)

    # ── performance ──────────────────────────────────────────────────────
    print(f"\n  {_YELLOW}━━  Performance  ━━{_RESET}\n")

    _explain([
        "Number of files processed in parallel for both clean and scan phases.",
        "↑ more workers  →  faster overall, but more CPU/memory pressure",
        "Set to the number of physical CPU cores for best throughput",
        "Start with 2–4; going beyond core count rarely helps",
    ])
    workers = _prompt_int("Parallel workers", 1, 1, 64)

    return Config(
        folder=folder,
        dry_run=dry_run,
        threshold=threshold,
        downscale_factor=downscale_factor,
        min_event_length=min_event_length,
        kernel_size=kernel_size,
        crf=crf,
        workers=workers,
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
    print(f"    Workers          : {cfg.workers}")
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
    parser.add_argument("--workers",      type=int,   default=1,    metavar="N",
                        help="Parallel jobs for clean and scan phases. Default: 1")
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
            workers=args.workers,
        )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
