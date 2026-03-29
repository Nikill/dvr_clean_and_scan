"""Quick diagnostic script to check GPU encoder detection on this machine."""

import ctypes
import os
import shutil
import subprocess
import sys
import winreg

# ---------------------------------------------------------------------------
# Resolve ffmpeg the same way the main script does
# ---------------------------------------------------------------------------

def resolve_ffmpeg() -> str:
    system = shutil.which("ffmpeg")
    if system:
        print(f"[ffmpeg] Found system ffmpeg: {system}")
        return system
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        bundled = get_ffmpeg_exe()
        print(f"[ffmpeg] System ffmpeg not found — using bundled imageio-ffmpeg: {bundled}")
        return bundled
    except ImportError:
        print("[ffmpeg] ERROR: no ffmpeg found (neither system nor imageio-ffmpeg).")
        sys.exit(1)


# ---------------------------------------------------------------------------
# GPU probe — mirrors _detect_encoder() from dvr_clean_and_scan.py
# ---------------------------------------------------------------------------

_GPU_ENCODERS = [
    ("h264_nvenc",  ["-hwaccel", "cuda"],        "-cq",             "NVIDIA NVENC"),
    ("h264_amf",    [],                           "-qp_i",           "AMD AMF"),
    ("h264_qsv",    ["-hwaccel", "qsv"],          "-global_quality", "Intel QSV"),
]

# ---------------------------------------------------------------------------
# AMD-specific diagnostics
# ---------------------------------------------------------------------------

def check_amd_runtime() -> None:
    """Check for AMF runtime DLL and AMD GPU presence via WMI/registry."""
    print("\n--- AMD-specific diagnostics ---")

    # 1. Check for amfrt64.dll (AMF runtime — required by h264_amf)
    search_dirs = [
        os.environ.get("SystemRoot", r"C:\Windows") + r"\System32",
        os.environ.get("SystemRoot", r"C:\Windows") + r"\SysWOW64",
    ]
    amf_found = False
    for d in search_dirs:
        dll = os.path.join(d, "amfrt64.dll")
        if os.path.exists(dll):
            print(f"  amfrt64.dll found: {dll}")
            amf_found = True
    if not amf_found:
        print("  amfrt64.dll NOT FOUND in System32/SysWOW64.")
        print("  => AMD AMF runtime is missing. Install/update AMD drivers from:")
        print("     https://www.amd.com/en/support/download/drivers.html")

    # 2. Check session type (RDP / console)
    session_name = os.environ.get("SESSIONNAME", "unknown")
    print(f"\n  Session type: {session_name}")
    if session_name.lower().startswith("rdp"):
        print("  WARNING: RDP session detected — AMD AMF is NOT supported over RDP.")
        print("  You must run the script locally (on the physical machine) for AMF to work.")

    # 3. List AMD GPU(s) from registry
    print("\n  AMD GPU(s) detected via registry:")
    amd_found = False
    try:
        key_path = r"SYSTEM\CurrentControlSet\Enum\PCI"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as pci_key:
            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(pci_key, i)
                    i += 1
                    # AMD vendor ID is 1002
                    if "VEN_1002" not in subkey_name.upper():
                        continue
                    with winreg.OpenKey(pci_key, subkey_name) as dev_key:
                        j = 0
                        while True:
                            try:
                                instance = winreg.EnumKey(dev_key, j)
                                j += 1
                                with winreg.OpenKey(dev_key, instance) as inst_key:
                                    try:
                                        desc, _ = winreg.QueryValueEx(inst_key, "DeviceDesc")
                                        print(f"    {desc} [{subkey_name}]")
                                        amd_found = True
                                    except FileNotFoundError:
                                        pass
                            except OSError:
                                break
                except OSError:
                    break
    except Exception as e:
        print(f"    Registry read error: {e}")

    if not amd_found:
        print("    No AMD GPU found in PCI registry (VEN_1002).")

    # 4. Quick WMI GPU list
    print("\n  GPU(s) via WMI (Win32_VideoController):")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-WmiObject Win32_VideoController | Select-Object Name,DriverVersion,Status | Format-List"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
    except Exception as e:
        print(f"    WMI query failed: {e}")


def probe_gpu_encoders(ffmpeg_bin: str) -> None:
    print("\n--- Available encoders (grepping GPU candidates) ---")
    result = subprocess.run(
        [ffmpeg_bin, "-encoders", "-loglevel", "quiet"],
        capture_output=True, text=True, check=False,
    )
    available = result.stdout

    for encoder, _, _, label in _GPU_ENCODERS:
        listed = encoder in available
        print(f"  {label:20s} ({encoder:15s}) — listed in ffmpeg: {'YES' if listed else 'NO'}")

    print("\n--- Probing each GPU encoder with a 1-frame test ---")
    working = []
    for encoder, hw_flags, _, label in _GPU_ENCODERS:
        if encoder not in available:
            print(f"  [{label}] SKIP — not listed in this ffmpeg build")
            continue

        cmd = [
            ffmpeg_bin, *hw_flags,
            # 128x128 min: AMD AMF (h264_amf) fails on resolutions below 128x128
            "-f", "lavfi", "-i", "color=s=128x128:r=25",
            "-frames:v", "1",
            "-c:v", encoder,
            "-loglevel", "error",   # show errors so we can diagnose failures
            "-f", "null", "-",
        ]
        print(f"\n  [{label}] Running probe...")
        print(f"    cmd: {' '.join(cmd)}")
        probe = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if probe.returncode == 0:
            print(f"    RESULT: SUCCESS — {label} is available!")
            working.append((label, encoder))
        else:
            print(f"    RESULT: FAILED (exit code {probe.returncode})")
            if probe.stderr.strip():
                for line in probe.stderr.strip().splitlines():
                    print(f"      stderr: {line}")

    print("\n--- Summary ---")
    if working:
        print(f"  Working GPU encoder(s): {', '.join(f'{lbl} ({enc})' for lbl, enc in working)}")
        print(f"  The main script will use: {working[0][1]}")
    else:
        print("  No GPU encoder worked — the main script will fall back to libx264 (CPU).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ffmpeg_bin = resolve_ffmpeg()
    check_amd_runtime()
    probe_gpu_encoders(ffmpeg_bin)
