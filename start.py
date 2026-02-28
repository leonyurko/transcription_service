#!/usr/bin/env python3
"""
Fixed launcher for Windows.
Fixes applied vs the original start.py:
  1. Skips Linux-only nvidia-* pip packages on Windows.
  2. Sets a Windows-compatible default for TRANSCRIBE_TMP_DIR.
  3. Auto-installs the ffmpeg binary via winget if it is not on PATH.
Run:  python start.py"""
import importlib.metadata
import os
import re
import site
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

# ── Fix 2: ensure a Windows-compatible tmp dir before config.py loads ─────────
os.environ.setdefault(
    "TRANSCRIBE_TMP_DIR",
    str(Path(tempfile.gettempdir()) / "transcribe_service"),
)


# ── NVIDIA DLL path fix (must run before ctranslate2 / faster-whisper import) ─
def _fix_nvidia_path() -> None:
    if os.name != "nt":
        return
    dirs_to_add: list[str] = []
    all_sites = list(site.getsitepackages()) + [site.getusersitepackages()]
    for site_dir in all_sites:
        nvidia_root = Path(site_dir) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for pkg in nvidia_root.iterdir():
            for sub in ("bin", "lib"):
                dll_dir = pkg / sub
                if dll_dir.is_dir():
                    dirs_to_add.append(str(dll_dir))
    if dirs_to_add:
        os.environ["PATH"] = os.pathsep.join(dirs_to_add) + os.pathsep + os.environ.get("PATH", "")
        for d in dirs_to_add:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
        print(f"[launcher] Registered {len(dirs_to_add)} NVIDIA DLL director{'y' if len(dirs_to_add)==1 else 'ies'} in PATH.")


_fix_nvidia_path()   # must be before ANY faster-whisper / ctranslate2 import


# ── Dependency check ──────────────────────────────────────────────────────────

def _missing_packages(req_file: Path) -> list[str]:
    missing = []
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Fix 1: skip Linux-only nvidia pip packages on Windows — they have no
        # Windows wheels and will cause pip (and this launcher) to abort.
        if os.name == "nt" and line.lower().startswith("nvidia-"):
            continue
        name = re.split(r"[>=<!;\[]", line)[0].strip().replace("-", "_")
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(line)
    return missing


def _check_and_install() -> None:
    req_file = DIR / "requirements.txt"
    if not req_file.exists():
        return
    print("Checking dependencies...")
    missing = _missing_packages(req_file)
    if not missing:
        print("All dependencies satisfied.\n")
        return
    print(f"Installing {len(missing)} missing package(s): {', '.join(missing)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"] + missing,
    )
    if result.returncode != 0:
        print("\n[ERROR] Installation failed. Fix the errors above and try again.")
        sys.exit(1)
    _fix_nvidia_path()
    print("\nDependencies installed.\n")


# ── ffmpeg binary check / auto-install ───────────────────────────────────────

def _ffmpeg_on_path() -> bool:
    """Return True if ffmpeg.exe is reachable on the current PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _refresh_path_from_registry() -> None:
    """Read the Machine + User PATH from the registry and update os.environ."""
    if os.name != "nt":
        return
    try:
        import winreg
        parts: list[str] = []
        for hive, key in [
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER,  r"Environment"),
        ]:
            try:
                with winreg.OpenKey(hive, key) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                    parts.append(val)
            except FileNotFoundError:
                pass
        if parts:
            os.environ["PATH"] = os.pathsep.join(parts) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass  # non-critical


def _check_and_install_ffmpeg() -> None:
    """Ensure the ffmpeg binary is available; install via winget if not."""
    if _ffmpeg_on_path():
        return

    print("[launcher] ffmpeg not found on PATH — attempting install via winget...")

    # Check winget is available
    try:
        subprocess.run(
            ["winget", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "\n[ERROR] winget is not available on this system.\n"
            "        Please install ffmpeg manually from https://ffmpeg.org/download.html\n"
            "        and add its 'bin' folder to your PATH, then restart this script."
        )
        sys.exit(1)

    result = subprocess.run(
        [
            "winget", "install",
            "--id", "Gyan.FFmpeg",
            "-e",
            "--accept-source-agreements",
            "--accept-package-agreements",
        ]
    )
    if result.returncode != 0:
        print(
            "\n[ERROR] winget failed to install ffmpeg.\n"
            "        Please install it manually from https://ffmpeg.org/download.html"
        )
        sys.exit(1)

    # Refresh PATH so the current process can find the new binary immediately
    _refresh_path_from_registry()

    if not _ffmpeg_on_path():
        print(
            "\n[WARNING] ffmpeg was installed but is still not on the active PATH.\n"
            "          Please restart this script (or open a new terminal) so the\n"
            "          updated PATH takes effect."
        )
        sys.exit(0)

    print("[launcher] ffmpeg installed successfully.\n")


# ── Browser opener ────────────────────────────────────────────────────────────

def _wait_and_open(url: str) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/api/v1/health", timeout=1)
            break
        except Exception:
            time.sleep(0.4)
    else:
        print("\n[launcher] Server did not respond within 60 s — open the browser manually.")
        return
    print(f"\n  ✓ Service ready  →  {url}\n")
    webbrowser.open(url)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _check_and_install()
    _check_and_install_ffmpeg()

    from config import settings
    import uvicorn

    url = f"http://{'localhost' if settings.host == '0.0.0.0' else settings.host}:{settings.port}"

    print("=" * 50)
    print("  Transcription Service")
    print(f"  {url}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    threading.Thread(target=_wait_and_open, args=(url,), daemon=True).start()
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
