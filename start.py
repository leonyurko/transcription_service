#!/usr/bin/env python3
"""
Launch the Transcription Service and open the UI automatically.
Run:  python start.py
"""
import importlib.metadata
import os
import re
import site
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))


# ── NVIDIA DLL path fix (must run before ctranslate2 / faster-whisper import) ─
# ctranslate2 loads cublas/cudnn via Windows LoadLibrary which searches PATH.
# The nvidia-* pip packages install DLLs in user site-packages, outside PATH.

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
        # Prepend to PATH so Windows LoadLibrary finds them first
        os.environ["PATH"] = os.pathsep.join(dirs_to_add) + os.pathsep + os.environ.get("PATH", "")
        # Also register with Python's DLL loader as a belt-and-suspenders measure
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
    # Re-run the path fix in case the install just added new nvidia packages
    _fix_nvidia_path()
    print("\nDependencies installed.\n")


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
