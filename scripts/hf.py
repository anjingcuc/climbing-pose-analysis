"""Silently run the hyperframes CLI (one hidden console, zero window flashes).

Why: the Bash tool starts processes without a console, so every console
child spawned by npx/node (render workers, ffmpeg, chrome helpers) gets its
own flashing black window. Running the whole tree under one HIDDEN console
makes all children inherit it silently.

Usage:
  python hf.py --cwd <project> [--log <file>] -- <hyperframes args...>
  e.g. python hf.py --cwd overlay_v2 -- check
       python hf.py --cwd overlay_v2 -- render
"""
import argparse
import subprocess
import sys
from pathlib import Path

from procutil import NO_WINDOW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--tail", type=int, default=60, help="log lines to print")
    ap.add_argument("args", nargs=argparse.REMAINDER,
                    help="hyperframes args after --")
    a = ap.parse_args()
    hf_args = [x for x in a.args if x != "--"] or ["check"]

    cwd = Path(a.cwd).resolve()
    log = cwd / "hf_last.log"
    batch = cwd / ".hf_run.cmd"
    inner = subprocess.list2cmdline(
        ["npx", "--yes", "hyperframes@0.8.6"] + hf_args)
    batch.write_text(f'@{inner} > "{log}" 2>&1\n', encoding="utf-8")

    ps = (
        f"$p = Start-Process -FilePath 'cmd.exe' "
        f"-ArgumentList '/c','{batch}' "
        f"-WorkingDirectory '{cwd}' -WindowStyle Hidden -PassThru -Wait; "
        f"exit $p.ExitCode"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       creationflags=NO_WINDOW)
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-a.tail:]))
    batch.unlink(missing_ok=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
