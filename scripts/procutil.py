"""Quiet subprocess launching on Windows.

Console children (ffmpeg / python / node) each get their own black console
window unless CREATE_NO_WINDOW is passed. Use run() everywhere to keep the
pipeline silent.
"""
import os
import subprocess

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def run(cmd, check=True, capture=False, **kw):
    """subprocess.run with the no-console-window flag preset."""
    kw.setdefault("creationflags", NO_WINDOW)
    if capture:
        kw.setdefault("stdout", subprocess.PIPE)
        kw.setdefault("stderr", subprocess.PIPE)
        kw.setdefault("text", True)
    return subprocess.run(cmd, check=check, **kw)
