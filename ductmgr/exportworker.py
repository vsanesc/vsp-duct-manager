# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The process that does one step of an export: ``python -m ductmgr.exportworker JOB RESULT``.

OpenVSP 3.55 sometimes corrupts its heap while it builds a Stack with Edit Curve XSecs and the process dies (exit code
0xC0000374 and friends), depending on the section data. ``exportrun.export_design`` therefore runs every OpenVSP step
(``exportrun.TASKS``) in a process of its own, so that a crash never takes down the caller (the explorer window), and
tries a Geom again with other sections when its process died. JOB is a pickle ``{"task": name, "args": {...}}`` written by
``exportrun``; progress lines go to stdout as ``PROGRESS <text>``; the result is pickled to RESULT.
"""
import os
import pickle
import sys


def no_crash_dialogs() -> None:
    """On Windows, let a crash end the process at once instead of waiting for the error-reporting dialog."""
    if sys.platform == "win32":
        import ctypes

        # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


def main(argv=None) -> int:
    no_crash_dialogs()
    job_path, result_path = (argv or sys.argv[1:])[:2]
    with open(job_path, "rb") as f:
        job = pickle.load(f)
    from ductmgr import exportrun

    result = exportrun.TASKS[job["task"]](**job["args"], progress=lambda text: print(f"PROGRESS {text}", flush=True))
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    sys.stdout.flush()
    # leave without running the interpreter's and OpenVSP's shutdown: that is where a corrupted OpenVSP heap tends to
    # show up, after everything has been written
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
