# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Phase 4: write the duct into a copy of the user's model and verify it.

Opens the input .vsp3, deletes only the Geoms whose name starts with the configured prefix (``model.duct_prefix``),
creates the duct Stacks (``vspexport`` plans them, ``vspio`` writes them), saves the model as ``<name>_ducts.vsp3`` and
reopens that file to run the checks of ``exportcheck``. The input file is never written.

``export_design`` does the OpenVSP work in child processes (``exportworker``). OpenVSP 3.55 sometimes corrupts its heap
while it builds a Stack with Edit Curve XSecs and the process dies; whether it does depends on the exact curves it is
given (some sets of sections crash it every time, others never), and it must not take the caller down. So every step is a
process of its own, chained through the model file: delete the old ``DUCT_*`` Geoms, then one process per Geom, then the
verification. A Geom whose process dies is made again with other sections (another number of Bezier segments and station
tolerance, ``RETRY_VARIANTS``) and tried again. Children run one after the other (two OpenVSP processes at once crash each
other). A crashed child can be kept around by the system for minutes, so one that neither prints nor uses any CPU for
``CHILD_IDLE`` seconds, or stays too long in one step, is taken for dead.
"""
import os
import pickle
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ductmgr import duct as dm
from ductmgr import exportcheck, vspexport, vspio
from ductmgr.config import skin_names_from_config
from ductmgr.solids import Solid, UnionSolid

OUTPUT_SUFFIX = "_ducts"
# per attempt at a Geom: (added to export.cedit_anchors, factor on export.station_tolerance, added to export.cedit_start)
RETRY_VARIANTS = ((0, 1.0, 0), (0, 1.0, 1), (0, 1.0, 2), (1, 1.25, 0), (3, 0.8, 1), (5, 1.6, 2), (8, 0.6, 0), (2, 2.0, 1),
                  (4, 0.5, 2), (6, 3.0, 0))
CHILD_IDLE = 20  # s without output and without CPU time: the child is taken for dead
STEP_LIMITS = {"prepare": 90, "write": 120, "verify": 1800}  # s a task of that kind may take at most (OpenVSP can spin)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PROCESS_ENV = "DUCTMGR_EXPORT_IN_PROCESS"  # set to 1 to run the whole export in this process (debugging)
CRASH_CODES = (3221226356, 3221225477, 3221226505)  # heap corruption, access violation, stack buffer overrun


@dataclass
class ExportResult:
    out_path: Path
    plan: vspexport.ExportPlan
    removed: list[str]
    verification: exportcheck.Verification | None
    lines: list[str] = field(default_factory=list)  # the report, ready to print or show

    @property
    def ok(self) -> bool:
        return self.verification is None or not self.verification.failed

    @property
    def warned(self) -> bool:
        return self.verification is not None and self.verification.warned

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def output_path(model_path) -> Path:
    """``<name>_ducts.vsp3`` next to the input model."""
    p = Path(model_path)
    return p.with_name(f"{p.stem}{OUTPUT_SUFFIX}{p.suffix}")


def prefix_from_config(cfg: dict) -> str:
    prefix = (cfg.get("model") or {}).get("duct_prefix") or vspexport.DEFAULT_PREFIX
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("model.duct_prefix must be a non-empty text: it is what marks the Geoms this program may touch")
    return prefix


class OpenVSPCrash(RuntimeError):
    """OpenVSP's process died; the message says in which Geom and XSec."""


def resolve_paths(model_path, out_path) -> tuple[Path, Path]:
    model_path = Path(model_path).resolve()
    out_path = Path(out_path).resolve() if out_path is not None else output_path(model_path).resolve()
    if out_path == model_path:
        raise ValueError(f"refusing to write the input model {model_path.name}: the output must be a different file")
    return model_path, out_path


def report_lines(out_path: Path, model_path: Path, removed: list[str], plan: vspexport.ExportPlan, design: dm.Duct,
                 prefix: str, verification: exportcheck.Verification | None, crashes: dict | None = None) -> list[str]:
    """The report of an export: what was written, per Geom, and the verification."""
    lines = [f"Wrote {out_path}", f"  input {model_path.name} was not modified"]
    if removed:
        lines.append(f"  replaced {len(removed)} existing {prefix}* Geom(s): {', '.join(removed)}")
    for g in plan.geoms:
        lines.append(f"  {g.name}: {g.summary}")
        for note in (crashes or {}).get(g.name, []):
            lines.append(f"      WARN {note}" if note.startswith("OpenVSP crashed") else
                         f"      WARN the sections of this Geom differ from the first plan: {note}")
        if g.prism_depth is not None and g.negative:
            lines.append(f"      prismatic start {g.prism_depth * 1000:.0f} mm after the opening plane, lip "
                         f"{design.spec.lip_extension * 1000:.0f} mm upstream")
        lines += [f"      {n}" for n in g.notes]
    if verification is None:
        lines.append("Verification skipped (export.verify is off)")
    else:
        lines.append("Verification of the saved file:")
        lines += verification.lines()
        lines.append("  => " + ("FAIL" if verification.failed else "PASS with warnings" if verification.warned else "PASS"))
    return lines


# ---- the steps (each runs in a process of its own, or all together in process) ----------------------------------------------
def task_prepare(model_path, out_file, prefix, progress=None) -> dict:
    """Open the input model, delete the Geoms that start with ``prefix`` and save the rest to ``out_file``. Returns the
    parameter hashes and names of the Geoms that stay (to check later that nothing else changed) and the names removed."""
    say = progress or (lambda text: None)
    say(f"Opening {Path(model_path).name}")
    vspio.open_model(model_path)
    before = vspio.param_hashes(prefix)
    names = {gid: vspio.vsp.GetGeomName(gid) for gid in before}
    removed = vspio.delete_geoms_with_prefix(prefix)
    vspio.save_model(out_file)
    return {"before": before, "names": names, "removed": removed}


def task_write(in_file, out_file, geom, settings, progress=None) -> bool:
    """Add the Stack of ``geom`` to the model in ``in_file`` and save it as ``out_file``."""
    say = progress or (lambda text: None)
    say(f"Writing {geom.name}")
    vspio.open_model(in_file)
    vspio.write_stack(geom, settings, say)
    vspio.save_model(out_file)
    return True


def _skin_to_meshes(skin) -> tuple[list, bool]:
    """(meshes, is_union) of a skin ``Solid`` / ``UnionSolid`` (None: no meshes), for pickling."""
    if skin is None:
        return [], False
    if isinstance(skin, UnionSolid):
        return [s.mesh for s in skin.solids], True
    return [skin.mesh], False


def _skin_from_meshes(meshes: list, union: bool):
    if not meshes:
        return None
    solids = [Solid(m) for m in meshes]
    return UnionSolid(solids, "skin") if union else solids[0]


def task_verify(saved_path, plan, spec, skin, skin_names, settings, before, names, prefix, viewer_los,
                progress=None) -> exportcheck.Verification:
    """Reopen the saved file and run the checks (the design is rebuilt from its ``spec``, the skin from its meshes)."""
    return exportcheck.verify_export(saved_path, plan, dm.Duct(spec), skin_names, _skin_from_meshes(*skin), settings,
                                     before, names, prefix, viewer_los, None, progress)


TASKS = {"prepare": task_prepare, "write": task_write, "verify": task_verify}


def export_in_process(design: dm.Duct, cfg: dict, model_path, out_path=None, skin=None, skin_names=None, bodies=None,
                      viewer_los=None, verify: bool | None = None, progress=None) -> ExportResult:
    """The whole export in this process, without any protection against an OpenVSP crash (debugging; see
    ``IN_PROCESS_ENV``). Arguments as for ``export_design``; ``bodies`` is the design bodies (default: built here)."""
    say = progress or (lambda text: None)
    model_path, out_path = resolve_paths(model_path, out_path)
    prefix = prefix_from_config(cfg)
    settings = vspexport.ExportSettings.from_config(cfg)
    skin_names = list(skin_names if skin_names is not None else skin_names_from_config(cfg))
    say("Planning the stations and sections")
    plan = vspexport.plan_export(vspexport.fine_duct(design, settings), design, settings, skin, prefix)
    with tempfile.TemporaryDirectory(prefix="ductmgr_export_") as tmp:
        work = Path(tmp) / "work.vsp3"
        prepared = task_prepare(model_path, work, prefix, say)
        for g in plan.geoms:
            task_write(work, work, g, settings, say)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work, out_path)
    verification = None
    if settings.verify if verify is None else verify:
        verification = exportcheck.verify_export(out_path, plan, design, skin_names, skin, settings, prepared["before"],
                                                 prepared["names"], prefix, viewer_los, bodies, say)
    return ExportResult(out_path, plan, prepared["removed"], verification,
                        report_lines(out_path, model_path, prepared["removed"], plan, design, prefix, verification))


# ---- in child processes ------------------------------------------------------------------------------------------------------
def _cpu_seconds(proc) -> float | None:
    """CPU time used so far by a child process (Windows; None elsewhere or when it cannot be read)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    times = [wintypes.FILETIME() for _ in range(4)]
    ok = ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(int(proc._handle)), *[ctypes.byref(t) for t in times])
    if not ok:
        return None
    return sum((t.dwHighDateTime << 32 | t.dwLowDateTime) for t in times[2:]) / 1e7


def _run_task(task: str, args: dict, tmp: Path, say):
    """Run one task in a child process. Returns (status, result, tail, where): "ok", "crash" (the child died, went silent
    or spun for too long) or "error" (an ordinary Python error in the child; ``tail`` has its output). ``where`` says what
    the child did last and how it ended, e.g. "DUCT_X_NEG: Edit Curve of XSec 7 of 14 (exit code 0xC0000374)"."""
    work = Path(tempfile.mkdtemp(dir=tmp))
    job_file, result_file = work / "job.pkl", work / "result.pkl"
    with open(job_file, "wb") as f:
        pickle.dump({"task": task, "args": args}, f)
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "ductmgr.exportworker", str(job_file), str(result_file)],
                            cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(ln) for ln in proc.stdout] or lines.put(None), daemon=True).start()
    tail, last = [], ""
    started = active = time.monotonic()
    cpu = _cpu_seconds(proc)
    while True:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            if proc.poll() is not None:  # the child has ended (its pipe may stay open in a process it started)
                break
            now = time.monotonic()
            used = _cpu_seconds(proc)
            if used is not None and cpu is not None and used - cpu > 0.2:
                active = now
            cpu = used if used is not None else cpu
            if now - active > CHILD_IDLE or now - started > STEP_LIMITS[task]:
                proc.kill()  # silent and idle for too long (a crashed child the system keeps around) or spinning
                if not result_file.is_file():
                    return "crash", None, tail, f"{last} (no sign of life for {CHILD_IDLE} s, or too long in one step)"
                break
            continue
        if line is None:  # the pipe is closed: the child has ended
            break
        active = time.monotonic()
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            last = line[len("PROGRESS "):]
            say(last)
        elif line:
            tail.append(line)
    try:
        code = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "crash", None, tail, last
    if result_file.is_file():  # written before anything went wrong: a crash while the process was ending does not matter
        try:
            with open(result_file, "rb") as f:
                return "ok", pickle.load(f), tail, last
        except (pickle.UnpicklingError, EOFError):
            pass  # the child died while writing it
    crashed = code in CRASH_CODES or (code == 1 and not any("Traceback" in t for t in tail))
    return ("crash" if crashed else "error"), None, tail, f"{last} (exit code 0x{code & 0xFFFFFFFF:X})"


def _checked(task: str, args: dict, tmp: Path, say, attempts: int = 3):
    """``_run_task`` again (the same arguments) when OpenVSP crashes; an ordinary error is raised."""
    tail = []
    for _ in range(attempts):
        status, result, tail, where = _run_task(task, args, tmp, say)
        if status == "ok":
            return result
        if status == "error":
            raise RuntimeError(f"the export failed in step '{task}':\n" + "\n".join(tail[-12:]))
        say(f"OpenVSP crashed in step '{task}' at: {where}; trying again")
    raise OpenVSPCrash(f"ERROR OpenVSP crashed in step '{task}' every time (last: {where})")


def export_design(design: dm.Duct, cfg: dict, model_path, out_path=None, skin=None, skin_names=None, bodies=None,
                  viewer_los=None, verify: bool | None = None, progress=None) -> ExportResult:
    """Export ``design`` into ``out_path`` (default ``<model>_ducts.vsp3``), a copy of ``model_path`` with the duct Geoms,
    every OpenVSP step in a child process (see the module text). ``skin`` is the fuselage as a ``Solid`` (for the prismatic
    start and the volume check), ``skin_names`` the Geoms that form it (default: the config's), ``viewer_los`` the viewer's
    line of sight as {"x": .., "n": ..}; ``bodies`` is accepted for compatibility (the verification builds its own).
    ``progress`` receives one line per step."""
    if os.environ.get(IN_PROCESS_ENV) == "1":
        return export_in_process(design, cfg, model_path, out_path, skin, skin_names, bodies, viewer_los, verify, progress)
    say = progress or (lambda text: None)
    model_path, out_path = resolve_paths(model_path, out_path)
    prefix = prefix_from_config(cfg)
    settings = vspexport.ExportSettings.from_config(cfg)
    skin_names = list(skin_names if skin_names is not None else skin_names_from_config(cfg))
    say("Planning the stations and sections")
    plan = vspexport.plan_export(vspexport.fine_duct(design, settings), design, settings, skin, prefix)
    crashes: dict[str, list] = {}  # per Geom: where OpenVSP crashed, and what was changed to get past it
    with tempfile.TemporaryDirectory(prefix="ductmgr_export_") as tmp:
        tmp = Path(tmp)
        current = tmp / "step0.vsp3"
        prepared = _checked("prepare", {"model_path": model_path, "out_file": current, "prefix": prefix}, tmp, say)
        for k, geom in enumerate(list(plan.geoms)):
            tail, where = [], ""
            attempts = RETRY_VARIANTS if settings.retry_on_crash else RETRY_VARIANTS[:1]
            for attempt, (extra, factor, shift) in enumerate(attempts):
                g = geom if attempt == 0 else vspexport.rebuild_geom(
                    geom, replace(settings, cedit_anchors=settings.cedit_anchors + extra,
                                  station_tolerance=settings.station_tolerance * factor,
                                  cedit_start=settings.cedit_start + shift))
                nxt = tmp / f"step{k + 1}.vsp3"
                status, _, tail, where = _run_task("write", {"in_file": current, "out_file": nxt, "geom": g,
                                                             "settings": settings}, tmp, say)
                if status == "ok":
                    plan.geoms[k], current = g, nxt
                    if attempt:
                        crashes.setdefault(g.name, []).append(
                            f"written again with +{extra} Bezier segments, station tolerance x{factor:g}, start +{shift}")
                    break
                if status == "error":  # an ordinary Python error: running it again would not help
                    raise RuntimeError(f"the export failed while writing {geom.name}:\n" + "\n".join(tail[-12:]))
                crashes.setdefault(geom.name, []).append(f"OpenVSP crashed at: {where}")
                say(f"OpenVSP crashed while writing {geom.name}: {where}")
            else:
                raise OpenVSPCrash(f"ERROR OpenVSP crashed while writing {geom.name} at: {where}"
                                   + ("; every version of the Geom crashed it" if settings.retry_on_crash
                                      else " (export.retry_on_crash is off)"))
        say(f"Saving {out_path.name}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, out_path)
        verification = None
        if settings.verify if verify is None else verify:
            verification = _checked("verify", {
                "saved_path": out_path, "plan": plan, "spec": design.spec, "skin": _skin_to_meshes(skin),
                "skin_names": skin_names, "settings": settings, "before": prepared["before"],
                "names": prepared["names"], "prefix": prefix, "viewer_los": viewer_los}, tmp, say)
    return ExportResult(out_path, plan, prepared["removed"], verification,
                        report_lines(out_path, model_path, prepared["removed"], plan, design, prefix, verification,
                                     crashes))
