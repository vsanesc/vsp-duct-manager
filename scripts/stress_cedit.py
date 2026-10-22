# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Stress test of writing Edit Curve XSecs into OpenVSP Stacks, all in one process.

    python scripts/stress_cedit.py [--sections 200] [--per-stack 20] [--seed 0] [--variant NAME]

Random sections (asymmetric, a mix of straight and curved segments, 4..20 segments, anchors that fall on u = 0.25 / 0.75
and on the last index) are written with ``vspio.write_stack`` into Stacks of ``--per-stack`` XSecs, a few Stacks per model,
the way an export does it. A crash kills the process, so run it from ``tests/test_cedit_stress.py`` (which reads the exit
code) or look at the exit code yourself. At the end every XSec's stored control points are read back and compared with the
ones that were written (``max deviation`` in units of the section size).

``--variant`` selects how an XSec is written (default: the export's ``cedit.apply_cedit``).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ductmgr import cedit  # noqa: E402
from ductmgr import vspexport as vx  # noqa: E402


def random_section(rng: np.random.Generator, m: int) -> cedit.CeditSection:
    """A random closed clockwise outline of ``m`` Bezier segments: star-shaped about the origin with random radii and
    angular spacing (so asymmetric), each segment straight or curved at random."""
    ang = np.sort(rng.random(m)) * 2 * math.pi
    ang = ang + (rng.random() * 2 * math.pi)
    r = 0.15 + 0.15 * rng.random(m)
    anchors = np.c_[r * np.cos(-ang), 0.7 * r * np.sin(-ang)]  # clockwise
    segs, flags = [], []
    for k in range(m):
        a, b = anchors[k], anchors[(k + 1) % m]
        if rng.random() < 0.5:
            segs.append(np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]))
        else:
            mid = 0.5 * (a + b)
            bulge = (b - a) @ np.array([[0, -1], [1, 0]]) * (rng.random() - 0.3) * 0.6
            segs.append(np.array([a, a + (mid + bulge - a) * 0.66, b + (mid + bulge - b) * 0.66, b]))
        flags.append(False)
    return cedit.segments_to_cedit(segs, flags, anchors[int(rng.integers(m))], clockwise=True)


def random_plan(rng, name: str, n: int, negative: bool):
    """A Stack plan of ``n`` random sections (the same segment count in all) along +x, with cap points."""
    m = int(rng.choice([4, 8, 12, 16, 5, 7, 9, 10, 13, 20]))  # multiples of 4 put anchors on u = 0.25, 0.5, 0.75
    secs = []
    for i in range(n):
        c = random_section(rng, m)
        secs.append(vx.SectionPlan(np.array([0.06 * i, 0.0, 0.0]), np.eye(3), c, 0.06 * i, "station", i, np.zeros((0, 2))))
    plan = vx.GeomPlan(name, "negative" if negative else "positive", "branch", None, False,
                       vx._with_caps(secs, 0.001), n, None)
    return plan


def deviation(xs_id: str, c: cedit.CeditSection) -> float:
    import openvsp as vsp

    got = vsp.GetEditXSecCtrlVec(xs_id, True)
    if len(got) != len(c.points):
        return math.inf
    return max(max(abs(g.x() - p[0]), abs(g.y() - p[1])) for g, p in zip(got, c.points))


def install_variant(name: str) -> None:
    """Replace ``cedit.apply_cedit`` by another way of writing an XSec (to find out what OpenVSP chokes on)."""
    if name == "export":
        return
    import openvsp as vsp

    def apply(xsec_id, c, g1=True):
        def parm(n):
            return vsp.GetXSecParm(xsec_id, n)

        if name == "old":  # the writing as it was before the symmetry hardening (no CloseFlag, no read-back)
            for n in ("SymType", "TBSym", "RLSym", "AllSym"):
                vsp.SetParmVal(parm(n), 0.0)
        elif name == "symon":  # the default SymType (SYM_RL) is left on: OpenVSP mirrors the points (distorts them)
            pass
        else:
            vsp.SetParmVal(parm("SymType"), float(vsp.SYM_NONE))
        if name == "open":
            vsp.SetParmVal(parm("CloseFlag"), 0.0)
        if name != "old":
            vsp.Update()
        vsp.SetParmVal(parm("CurveType"), float(vsp.CEDIT))
        vsp.Update()
        vsp.SetXSecWidthHeight(xsec_id, c.width, c.height)
        vsp.Update()
        ctrl = [vsp.vec3d(x, y, 0.0) for x, y in c.points]
        vsp.SetEditXSecPnts(xsec_id, list(c.u), ctrl, [0.0] * len(ctrl))
        vsp.Update()

    cedit.apply_cedit = apply


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", type=int, default=200)
    ap.add_argument("--per-stack", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="export")
    ap.add_argument("--only-stack", type=int, default=None, help="write only this stack (1-based) of the random series")
    ap.add_argument("--pick", default=None, help="with --only-stack: only these interior sections (comma separated, 1-based)")
    ap.add_argument("--sections-of", type=int, default=None, help="with --only-stack: only its first N sections")
    args = ap.parse_args(argv)

    import openvsp as vsp

    from ductmgr import vspio

    install_variant(args.variant)
    rng = np.random.default_rng(args.seed)
    settings = vx.ExportSettings()
    vsp.ClearVSPModel()
    done, worst, stacks = 0, 0.0, []
    while done < args.sections:
        n = min(args.per_stack, args.sections - done)
        plan = random_plan(rng, f"STRESS_{len(stacks)}", n, negative=len(stacks) % 2 == 1)
        if args.only_stack is not None and len(stacks) + 1 != args.only_stack:
            stacks.append((None, plan))
            done += n
            continue
        if args.pick:
            plan.sections = [plan.sections[0]] + [plan.sections[int(i)] for i in args.pick.split(",")] + [plan.sections[-1]]
        elif args.sections_of:
            keep = [plan.sections[0]] + [x for x in plan.sections[1:-1]][:args.sections_of] + [plan.sections[-1]]
            plan.sections = keep
        gid = vspio.write_stack(plan, settings)
        stacks.append((gid, plan))
        done += n
        print(f"wrote stack {len(stacks)} ({n} sections, {plan.sections[1].cedit.n_segments} segments)", flush=True)
    for gid, plan in stacks:
        if gid is None:
            continue
        surf = vsp.GetXSecSurf(gid, 0)
        for i, s in enumerate(plan.sections):
            if s.cedit is not None:
                worst = max(worst, deviation(vsp.GetXSec(surf, i), s.cedit))
    print(f"done {done} sections in {len(stacks)} stacks; max deviation of the stored control points {worst:.2e} "
          "(normalised by the section size)", flush=True)
    return 0 if worst < 1e-3 else 3  # 3: OpenVSP did not keep the points it was given


if __name__ == "__main__":
    sys.exit(main())
