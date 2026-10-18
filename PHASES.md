# Claude Code prompts — one phase at a time

Paste one phase, let it finish and pass its checks, then paste the next.

---
## Phase 0 — Environment

Read CLAUDE.md. Set up the environment:
- Create a conda env `ductmgr` with python=3.13 (or a venv on a Python 3.13 interpreter).
- Install OpenVSP's packages from `OpenVSP-3.55.0-win64/python/` with
  `pip install -e` for `openvsp_config` and `openvsp` only (no CHARM/AvlPy/pyPMARC).
- Install numpy, scipy, pyyaml, matplotlib, pyvista, pytest.
- Write `env_check.py` that imports openvsp, prints `vsp.GetVSPVersion()`,
  opens the .vsp3 given in duct_config.yaml and lists every Geom: name, type, ID,
  bounding box.
- Create `duct_config.yaml` with placeholders (fields listed in Phases 1–3) and
  `model.vsp3_path`, `model.fuselage_name`.
- `git init` (local only), .gitignore for *.vsp3 outputs, meshes, __pycache__.
Acceptance: env_check.py runs and lists the components of the user's model.

---
## Phase 1 — Engine and area sizing (`ductmgr/engine.py`)

Inputs (yaml `engine:` / `design_point:`): face diameter, hub-to-tip ratio
(0 = full disc), corrected flow per unit face area [kg/s/m²] (default 190),
design Mach, altitude, total-pressure recovery (default 0.97), capture margin
(default 1.05), optional reference thrust and specific-thrust range for a sanity check.

Compute:
- ISA static T, p, rho at altitude; freestream V; total T0, p0.
- Face area; corrected mass flow; actual mass flow
  m = mc * (p0*eta/101325) / sqrt(T0/288.15).
- Face Mach by inverting the isentropic mass-flow function (subsonic root).
- Capture area total = m / (rho*V) * margin; per inlet = half.
- Optional check: thrust / m within the specific-thrust range -> warn otherwise.
- Pretty summary table printed + returned as a dataclass.
Tests: ISA vs table at 0 / 11 km; mass-flow function inversion round-trip;
face Mach within 0.3–0.7 for default inputs or warning raised.
Acceptance: pytest green, summary printed for the yaml values.

---
## Phase 2 — Duct geometry (`ductmgr/geometry.py`)

Yaml `ducts:`:
- inlet: centroid (x,y,z) of the right inlet (left is mirrored), inlet plane
  normal, polygon vertices in the inlet plane (normalised, any 4–6 vertices,
  CCW), lip extension outside the skin [m].
- junction: point (x, 0, z) where the branches merge; trunk to engine face
  centre (x, 0, z); face normal (default +X).
- n_stations per branch and per trunk (default 12 each).
- area law: "linear" or "cosine" between capture area and face area, with the
  junction area = 2 × branch area at that point (continuity).

Build:
- Centrelines as cubic Bezier/Hermite curves with tangent continuity at the
  junction; frames by parallel transport (no twist flips).
- Sections: N resampled points (e.g. 64) morphing from the polygon to a circle
  (blend by s), each scaled so its area equals A(s) exactly.
- Station output: position, frame, point list, area.
Tests: polygon area routine; every station area within 0.5% of A(s);
frames orthonormal; mirror symmetry exact.
Acceptance: tests green; `run_design.py` prints the station table.

---
## Phase 3 — Checks + visualisation (`checks.py`, `viz.py`, `run_design.py`)

- Get the fuselage surface from the .vsp3 via the API (export the fuselage alone
  to STL through a temporary set, load with PyVista). No manual STL export.
- Signed distance of every duct section point to the fuselage skin: flag stations
  that leave the fuselage (except the lip region) and min wall clearance
  (yaml `checks.min_clearance`).
- Equivalent conical diffuser half-angle per segment -> warn above yaml limit (default 6°).
- Centreline offset/length ratio per branch (S-duct severity) printed.
- PyVista scene: translucent fuselage, duct surfaces coloured by local area
  error or clearance, engine-face disc, inlet polygons. Matplotlib A(s) plot.
Acceptance: `python run_design.py` opens the 3D view and the A(s) plot, prints
all checks with PASS/WARN.

---
## Phase 4 — Export to OpenVSP (`vspio.py`, `run_export.py`)

- Open the user's .vsp3, delete existing `DUCT_*` geoms, create:
  `DUCT_BRANCH` (Stack, planar XZ symmetry -> gives both inlets) and `DUCT_TRUNK` (Stack).
- One Edit Curve XSec per station; polygon-like stations use linear
  interpolation, near-circular ones smooth. Convert station positions/frames to
  Stack XDelta/YDelta/ZDelta + rotations. Flat end caps. Negative_Volume_Flag = 1.
  Branch starts `lip_extension` outside the skin and overlaps into the trunk.
- Save as `<name>_ducts.vsp3`.
- Automatic verification: reopen the saved file, run CompGeom on the set with
  fuselage + DUCT_*, check it returns a non-empty mesh and that fuselage volume
  drops by roughly the duct volume inside it (report %). Slice the VSP Stacks at
  each station and compare area vs design (report max error).
Acceptance: saved file opens in the GUI; CompGeom check PASS; max area error
reported (< 3% expected, otherwise increase stations).

---
## Phase 5 (later, optional) — Round trip
Read DUCT_* back from a GUI-edited .vsp3 into the station format so GUI edits
can be re-checked in Phase 3 without being lost.
