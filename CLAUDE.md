# VSP Duct Manager — project context for Claude Code

## Purpose
Design the air-intake ducts of an aircraft in Python, check them against the real
fuselage geometry, and write them into the OpenVSP model as native, GUI-editable
components. The user keeps working in the OpenVSP GUI afterwards; the API is only
used to generate/update the duct components.

## Project-specific configuration
Lives in `CLAUDE.local.md` (gitignored, never published). Imported below:
@CLAUDE.local.md

## Environment
- Windows. OpenVSP 3.55.0 in `OpenVSP-3.55.0-win64/`.
- The `openvsp` .pyd modules are built for **Python 3.13** (they link python313.dll).
  Any other Python version will fail to import.
- `OpenVSP-3.55.0-win64/python/environment.yml` ships with an empty `python=`
  pin. Do NOT edit the vendor folder; create our own env instead (Phase 0).
- Units: metres, seconds, kg, K, Pa. VSP axes: +X aft, +Y right, +Z up.

## Hard rules
1. Never overwrite the user's input .vsp3. Always write `<name>_ducts.vsp3`.
2. Only touch Geoms whose name starts with `DUCT_`. On re-run, delete and recreate
   those; leave every other component exactly as it was.
3. Ducts are modelled as **closed Stack volumes with Negative_Volume_Flag = 1**
   (flat end caps). Do not use the Engine/flowthrough model. Each negative must
   start slightly outside the skin (pierce it) and end inside the fuselage.
   Branch negatives overlap into the trunk at the junction.
4. Cross-sections are **Edit Curve XSecs** so the user can move points in the GUI.
5. All inputs live in `duct_config.yaml`. No magic numbers in code.
6. The repo is **public**. Never commit: any `.vsp3`, `duct_config.yaml`
   (only `duct_config.example.yaml` with generic dummy values), `CLAUDE.local.md`,
   generated outputs, `.env*`, keys/tokens/credentials, absolute personal paths.
   Run the secret scan (pre-commit) before every commit; never push without the
   user's explicit OK.
7. Each phase ends with its acceptance checks passing; report results, then stop.
8. Never use `git add .`, `git add -A` or `git commit -a`. Stage files by name and
   show `git diff --cached --stat` before every commit.

## Layout
```
duct_config.yaml
ductmgr/
  vspio.py        # open model, find geoms, tessellate, write DUCT_* stacks
  engine.py       # ISA, mass flow, capture/face areas, Mach checks
  geometry.py     # inlet plane frame, projected area, centreline, frames
  section.py      # plane ∩ fuselage skin as a (u, v) curve, snapping, signed distance
  outline.py      # SectionShape: any number of vertices, straight / Bezier edges, continuity
  sketch.py       # inlet section on the skin: snapping, validity, Fit area
  area_law.py     # A(s) along a branch (linear / cosine, junction = n x branch)
  interp.py       # arc-length resampling from a reference vertex (no twist)
  cedit.py        # section -> OpenVSP Edit Curve (CEDIT) XSec mapping
  centreline.py   # branch / trunk cubic Beziers, G1 at the junction, arc-length mapping
  areacurve.py    # A(s) as monotone PCHIP curves for the branch and the trunk
  duct.py         # 1..N branches: stations, frames, sections at A(s), the junction (D-split / sectors), POS + NEG bodies
  ductcheck.py    # external region, wall left, other Geoms, line of sight; solids.py = mesh queries
  qt_paths.py     # side / top / area-law 2D editors
  explore_model.py / qt_sketch.py / qt_explorer.py   # section editor model + Qt UI
  checks.py       # clearance vs fuselage, area error, diffusion angle
  viz.py          # PyVista scene + matplotlib A(s) plot
run_explore.py    # interactive viewer: place the inlets, save them to the config
run_design.py     # design + visualise (no VSP write)
run_export.py     # write DUCT_* into <name>_ducts.vsp3 + CompGeom check
tests/
```
