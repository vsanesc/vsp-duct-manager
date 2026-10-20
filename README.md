# VSP Duct Manager

Design the air-intake ducts of an aircraft in Python, check them against the real fuselage
geometry, and write them into an [OpenVSP](https://openvsp.org) model as native,
GUI-editable components. You keep working in the OpenVSP GUI afterwards; the Python API is
only used to generate and update the duct components.

> **Status: early development.** Implemented so far: the environment/config scaffolding
> (config loader, `env_check.py`, example model), Phase 1 engine/capture-area sizing, and
> Phase 2: an interactive 3D viewer to place the inlets on the fuselage. The duct geometry,
> checks and export stages below are planned and not implemented yet.

## What it will do

Configuration modelled: one engine with a single circular engine face, fed by two inlets
that are mirror-symmetric about the XZ plane, merging into a Y-duct and a single trunk.

1. **Sizing** (`ductmgr/engine.py`): ISA conditions, engine mass flow, capture and face areas, Mach checks.
2. **Geometry** (`ductmgr/geometry.py`): centrelines, frames, area law, cross-section morphing from an inlet polygon to a circle.
3. **Checks and visualisation** (`ductmgr/checks.py`, `viz.py`): clearance to the fuselage skin, diffuser half-angle, S-duct severity, PyVista 3D view and an A(s) plot.
4. **Export** (`ductmgr/vspio.py`): write `DUCT_*` components into `<name>_ducts.vsp3` as closed Stack volumes with the negative-volume flag set and Edit Curve cross-sections, then verify with CompGeom.

Safety rules: your input `.vsp3` is never overwritten (output is `<name>_ducts.vsp3`), and
only Geoms whose name starts with `DUCT_` are ever created, deleted or modified.

Units are metres, seconds, kg, K and Pa. OpenVSP axes: +X aft, +Y right, +Z up.

## Requirements

- Windows (developed and tested there).
- **Python 3.13.** OpenVSP's compiled Python modules link `python313.dll`; other versions fail to import.
- **OpenVSP 3.55.0** with its bundled Python packages. OpenVSP is not included in this repository;
  download it from <https://openvsp.org>.

## Install

From the repository root, in PowerShell. Set `$VSP` to the folder that contains OpenVSP's `python` directory
(for example the unpacked `OpenVSP-3.55.0-win64`):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

$VSP = "path\to\OpenVSP-3.55.0-win64"
foreach ($p in "openvsp_config","utilities","degen_geom","openvsp") {
    .venv\Scripts\python -m pip install --no-deps -e "$VSP\python\$p"
}
```

`--no-deps` matters: without it pip would try to resolve the name `utilities` against PyPI, where it is an
unrelated package. `utilities` and `degen_geom` are needed because `openvsp` imports them on load.
The optional CHARM, AvlPy and pyPMARC packages are not needed.

Check the install:

```powershell
.venv\Scripts\python -c "import openvsp as vsp; print(vsp.GetVSPVersion())"
```

## Quickstart (example config)

`duct_config.example.yaml` holds generic dummy values. The tools use `duct_config.yaml` if it exists (copy
the example to that name for your own model; it is gitignored) and fall back to the example otherwise.
Pass `--config <path>` to choose a file explicitly.

```powershell
# 1. build a dummy model (one plain fuselage named "Test Fuselage"); refuses to overwrite an existing file
.venv\Scripts\python make_example_model.py

# 2. open it through the OpenVSP API and list every Geom (name, type, ID, bounding box)
.venv\Scripts\python env_check.py --config duct_config.example.yaml
```

### Engine and capture-area sizing

```powershell
.venv\Scripts\python -m ductmgr.engine --config duct_config.example.yaml
```

Prints the ISA conditions, free-stream velocity, total temperature and pressure, face area, corrected and
actual mass flow, face Mach number (subsonic root of the isentropic flow function), total and per-inlet
capture area, and a specific-thrust sanity check. The thrust rating is divided by the mass flow at its own
reference condition (`engine.reference_thrust_condition`, default sea-level static), not the design-point
flow. Out-of-range results are flagged `WARN`, never silently
corrected. If the requested corrected flow cannot pass the face (choked), it stops with an error that states
the maximum mass flow for that face area.

Run the tests (they use only hard-coded values and the example config):

```powershell
.venv\Scripts\python -m pytest
```

### Interactive inlet placement

```powershell
.venv\Scripts\python make_example_model.py                      # once, for the example
.venv\Scripts\python run_explore.py --config duct_config.example.yaml
```

The viewer reads the fuselage, engine and every other component straight from the `.vsp3` through the
OpenVSP API (a temporary set and a temporary STL, both removed afterwards; the model is never written). It is a
Qt window (PySide6 + pyvistaqt): the 3D view in the centre, the **section editor** docked on the left and a
compact dark panel on the right. Nothing is drawn inside the 3D view except the scene itself (translucent
fuselage, other components as grey context, the engine, the engine-face disc and the inlets).

**The inlet plane.** The opening is drawn in a plane defined by an origin `O` (on or near the skin) and two angles:
`pitch_deg` (rotation about Y, the opening tilted up for positive values) and `sweep_deg` (rotation about Z, the
opening swept in planform). The plane normal, facing the freestream, is `n = Rz(sweep) . Ry(pitch) . (-X)`, so
both angles 0 is the plane x = const facing forward. In the plane, `v` is the component of +Z orthogonal to `n`
and `u = v x n` (seen from the front, `u` to the right and `v` up, so the aircraft's +Y side is on the left). All
sketch vertices are stored as `(u, v)` in metres relative to `O`, never as `(y, z)`. The left inlet is the exact
mirror image of the right one about XZ (origin mirrored, sweep negated, `u` negated). The capture area is the
opening area projected normal to the freestream, `area_in_plane * |n . X|`; the readouts show both, and warn when
`|n . X| < 0.5` (the opening is more than 60 degrees off the flow).

**Align to edge...** sets the sweep so the opening's *planform trace* (the plane's intersection with the
horizontal plane through `O`; its direction does not depend on the pitch) is parallel to a chosen edge. The picker
offers any Wing-type Geom of the model, with its leading or trailing edge, or a manual angle. A wing's sweep is
read from its VSP parameters (not from the mesh) in the section that spans the inlet's y: the section `Sweep` is
measured at the chord fraction `Sweep_Location`, so the leading edge follows from it, the taper and the dihedral
(`tan LE = tan(sweep) + s (root - tip) / span`, divided by `cos(dihedral)` for the planform; the trailing edge adds
`(tip - root) / span`). For an edge swept back by an angle L the result is `sweep_deg = -L`, which makes the right
inlet's outer edge trail aft (the left inlet is the mirror image). The readouts show `aligned to: <geom> LE
(Λ = xx.x°)` while the sweep is the aligned one and `custom` as soon as you edit it; the alignment itself is not
saved, only the sweep. An alignment that would leave `|n . X|` below 0.2 is refused, with the reason.

**The section editor.** The curve where the plane cuts the fuselage skin is computed in `(u, v)`, with chine
corners (turning angle above 30 degrees) marked. The inlet opening is a closed outline of any number of vertices
(at least 3) that you edit with the mouse:

- **Vertices** are dragged. A vertex dropped within 12 pixels of the section curve **snaps** onto it (drawn
  filled); drag it away to unsnap, or hold `Alt` to disable snapping. **Double-click an edge** to insert a vertex
  there (on a skin-following edge it lands on the skin); **right-click a vertex** (or select it and press `Del`)
  to remove it, down to 3.
- **Edges** are straight or curved. Select an edge and press *Curved* to turn it into a cubic Bezier with two
  draggable tangent handles (blue squares). An edge whose two end vertices are both snapped **follows the
  section curve** between them (the shorter arc, chine corners included), drawn in the accent colour; it is not
  editable.
- **Vertex continuity** (buttons *Corner*, *Smooth*, *Symmetric*): a corner keeps the two edges at an angle,
  *Smooth* keeps the tangent continuous (G1) with independent handle lengths, *Symmetric* also equalises them.
  Markers: square = corner, circle = smooth, circle with a ring = symmetric.
- Live readouts: the projected area against the Phase 1 target, the snapped edge length and the minimum clearance
  of the free edges above the skin, all on the true curved outline. **Fit area** scales only the free vertices
  about the centre of the snapped edges until the area matches the target (within 0.5 %); the snapped vertices
  never move.
- If any part of the opening lies inside the fuselage section the outline turns red, the readout says ERROR and
  **Save is disabled**.
- The mirrored left inlet is shown greyed, with the symmetry plane y = 0, a grid and a scale bar. The wheel zooms
  and a middle-drag pans.

**Key sections.** A branch is described by key sections at a normalised arc length `s`, from the inlet to the
junction: `s = 0` is the inlet section above. (The trunk has its own key sections, see **The trunk** below; the engine
face, a fixed circle of `engine.face_diameter`, closes the trunk at its `s = 1`.) The
**Key sections** list in the panel adds (*Add...*, at a chosen `s` before the junction) and removes
intermediate sections; selecting a row opens that section in the editor. A new section starts as a copy of the one
before it. Its *shape* is free (same editing tools, no skin to snap to) but its *size* always follows the area law
`A(s)`: the editor rescales it to exactly that area when an edit ends. `A(s)` runs from the inlet's in-plane area
to the face area (`ducts.area_law`, `linear` or `cosine`), and the trunk carries all the branches, so the area at the
junction is the sum of the branch areas there.

Between key sections, outlines are resampled to N points by arc length from a reference vertex (the one closest
to the section's local `+v`), so a pentagon morphs into a circle without twisting. `ductmgr/cedit.py` maps a
section to an OpenVSP **Edit Curve** cross-section (type `CEDIT`) for the export phase: straight edges as
degenerate Beziers with handles at 1/3 and 2/3, curved edges as Beziers, corners kept, G1 flags only where the
handles are collinear, so every point stays editable in the OpenVSP GUI.

**The duct (Phase 3b).** Below the 3D view a **Duct** dock has three tabs; every edit goes through the same model
and the same undo history as the section editor, and the 3D view follows at once.

- **Topology.** Two branches (the right inlet and its mirror image) join in a Y junction on `y = 0`, and one trunk
  runs to the engine face (centre from the engine Geom, axis `ducts.engine_face.normal`, default +X).
- **The Y junction.** The trunk start section (the trunk's own `s = 0` key section, see **The trunk**; mirror
  symmetric with symmetry `xz`) is split by the plane `y = 0` into two halves; each branch ends with exactly its half
  (same shape, position and half the area), so the two branch ends tile the trunk start section with no overlap and no
  gap. That branch end is a fixed key section in the list (like the engine face): derived, not editable, and the
  branch interpolation morphs into it. Editing the trunk start therefore reshapes every branch end. The branch
  centrelines end at the halves' centroids rather than on `y = 0`, with the junction tangent (G1). For clean
  booleans each branch negative continues into the trunk by `ducts.junction.overlap` (default 0.02 m) as a
  half-section and the trunk negative starts as far upstream, all inside the duct. With
  `ducts.junction.splitter_length` above 0 the branch negatives stay apart for that length, separated by a wall of
  `wall_thickness` on `y = 0` (the trunk negative then starts where the splitter ends).
- **Centrelines** (SIDE x-z and TOP x-y tabs). A cubic Bezier per branch, from the inlet section centroid
  (starting along the inward plane normal) to the junction, and one for the trunk, from the junction to the face
  centre (ending along the engine axis). The junction tangent lies in the XZ plane, so the branches and the trunk
  are G1 at the junction and everything is symmetric about `y = 0`; the two junction handles are collinear with
  independent lengths. The tabs show the fuselage silhouette, the engine, the other components, the opening and the
  centrelines with CAD-style handles. Drag the junction (side: x and z; top: x only) and the handles: the inlet
  handle slides along the inlet normal and the face handle along the engine axis (length only), the junction
  handles set angle and length in the side view and length in the top view. Double-click a handle for its default
  (`ducts.centreline.handle_fraction` x the chord length). Wheel zooms, middle-drag pans.
- **Area law** (third tab). `A(s)` for the branch and the trunk as monotone (PCHIP) curves with draggable control
  points. The branch runs from the inlet area to half the junction area, the trunk from the junction area to the
  face area, so the trunk starts with `n_inlets` times the area a branch ends with. The junction area (drag the
  right end of the branch curve or the left end of the trunk curve) defaults to the geometric mean of the combined
  inlet area and the face area; the interior points default to `ducts.area_law` (`linear` or `cosine`).
  Double-click a curve to add a point, right-click a point to remove it. The curve is red where it decreases or
  where the equivalent cone half-angle exceeds `checks.max_diffuser_half_angle_deg`. Key sections are held at
  `A(s)` of these curves: an inlet's sections by the arc-length fraction of its own branch, the trunk's by the
  fraction of the trunk. Selecting the **Trunk** row of the Inlets list shows the trunk curve alone, full width.
- **Bodies.** Stations lie along the centrelines (`ducts.n_stations_branch` / `n_stations_trunk`); the section at a
  station is the blend of the neighbouring key sections (arc-length correspondence, no twist), scaled to exactly
  `A(s)`, with frames from parallel transport (the branch's accumulated roll is spread so it meets the trunk's
  symmetric frame at the junction). Two bodies per branch and for the trunk come from the same stations:
  the **negative** is the inner sections, starting `ducts.lip_extension` *upstream* of the opening plane (a prism
  along the inlet normal) so it never shares a face with the positive, and ending at the engine face; the
  **positive** is the inner section offset outward by `ducts.wall_thickness` (exactly that perpendicular to every
  edge, curved ones and corners included), starting exactly at the opening plane (`ducts.lip_mode: blunt`, a blunt
  lip of that thickness) and running to the face. The model intended in OpenVSP is
  **(fuselage ∪ positives) − negatives**. The part of the positive outside the fuselage skin, the *external region*,
  is a result: it is drawn in orange and reported with its length and wetted area.
- **Live checks** (readouts; the bodies turn red where they apply). The positive may leave the skin only within
  `checks.lip_length` of the opening, otherwise WARN (an unplanned bump); `checks.min_clearance` applies only to the
  internal part, the distance from the negative (the inner wall) to the skin where the positive is inside it; in the
  external region the wall is `wall_thickness` by definition, reported separately and never compared with the
  clearance; neither body may intersect another Geom (the engine
  is not one of them); the line of sight reports the % of the engine face seen straight through an opening along
  +X and along the inlet normal (0 % = hidden); offset / length, the maximum centreline curvature and the cone
  half-angle per part; A(s) must not decrease. The cheap numbers update with every edit; the bodies and the mesh
  checks are rebuilt a moment after the last one.

**The trunk.** The trunk (junction to engine face) is its own entity, not derived from any inlet: `ducts.trunk` holds
its key sections (`key_sections`, with `s` along the trunk: `0` is the trunk start at the junction, `0 < s < 1` are
intermediate ones, the engine face at `1` is the fixed circle), its area curve (`junction_area`, interior
`area_points`) and its two handles (`junction_out_handle`, `end_handle`). The trunk start (`s = 0`) is an editable
section like any other, by default a superellipse of exponent 2.5; its size is held at the junction area, and the
other trunk sections at the trunk's `A(s)`. Every branch end is cut from the trunk start section (D-split or
sectors), so editing it reshapes all of them and the pieces keep tiling it exactly. With symmetry `xz` the trunk
sections are kept mirror symmetric about `u = 0` (edit one side). The **Inlets** list ends with a fixed **Trunk** row
(it cannot be removed, renamed or moved): select it to edit the trunk's key sections in the section editor and its
area curve in the Area tab. While the trunk start is the section in the editor, two actions act on it (both one undo
step): **Reset to default…** replaces it by a superellipse of an exponent you choose (2 = ellipse, larger = squarer,
1 to 8), held at the junction area; **Simplify…** reduces it to N vertices (a subset of its own) with curved edges fitted
to the outline, keeping its area and, with symmetry `xz`, its mirror symmetry (vertices go in mirror pairs, so the
count may need one more or less); corners stay corners, and the deviation from the previous outline (max and rms, mm and
% of the equivalent radius) is shown in the status line, the section info and the readouts. This is how a migrated
64-vertex polygon becomes a handful of editable points. Inlets take no key sections beyond the junction. Nothing about the trunk depends on how
many inlets there are, their order or their names (the junction point and the junction area, when not given, are
symmetric functions of the inlets, so reordering or renaming never moves the trunk; give them explicitly and adding
or removing an inlet leaves it bit for bit as it was). The panel's *Up* / *Down* buttons reorder the inlets.

Configs from before (no `ducts.trunk`; the trunk start section was derived from the first inlet's sections at the
junction, its key sections beyond the junction shaped the trunk, and the junction / trunk area points and handles
were in `ducts.area_points` and `ducts.centreline`) are migrated on load with a one-line notice: the derived trunk
start becomes the trunk's `s = 0` section (a polygon of the section samples), the first inlet's sections beyond the
junction move to the trunk (their `s` converted to the trunk's), sections beyond the junction on other inlets are
dropped, and the old locations of the values are still read. Save writes the new format.

The config keeps the junction in `ducts.junction.point` `[x, y, z]` (y is 0 with symmetry xz), the shared tangent in
`ducts.centreline` (`null` = default), the trunk in `ducts.trunk` (`null` = derived from `ducts.area_law` and the
inlets), and `ducts.wall_thickness`, `ducts.lip_extension` (formerly `ducts.inlet.lip_extension`, still read) and
`ducts.lip_mode`. Each inlet keeps its own handles and branch area points in `ducts.inlets`. Save writes them all,
keeping comments.

**Bodies.** The **Bodies** list in the panel has every Geom of the model (and the engine) with two checkboxes.
*Show* hides a body in all three views (3D, Side, Top) at once, and the toolbar's *Context* switch hides every body
that is neither skin nor the engine; hiding is a view setting, hidden bodies stay in the checks, and the clash checks
always use every Geom except the engine and the skin. *Skin* marks the Geom(s) that form the fuselage skin for the
section sketch, snapping, clearance and the external-region check; several skin bodies are used as their union (the
section is the union of their cuts, a point is inside when it is inside any of them). Changing it rebuilds the
section and reruns the checks, with undo, and Save writes `model.skin_bodies` (a list; the older single
`model.fuselage_name` is still read when it is absent and is never rewritten). Bodies are never treated specially by
name: a missile is a skin only if you say so. The Side and Top views draw the exact projected outline of each body
(the union of its projected triangles), so a coarse nose tessellation gives the true silhouette.

**Inlets (1..N).** The design has any number of inlets, listed in the **Inlets** group of the panel (add, duplicate,
remove, rename); the section editor, the 2D views and the panel edit the *selected* one (click an opening in a 2D
view to select it). Each inlet has its own plane and key sections, a **flow share** (the fraction of the total flow
its branch carries; default an equal split over all physical branches, always normalised to sum to 1 with a notice)
and a **mirrored twin** box. Its capture area is its flow share of the total of Phase 1, and its area law scales
accordingly. The global **symmetry** is `xz` or `none`:

- `xz`: an inlet with the mirror box set gets an automatic mirrored twin about y = 0 (generated, greyed, not
  editable); the junction, the trunk and all centrelines are held on y = 0. An inlet centred on y = 0 (origin y = 0,
  no sweep) is its own mirror image: it is never mirrored and its sections are kept symmetric about u = 0, so you edit
  one side and the other follows.
- `none`: every inlet is independent and the junction position and tangent are free (also in the top view).

What happens where the branches meet depends on their number (twins counted): one branch is one continuous duct with
the junction point as an intermediate G1 point; two branches split the trunk start section by the plane through the
trunk axis equidistant to both branches (y = 0 when symmetric), offset to give each its flow share of the area; three
or more split it into sectors by rays from its centroid, ordered by each branch's angular position around the trunk
axis, with the ray angles solved so every sector has its flow share. Each branch ends exactly on its piece, so the
pieces tile the section with no gap and no overlap.

The config holds `ducts.symmetry` and `ducts.inlets`, a list of `{name, origin, pitch_deg, sweep_deg, flow_fraction,
mirror, start_handle, junction_in_handle, area_points, key_sections}`. The old single-inlet format (`ducts.inlet` +
`ducts.key_sections`) still loads, as one mirrored inlet with symmetry xz, with a one-line notice; Save migrates it.

The 3D view draws both openings on their planes, the section curve and the plane's `u`/`v`/`n` arrows at `O`.

Other controls:

- **Placement is explicit.** Clicking the geometry does nothing by default. Press **Place inlet** (`P`): the next
  click on the fuselage sets the plane origin there (creating a default sketch the first time; a click on the left
  side is mirrored onto the right) and place mode switches itself off. `Esc` cancels. The status bar shows
  `View` or `Placing inlet - click on fuselage, Esc to cancel`. Dragging still rotates the camera.
- **Lock inlet** disables placement and nudges (the origin).
- **Undo / redo** (`Ctrl+Z` / `Ctrl+Y`) cover every change, placement, origin, angles and sketch drags alike (up
  to 200 steps; a drag or rapid edits of one value merge into one step).
- **Panel**: origin x/y/z with a nudge-step selector, pitch, sweep, Align to edge..., and read-only monospace
  readouts with PASS in green, WARN in amber and ERROR in red. Spin boxes take the mouse wheel only while focused.
- **Save to config** (`Ctrl+S`) writes the inlet into `duct_config.yaml`, keeping its comments and every other
  key; **Revert to saved** re-reads it. The title bar shows `*` while there are unsaved changes, and closing the
  window then asks first.
- **View** toolbar: side / top / front / iso (`1`-`4`), toggles for the context components, the engine envelope
  and the section editor dock.

Each inlet in `ducts.inlets` stores its plane as `origin` `[x, y, z]`, `pitch_deg` and `sweep_deg` and its sections as
`key_sections`, a list of `{s, vertices, snapped, edges, handles, continuity}` (`vertices` as `[u, v]` pairs, `edges` per
edge `straight` / `curved`, `handles` per edge `null` or two `[du, dv]` offsets from its two vertices, `continuity`
per vertex `corner` / `smooth` / `symmetric`); see `duct_config.example.yaml`. The older formats (a single
`ducts.inlet` with `ducts.key_sections`, and the 5-vertex `ducts.inlet.vertices` / `snapped`) still load as one
inlet with a one-line notice, and Save migrates them. Other old keys (`shape`, `skin_fit`, `centroid`, `normal`,
`tilt_deg`, `attach_point`, `station_x`, `polygon`) are ignored with a one-line notice.

The engine face is centred on the front (min-x) end of the Geom named by `engine.geom_name`. Its diameter is
`engine.face_diameter` from the config (the fan); the Geom's bounding box is only the engine envelope (casing),
drawn as a faint outline around the face disc. The viewer prints both diameters and the ratio
face / envelope, and warns when it is outside `engine.face_to_envelope_range` (default `[0.80, 0.98]`).
Neither value is ever changed. Save never writes to the tracked example config (the button is disabled): copy it
to `duct_config.yaml` first. `--screenshot out.png` renders the scene off-screen, without Qt, instead of opening
a window.

Tests: `python -m pytest` runs without a display (the window is tested on Qt's offscreen platform, without its 3D
view). The embedded 3D view and real clicks need a desktop session:
`$env:DUCTMGR_GUI_TESTS=1; python -m pytest tests/test_qt_gui_3d.py` (opens a window briefly).

## Using your own model

```powershell
Copy-Item duct_config.example.yaml duct_config.yaml
# edit duct_config.yaml: model.vsp3_path, model.fuselage_name, engine.geom_name, engine and duct values
.venv\Scripts\python env_check.py
.venv\Scripts\python -m ductmgr.engine
.venv\Scripts\python run_explore.py
```

All inputs live in the YAML config; the example file documents every field.

## Development

Hooks run [gitleaks](https://github.com/gitleaks/gitleaks), a 1 MB file-size limit, and a local check
(`scripts/check_private_values.py`) that blocks a commit if its added lines contain a value from your private
`duct_config.yaml` (it reports file:line and the config key, never the value; it does nothing if that file is
absent). Stage files by name rather than with `git add .`.

```powershell
.venv\Scripts\python -m pre_commit install
.venv\Scripts\python -m pre_commit run --all-files
```

`.vsp3` models, `duct_config.yaml`, generated outputs and meshes are gitignored on purpose; do not commit them.

## License

Copyright (C) 2026 Víctor Sánchez Escalante.

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version (**GPL-3.0-or-later**). See [LICENSE](LICENSE).

It is distributed in the hope that it will be useful, but without any warranty. OpenVSP is a separate
program with its own license and is not part of this repository.
