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
Qt window (PySide6 + pyvistaqt) with the 3D view in the centre and a compact dark panel on the right; nothing is
drawn inside the 3D view except the scene itself (translucent fuselage, other components as grey context, the
engine, the engine-face disc and the inlets).

- **Placement is explicit.** Clicking the geometry does nothing by default. Press **Place inlet** (`P`): the next
  click on the fuselage places the right inlet (the left one is its mirror image about XZ; a click on the left
  side is mirrored onto the right) and place mode switches itself off. `Esc` cancels. The status bar shows
  `View` or `Placing inlet - click on fuselage, Esc to cancel`. Dragging still rotates the camera.
- **Lock inlet** disables placement and nudges.
- **Undo / redo** (`Ctrl+Z` / `Ctrl+Y`) cover every change, placement and parameters alike (up to 200 steps;
  rapid edits of one value merge into one step).
- **Panel** (collapsible groups): position x/y/z with a nudge-step selector (the inlet snaps to the skin), the
  shape ratios `width_bottom`, `width_top`, `height`, `apex_height` (0 gives a quadrilateral) and `skew`, the
  tilt, the skin-fit mode, and read-only monospace readouts: centroid, capture area target vs actual, aspect
  ratio, distance to the face, (dy, dz) offset, offset/length, maximum skin gap, opening height and the engine
  face/envelope check, with WARN in amber and PASS in green. Spin boxes take the mouse wheel only while focused.
- **Save to config** (`Ctrl+S`) writes the inlet into `duct_config.yaml`, keeping its comments and every other
  key; **Revert to saved** re-reads it. The title bar shows `*` while there are unsaved changes, and closing the
  window then asks first.
- **View** toolbar: side / top / front / iso (`1`-`4`), and toggles for the context components and the engine
  envelope.

The polygon is always scaled to the per-inlet capture area from the sizing step.

**Skin fit** (`ducts.inlet.skin_fit`): `none` is the planar polygon centred on the clicked point. `bottom_edge`
(the example default) keeps the outline planar but replaces its bottom edge with the curve where the inlet plane
cuts the fuselage skin, so the opening sits on the skin; the other vertices keep the shape ratios above it.
`flush` projects the whole outline onto the skin along the inlet axis (a non-planar outline) and always warns that
a flush inlet means low pressure recovery and boundary-layer ingestion. For `bottom_edge` and `flush` the shape is
scaled iteratively until the outline area projected normal to the freestream matches the per-inlet capture area
(within 0.5 %). A flush outline on skin that is almost parallel to the freestream has almost no frontal area, so
the fit can fail there; the viewer says so. The readout shows the mode, the maximum gap between outline and skin,
and the opening height above the skin.

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
