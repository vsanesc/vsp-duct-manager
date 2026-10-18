# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Inlet explorer: see the aircraft and place the inlets before any duct exists.

    python run_explore.py [--config PATH] [--screenshot OUT.png]

A window opens with the 3D scene and a control panel. Clicking the geometry does nothing until you press
"Place inlet" (P); the next click on the fuselage then places the right inlet (the left one is mirrored about
XZ) and the mode switches off again. Save to config (Ctrl+S) writes the inlet into duct_config.yaml.
With ``--screenshot`` the scene is rendered off-screen to a PNG and the program exits (no window, no Qt).
"""
import argparse
import sys

from ductmgr.config import EXAMPLE_CONFIG, PRIVATE_CONFIG, add_config_argument, load_config

WINDOW_SIZE = (1440, 880)  # screenshot size


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--screenshot", metavar="PNG", help="render off-screen to this file and exit")
    args = parser.parse_args(argv)

    try:
        cfg, config_path = load_config(args.config)
        from ductmgr.explore_model import InletModel
        from ductmgr.viz import load_scene

        scene = load_scene(cfg, config_path)
    except (FileNotFoundError, LookupError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Save writes back into the config in use, but never into the tracked example config.
    save_target = None if config_path == EXAMPLE_CONFIG.resolve() else config_path
    model = InletModel(scene, cfg, config_path, save_target)

    print(f"Config: {config_path.name}   " + (
        f"(Save writes to {save_target.name})" if save_target else
        f"(Save disabled: copy it to {PRIVATE_CONFIG.name} to save)"
    ))
    check = scene.engine_check
    if check is not None and check.status == "PASS":
        print(check.message)  # a WARN is already among scene.warnings
    for msg in scene.warnings:
        print(msg)
    if model.notice:
        print(f"NOTE {model.notice}")

    if args.screenshot:
        import pyvista as pv

        from ductmgr.viz import SceneView

        plotter = pv.Plotter(window_size=WINDOW_SIZE, off_screen=True)
        view = SceneView(plotter, scene)
        view.update_inlet(model)
        plotter.screenshot(args.screenshot)
        plotter.close()
        print(f"Wrote {args.screenshot}")
        return 0

    try:
        from ductmgr.qt_explorer import run
    except ImportError as exc:
        print(f"error: the Qt viewer needs PySide6 and pyvistaqt ({exc}); see requirements.txt", file=sys.stderr)
        return 1
    return run(scene, model)


if __name__ == "__main__":
    sys.exit(main())
