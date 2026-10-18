# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Interactive viewer: see the aircraft and place the inlets before any duct exists.

    python run_explore.py [--config PATH] [--screenshot OUT.png]

Click the fuselage to place the right inlet (the left one is mirrored about XZ), adjust the sliders,
press ``s`` to save the inlet definition into duct_config.yaml and ``r`` to reload it. With
``--screenshot`` the scene is rendered off-screen to a PNG and the program exits.
"""
import argparse
import sys

import pyvista as pv

from ductmgr.config import EXAMPLE_CONFIG, PRIVATE_CONFIG, add_config_argument, load_config

WINDOW_SIZE = (1500, 900)
TITLE = "VSP Duct Manager - inlet placement"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--screenshot", metavar="PNG", help="render off-screen to this file and exit")
    args = parser.parse_args(argv)

    try:
        cfg, config_path = load_config(args.config)
        from ductmgr.viz import ExploreApp, load_scene

        scene = load_scene(cfg, config_path)
    except (FileNotFoundError, LookupError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # `s` writes back into the config in use, but never into the tracked example config.
    save_target = None if config_path == EXAMPLE_CONFIG.resolve() else config_path

    print(f"Config: {config_path.name}   " + (
        f"(s saves to {save_target.name})" if save_target else f"(s disabled: copy it to {PRIVATE_CONFIG.name} to save)"
    ))
    for msg in scene.warnings:
        print(msg)

    plotter = pv.Plotter(window_size=WINDOW_SIZE, title=TITLE, off_screen=bool(args.screenshot))
    app = ExploreApp(plotter, scene, cfg, config_path, save_target)
    if args.screenshot:
        plotter.screenshot(args.screenshot)
        plotter.close()
        print(f"Wrote {args.screenshot}")
        return 0
    app.add_controls()
    plotter.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
