# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Create the dummy model used by duct_config.example.yaml: a single default fuselage.

Uses duct_config.example.yaml unless --config is given. Refuses to overwrite an existing
file, so it can never clobber a real model.
"""
import argparse
import sys

from ductmgr.config import EXAMPLE_CONFIG, add_config_argument, load_config, resolve_model_path

FUSELAGE_LENGTH = 8.0  # m


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args(argv)

    import openvsp as vsp

    cfg, config_path = load_config(args.config or EXAMPLE_CONFIG)
    out = resolve_model_path(cfg, config_path)
    if out.exists():
        print(f"{out.name} already exists; not overwriting.")
        return 1

    vsp.ClearVSPModel()
    fus = vsp.AddGeom("FUSELAGE")
    vsp.SetGeomName(fus, cfg["model"]["fuselage_name"])
    vsp.SetParmVal(fus, "Length", "Design", FUSELAGE_LENGTH)
    vsp.Update()
    vsp.WriteVSPFile(str(out), vsp.SET_ALL)
    print(f"Wrote {out.name} (fuselage '{cfg['model']['fuselage_name']}', {FUSELAGE_LENGTH} m).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
