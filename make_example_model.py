# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Create the dummy model used by duct_config.example.yaml: a plain fuselage and a stand-in engine pod.

Uses duct_config.example.yaml unless --config is given. Refuses to overwrite an existing
file, so it can never clobber a real model.
"""
import argparse
import sys
from pathlib import Path

from ductmgr.config import EXAMPLE_CONFIG, add_config_argument, load_config, resolve_model_path

FUSELAGE_LENGTH = 8.0  # m
ENGINE_ENVELOPE_OVER_FACE = 1.15  # the stand-in pod (casing) is this much wider than the fan face
ENGINE_LENGTH_OVER_DIAMETER = 3.0  # the stand-in engine pod is this many envelope diameters long
ENGINE_X_FRACTION = 0.6  # of the fuselage length, used when the config has no engine face position


def build_example_model(out: Path, cfg: dict) -> None:
    """Write a fuselage and an engine pod, named and sized from ``cfg``, to ``out``."""
    import openvsp as vsp

    vsp.ClearVSPModel()
    fus = vsp.AddGeom("FUSELAGE")
    vsp.SetGeomName(fus, cfg["model"]["fuselage_name"])
    vsp.SetParmVal(fus, "Length", "Design", FUSELAGE_LENGTH)

    diameter = ENGINE_ENVELOPE_OVER_FACE * cfg["engine"]["face_diameter"]
    centre = ((cfg.get("ducts") or {}).get("engine_face") or {}).get("centre")
    x_face = centre[0] if centre else ENGINE_X_FRACTION * FUSELAGE_LENGTH
    pod = vsp.AddGeom("POD")
    vsp.SetGeomName(pod, cfg["engine"]["geom_name"])
    vsp.SetParmVal(pod, "Length", "Design", ENGINE_LENGTH_OVER_DIAMETER * diameter)
    # OpenVSP's pod fineness ratio is length / radius, so diameter = 2 * length / FineRatio
    vsp.SetParmVal(pod, "FineRatio", "Design", 2.0 * ENGINE_LENGTH_OVER_DIAMETER)
    vsp.SetParmVal(pod, "X_Rel_Location", "XForm", x_face)
    vsp.Update()
    vsp.WriteVSPFile(str(out), vsp.SET_ALL)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args(argv)

    cfg, config_path = load_config(args.config or EXAMPLE_CONFIG)
    out = resolve_model_path(cfg, config_path)
    if out.exists():
        print(f"{out.name} already exists; not overwriting.")
        return 1

    build_example_model(out, cfg)
    print(
        f"Wrote {out.name} (fuselage '{cfg['model']['fuselage_name']}', {FUSELAGE_LENGTH} m; "
        f"engine pod '{cfg['engine']['geom_name']}')."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
