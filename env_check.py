# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Environment check: import openvsp, open the model from the config, list every Geom."""
import argparse
import sys

from ductmgr.config import add_config_argument, load_config, resolve_model_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args(argv)

    import openvsp as vsp

    cfg, config_path = load_config(args.config)
    model_cfg = cfg["model"]
    vsp3 = resolve_model_path(cfg, config_path)

    print(f"Python  : {sys.version.split()[0]}")
    print(f"OpenVSP : {vsp.GetVSPVersion()}")
    print(f"Config  : {config_path.name}")
    print(f"Model   : {vsp3.name}")

    if not vsp3.is_file():
        print(f"\nModel file not found: {vsp3}")
        print("For the example config, create the dummy model with: python make_example_model.py")
        return 2

    vsp.ClearVSPModel()
    vsp.ReadVSPFile(str(vsp3))
    vsp.Update()

    geoms = vsp.FindGeoms()
    print(f"\n{len(geoms)} geom(s):\n")
    header = f"{'name':<28}{'type':<16}{'id':<12}{'bbox min (x, y, z)':<32}bbox max (x, y, z)"
    print(header)
    print("-" * len(header))
    names = []
    for gid in geoms:
        name = vsp.GetGeomName(gid)
        names.append(name)
        lo = vsp.GetGeomBBoxMin(gid)
        hi = vsp.GetGeomBBoxMax(gid)
        lo_s = f"({lo.x():.3f}, {lo.y():.3f}, {lo.z():.3f})"
        hi_s = f"({hi.x():.3f}, {hi.y():.3f}, {hi.z():.3f})"
        print(f"{name:<28}{vsp.GetGeomTypeName(gid):<16}{gid:<12}{lo_s:<32}{hi_s}")

    fus = model_cfg.get("fuselage_name")
    print()
    if fus is None:
        print("model.fuselage_name is not set in the config.")
    elif fus in names:
        print(f"model.fuselage_name = '{fus}': found.")
    else:
        print(f"WARNING: model.fuselage_name = '{fus}' not found in the model.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
