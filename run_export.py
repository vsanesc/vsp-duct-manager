# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Export the duct into OpenVSP and verify it.

    python run_export.py [--config PATH] [--out PATH] [--no-verify]

Reads the design from the config (the same one the explorer saves), opens the input model, deletes only the Geoms whose
name starts with ``model.duct_prefix``, creates the duct Stacks (a positive and a negative per inlet branch and for the
trunk, with Edit Curve sections) and saves the model as ``<name>_ducts.vsp3`` next to the input; the input file is never
written. The saved file is then reopened and checked (see ``ductmgr/exportcheck.py``). Exit status: 0 when every check
passes (warnings included), 2 when a check fails, 1 for an error before the export.
"""
import argparse
import sys

from ductmgr.config import add_config_argument, load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--out", metavar="VSP3", help="output file (default: <input>_ducts.vsp3); never the input file")
    parser.add_argument("--no-verify", action="store_true", help="skip the verification of the saved file")
    args = parser.parse_args(argv)

    try:
        cfg, config_path = load_config(args.config)
        from ductmgr import exportrun
        from ductmgr.explore_model import InletModel
        from ductmgr.viz import load_scene

        scene = load_scene(cfg, config_path)
        model = InletModel(scene, cfg, config_path, None)
        for msg in scene.warnings:
            print(msg)
        model.ensure_built()
        reason = model.export_blocker()
        if reason:
            print(f"error: cannot export: {reason}", file=sys.stderr)
            return 1
        inputs = model.export_inputs()
        result = exportrun.export_design(**inputs, out_path=args.out, verify=False if args.no_verify else None,
                                         progress=lambda text: print(f"... {text}", flush=True))
    except (FileNotFoundError, LookupError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.text)
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
