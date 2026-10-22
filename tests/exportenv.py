# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Test environments for the export: a real OpenVSP model (the example fuselage and engine), and explorer models with
1..N inlets placed on its skin."""
import copy

import numpy as np
import openvsp as vsp
import yaml

from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.explore_model import InletModel
from ductmgr.viz import load_scene
from make_example_model import build_example_model


def build_env(folder, engine_z: float | None = None):
    """(cfg, config_path, model_path): the example config pointing at a fresh example model; ``engine_z`` raises the
    engine (and so the engine face) to that height, which makes a nearly straight duct from a dorsal inlet."""
    cfg, _ = load_config(EXAMPLE_CONFIG)
    model = folder / "model.vsp3"
    build_example_model(model, cfg)
    if engine_z is not None:
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(str(model))
        for gid in vsp.FindGeoms():
            if vsp.GetGeomName(gid) == cfg["engine"]["geom_name"]:
                vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", engine_z)
        vsp.Update()
        vsp.WriteVSPFile(str(model), vsp.SET_ALL)
    cfg["model"]["vsp3_path"] = str(model)
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg, path, model


def skin_point(scene, x: float, y: float) -> list[float]:
    """The dorsal skin point above (x, y)."""
    z = np.array(scene.fuselage.ray_trace([x, y, 5.0], [x, y, -5.0])[0])[:, 2].max()
    return [float(x), float(y), float(z)]


def inlet(scene, name: str, x: float, y: float, mirror: bool = False, fraction=None) -> dict:
    return {"name": name, "origin": skin_point(scene, x, y), "pitch_deg": 0.0, "sweep_deg": 0.0, "mirror": mirror,
            "flow_fraction": fraction, "key_sections": None}


def explorer_model(cfg, path, inlets, symmetry="xz", junction=None):
    """An ``InletModel`` on the example model with the given inlets (a default sketch is made for each)."""
    scene = load_scene(cfg, path)
    c = copy.deepcopy(cfg)
    c["ducts"]["symmetry"] = symmetry
    c["ducts"]["inlets"] = [i(scene) if callable(i) else i for i in inlets]
    c["ducts"]["trunk"] = {}
    if junction is not None:
        c["ducts"]["junction"]["point"] = junction
    return InletModel(scene, c, path, None)
