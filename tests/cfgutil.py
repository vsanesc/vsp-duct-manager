# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Helpers to reshape the example config (ducts.inlets format) in tests."""


def first(cfg: dict) -> dict:
    """The first inlet entry of ``ducts.inlets``."""
    return cfg["ducts"]["inlets"][0]


def unplace(cfg: dict) -> dict:
    """The config with its first inlet not placed yet (no origin, no sketch)."""
    entry = first(cfg)
    entry["origin"] = None
    entry["key_sections"] = None
    return cfg


def to_legacy(cfg: dict) -> dict:
    """The old single-inlet format: ducts.inlet (plane) + ducts.key_sections, no ducts.inlets / symmetry."""
    ducts = cfg["ducts"]
    entry = ducts.pop("inlets")[0]
    ducts.pop("symmetry", None)
    ducts["inlet"] = {"origin": entry["origin"], "pitch_deg": entry["pitch_deg"], "sweep_deg": entry["sweep_deg"]}
    ducts["key_sections"] = entry["key_sections"]
    return cfg


def to_five_vertex(cfg: dict) -> dict:
    """The oldest format: ducts.inlet.vertices / snapped and no key sections."""
    to_legacy(cfg)
    section = cfg["ducts"].pop("key_sections")[0]
    cfg["ducts"]["inlet"]["vertices"] = section["vertices"]
    cfg["ducts"]["inlet"]["snapped"] = section["snapped"]
    return cfg
