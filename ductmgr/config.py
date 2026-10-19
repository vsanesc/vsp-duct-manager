# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Config loading shared by every entry point.

Resolution order: ``--config <path>``, then ``duct_config.yaml`` (private, gitignored),
then ``duct_config.example.yaml`` (public, generic dummy values).
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_CONFIG = ROOT / "duct_config.yaml"
EXAMPLE_CONFIG = ROOT / "duct_config.example.yaml"


def default_config_path() -> Path:
    return PRIVATE_CONFIG if PRIVATE_CONFIG.is_file() else EXAMPLE_CONFIG


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: duct_config.yaml if present, else duct_config.example.yaml)",
    )


def load_config(path: Path | None = None) -> tuple[dict, Path]:
    """Return ``(config, resolved_path)``."""
    path = Path(path) if path is not None else default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f), path.resolve()


SAVE_DECIMALS = 6


def _round_values(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, SAVE_DECIMALS)
    if isinstance(value, dict):
        return {k: _round_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_values(v) for v in value]
    return value


def save_inlet_definition(inlet: dict | None, target: Path | None = None, key_sections: list | None = None,
                          remove_inlet_keys=(), ducts_values: dict | None = None, remove_ducts_keys=()) -> Path:
    """Write ``inlet`` into ``ducts.inlet`` of the private config (skipped when None), keeping comments and every
    other key. ``remove_ducts_keys`` names superseded keys to delete from ``ducts`` (the old ``inlet`` and
    ``key_sections``). In ``ducts_values`` the key ``inlets`` is a list of mappings (one per inlet, with its own
    ``key_sections``) that replaces ``ducts.inlets`` wholesale.

    ``ducts_values`` is a (nested) mapping merged into ``ducts`` (junction point, centreline, area points, wall
    thickness, lip); ``None`` values are written as null.

    ``key_sections`` (a list of mappings: ``s``, ``vertices``, ``snapped``, ``edges``, ``handles``, ``continuity``)
    replaces ``ducts.key_sections`` wholesale when given. ``remove_inlet_keys`` names superseded keys to delete
    from ``ducts.inlet`` (the old 5-vertex ``vertices`` / ``snapped``).

    ``target`` defaults to ``duct_config.yaml``. If it does not exist it is first created as a copy of the
    example config. The tracked example config is never written.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    target = Path(target) if target is not None else PRIVATE_CONFIG
    if target.resolve() == EXAMPLE_CONFIG.resolve():
        raise PermissionError(
            f"refusing to write the tracked example config; copy it to {PRIVATE_CONFIG.name} first"
        )
    if not target.is_file():
        target.write_bytes(EXAMPLE_CONFIG.read_bytes())

    raw = target.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    yaml = YAML()  # round-trip: keeps comments, order and quoting
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    doc = yaml.load(raw.decode("utf-8"))

    def mapping(parent, key):
        if parent.get(key) is None:
            parent[key] = CommentedMap()
        return parent[key]

    def flow_seq(items):
        seq = CommentedSeq(items)
        seq.fa.set_flow_style()  # [a, b, c] on one line
        return seq

    def assign_list(slot, key, value):
        """Set a list value. An existing list of the same shape is updated in place, which keeps its comments."""
        existing = slot.get(key)
        nested = bool(value) and all(isinstance(item, list) for item in value)
        if isinstance(existing, CommentedSeq) and len(existing) == len(value):
            for i, item in enumerate(value):
                if nested and isinstance(existing[i], CommentedSeq) and len(existing[i]) == len(item):
                    for j, x in enumerate(item):
                        existing[i][j] = x
                else:
                    existing[i] = flow_seq(item) if isinstance(item, list) else item
        elif nested:
            slot[key] = CommentedSeq([flow_seq(item) for item in value])  # one [u, v] pair per line
        else:
            slot[key] = flow_seq(value)

    def assign_tree(slot, mapping):
        """Set nested mapping values in place (keeping the comments of keys that already exist); lists are flow."""
        for key, value in mapping.items():
            if key == "inlets" and isinstance(value, list):
                slot[key] = CommentedSeq([entry_map(e) for e in value])
            elif isinstance(value, dict):
                assign_tree(mapping_of(slot, key), value)
            elif isinstance(value, list):
                assign_list(slot, key, value)
            else:
                slot[key] = value

    def mapping_of(parent, key):
        if not isinstance(parent.get(key), CommentedMap):
            parent[key] = CommentedMap()
        return parent[key]

    def section_map(entry):
        """One key section: lists in flow style, one [u, v] vertex / handle pair per line."""
        out = CommentedMap()
        for key, value in entry.items():
            if key == "vertices":
                out[key] = CommentedSeq([flow_seq(v) for v in value])
            elif key == "handles":
                out[key] = flow_seq([None if h is None else flow_seq([flow_seq(h[0]), flow_seq(h[1])])
                                     for h in value])  # one line: [null, [[du, dv], [du, dv]], null]
            elif isinstance(value, list):
                out[key] = flow_seq(value)
            else:
                out[key] = value
        return out

    def entry_map(entry):
        """One inlet: its key sections as section maps, other lists in flow style."""
        out = CommentedMap()
        for key, value in entry.items():
            if key == "key_sections":
                out[key] = CommentedSeq([section_map(e) for e in value])
            elif key == "area_points" and value is not None:
                out[key] = CommentedSeq([flow_seq(p) for p in value])
            elif isinstance(value, list):
                out[key] = flow_seq(value)
            else:
                out[key] = value
        return out

    ducts = mapping(doc, "ducts")
    for key in remove_ducts_keys:
        if key in ducts:
            del ducts[key]
    if inlet is not None:
        slot = mapping(ducts, "inlet")
        for key in remove_inlet_keys:
            if key in slot:
                del slot[key]
        for key, value in _round_values(inlet).items():
            if isinstance(value, dict):
                sub = mapping(slot, key)
                for k, v in value.items():
                    sub[k] = v
            elif isinstance(value, list):
                assign_list(slot, key, value)
            else:
                slot[key] = value
    if key_sections is not None:
        ducts["key_sections"] = CommentedSeq([section_map(e) for e in _round_values(key_sections)])
    if ducts_values:
        assign_tree(ducts, _round_values(ducts_values))

    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline=newline) as f:
        yaml.dump(doc, f)
    tmp.replace(target)
    return target


def resolve_model_path(cfg: dict, config_path: Path) -> Path:
    """model.vsp3_path, relative paths resolved against the config file's folder."""
    p = Path(cfg["model"]["vsp3_path"])
    return p if p.is_absolute() else config_path.parent / p
