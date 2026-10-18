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


def resolve_model_path(cfg: dict, config_path: Path) -> Path:
    """model.vsp3_path, relative paths resolved against the config file's folder."""
    p = Path(cfg["model"]["vsp3_path"])
    return p if p.is_absolute() else config_path.parent / p
