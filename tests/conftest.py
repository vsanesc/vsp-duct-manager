# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Shared fixtures. Nothing here reads the private duct_config.yaml: the model is built from the example."""
import yaml
import pytest

from ductmgr.config import EXAMPLE_CONFIG, load_config
from make_example_model import build_example_model


@pytest.fixture(scope="session")
def example_env(tmp_path_factory):
    """(cfg, config_path, model_path): the example config pointing at a freshly built example model."""
    folder = tmp_path_factory.mktemp("example_model")
    cfg, _ = load_config(EXAMPLE_CONFIG)
    model = folder / "model.vsp3"
    build_example_model(model, cfg)
    cfg["model"]["vsp3_path"] = str(model)
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg, path, model
