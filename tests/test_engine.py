# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Phase 1 tests. Inputs are hard-coded or come from duct_config.example.yaml, never the private config."""
import copy
import math
import warnings

import pytest
import yaml

from ductmgr import engine
from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.engine import ChokedFlowError, EngineSizingWarning

# Independent reference constants (deliberately not imported from the module under test).
GAMMA = 1.4
R = 287.05287
T_SL, P_SL = 288.15, 101325.0

BASE = dict(
    face_diameter=0.5,
    hub_to_tip_ratio=0.0,
    corrected_flow_per_area=190.0,
    mach=0.6,
    altitude=5000.0,
    pressure_recovery=0.97,
    capture_margin=1.05,
)


def example_cfg() -> dict:
    cfg, _ = load_config(EXAMPLE_CONFIG)
    return cfg


def phi(m: float) -> float:
    """Isentropic flow function, written out independently of the module."""
    return math.sqrt(GAMMA / R) * m * (1 + 0.2 * m * m) ** -3


# ---- ISA ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "h, T, p, rho",
    [
        (0.0, 288.15, 101325.0, 1.2250),
        (11000.0, 216.65, 22632.06, 0.36392),
        (20000.0, 216.65, 5474.89, 0.08803),
    ],
)
def test_isa_against_table(h, T, p, rho):
    Ts, ps, rhos = engine.isa(h)
    assert Ts == pytest.approx(T, abs=0.01)
    assert ps == pytest.approx(p, rel=1e-4)
    assert rhos == pytest.approx(rho, rel=2e-4)


def test_isa_is_continuous_at_tropopause():
    below = engine.isa(11000.0 - 1e-6)
    above = engine.isa(11000.0 + 1e-6)
    assert below[1] == pytest.approx(above[1], rel=1e-8)


@pytest.mark.parametrize("h", [-1.0, 20001.0])
def test_isa_out_of_range(h):
    with pytest.raises(ValueError):
        engine.isa(h)


# ---- mass-flow function --------------------------------------------------------------------
def test_flow_function_matches_independent_formula():
    for m in (0.1, 0.5, 0.9, 1.0):
        assert engine.mass_flow_function(m) == pytest.approx(phi(m), rel=1e-12)


def test_flow_function_peaks_at_sonic():
    assert engine.max_corrected_flux() == pytest.approx(phi(1.0) * P_SL / math.sqrt(T_SL), rel=1e-12)
    assert engine.corrected_flux(0.99) < engine.max_corrected_flux()
    assert engine.corrected_flux(1.01) < engine.max_corrected_flux()


@pytest.mark.parametrize("mach", [0.01, 0.2, 0.4, 0.6, 0.8, 0.95, 0.999])
def test_mach_inversion_round_trip(mach):
    flux = engine.corrected_flux(mach)
    assert engine.mach_from_corrected_flux(flux) == pytest.approx(mach, abs=1e-9)


def test_inversion_returns_sonic_at_exact_maximum():
    assert engine.mach_from_corrected_flux(engine.max_corrected_flux()) == pytest.approx(1.0)


def test_inversion_beyond_maximum_raises():
    with pytest.raises(ChokedFlowError):
        engine.mach_from_corrected_flux(engine.max_corrected_flux() * 1.001)


# ---- face Mach -------------------------------------------------------------------------------
def test_face_mach_in_band_for_default_flux_without_warning():
    assert 0.3 <= engine.mach_from_corrected_flux(190.0) <= 0.7
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = engine.size_engine(**BASE)
    assert r.face_mach_status == "PASS"
    assert r.warnings == ()


def test_face_mach_outside_band_warns():
    kwargs = {**BASE, "corrected_flow_per_area": 225.0}
    with pytest.warns(EngineSizingWarning, match="face Mach"):
        r = engine.size_engine(**kwargs)
    assert r.face_mach > 0.7
    assert r.face_mach_status == "WARN"
    assert any("face Mach" in w for w in r.warnings)


def test_face_mach_band_is_configurable():
    r = engine.size_engine(**BASE, face_mach_range=(0.1, 0.9))
    assert r.face_mach_status == "PASS"


# ---- choking -----------------------------------------------------------------------------------
def test_choked_face_raises_clear_error_with_max_mass_flow():
    flux_max = phi(1.0) * P_SL / math.sqrt(T_SL)
    kwargs = {**BASE, "corrected_flow_per_area": flux_max * 1.1}
    with pytest.raises(ChokedFlowError) as exc:
        engine.size_engine(**kwargs)

    area = math.pi / 4 * BASE["face_diameter"] ** 2
    ram = 1 + 0.2 * BASE["mach"] ** 2
    T, p = engine.isa(BASE["altitude"])[:2]
    p0 = p * ram**3.5
    T0 = T * ram
    m_max = flux_max * area * (BASE["pressure_recovery"] * p0 / P_SL) / math.sqrt(T0 / T_SL)

    msg = str(exc.value)
    assert "choked" in msg
    assert "Maximum mass flow" in msg
    assert f"{m_max:.4g}" in msg


# ---- sizing identities ---------------------------------------------------------------------------
def test_sizing_matches_hand_calculation():
    r = engine.size_engine(**BASE)
    T = 288.15 - 0.0065 * 5000.0
    p = 101325.0 * (T / 288.15) ** 5.255876
    rho = p / (R * T)
    ram = 1 + 0.2 * 0.6**2
    T0, p0 = T * ram, p * ram**3.5
    V = 0.6 * math.sqrt(GAMMA * R * T)
    area = math.pi / 4 * 0.5**2
    mc = 190.0 * area
    m = mc * (0.97 * p0 / P_SL) / math.sqrt(T0 / T_SL)

    assert r.velocity == pytest.approx(V, rel=1e-6)
    assert r.T0 == pytest.approx(T0, rel=1e-6)
    assert r.p0 == pytest.approx(p0, rel=1e-5)
    assert r.face_area == pytest.approx(area, rel=1e-12)
    assert r.corrected_flow == pytest.approx(mc, rel=1e-12)
    assert r.mass_flow == pytest.approx(m, rel=1e-5)
    assert r.capture_area_total == pytest.approx(m / (rho * V) * 1.05, rel=1e-5)
    assert r.capture_area_per_inlet == pytest.approx(r.capture_area_total / 2, rel=1e-12)
    assert phi(r.face_mach) * P_SL / math.sqrt(T_SL) == pytest.approx(190.0, rel=1e-9)


def test_hub_to_tip_ratio_reduces_face_area_and_flow():
    full = engine.size_engine(**BASE)
    hub = engine.size_engine(**{**BASE, "hub_to_tip_ratio": 0.4})
    assert hub.face_area == pytest.approx(full.face_area * (1 - 0.4**2), rel=1e-12)
    assert hub.mass_flow == pytest.approx(full.mass_flow * (1 - 0.4**2), rel=1e-12)
    assert hub.face_mach == pytest.approx(full.face_mach, rel=1e-12)


def test_capture_area_scales_with_margin_and_inlet_count():
    a = engine.size_engine(**BASE)
    b = engine.size_engine(**{**BASE, "capture_margin": 1.05 * 1.2}, n_inlets=3)
    assert b.capture_area_total == pytest.approx(a.capture_area_total * 1.2, rel=1e-12)
    assert b.capture_area_per_inlet == pytest.approx(b.capture_area_total / 3, rel=1e-12)


# ---- specific-thrust check -----------------------------------------------------------------------
def test_specific_thrust_sls_static_hand_calculation():
    # SLS static: T0 = 288.15 K, p0 = 101325 Pa, so m_ref = corrected flow * recovery exactly.
    area = math.pi / 4 * 0.5**2  # 0.196349541 m^2
    m_ref = 190.0 * area * 0.97  # 36.18722 kg/s
    r = engine.size_engine(**BASE, reference_thrust=10000.0, specific_thrust_range=(250.0, 300.0))
    assert r.reference_condition == (0.0, 0.0)
    assert r.reference_mass_flow == pytest.approx(36.18722, abs=1e-5)
    assert r.reference_mass_flow == pytest.approx(m_ref, rel=1e-9)
    assert r.specific_thrust == pytest.approx(276.3406, abs=1e-4)
    assert r.specific_thrust_status == "PASS"


def test_specific_thrust_independent_of_design_point():
    kwargs = dict(reference_thrust=10000.0, specific_thrust_range=(250.0, 300.0))
    a = engine.size_engine(**BASE, **kwargs)
    b = engine.size_engine(**{**BASE, "mach": 0.8, "altitude": 9000.0}, **kwargs)
    assert a.mass_flow != pytest.approx(b.mass_flow)  # design-point flow differs...
    assert a.specific_thrust == pytest.approx(b.specific_thrust, rel=1e-12)  # ...the check does not


def test_specific_thrust_non_default_reference_condition():
    # Mach 0.3 at sea level: T0/T = 1.018, p0/p = 1.018^3.5, so m = mc * 0.97 * 1.018^3.
    area = math.pi / 4 * 0.5**2
    m_ref = 190.0 * area * 0.97 * 1.018**3  # 38.17672 kg/s
    r = engine.size_engine(
        **BASE,
        reference_thrust=10000.0,
        reference_thrust_condition=(0.3, 0.0),
        specific_thrust_range=(200.0, 300.0),
    )
    assert r.reference_mass_flow == pytest.approx(38.17672, abs=1e-5)
    assert r.reference_mass_flow == pytest.approx(m_ref, rel=1e-9)
    assert r.specific_thrust == pytest.approx(10000.0 / m_ref, rel=1e-9)


def test_specific_thrust_warn_names_reference_condition():
    with pytest.warns(EngineSizingWarning, match="reference condition"):
        bad = engine.size_engine(**BASE, reference_thrust=20000.0, specific_thrust_range=(250.0, 300.0))
    assert bad.specific_thrust_status == "WARN"
    assert bad.specific_thrust == pytest.approx(20000.0 / 36.18722, rel=1e-6)


def test_negative_reference_mach_rejected():
    with pytest.raises(ValueError, match="Mach"):
        engine.size_engine(**BASE, reference_thrust_condition=(-0.1, 0.0))


def test_specific_thrust_skipped_when_inputs_missing():
    assert engine.size_engine(**BASE).specific_thrust_status == "SKIPPED"
    assert engine.size_engine(**BASE, reference_thrust=1000.0).specific_thrust_status == "SKIPPED"
    assert engine.size_engine(**BASE, specific_thrust_range=(1.0, 2.0)).specific_thrust_status == "SKIPPED"


# ---- input validation ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key, value",
    [
        ("face_diameter", 0.0),
        ("hub_to_tip_ratio", 1.0),
        ("hub_to_tip_ratio", -0.1),
        ("corrected_flow_per_area", 0.0),
        ("mach", 0.0),
        ("altitude", 25000.0),
        ("pressure_recovery", 1.2),
        ("pressure_recovery", 0.0),
        ("capture_margin", 0.0),
    ],
)
def test_invalid_inputs_rejected(key, value):
    with pytest.raises(ValueError):
        engine.size_engine(**{**BASE, key: value})


# ---- config path -------------------------------------------------------------------------------------
def test_example_config_sizes_and_passes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = engine.size_from_config(example_cfg())
    assert r.face_mach_status == "PASS"
    assert r.specific_thrust_status == "PASS"
    assert r.n_inlets == 2
    assert r.capture_area_per_inlet == pytest.approx(r.capture_area_total / 2)


def test_missing_config_value_named_in_error():
    cfg = copy.deepcopy(example_cfg())
    cfg["engine"]["face_diameter"] = None
    with pytest.raises(ValueError, match=r"engine\.face_diameter"):
        engine.size_from_config(cfg)


def test_optional_config_keys_default():
    cfg = copy.deepcopy(example_cfg())
    del cfg["ducts"]["n_inlets"]
    del cfg["engine"]["face_mach_range"]
    del cfg["engine"]["reference_thrust_condition"]
    r = engine.size_from_config(cfg)
    assert r.n_inlets == 2
    assert r.face_mach_range == (0.3, 0.7)
    assert r.reference_condition == (0.0, 0.0)  # defaults to sea-level static
    cfg["engine"]["reference_thrust"] = None
    assert engine.size_from_config(cfg).specific_thrust_status == "SKIPPED"


def test_reference_condition_read_from_config():
    cfg = copy.deepcopy(example_cfg())
    cfg["engine"]["reference_thrust_condition"] = {"mach": 0.3, "altitude": 0.0}
    assert engine.size_from_config(cfg).reference_condition == (0.3, 0.0)
    del cfg["engine"]["reference_thrust_condition"]["altitude"]
    with pytest.raises(ValueError, match="reference_thrust_condition.altitude"):
        engine.size_from_config(cfg)


# ---- CLI ------------------------------------------------------------------------------------------------
def test_cli_prints_summary(capsys):
    assert engine.main(["--config", str(EXAMPLE_CONFIG)]) == 0
    out = capsys.readouterr().out
    for label in (
        "Static temperature",
        "Velocity",
        "Total pressure",
        "Face area",
        "Corrected mass flow",
        "Actual mass flow",
        "Face Mach",
        "Capture area, total",
        "per inlet",
        "Reference condition",
        "Mass flow at reference",
        "Specific thrust",
    ):
        assert label in out
    assert "PASS" in out


def test_cli_choked_config_reports_error(tmp_path, capsys):
    cfg = example_cfg()
    cfg["engine"]["corrected_flow_per_area"] = 300.0
    path = tmp_path / "choked.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert engine.main(["--config", str(path)]) == 1
    err = capsys.readouterr().err
    assert "choked" in err and "Maximum mass flow" in err


def test_cli_missing_config_file(tmp_path, capsys):
    assert engine.main(["--config", str(tmp_path / "nope.yaml")]) == 1
    assert "not found" in capsys.readouterr().err
