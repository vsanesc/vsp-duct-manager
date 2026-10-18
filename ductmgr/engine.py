# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Engine and capture-area sizing (Phase 1).

Given an engine face, a design point and a corrected flow per unit face area, compute the ISA
conditions, free-stream and total conditions, engine mass flow, face Mach number and the
free-stream capture area. Run ``python -m ductmgr.engine`` to print the summary for a config.

Units: metres, seconds, kg, K, Pa.
"""
import argparse
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from scipy.optimize import brentq

from ductmgr.config import add_config_argument, load_config

# ---- physical constants ------------------------------------------------------
GAMMA = 1.4  # ratio of specific heats of air
R_AIR = 287.05287  # J/(kg K)
G0 = 9.80665  # m/s^2
T_REF = 288.15  # K, sea-level ISA: reference for corrected flow
P_REF = 101325.0  # Pa, sea-level ISA: reference for corrected flow

# ---- ISA layers (troposphere, then isothermal lower stratosphere) -------------
H_TROPOPAUSE = 11000.0  # m
H_MAX = 20000.0  # m, upper limit of this ISA implementation
LAPSE_RATE = -0.0065  # K/m in the troposphere

# ---- defaults for optional config keys -----------------------------------------
DEFAULT_N_INLETS = 2
DEFAULT_FACE_MACH_RANGE = (0.3, 0.7)

_CHOKE_RTOL = 1e-12  # a flux this close above the M=1 maximum is treated as exactly sonic


class EngineSizingWarning(UserWarning):
    """A sizing result is outside its expected range (flagged, never silently corrected)."""


class ChokedFlowError(ValueError):
    """The requested corrected flow cannot pass the engine face at subsonic Mach."""


# ---- atmosphere -----------------------------------------------------------------
def isa(altitude: float) -> tuple[float, float, float]:
    """ISA static temperature [K], pressure [Pa] and density [kg/m^3] at geopotential altitude [m]."""
    if not 0.0 <= altitude <= H_MAX:
        raise ValueError(f"altitude {altitude} m outside the supported ISA range 0..{H_MAX:.0f} m")
    exponent = -G0 / (LAPSE_RATE * R_AIR)
    if altitude <= H_TROPOPAUSE:
        T = T_REF + LAPSE_RATE * altitude
        p = P_REF * (T / T_REF) ** exponent
    else:
        T11 = T_REF + LAPSE_RATE * H_TROPOPAUSE
        p11 = P_REF * (T11 / T_REF) ** exponent
        T = T11
        p = p11 * math.exp(-G0 * (altitude - H_TROPOPAUSE) / (R_AIR * T11))
    return T, p, p / (R_AIR * T)


# ---- isentropic mass-flow function ------------------------------------------------
def mass_flow_function(mach: float) -> float:
    """Isentropic flow function m*sqrt(T0)/(A*p0) [kg sqrt(K)/(s m^2 Pa)] at a given Mach."""
    g = GAMMA
    return math.sqrt(g / R_AIR) * mach * (1.0 + 0.5 * (g - 1.0) * mach**2) ** (-(g + 1.0) / (2.0 * (g - 1.0)))


def corrected_flux(mach: float) -> float:
    """Corrected flow per unit area [kg/s/m^2] (referred to sea-level ISA) at a given Mach."""
    return mass_flow_function(mach) * P_REF / math.sqrt(T_REF)


def max_corrected_flux() -> float:
    """Choked (M = 1) corrected flow per unit area [kg/s/m^2]."""
    return corrected_flux(1.0)


def mach_from_corrected_flux(flux: float) -> float:
    """Invert :func:`corrected_flux` for the subsonic root. Raises :class:`ChokedFlowError` if none exists."""
    if flux < 0.0:
        raise ValueError("corrected flow per unit area must be non-negative")
    flux_max = max_corrected_flux()
    if flux > flux_max * (1.0 + _CHOKE_RTOL):
        raise ChokedFlowError(
            f"no subsonic root: corrected flow per unit area {flux:.6g} kg/s/m^2 exceeds the "
            f"choked maximum {flux_max:.6g} kg/s/m^2"
        )
    if flux >= flux_max:
        return 1.0
    return brentq(lambda m: corrected_flux(m) - flux, 0.0, 1.0, xtol=1e-14, rtol=1e-14)


# ---- sizing --------------------------------------------------------------------------
@dataclass(frozen=True)
class SizingResult:
    # atmosphere and flight
    altitude: float
    T_static: float
    p_static: float
    rho: float
    mach: float
    velocity: float
    T0: float
    p0: float
    # engine face
    face_diameter: float
    hub_to_tip_ratio: float
    face_area: float
    pressure_recovery: float
    p0_face: float
    corrected_flux: float
    corrected_flow: float
    mass_flow: float
    face_mach: float
    face_mach_range: tuple[float, float]
    face_mach_status: str
    # capture
    capture_margin: float
    n_inlets: int
    streamtube_area: float
    capture_area_total: float
    capture_area_per_inlet: float
    # optional specific-thrust check
    thrust: float | None
    specific_thrust: float | None
    specific_thrust_range: tuple[float, float] | None
    specific_thrust_status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return format_summary(self)


def _check_range(value: float, lo: float, hi: float) -> str:
    return "PASS" if lo <= value <= hi else "WARN"


def size_engine(
    *,
    face_diameter: float,
    hub_to_tip_ratio: float,
    corrected_flow_per_area: float,
    mach: float,
    altitude: float,
    pressure_recovery: float,
    capture_margin: float,
    n_inlets: int = DEFAULT_N_INLETS,
    face_mach_range: tuple[float, float] = DEFAULT_FACE_MACH_RANGE,
    reference_thrust: float | None = None,
    specific_thrust_range: tuple[float, float] | None = None,
) -> SizingResult:
    if face_diameter <= 0.0:
        raise ValueError("face_diameter must be positive")
    if not 0.0 <= hub_to_tip_ratio < 1.0:
        raise ValueError("hub_to_tip_ratio must be in [0, 1)")
    if corrected_flow_per_area <= 0.0:
        raise ValueError("corrected_flow_per_area must be positive")
    if mach <= 0.0:
        raise ValueError("design Mach must be positive")
    if not 0.0 < pressure_recovery <= 1.0:
        raise ValueError("pressure_recovery must be in (0, 1]")
    if capture_margin <= 0.0:
        raise ValueError("capture_margin must be positive")
    if n_inlets < 1:
        raise ValueError("n_inlets must be at least 1")

    # ISA, free stream, total conditions
    T, p, rho = isa(altitude)
    V = mach * math.sqrt(GAMMA * R_AIR * T)
    ram = 1.0 + 0.5 * (GAMMA - 1.0) * mach**2
    T0 = T * ram
    p0 = p * ram ** (GAMMA / (GAMMA - 1.0))

    # engine face and mass flow
    face_area = 0.25 * math.pi * face_diameter**2 * (1.0 - hub_to_tip_ratio**2)
    mc = corrected_flow_per_area * face_area
    p0_face = p0 * pressure_recovery
    m = mc * (p0_face / P_REF) / math.sqrt(T0 / T_REF)

    try:
        face_mach = mach_from_corrected_flux(corrected_flow_per_area)
    except ChokedFlowError as exc:
        mc_max = max_corrected_flux() * face_area
        m_max = mc_max * (p0_face / P_REF) / math.sqrt(T0 / T_REF)
        raise ChokedFlowError(
            f"engine face is choked: corrected flow {mc:.4g} kg/s ({corrected_flow_per_area:.4g} kg/s/m^2 over "
            f"{face_area:.4g} m^2) exceeds the maximum {mc_max:.4g} kg/s ({max_corrected_flux():.4g} kg/s/m^2). "
            f"Maximum mass flow for this face area at this design point: {m_max:.4g} kg/s "
            f"(requested {m:.4g} kg/s)."
        ) from exc

    # capture area
    streamtube = m / (rho * V)
    capture_total = streamtube * capture_margin

    msgs: list[str] = []
    lo, hi = face_mach_range
    face_status = _check_range(face_mach, lo, hi)
    if face_status == "WARN":
        msgs.append(f"face Mach {face_mach:.3f} outside the expected range {lo:g}-{hi:g}")

    specific = None
    st_status = "SKIPPED"
    if reference_thrust is not None and specific_thrust_range is not None:
        specific = reference_thrust / m
        st_status = _check_range(specific, *specific_thrust_range)
        if st_status == "WARN":
            msgs.append(
                f"specific thrust {specific:.1f} N/(kg/s) outside the range "
                f"{specific_thrust_range[0]:g}-{specific_thrust_range[1]:g}"
            )
    for msg in msgs:
        warnings.warn(msg, EngineSizingWarning, stacklevel=2)

    return SizingResult(
        altitude=altitude,
        T_static=T,
        p_static=p,
        rho=rho,
        mach=mach,
        velocity=V,
        T0=T0,
        p0=p0,
        face_diameter=face_diameter,
        hub_to_tip_ratio=hub_to_tip_ratio,
        face_area=face_area,
        pressure_recovery=pressure_recovery,
        p0_face=p0_face,
        corrected_flux=corrected_flow_per_area,
        corrected_flow=mc,
        mass_flow=m,
        face_mach=face_mach,
        face_mach_range=(lo, hi),
        face_mach_status=face_status,
        capture_margin=capture_margin,
        n_inlets=n_inlets,
        streamtube_area=streamtube,
        capture_area_total=capture_total,
        capture_area_per_inlet=capture_total / n_inlets,
        thrust=reference_thrust,
        specific_thrust=specific,
        specific_thrust_range=tuple(specific_thrust_range) if specific_thrust_range is not None else None,
        specific_thrust_status=st_status,
        warnings=tuple(msgs),
    )


def _require(section: dict, name: str, key: str):
    value = (section or {}).get(key)
    if value is None:
        raise ValueError(f"config value '{name}.{key}' is not set")
    return value


def size_from_config(cfg: dict) -> SizingResult:
    """Size the engine face and capture area from a loaded config dict."""
    eng = cfg.get("engine") or {}
    dp = cfg.get("design_point") or {}
    ducts = cfg.get("ducts") or {}
    return size_engine(
        face_diameter=_require(eng, "engine", "face_diameter"),
        hub_to_tip_ratio=_require(eng, "engine", "hub_to_tip_ratio"),
        corrected_flow_per_area=_require(eng, "engine", "corrected_flow_per_area"),
        mach=_require(dp, "design_point", "mach"),
        altitude=_require(dp, "design_point", "altitude"),
        pressure_recovery=_require(dp, "design_point", "pressure_recovery"),
        capture_margin=_require(dp, "design_point", "capture_margin"),
        n_inlets=ducts.get("n_inlets") or DEFAULT_N_INLETS,
        face_mach_range=tuple(eng.get("face_mach_range") or DEFAULT_FACE_MACH_RANGE),
        reference_thrust=eng.get("reference_thrust"),
        specific_thrust_range=(
            tuple(eng["specific_thrust_range"]) if eng.get("specific_thrust_range") is not None else None
        ),
    )


# ---- reporting ----------------------------------------------------------------------------
def format_summary(r: SizingResult) -> str:
    rows: list[tuple[str, str] | None] = [
        None,
        ("ISA / flight", ""),
        ("Altitude", f"{r.altitude:,.0f} m"),
        ("Static temperature T", f"{r.T_static:.2f} K"),
        ("Static pressure p", f"{r.p_static:,.1f} Pa"),
        ("Density rho", f"{r.rho:.4f} kg/m^3"),
        ("Mach", f"{r.mach:.3f}"),
        ("Velocity V", f"{r.velocity:.2f} m/s"),
        ("Total temperature T0", f"{r.T0:.2f} K"),
        ("Total pressure p0", f"{r.p0:,.1f} Pa"),
        None,
        ("Engine face", ""),
        ("Diameter", f"{r.face_diameter:.4f} m"),
        ("Hub-to-tip ratio", f"{r.hub_to_tip_ratio:.3f}"),
        ("Face area", f"{r.face_area:.5f} m^2"),
        ("Pressure recovery", f"{r.pressure_recovery:.3f}"),
        ("Face total pressure", f"{r.p0_face:,.1f} Pa"),
        ("Corrected flow / area", f"{r.corrected_flux:.2f} kg/s/m^2"),
        ("Corrected mass flow", f"{r.corrected_flow:.3f} kg/s"),
        ("Actual mass flow", f"{r.mass_flow:.3f} kg/s"),
        (
            "Face Mach",
            f"{r.face_mach:.3f}   [{r.face_mach_status}: expected "
            f"{r.face_mach_range[0]:g}-{r.face_mach_range[1]:g}]",
        ),
        None,
        ("Capture", ""),
        ("Free-stream tube area", f"{r.streamtube_area:.5f} m^2"),
        ("Capture margin", f"{r.capture_margin:.3f}"),
        ("Capture area, total", f"{r.capture_area_total:.5f} m^2"),
        (f"Capture area, per inlet (/{r.n_inlets})", f"{r.capture_area_per_inlet:.5f} m^2"),
        None,
        ("Specific-thrust check", ""),
    ]
    if r.specific_thrust is None:
        rows.append(("Specific thrust", "SKIPPED (reference thrust or range not set)"))
    else:
        lo, hi = r.specific_thrust_range
        rows.append(("Reference thrust", f"{r.thrust:,.0f} N"))
        rows.append(("Specific thrust", f"{r.specific_thrust:.1f} N/(kg/s)   [{r.specific_thrust_status}: expected {lo:g}-{hi:g}]"))

    width = max(len(a) for row in rows if row for a in row[:1]) + 2
    lines = ["=" * 74, "ENGINE AND CAPTURE-AREA SIZING", "=" * 74]
    for row in rows:
        if row is None:
            lines.append("")
        elif row[1] == "":
            lines.append(row[0])
            lines.append("-" * len(row[0]))
        else:
            lines.append(f"  {row[0]:<{width}}{row[1]}")
    lines.append("")
    if r.warnings:
        lines.extend(f"WARN: {w}" for w in r.warnings)
    else:
        lines.append("All checks PASS.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Print the engine and capture-area sizing summary.")
    add_config_argument(parser)
    args = parser.parse_args(argv)
    try:
        cfg, config_path = load_config(args.config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineSizingWarning)  # shown in the summary instead
            result = size_from_config(cfg)
    except (FileNotFoundError, ValueError) as exc:  # includes ChokedFlowError
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Config: {Path(config_path).name}")
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
