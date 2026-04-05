#!/usr/bin/env python3
"""
Full electromagnetic characterization comparison: FEM (Elmer) vs PyOM analytical.

For each MAS example, runs:
  1. Magnetizing inductance (self, per primary winding)
  2. Inductance matrix + leakage inductance + coupling coefficients
  3. AC winding losses / resistance matrix
  4. Core losses (Steinmetz from FEM B-field)
  5. Stray capacitance matrix (where bobbin is available)

Compares each against the PyOM analytical model and reports errors.
"""

import os
import sys
import json
import glob
import time
import math
import subprocess
import traceback
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.expanduser("~/OpenMagnetics/MVB/src"))
sys.path.insert(0, os.path.expanduser("~/wuerth/Ansyas/src"))
sys.path.insert(0, os.path.expanduser("~/wuerth/Ansyas/tests"))

EXAMPLES_DIR = "/home/alfonso/OpenMagnetics/MAS/examples"
OUTPUT_BASE = "output/full_comparison"
SCRIPT = "tests/validate_elmer_inductance.py"
TIMEOUT = 900
FREQUENCY = 100000.0  # Hz
TEMPERATURE = 25.0    # °C

try:
    from PyOpenMagnetics import PyOpenMagnetics as PyOM
    HAS_PYOM = True
except ImportError:
    HAS_PYOM = False
    print("WARNING: PyOpenMagnetics not available — no analytical reference")


# ---------------------------------------------------------------------------
# PyOM reference computations
# ---------------------------------------------------------------------------

def pyom_inductance_matrix(magnetic: Dict, models: Dict) -> Optional[Dict]:
    """NxN inductance matrix from PyOM. Returns {winding: {winding: H}}."""
    try:
        r = PyOM.calculate_inductance_matrix(magnetic, FREQUENCY, models)
        magnitude = r.get("magnitude", {})
        # Flatten to {winding: {winding: nominal_H}}
        result = {}
        for w1, row in magnitude.items():
            result[w1] = {}
            for w2, val in row.items():
                result[w1][w2] = val.get("nominal", 0.0) if isinstance(val, dict) else 0.0
        return result
    except Exception as e:
        return {"error": str(e)}


def pyom_leakage_inductance(magnetic: Dict, winding_index: int) -> Optional[float]:
    """Leakage inductance for a given winding index from PyOM. Returns H."""
    try:
        r = PyOM.calculate_leakage_inductance(magnetic, FREQUENCY, winding_index)
        per_winding = r.get("leakageInductancePerWinding", [])
        if per_winding:
            # Sum all non-zero entries (leakage w.r.t all other windings)
            total = sum(
                v.get("nominal", 0.0) if isinstance(v, dict) else 0.0
                for v in per_winding
                if (v.get("nominal") if isinstance(v, dict) else v) is not None
            )
            return total
        return None
    except Exception:
        return None


def pyom_resistance_matrix(magnetic: Dict) -> Optional[Dict]:
    """AC resistance matrix from PyOM at FREQUENCY. Returns {winding: {winding: Ohm}}."""
    try:
        r = PyOM.calculate_resistance_matrix(magnetic, TEMPERATURE, FREQUENCY)
        magnitude = r.get("magnitude", {})
        result = {}
        for w1, row in magnitude.items():
            result[w1] = {}
            for w2, val in row.items():
                result[w1][w2] = val.get("nominal", 0.0) if isinstance(val, dict) else 0.0
        return result
    except Exception as e:
        return {"error": str(e)}


def pyom_core_losses(core: Dict, coil: Dict, inputs: Dict, models: Dict) -> Optional[float]:
    """Total core losses from PyOM using IGSE. Returns W."""
    try:
        r = PyOM.calculate_core_losses(core, coil, inputs, models)
        if isinstance(r, dict):
            return r.get("coreLosses")
        return None
    except Exception:
        return None


def pyom_winding_losses(magnetic: Dict, op: Dict) -> Optional[float]:
    """Total winding losses from PyOM for given operating point. Returns W."""
    try:
        r = PyOM.calculate_winding_losses(magnetic, op, TEMPERATURE)
        if isinstance(r, dict):
            data = r.get("data", {})
            if isinstance(data, str):
                return None  # Exception string
            if isinstance(data, dict):
                return data.get("totalLosses") or data.get("losses")
        return None
    except Exception:
        return None


def pyom_capacitance_matrix(core: Dict, coil: Dict) -> Optional[Dict]:
    """Maxwell capacitance matrix from PyOM. Returns {winding: {winding: F}}."""
    try:
        r = PyOM.calculate_maxwell_capacitance_matrix(core, coil)
        if isinstance(r, dict):
            data = r.get("data", {})
            if isinstance(data, str):
                return None  # Exception: missing bobbin
        return r if isinstance(r, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FEM computations (subprocess calls to validate_elmer_inductance.py)
# ---------------------------------------------------------------------------

def _get_python() -> str:
    """Return python executable with cadquery available."""
    import shutil
    # Prefer venv python (has cadquery), fall back to sys.executable
    venv_py = os.path.expanduser("~/wuerth/Ansyas/.venv/bin/python3")
    if os.path.isfile(venv_py):
        return venv_py
    return sys.executable


def run_elmer_subprocess(args: List[str]) -> Dict:
    """Run validate_elmer_inductance.py as subprocess; parse Final Results JSON."""
    cmd = [_get_python(), SCRIPT] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT
        )
        output = result.stdout + result.stderr
        start = output.rfind("Final Results:")
        if start >= 0:
            rest = output[start:]
            brace = rest.find("{")
            if brace >= 0:
                depth, end = 0, -1
                for i, c in enumerate(rest[brace:]):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = brace + i + 1
                            break
                if end > 0:
                    try:
                        return json.loads(rest[brace:end])
                    except json.JSONDecodeError:
                        pass
        return {"success": False, "error": f"No result (exit={result.returncode})"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"TIMEOUT {TIMEOUT}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def fem_inductance(mas_file: str, output_dir: str) -> Dict:
    """Single-winding DC inductance via FEM (CoilSolver)."""
    return run_elmer_subprocess([
        mas_file, "-o", output_dir, "-m", "coilsolver"
    ])


def fem_inductance_matrix(mas_file: str, output_dir: str, magnetic: Dict) -> Dict:
    """NxN inductance matrix via FEM energy method."""
    # Import compute_inductance_matrix from the test module
    try:
        from validate_elmer_inductance import compute_inductance_matrix
        return compute_inductance_matrix(mas_file, output_dir)
    except Exception as e:
        return {"error": str(e)}


def fem_core_losses(mas_file: str, inductance_dir: str, magnetic: Dict) -> Dict:
    """
    Core losses from FEM B-field + Steinmetz.

    Reuses the VTU from the inductance simulation (must run phase 1 first).
    Extracts B-field from core body in the VTU and integrates Steinmetz element-by-element.
    """
    try:
        from validate_elmer_inductance import compute_core_losses, get_core_data
        import json as _json

        # Get material name from the magnetic
        core_data = get_core_data(magnetic)
        func_desc = core_data.get("functionalDescription", {})
        mat = func_desc.get("material", "N87")
        if isinstance(mat, dict):
            mat = mat.get("name", "N87")

        # The inductance simulation VTU is in inductance_dir/mesh/ — pass parent dir
        mesh_dir = os.path.join(inductance_dir, "mesh")
        if not os.path.isdir(mesh_dir):
            return {"error": f"No mesh dir at {mesh_dir} — run inductance phase first"}

        r = compute_core_losses(
            sim_dir=inductance_dir,  # compute_core_losses appends /mesh/ internally
            material_name=mat,
            frequency=FREQUENCY,
            core_body_id=1,
            temperature=TEMPERATURE,
        )
        return r if r else {"error": "compute_core_losses returned None"}
    except Exception as e:
        return {"error": str(e)}


def fem_winding_losses(mas_file: str, output_dir: str) -> Dict:
    """AC winding losses from FEM harmonic solver."""
    try:
        from validate_elmer_inductance import run_harmonic_simulation
        return run_harmonic_simulation(
            mas_file, output_dir, frequency=FREQUENCY
        )
    except Exception as e:
        return {"error": str(e)}


def fem_capacitance(mas_file: str, output_dir: str, magnetic: Dict) -> Dict:
    """Capacitance matrix from electrostatic FEM."""
    try:
        from validate_elmer_inductance import compute_capacitance_matrix
        return compute_capacitance_matrix(mas_file, output_dir)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Per-example comparison
# ---------------------------------------------------------------------------

def get_winding_turns(coil: Dict) -> List[int]:
    """Return full turn count per winding from coil functionalDescription."""
    return [w.get("numberTurns", 1) for w in coil.get("functionalDescription", [])]



def compare_example(mas_file: str, output_dir: str) -> Dict:
    """Run full FEM characterization and compare against PyOM for one MAS file."""
    t0 = time.time()
    result = {
        "mas_file": mas_file,
        "name": os.path.splitext(os.path.basename(mas_file))[0],
        "phases": {},
    }

    # Load and autocomplete MAS data
    try:
        with open(mas_file) as f:
            data = json.load(f)
        mag = data.get("magnetic", data)
        mag = PyOM.magnetic_autocomplete(mag, {})
        core = mag.get("core", {})
        coil = mag.get("coil", {})
        inputs = data.get("inputs", {})
        op = (inputs.get("operatingPoints") or [{}])[0]
        models = {"gapReluctance": "ZHANG", "coreLosses": "IGSE"}
    except Exception as e:
        result["error"] = f"Load failed: {e}"
        result["time_s"] = round(time.time() - t0, 1)
        return result

    winding_names = [w.get("name", f"W{i}") for i, w in
                     enumerate(coil.get("functionalDescription", []))]
    full_turns = get_winding_turns(coil)  # actual turn counts per winding
    result["winding_names"] = winding_names
    result["full_turns"] = full_turns
    N = len(winding_names)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Magnetizing inductance (primary winding self-inductance)
    # ------------------------------------------------------------------
    print(f"  [1] Inductance...")
    try:
        fem_L = fem_inductance(mas_file, os.path.join(output_dir, "inductance"))
        # Use the turn-normalized reference from validate_mas_file (already N_fem turns)
        fem_val = fem_L.get("elmer_inductance_H")
        pyom_val = fem_L.get("analytical_inductance_H")

        err = _pct_err(fem_val, pyom_val)
        result["phases"]["inductance"] = {
            "fem_H": fem_val,
            "pyom_H": pyom_val,
            "full_turns": full_turns[0] if full_turns else None,
            "error_pct": err,
            "pass": abs(err) < 25 if err is not None else False,
        }
        print(f"     FEM={_fmt_uH(fem_val)}  PyOM={_fmt_uH(pyom_val)}  err={_fmt_err(err)}"
              f"  (N={full_turns[0] if full_turns else '?'})")
    except Exception as e:
        result["phases"]["inductance"] = {"error": str(e)}
        print(f"     ERROR: {e}")

    # ------------------------------------------------------------------
    # Phase 2: Inductance matrix (mutual, leakage, coupling)
    # ------------------------------------------------------------------
    if N > 1:
        print(f"  [2] Inductance matrix ({N}x{N})...")
        try:
            ind_mat_dir = os.path.join(output_dir, "inductance_matrix")
            fem_mat = fem_inductance_matrix(mas_file, ind_mat_dir, mag)
            pyom_mat = pyom_inductance_matrix(mag, models)

            fem_Lmat = fem_mat.get("inductance_matrix_H", [])
            fem_kmat = fem_mat.get("coupling_matrix", [])
            fem_leak = fem_mat.get("leakage_inductance_H", {})

            # Per-winding comparison
            mat_phase = {"windings": winding_names, "L_matrix": {}, "coupling": {}, "leakage": {}}

            for i, wi in enumerate(winding_names):
                for j, wj in enumerate(winding_names):
                    fem_v = fem_Lmat[i][j] if fem_Lmat and i < len(fem_Lmat) and j < len(fem_Lmat[i]) else None
                    pyom_v = (pyom_mat or {}).get(wi, {}).get(wj) if pyom_mat else None
                    err = _pct_err(fem_v, pyom_v)
                    mat_phase["L_matrix"][f"{wi}_{wj}"] = {
                        "fem_H": fem_v, "pyom_H": pyom_v, "error_pct": err
                    }
                # Leakage (scale-independent: leakage/self ratio is what matters)
                if N == 2:
                    fem_leak_v = fem_leak.get(wi)
                    pyom_leak_v = pyom_leakage_inductance(mag, i)
                    err_l = _pct_err(fem_leak_v, pyom_leak_v)
                    mat_phase["leakage"][wi] = {
                        "fem_H": fem_leak_v, "pyom_H": pyom_leak_v, "error_pct": err_l
                    }

            # Coupling coefficients (turn-count independent — most reliable comparison)
            pyom_kmat = {}
            if pyom_mat:
                for i, wi in enumerate(winding_names):
                    for j, wj in enumerate(winding_names):
                        if i != j:
                            Lii = (pyom_mat.get(wi, {}).get(wi) or 0)
                            Ljj = (pyom_mat.get(wj, {}).get(wj) or 0)
                            Mij = (pyom_mat.get(wi, {}).get(wj) or 0)
                            if Lii > 0 and Ljj > 0:
                                pyom_kmat[f"{wi}_{wj}"] = Mij / math.sqrt(Lii * Ljj)
            mat_phase["coupling_pyom"] = pyom_kmat
            mat_phase["coupling_fem"] = {
                f"{winding_names[i]}_{winding_names[j]}": fem_kmat[i][j]
                for i in range(len(winding_names))
                for j in range(len(winding_names))
                if i != j and fem_kmat and i < len(fem_kmat) and j < len(fem_kmat[i])
            }

            result["phases"]["inductance_matrix"] = mat_phase

            # Print summary
            for i, wi in enumerate(winding_names):
                for j, wj in enumerate(winding_names):
                    entry = mat_phase["L_matrix"][f"{wi}_{wj}"]
                    print(f"     L[{wi}][{wj}]: FEM={_fmt_uH(entry['fem_H'])}  "
                          f"PyOM={_fmt_uH(entry['pyom_H'])}  err={_fmt_err(entry['error_pct'])}")
            for key in mat_phase.get("coupling_fem", {}):
                k_fem = mat_phase["coupling_fem"].get(key, 0)
                k_pyom = mat_phase["coupling_pyom"].get(key)
                k_err = _pct_err(k_fem, k_pyom)
                pyom_k_str = f"{k_pyom:.4f}" if k_pyom is not None else "N/A"
                print(f"     k[{key}]: FEM={k_fem:.4f}  PyOM={pyom_k_str}  err={_fmt_err(k_err)}")
            if mat_phase["leakage"]:
                for wi, v in mat_phase["leakage"].items():
                    print(f"     Leak[{wi}]: FEM={_fmt_uH(v['fem_H'])}  "
                          f"PyOM={_fmt_uH(v['pyom_H'])}  err={_fmt_err(v['error_pct'])}")
        except Exception as e:
            result["phases"]["inductance_matrix"] = {"error": str(e)}
            print(f"     ERROR: {e}")
    else:
        print(f"  [2] Inductance matrix: skip (single winding)")

    # ------------------------------------------------------------------
    # Phase 3: AC winding losses / resistance
    # ------------------------------------------------------------------
    print(f"  [3] AC winding losses @ {FREQUENCY/1e3:.0f} kHz...")
    try:
        harm_dir = os.path.join(output_dir, "harmonic")
        fem_harm = fem_winding_losses(mas_file, harm_dir)
        pyom_R_mat = pyom_resistance_matrix(mag)

        fem_R = fem_harm.get("ac_resistance_Ohm")
        fem_P = fem_harm.get("joule_losses_W")

        w0 = winding_names[0] if winding_names else "Primary"
        pyom_R_ac = (pyom_R_mat or {}).get(w0, {}).get(w0) if pyom_R_mat else None

        result["phases"]["winding_losses"] = {
            "frequency_Hz": FREQUENCY,
            "fem_R_ac_Ohm": fem_R,
            "fem_joule_W": fem_P,
            "pyom_R_ac_Ohm": pyom_R_ac,
            "error_pct": _pct_err(fem_R, pyom_R_ac),
        }
        print(f"     FEM R_ac={_fmt_R(fem_R)}  FEM P={_fmt_W(fem_P)}  "
              f"PyOM R_ac={_fmt_R(pyom_R_ac)}  err={_fmt_err(_pct_err(fem_R, pyom_R_ac))}")
    except Exception as e:
        result["phases"]["winding_losses"] = {"error": str(e)}
        print(f"     ERROR: {e}")

    # ------------------------------------------------------------------
    # Phase 4: Core losses (from inductance sim VTU + Steinmetz)
    # ------------------------------------------------------------------
    print(f"  [4] Core losses...")
    try:
        # Reuse VTU from inductance phase (mesh/ subdirectory)
        ind_sim_dir = os.path.join(output_dir, "inductance")
        fem_cl = fem_core_losses(mas_file, ind_sim_dir, mag)
        pyom_cl = pyom_core_losses(core, coil, inputs, models)

        fem_P_raw = fem_cl.get("core_loss_W") if isinstance(fem_cl, dict) else None

        # Scale FEM result from simulation conditions (I=1A) to actual
        # operating conditions. Core losses scale as B^β, and B ∝ N*I.
        # P_actual = P_sim × (N_actual × I_peak / (N_sim × 1.0))^β
        fem_P_core = fem_P_raw
        scale_factor = None
        if fem_P_raw is not None:
            beta = fem_cl.get("steinmetz_beta", 2.5) if isinstance(fem_cl, dict) else 2.5
            # FEM runs at 1A; scale to actual operating peak current (B ∝ I, P ∝ I^β)
            I_peak = 1.0
            exc = op.get("excitationsPerWinding", [{}])
            if exc:
                cur = exc[0].get("magnetizingCurrent") or exc[0].get("current")
                if isinstance(cur, dict):
                    processed = cur.get("processed", {})
                    I_peak = processed.get("peakToPeak", 0) / 2 if processed.get("peakToPeak") else \
                             processed.get("rms", 1.0) * math.sqrt(2)
                    dc_offset = processed.get("offset", 0) or 0
                    I_peak = abs(I_peak + dc_offset) if I_peak else abs(dc_offset) or 1.0
            if I_peak != 1.0 and I_peak > 0:
                scale_factor = I_peak ** beta
                fem_P_core = fem_P_raw * scale_factor

        err_c = _pct_err(fem_P_core, pyom_cl)
        result["phases"]["core_losses"] = {
            "fem_W_raw": fem_P_raw,
            "fem_W_scaled": fem_P_core,
            "pyom_W": pyom_cl,
            "scale_factor": scale_factor,
            "error_pct": err_c,
        }
        sf_str = f"  scale={scale_factor:.1f}x" if scale_factor else ""
        print(f"     FEM_scaled={_fmt_W(fem_P_core)}{sf_str}  PyOM={_fmt_W(pyom_cl)}  err={_fmt_err(err_c)}")
    except Exception as e:
        result["phases"]["core_losses"] = {"error": str(e)}
        print(f"     ERROR: {e}")

    # ------------------------------------------------------------------
    # Phase 5: Capacitance matrix
    # ------------------------------------------------------------------
    if N > 1:
        print(f"  [5] Capacitance matrix...")
        try:
            cap_dir = os.path.join(output_dir, "capacitance")
            fem_cap = fem_capacitance(mas_file, cap_dir, mag)
            pyom_cap = pyom_capacitance_matrix(core, coil)

            if isinstance(fem_cap, dict) and "capacitance_matrix_F" in fem_cap:
                fem_C = fem_cap["capacitance_matrix_F"]
                cap_phase = {"fem_C_pF": {}, "pyom_C_pF": {}}
                for i, wi in enumerate(winding_names):
                    for j, wj in enumerate(winding_names):
                        fem_v = fem_C[i][j] if fem_C and i < len(fem_C) else None
                        cap_phase["fem_C_pF"][f"{wi}_{wj}"] = (fem_v * 1e12) if fem_v else None
                result["phases"]["capacitance"] = cap_phase
                print(f"     FEM cap matrix computed")
            elif isinstance(fem_cap, dict) and "error" in fem_cap:
                result["phases"]["capacitance"] = {"error": fem_cap["error"]}
                print(f"     FEM ERROR: {fem_cap['error'][:80]}")
            else:
                result["phases"]["capacitance"] = {"error": "No capacitance data"}
                print(f"     No capacitance data")
        except Exception as e:
            result["phases"]["capacitance"] = {"error": str(e)}
            print(f"     ERROR: {e}")
    else:
        print(f"  [5] Capacitance: skip (single winding, no inter-winding cap)")

    result["time_s"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct_err(fem, ref):
    if fem is None or ref is None or ref == 0:
        return None
    return 100.0 * (fem - ref) / abs(ref)

def _fmt_uH(v):
    return f"{v*1e6:.3f}µH" if v is not None else "N/A"

def _fmt_R(v):
    return f"{v*1e3:.2f}mΩ" if v is not None else "N/A"

def _fmt_W(v):
    return f"{v:.4f}W" if v is not None else "N/A"

def _fmt_err(v):
    return f"{v:+.1f}%" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full FEM vs PyOM comparison for all MAS examples")
    parser.add_argument("--examples", default=EXAMPLES_DIR, help="Directory with MAS JSON files")
    parser.add_argument("--output", default=OUTPUT_BASE, help="Output base directory")
    parser.add_argument("--filter", default=None, help="Only run examples matching this substring")
    parser.add_argument("--phases", default="all",
                        help="Comma-separated phases: inductance,matrix,winding,core,capacitance,all")
    args = parser.parse_args()

    requested_phases = set(args.phases.split(","))
    run_all = "all" in requested_phases

    files = sorted(glob.glob(os.path.join(args.examples, "*.json")))
    if args.filter:
        files = [f for f in files if args.filter in os.path.basename(f)]

    print(f"Running full comparison on {len(files)} examples")
    print(f"Output: {args.output}")
    print("=" * 70)

    os.makedirs(args.output, exist_ok=True)
    all_results = []

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        output_dir = os.path.join(args.output, name)
        print(f"\n{'='*70}")
        print(f"Example: {name}")
        print(f"{'='*70}")

        r = compare_example(f, output_dir)
        r["name"] = name
        all_results.append(r)

        # Save intermediate results
        with open(os.path.join(args.output, "all_results.json"), "w") as fout:
            json.dump(all_results, fout, indent=2, default=str)

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print_summary(all_results)

    results_file = os.path.join(args.output, "all_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {results_file}")


def print_summary(results: List[Dict]):
    print(f"\n{'Example':<45} {'L err':>8} {'L11 err':>8} {'Leak err':>9} "
          f"{'Rac err':>9} {'Pcore err':>10}")
    print("-" * 95)
    for r in results:
        name = r.get("name", "?")[:44]
        phases = r.get("phases", {})

        L_err = _fmt_err(phases.get("inductance", {}).get("error_pct"))
        L11_err = "N/A"
        leak_err = "N/A"
        if "inductance_matrix" in phases:
            wnames = phases["inductance_matrix"].get("windings", [])
            if wnames:
                w0 = wnames[0]
                L11 = phases["inductance_matrix"].get("L_matrix", {}).get(f"{w0}_{w0}", {})
                L11_err = _fmt_err(L11.get("error_pct"))
                leaks = phases["inductance_matrix"].get("leakage", {})
                if leaks and w0 in leaks:
                    leak_err = _fmt_err(leaks[w0].get("error_pct"))
        Rac_err = _fmt_err(phases.get("winding_losses", {}).get("error_pct"))
        Pcore_err = _fmt_err(phases.get("core_losses", {}).get("error_pct"))

        print(f"{name:<45} {L_err:>8} {L11_err:>8} {leak_err:>9} {Rac_err:>9} {Pcore_err:>10}")


if __name__ == "__main__":
    sys.exit(main())
