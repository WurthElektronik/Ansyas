#!/usr/bin/env python3
"""Run all MAS examples through Elmer validation with timeout and error handling."""

import json
import os
import sys
import subprocess
import glob
import time

EXAMPLES_DIR = "/home/alfonso/OpenMagnetics/MAS/examples"
OUTPUT_BASE = "output/mas_examples"
SCRIPT = "tests/validate_elmer_inductance.py"
TIMEOUT = 900  # 15 minutes per example
MAX_TURNS = 4
METHOD = "coilsolver"


def run_example(json_file, output_dir):
    """Run a single example with timeout."""
    cmd = [
        sys.executable, SCRIPT, json_file,
        "-o", output_dir,
        "-t", str(MAX_TURNS),
        "-m", METHOD,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT
        )
        output = result.stdout + result.stderr
        # Extract the JSON block between { and } after "Final Results:"
        start = output.rfind("Final Results:")
        if start >= 0:
            rest = output[start:]
            brace_start = rest.find("{")
            if brace_start >= 0:
                # Find matching closing brace
                depth = 0
                for i, c in enumerate(rest[brace_start:]):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = rest[brace_start:brace_start + i + 1]
                            try:
                                return json.loads(json_str), output
                            except json.JSONDecodeError:
                                pass
                            break
        return {"success": False, "error": f"exit_code={result.returncode}"}, output
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"TIMEOUT after {TIMEOUT}s"}, "TIMEOUT"
    except Exception as e:
        return {"success": False, "error": str(e)}, str(e)


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Gather all JSON files (skip complete/ - different MAS structure without winding support)
    files = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.json")))

    results = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if "/complete/" in f:
            name = "complete_" + name
        output_dir = os.path.join(OUTPUT_BASE, name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        t0 = time.time()
        result, output = run_example(f, output_dir)
        elapsed = time.time() - t0
        result["name"] = name
        result["time_s"] = round(elapsed, 1)
        results.append(result)

        if result.get("success"):
            print(f"  PASS  analytical={result.get('analytical_inductance_uH', '?'):.2f} uH  "
                  f"elmer={result.get('elmer_inductance_uH', '?'):.2f} uH  "
                  f"error={result.get('error_percent', '?'):.1f}%  "
                  f"({elapsed:.0f}s)")
        else:
            err = result.get("error", "unknown")
            if len(err) > 100:
                err = err[:100] + "..."
            print(f"  FAIL  {err}  ({elapsed:.0f}s)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    print(f"Total: {len(results)}  Passed: {len(passed)}  Failed: {len(failed)}")

    if passed:
        print(f"\nPassed examples:")
        for r in passed:
            print(f"  {r['name']:50s}  L_ana={r.get('analytical_inductance_uH', 0):.2f} uH  "
                  f"L_fem={r.get('elmer_inductance_uH', 0):.2f} uH  "
                  f"err={r.get('error_percent', 0):.1f}%  ({r['time_s']}s)")

    if failed:
        print(f"\nFailed examples:")
        for r in failed:
            err = r.get("error", "unknown")
            if len(err) > 80:
                err = err[:80] + "..."
            print(f"  {r['name']:50s}  {err}  ({r['time_s']}s)")

    # Save full results
    results_file = os.path.join(OUTPUT_BASE, "all_results.json")
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nFull results saved to: {results_file}")


if __name__ == "__main__":
    main()
