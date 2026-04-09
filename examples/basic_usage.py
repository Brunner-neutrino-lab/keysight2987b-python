"""
B2987B basic usage example — simulation mode.

Demonstrates all three operating modes:
  1. Bias-only (set voltage, no measurement)
  2. Single current measurement
  3. Full IV sweep
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from b2987b import B2987BController

ctrl = B2987BController(mode="simulation")
ctrl.sim_set_vbd(47.5)

with ctrl:
    ctrl.configure_sweep(source_range=1000, n_per_voltage=5, delay_s=0.05)

    # --- 1. Bias only ---
    print("1. Bias only (no measurement)")
    ctrl.set_bias(45.0)
    print(f"   Bias set to 45.0 V, output ON, no current measurement")
    ctrl.bias_off()
    print(f"   Output OFF")

    # --- 2. Single point ---
    print("\n2. Single current measurement")
    current = ctrl.measure_current(voltage=46.0)
    print(f"   V=46.0 V  →  I={current:.4e} A")

    # --- 3. IV sweep ---
    print("\n3. IV sweep (40–52 V, 0.5 V steps, 5 pts/V)")
    voltages = np.arange(40.0, 52.5, 0.5).tolist()
    result   = ctrl.sweep(voltages, n_per_voltage=5, delay_s=0.0)

    print(f"   {len(result.avg_source_v)} voltage points acquired")
    for v, i, ei in zip(result.avg_source_v, result.avg_current_a, result.err_current_a):
        print(f"   V={v:6.2f} V  I={i:.3e} A  ±{ei:.1e}")

print("\nDone.")
