# b2987b

Python driver and GUI for the Keysight B2987B electrometer/picoammeter.

Replaces `B2987b-Control-Program/Python_Control/` with a cleaner API,
simulation mode, and standalone GUI. The old directory is kept for reference.

## Three Operating Modes

| Mode | Method | Use case |
|------|--------|----------|
| **Bias only** | `set_bias(v)` / `bias_off()` | Set HV during pulse counting — no measurement |
| **Single point** | `measure_current(v)` | One-shot current readout |
| **IV sweep** | `sweep(voltages)` | Full IV curve with averaging |

## Quick Start

```bash
pip install -r requirements.txt

python -m b2987b.gui          # standalone GUI
python examples/basic_usage.py  # headless example
```

## API

```python
from b2987b import B2987BController

with B2987BController(visa="USB0::...", mode="simulation") as elec:

    elec.configure_sweep(source_range=1000, n_per_voltage=5, delay_s=0.1)

    # Bias only (for pulse counting)
    elec.set_bias(48.5)
    # ... digitizer acquires ...
    elec.bias_off()

    # Single measurement
    current = elec.measure_current(voltage=48.5)

    # IV sweep
    import numpy as np
    voltages = np.arange(40.0, 55.0, 0.05).tolist()
    result = elec.sweep(voltages)

# result.avg_source_v   -> np.ndarray (V)
# result.avg_current_a  -> np.ndarray (A)
# result.err_current_a  -> np.ndarray (A) standard error in the mean
# result.source_v       -> raw voltage array (all n_per_voltage points)
# result.current_a      -> raw current array
```

## Improvements over B2987b-Control-Program

- No CSV config files — parameters passed directly as Python arguments
- Simulation mode — full IV curve without hardware
- `set_bias()` / `bias_off()` — bias-only control for use during pulse counting
- `ramp_bias()` — safe voltage ramping with configurable step size
- `SweepResult` dataclass with both raw and averaged data
- Statistical averaging built in (standard error in the mean)
- Standalone GUI with IV plot
- Plugin interface for ETS DAQ auto-discovery
