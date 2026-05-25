"""
b2987b/controller.py

High-level controller for the Keysight B2987B electrometer.

Three operating modes:
  1. Bias-only        — set voltage, output on, no measurement (for pulse counting)
  2. Single-point     — set voltage, measure current once
  3. IV sweep         — automated list sweep across a voltage array

All three can be used independently. The DAQ uses bias-only during pulse
counting and IV sweep for characterization measurements.

Usage (headless):

    from b2987b.controller import B2987BController

    with B2987BController(mode="simulation") as elec:

        # 1. Bias only (no measurement — for pulse counting windows)
        elec.set_bias(48.5)
        # ... digitizer acquires pulses ...
        elec.bias_off()

        # 2. Single point
        elec.set_bias(48.5)
        current = elec.measure_current()

        # 3. IV sweep
        import numpy as np
        voltages = np.arange(40.0, 55.0, 0.05).tolist()
        result = elec.sweep(voltages, n_per_voltage=5, delay_s=0.1)
        # result["source_v"], result["current_a"], result["voltage_v"]
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .driver import B2987BDriver, DEFAULT_VISA, SIM_VBD


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """
    Result of one IV sweep.

    Raw data (one entry per trigger):
        source_v    : np.ndarray   commanded voltage (V)
        current_a   : np.ndarray   measured current (A)
        voltage_v   : np.ndarray   sense voltage (V), NaN if not measured
        timestamp_s : np.ndarray   UTC time (s)

    Averaged data (one entry per unique voltage step):
        avg_source_v    : np.ndarray
        avg_current_a   : np.ndarray
        avg_voltage_v   : np.ndarray
        err_current_a   : np.ndarray   standard error in the mean
        err_voltage_v   : np.ndarray

    Metadata:
        channel_id    : int
        bias_voltage_V: float   (for single-voltage sweeps; 0 otherwise)
        temperature_K : float   (set externally before saving)
        run_timestamp : float   UTC time at sweep start
        n_per_voltage : int
    """
    source_v:       np.ndarray = field(default_factory=lambda: np.array([]))
    current_a:      np.ndarray = field(default_factory=lambda: np.array([]))
    voltage_v:      np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp_s:    np.ndarray = field(default_factory=lambda: np.array([]))

    avg_source_v:   np.ndarray = field(default_factory=lambda: np.array([]))
    avg_current_a:  np.ndarray = field(default_factory=lambda: np.array([]))
    avg_voltage_v:  np.ndarray = field(default_factory=lambda: np.array([]))
    err_current_a:  np.ndarray = field(default_factory=lambda: np.array([]))
    err_voltage_v:  np.ndarray = field(default_factory=lambda: np.array([]))

    channel_id:     int   = 0
    bias_voltage_V: float = 0.0
    temperature_K:  float = 0.0
    run_timestamp:  float = field(default_factory=time.time)
    n_per_voltage:  int   = 1


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class B2987BController:
    """
    High-level controller for the Keysight B2987B.

    Parameters
    ----------
    visa : str
        VISA resource string.
    mode : str
        "hardware" or "simulation".
    """

    # ------------------------------------------------------------------
    # Plugin interface
    # ------------------------------------------------------------------
    MODULE_NAME  = "B2987B"
    DEVICE_NAME  = "Keysight B2987B Electrometer"
    CONFIG_FIELDS = [
        {"key": "visa",           "label": "VISA Resource",           "type": "str",    "default": DEFAULT_VISA},
        {"key": "mode",           "label": "Mode",                    "type": "choice", "default": "simulation",
         "choices": ["simulation", "hardware"]},
        {"key": "source_range",   "label": "Source Range (V)",        "type": "choice", "default": "1000",
         "choices": ["20", "1000"]},
        {"key": "current_limit",  "label": "Current-limiting resistor","type": "bool",  "default": False},
        {"key": "measure_voltage","label": "Measure sense voltage",    "type": "bool",  "default": False},
        {"key": "delay_s",        "label": "Trigger delay (s)",        "type": "float", "default": 0.1},
        {"key": "n_per_voltage",  "label": "Points per voltage",       "type": "int",   "default": 5},
    ]
    DEFAULTS = {
        "visa":           DEFAULT_VISA,
        "mode":           "simulation",
        "source_range":   "1000",
        "current_limit":  False,
        "measure_voltage": False,
        "delay_s":        0.1,
        "n_per_voltage":  5,
    }

    @staticmethod
    def test(config: dict) -> tuple[bool, str]:
        try:
            ctrl = B2987BController(
                visa=config.get("visa", DEFAULT_VISA),
                mode=config.get("mode", "simulation"),
            )
            ctrl.connect()
            idn = ctrl.identify()
            ctrl.disconnect()
            return True, f"OK — {idn}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    @staticmethod
    def read(config: dict) -> dict:
        return {
            "visa": config.get("visa", ""),
            "mode": config.get("mode", "simulation"),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, visa: str = DEFAULT_VISA, mode: str = "simulation"):
        self._driver = B2987BDriver(visa=visa, mode=mode)

        # Sweep defaults (overridden by configure_sweep)
        self._source_range    = 1000       # V
        self._current_limit   = False
        self._measure_voltage = False
        self._delay_s         = 0.1
        self._n_per_voltage   = 5
        self._current_aperture: float | None = None

        # Bias state tracking
        self._bias_voltage    = 0.0
        self._bias_active     = False

        # Progress callback: fn(step, total)
        self.on_progress: Callable | None = None

    def connect(self):
        self._driver.connect()

    def disconnect(self):
        if self._bias_active:
            self.bias_off()
        self._driver.disconnect()

    def identify(self) -> str:
        if self._driver.mode == "simulation":
            return f"Keysight B2987B [simulation] @ {self._driver._visa_str}"
        return f"Keysight B2987B [hardware] @ {self._driver._visa_str}"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_sweep(self,
                        source_range: float = 1000,
                        n_per_voltage: int = 5,
                        delay_s: float = 0.1,
                        measure_voltage: bool = False,
                        current_limit: bool = False,
                        current_aperture_s: float | None = None):
        """
        Set defaults for IV sweeps.

        Parameters
        ----------
        source_range : float
            Voltage source range: 20 or 1000 V.
        n_per_voltage : int
            Number of current measurements averaged at each voltage step.
        delay_s : float
            Trigger delay between voltage steps (s). Longer = more settling.
        measure_voltage : bool
            Also measure sense voltage at each step (slower; useful for verification).
        current_limit : bool
            Enable the internal current-limiting resistor.
        current_aperture_s : float, optional
            Current measurement integration time (s). None = auto (LONG mode).
        """
        self._source_range      = source_range
        self._n_per_voltage     = n_per_voltage
        self._delay_s           = delay_s
        self._measure_voltage   = measure_voltage
        self._current_limit     = current_limit
        self._current_aperture  = current_aperture_s

        self._driver.set_source_range(source_range)
        self._driver.set_current_limit(current_limit)

    # ------------------------------------------------------------------
    # Bias-only control (no measurement)
    # ------------------------------------------------------------------

    def set_bias(self, voltage: float, settle_s: float = 0.0):
        """
        Set the source voltage and enable the output without measuring.

        Use this during pulse counting windows when only bias is needed.

        Parameters
        ----------
        voltage : float
            Target voltage in volts.
        settle_s : float
            Optional settling delay after enabling output (s).
        """
        self._driver.set_source_range(self._source_range)
        self._driver.set_current_limit(self._current_limit)
        self._driver.set_voltage(voltage)
        self._driver.output_on()
        self._bias_voltage = voltage
        self._bias_active  = True

        if settle_s > 0:
            time.sleep(settle_s)

    def bias_off(self):
        """Disable the source output."""
        self._driver.output_off()
        self._bias_active = False

    def ramp_bias(self, target_v: float, step_v: float = 1.0, step_delay_s: float = 0.1):
        """
        Ramp from current bias to target_v in steps.

        Safer for large voltage changes — avoids large step transients.

        Parameters
        ----------
        target_v : float
            Target voltage (V).
        step_v : float
            Voltage increment per step (V). Always positive; direction is automatic.
        step_delay_s : float
            Delay between steps (s).
        """
        current_v = self._bias_voltage
        direction = 1 if target_v >= current_v else -1
        step      = abs(step_v) * direction

        v = current_v + step
        while direction * v < direction * target_v:
            self.set_bias(v)
            time.sleep(step_delay_s)
            v += step

        self.set_bias(target_v)

    # ------------------------------------------------------------------
    # Single-point measurement
    # ------------------------------------------------------------------

    def measure_current(self, voltage: float | None = None) -> float:
        """
        Measure current at the given voltage (or current bias if None).

        Enables output and ammeter, takes one reading, returns the value.
        Output stays on after measurement.

        Parameters
        ----------
        voltage : float, optional
            If given, sets this voltage before measuring.

        Returns
        -------
        float
            Current in amperes.
        """
        if voltage is not None:
            self._driver.set_voltage(voltage)
            self._driver.output_on()
            self._bias_voltage = voltage
            self._bias_active  = True
            time.sleep(0.05)

        self._driver.ammeter_on()
        current = self._driver.measure_current()
        self._driver.ammeter_off()
        return current

    # ------------------------------------------------------------------
    # IV sweep
    # ------------------------------------------------------------------

    def sweep(self,
              voltages: list[float] | np.ndarray,
              n_per_voltage: int | None = None,
              delay_s: float | None = None,
              measure_voltage: bool | None = None,
              timeout_s: float = 600.0) -> SweepResult:
        """
        Run a full IV sweep over the given voltage list.

        Parameters
        ----------
        voltages : list or ndarray
            Ordered voltage points for the sweep (V).
        n_per_voltage : int, optional
            Overrides the configured n_per_voltage for this sweep.
        delay_s : float, optional
            Overrides the configured delay_s for this sweep.
        measure_voltage : bool, optional
            Overrides the configured measure_voltage for this sweep.
        timeout_s : float
            Maximum time to wait for sweep completion.

        Returns
        -------
        SweepResult
        """
        voltages       = list(voltages)
        n_per          = n_per_voltage   if n_per_voltage   is not None else self._n_per_voltage
        delay          = delay_s         if delay_s         is not None else self._delay_s
        meas_v         = measure_voltage if measure_voltage is not None else self._measure_voltage

        # The B2987's source range defaults to 20 V after *RST.  If sweep()
        # is called without a prior set_bias() (which would have set the
        # range), any voltage above the small-range limit returns
        # -222 "Data out of range" at INIT and FETCH returns 9.91e+37
        # (the "no data" sentinel).  Push the configured range every time
        # so sweeps work standalone.
        self._driver.set_source_range(self._source_range)
        self._driver.set_current_limit(self._current_limit)

        self._driver.configure_list_sweep(
            voltages            = voltages,
            n_points_per_voltage = n_per,
            delay_s             = delay,
            measure_voltage     = meas_v,
            current_range_auto  = True,
            current_aperture_s  = self._current_aperture,
        )

        run_ts = time.time()
        raw    = self._driver.run_sweep(timeout_s=timeout_s)

        # Convert to numpy
        source_v  = np.array(raw["source_v"],    dtype=np.float64)
        current_a = np.array(raw["current_a"],   dtype=np.float64)
        voltage_v = np.array(raw["voltage_v"],   dtype=np.float64)
        timestamp = np.array(raw["timestamp_s"], dtype=np.float64)

        # Statistical averaging
        avg_s, avg_i, avg_v, err_i, err_v = self._stat_analysis(
            source_v, current_a, voltage_v
        )

        result = SweepResult(
            source_v      = source_v,
            current_a     = current_a,
            voltage_v     = voltage_v,
            timestamp_s   = timestamp,
            avg_source_v  = avg_s,
            avg_current_a = avg_i,
            avg_voltage_v = avg_v,
            err_current_a = err_i,
            err_voltage_v = err_v,
            n_per_voltage = n_per,
            run_timestamp = run_ts,
        )
        return result

    # ------------------------------------------------------------------
    # Statistical analysis (carried over from DataAnalysis.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _stat_analysis(source_v: np.ndarray,
                        current_a: np.ndarray,
                        voltage_v: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
        """
        Average repeated measurements at each unique voltage step.

        Returns (avg_source, avg_current, avg_voltage, err_current, err_voltage).
        Standard error = std / sqrt(n).
        """
        unique_v  = []
        seen      = set()
        for v in source_v:
            if v not in seen:
                unique_v.append(v)
                seen.add(v)

        avg_s = np.array(unique_v, dtype=np.float64)
        avg_i = np.zeros(len(unique_v))
        avg_v = np.zeros(len(unique_v))
        err_i = np.zeros(len(unique_v))
        err_v = np.zeros(len(unique_v))

        for j, v in enumerate(unique_v):
            mask      = source_v == v
            i_vals    = current_a[mask]
            v_vals    = voltage_v[mask]
            n         = len(i_vals)
            avg_i[j]  = np.mean(i_vals)
            err_i[j]  = np.std(i_vals) / np.sqrt(n) if n > 1 else 0.0
            if not np.isnan(v_vals[0]):
                avg_v[j] = np.mean(v_vals)
                err_v[j] = np.std(v_vals) / np.sqrt(n) if n > 1 else 0.0
            else:
                avg_v[j] = float("nan")
                err_v[j] = float("nan")

        return avg_s, avg_i, avg_v, err_i, err_v

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def sim_set_vbd(self, vbd: float):
        """Set simulated breakdown voltage (simulation mode only)."""
        self._driver.sim_set_vbd(vbd)

    def sim_set_dark_current(self, i_dark: float):
        """Set simulated dark current floor (simulation mode only)."""
        self._driver.sim_set_dark_current(i_dark)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
