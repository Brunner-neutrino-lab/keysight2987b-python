"""
b2987b/driver.py

Low-level SCPI interface to the Keysight B2987B electrometer/picoammeter.

Handles only instrument communication — no experiment logic, no Qt.

Two modes:
  "hardware"   — connects via pyvisa (USB or Ethernet)
  "simulation" — returns synthetic IV data for development without hardware

SCPI commands verified against the existing B2987b-Control-Program/Python_Control/
electrometer.py implementation.
"""

import time
import math
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VISA    = "USB0::2391::37912::MY54321112::0::INSTR"
TIMEOUT_MS      = 10_000
IDN_EXPECTED    = "Keysight Technologies,B2987"

# Source voltage limits
VMAX_20V   =   20.0    # V (20 V range)
VMAX_1000V = 1000.0    # V (1000 V range)

# Simulation: simple SiPM dark IV model
# I(V) = I_dark * exp(alpha*(V - VBD)) for V < VBD  (leakage)
#       = I_dark * (1 + gain*(V-VBD))  for V >= VBD (avalanche)
SIM_VBD      = 47.0     # V  approximate breakdown
SIM_I_DARK   = 1e-12   # A  dark current floor
SIM_ALPHA    = 0.05    # 1/V leakage slope
SIM_GAIN_K   = 5e-9   # A/V post-breakdown slope (current per overvoltage)
SIM_NOISE    = 5e-14   # A  current noise floor


# ---------------------------------------------------------------------------
# SCPI helpers (range and aperture composition)
# ---------------------------------------------------------------------------

def _apply_range(inst, sense: str,
                 auto: bool, fixed_value: float | None,
                 lower: float, upper: float) -> None:
    """Set :SENS1:<sense>:RANG to auto-with-limits or a fixed value.

    `sense` is "CURR" or "VOLT". Limits and fixed are in instrument units
    (A for CURR, V for VOLT).
    """
    if auto:
        inst.write(f':SENS1:{sense}:RANG:AUTO ON;'
                   f'AUTO:ULIM {upper:.6e};LLIM {lower:.6e}')
    else:
        if fixed_value is None:
            raise ValueError(f"{sense}: fixed range requested but no value given")
        inst.write(f':SENS1:{sense}:RANG:AUTO OFF;'
                   f':SENS1:{sense}:RANG {fixed_value:.6e}')


def _apply_aperture(inst, sense: str,
                    mode: str, fixed_s: float | None) -> None:
    """Set :SENS1:<sense>:APER auto-mode or fixed seconds.

    mode is one of AUTO|SHORT|MEDIUM|LONG|FIXED (case-insensitive).
    AUTO|SHORT|MEDIUM|LONG → :APER:AUTO ON with AUTO:MODE accordingly
    (AUTO maps to LONG, the firmware's documented default).
    FIXED → :APER:AUTO OFF with explicit seconds.
    """
    mode = mode.upper()
    speed_map = {"AUTO": "LONG", "SHORT": "SHOR", "MEDIUM": "MED", "LONG": "LONG"}
    if mode in speed_map:
        inst.write(f':SENS1:{sense}:APER:AUTO ON;'
                   f'AUTO:MODE {speed_map[mode]}')
    elif mode == "FIXED":
        if fixed_s is None:
            raise ValueError(f"{sense}: FIXED aperture requested but no value given")
        inst.write(f':SENS1:{sense}:APER:AUTO OFF;'
                   f':SENS1:{sense}:APER {fixed_s:.6e}')
    else:
        raise ValueError(f"{sense}: unknown aperture mode {mode!r}; "
                         "expected AUTO|SHORT|MEDIUM|LONG|FIXED")


class B2987BDriver:
    """
    Low-level SCPI driver for the Keysight B2987B.

    Parameters
    ----------
    visa : str
        VISA resource string. USB or TCPIP.
    mode : str
        "hardware" or "simulation".
    """

    def __init__(self, visa: str = DEFAULT_VISA, mode: str = "simulation"):
        if mode not in ("hardware", "simulation"):
            raise ValueError(f"mode must be 'hardware' or 'simulation', got {mode!r}")
        self._visa_str   = visa
        self._mode       = mode
        self._inst       = None
        self._rm         = None
        self._connected  = False

        # Current instrument state
        self._source_voltage  = 0.0     # V — last commanded voltage
        self._output_on       = False
        self._ammeter_on      = False

        # Simulation tuning
        self._sim_vbd     = SIM_VBD
        self._sim_i_dark  = SIM_I_DARK
        self._sim_alpha   = SIM_ALPHA
        self._sim_gain_k  = SIM_GAIN_K
        self._sim_noise   = SIM_NOISE
        self._sim_rng     = np.random.default_rng()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self._connected:
            return
        if self._mode == "hardware":
            self._connect_hardware()
        self._connected = True

    def disconnect(self):
        if not self._connected:
            return
        if self._mode == "hardware":
            self._disconnect_hardware()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Source control
    # ------------------------------------------------------------------

    def set_voltage(self, voltage: float):
        """
        Set the source voltage (does not enable output).

        Parameters
        ----------
        voltage : float
            Target voltage in volts. Sign determines polarity.
        """
        self._source_voltage = voltage
        if self._mode == "hardware":
            self._inst.write(f":SOUR1:VOLT:MODE FIX")
            self._inst.write(f":SOUR1:VOLT {voltage:.6f}")

    def set_source_range(self, range_v: float):
        """
        Set source voltage range.

        Parameters
        ----------
        range_v : float
            20 for ±20 V range; 1000 for ±1000 V range.
        """
        if range_v not in (20, 1000, -1000):
            raise ValueError(f"source_range must be 20 or 1000, got {range_v}")
        if self._mode == "hardware":
            self._inst.write(f":SOUR1:VOLT:RANG {range_v:.0f}")

    def set_current_limit(self, enable: bool):
        """Enable or disable the internal current-limiting resistor."""
        if self._mode == "hardware":
            state = "ON" if enable else "OFF"
            self._inst.write(f":SOUR1:VOLT:RLIM:STAT {state}")

    def output_on(self):
        """Enable the voltage source output."""
        self._output_on = True
        if self._mode == "hardware":
            self._inst.write(":OUTP1 ON")

    def output_off(self):
        """Disable the voltage source output."""
        self._output_on = False
        if self._mode == "hardware":
            self._inst.write(":OUTP1 OFF")

    def ammeter_on(self):
        """Enable the ammeter input."""
        self._ammeter_on = True
        if self._mode == "hardware":
            self._inst.write(":INP1 ON")

    def ammeter_off(self):
        """Disable the ammeter input."""
        self._ammeter_on = False
        if self._mode == "hardware":
            self._inst.write(":INP1 OFF")

    # ------------------------------------------------------------------
    # Single-point measurement
    # ------------------------------------------------------------------

    def measure_current(self) -> float:
        """
        Read a single current measurement at the current source voltage.

        Returns
        -------
        float
            Current in amperes.
        """
        if self._mode == "hardware":
            self._inst.write(":SENS1:FUNC 'CURR'")
            self._inst.write(":TRIG1:ALL:SOUR AINT;COUN 1")
            self._inst.write(":INIT:ALL (@1)")
            self._inst.query("*OPC?")
            self._inst.write(":FETC:ARR:CURR? (@1)")
            raw = self._inst.read_ascii_values()
            return float(raw[0])
        else:
            return self._sim_current(self._source_voltage)

    def measure_voltage(self) -> float:
        """
        Read a single voltage measurement (sense terminal).

        Returns
        -------
        float
            Voltage in volts.
        """
        if self._mode == "hardware":
            self._inst.write(":SENS1:FUNC 'CURR','VOLT'")
            self._inst.write(":TRIG1:ALL:SOUR AINT;COUN 1")
            self._inst.write(":INIT:ALL (@1)")
            self._inst.query("*OPC?")
            self._inst.write(":FETC:ARR:VOLT? (@1)")
            raw = self._inst.read_ascii_values()
            return float(raw[0])
        else:
            return self._source_voltage

    # ------------------------------------------------------------------
    # List sweep
    # ------------------------------------------------------------------

    def configure_list_sweep(self,
                              voltages: list[float],
                              n_points_per_voltage: int = 1,
                              delay_s: float = 0.1,
                              measure_voltage: bool = False,
                              # Current sense
                              current_range_auto:  bool        = True,
                              current_range_v:     float | None = None,
                              current_range_lower_a: float     = 2e-12,
                              current_range_upper_a: float     = 2e-2,
                              current_aperture_mode: str       = "AUTO",   # AUTO|SHORT|MEDIUM|LONG|FIXED
                              current_aperture_s:  float | None = None,
                              # Voltage sense (only if measure_voltage=True)
                              voltage_range_auto:  bool        = True,
                              voltage_range_v:     float | None = None,
                              voltage_range_lower_v: float     = 2.0,
                              voltage_range_upper_v: float     = 20.0,
                              voltage_aperture_mode: str       = "AUTO",   # AUTO|SHORT|MEDIUM|LONG|FIXED
                              voltage_aperture_s:  float | None = None,
                              # Zero-reference subtraction (defaults preserve prior behaviour)
                              zero_reference:      bool        = True):
        """
        Configure a list sweep.

        Parameters
        ----------
        voltages : list[float]
            Ordered list of source voltages for the sweep (V).
        n_points_per_voltage : int
            Number of current measurements at each voltage step.
        delay_s : float
            Trigger delay between points (s).
        measure_voltage : bool
            If True, also measure the sense voltage at each point.
        current_range_auto : bool
            True → auto-range with [current_range_lower_a, current_range_upper_a].
            False → fixed at current_range_v.
        current_range_v : float, optional
            Fixed current range in A (2e-12 … 2e-2). Used when current_range_auto=False.
        current_range_lower_a, current_range_upper_a : float
            Lower/upper bound for current auto-range. Defaults: 2e-12 … 2e-2.
        current_aperture_mode : str
            "AUTO"   → auto-aperture (firmware picks default speed, LONG).
            "SHORT", "MEDIUM", "LONG" → auto-aperture pinned to that speed.
            "FIXED"  → use current_aperture_s explicitly.
        current_aperture_s : float, optional
            Fixed integration aperture in seconds (1e-5 … 2). Used when
            current_aperture_mode="FIXED".
        voltage_range_auto, voltage_range_v, voltage_range_lower_v,
        voltage_range_upper_v, voltage_aperture_mode, voltage_aperture_s :
            Same shape as the current_* equivalents but for the voltmeter.
            Only applied if measure_voltage=True. Voltage range is 2 … 20 V.
        zero_reference : bool
            If True, acquire and subtract a zero-current reference at the
            start of the sweep (preserves prior driver behaviour).
        """
        # Build the expanded voltage list (each voltage repeated n times)
        sweep_list = []
        for v in voltages:
            sweep_list.extend([v] * n_points_per_voltage)
        n_total = len(sweep_list)

        self._sweep_voltages        = voltages
        self._sweep_list            = sweep_list
        self._sweep_n_per_v         = n_points_per_voltage
        self._sweep_measure_voltage = measure_voltage
        self._sweep_n_total         = n_total

        if self._mode == "hardware":
            inst = self._inst
            # Build SCPI list string
            list_str = ",".join(f"{v:.6f}" for v in sweep_list)
            inst.write(f":SOUR1:LIST:VOLT {list_str};:SOUR1:LIST:VOLT:STAR 1")
            inst.write(":SOUR1:VOLT:MODE LIST")

            # Sense functions
            if measure_voltage:
                inst.write(':SENS1:FUNC "CURR","VOLT"')
                _apply_range(inst, "VOLT",
                             voltage_range_auto, voltage_range_v,
                             voltage_range_lower_v, voltage_range_upper_v)
                _apply_aperture(inst, "VOLT",
                                voltage_aperture_mode, voltage_aperture_s)
            else:
                inst.write(':SENS1:FUNC "CURR"')

            # Current sense range + aperture
            _apply_range(inst, "CURR",
                         current_range_auto, current_range_v,
                         current_range_lower_a, current_range_upper_a)
            _apply_aperture(inst, "CURR",
                            current_aperture_mode, current_aperture_s)

            # Trigger: auto internal, with delay, n_total counts
            inst.write(f":TRIG1:ALL:SOUR AINT;DEL {delay_s:.4f};COUN {n_total}")

            # Zero reference (offset current subtraction)
            if zero_reference:
                inst.write(":SENS1:CURR:REF:ACQ")
                inst.write(":SENS1:CURR:REF:STAT 1")
            else:
                inst.write(":SENS1:CURR:REF:STAT 0")

    def run_sweep(self, timeout_s: float = 600.0) -> dict:
        """
        Execute the configured list sweep and return the results.

        Returns
        -------
        dict with keys:
            "source_v"   : list[float]  commanded voltage at each point
            "current_a"  : list[float]  measured current (A)
            "voltage_v"  : list[float]  measured voltage (V) or NaN if not measured
            "timestamp_s": list[float]  UTC time at each point (s)
        """
        if self._mode == "hardware":
            return self._run_sweep_hardware(timeout_s)
        else:
            return self._run_sweep_simulation()

    def abort(self):
        """Abort any ongoing acquisition."""
        if self._mode == "hardware":
            try:
                self._inst.write(":ABOR:ALL (@1)")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Hardware internals
    # ------------------------------------------------------------------

    def _connect_hardware(self):
        try:
            import pyvisa
        except ImportError as e:
            raise ImportError(
                "pyvisa not installed. Run: pip install pyvisa pyvisa-py"
            ) from e

        self._rm   = pyvisa.ResourceManager()
        count = 0
        while count < 3:
            try:
                self._inst = self._rm.open_resource(self._visa_str)
                break
            except Exception:
                count += 1
                time.sleep(1)
        else:
            raise RuntimeError(
                f"Could not open VISA resource {self._visa_str!r} after 3 attempts."
            )

        # Raw SOCKET sessions (TCPIP::host::5025::SOCKET) are stateless on the
        # instrument side — no VXI-11 RPC link table to leak — and need
        # explicit terminations.  Default pyvisa values differ between
        # transport types; setting both is safe and avoids hangs on read.
        is_socket = "SOCKET" in self._visa_str.upper()
        if is_socket:
            self._inst.read_termination  = "\n"
            self._inst.write_termination = "\n"

        # device_clear() is supposed to abort pending operations from a prior
        # session, but the B2987's VXI-11 service occasionally hangs on the
        # clear RPC when the previous session didn't shut down cleanly. The
        # *RST that follows accomplishes the same software reset, so a
        # failed clear is non-fatal — log and continue.  SOCKET sessions
        # don't expose device_clear, so we just skip it.
        if not is_socket:
            try:
                self._inst.timeout = 2000
                self._inst.clear()
            except Exception:
                pass
        self._inst.timeout = TIMEOUT_MS

        idn = self._inst.query("*IDN?")
        if IDN_EXPECTED not in idn:
            self._inst.close()
            raise RuntimeError(
                f"IDN mismatch. Expected {IDN_EXPECTED!r}, got {idn!r}\n"
                f"Check VISA string: {self._visa_str!r}"
            )
        self._reset()

    def _reset(self):
        """Reset to default state."""
        self._inst.write("*RST")
        self._inst.write(
            ":FORM ASC;:FORM:DIG ASC;:FORM:ELEM:CALC CALC,TIME,STAT;"
            ":FORM:SREG ASC;:FORM:BORD NORM;*ESE 60;*SRE 48;*CLS;"
        )

    def _disconnect_hardware(self):
        if self._inst is not None:
            is_socket = "SOCKET" in (self._visa_str or "").upper()
            try:
                self.abort()
                self.ammeter_off()
                self.output_off()
            except Exception:
                pass
            # device_clear is a VXI-11/USB IEEE-488 operation; SOCKET
            # sessions don't implement it and pyvisa raises.  Skip it there.
            if not is_socket:
                try:
                    self._inst.clear()
                except Exception:
                    pass
            try:
                self._inst.close()
            except Exception:
                pass
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass

    def _run_sweep_hardware(self, timeout_s: float) -> dict:
        inst      = self._inst
        n_total   = self._sweep_n_total
        meas_volt = self._sweep_measure_voltage

        self.output_on()
        self.ammeter_on()
        time.sleep(0.5)

        inst.write(":INIT:ALL (@1)")
        start_utc = time.time()

        # Wait for acquisition
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            inst.write(":STAT:OPER:COND?")
            resp = inst.read()
            if len(resp) >= 3 and resp[2] == "7":
                break
            time.sleep(0.1)
        else:
            self.abort()
            self.ammeter_off()
            self.output_off()
            raise TimeoutError(f"Sweep did not complete within {timeout_s:.0f}s")

        # Fetch results
        inst.write(":FETC:ARR:SOUR? (@1)")
        source = inst.read_ascii_values()

        inst.write(":FETC:ARR:CURR? (@1)")
        current = inst.read_ascii_values()

        if meas_volt:
            inst.write(":FETC:ARR:VOLT? (@1)")
            voltage = inst.read_ascii_values()
        else:
            voltage = [float("nan")] * n_total

        inst.write(":FETC:ARR:TIME? (@1)")
        rel_time = inst.read_ascii_values()
        utc_time = [start_utc + t for t in rel_time]

        self.ammeter_off()
        self.output_off()

        return {
            "source_v":    list(source),
            "current_a":   list(current),
            "voltage_v":   list(voltage),
            "timestamp_s": utc_time,
        }

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------

    def _sim_current(self, voltage: float) -> float:
        """Simple SiPM dark IV model."""
        rng  = self._sim_rng
        vbd  = self._sim_vbd
        ov   = voltage - vbd

        if voltage <= 0:
            base = self._sim_i_dark * 0.01
        elif voltage < vbd:
            base = self._sim_i_dark * math.exp(self._sim_alpha * voltage)
        else:
            base = self._sim_i_dark * math.exp(self._sim_alpha * vbd) \
                   + self._sim_gain_k * ov ** 2

        noise = abs(rng.normal(0, self._sim_noise))
        return base + noise

    def _run_sweep_simulation(self) -> dict:
        """Generate synthetic IV sweep data."""
        source    = []
        current   = []
        voltage   = []
        timestamp = []
        t0 = time.time()
        dt = 0.05  # simulated seconds per point

        for i, v in enumerate(self._sweep_list):
            source.append(v)
            current.append(self._sim_current(v))
            voltage.append(v if self._sweep_measure_voltage else float("nan"))
            timestamp.append(t0 + i * dt)

        return {
            "source_v":    source,
            "current_a":   current,
            "voltage_v":   voltage,
            "timestamp_s": timestamp,
        }

    def sim_set_vbd(self, vbd: float):
        """Set simulated breakdown voltage (simulation mode only)."""
        self._sim_vbd = vbd

    def sim_set_dark_current(self, i_dark: float):
        """Set simulated dark current floor in A (simulation mode only)."""
        self._sim_i_dark = i_dark

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
