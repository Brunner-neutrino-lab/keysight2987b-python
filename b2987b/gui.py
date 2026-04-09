"""
b2987b/gui.py

Standalone PyQt5 GUI for the Keysight B2987B electrometer.

Launch directly:
    python -m b2987b.gui

Tabs:
    Connection  — VISA string, mode, connect/disconnect
    Bias        — set voltage, enable/disable output (bias-only, no measurement)
    Single      — measure current at a single voltage
    IV Sweep    — configure and run a full IV sweep with live plot
"""

import sys
import time
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTextEdit, QTabWidget, QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .controller import B2987BController, SweepResult
from .driver import DEFAULT_VISA


# ---------------------------------------------------------------------------
# Worker signals
# ---------------------------------------------------------------------------

class _Signals(QObject):
    status       = pyqtSignal(str)
    connected    = pyqtSignal(bool, str)
    single_done  = pyqtSignal(float)           # current value (A)
    sweep_done   = pyqtSignal(object)          # SweepResult
    bias_changed = pyqtSignal(float, bool)     # (voltage, output_on)


class _ConnectWorker(QThread):
    def __init__(self, ctrl, signals):
        super().__init__()
        self._ctrl = ctrl; self._signals = signals

    def run(self):
        try:
            self._ctrl.connect()
            self._signals.connected.emit(True, self._ctrl.identify())
        except Exception as e:
            self._signals.connected.emit(False, str(e))


class _SingleMeasureWorker(QThread):
    def __init__(self, ctrl, voltage, signals):
        super().__init__()
        self._ctrl = ctrl; self._v = voltage; self._signals = signals

    def run(self):
        try:
            current = self._ctrl.measure_current(self._v)
            self._signals.single_done.emit(current)
        except Exception as e:
            self._signals.status.emit(f"Measure error: {e}")


class _SweepWorker(QThread):
    def __init__(self, ctrl, voltages, kwargs, signals):
        super().__init__()
        self._ctrl = ctrl; self._voltages = voltages
        self._kwargs = kwargs; self._signals = signals

    def run(self):
        try:
            result = self._ctrl.sweep(self._voltages, **self._kwargs)
            self._signals.sweep_done.emit(result)
        except Exception as e:
            self._signals.status.emit(f"Sweep error: {e}")


class _BiasWorker(QThread):
    def __init__(self, ctrl, voltage, on, signals):
        super().__init__()
        self._ctrl = ctrl; self._v = voltage
        self._on = on; self._signals = signals

    def run(self):
        try:
            if self._on:
                self._ctrl.set_bias(self._v)
            else:
                self._ctrl.bias_off()
            self._signals.bias_changed.emit(self._v, self._on)
        except Exception as e:
            self._signals.status.emit(f"Bias error: {e}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class B2987BWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keysight B2987B Electrometer Control")
        self.resize(860, 680)

        self._ctrl:   B2987BController | None = None
        self._signals = _Signals()
        self._worker  = None
        self._last_result: SweepResult | None = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        tabs = QTabWidget()
        tabs.addTab(self._build_connection_tab(), "Connection")
        tabs.addTab(self._build_bias_tab(),       "Bias")
        tabs.addTab(self._build_single_tab(),     "Single Measure")
        tabs.addTab(self._build_sweep_tab(),      "IV Sweep")

        lay.addWidget(tabs)
        lay.addWidget(self._build_log())

    # --- Connection ---
    def _build_connection_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox("Instrument Connection")
        g   = QGridLayout(box)

        g.addWidget(QLabel("VISA Resource:"), 0, 0)
        self._visa_edit = QLineEdit(DEFAULT_VISA)
        g.addWidget(self._visa_edit, 0, 1)

        g.addWidget(QLabel("Mode:"), 1, 0)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["simulation", "hardware"])
        g.addWidget(self._mode_combo, 1, 1)

        btn_row = QHBoxLayout()
        self._connect_btn    = QPushButton("Connect")
        self._disconnect_btn = QPushButton("Disconnect")
        self._test_btn       = QPushButton("Test")
        self._disconnect_btn.setEnabled(False)
        btn_row.addWidget(self._connect_btn)
        btn_row.addWidget(self._disconnect_btn)
        btn_row.addWidget(self._test_btn)
        g.addLayout(btn_row, 2, 0, 1, 2)

        self._conn_label = QLabel("Not connected")
        self._conn_label.setStyleSheet("color: red; font-weight: bold;")
        g.addWidget(self._conn_label, 3, 0, 1, 2)

        lay.addWidget(box)
        lay.addStretch()
        return w

    # --- Bias only ---
    def _build_bias_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Bias Control (no measurement)")
        g   = QGridLayout(box)

        g.addWidget(QLabel("Voltage (V):"), 0, 0)
        self._bias_v_spin = QDoubleSpinBox()
        self._bias_v_spin.setRange(-1000.0, 1000.0)
        self._bias_v_spin.setValue(0.0)
        self._bias_v_spin.setSingleStep(0.5)
        self._bias_v_spin.setDecimals(3)
        g.addWidget(self._bias_v_spin, 0, 1)

        g.addWidget(QLabel("Source range (V):"), 1, 0)
        self._bias_range_combo = QComboBox()
        self._bias_range_combo.addItems(["20", "1000"])
        self._bias_range_combo.setCurrentText("1000")
        g.addWidget(self._bias_range_combo, 1, 1)

        g.addWidget(QLabel("Current-limiting resistor:"), 2, 0)
        self._bias_rlim_check = QCheckBox()
        g.addWidget(self._bias_rlim_check, 2, 1)

        btn_row = QHBoxLayout()
        self._bias_on_btn  = QPushButton("Output ON")
        self._bias_off_btn = QPushButton("Output OFF")
        self._bias_on_btn.setEnabled(False)
        self._bias_off_btn.setEnabled(False)
        btn_row.addWidget(self._bias_on_btn)
        btn_row.addWidget(self._bias_off_btn)
        g.addLayout(btn_row, 3, 0, 1, 2)

        self._bias_status_label = QLabel("Output: OFF")
        self._bias_status_label.setStyleSheet("color: red;")
        g.addWidget(self._bias_status_label, 4, 0, 1, 2)

        lay.addWidget(box)
        lay.addStretch()
        return w

    # --- Single measure ---
    def _build_single_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Single Current Measurement")
        g   = QGridLayout(box)

        g.addWidget(QLabel("Voltage (V):"), 0, 0)
        self._single_v_spin = QDoubleSpinBox()
        self._single_v_spin.setRange(-1000.0, 1000.0)
        self._single_v_spin.setValue(0.0)
        self._single_v_spin.setSingleStep(1.0)
        self._single_v_spin.setDecimals(3)
        g.addWidget(self._single_v_spin, 0, 1)

        self._single_btn = QPushButton("Measure")
        self._single_btn.setEnabled(False)
        g.addWidget(self._single_btn, 1, 0, 1, 2)

        self._single_result_label = QLabel("—")
        self._single_result_label.setFont(QFont("Courier", 14))
        self._single_result_label.setAlignment(Qt.AlignCenter)
        g.addWidget(self._single_result_label, 2, 0, 1, 2)

        lay.addWidget(box)
        lay.addStretch()
        return w

    # --- IV sweep ---
    def _build_sweep_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        # Voltage range
        rng_box = QGroupBox("Voltage Sweep Range")
        rg      = QGridLayout(rng_box)
        rg.addWidget(QLabel("Start (V):"), 0, 0)
        self._sweep_start = QDoubleSpinBox()
        self._sweep_start.setRange(-1000, 1000); self._sweep_start.setValue(40.0)
        rg.addWidget(self._sweep_start, 0, 1)

        rg.addWidget(QLabel("Stop (V):"), 1, 0)
        self._sweep_stop = QDoubleSpinBox()
        self._sweep_stop.setRange(-1000, 1000); self._sweep_stop.setValue(55.0)
        rg.addWidget(self._sweep_stop, 1, 1)

        rg.addWidget(QLabel("Step (V):"), 2, 0)
        self._sweep_step = QDoubleSpinBox()
        self._sweep_step.setRange(0.001, 100); self._sweep_step.setValue(0.05)
        self._sweep_step.setDecimals(3)
        rg.addWidget(self._sweep_step, 2, 1)
        lay.addWidget(rng_box)

        # Sweep parameters
        cfg_box = QGroupBox("Sweep Parameters")
        cg      = QGridLayout(cfg_box)

        cg.addWidget(QLabel("Points per voltage:"), 0, 0)
        self._sweep_npts = QSpinBox()
        self._sweep_npts.setRange(1, 1000); self._sweep_npts.setValue(5)
        cg.addWidget(self._sweep_npts, 0, 1)

        cg.addWidget(QLabel("Delay per point (s):"), 1, 0)
        self._sweep_delay = QDoubleSpinBox()
        self._sweep_delay.setRange(0, 100); self._sweep_delay.setValue(0.1)
        self._sweep_delay.setDecimals(3)
        cg.addWidget(self._sweep_delay, 1, 1)

        cg.addWidget(QLabel("Measure sense voltage:"), 2, 0)
        self._sweep_measv = QCheckBox(); self._sweep_measv.setChecked(False)
        cg.addWidget(self._sweep_measv, 2, 1)

        cg.addWidget(QLabel("Log scale:"), 3, 0)
        self._sweep_log = QCheckBox(); self._sweep_log.setChecked(True)
        cg.addWidget(self._sweep_log, 3, 1)

        lay.addWidget(cfg_box)

        # Run controls
        btn_row = QHBoxLayout()
        self._sweep_btn = QPushButton("Run Sweep")
        self._sweep_btn.setEnabled(False)
        btn_row.addWidget(self._sweep_btn)
        self._sweep_progress = QLabel("Ready")
        btn_row.addWidget(self._sweep_progress)
        lay.addLayout(btn_row)

        # Plot
        if HAS_MPL:
            self._iv_fig    = Figure(figsize=(7, 3))
            self._iv_canvas = FigureCanvas(self._iv_fig)
            self._iv_ax     = self._iv_fig.add_subplot(111)
            self._iv_ax.set_xlabel("Bias (V)")
            self._iv_ax.set_ylabel("Current (A)")
            self._iv_ax.set_title("IV Curve")
            lay.addWidget(self._iv_canvas)

        return w

    def _build_log(self) -> QWidget:
        box = QGroupBox("Status Log")
        lay = QVBoxLayout(box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        self._log.setFont(QFont("Courier", 9))
        lay.addWidget(self._log)
        return box

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._test_btn.clicked.connect(self._on_test)
        self._bias_on_btn.clicked.connect(self._on_bias_on)
        self._bias_off_btn.clicked.connect(self._on_bias_off)
        self._single_btn.clicked.connect(self._on_single_measure)
        self._sweep_btn.clicked.connect(self._on_sweep)

        self._signals.status.connect(self._log_msg)
        self._signals.connected.connect(self._on_connect_result)
        self._signals.single_done.connect(self._on_single_done)
        self._signals.sweep_done.connect(self._on_sweep_done)
        self._signals.bias_changed.connect(self._on_bias_changed)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connect(self):
        self._ctrl = B2987BController(
            visa=self._visa_edit.text().strip(),
            mode=self._mode_combo.currentText(),
        )
        self._log_msg("Connecting...")
        self._connect_btn.setEnabled(False)
        w = _ConnectWorker(self._ctrl, self._signals)
        w.start(); self._worker = w

    def _on_connect_result(self, ok: bool, msg: str):
        self._connect_btn.setEnabled(True)
        if ok:
            self._conn_label.setText(f"Connected: {msg}")
            self._conn_label.setStyleSheet("color: green; font-weight: bold;")
            self._disconnect_btn.setEnabled(True)
            for btn in (self._bias_on_btn, self._bias_off_btn,
                        self._single_btn, self._sweep_btn):
                btn.setEnabled(True)
        else:
            self._conn_label.setText("Failed")
            self._conn_label.setStyleSheet("color: red; font-weight: bold;")
            self._ctrl = None
        self._log_msg(("Connected: " if ok else "FAILED: ") + msg)

    def _on_disconnect(self):
        if self._ctrl:
            try: self._ctrl.disconnect()
            except Exception as e: self._log_msg(f"Disconnect: {e}")
            self._ctrl = None
        self._conn_label.setText("Not connected")
        self._conn_label.setStyleSheet("color: red; font-weight: bold;")
        self._disconnect_btn.setEnabled(False)
        for btn in (self._bias_on_btn, self._bias_off_btn,
                    self._single_btn, self._sweep_btn):
            btn.setEnabled(False)
        self._log_msg("Disconnected.")

    def _on_test(self):
        config = {"visa": self._visa_edit.text().strip(),
                  "mode": self._mode_combo.currentText()}
        class _T(QThread):
            done = pyqtSignal(bool, str)
            def run(self_):
                ok, msg = B2987BController.test(config)
                self_.done.emit(ok, msg)
        t = _T(self)
        t.done.connect(lambda ok, m: self._log_msg(f"Test {'OK' if ok else 'FAILED'}: {m}"))
        t.start(); self._worker = t

    def _on_bias_on(self):
        if not self._ctrl: return
        v = self._bias_v_spin.value()
        self._ctrl.configure_sweep(
            source_range=float(self._bias_range_combo.currentText()),
            current_limit=self._bias_rlim_check.isChecked(),
        )
        self._bias_on_btn.setEnabled(False)
        w = _BiasWorker(self._ctrl, v, True, self._signals)
        w.start(); self._worker = w

    def _on_bias_off(self):
        if not self._ctrl: return
        w = _BiasWorker(self._ctrl, 0.0, False, self._signals)
        w.start(); self._worker = w

    def _on_bias_changed(self, voltage: float, on: bool):
        self._bias_on_btn.setEnabled(True)
        if on:
            self._bias_status_label.setText(f"Output: ON  {voltage:.3f} V")
            self._bias_status_label.setStyleSheet("color: green;")
        else:
            self._bias_status_label.setText("Output: OFF")
            self._bias_status_label.setStyleSheet("color: red;")
        self._log_msg(f"Bias {'ON ' + str(voltage) + ' V' if on else 'OFF'}")

    def _on_single_measure(self):
        if not self._ctrl: return
        v = self._single_v_spin.value()
        self._single_btn.setEnabled(False)
        self._single_result_label.setText("Measuring…")
        w = _SingleMeasureWorker(self._ctrl, v, self._signals)
        w.start(); self._worker = w

    def _on_single_done(self, current: float):
        self._single_btn.setEnabled(True)
        self._single_result_label.setText(f"{current:.4e} A")
        self._log_msg(f"Single measurement: {current:.4e} A")

    def _on_sweep(self):
        if not self._ctrl: return
        start = self._sweep_start.value()
        stop  = self._sweep_stop.value()
        step  = self._sweep_step.value()
        if step <= 0:
            self._log_msg("Step must be > 0.")
            return
        voltages = np.arange(start, stop + step * 0.5, step).tolist()
        kwargs   = {
            "n_per_voltage":  self._sweep_npts.value(),
            "delay_s":        self._sweep_delay.value(),
            "measure_voltage": self._sweep_measv.isChecked(),
        }
        self._sweep_btn.setEnabled(False)
        self._sweep_progress.setText(f"Running sweep ({len(voltages)} points)…")
        self._log_msg(f"Starting IV sweep: {start} → {stop} V, step={step} V")
        w = _SweepWorker(self._ctrl, voltages, kwargs, self._signals)
        w.start(); self._worker = w

    def _on_sweep_done(self, result: SweepResult):
        self._last_result = result
        self._sweep_btn.setEnabled(True)
        n = len(result.avg_source_v)
        self._sweep_progress.setText(f"Done — {n} voltage points")
        self._log_msg(f"Sweep complete: {n} points, "
                      f"I_max={result.avg_current_a.max():.3e} A")
        if HAS_MPL:
            self._refresh_iv_plot(result)

    def _refresh_iv_plot(self, result: SweepResult):
        self._iv_ax.clear()
        v = result.avg_source_v
        i = np.abs(result.avg_current_a)
        self._iv_ax.errorbar(v, i, yerr=result.err_current_a,
                             fmt='o-', ms=3, lw=1, capsize=2)
        if self._sweep_log.isChecked():
            self._iv_ax.set_yscale("log")
        else:
            self._iv_ax.set_yscale("linear")
        self._iv_ax.set_xlabel("Bias (V)")
        self._iv_ax.set_ylabel("|Current| (A)")
        self._iv_ax.set_title("IV Curve")
        self._iv_ax.grid(True, which="both", alpha=0.3)
        self._iv_fig.tight_layout()
        self._iv_canvas.draw()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_msg(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        if self._ctrl:
            try: self._ctrl.disconnect()
            except Exception: pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = B2987BWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
