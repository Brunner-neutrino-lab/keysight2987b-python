"""
b2987b/gui.py

NiceGUI control panel for the Keysight B2987B electrometer.

Two modes:

  - Standalone: `python -m b2987b.gui` opens a browser tab with full
    control including a Connection panel that creates and owns its own
    B2987BController. Useful for non-automated bench work.

  - Embedded: `build_page(get_controller=..., show_connection=False)` is
    called from inside a parent NiceGUI app (e.g. the ETS DAQ) and
    operates on a shared controller. The Connection panel is hidden;
    connect/disconnect happens in the parent's own Connections tab.

The four feature tabs (Bias, Single Measure, IV Sweep) are identical
between modes. All blocking instrument calls run in a thread pool via
asyncio.to_thread so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

import numpy as np
from nicegui import ui

from .controller import B2987BController, SweepResult
from .driver     import DEFAULT_VISA


# ---------------------------------------------------------------------------
# Style — matches xsphere/DAQ "register" theme; harmless if the host page
# already injects the same vars.
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg:#11151c; --panel:#1b2230; --panel2:#232c3d;
  --fg:#dde3ee; --mut:#8a93a6;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --acc:#58a6ff;
  --line:#2d3648;
}
html, body, .nicegui-content { background:var(--bg) !important; color:var(--fg);
  font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:0; }
.pill { padding:.15rem .55rem; border-radius:999px; font-size:.78rem;
  font-weight:600; white-space:nowrap; display:inline-flex; align-items:center; gap:.3rem; }
.pill.ok   { background:rgba(63,185,80,.18);  color:var(--ok); }
.pill.bad  { background:rgba(248,81,73,.18);  color:var(--bad); }
.pill.warn { background:rgba(210,153,34,.18); color:var(--warn); }
.pill.mut  { background:rgba(138,147,166,.15);color:var(--mut); }
.q-card, .b2987-card {
  background:var(--panel) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:10px;
  box-shadow:none !important; padding:.55rem .85rem .7rem !important;
}
.b2987-card h2 { font-size:.92rem; margin:.05rem 0 .45rem; color:var(--acc);
  font-weight:600; letter-spacing:.3px; }
.q-btn { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line) !important; border-radius:6px !important;
  box-shadow:none !important; padding:.18rem .65rem !important;
  min-height:32px !important; text-transform:none !important; }
.q-btn:hover { border-color:var(--acc) !important; }
.q-btn[data-q-color="primary"], .q-btn.bg-primary {
  background:var(--acc) !important; color:#08111f !important;
  border-color:var(--acc) !important; font-weight:600 !important; }
.q-btn[data-q-color="negative"], .q-btn.bg-negative {
  background:transparent !important; color:var(--bad) !important;
  border-color:var(--bad) !important; }
.q-field__control, .q-field--filled .q-field__control {
  background:var(--panel2) !important; border:1px solid var(--line) !important;
  border-radius:6px !important; min-height:32px !important; color:var(--fg) !important; }
.q-field__label, .q-field__native, .q-field input { color:var(--fg) !important; }
.q-field__label { color:var(--mut) !important; }
.q-field--filled .q-field__control:before,
.q-field--filled .q-field__control:after { display:none !important; }
.q-tab { color:var(--mut) !important; text-transform:none !important; }
.q-tab--active { color:var(--acc) !important; }
.q-tab__indicator { background:var(--acc) !important; }
.q-log, .nicegui-log { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:6px;
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; }
.num { font-variant-numeric:tabular-nums; }
"""


# ---------------------------------------------------------------------------
# build_page — the reusable GUI
# ---------------------------------------------------------------------------

def build_page(get_controller: Optional[Callable[[], Optional[B2987BController]]] = None,
               *,
               show_connection: Optional[bool] = None) -> None:
    """
    Render the B2987B control panel into the current NiceGUI container.

    Parameters
    ----------
    get_controller : callable returning B2987BController | None, optional
        If given, the GUI calls this on every action to get the current
        controller. Lets an embedding parent share its live controller
        (e.g. `lambda: HUB.elec`).  If `None` (default), the GUI owns
        its own controller managed via the Connection panel.
    show_connection : bool, optional
        Whether to render the Connection panel. Defaults to:
        `True` for standalone (get_controller=None), `False` for embedded.
    """
    if show_connection is None:
        show_connection = (get_controller is None)

    # In standalone mode, hold a controller in closure state.
    _own = {"ctrl": None, "last_sweep": None}
    if get_controller is None:
        def get_controller():
            return _own["ctrl"]

    # --- Helpers -----------------------------------------------------------

    log = ui.log(max_lines=120).classes("h-32 w-full")

    def log_msg(s: str):
        log.push(f"[{time.strftime('%H:%M:%S')}] {s}")

    async def in_thread(fn, *a, **kw):
        return await asyncio.to_thread(fn, *a, **kw)

    def ensure_ctrl() -> Optional[B2987BController]:
        c = get_controller()
        if c is None:
            log_msg("not connected — connect on the parent's Connections tab"
                    if not show_connection else
                    "not connected — use the Connection panel")
            return None
        return c

    # --- Tabs --------------------------------------------------------------

    with ui.tabs().classes("w-full") as tabs:
        t_conn   = ui.tab("connection") if show_connection else None
        t_bias   = ui.tab("bias")
        t_single = ui.tab("single measure")
        t_sweep  = ui.tab("iv sweep")

    initial = t_conn if t_conn is not None else t_bias
    with ui.tab_panels(tabs, value=initial).classes("w-full"):

        # ----------- Connection tab (standalone only) ------------
        if t_conn is not None:
            with ui.tab_panel(t_conn):
                with ui.card().classes("b2987-card"):
                    ui.html("<h2>instrument connection</h2>")
                    visa_in = ui.input(label="VISA resource",
                                       value=DEFAULT_VISA).classes("w-96 num")
                    mode_in = ui.select(["simulation", "hardware"],
                                        value="simulation",
                                        label="mode").classes("w-40")
                    conn_pill = ui.html('<span class="pill mut">disconnected</span>')

                    def set_pill(text: str, cls: str):
                        conn_pill.content = f'<span class="pill {cls}">{text}</span>'

                    async def do_connect():
                        c = B2987BController(visa=visa_in.value.strip(),
                                             mode=mode_in.value)
                        set_pill("connecting…", "warn")
                        log_msg(f"connecting to {visa_in.value.strip()} ({mode_in.value})…")
                        try:
                            await in_thread(c.connect)
                            _own["ctrl"] = c
                            set_pill(f"OK — {c.identify()[:60]}", "ok")
                            log_msg(f"connected: {c.identify()}")
                        except Exception as e:
                            set_pill(f"FAIL: {type(e).__name__}", "bad")
                            log_msg(f"connect FAIL: {type(e).__name__}: {e}")

                    async def do_disconnect():
                        c = _own["ctrl"]
                        if c is None:
                            return
                        try:
                            await in_thread(c.disconnect)
                        except Exception as e:
                            log_msg(f"disconnect warn: {e}")
                        _own["ctrl"] = None
                        set_pill("disconnected", "mut")
                        log_msg("disconnected")

                    async def do_test():
                        cfg = {"visa": visa_in.value.strip(), "mode": mode_in.value}
                        try:
                            ok, msg = await in_thread(B2987BController.test, cfg)
                            log_msg(f"test {'OK' if ok else 'FAIL'}: {msg}")
                        except Exception as e:
                            log_msg(f"test FAIL: {type(e).__name__}: {e}")

                    with ui.row().classes("mt-1 gap-2"):
                        ui.button("connect",    on_click=do_connect).props("color=primary")
                        ui.button("disconnect", on_click=do_disconnect).props("color=negative flat")
                        ui.button("test",       on_click=do_test)

        # ----------- Bias tab ------------
        with ui.tab_panel(t_bias):
            with ui.card().classes("b2987-card"):
                ui.html("<h2>bias control (no measurement)</h2>")
                bias_v_in    = ui.number(label="voltage (V)", value=0.0, step=0.5).classes("w-40 num")
                bias_range   = ui.select(["20", "1000"], value="1000",
                                         label="source range (V)").classes("w-40")
                bias_rlim    = ui.switch("current-limiting resistor", value=False)
                bias_pill    = ui.html('<span class="pill mut">output: off</span>')

                async def bias_on():
                    c = ensure_ctrl()
                    if c is None:
                        return
                    try:
                        await in_thread(c.configure_sweep,
                                        source_range=float(bias_range.value),
                                        current_limit=bias_rlim.value)
                        await in_thread(c.set_bias, float(bias_v_in.value), 0.0)
                        bias_pill.content = (f'<span class="pill ok">'
                                             f'output: on  {bias_v_in.value:.3f} V</span>')
                        log_msg(f"bias on  {bias_v_in.value:.3f} V")
                    except Exception as e:
                        log_msg(f"bias on FAIL: {type(e).__name__}: {e}")

                async def bias_off():
                    c = ensure_ctrl()
                    if c is None:
                        return
                    try:
                        await in_thread(c.bias_off)
                        bias_pill.content = '<span class="pill mut">output: off</span>'
                        log_msg("bias off")
                    except Exception as e:
                        log_msg(f"bias off FAIL: {type(e).__name__}: {e}")

                with ui.row().classes("mt-1 gap-2"):
                    ui.button("output ON",  on_click=bias_on).props("color=primary")
                    ui.button("output OFF", on_click=bias_off).props("color=negative")

        # ----------- Single measure tab ------------
        with ui.tab_panel(t_single):
            with ui.card().classes("b2987-card"):
                ui.html("<h2>single current measurement</h2>")
                single_v_in = ui.number(label="voltage (V)", value=0.0, step=1.0).classes("w-40 num")
                single_result = ui.label("—").classes("num text-2xl")

                async def single_measure():
                    c = ensure_ctrl()
                    if c is None:
                        return
                    single_result.text = "measuring…"
                    try:
                        i = await in_thread(c.measure_current, float(single_v_in.value))
                        single_result.text = f"{i:.4e} A"
                        log_msg(f"single  V={single_v_in.value:.3f}  I={i:.4e} A")
                    except Exception as e:
                        single_result.text = "—"
                        log_msg(f"single FAIL: {type(e).__name__}: {e}")

                ui.button("measure", on_click=single_measure).props("color=primary")

        # ----------- IV sweep tab ------------
        with ui.tab_panel(t_sweep):
            with ui.row().classes("w-full gap-3 items-start"):
                with ui.card().classes("b2987-card"):
                    ui.html("<h2>voltage range</h2>")
                    sw_start = ui.number(label="start (V)", value=40.0, step=0.1).classes("w-32 num")
                    sw_stop  = ui.number(label="stop (V)",  value=55.0, step=0.1).classes("w-32 num")
                    sw_step  = ui.number(label="step (V)",  value=0.05, step=0.01).classes("w-32 num")

                with ui.card().classes("b2987-card"):
                    ui.html("<h2>sweep parameters</h2>")
                    sw_npts  = ui.number(label="pts / V",     value=5,    step=1).classes("w-32 num")
                    sw_delay = ui.number(label="delay (s)",   value=0.1,  step=0.05).classes("w-32 num")
                    sw_measv = ui.switch("measure sense V", value=False)
                    sw_logy  = ui.switch("log Y", value=True)

                with ui.card().classes("b2987-card"):
                    ui.html("<h2>run</h2>")
                    sw_progress = ui.label("ready").classes("num text-sm")

                    async def run_sweep():
                        c = ensure_ctrl()
                        if c is None:
                            return
                        start, stop, step = float(sw_start.value), float(sw_stop.value), float(sw_step.value)
                        if step <= 0:
                            log_msg("step must be > 0"); return
                        voltages = np.arange(start, stop + step * 0.5, step).tolist()
                        sw_progress.text = f"running ({len(voltages)} pts)…"
                        log_msg(f"sweep {start} → {stop} V, step={step} V, {len(voltages)} pts")
                        try:
                            result: SweepResult = await in_thread(
                                c.sweep, voltages,
                                int(sw_npts.value), float(sw_delay.value),
                                bool(sw_measv.value),
                            )
                            _own["last_sweep"] = result
                            n = len(result.avg_source_v)
                            sw_progress.text = f"done — {n} pts, I_max={float(np.abs(result.avg_current_a).max()):.3e} A"
                            log_msg(sw_progress.text)
                            refresh_plot()
                        except Exception as e:
                            sw_progress.text = "FAIL"
                            log_msg(f"sweep FAIL: {type(e).__name__}: {e}")

                    ui.button("run sweep", on_click=run_sweep).props("color=primary")

            # Plot below the controls
            plot = ui.matplotlib(figsize=(8, 3.2)).classes("w-full")
            ax   = plot.figure.add_subplot(111)
            ax.set_xlabel("bias (V)"); ax.set_ylabel("|current| (A)")
            ax.set_title("IV curve"); ax.grid(True, alpha=.3)
            plot.figure.tight_layout()

            def refresh_plot():
                r = _own["last_sweep"]
                if r is None:
                    return
                ax.clear()
                v = np.asarray(r.avg_source_v)
                i = np.abs(np.asarray(r.avg_current_a))
                err = np.asarray(r.err_current_a) if r.err_current_a is not None else None
                ax.errorbar(v, i, yerr=err, fmt="o-", ms=3, lw=1, capsize=2,
                            color="#58a6ff", ecolor="#8a93a6")
                ax.set_yscale("log" if sw_logy.value else "linear")
                ax.set_xlabel("bias (V)"); ax.set_ylabel("|current| (A)")
                ax.set_title("IV curve"); ax.grid(True, which="both", alpha=.3)
                plot.figure.tight_layout()
                plot.update()


# ---------------------------------------------------------------------------
# Standalone entry — `python -m b2987b.gui`
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8766):
    """Launch the standalone web GUI."""
    import argparse
    p = argparse.ArgumentParser(description="Keysight B2987B web GUI")
    p.add_argument("--host", default=host)
    p.add_argument("--port", type=int, default=port)
    args = p.parse_args()

    @ui.page("/")
    def index():
        ui.add_head_html(f"<style>{_CSS}</style>")
        ui.dark_mode().enable()
        with ui.element("header").style(
            "display:flex;align-items:center;gap:.8rem;"
            "padding:.55rem 1rem;background:var(--panel);"
            "border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5"
        ):
            ui.html("<h1 style='font-size:1.05rem;font-weight:600;margin:0'>"
                    "B2987B · electrometer</h1>")
        build_page()

    ui.run(host=args.host, port=args.port, reload=False,
           title="B2987B Electrometer", show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
