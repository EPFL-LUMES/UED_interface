"""
Launch UED_GUI_v14.py while replacing the Glaz DLL with a NI-USB adapter.

This file does not modify UED_GUI_v14.py. It monkey-patches ctypes.cdll.LoadLibrary
so that only GlazLib.dll is replaced by a Python object exposing compatible methods:
- initialiseSession
- setScanCount
- startMeasurement
- getPDValues
- getLastErrorMessage

Environment variables (optional):
- UED_GUI_PATH: absolute path to UED_GUI_v14.py
- NIUSB_DEVICE: default "Dev1"
- NIUSB_AI_CHAN: default "ai0"
- NIUSB_TERMINAL: DIFF|RSE|NRSE|PSEUDODIFFERENTIAL (default DIFF)
- NIUSB_SAMPLE_RATE: default "10000"
- NIUSB_TRIGGER_SOURCE: default "/Dev1/PFI0" (set empty string to disable trigger)
- NIUSB_TRIGGER_EDGE: RISING|FALLING (default RISING)
- NIUSB_READ_TIMEOUT_S: default "10.0"
"""

from __future__ import annotations

import ctypes
import os
import runpy
from dataclasses import dataclass
from typing import Optional

import h5py
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration


class _CtypesCallable:
    """Callable wrapper that accepts ctypes-style .argtypes/.restype assignment."""

    def __init__(self, fn):
        self._fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


@dataclass
class _NIUSBConfig:
    device: str = "Dev1"
    ai_chan: str = "ai0"
    terminal: str = "DIFF"
    sample_rate: float = 10000.0
    trigger_source: str = "/Dev1/PFI0"
    trigger_edge: str = "RISING"
    read_timeout_s: float = 10.0

    @classmethod
    def from_env(cls) -> "_NIUSBConfig":
        return cls(
            device=os.getenv("NIUSB_DEVICE", "Dev1"),
            ai_chan=os.getenv("NIUSB_AI_CHAN", "ai0"),
            terminal=os.getenv("NIUSB_TERMINAL", "DIFF"),
            sample_rate=float(os.getenv("NIUSB_SAMPLE_RATE", "10000")),
            trigger_source=os.getenv("NIUSB_TRIGGER_SOURCE", "/Dev1/PFI0"),
            trigger_edge=os.getenv("NIUSB_TRIGGER_EDGE", "RISING"),
            read_timeout_s=float(os.getenv("NIUSB_READ_TIMEOUT_S", "10.0")),
        )


class NIUSBGlazCompat:
    """Drop-in replacement for the subset of GlazLib used by UED_GUI_v14.py."""

    def __init__(self, config: _NIUSBConfig):
        self._cfg = config
        self._scan_count = 1
        self._armed_task: Optional[nidaqmx.Task] = None
        self._last_error = b""

        self.initialiseSession = _CtypesCallable(self._initialise_session)
        self.setScanCount = _CtypesCallable(self._set_scan_count)
        self.startMeasurement = _CtypesCallable(self._start_measurement)
        self.getPDValues = _CtypesCallable(self._get_pd_values)
        self.getLastErrorMessage = _CtypesCallable(self._get_last_error_message)

    def _set_error(self, exc: Exception) -> int:
        self._last_error = str(exc).encode("utf-8", errors="ignore")
        print("[NIUSB->GlazCompat] Error:", str(exc))
        return -1

    def _terminal_cfg(self) -> TerminalConfiguration:
        m = self._cfg.terminal.strip().upper()
        if m == "DIFF":
            return TerminalConfiguration.DIFF
        if m == "RSE":
            return TerminalConfiguration.RSE
        if m == "NRSE":
            return TerminalConfiguration.NRSE
        if m in {"PSEUDODIFFERENTIAL", "PSEUDO_DIFFERENTIAL"}:
            return TerminalConfiguration.PSEUDODIFFERENTIAL
        return TerminalConfiguration.DEFAULT

    def _edge(self) -> Edge:
        return Edge.FALLING if self._cfg.trigger_edge.strip().upper() == "FALLING" else Edge.RISING

    def _build_task(self) -> nidaqmx.Task:
        task = nidaqmx.Task()
        physical = f"{self._cfg.device}/{self._cfg.ai_chan}"
        task.ai_channels.add_ai_voltage_chan(
            physical,
            terminal_config=self._terminal_cfg(),
        )
        task.timing.cfg_samp_clk_timing(
            rate=self._cfg.sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=self._scan_count,
        )
        if self._cfg.trigger_source.strip():
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=self._cfg.trigger_source,
                trigger_edge=self._edge(),
            )
        return task

    def _close_armed_task(self) -> None:
        if self._armed_task is not None:
            try:
                self._armed_task.close()
            finally:
                self._armed_task = None

    def _initialise_session(self, _script_path) -> int:
        self._last_error = b""
        return 0

    def _set_scan_count(self, scan_count) -> int:
        try:
            self._scan_count = max(1, int(getattr(scan_count, "value", scan_count)))
            self._last_error = b""
            return 0
        except Exception as exc:  # pragma: no cover - defensive for ctypes inputs
            return self._set_error(exc)

    def _start_measurement(self) -> int:
        try:
            self._close_armed_task()
            task = self._build_task()
            task.start()
            self._armed_task = task
            self._last_error = b""
            return 0
        except Exception as exc:
            self._close_armed_task()
            return self._set_error(exc)

    def _get_pd_values(self, _pd_number, _pd_channel, scan_cnt_out_ptr, data_ptr) -> int:
        try:
            task = self._armed_task
            owns_task = False
            if task is None:
                # Fallback path if getPDValues is called without startMeasurement.
                task = self._build_task()
                task.start()
                owns_task = True

            read_vals = task.read(
                number_of_samples_per_channel=self._scan_count,
                timeout=self._cfg.read_timeout_s,
            )

            if isinstance(read_vals, (float, int)):
                values = [float(read_vals)]
            else:
                values = [float(v) for v in read_vals]

            n = min(len(values), self._scan_count)

            out_ptr = ctypes.cast(scan_cnt_out_ptr, ctypes.POINTER(ctypes.c_int))
            out_ptr[0] = n

            for i in range(n):
                data_ptr[i] = values[i]

            if owns_task:
                task.close()
            else:
                self._close_armed_task()

            self._last_error = b""
            return 0
        except Exception as exc:
            self._close_armed_task()
            return self._set_error(exc)

    def _get_last_error_message(self, message_ptr) -> int:
        try:
            if hasattr(message_ptr, "value"):
                message_ptr.value = self._last_error
            return 0
        except Exception:
            return 0


_ORIGINAL_LOADLIBRARY = ctypes.cdll.LoadLibrary
_ORIGINAL_CREATE_DATASET = h5py.Group.create_dataset


def _patched_create_dataset(self, name, *args, **kwargs):
    if name == "data_diode":
        name = "data_NIUSB"
    return _ORIGINAL_CREATE_DATASET(self, name, *args, **kwargs)


def _patched_load_library(path):
    basename = os.path.basename(str(path)).lower()
    if basename == "glazlib.dll":
        cfg = _NIUSBConfig.from_env()
        print("[NIUSB->GlazCompat] Intercepted Glaz DLL load:", path)
        print(
            "[NIUSB->GlazCompat] Config:",
            f"device={cfg.device}, ai={cfg.ai_chan}, term={cfg.terminal}, "
            f"rate={cfg.sample_rate}, trig={cfg.trigger_source}, edge={cfg.trigger_edge}",
        )
        return NIUSBGlazCompat(cfg)
    return _ORIGINAL_LOADLIBRARY(path)


def _resolve_gui_path() -> str:
    env_path = os.getenv("UED_GUI_PATH", "").strip()
    if env_path:
        return env_path
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GUI", "UED_GUI_v14.py")
    )


def main() -> None:
    gui_path = _resolve_gui_path()
    if not os.path.isfile(gui_path):
        raise FileNotFoundError(f"Could not find GUI file: {gui_path}")

    gui_dir = os.path.dirname(os.path.abspath(gui_path))
    old_cwd = os.getcwd()
    os.chdir(gui_dir)

    ctypes.cdll.LoadLibrary = _patched_load_library
    h5py.Group.create_dataset = _patched_create_dataset
    try:
        runpy.run_path(gui_path, run_name="__main__")
    finally:
        ctypes.cdll.LoadLibrary = _ORIGINAL_LOADLIBRARY
        h5py.Group.create_dataset = _ORIGINAL_CREATE_DATASET
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
