"""Direct NVML access through :mod:`ctypes` -- standard library only.

Chapter 3 specifies that GPU power, utilization and memory are read by loading
the NVIDIA Management Library directly and calling ``nvmlDeviceGetPowerUsage``,
``nvmlDeviceGetUtilizationRates`` and ``nvmlDeviceGetMemoryInfo``, bypassing any
third-party Python wrapper (no ``pynvml``). This module is that interface.

Energy-measurement ladder (amendment A-8 / A-18)
------------------------------------------------
Not every GPU supports power telemetry. Measured on the development machine, an
NVIDIA MX330 returns ``NVML_ERROR_NOT_SUPPORTED`` (rc=3) for *both*
``nvmlDeviceGetPowerUsage`` and ``nvmlDeviceGetTotalEnergyConsumption``, while
utilization and memory queries succeed. A study that silently recorded zeros
there would be worse than one that records which instrument it actually used, so
every measurement carries an :class:`EnergyMethod` tag:

``NVML_ENERGY_COUNTER``
    ``nvmlDeviceGetTotalEnergyConsumption`` -- a monotonic millijoule counter
    maintained by the driver (Volta and newer). Preferred: it is exact, needs no
    idle-power subtraction, and is immune to polling-interval integration error.
``NVML_POWER_INTEGRATION``
    Chapter 3's documented method: integrate ``(P_t - P_idle)`` over the
    inference window. Used when the energy counter is unavailable.
``UNAVAILABLE``
    Neither is supported. Energy is recorded as ``None``; the caller must pass
    ``--allow-no-power`` to proceed, and no energy claim may be published.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13

_RC_NAMES = {
    NVML_SUCCESS: "SUCCESS",
    NVML_ERROR_UNINITIALIZED: "UNINITIALIZED",
    NVML_ERROR_INVALID_ARGUMENT: "INVALID_ARGUMENT",
    NVML_ERROR_NOT_SUPPORTED: "NOT_SUPPORTED",
    NVML_ERROR_NO_PERMISSION: "NO_PERMISSION",
    NVML_ERROR_NOT_FOUND: "NOT_FOUND",
    NVML_ERROR_DRIVER_NOT_LOADED: "DRIVER_NOT_LOADED",
    NVML_ERROR_LIBRARY_NOT_FOUND: "LIBRARY_NOT_FOUND",
    NVML_ERROR_FUNCTION_NOT_FOUND: "FUNCTION_NOT_FOUND",
}


def rc_name(rc: int) -> str:
    return _RC_NAMES.get(rc, f"RC_{rc}")


class EnergyMethod:
    """Provenance tag recorded with every energy measurement."""

    NVML_ENERGY_COUNTER = "nvml_energy_counter"
    NVML_POWER_INTEGRATION = "nvml_power_integration"
    UNAVAILABLE = "unavailable"


#: Candidate NVML library names, in load order, per platform.
_LIB_CANDIDATES = {
    "Windows": ("nvml.dll", r"C:\Windows\System32\nvml.dll", "nvidia-ml.dll"),
    "Linux": (
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        # WSL2 exposes the host driver's NVML here and it is not always on the
        # default loader path.
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    ),
    "Darwin": (),  # NVML is not available on macOS.
}


# --------------------------------------------------------------------------- #
# ctypes structures
# --------------------------------------------------------------------------- #


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlUnavailable(RuntimeError):
    """NVML could not be loaded or initialized."""


# --------------------------------------------------------------------------- #
# Device handle
# --------------------------------------------------------------------------- #


class GpuDevice:
    """A single NVML device, with per-capability support flags.

    Capability probing happens once at construction: each optional query is
    attempted and its return code recorded, so callers never have to guess
    whether a metric is available and the manifest can state it plainly.
    """

    def __init__(self, nvml: "Nvml", index: int) -> None:
        self._nvml = nvml
        self.index = index
        self._handle = ctypes.c_void_p()
        rc = nvml.lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(index), ctypes.byref(self._handle)
        )
        if rc != NVML_SUCCESS:
            raise NvmlUnavailable(
                f"nvmlDeviceGetHandleByIndex_v2({index}) -> {rc_name(rc)}"
            )

        self.name = self._read_name()
        self.compute_capability = self._read_compute_capability()

        total, _free, _used = self._raw_memory()
        self.vram_total_bytes = total or 0

        # Probe optional telemetry.
        self.power_rc = self._probe_power()
        self.energy_counter_rc = self._probe_energy_counter()
        self.power_supported = self.power_rc == NVML_SUCCESS
        self.energy_counter_supported = self.energy_counter_rc == NVML_SUCCESS

    # -- capability probes --------------------------------------------------- #

    def _probe_power(self) -> int:
        mw = ctypes.c_uint()
        return self._nvml.lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(mw))

    def _probe_energy_counter(self) -> int:
        fn = getattr(self._nvml.lib, "nvmlDeviceGetTotalEnergyConsumption", None)
        if fn is None:
            return NVML_ERROR_FUNCTION_NOT_FOUND
        mj = ctypes.c_ulonglong()
        return fn(self._handle, ctypes.byref(mj))

    @property
    def energy_method(self) -> str:
        """The best energy instrument this device supports."""
        if self.energy_counter_supported:
            return EnergyMethod.NVML_ENERGY_COUNTER
        if self.power_supported:
            return EnergyMethod.NVML_POWER_INTEGRATION
        return EnergyMethod.UNAVAILABLE

    # -- metadata ------------------------------------------------------------ #

    def _read_name(self) -> str:
        buf = ctypes.create_string_buffer(96)
        rc = self._nvml.lib.nvmlDeviceGetName(self._handle, buf, ctypes.c_uint(96))
        if rc != NVML_SUCCESS:
            return "unknown"
        return buf.value.decode("utf-8", "replace")

    def _read_compute_capability(self) -> Optional[str]:
        fn = getattr(self._nvml.lib, "nvmlDeviceGetCudaComputeCapability", None)
        if fn is None:
            return None
        major, minor = ctypes.c_int(), ctypes.c_int()
        rc = fn(self._handle, ctypes.byref(major), ctypes.byref(minor))
        if rc != NVML_SUCCESS:
            return None
        return f"{major.value}.{minor.value}"

    # -- live metrics -------------------------------------------------------- #

    def _raw_memory(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        mem = _Memory()
        rc = self._nvml.lib.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(mem))
        if rc != NVML_SUCCESS:
            return None, None, None
        return int(mem.total), int(mem.free), int(mem.used)

    def memory_used_bytes(self) -> Optional[int]:
        return self._raw_memory()[2]

    def memory_free_bytes(self) -> Optional[int]:
        return self._raw_memory()[1]

    def utilization(self) -> Tuple[Optional[int], Optional[int]]:
        """``(gpu_percent, memory_bus_percent)``."""
        util = _Utilization()
        rc = self._nvml.lib.nvmlDeviceGetUtilizationRates(
            self._handle, ctypes.byref(util)
        )
        if rc != NVML_SUCCESS:
            return None, None
        return int(util.gpu), int(util.memory)

    def power_milliwatts(self) -> Optional[int]:
        """Instantaneous board power draw in mW, or ``None`` if unsupported."""
        if not self.power_supported:
            return None
        mw = ctypes.c_uint()
        rc = self._nvml.lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(mw))
        if rc != NVML_SUCCESS:
            return None
        return int(mw.value)

    def total_energy_millijoules(self) -> Optional[int]:
        """Monotonic driver-side energy counter in mJ, or ``None``.

        Reset only on driver reload, so callers take a difference across the
        inference window.
        """
        if not self.energy_counter_supported:
            return None
        fn = self._nvml.lib.nvmlDeviceGetTotalEnergyConsumption
        mj = ctypes.c_ulonglong()
        rc = fn(self._handle, ctypes.byref(mj))
        if rc != NVML_SUCCESS:
            return None
        return int(mj.value)

    # -- reporting ----------------------------------------------------------- #

    def describe(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_total_mib": self.vram_total_bytes // (1024 * 1024),
            "compute_capability": self.compute_capability,
            "power_supported": self.power_supported,
            "power_rc": rc_name(self.power_rc),
            "energy_counter_supported": self.energy_counter_supported,
            "energy_counter_rc": rc_name(self.energy_counter_rc),
            "energy_method": self.energy_method,
        }


# --------------------------------------------------------------------------- #
# Library handle
# --------------------------------------------------------------------------- #


class Nvml:
    """Loaded, initialized NVML library. Use as a context manager."""

    def __init__(self, lib: ctypes.CDLL, lib_path: str) -> None:
        self.lib = lib
        self.lib_path = lib_path
        self._initialized = False

    # -- lifecycle ----------------------------------------------------------- #

    @staticmethod
    def _load() -> Tuple[ctypes.CDLL, str]:
        system = platform.system()
        errors: List[str] = []
        for candidate in _LIB_CANDIDATES.get(system, ()):
            try:
                return ctypes.CDLL(candidate), candidate
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        raise NvmlUnavailable(
            "could not load NVML ("
            + (f"platform {system} unsupported" if not errors else "; ".join(errors))
            + ")"
        )

    @classmethod
    def open(cls) -> "Nvml":
        lib, path = cls._load()
        cls._declare_signatures(lib)
        inst = cls(lib, path)
        init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
        rc = init()
        if rc != NVML_SUCCESS:
            raise NvmlUnavailable(f"nvmlInit -> {rc_name(rc)}")
        inst._initialized = True
        return inst

    @staticmethod
    def _declare_signatures(lib: ctypes.CDLL) -> None:
        """Pin argtypes/restype so pointers are not truncated on 64-bit."""
        void_p = ctypes.c_void_p
        specs = {
            "nvmlInit_v2": ([], ctypes.c_int),
            "nvmlInit": ([], ctypes.c_int),
            "nvmlShutdown": ([], ctypes.c_int),
            "nvmlDeviceGetCount_v2": ([ctypes.POINTER(ctypes.c_uint)], ctypes.c_int),
            "nvmlDeviceGetHandleByIndex_v2": (
                [ctypes.c_uint, ctypes.POINTER(void_p)],
                ctypes.c_int,
            ),
            "nvmlDeviceGetName": (
                [void_p, ctypes.c_char_p, ctypes.c_uint],
                ctypes.c_int,
            ),
            "nvmlDeviceGetMemoryInfo": (
                [void_p, ctypes.POINTER(_Memory)],
                ctypes.c_int,
            ),
            "nvmlDeviceGetUtilizationRates": (
                [void_p, ctypes.POINTER(_Utilization)],
                ctypes.c_int,
            ),
            "nvmlDeviceGetPowerUsage": (
                [void_p, ctypes.POINTER(ctypes.c_uint)],
                ctypes.c_int,
            ),
            "nvmlDeviceGetTotalEnergyConsumption": (
                [void_p, ctypes.POINTER(ctypes.c_ulonglong)],
                ctypes.c_int,
            ),
            "nvmlDeviceGetCudaComputeCapability": (
                [void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
                ctypes.c_int,
            ),
            "nvmlSystemGetDriverVersion": (
                [ctypes.c_char_p, ctypes.c_uint],
                ctypes.c_int,
            ),
            "nvmlSystemGetCudaDriverVersion_v2": (
                [ctypes.POINTER(ctypes.c_int)],
                ctypes.c_int,
            ),
        }
        for name, (argtypes, restype) in specs.items():
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.argtypes = argtypes
                fn.restype = restype

    def close(self) -> None:
        if self._initialized:
            try:
                self.lib.nvmlShutdown()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
            self._initialized = False

    def __enter__(self) -> "Nvml":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- queries ------------------------------------------------------------- #

    def device_count(self) -> int:
        n = ctypes.c_uint()
        rc = self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
        return int(n.value) if rc == NVML_SUCCESS else 0

    def device(self, index: int = 0) -> GpuDevice:
        return GpuDevice(self, index)

    def driver_version(self) -> Optional[str]:
        buf = ctypes.create_string_buffer(80)
        rc = self.lib.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(80))
        if rc != NVML_SUCCESS:
            return None
        return buf.value.decode("utf-8", "replace")

    def cuda_driver_version(self) -> Optional[str]:
        fn = getattr(self.lib, "nvmlSystemGetCudaDriverVersion_v2", None)
        if fn is None:
            return None
        v = ctypes.c_int()
        if fn(ctypes.byref(v)) != NVML_SUCCESS:
            return None
        # NVML encodes CUDA version as 1000*major + 10*minor.
        return f"{v.value // 1000}.{(v.value % 1000) // 10}"


def probe() -> dict:
    """Best-effort GPU description that never raises.

    Returns a mapping always containing ``available``; when that is ``False`` the
    ``error`` key explains why. Used by ``bopis profile`` and by the run manifest
    so the recorded provenance is honest on machines without a usable GPU.
    """
    try:
        with Nvml.open() as nvml:
            count = nvml.device_count()
            if count == 0:
                return {"available": False, "error": "no NVML devices reported"}
            dev = nvml.device(0)
            info = dev.describe()
            info.update(
                {
                    "available": True,
                    "device_count": count,
                    "nvml_library": nvml.lib_path,
                    "driver_version": nvml.driver_version(),
                    "cuda_driver_version": nvml.cuda_driver_version(),
                }
            )
            return info
    except NvmlUnavailable as exc:
        return {"available": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import json

    json.dump(probe(), sys.stdout, indent=2)
    sys.stdout.write(os.linesep)
