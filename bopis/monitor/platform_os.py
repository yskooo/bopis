"""Cross-platform CPU / RAM telemetry using only the standard library.

Chapter 3 reads CPU utilization, memory and thread counts from the Linux
``/proc`` filesystem (``/proc/stat``, ``/proc/meminfo``, ``/proc/cpuinfo``,
``/proc/[pid]/status``). ``/proc`` does not exist on Windows, and the tool is
hardware-agnostic by design, so this module presents one interface over two
implementations (amendment A-9):

* **Linux / WSL2** -- parse ``/proc`` exactly as the manuscript specifies.
* **Windows** -- ``ctypes`` calls into ``kernel32``: ``GetSystemTimes``,
  ``GetProcessTimes``, ``GlobalMemoryStatusEx``,
  ``GetLogicalProcessorInformation``, ``CreateToolhelp32Snapshot`` and
  ``K32GetProcessMemoryInfo``.

``psutil`` is deliberately not used: it is a third-party package and the
measurement core must stay standard-library-only.

CPU utilization is inherently a *rate*, so it is measured as a delta between two
:class:`CpuTimes` snapshots rather than as an instantaneous value.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
from typing import Dict, NamedTuple, Optional, Tuple

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


class CpuTimes(NamedTuple):
    """Cumulative CPU time counters, in seconds, aggregated over all cores."""

    idle: float
    busy: float

    @property
    def total(self) -> float:
        return self.idle + self.busy


class ProcessCpuTimes(NamedTuple):
    """Cumulative CPU seconds charged to one process, summed over its threads.

    Needed because system-wide utilization cannot answer "how much of this
    machine did the inference process use". On a laptop that is also running a
    browser, the system-wide figure attributes their load to the workload under
    test. Both backends here aggregate over every thread of the process, which
    is what a multi-threaded ``llama-server`` requires.
    """

    user: float
    kernel: float

    @property
    def total(self) -> float:
        return self.user + self.kernel


class MemoryInfo(NamedTuple):
    total_bytes: int
    available_bytes: int

    @property
    def used_bytes(self) -> int:
        return max(0, self.total_bytes - self.available_bytes)

    @property
    def used_mib(self) -> float:
        return self.used_bytes / (1024.0 * 1024.0)

    @property
    def percent_used(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return 100.0 * self.used_bytes / self.total_bytes


# --------------------------------------------------------------------------- #
# Linux / WSL2: /proc
# --------------------------------------------------------------------------- #


def _linux_cpu_times() -> CpuTimes:
    with open("/proc/stat", "r", encoding="utf-8") as fh:
        line = fh.readline()
    # "cpu  user nice system idle iowait irq softirq steal guest guest_nice"
    parts = [float(v) for v in line.split()[1:]]
    hz = os.sysconf("SC_CLK_TCK") or 100
    user, nice, system, idle = parts[0], parts[1], parts[2], parts[3]
    iowait = parts[4] if len(parts) > 4 else 0.0
    rest = sum(parts[5:8]) if len(parts) > 5 else 0.0
    # iowait counts as idle: the CPU is not executing instructions.
    idle_total = (idle + iowait) / hz
    busy_total = (user + nice + system + rest) / hz
    return CpuTimes(idle=idle_total, busy=busy_total)


def _linux_memory() -> MemoryInfo:
    fields: Dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            tokens = rest.split()
            if tokens:
                fields[key] = int(tokens[0]) * 1024  # kB -> bytes
    total = fields.get("MemTotal", 0)
    # MemAvailable is the kernel's own estimate and is preferred over free.
    available = fields.get("MemAvailable", fields.get("MemFree", 0))
    return MemoryInfo(total_bytes=total, available_bytes=available)


def _linux_cpu_model_and_cores() -> Tuple[str, int, int]:
    model = "unknown"
    logical = 0
    core_ids = set()
    physical_id = None
    core_id = None
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "model name":
                    model = value
                elif key == "processor":
                    logical += 1
                elif key == "physical id":
                    physical_id = value
                elif key == "core id":
                    core_id = value
                    core_ids.add((physical_id, core_id))
    except OSError:
        pass
    physical = len(core_ids) if core_ids else (logical or os.cpu_count() or 1)
    return model, physical, logical or (os.cpu_count() or 1)


def _linux_process_memory(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _linux_process_cpu_times(pid: int) -> Optional[ProcessCpuTimes]:
    """``utime``/``stime`` from ``/proc/[pid]/stat``, already thread-aggregated.

    The comm field is parenthesised and may itself contain spaces and
    parentheses, so the split starts after the *last* ``)`` rather than at a
    fixed token index.
    """
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 1 :].split()
    # fields[0] is `state` (stat field 3), so utime (14) and stime (15) sit at
    # offsets 11 and 12.
    if len(fields) < 13:
        return None
    hz = os.sysconf("SC_CLK_TCK") or 100
    try:
        return ProcessCpuTimes(
            user=float(fields[11]) / hz, kernel=float(fields[12]) / hz
        )
    except ValueError:
        return None


def _linux_process_threads(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- #
# Windows: ctypes -> kernel32
# --------------------------------------------------------------------------- #

if IS_WINDOWS:
    import ctypes.wintypes as wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def _filetime_to_100ns(ft: "wintypes.FILETIME") -> int:
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    def _windows_cpu_times() -> CpuTimes:
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not _k32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise OSError(ctypes.get_last_error(), "GetSystemTimes failed")
        # Units are 100 ns. Note: kernel time *includes* idle time on Windows.
        idle_s = _filetime_to_100ns(idle) / 1e7
        kernel_s = _filetime_to_100ns(kernel) / 1e7
        user_s = _filetime_to_100ns(user) / 1e7
        busy_s = (kernel_s - idle_s) + user_s
        return CpuTimes(idle=idle_s, busy=busy_s)

    def _windows_memory() -> MemoryInfo:
        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not _k32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
        return MemoryInfo(
            total_bytes=int(stat.ullTotalPhys),
            available_bytes=int(stat.ullAvailPhys),
        )

    _RELATION_PROCESSOR_CORE = 0

    def _windows_physical_cores() -> int:
        """Count ``RelationProcessorCore`` entries from GetLogicalProcessorInformation."""
        length = ctypes.c_ulong(0)
        _k32.GetLogicalProcessorInformation(None, ctypes.byref(length))
        buf = ctypes.create_string_buffer(length.value)
        if not _k32.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
            return os.cpu_count() or 1
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION on x64:
        #   ULONG_PTR ProcessorMask (8) + DWORD Relationship (4) + pad (4)
        #   + union (16) = 32 bytes
        stride = 32 if ctypes.sizeof(ctypes.c_void_p) == 8 else 24
        count = 0
        for offset in range(0, length.value, stride):
            relationship = int.from_bytes(
                buf.raw[offset + ctypes.sizeof(ctypes.c_void_p) : offset
                        + ctypes.sizeof(ctypes.c_void_p) + 4],
                "little",
            )
            if relationship == _RELATION_PROCESSOR_CORE:
                count += 1
        return count or (os.cpu_count() or 1)

    def _windows_cpu_model() -> str:
        # The registry is the only stdlib-reachable source for the brand string.
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            return platform.processor() or "unknown"

    def _windows_process_cpu_times(pid: int) -> Optional[ProcessCpuTimes]:
        """``GetProcessTimes``, which sums the kernel and user time of every
        thread the process has ever run.

        ``PROCESS_QUERY_LIMITED_INFORMATION`` is enough here and, unlike
        ``PROCESS_QUERY_INFORMATION``, is granted for a child process without
        elevation -- which matters because the measured process is the
        ``llama-server`` this tool spawned.
        """
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = _k32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return None
        try:
            creation = wintypes.FILETIME()
            exited = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not _k32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            # Units are 100 ns intervals.
            return ProcessCpuTimes(
                user=_filetime_to_100ns(user) / 1e7,
                kernel=_filetime_to_100ns(kernel) / 1e7,
            )
        finally:
            _k32.CloseHandle(handle)

    def _windows_process_threads(pid: int) -> Optional[int]:
        """Thread count of *pid* via a ``Thread32`` snapshot walk.

        ``/proc/[pid]/status`` has no Windows equivalent, so the thread count
        comes from ``CreateToolhelp32Snapshot``. Without this the manifest
        would leave ``n_threads`` empty on Windows, and the ``c`` (CPU-threads)
        dimension of the search space would have no observed counterpart to
        check the requested value against.
        """
        TH32CS_SNAPTHREAD = 0x00000004

        class _ThreadEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_ulong),
                ("cntUsage", ctypes.c_ulong),
                ("th32ThreadID", ctypes.c_ulong),
                ("th32OwnerProcessID", ctypes.c_ulong),
                ("tpBasePri", ctypes.c_long),
                ("tpDeltaPri", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong),
            ]

        # OpenProcess and friends are declared with the default ``c_int``
        # restype in this module, so a failed call arrives as -1 rather than as
        # the pointer-width INVALID_HANDLE_VALUE. Both are rejected.
        snapshot = _k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if not snapshot or snapshot == -1:
            return None
        try:
            entry = _ThreadEntry32()
            entry.dwSize = ctypes.sizeof(_ThreadEntry32)
            if not _k32.Thread32First(snapshot, ctypes.byref(entry)):
                return None
            count = 0
            while True:
                if entry.th32OwnerProcessID == pid:
                    count += 1
                if not _k32.Thread32Next(snapshot, ctypes.byref(entry)):
                    break
            # A live process always owns at least one thread, so a count of
            # zero means the PID was not found -- report that as "not
            # observable" rather than as a process with no threads.
            return count or None
        finally:
            _k32.CloseHandle(snapshot)

    def _windows_process_memory(pid: int) -> Optional[int]:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        PROCESS_VM_READ = 0x0010
        handle = _k32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid
        )
        if not handle:
            return None
        try:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            fn = getattr(_k32, "K32GetProcessMemoryInfo", None)
            if fn is None:
                return None
            if not fn(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        finally:
            _k32.CloseHandle(handle)


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #


def cpu_times() -> CpuTimes:
    """Cumulative idle/busy CPU seconds. Take two and difference them."""
    if IS_LINUX:
        return _linux_cpu_times()
    if IS_WINDOWS:
        return _windows_cpu_times()
    raise NotImplementedError(f"CPU telemetry unsupported on {platform.system()}")


def cpu_percent_between(before: CpuTimes, after: CpuTimes) -> float:
    """System-wide CPU utilization (%) over the interval between two snapshots."""
    delta_total = after.total - before.total
    if delta_total <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (after.busy - before.busy) / delta_total))


def process_cpu_times(pid: int) -> Optional[ProcessCpuTimes]:
    """Cumulative CPU seconds charged to *pid*, or ``None`` if not observable.

    ``None`` is a real outcome, not just an error path: the process may have
    exited between the two snapshots that bracket a window, and the caller must
    degrade to the system-wide signal rather than treat zero as the answer.
    """
    if IS_LINUX:
        return _linux_process_cpu_times(pid)
    if IS_WINDOWS:
        return _windows_process_cpu_times(pid)
    return None


def process_cpu_fraction(
    before: ProcessCpuTimes,
    after: ProcessCpuTimes,
    wall_s: float,
    logical_cores: Optional[int] = None,
) -> Optional[float]:
    """Share of total machine CPU capacity used by one process, in ``[0, 1]``.

    The denominator is ``logical_cores * wall_s`` -- the CPU-seconds the whole
    machine could have delivered over the window -- so a process pinning four
    of eight logical processors returns ``0.5``. That is the quantity the energy
    estimator multiplies by the CPU package's dynamic power range.

    Normalising by *logical* rather than physical cores assumes package power
    scales with logical saturation. It does not do so exactly, since the two
    threads sharing a core do not double its power, so a hyperthreaded host
    running few threads is charged somewhat less than its true share. The
    alternative -- normalising by physical cores -- lets a fully loaded
    hyperthreaded machine exceed 1.0, which is worse.

    Returns ``None`` when the counters moved backwards, which means the PID was
    reused or the process restarted mid-window; that difference is meaningless
    and must not be silently reported as low utilization.
    """
    cores = logical_cores or os.cpu_count() or 1
    if wall_s <= 0 or cores <= 0:
        return None
    busy = after.total - before.total
    if busy < 0:
        return None
    return max(0.0, min(1.0, busy / (wall_s * cores)))


def memory_info() -> MemoryInfo:
    if IS_LINUX:
        return _linux_memory()
    if IS_WINDOWS:
        return _windows_memory()
    raise NotImplementedError(f"memory telemetry unsupported on {platform.system()}")


def cpu_info() -> Dict[str, object]:
    """CPU model plus physical and logical core counts.

    The Table H1 thread rules key on *physical* cores, matching llama.cpp's own
    guidance that ``--threads`` should not exceed the physical core count. Both
    figures are recorded in the manifest so the choice is auditable.
    """
    if IS_LINUX:
        model, physical, logical = _linux_cpu_model_and_cores()
    elif IS_WINDOWS:
        model = _windows_cpu_model()
        physical = _windows_physical_cores()
        logical = os.cpu_count() or physical
    else:
        model = platform.processor() or "unknown"
        logical = os.cpu_count() or 1
        physical = logical
    return {
        "model": re.sub(r"\s+", " ", model).strip(),
        "physical_cores": physical,
        "logical_cores": logical,
    }


def process_memory_bytes(pid: int) -> Optional[int]:
    """Resident set size of *pid*, or ``None`` if not observable."""
    if IS_LINUX:
        return _linux_process_memory(pid)
    if IS_WINDOWS:
        return _windows_process_memory(pid)
    return None


def process_threads(pid: int) -> Optional[int]:
    """Thread count of *pid*, or ``None`` if not observable."""
    if IS_LINUX:
        return _linux_process_threads(pid)
    if IS_WINDOWS:
        try:
            return _windows_process_threads(pid)
        except Exception:  # pragma: no cover - telemetry is best-effort
            return None
    return None


def describe() -> Dict[str, object]:
    """Platform summary for the run manifest."""
    mem = memory_info()
    info = dict(cpu_info())
    info.update(
        {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "is_wsl": _detect_wsl(),
            "ram_total_bytes": mem.total_bytes,
            "ram_total_gib": round(mem.total_bytes / (1024**3), 2),
            "ram_available_bytes": mem.available_bytes,
            "telemetry_source": "/proc" if IS_LINUX else ("kernel32" if IS_WINDOWS else "none"),
        }
    )
    return info


def _detect_wsl() -> bool:
    """True when running inside WSL.

    Matters for Table H1: WSL reports the memory ceiling from ``.wslconfig``, not
    the host's physical RAM, and the batch-size rule must key on the memory
    actually visible to the inference process (amendment A-14).
    """
    if not IS_LINUX:
        return False
    if "microsoft" in platform.release().lower():
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False
