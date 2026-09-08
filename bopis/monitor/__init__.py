"""Performance-measurement layer: GPU energy/utilization and CPU/RAM telemetry.

Standard library only. See :mod:`bopis.monitor.nvml` for the NVML ctypes
interface and the energy-method ladder, and :mod:`bopis.monitor.platform_os` for
the ``/proc`` (Linux/WSL2) and ``kernel32`` (Windows) CPU/RAM paths.

The 100 ms sampler and energy integration live in :mod:`bopis.monitor.sampler`.
"""

from __future__ import annotations

from bopis.monitor import nvml, platform_os
from bopis.monitor.nvml import EnergyMethod

__all__ = ["nvml", "platform_os", "EnergyMethod"]
