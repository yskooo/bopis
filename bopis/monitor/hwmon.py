"""CPU package power from a running hardware monitor (RAPL, read second-hand).

Why this module exists
----------------------
The study GPU (GeForce MX330) exposes no power telemetry, so Chapter 3's GPU
energy objective cannot be measured on the study laptop. The *CPU* on that
laptop can be measured: Intel processors since Sandy Bridge keep a Running
Average Power Limit (RAPL) energy counter for the whole package
(``MSR_PKG_ENERGY_STATUS``). On an 11th-gen Tiger Lake part that counter is fed
by on-die power telemetry, not a nameplate model.

Reading an MSR on Windows needs a kernel driver, which the measurement core
cannot ship and still claim to be standard-library only. Open Hardware Monitor
(OHM) and its maintained fork LibreHardwareMonitor (LHM) already load such a
driver, convert the counter into package watts, and publish every sensor over
their built-in "Remote Web Server" as ``data.json``. This module reads that
endpoint with :mod:`urllib` -- no WMI, no ``pywin32``, no third-party imports.

So the instrument is RAPL; OHM/LHM is only the transport. When inference runs
CPU-only (``g = 0``), the package counter covers essentially all of the
inference compute, which is what makes this a *measured* energy objective on a
machine whose GPU cannot report one.

What the reading is, precisely
------------------------------
Both monitors publish ``P = (E_k - E_{k-1}) / (t_k - t_{k-1})`` -- the mean
package power over the **previous** refresh interval (1 s by default). Two
consequences, both handled in :mod:`bopis.monitor.sampler`:

1. **Lag.** A value observed at time ``t`` describes ``[t - Δ, t]``. Integrating
   the observed signal over the inference window ``[t0, t1]`` would measure
   ``[t0 - Δ, t1 - Δ]`` instead. The sampler keeps reading for one refresh
   interval after inference ends and integrates over ``[t0 + Δ, t1 + Δ]``.
2. **Resolution.** Energy inside one refresh bin is spread uniformly across it,
   so each window edge is uncertain by up to one bin. The sampler reports that
   bound as ``energy_low_j`` / ``energy_high_j``. Short windows (a 2 s chat
   reply) are therefore much less precise than the 5-30 s study windows;
   LibreHardwareMonitor allows a shorter refresh interval, which narrows it.

Scope
-----
Package power covers cores, uncore, and the integrated GPU. It excludes DRAM,
storage, display, and the discrete GPU. At ``g = 0`` the discrete GPU does no
inference work, so the scope matches the work performed; at ``g > 0`` it does
not, and :mod:`bopis.measure` flags such rows scope-invalid.

Standard library only.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

#: Where both OHM and LHM serve their sensor tree by default
#: (Options -> Remote Web Server -> Run).
DEFAULT_URL = "http://127.0.0.1:8085/data.json"

#: Refresh interval both monitors ship with. Measured at setup when possible.
DEFAULT_REFRESH_S = 1.0

#: Leaf names that denote whole-package power, in preference order. OHM names
#: the sensor "CPU Package"; some LHM versions and AMD parts use "Package".
PACKAGE_SENSOR_NAMES: Tuple[str, ...] = ("CPU Package", "Package")

_NUMBER = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*W\s*$")


class HwmonUnavailable(RuntimeError):
    """The hardware monitor could not be reached or exposes no package power."""


def setup_instructions(url: str = DEFAULT_URL) -> str:
    """What to tell the operator when the endpoint is not answering."""
    return (
        f"No hardware-monitor sensor feed at {url}.\n"
        "  1. Install LibreHardwareMonitor (maintained) or Open Hardware "
        "Monitor.\n"
        "  2. Run it AS ADMINISTRATOR -- RAPL is read through a kernel driver.\n"
        "  3. Options -> Remote Web Server -> Run (default port 8085).\n"
        "  4. Confirm the CPU node lists Powers -> CPU Package, then retry.\n"
        "  Check with:  python -m bopis profile --hwmon-url " + url
    )


def parse_watts(text: object) -> Optional[float]:
    """``"7.3 W"`` -> ``7.3``. Accepts a decimal comma; anything else -> None.

    Both monitors format values with the host's culture, so a Filipino or
    European locale can produce ``"7,3 W"``. Values in other units (``"%"``,
    ``"°C"``, ``"MHz"``) are not power and return None.
    """
    if not isinstance(text, str):
        return None
    match = _NUMBER.match(text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


class PowerSensor:
    """One power leaf in the sensor tree."""

    __slots__ = ("path", "watts", "sensor_id")

    def __init__(
        self, path: Tuple[str, ...], watts: float, sensor_id: Optional[str]
    ) -> None:
        self.path = path
        self.watts = watts
        self.sensor_id = sensor_id

    @property
    def label(self) -> str:
        return " / ".join(self.path)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PowerSensor({self.label!r}, {self.watts} W)"


def _walk(
    node: Dict[str, object], path: Tuple[str, ...]
) -> Iterator[PowerSensor]:
    text = str(node.get("Text", ""))
    here = path + (text,) if text else path
    children = node.get("Children") or []
    if not children:
        watts = parse_watts(node.get("Value"))
        if watts is not None:
            sensor_id = node.get("SensorId")
            yield PowerSensor(
                here, watts, str(sensor_id) if sensor_id is not None else None
            )
        return
    for child in children:  # type: ignore[union-attr]
        if isinstance(child, dict):
            yield from _walk(child, here)


def power_sensors(tree: Dict[str, object]) -> List[PowerSensor]:
    """Every leaf in *tree* whose value is a wattage."""
    return list(_walk(tree, ()))


def find_package_sensor(
    sensors: Sequence[PowerSensor], hint: Optional[str] = None
) -> Optional[PowerSensor]:
    """The whole-package power sensor, or None.

    *hint* is a case-insensitive substring matched against the full path (or the
    LHM ``SensorId``), for hosts where the default names do not apply.
    """
    if hint:
        needle = hint.lower()
        for sensor in sensors:
            if needle in sensor.label.lower() or (
                sensor.sensor_id and needle in sensor.sensor_id.lower()
            ):
                return sensor
        return None
    for name in PACKAGE_SENSOR_NAMES:
        for sensor in sensors:
            if sensor.path and sensor.path[-1] == name:
                return sensor
    return None


class HwmonPowerSource:
    """Reads CPU package watts from an OHM/LHM ``data.json`` endpoint.

    Usage::

        source = HwmonPowerSource()
        source.discover()          # raises HwmonUnavailable with instructions
        source.calibrate_refresh() # measures the monitor's update interval
        watts = source.power_watts()
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        sensor_hint: Optional[str] = None,
        timeout_s: float = 1.0,
    ) -> None:
        self.url = url
        self.sensor_hint = sensor_hint
        self.timeout_s = timeout_s
        self.sensor_path: Optional[Tuple[str, ...]] = None
        self.sensor_id: Optional[str] = None
        self.refresh_s: float = DEFAULT_REFRESH_S
        self.refresh_measured = False

    # ------------------------------------------------------------------ #

    def fetch_tree(self) -> Dict[str, object]:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as resp:
                payload = resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise HwmonUnavailable(
                f"{setup_instructions(self.url)}\n  (underlying error: {exc})"
            ) from exc
        try:
            tree = json.loads(payload.decode("utf-8-sig"))
        except ValueError as exc:
            raise HwmonUnavailable(
                f"{self.url} answered, but not with a sensor tree: {exc}"
            ) from exc
        if not isinstance(tree, dict):
            raise HwmonUnavailable(f"{self.url} returned {type(tree).__name__}")
        return tree

    def discover(self) -> PowerSensor:
        """Locate the package sensor and pin its path for later reads."""
        sensors = power_sensors(self.fetch_tree())
        sensor = find_package_sensor(sensors, self.sensor_hint)
        if sensor is None:
            found = ", ".join(s.label for s in sensors) or "none"
            raise HwmonUnavailable(
                "The hardware monitor is running but exposes no CPU package "
                f"power sensor. Power sensors found: {found}.\n"
                "  If the list is empty, the monitor is not running as "
                "administrator, so it cannot read RAPL.\n"
                "  Otherwise pass --hwmon-sensor with part of the right path."
            )
        self.sensor_path = sensor.path
        self.sensor_id = sensor.sensor_id
        return sensor

    def power_watts(self) -> Optional[float]:
        """The pinned sensor's current value, or None on any read failure.

        Telemetry is best-effort inside a sampling loop: a single failed read
        becomes a gap in the series, never an exception on the sampler thread.
        """
        if self.sensor_path is None:
            try:
                self.discover()
            except HwmonUnavailable:
                return None
        try:
            sensors = power_sensors(self.fetch_tree())
        except HwmonUnavailable:
            return None
        for sensor in sensors:
            if self.sensor_id and sensor.sensor_id == self.sensor_id:
                return sensor.watts
            if sensor.path == self.sensor_path:
                return sensor.watts
        return None

    def calibrate_refresh(
        self, seconds: float = 4.0, poll_s: float = 0.05
    ) -> float:
        """Measure how often the monitor publishes a new value.

        The median gap between value changes. Kept at the default when the
        value never changes in the window (a perfectly flat reading is possible
        on a quiet package), and ``refresh_measured`` records which applies.
        """
        changes: List[float] = []
        last: Optional[float] = None
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            value = self.power_watts()
            now = time.monotonic()
            if value is not None and last is not None and value != last:
                changes.append(now)
            if value is not None:
                last = value
            time.sleep(poll_s)
        gaps = sorted(b - a for a, b in zip(changes, changes[1:]))
        if gaps:
            self.refresh_s = gaps[len(gaps) // 2]
            self.refresh_measured = True
        return self.refresh_s

    def describe(self) -> Dict[str, object]:
        return {
            "instrument": "Intel RAPL package energy (MSR_PKG_ENERGY_STATUS)",
            "transport": "OpenHardwareMonitor/LibreHardwareMonitor data.json",
            "url": self.url,
            "sensor": " / ".join(self.sensor_path) if self.sensor_path else None,
            "sensor_id": self.sensor_id,
            "refresh_s": self.refresh_s,
            "refresh_measured": self.refresh_measured,
            "scope": "CPU package (cores + uncore + integrated GPU); excludes "
            "DRAM, storage, display and the discrete GPU",
        }


def probe(url: str = DEFAULT_URL, sensor_hint: Optional[str] = None) -> Dict:
    """One-shot status for ``bopis profile``: reachable, sensor, current watts."""
    source = HwmonPowerSource(url=url, sensor_hint=sensor_hint)
    try:
        sensor = source.discover()
    except HwmonUnavailable as exc:
        return {"available": False, "url": url, "reason": str(exc)}
    return {
        "available": True,
        "url": url,
        "sensor": sensor.label,
        "watts": sensor.watts,
    }


__all__ = [
    "DEFAULT_URL",
    "DEFAULT_REFRESH_S",
    "HwmonPowerSource",
    "HwmonUnavailable",
    "PowerSensor",
    "find_package_sensor",
    "parse_watts",
    "power_sensors",
    "probe",
    "setup_instructions",
]
