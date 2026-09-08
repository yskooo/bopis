"""Run directories, CSV table writers, the manifest, and the dashboard export.

Every artifact BOPIS produces lands under ``runs/<UTC-timestamp>/``. The layout
mirrors the manuscript's appendix so a reader can go from a table number in the
document to a file on disk without a lookup.

Two design points worth stating
-------------------------------
* **Per-prompt rows are appended immediately, never buffered.** The full
  validation is 500 prompts x 3 conditions ~= 1500 generations; at realistic
  throughput that is a multi-hour run, and it *will* be interrupted. Writing
  each row as it is measured, plus :meth:`RunDirectory.completed_keys`, is what
  makes ``--resume`` possible rather than a rewrite.
* **Rows are schema-validated on write.** An unknown column raises instead of
  being silently dropped, so a typo in a measurement key fails loudly at the
  first row rather than producing a quietly incomplete table.

Standard library only.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import json
import os
import platform
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from bopis import __version__, schemas

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
DASHBOARD_DATA_NAME = "dashboard_data.js"
AMENDMENTS_NAME = "amendments.json"


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)


def utc_stamp() -> str:
    """Filesystem-safe UTC timestamp, e.g. ``20260907T041530Z``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class TableWriter:
    """Append-only CSV writer bound to one schema.

    Writes the header on creation and flushes after every row, so an interrupted
    run leaves a valid, readable partial table.
    """

    def __init__(self, path: str, table_id: str, resume: bool = False) -> None:
        self.path = path
        self.table_id = table_id
        self.columns = schemas.TABLES[table_id]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        existed = os.path.exists(path) and os.path.getsize(path) > 0
        self._handle = open(
            path, "a" if (resume and existed) else "w", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(
            self._handle, fieldnames=list(self.columns), restval=""
        )
        if not (resume and existed):
            self._writer.writeheader()
            self._handle.flush()
        self.n_rows = 0

    def write(self, row: Mapping[str, object]) -> None:
        unknown = schemas.validate_row(self.table_id, dict(row))
        if unknown:
            raise KeyError(
                f"table {self.table_id} ({self.path}) has no column(s) "
                f"{unknown}; known columns are {list(self.columns)}"
            )
        self._writer.writerow(dict(row))
        self._handle.flush()
        self.n_rows += 1

    def write_all(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            self.write(row)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TableWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class RunDirectory:
    """One run's output tree."""

    SUBDIRS = ("calibration", "validation", "dataset", "raw")

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        for name in self.SUBDIRS:
            os.makedirs(os.path.join(self.root, name), exist_ok=True)

    # -- construction -------------------------------------------------------- #

    @classmethod
    def create(cls, base: str = "runs", label: Optional[str] = None) -> "RunDirectory":
        name = utc_stamp() + (f"_{label}" if label else "")
        return cls(os.path.join(base, name))

    @classmethod
    def latest(cls, base: str = "runs") -> Optional["RunDirectory"]:
        if not os.path.isdir(base):
            return None
        entries = sorted(
            entry
            for entry in os.listdir(base)
            if os.path.isdir(os.path.join(base, entry))
        )
        if not entries:
            return None
        return cls(os.path.join(base, entries[-1]))

    # -- paths --------------------------------------------------------------- #

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def table_path(self, table_id: str) -> str:
        return self.path(schemas.TABLE_FILES[table_id])

    def writer(self, table_id: str, resume: bool = False) -> TableWriter:
        return TableWriter(self.table_path(table_id), table_id, resume=resume)

    # -- whole-table helpers ------------------------------------------------- #

    def write_table(
        self, table_id: str, rows: Sequence[Mapping[str, object]]
    ) -> str:
        with self.writer(table_id) as writer:
            writer.write_all(rows)
        return self.table_path(table_id)

    def read_table(self, table_id: str) -> List[Dict[str, str]]:
        path = self.table_path(table_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def completed_keys(self, table_id: str, *key_columns: str) -> Set[tuple]:
        """Composite keys already present in *table_id*, for ``--resume``.

        Example: ``completed_keys("B.3", "condition", "prompt_index")`` returns
        the ``(condition, prompt_index)`` pairs already measured, so the runner
        can skip them.
        """
        rows = self.read_table(table_id)
        return {tuple(row.get(col, "") for col in key_columns) for row in rows}

    # -- JSON artifacts ------------------------------------------------------ #

    def write_json(self, name: str, payload: Mapping[str, object]) -> str:
        target = self.path(name)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
            handle.write("\n")
        return target

    def read_json(self, name: str) -> Optional[Dict[str, object]]:
        target = self.path(name)
        if not os.path.exists(target):
            return None
        with open(target, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_manifest(self, payload: Mapping[str, object]) -> str:
        return self.write_json(MANIFEST_NAME, payload)

    def write_metrics(self, payload: Mapping[str, object]) -> str:
        return self.write_json(METRICS_NAME, payload)

    # -- dashboard ----------------------------------------------------------- #

    def write_dashboard_data(self, payload: Mapping[str, object]) -> str:
        """Emit ``dashboard_data.js`` assigning ``window.BOPIS_DATA``.

        A ``.js`` assignment rather than a ``.json`` fetch, because the dashboard
        must open from ``file://`` with no server or build step, and browsers
        block ``fetch`` of local files under the same-origin policy.
        """
        target = self.path(DASHBOARD_DATA_NAME)
        body = json.dumps(payload, indent=2, default=_json_default)
        header = (
            "// Generated by BOPIS -- do not edit.\n"
            f"// Run: {os.path.basename(self.root)}\n"
            "// Every value below is derived from this run's measured "
            "artifacts.\n"
        )
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(header)
            handle.write("window.BOPIS_DATA = ")
            handle.write(body)
            handle.write(";\n")
        return target

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        return f"RunDirectory({os.path.basename(self.root)!r})"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def build_manifest(
    host_profile: Mapping[str, object],
    space_summary: Mapping[str, object],
    backend: Mapping[str, object],
    settings: Mapping[str, object],
    dataset: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Assemble the run manifest.

    This is the provenance record: which machine, which rules narrowed the
    search space, which backend and model, which energy instrument, which seed.
    Without it, none of the CSVs can be interpreted -- and with it, a reader can
    tell at a glance whether an energy figure is measured, cross-checked or
    simulated.
    """
    return {
        "bopis_version": __version__,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "host_profile": dict(host_profile),
        "configuration_space": dict(space_summary),
        "backend": dict(backend),
        "settings": dict(settings),
        "dataset": dict(dataset) if dataset else None,
        "dependency_policy": {
            "core": "python standard library only",
            "evaluation_only": ["transformers", "torch (BERTScore F1)"],
            "note": (
                "The measurement core imports no third-party package. BERTScore "
                "runs as a separate offline scoring stage; see "
                "bopis.quality.bertscore."
            ),
        },
    }
