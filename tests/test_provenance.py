"""Two guards that keep the artifact's claims true.

Both failure modes these prevent are quiet ones: a "standard-library-only"
measurement core that gradually acquires third-party imports, and a beautiful
dashboard full of numbers nobody can trace back to a measurement. Neither shows
up as a crash, so neither is caught by any other test.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(REPO_ROOT, "bopis")

#: The one module permitted to import third-party packages. BERTScore requires a
#: transformer forward pass, which no standard-library implementation can do, so
#: it is isolated here and runs as a separate offline scoring stage.
QUALITY_EXEMPTION = os.path.join("bopis", "quality", "bertscore.py")


def _stdlib_names() -> frozenset:
    names = set(sys.stdlib_module_names)
    names.discard("this")
    names.discard("antigravity")
    return frozenset(names)


STDLIB = _stdlib_names()


def _python_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _imported_roots(path: str):
    """Top-level module names imported by *path*, ignoring relative imports."""
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


class TestStandardLibraryOnly(unittest.TestCase):
    """Amendment A-6: the measurement core imports nothing third-party."""

    def test_core_modules_import_only_stdlib(self) -> None:
        offenders = {}
        for path in _python_files(PACKAGE_ROOT):
            relative = os.path.relpath(path, REPO_ROOT)
            if relative.replace("/", os.sep) == QUALITY_EXEMPTION:
                continue
            foreign = {
                name
                for name in _imported_roots(path)
                if name not in STDLIB and name != "bopis"
            }
            if foreign:
                offenders[relative] = sorted(foreign)
        self.assertEqual(
            offenders,
            {},
            "the BOPIS core must import only the standard library; "
            f"found third-party imports: {offenders}",
        )

    def test_exemption_is_a_single_documented_file(self) -> None:
        """The exemption list must stay at exactly one module."""
        self.assertEqual(
            QUALITY_EXEMPTION, os.path.join("bopis", "quality", "bertscore.py")
        )

    def test_core_imports_in_a_clean_interpreter(self) -> None:
        """Import the whole core with third-party paths stripped.

        A static scan can miss a lazy import inside a function body, so this
        actually executes the imports with ``sys.path`` reduced to the standard
        library and the repo itself.
        """
        script = (
            "import sys, os\n"
            "root = sys.argv[1]\n"
            "sys.path = [p for p in sys.path "
            "if 'site-packages' not in p and 'dist-packages' not in p]\n"
            "sys.path.insert(0, root)\n"
            "import bopis.acquisition, bopis.artifacts, bopis.cli\n"
            "import bopis.config_space, bopis.dashboard, bopis.dataset\n"
            "import bopis.gp, bopis.hardware, bopis.measure, bopis.metrics\n"
            "import bopis.monitor, bopis.monitor.nvml, bopis.monitor.platform_os\n"
            "import bopis.optimizer, bopis.pareto, bopis.runner\n"
            "import bopis.schemas, bopis.stats, bopis.tasks\n"
            "import bopis.backends, bopis.backends.simulator\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, REPO_ROOT],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"core failed to import without site-packages:\n{result.stderr}",
        )
        self.assertIn("ok", result.stdout)

    def test_psutil_is_not_used(self) -> None:
        """CPU/RAM telemetry must go through /proc or ctypes, not psutil."""
        for path in _python_files(PACKAGE_ROOT):
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn(
                "import psutil",
                source,
                f"{os.path.relpath(path, REPO_ROOT)} imports psutil; the core "
                "must use /proc (Linux) or ctypes/kernel32 (Windows)",
            )

    def test_pynvml_is_not_used(self) -> None:
        """NVML must be reached through ctypes, per Chapter 3."""
        for path in _python_files(PACKAGE_ROOT):
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            for banned in ("import pynvml", "from pynvml"):
                self.assertNotIn(
                    banned,
                    source,
                    f"{os.path.relpath(path, REPO_ROOT)} uses pynvml; NVML must "
                    "be accessed directly through ctypes",
                )


class TestDashboardHasNoHardcodedMetrics(unittest.TestCase):
    """The mockup asserted its results; the dashboard must derive them."""

    def setUp(self) -> None:
        self.path = os.path.join(REPO_ROOT, "dashboard", "index.html")
        if not os.path.exists(self.path):
            self.skipTest("dashboard/index.html not present")
        with open(self.path, "r", encoding="utf-8") as handle:
            self.source = handle.read()

    def _script_body(self) -> str:
        """Everything inside <script> tags, with comments stripped."""
        blocks = re.findall(r"<script[^>]*>(.*?)</script>", self.source, re.S)
        body = "\n".join(blocks)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        body = re.sub(r"(?m)^\s*//.*$", "", body)
        return body

    def test_no_leftover_mockup_values(self) -> None:
        """The fabricated numbers from bopis.html must be gone.

        Only distinctive strings are checked. A bare integer like ``42`` cannot
        be tested for, because it legitimately appears in CSS lengths and colour
        values -- so the general prohibition is enforced by
        :meth:`test_no_numeric_literals_in_rendered_markup` instead.
        """
        for literal in (
            "1,847",       # fake energy total
            "N*=23",       # fake break-even prompt count
            "1,014 J",     # fake calibration cost
            "44.1 J",      # fake per-prompt energy
            "96.2 J",      # fake default per-prompt energy
            "0.81 F1",     # fake quality score
            "34 configs",  # fake calibration count
            "₱0.003", # fake peso saving
            "2.1x less",   # fake energy multiplier
            "tok/s</strong>",
        ):
            self.assertNotIn(
                literal,
                self.source,
                f"dashboard still contains the hardcoded mockup value {literal!r}",
            )

    #: Numeric strings that form part of a metric's *definition* rather than
    #: being a measured value or a tunable threshold. These are fixed by the
    #: statistics, not by configuration, so hardcoding them is correct:
    #:
    #: ``within 5%``
    #:     names the convergence metric ``iterations_to_within(0.05)``.
    #: ``95% predictive intervals`` / ``the 95% interval``
    #:     name the +/- 1.96 sigma interval that UCR is defined against.
    #:
    #: Everything else -- results and thresholds alike -- must be interpolated.
    DEFINITIONAL_LITERALS = ("within 5%", "95% predictive", "the 95% interval")

    def test_no_numeric_literals_in_rendered_markup(self) -> None:
        """Template literals must interpolate, never hardcode, metric values.

        Scans the HTML body (outside ``<style>`` and comments) for a number
        immediately followed by a unit only a measurement or threshold would
        carry. A hit is a value the reader cannot trace back to an artifact.
        """
        body = re.sub(r"<style[^>]*>.*?</style>", "", self.source, flags=re.S)
        # Comments document the method and may quote figures freely; only
        # markup that actually reaches the reader is in scope.
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        body = re.sub(r"(?m)^\s*//.*$", "", body)
        # e.g. a literal "41.78%", "137 J", "22.4 tok/s" or "0.81 F1".
        pattern = re.compile(r"(?<![\w.$])\d+(?:[.,]\d+)?\s*(?:%|J\b|tok/s|F1\b)")

        hits = []
        for match in pattern.finditer(body):
            context = body[max(0, match.start() - 30) : match.end() + 20]
            if any(allowed in context for allowed in self.DEFINITIONAL_LITERALS):
                continue
            hits.append(match.group(0).strip())

        self.assertEqual(
            hits,
            [],
            "dashboard markup contains literal metric values that should be "
            f"interpolated from the run payload: {hits}",
        )

    def test_no_hardcoded_pareto_coordinates(self) -> None:
        """The mockup's literal front array must not reappear."""
        body = self._script_body()
        self.assertNotIn(
            "par: true",
            body,
            "Pareto membership must be computed by dominance testing, not "
            "asserted with a literal flag",
        )
        # An array of objects carrying e/q/s keys is the mockup's shape.
        self.assertIsNone(
            re.search(r"\{\s*e\s*:\s*\d+\s*,\s*q\s*:", body),
            "dashboard contains a literal energy/quality point array",
        )

    def test_no_stale_precision_vocabulary(self) -> None:
        """FP32/FP16/INT8 belong to an earlier draft (amendment A-23)."""
        for stale in ("FP32", "FP16", "INT8"):
            self.assertNotIn(
                stale,
                self.source,
                f"dashboard uses the stale precision label {stale!r}; the "
                "variants are F32 / F16 / Q8_0 / Q4_K_M",
            )

    def test_no_contradictory_backend_claims(self) -> None:
        """The mockup named a stack Chapter 3 does not use."""
        for wrong in ("pyRAPL", "pynvml", "ROUGE-L", "CodeCarbon"):
            self.assertNotIn(
                wrong,
                self.source,
                f"dashboard names {wrong!r}, which contradicts Chapter 3",
            )

    def test_reads_from_the_generated_payload(self) -> None:
        self.assertIn("BOPIS_DATA", self.source)
        self.assertIn("dashboard_data.js", self.source)

    def test_renders_a_dash_for_missing_values(self) -> None:
        """Absent data must show an em dash, never a fabricated default."""
        body = self._script_body()
        self.assertIn("DASH", body)
        self.assertIn("isFinite", body)


class TestIntegratedUiHasNoHardcodedThresholds(unittest.TestCase):
    """``bopis.html`` must derive the calibration floors, not restate them.

    The guard above only reads ``dashboard/index.html``, which is how the System
    Settings card came to carry four invented ``Q_min`` values and a literal
    iteration count: nothing was checking the integrated UI. ``Q_min(task)`` is
    ``metrics.q_min`` of that task's unoptimized mean F1 and ``S_min`` is p10 of
    the unoptimized default's speed -- both per-run, per-machine quantities the
    page cannot recompute, so it may only display them or show a dash.
    """

    CARD = "Calibration Thresholds"

    def setUp(self) -> None:
        self.path = os.path.join(REPO_ROOT, "bopis.html")
        if not os.path.exists(self.path):
            self.skipTest("bopis.html not present")
        with open(self.path, "r", encoding="utf-8") as handle:
            self.source = handle.read()
        start = self.source.find(self.CARD)
        self.assertNotEqual(start, -1, f"{self.CARD} card not found in bopis.html")
        # The card runs to the next card header, which is where its markup ends.
        end = self.source.find("card-header", start + len(self.CARD))
        self.card = self.source[start : end if end != -1 else len(self.source)]

    def test_card_states_no_literal_quality_floor(self) -> None:
        for match in re.finditer(r"\d+\.\d+\s*F1", self.card):
            self.fail(
                f"the {self.CARD} card hardcodes a quality floor "
                f"({match.group(0)!r}); Q_min is derived per run from "
                f"bopis.metrics.q_min and must be interpolated"
            )

    def test_card_reads_the_generated_payload(self) -> None:
        for attribute in ("data-qmin", "data-smin", "data-iterations"):
            self.assertIn(
                attribute,
                self.card,
                f"the {self.CARD} card has no {attribute} placeholder, so nothing "
                f"can populate it from the run payload",
            )
        self.assertIn("q_min_by_task", self.source)
        self.assertIn("renderThresholds", self.source)

    def test_card_defaults_to_a_dash(self) -> None:
        """No run loaded must render an em dash, never a plausible number."""
        self.assertGreaterEqual(
            self.card.count("&mdash;"), 6, "every threshold row needs a dash default"
        )


class TestIntegratedUiExplainsAMissingReference(unittest.TestCase):
    """A dropped BERTScore reference must be named, not misattributed.

    BERTScore is reference-based, so a free-typed turn legitimately has no F1.
    A loaded Dolly prompt that the user then edited is a different case: the
    reference existed and was discarded because it no longer answers the edited
    text. Both branches used to render the same "free chat has no reference"
    badge, which told someone who had clicked *Dolly prompt* that they had not.
    """

    def setUp(self) -> None:
        self.path = os.path.join(REPO_ROOT, "bopis.html")
        if not os.path.exists(self.path):
            self.skipTest("bopis.html not present")
        with open(self.path, "r", encoding="utf-8") as handle:
            self.source = handle.read()

    def test_all_three_states_are_distinguished(self) -> None:
        self.assertIn(
            "edited",
            self.source,
            "bopis.html has no branch for a loaded-but-edited Dolly prompt, so "
            "the drop cannot be explained separately from free chat",
        )
        self.assertIn("q-dropped", self.source)
        # The free-chat wording must survive for the case it is actually true for.
        self.assertIn("free chat has no reference", self.source)

    def test_reference_is_not_scored_after_an_edit(self) -> None:
        """The reference must still be withheld when the text was edited.

        Guards the fix against being "solved" by keeping the stale reference,
        which would report a real F1 for a question the reference does not
        answer and quietly corrupt the session mean.
        """
        body = re.sub(r"/\*.*?\*/", "", self.source, flags=re.S)
        self.assertIn(
            "loaded.text.trim() === txt",
            body,
            "the exact-match gate on the Dolly reference is gone; it is the only "
            "thing preventing a stale reference from being scored",
        )
        self.assertIn("const edited = (loaded && !reference) ? loaded : null;", body)


if __name__ == "__main__":
    unittest.main()
