"""CPU package energy (RAPL via OHM/LHM), the quality stage, and ``bopis ui``.

No hardware monitor or llama-server is needed: both are replaced by local
``http.server`` fakes serving the same JSON shapes, so the full path -- urllib
read, sensor discovery, lag-compensated integration, scope labelling, and the
UI bridge's chat endpoint -- is exercised on any machine.
"""

from __future__ import annotations

import json
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bopis import hardware, runner
from bopis import config_space as cs
from bopis.dataset import Prompt
from bopis.measure import EnergyScope, PromptMeasurement
from bopis.monitor import hwmon
from bopis.monitor.nvml import EnergyMethod
from bopis.monitor.sampler import TelemetrySampler, integrate_held
from bopis.quality import QualityScore


def ohm_tree(package_watts: str = "7.5 W", sensor_id=None) -> dict:
    """The shape Open Hardware Monitor's data.json serves."""
    package = {"id": 9, "Text": "CPU Package", "Min": "2.0 W", "Value": package_watts,
               "Max": "25.0 W", "ImageURL": "", "Children": []}
    if sensor_id:
        package["SensorId"] = sensor_id
    return {
        "id": 0, "Text": "Sensor", "Children": [{
            "id": 1, "Text": "LAPTOP", "Children": [{
                "id": 2, "Text": "Intel Core i5-1135G7", "Children": [
                    {"id": 3, "Text": "Load", "Children": [
                        {"id": 4, "Text": "CPU Total", "Value": "12.0 %",
                         "Children": []}]},
                    {"id": 5, "Text": "Powers", "Children": [
                        package,
                        {"id": 10, "Text": "CPU Cores", "Value": "4.1 W",
                         "Children": []}]},
                ]}, {
                "id": 6, "Text": "NVIDIA GeForce MX330", "Children": [
                    {"id": 7, "Text": "Temperatures", "Children": [
                        {"id": 8, "Text": "GPU Core", "Value": "45.0 °C",
                         "Children": []}]}]},
            ]}],
    }


class _Server:
    """A throwaway local HTTP server with a scripted handler."""

    def __init__(self, handler_cls) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def ohm_server(watts_fn) -> _Server:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps(ohm_tree(f"{watts_fn():.1f} W")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    return _Server(Handler)


class FakeSource:
    """A package-power source whose value is a function of time."""

    def __init__(self, watts_fn, refresh_s: float = 0.2) -> None:
        self.watts_fn = watts_fn
        self.refresh_s = refresh_s
        self.refresh_measured = True

    def power_watts(self):
        return self.watts_fn(time.monotonic())

    def describe(self):
        return {"sensor": "fake"}


class TestParsing(unittest.TestCase):
    def test_parse_watts(self) -> None:
        self.assertEqual(hwmon.parse_watts("7.3 W"), 7.3)
        self.assertEqual(hwmon.parse_watts("7,3 W"), 7.3)  # decimal-comma locale
        self.assertEqual(hwmon.parse_watts("12 W"), 12.0)
        self.assertIsNone(hwmon.parse_watts("45.0 °C"))
        self.assertIsNone(hwmon.parse_watts("12.0 %"))
        self.assertIsNone(hwmon.parse_watts(None))

    def test_finds_only_power_leaves(self) -> None:
        labels = [s.label for s in hwmon.power_sensors(ohm_tree())]
        self.assertEqual(len(labels), 2)
        self.assertTrue(all("Powers" in label for label in labels))

    def test_package_sensor_preferred(self) -> None:
        sensor = hwmon.find_package_sensor(hwmon.power_sensors(ohm_tree("9.9 W")))
        self.assertEqual(sensor.path[-1], "CPU Package")
        self.assertEqual(sensor.watts, 9.9)

    def test_hint_overrides_default(self) -> None:
        sensors = hwmon.power_sensors(ohm_tree())
        self.assertEqual(
            hwmon.find_package_sensor(sensors, hint="cores").path[-1], "CPU Cores"
        )
        self.assertIsNone(hwmon.find_package_sensor(sensors, hint="nonexistent"))

    def test_hint_matches_lhm_sensor_id(self) -> None:
        sensors = hwmon.power_sensors(ohm_tree(sensor_id="/intelcpu/0/power/0"))
        self.assertEqual(
            hwmon.find_package_sensor(sensors, hint="/power/0").sensor_id,
            "/intelcpu/0/power/0",
        )


class TestHttpSource(unittest.TestCase):
    def test_reads_over_http(self) -> None:
        server = ohm_server(lambda: 11.0)
        try:
            source = hwmon.HwmonPowerSource(server.url + "/data.json")
            self.assertEqual(source.discover().watts, 11.0)
            self.assertEqual(source.power_watts(), 11.0)
            probe = hwmon.probe(server.url + "/data.json")
            self.assertTrue(probe["available"])
        finally:
            server.close()

    def test_unreachable_is_actionable(self) -> None:
        source = hwmon.HwmonPowerSource("http://127.0.0.1:9/data.json", timeout_s=0.3)
        with self.assertRaises(hwmon.HwmonUnavailable) as caught:
            source.discover()
        self.assertIn("administrator", str(caught.exception).lower())
        self.assertIsNone(source.power_watts())  # never raises inside a sampler
        self.assertFalse(hwmon.probe("http://127.0.0.1:9/data.json")["available"])


class TestIntegrateHeld(unittest.TestCase):
    def test_constant_power(self) -> None:
        readings = [(0.0, 10.0), (1.0, 10.0), (2.0, 10.0)]
        result = integrate_held(readings, 0.0, 2.0)
        self.assertAlmostEqual(result["gross_j"], 20.0)
        self.assertAlmostEqual(result["net_j"], 20.0)

    def test_zero_order_hold_not_trapezoid(self) -> None:
        # 5 W for one second, then 15 W: a hold gives 5 + 15, a trapezoid would
        # invent a ramp and give 10 + 15.
        readings = [(0.0, 5.0), (1.0, 15.0)]
        self.assertAlmostEqual(integrate_held(readings, 0.0, 2.0)["gross_j"], 20.0)

    def test_idle_subtraction_and_clamps(self) -> None:
        readings = [(0.0, 3.0), (1.0, 12.0)]
        result = integrate_held(readings, 0.0, 2.0, p_idle_w=4.0)
        self.assertAlmostEqual(result["net_j"], 8.0)  # (3-4 -> 0) + (12-4)
        self.assertEqual(result["clamped"], 1.0)

    def test_window_starts_between_readings(self) -> None:
        readings = [(0.0, 10.0), (1.0, 20.0)]
        # [0.5, 1.5]: half a second at 10 W then half at 20 W.
        self.assertAlmostEqual(integrate_held(readings, 0.5, 1.5)["gross_j"], 15.0)

    def test_empty(self) -> None:
        self.assertIsNone(integrate_held([], 0.0, 1.0))
        self.assertIsNone(integrate_held([(0.0, 1.0)], 1.0, 1.0))


class TestSamplerWithPackageSource(unittest.TestCase):
    def test_measures_and_labels(self) -> None:
        source = FakeSource(lambda _t: 12.0, refresh_s=0.2)
        sampler = TelemetrySampler(
            cpu_power_source=source, p_cpu_idle_w=2.0, interval_s=0.02
        )
        sampler.start()
        time.sleep(0.5)
        window = sampler.stop()
        self.assertEqual(window.energy_method, EnergyMethod.RAPL_HWMON_POWER_INTEGRATION)
        # 10 W net over the window's duration.
        self.assertAlmostEqual(window.energy_j, 10.0 * window.duration_s, delta=0.15)
        self.assertAlmostEqual(window.energy_gross_j, 12.0 * window.duration_s, delta=0.2)
        self.assertEqual(window.hwmon_refresh_s, 0.2)
        self.assertLessEqual(window.energy_low_j, window.energy_j)
        self.assertGreaterEqual(window.energy_high_j, window.energy_j)

    def test_lag_compensation(self) -> None:
        """The monitor publishes each interval's mean one interval late.

        The fake reports 20 W starting 0.2 s *after* the load starts, exactly as
        a monitor with a 0.2 s refresh would. Integrating the unshifted window
        would miss that last stretch; the shifted one recovers it.
        """
        started = {}

        def watts(t):
            if "t0" not in started:
                return 0.0
            return 20.0 if t >= started["t0"] + 0.2 else 0.0

        source = FakeSource(watts, refresh_s=0.2)
        sampler = TelemetrySampler(cpu_power_source=source, interval_s=0.01)
        sampler.start()
        started["t0"] = sampler._started_at
        time.sleep(0.6)
        window = sampler.stop()
        # Load ran for the whole window at 20 W.
        self.assertAlmostEqual(window.energy_j, 20.0 * window.duration_s, delta=0.6)

    def test_estimator_not_reached_when_measured(self) -> None:
        from bopis.monitor import estimator

        sampler = TelemetrySampler(
            cpu_power_source=FakeSource(lambda _t: 5.0, refresh_s=0.05),
            estimator_budget=estimator.PowerBudget(),
            interval_s=0.01,
        )
        sampler.start()
        time.sleep(0.1)
        self.assertEqual(
            sampler.stop().energy_method, EnergyMethod.RAPL_HWMON_POWER_INTEGRATION
        )


class TestScope(unittest.TestCase):
    def _m(self, g: int) -> PromptMeasurement:
        return PromptMeasurement(
            prompt_index=0, prompt_id="p", task_type="open_qa", condition="search",
            config=cs.Config(t=128, b=1, p="Q4_K_M", g=g, c=4),
            energy_j=1.0, energy_scope=EnergyScope.CPU_PACKAGE_RAPL,
        )

    def test_package_scope_requires_cpu_only(self) -> None:
        self.assertTrue(self._m(0).scope_valid)
        self.assertFalse(self._m(14).scope_valid)

    def test_max_gpu_layers_pins_cpu_only(self) -> None:
        profile = hardware.profile_host()
        space, rejections = hardware.feasible_space(profile, max_gpu_layers=0)
        self.assertTrue(space)
        self.assertTrue(all(cfg.g == 0 for cfg in space))
        if any(g != 0 for g in profile.permitted_gpu_layers):
            self.assertTrue(any(r.rule == "ENERGY-SCOPE" for r in rejections))


class FakeScorer:
    name = "bertscore"

    def __init__(self) -> None:
        self.calls = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        return [QualityScore(f1=0.5, precision=0.4, recall=0.6, scorer="bertscore",
                             baseline_rescaled=True) for _ in candidates]

    def describe(self):
        return {"scorer": "fake"}


class TestQualityStage(unittest.TestCase):
    def test_scores_against_dolly_reference(self) -> None:
        config = cs.Config(t=128, b=1, p="Q4_K_M", g=0, c=4)
        prompts = [
            Prompt(0, "a", "open_qa", "Q1", "", "ref one"),
            Prompt(1, "b", "open_qa", "Q2", "", "ref two"),
            Prompt(2, "c", "open_qa", "Q3", "", "ref three"),
        ]
        results = [
            PromptMeasurement(0, "a", "open_qa", "search", config, text="answer"),
            PromptMeasurement(1, "b", "open_qa", "search", config, text="   "),
            PromptMeasurement(2, "c", "open_qa", "search", config, text="x", error="boom"),
        ]
        scorer = FakeScorer()
        runner.Study._score_quality(types.SimpleNamespace(scorer=scorer), results, prompts)
        self.assertEqual(scorer.calls, [(["answer"], ["ref one"])])
        self.assertEqual(results[0].quality_f1, 0.5)
        self.assertTrue(results[0].baseline_rescaled)
        # An empty answer is a bad answer, not a missing one.
        self.assertEqual(results[1].quality_f1, 0.0)
        # A failed request has no answer to score.
        self.assertIsNone(results[2].quality_f1)

    def test_no_scorer_is_a_noop(self) -> None:
        config = cs.Config(t=128, b=1, p="Q4_K_M", g=0, c=4)
        result = PromptMeasurement(0, "a", "open_qa", "search", config, text="x")
        runner.Study._score_quality(types.SimpleNamespace(scorer=None), [result], [])
        self.assertIsNone(result.quality_f1)


class TestUiBridge(unittest.TestCase):
    """``bopis ui``'s endpoints against fake llama-server and OHM servers."""

    @classmethod
    def setUpClass(cls) -> None:
        from bopis import ui_server

        class Llama(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                time.sleep(0.3)  # "inference"
                body = json.dumps({
                    "choices": [{"message": {"content": "Paris."}}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 2},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        cls.llama = _Server(Llama)
        cls.ohm = ohm_server(lambda: 9.0)
        cls.instruments = ui_server.Instruments(
            llama_url=cls.llama.url,
            hwmon_url=cls.ohm.url + "/data.json",
            idle_seconds=0.3,
            llama_pid=None,
            data_dir="nonexistent-data-dir",
            log=lambda _m: None,
        )
        cls.instruments.source.refresh_s = 0.1  # the fake updates every read
        cls.ui = _Server(ui_server.make_handler(cls.instruments, 0))

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.ui, cls.llama, cls.ohm):
            server.close()

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.ui.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def test_chat_is_measured(self) -> None:
        result = self._post("/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(result["completion"]["choices"][0]["message"]["content"], "Paris.")
        energy = result["energy"]
        self.assertTrue(energy["measured"])
        self.assertEqual(energy["scope"], "cpu_package_rapl")
        # Constant 9 W with 9 W idle -> net ~0; gross ~9 W x duration.
        self.assertAlmostEqual(energy["energy_gross_j"], 9.0 * energy["duration_s"], delta=0.5)
        self.assertIsNotNone(energy["cost_php"])

    def test_status(self) -> None:
        with urllib.request.urlopen(self.ui.url + "/api/status", timeout=10) as r:
            status = json.loads(r.read())
        self.assertTrue(status["energy"]["measured"])
        self.assertTrue(status["llama"]["ready"])
        self.assertEqual(status["dolly"]["n"], 0)

    def test_score_requires_both_texts(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._post("/api/score", {"candidate": "x"})
        self.assertEqual(caught.exception.code, 400)

    def test_serves_only_known_files(self) -> None:
        with urllib.request.urlopen(self.ui.url + "/", timeout=10) as r:
            self.assertIn(b"BOPIS", r.read())
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.ui.url + "/bopis/cli.py", timeout=10)
        self.assertEqual(caught.exception.code, 404)

    def test_cors_only_for_local_origins(self) -> None:
        def origin_header(origin):
            request = urllib.request.Request(
                self.ui.url + "/api/status", headers={"Origin": origin}
            )
            with urllib.request.urlopen(request, timeout=10) as r:
                return r.headers.get("Access-Control-Allow-Origin")

        self.assertEqual(origin_header("null"), "null")  # file:// page
        self.assertIsNone(origin_header("https://example.com"))


if __name__ == "__main__":
    unittest.main()
