"""Smoke tests for NVML failure/recovery handling in _gather_metrics.

Runs without a GPU by swapping in a fake pynvml module.
"""
import sys
import types

sys.path.insert(0, ".")

import main  # noqa: E402


class FakeNVMLError(Exception):
    def __init__(self, code=999):
        super().__init__("Unknown Error")
        self.value = code


def make_fake_pynvml(init_failures=0, gather_failures=0):
    state = {"init_calls": 0, "gather_calls": 0, "shutdown_calls": 0}
    fake = types.SimpleNamespace()
    fake.NVMLError = FakeNVMLError

    def nvmlInit():
        state["init_calls"] += 1
        if state["init_calls"] <= init_failures:
            raise FakeNVMLError()

    def nvmlShutdown():
        state["shutdown_calls"] += 1

    def driver_version():
        state["gather_calls"] += 1
        if state["gather_calls"] <= gather_failures:
            raise FakeNVMLError()
        return b"535.154.05"

    fake.nvmlInit = nvmlInit
    fake.nvmlShutdown = nvmlShutdown
    fake.nvmlSystemGetDriverVersion = driver_version
    fake.nvmlSystemGetNVMLVersion = lambda: b"12.2"
    fake.nvmlDeviceGetCount = lambda: 0
    fake._state = state
    return fake


def check(name, cond):
    print(("PASS" if cond else "FAIL") + f": {name}")
    if not cond:
        sys.exit(1)


# 1. pynvml missing entirely -> graceful error response, no exception
main.pynvml = None
main._nvml_ready = False
r = main._gather_metrics()
check("pynvml None -> status error", r.status == "error" and r.gpu_count == 0)

# 2. init always fails -> graceful error response, no exception, shutdown attempted
fake = make_fake_pynvml(init_failures=10**9)
main.pynvml = fake
main._nvml_ready = False
r = main._gather_metrics()
check("init always fails -> status error", r.status == "error")
check("reset attempted shutdown", fake._state["shutdown_calls"] >= 1)

# 3. init fails twice then recovers -> next request succeeds via reset path
fake = make_fake_pynvml(init_failures=2)
main.pynvml = fake
main._nvml_ready = False
r1 = main._gather_metrics()
check("first request during outage -> error", r1.status == "error")
r2 = main._gather_metrics()
check("second request recovers -> ok", r2.status == "ok")
check("driver version parsed", r2.system_info["driver_version"] == "535.154.05")

# 4. init ok but gather raises NVMLError once -> reset + retry within same request
fake = make_fake_pynvml(gather_failures=1)
main.pynvml = fake
main._nvml_ready = False
r = main._gather_metrics()
check("mid-gather NVMLError -> retry succeeds", r.status == "ok")
check("retry performed shutdown+init", fake._state["shutdown_calls"] == 1 and fake._state["init_calls"] == 2)

# 5. gather keeps failing -> error response after single retry (2 gather attempts)
fake = make_fake_pynvml(gather_failures=10**9)
main.pynvml = fake
main._nvml_ready = False
r = main._gather_metrics()
check("persistent gather failure -> error", r.status == "error")
check("exactly one retry", fake._state["gather_calls"] == 2)

# 6. init happens once across many requests (no per-request init/shutdown)
fake = make_fake_pynvml()
main.pynvml = fake
main._nvml_ready = False
for _ in range(5):
    r = main._gather_metrics()
check("5 requests all ok", r.status == "ok")
check("nvmlInit called exactly once", fake._state["init_calls"] == 1)
check("no per-request shutdown", fake._state["shutdown_calls"] == 0)

print("All smoke tests passed.")
