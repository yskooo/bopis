# Energy measurement modes

This document exists because of one hardware fact and one methodological
consequence of it.

**The fact.** Chapter 3 defines the energy objective as GPU energy, obtained
from NVML — either `nvmlDeviceGetTotalEnergyConsumption` (a monotonic
millijoule counter) or the integral of `nvmlDeviceGetPowerUsage` over the
inference window. The GeForce MX330 on the development machine, like most
entry-level mobile GPUs, implements neither. Both calls return
`NVML_ERROR_NOT_SUPPORTED`. There is no driver flag, no elevation, and no
library that changes this: the board has no power-monitoring circuit for the
driver to read.

**The consequence.** No amount of software can manufacture a measurement from a
machine with no instrument. If the requirement is a *measured* GPU-only energy
figure, the only path is a GPU whose driver exposes the power query. Everything
below is about what can honestly be done when that GPU is not available, and
about labelling the difference so precisely that a reader can never mistake one
for the other.

---

## The three modes

| Mode | How energy is obtained | What may be claimed | Flag |
|---|---|---|---|
| **A. Measured (NVML)** | Driver energy counter, or trapezoidal integration of board power at 100 ms with `P_idle` subtracted | Absolute GPU-only energy, in joules. Chapter 3's objective as written. | default (`--energy-mode auto`) |
| **B. Measured (external meter)** | A wall-socket or DC-inline meter reads whole-system power; `P_idle` is subtracted manually | Absolute whole-system energy. Scope is *wider* than Chapter 3's, so it is not directly comparable to Mode A numbers | procedure, not a code path — see [Using an external meter](#using-an-external-meter) |
| **C. Estimated (resource allocation)** | Observed CPU-time and GPU-utilization scaled by declared power budgets | *Relative* comparison between configurations on the same host in the same session. **Not** absolute energy. | `--energy-mode resource-estimate` |

Mode A is the default and is never overridden. Mode C sits *below* the NVML
ladder: `bopis/monitor/sampler.py` consults it only after both measured paths
have returned nothing, so on a host that does report power the estimator is
unreachable even if the flag is passed. With no flag and no instrument, the run
is refused rather than silently downgraded.

---

## Mode C: the resource-allocation estimate

### What is still observable without a power sensor

A GPU that withholds power still reports, at full fidelity:

- **utilization** — `nvmlDeviceGetUtilizationRates`, the percentage of the
  sample period during which at least one kernel was resident;
- **VRAM** — allocated bytes.

And the operating system reports, independently of the GPU:

- **per-process CPU time** — `GetProcessTimes` on Windows, `/proc/[pid]/stat`
  on Linux, both aggregated over every thread of the process;
- **system-wide CPU time** — `GetSystemTimes`, `/proc/stat`.

The workload's *resource allocation* is therefore fully measured. Only the
conversion from allocation to watts is unmeasured, and that conversion is where
the assumptions live.

### The model

For one inference window of wall duration `T`:

```
E_est = [ (P_cpu_tdp - P_cpu_idle) * u_cpu_proc
        + (P_gpu_tdp - P_gpu_idle) * max(u_gpu - u_gpu_idle, 0) ] * T
```

| Symbol | Meaning | Source |
|---|---|---|
| `T` | Window duration, seconds | `time.monotonic()` around the request |
| `u_cpu_proc` | Inference process's CPU-seconds ÷ (`N_logical` × `T`), in [0, 1] | `GetProcessTimes` / `/proc/[pid]/stat` |
| `u_gpu` | Mean GPU duty cycle over the window, as a fraction | NVML utilization, 100 ms samples |
| `u_gpu_idle` | Mean GPU duty cycle at rest | 60 s idle calibration |
| `P_cpu_tdp`, `P_gpu_tdp` | Power at full utilization | vendor specification, `--cpu-tdp-w` / `--gpu-tdp-w` |
| `P_cpu_idle`, `P_gpu_idle` | Power at rest | vendor specification or meter, `--cpu-idle-w` / `--gpu-idle-w` |

### Why the model takes that form

**1. The coefficient is a dynamic range, not a nameplate.**
Multiplying TDP directly by utilization charges the idle draw again at every
load level. A package that burns 2 W at rest and 15 W saturated does not draw
`0.5 × 15 = 7.5 W` at half load; it draws roughly `2 + 0.5 × 13`. The objective
is the *marginal* energy of inference — the same quantity that
`E = Σ (P_t − P_idle) dt` isolates in Mode A — so the idle component is excluded
from the coefficient.

It is specifically **not** subtracted afterwards. An earlier draft of this
estimator computed `max(P_cpu_tdp·u_cpu + P_gpu_tdp·u_gpu − P_idle, 0)`, which
discounts idle twice: once by never adding it, and again by subtracting it. That
form also drives short, light windows to a spurious exact zero, which would have
told the optimizer that some configurations are free.

**2. The CPU term is attributed per process.**
System-wide CPU utilization is the wrong signal on a laptop that is also running
a browser and an editor: their load would be charged to inference energy, and
the reported figure would move with whatever the operator happened to have open.
The estimator uses the `llama-server` child process's own CPU time instead, so
concurrent applications are excluded by construction rather than by asking the
operator to close them.

When the process handle cannot be read — the server exited mid-window, or the
PID was reused — the estimator falls back to the system-wide signal and records
`cpu_attribution = system_wide_cpu_time` on that row. A silent substitution
would change what the number means without recording it; rows carrying the
fallback should be treated as contaminated by whatever else was running.

**3. The GPU term is baseline-corrected but cannot be per-process.**
NVML's consumer-driver utilization is a device-level duty cycle. There is no
per-process decomposition without accounting-mode support, which these drivers
also withhold. Subtracting the measured idle duty cycle removes the desktop
compositor's steady contribution; the remainder is charged entirely to
inference. That is assumption A4 below, and it is why the calibration step warns
when the machine was not actually idle.

### The idle baseline

Chapter 3 calibrates `P_idle` over 60 seconds at 100 ms before any inference
runs. Mode C runs the same protocol over every signal the host *does* expose:

```
Calibrating the idle baseline over 60s with no inference running...
  Idle GPU utilization         3.2% (sd 1.8, peak 11%)
  Idle CPU utilization         6.4% (sd 3.1, peak 22%)
```

The spread is the point, not the mean. A GPU idling at 40% duty cycle or a CPU
at 30% means the machine was not idle, the baseline is contaminated, and every
subsequent figure is biased. The CLI prints an explicit warning in that case and
the run should be restarted with other applications closed.

Note the asymmetry: the idle GPU duty cycle *is* subtracted, because the GPU
signal is device-wide. The idle CPU figure is recorded for diagnosis but *not*
subtracted, because the CPU term is already process-exclusive — subtracting it
would remove other applications' load from a number that never contained it.

### Uncertainty

`energy_low_j` and `energy_high_j` propagate the declared fractional
uncertainty on each power budget, combined in quadrature:

```
sigma = sqrt( (f * E_cpu)^2 + (f * E_gpu)^2 ),    f = --estimate-uncertainty
```

Quadrature rather than a linear sum because the two nameplate figures are
independent sources of error; summing them would assume both budgets are wrong
in the same direction by the same proportion. The default `f = 0.30` is
deliberately wide: vendor TDP is a sustained thermal rating, not an electrical
maximum, and the mapping from either to instantaneous draw is loose.

**This band does not bound the total error.** It covers only uncertainty in the
budgets. The error in the model's *form* — assumption A1's linearity above all —
is not captured by it and cannot be, without an instrument to compare against.

---

## Assumptions

Enumerated in `bopis/monitor/estimator.py` as `ASSUMPTIONS`, recorded verbatim
in every run manifest, and rendered in the dashboard.

| ID | Assumption | Direction of error |
|---|---|---|
| **A1** | Power is linear in utilization between the declared idle and TDP points. | Real CPU power is convex in frequency, so mid-load windows are the least accurate. |
| **A2** | Vendor TDP is the power drawn at full utilization. | TDP is a sustained thermal rating; brief bursts can exceed it and a thermally throttled part never reaches it. Overstates a throttled laptop. |
| **A3** | The CPU term is charged from the inference process's own CPU time, excluding concurrent applications. | Holds when `cpu_attribution = process_cpu_time`. Rows reading `system_wide_cpu_time` overstate energy by whatever else was running. |
| **A4** | After subtracting the idle duty cycle, all remaining GPU utilization is inference. | Overstates if another GPU client is active. Understates nothing. |
| **A5** | Only CPU package and GPU board are modelled — no DRAM, storage, display, chipset or VRM losses. | Understates total system energy. Matches Mode A's exclusion of everything off the board. |
| **A6** | Frequency and voltage scaling are not observed. | An unplugged or power-limited run is modelled as though it ran at the same operating point as a plugged-in one. Overstates a power-limited run. |

---

## What the estimate supports, and what it does not

The model is linear in utilization, ignores DVFS, and treats a duty cycle as
though it were proportional to power. The **absolute** joule figure therefore
carries an error that the uncertainty band above does not bound.

What survives those limitations is **relative** comparison. Two configurations
measured on the same host, in the same session, against the same idle baseline
share the same systematic error, so the ratio between them is far better
determined than either value alone.

| Claim | Permitted? |
|---|---|
| "Configuration A allocates ~30% less compute-energy than B on this host" | Yes — this is the ratio the model determines |
| "BOPIS reduced energy relative to the manual baseline on this host" | Yes, if reported as an estimated reduction with the caveat attached |
| "Inference consumed 41.2 J" | **No** — absolute figures are not supported |
| "Configuration A uses less energy than on machine X" | **No** — cross-host comparison compounds two sets of budget errors |
| Any of the above described as *measured* energy | **No** |

The EIR success indicator is a ratio and is therefore the indicator that
survives Mode C most intact. It must still be reported as *estimated* EIR.

---

## Using an external meter

A wall-socket or DC-inline meter is the honest upgrade path from Mode C, and it
composes with it rather than replacing it: the meter supplies *measured* values
for the budgets the estimator otherwise takes from a datasheet.

1. Close every application. Let the machine settle for five minutes.
2. Record whole-system idle power `W_idle` from the meter over 60 s.
3. Run a CPU-only saturating load (`--gpu-tdp-w 0`, all threads) and record
   steady-state `W_cpu_max`. Then `P_cpu_tdp ≈ W_cpu_max − W_idle` plus the
   package's own idle share.
4. Run a GPU-saturating load and record `W_gpu_max`, giving
   `P_gpu_tdp ≈ W_gpu_max − W_idle`.
5. Pass those figures as `--cpu-tdp-w` / `--gpu-tdp-w` and the idle components
   as `--cpu-idle-w` / `--gpu-idle-w`, and narrow `--estimate-uncertainty` to
   the meter's stated accuracy.

The result is still Mode C — the *allocation* model is unchanged, so A1 and A4
still apply — but the budgets are measured rather than assumed, which is the
single largest error term removed. Record the meter's make, model and stated
accuracy alongside the run.

A meter reading alone, without this model, gives whole-system energy at a scope
wider than Chapter 3's GPU-only definition. That is a legitimate measurement but
a different objective, and mixing it with Mode A figures in one table is not
valid.

---

## Where the data lands

Every artifact carries the labels, so a figure cannot be separated from its
provenance.

| Artifact | Field |
|---|---|
| Table B.3 (`validation/per_prompt_energy.csv`) | `energy_method` = `resource_allocation_estimate`, `energy_scope` = `estimated_resource_allocation`, `energy_basis` = the formula, `energy_low_j`, `energy_high_j`, `cpu_attribution` |
| Table B.6 (`validation/per_prompt_resources.csv`) | `cpu_percent` (machine) alongside `process_cpu_percent` (this workload) |
| Table B.1 | `energy_method`, `energy_scope` per condition |
| `manifest.json` | `settings.energy_estimator` — formula, budgets, all six assumptions, caveat |
| `dashboard_data.js` | `meta.energy_caveat`, `meta.energy_estimator`; the dashboard renders both |
| CLI | Banner before the run, caveat block after it |

---

## Commands

Check what the host supports first:

```bash
python -m bopis profile
```

On a host with no power telemetry this prints the three options and their
respective claims. To run in Mode C:

```bash
python -m bopis run \
  --backend llama-server --model-aware \
  --llama-binary /path/to/llama-server \
  --model Q4_K_M=/models/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  --energy-mode resource-estimate \
  --cpu-tdp-w 15 --cpu-idle-w 2.5 \
  --gpu-tdp-w 25 --gpu-idle-w 1.5 \
  --estimate-uncertainty 0.30 \
  --idle-seconds 60 \
  --out runs --label estimated
```

Replace the four power figures with the values for the host actually under
test — the defaults describe a 15 W U-series CPU and a 25 W MX330, and using
them on different hardware invalidates A2 immediately. Leaving the two
`--*-idle-w` values at their default of 0 makes each coefficient the full
nameplate figure, which is the conservative (largest) estimate.

Note that `--min-gpu-layers` is not needed in Mode C. It exists to keep Mode A's
GPU-only accounting from telling the optimizer that `g = 0` is free; the
estimator has a CPU term, so when work moves to the CPU the estimate follows it.

---

## Text for Chapter 3

A paste-ready statement of the limitation:

> Energy for this run was not measured. The GPU available for this study
> (NVIDIA GeForce MX330) implements neither `nvmlDeviceGetPowerUsage` nor
> `nvmlDeviceGetTotalEnergyConsumption`, so the board exposes no power signal to
> integrate and no energy counter to difference. Energy figures reported here
> are instead produced by a resource-allocation model:
> `E = [(P_cpu_tdp − P_cpu_idle)·u_cpu + (P_gpu_tdp − P_gpu_idle)·Δu_gpu]·T`,
> where `u_cpu` is the inference process's share of total CPU capacity measured
> from per-process CPU time, `Δu_gpu` is the GPU duty cycle net of a 60-second
> idle baseline, and the power budgets are the vendor's nameplate figures for
> the host. The model is linear in utilization and excludes DRAM, storage and
> display power; its six assumptions are enumerated in the run manifest under
> `settings.energy_estimator`. Because every configuration was evaluated on the
> same host, in the same session, against the same idle baseline, the
> systematic error is shared and the *ratios* between configurations — and
> hence the Energy Improvement Rate — are far better determined than any
> individual joule value. Absolute energy figures from this run are therefore
> not reported as measured quantities, and the study's energy claims are stated
> as relative improvements only. Reproducing this study with an absolute energy
> objective requires a GPU whose driver exposes NVML power telemetry; the
> software supports that path unchanged and prefers it automatically whenever it
> is available.

---

## Related

- `bopis/monitor/estimator.py` — the model, its assumptions, the uncertainty
  propagation, and the reasoning for each, in the module docstring
- `bopis/monitor/sampler.py` — the NVML ladder the estimator sits below, and
  the idle-baseline calibration
- `bopis/monitor/platform_os.py` — per-process CPU accounting on both platforms
- `tests/test_estimator.py` — the model's properties, asserted
