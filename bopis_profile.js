window.BOPIS_PROFILE = {
  "generated_utc": "2026-09-24T20:33:57.550240+00:00",
  "host_profile": {
    "cpu_model": "11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz",
    "physical_cores": 4,
    "logical_cores": 8,
    "ram_total_bytes": 16948453376,
    "ram_available_bytes": 1120456704,
    "is_wsl": false,
    "telemetry_source": "kernel32",
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce MX330",
    "vram_total_bytes": 2147483648,
    "vram_free_bytes": 2078732288,
    "compute_capability": "6.1",
    "driver_version": "528.96",
    "cuda_driver_version": "12.0",
    "power_supported": false,
    "energy_counter_supported": false,
    "energy_method": "unavailable",
    "gpu_error": null,
    "python_version": "3.11.6",
    "platform_name": "Windows",
    "rules_fired": [
      "HW-P1",
      "HW-G1",
      "HW-B2",
      "HW-C2"
    ],
    "permitted_precisions": [
      "Q8_0",
      "Q4_K_M"
    ],
    "permitted_gpu_layers": [
      0,
      14
    ],
    "permitted_batch_sizes": [
      1,
      2
    ],
    "permitted_cpu_threads": [
      2,
      4
    ],
    "vram_total_gib": 2.0,
    "ram_total_gib": 15.78
  },
  "configuration_space": {
    "n_feasible": 128,
    "n_rejected": 256,
    "rejected_by_rule": {
      "HW-P1": 64,
      "HW-P0": 192
    },
    "n_unconstrained": 2304,
    "precision_variants_present": [
      "F16",
      "Q8_0",
      "Q4_K_M"
    ],
    "models_present": [
      "qwen2.5-0.5b",
      "qwen2.5-1.5b"
    ],
    "model_precision_pairs": [
      "qwen2.5-0.5b:F16",
      "qwen2.5-0.5b:Q4_K_M",
      "qwen2.5-0.5b:Q8_0",
      "qwen2.5-1.5b:Q4_K_M",
      "qwen2.5-1.5b:Q8_0"
    ]
  },
  "rejections": [
    {
      "config": "qwen2.5-0.5b_t128_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t128_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t128_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t128_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t256_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t256_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t256_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t256_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t512_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t512_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t512_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t512_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t1024_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t1024_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t1024_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-0.5b_t1024_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t128_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t128_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t128_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t128_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t128_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t128_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t128_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t128_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t128_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t128_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t128_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t128_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t256_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t256_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t256_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t256_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t256_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t512_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t512_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t512_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t512_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t512_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.92 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.58 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 2.98 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-1.5b_t1024_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.63 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t128_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t128_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t128_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t256_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t256_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t256_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t512_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t512_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t512_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.83 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.13 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.81 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 5.90 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 3.20 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.87 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 1.88 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-3b_t1024_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "host share 1.06 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t128_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t128_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t128_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t128_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t256_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t256_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t256_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t512_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t512_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t512_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.30 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.65 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.88 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.39 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.25 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_F16_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_F16_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 14.41 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_F16_g14_c2",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_F16_g14_c4",
      "rule": "HW-P1",
      "detail": "F16 is not permitted for GPU offload on 2.00 GiB VRAM (g=14); at g=0 the weights stay in system RAM and this rule does not apply"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.76 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.99 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.50 GiB exceeds 1.04 GiB available system RAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "qwen2.5-7b_t1024_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.36 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    }
  ],
  "requirements": [
    {
      "model": "qwen2.5-0.5b",
      "precision": "F32",
      "searched": false,
      "weights_gib": 1.8402934074401855,
      "kv_cache_gib": 0.0234375,
      "cpu_only_ram_gib": 1.8637309074401855,
      "full_gpu_vram_gib": 1.8637309074401855,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": true,
      "shortfall_gib": 0.8202242851257324
    },
    {
      "model": "qwen2.5-0.5b",
      "precision": "F16",
      "searched": true,
      "weights_gib": 0.9201467037200928,
      "kv_cache_gib": 0.0234375,
      "cpu_only_ram_gib": 0.9435842037200928,
      "full_gpu_vram_gib": 0.9435842037200928,
      "cpu_only_verdict": "fits",
      "full_gpu_fits": true,
      "shortfall_gib": 0.0
    },
    {
      "model": "qwen2.5-0.5b",
      "precision": "Q8_0",
      "searched": true,
      "weights_gib": 0.4888279363512993,
      "kv_cache_gib": 0.0234375,
      "cpu_only_ram_gib": 0.5122654363512993,
      "full_gpu_vram_gib": 0.5122654363512993,
      "cpu_only_verdict": "fits",
      "full_gpu_fits": true,
      "shortfall_gib": 0.0
    },
    {
      "model": "qwen2.5-0.5b",
      "precision": "Q4_K_M",
      "searched": true,
      "weights_gib": 0.277769286185503,
      "kv_cache_gib": 0.0234375,
      "cpu_only_ram_gib": 0.301206786185503,
      "full_gpu_vram_gib": 0.301206786185503,
      "cpu_only_verdict": "fits",
      "full_gpu_fits": true,
      "shortfall_gib": 0.0
    },
    {
      "model": "qwen2.5-1.5b",
      "precision": "F32",
      "searched": false,
      "weights_gib": 5.736947059631348,
      "kv_cache_gib": 0.0546875,
      "cpu_only_ram_gib": 5.791634559631348,
      "full_gpu_vram_gib": 5.791634559631348,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 4.7481279373168945
    },
    {
      "model": "qwen2.5-1.5b",
      "precision": "F16",
      "searched": true,
      "weights_gib": 2.868473529815674,
      "kv_cache_gib": 0.0546875,
      "cpu_only_ram_gib": 2.923161029815674,
      "full_gpu_vram_gib": 2.923161029815674,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 1.8796544075012207
    },
    {
      "model": "qwen2.5-1.5b",
      "precision": "Q8_0",
      "searched": true,
      "weights_gib": 1.5238765627145767,
      "kv_cache_gib": 0.0546875,
      "cpu_only_ram_gib": 1.5785640627145767,
      "full_gpu_vram_gib": 1.5785640627145767,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": true,
      "shortfall_gib": 0.5350574404001236
    },
    {
      "model": "qwen2.5-1.5b",
      "precision": "Q4_K_M",
      "searched": true,
      "weights_gib": 0.8659204468131065,
      "kv_cache_gib": 0.0546875,
      "cpu_only_ram_gib": 0.9206079468131065,
      "full_gpu_vram_gib": 0.9206079468131065,
      "cpu_only_verdict": "fits",
      "full_gpu_fits": true,
      "shortfall_gib": 0.0
    },
    {
      "model": "qwen2.5-3b",
      "precision": "F32",
      "searched": false,
      "weights_gib": 11.511147022247314,
      "kv_cache_gib": 0.0703125,
      "cpu_only_ram_gib": 11.581459522247314,
      "full_gpu_vram_gib": 11.581459522247314,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 10.537952899932861
    },
    {
      "model": "qwen2.5-3b",
      "precision": "F16",
      "searched": true,
      "weights_gib": 5.755573511123657,
      "kv_cache_gib": 0.0703125,
      "cpu_only_ram_gib": 5.825886011123657,
      "full_gpu_vram_gib": 5.825886011123657,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 4.782379388809204
    },
    {
      "model": "qwen2.5-3b",
      "precision": "Q8_0",
      "searched": true,
      "weights_gib": 3.057648427784443,
      "kv_cache_gib": 0.0703125,
      "cpu_only_ram_gib": 3.127960927784443,
      "full_gpu_vram_gib": 3.127960927784443,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 2.0844543054699898
    },
    {
      "model": "qwen2.5-3b",
      "precision": "Q4_K_M",
      "searched": true,
      "weights_gib": 1.737463753670454,
      "kv_cache_gib": 0.0703125,
      "cpu_only_ram_gib": 1.807776253670454,
      "full_gpu_vram_gib": 1.807776253670454,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": true,
      "shortfall_gib": 0.7642696313560009
    },
    {
      "model": "qwen2.5-7b",
      "precision": "F32",
      "searched": false,
      "weights_gib": 28.386712074279785,
      "kv_cache_gib": 0.109375,
      "cpu_only_ram_gib": 28.496087074279785,
      "full_gpu_vram_gib": 28.496087074279785,
      "cpu_only_verdict": "needs more RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 27.452580451965332
    },
    {
      "model": "qwen2.5-7b",
      "precision": "F16",
      "searched": true,
      "weights_gib": 14.193356037139893,
      "kv_cache_gib": 0.109375,
      "cpu_only_ram_gib": 14.302731037139893,
      "full_gpu_vram_gib": 14.302731037139893,
      "cpu_only_verdict": "needs more RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 13.25922441482544
    },
    {
      "model": "qwen2.5-7b",
      "precision": "Q8_0",
      "searched": true,
      "weights_gib": 7.540220394730568,
      "kv_cache_gib": 0.109375,
      "cpu_only_ram_gib": 7.649595394730568,
      "full_gpu_vram_gib": 7.649595394730568,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 6.606088772416115
    },
    {
      "model": "qwen2.5-7b",
      "precision": "Q4_K_M",
      "searched": true,
      "weights_gib": 4.284619353711605,
      "kv_cache_gib": 0.109375,
      "cpu_only_ram_gib": 4.393994353711605,
      "full_gpu_vram_gib": 4.393994353711605,
      "cpu_only_verdict": "free RAM",
      "full_gpu_fits": false,
      "shortfall_gib": 3.350487731397152
    }
  ],
  "models": {
    "qwen2.5-0.5b": {
      "label": "Qwen2.5-0.5B-Instruct",
      "n_params": 494000000.0,
      "n_layers": 24,
      "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    },
    "qwen2.5-1.5b": {
      "label": "Qwen2.5-1.5B-Instruct",
      "n_params": 1540000000.0,
      "n_layers": 28,
      "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    },
    "qwen2.5-3b": {
      "label": "Qwen2.5-3B-Instruct",
      "n_params": 3090000000.0,
      "n_layers": 36,
      "hf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF"
    },
    "qwen2.5-7b": {
      "label": "Qwen2.5-7B-Instruct",
      "n_params": 7620000000.0,
      "n_layers": 28,
      "hf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF"
    }
  },
  "energy_estimate": {
    "mode": "C",
    "basis": "estimated_resource_allocation",
    "cpu_tdp_w": 15.0,
    "gpu_tdp_w": 25.0,
    "tariff_php_per_kwh": 14.35,
    "joules_per_kwh": 3600000.0,
    "power_supported": false,
    "energy_counter_supported": false,
    "caveat": "Mode C resource-allocation estimate. This GPU exposes no power sensor, so no joule figure here is measured. Valid for comparing configurations on this host; not valid as an absolute energy claim."
  },
  "cpu_package_power": {
    "available": false,
    "url": "http://127.0.0.1:8085/data.json",
    "reason": "No hardware-monitor sensor feed at http://127.0.0.1:8085/data.json.\n  1. Install LibreHardwareMonitor (maintained) or Open Hardware Monitor.\n  2. Run it AS ADMINISTRATOR -- RAPL is read through a kernel driver.\n  3. Options -> Remote Web Server -> Run (default port 8085).\n  4. Confirm the CPU node lists Powers -> CPU Package, then retry.\n  Check with:  python -m bopis profile --hwmon-url http://127.0.0.1:8085/data.json\n  (underlying error: <urlopen error timed out>)"
  }
};
