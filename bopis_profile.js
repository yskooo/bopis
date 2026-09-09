window.BOPIS_PROFILE = {
  "generated_utc": "2026-09-09T14:03:52.506917+00:00",
  "host_profile": {
    "cpu_model": "11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz",
    "physical_cores": 4,
    "logical_cores": 8,
    "ram_total_bytes": 16948453376,
    "ram_available_bytes": 2289934336,
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
    "n_feasible": 0,
    "n_rejected": 64,
    "rejected_by_rule": {
      "HW-P0": 64
    },
    "n_unconstrained": 768,
    "precision_variants_present": []
  },
  "rejections": [
    {
      "config": "t128_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t128_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t128_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t128_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t128_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t128_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t128_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t128_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t128_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t256_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t256_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t512_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t512_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b1_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b1_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.42 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b1_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b1_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.39 GiB (Q8_0, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b1_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b1_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.33 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b1_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b1_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.03 GiB (Q4_K_M, g=14, kv b=1) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b2_Q8_0_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b2_Q8_0_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 7.67 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b2_Q8_0_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b2_Q8_0_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 3.64 GiB (Q8_0, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b2_Q4_K_M_g0_c2",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b2_Q4_K_M_g0_c4",
      "rule": "HW-P0",
      "detail": "host share 4.58 GiB exceeds 2.13 GiB available system RAM"
    },
    {
      "config": "t1024_b2_Q4_K_M_g14_c2",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    },
    {
      "config": "t1024_b2_Q4_K_M_g14_c4",
      "rule": "HW-P0",
      "detail": "GPU share 2.28 GiB (Q4_K_M, g=14, kv b=2) exceeds 1.94 GiB available VRAM"
    }
  ]
};
