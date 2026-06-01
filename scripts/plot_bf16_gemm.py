#!/usr/bin/env python3
"""Bar chart comparing bf16 GEMM TFLOPS: custom kernel vs current FlyDSL best."""

import matplotlib.pyplot as plt
import numpy as np

sizes = ["1K", "2K", "4K", "8K", "16K"]
custom = [75.08, 319.85, 1107.20, 1333.87, 1281.72] # Custom FlyDSL kernel with 32x16 blocking
best = [297.52, 726.72, 997.45, 1157.21, 1084.10] # Numbers taken from AITER GemmTuner

x = np.arange(len(sizes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, custom, width, label="FlyDSL Custom", color="#4c72b0")
bars2 = ax.bar(x + width / 2, best, width, label="Current FlyDSL Best", color="#dd8452")

ax.set_xlabel("M = N = K")
ax.set_ylabel("TFLOPS")
ax.set_title("BF16 GEMM Performance — MI355X (gfx950)")
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()

ax.bar_label(bars1, fmt="%.0f", padding=3, fontsize=8)
ax.bar_label(bars2, fmt="%.0f", padding=3, fontsize=8)

ax.set_ylim(0, max(max(custom), max(best)) * 1.15)
fig.tight_layout()
fig.savefig("bf16_gemm_comparison.png", dpi=150)
print("Saved bf16_gemm_comparison.png")
plt.show()
