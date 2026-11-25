"""Generate publication-ready high-DPI charts for the Medium article from measured TPU v6e JSONs."""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = pathlib.Path("results_from_vm/outputs")
FIG_DIR = pathlib.Path("medium/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Clean modern publication style
plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#2563EB"
RED = "#DC2626"
AMBER = "#D97706"
GREEN = "#059669"
SLATE = "#334155"
MUTED = "#64748B"


def make_fig1():
    d = json.loads((RESULTS_DIR / "lab01_metrics.json").read_text())
    peak_tf = d["peak_tflops_bf16"]
    peak_bw = d["peak_hbm_bw_gbs"]
    ridge = d["ridge_point_flops_per_byte"]
    modes = d["modes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Left: Roofline
    ai_x = np.logspace(0, 4.2, 500)
    roof_y = np.minimum(peak_tf, (ai_x * peak_bw) / 1e3)
    ax1.loglog(ai_x, roof_y, color="#0F172A", lw=2.3, label=f"TPU v6e Ceiling ({peak_tf:.0f} TF/s, {peak_bw:.0f} GB/s)")
    ax1.axvline(ridge, color=MUTED, ls=":", lw=1.4, label=f"Ridge Point ({ridge:.0f} FLOPs/B)")
    ax1.axvspan(1.0, ridge, color="#FEF2F2", alpha=0.45, zorder=0)
    ax1.axvspan(ridge, 10**4.2, color="#ECFDF5", alpha=0.35, zorder=0)
    ax1.text(12, 1.6, "HBM Memory-Bound", color="#991B1B", fontsize=9, fontweight="bold")
    ax1.text(850, 1.6, "MXU Compute-Bound", color="#065F46", fontsize=9, fontweight="bold")

    style_map = {
        "2. Debug Eager": (AMBER, "Debug Eager\n(153.4 TF/s)", (-112, -32)),
        "1. Strict Eager": (RED, "Strict Eager\n(228.1 TF/s)", (-118, 2)),
        "3. DeferAndFuse": (BLUE, "DeferAndFuse\n(583.5 TF/s)", (-122, 18)),
        "4. torch.compile": (GREEN, "torch.compile\n(660.5 TF/s)", (-42, -68)),
    }
    for key, (color, text, offset) in style_map.items():
        m = modes[key]
        ax1.scatter([m["arithmetic_intensity"]], [m["achieved_tflops"]], color=color, s=110, zorder=5, edgecolors="white", linewidth=1.2)
        ax1.annotate(
            text,
            xy=(m["arithmetic_intensity"], m["achieved_tflops"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.8,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0, shrinkA=2, shrinkB=6),
        )

    ax1.set_xlim(1, 10**4.2)
    ax1.set_ylim(1, 2200)
    ax1.set_xlabel("Effective Arithmetic Intensity (FLOPs / Byte)")
    ax1.set_ylabel("Achieved Throughput (TFLOPS)")
    ax1.set_title("A. TPU v6e (Trillium) Roofline Trajectory")
    ax1.grid(True, which="major", alpha=0.7)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.92)

    # Right: Bar comparison
    order = ["2. Debug Eager", "1. Strict Eager", "3. DeferAndFuse", "4. torch.compile"]
    labels = ["Debug Eager\n(Blocking)", "Strict Eager\n(Default)", "DeferAndFuse\n(Opt. Eager)", "torch.compile\n(backend='tpu')"]
    colors = [AMBER, RED, BLUE, GREEN]
    tflops_vals = [modes[k]["achieved_tflops"] for k in order]
    strict_ms = modes["1. Strict Eager"]["latency_ms"]
    speedups = [strict_ms / modes[k]["latency_ms"] for k in order]

    bars = ax2.bar(labels, tflops_vals, color=colors, width=0.55, edgecolor="white", linewidth=1.0)
    ax2.axhline(peak_tf, color="#0F172A", ls="--", lw=1.2, alpha=0.7)
    ax2.text(3.45, peak_tf + 18, f"Peak: {peak_tf:.0f} TF/s", ha="right", fontsize=8.5, color="#0F172A", fontweight="bold")

    for bar, tf, sp in zip(bars, tflops_vals, speedups):
        mfu = (tf / peak_tf) * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            tf + 16,
            f"{tf:.1f} TF/s\n({sp:.2f}x | {mfu:.1f}% MFU)",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=SLATE,
        )

    ax2.set_xlim(-0.62, 3.62)
    ax2.set_ylim(0, 1030)
    ax2.set_ylabel("Achieved Throughput (TFLOPS)")
    ax2.set_title("B. RMSNorm + SwiGLU Throughput (tokens=8,192)")
    ax2.grid(True, axis="y", alpha=0.7)

    plt.tight_layout(pad=1.6)
    out = FIG_DIR / "fig1_execution_modes_roofline.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def make_fig2():
    d = json.loads((RESULTS_DIR / "lab02_metrics.json").read_text())
    static_ms = d["dynamic_shapes_ms"]["dynamic_false"]
    dyn_ms = d["dynamic_shapes_ms"]["dynamic_none"]
    steps = list(range(1, len(static_ms) + 1))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(steps, static_ms, marker="o", lw=2.3, ms=7, color=RED, label="dynamic=False (Recompiles every new seq_len)")
    ax.plot(steps, dyn_ms, marker="s", lw=2.3, ms=7, color=GREEN, label="dynamic=None (128-lane Bounded Dynamism)")

    ax.axvspan(3.6, 8.3, color="#ECFDF5", alpha=0.55, zorder=0)
    ax.text(5.95, 1.8, "Steady-State Cached Bucket [17, 128]\n~0.53–0.55 ms/step (~240x faster)", ha="center", va="bottom",
            fontsize=9.2, fontweight="bold", color="#065F46",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#10B981", alpha=0.95))

    ax.annotate("Static guard fails every step\n(~127 ms XLA recompile)", xy=(6, static_ms[5]), xytext=(4.3, 280),
                fontsize=8.8, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.annotate("Warmup + padded\nlayout compile", xy=(2, dyn_ms[1]), xytext=(2.25, 520),
                fontsize=8.3, color="#047857",
                arrowprops=dict(arrowstyle="->", color="#047857", lw=1.0))

    ax.set_yscale("log")
    ax.set_ylim(0.25, 1500)
    ax.set_xlim(0.7, 8.3)
    ax.set_xticks(steps)
    ax.set_xticklabels([f"Step {s}\n(seq={16+s})" for s in steps])
    ax.set_ylabel("Step Latency (ms, log scale)")
    ax.set_title("Autoregressive KV-Cache Growth: Avoiding Recompilations with Bounded Dynamism")
    ax.grid(True, which="major", alpha=0.7)
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white")

    plt.tight_layout()
    out = FIG_DIR / "fig2_bounded_dynamism.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def make_fig3():
    d = json.loads((RESULTS_DIR / "lab02_metrics.json").read_text())
    align = d["mxu_alignment_tflops"]
    prec = d["precision_modes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # Left: MXU Alignment
    keys = [
        "128-Aligned (3968x3968x3968)",
        "Unaligned   (3997x3999x4001)",
        "256-Aligned (4096x4096x4096)",
    ]
    labels = ["128-Aligned (Odd)\n3968×3968×3968\n(128 × 31)", "Unaligned\n3997×3999×4001", "256-Aligned (v6e MXU)\n4096×4096×4096\n(256 × 16)"]
    vals = [align[k]["effective_tflops"] for k in keys]
    colors = [AMBER, RED, GREEN]

    bars = ax1.bar(labels, vals, color=colors, width=0.52, edgecolor="white")
    for bar, v in zip(bars, vals):
        mfu = (v / 918.0) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v:.1f} TF/s\n({mfu:.1f}% MFU)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=SLATE)
    ax1.set_ylim(550, 845)
    ax1.set_ylabel("Effective BF16 Throughput (TFLOPS)")
    ax1.set_title("A. TPU v6e 256×256 MXU Boundary Alignment")
    ax1.grid(True, axis="y", alpha=0.7)

    # Right: Precision Latency vs Max Error
    p_order = [
        ("Native bfloat16 (1-pass)", "Native bf16\n(1-pass)", GREEN, (8, -34)),
        ("FP32 Precision.DEFAULT (1p)", "FP32 DEFAULT\n(1-pass)", AMBER, (12, 4)),
        ("FP32 Precision.HIGH    (3p)", "FP32 HIGH\n(3-pass)", BLUE, (12, -8)),
        ("FP32 Precision.HIGHEST (6p)", "FP32 HIGHEST\n(6-pass)", RED, (-115, 16)),
    ]
    lats = [prec[k]["latency_ms"] for k, *_ in p_order]
    errs = [prec[k]["max_abs_err"] for k, *_ in p_order]

    ax2.plot(lats, errs, color="#94A3B8", lw=1.5, ls="--", zorder=2)
    for (k, lbl, col, offs), x, y in zip(p_order, lats, errs):
        ax2.scatter([x], [y], color=col, s=115, zorder=5, edgecolors="white", lw=1.2)
        ax2.annotate(f"{lbl}\n({x:.2f} ms, err={y:.2e})", xy=(x, y), xytext=offs,
                     textcoords="offset points", fontsize=8.4, fontweight="bold", color=col)

    ax2.set_yscale("log")
    ax2.set_xlim(0.18, 1.45)
    ax2.set_ylim(4e-5, 6.0)
    ax2.set_xlabel("Matmul Latency for 4096×4096 (ms) — Lower is Faster")
    ax2.set_ylabel("Max Abs Error vs. FP64 Reference (log) — Lower is Better")
    ax2.set_title("B. MXU Precision Modes: Speed vs. Accuracy Pareto")
    ax2.grid(True, which="major", alpha=0.7)

    plt.tight_layout(pad=1.6)
    out = FIG_DIR / "fig3_mxu_alignment_and_precision.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def make_fig4():
    d = json.loads((RESULTS_DIR / "lab03_metrics.json").read_text())
    micro = d["microbenchmark"]
    sweep = d["tile_sweep"]
    peak_bw = 1638.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # Left: Effective HBM Bandwidth comparison
    labels = ["Strict Eager\n(Unfused PyTorch)", "torch.compile\n(XLA Fused)", "Custom Pallas\n(block_m=256)"]
    bws = [
        micro["strict_eager"]["effective_bw_gbs"],
        micro["torch_compile"]["effective_bw_gbs"],
        sweep["256"]["effective_bw_gbs"],
    ]
    lats = [
        micro["strict_eager"]["latency_ms"],
        micro["torch_compile"]["latency_ms"],
        sweep["256"]["latency_ms"],
    ]
    colors = [RED, BLUE, GREEN]
    bars = ax1.bar(labels, bws, color=colors, width=0.52, edgecolor="white")
    ax1.axhline(peak_bw, color="#0F172A", ls="--", lw=1.3)
    ax1.text(2.35, peak_bw + 28, f"TPU v6e HBM Peak: {peak_bw:.0f} GB/s", ha="right", fontsize=8.8, fontweight="bold", color="#0F172A")

    for bar, bw, lat in zip(bars, bws, lats):
        pct = (bw / peak_bw) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bw + 28, f"{bw:.1f} GB/s\n({pct:.1f}% Peak | {lat:.3f} ms)",
                 ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=SLATE)
    ax1.set_ylim(0, 1850)
    ax1.set_ylabel("Effective HBM Bandwidth (GB/s)")
    ax1.set_title("A. Fused Add + RMSNorm (16,384 × 2,048 BF16)")
    ax1.grid(True, axis="y", alpha=0.7)

    # Right: VMEM Tile Size Sweep
    bms = [8, 64, 128, 256, 512]
    x_pos = np.arange(len(bms))
    sweep_bws = [sweep[str(bm)]["effective_bw_gbs"] for bm in bms]
    tile_kbs = [bm * 2048 * 2 // 1024 for bm in bms]

    ax2.plot(x_pos, sweep_bws, marker="o", lw=2.4, ms=8.5, color=BLUE, zorder=4)
    ax2.scatter([3], [sweep_bws[3]], color=GREEN, s=160, zorder=5, edgecolors="white", lw=1.5)
    ax2.axhline(peak_bw, color="#0F172A", ls="--", lw=1.2, alpha=0.75)

    for idx, (bw, kb) in enumerate(zip(sweep_bws, tile_kbs)):
        dy = 45 if idx != 3 else 55
        col = GREEN if idx == 3 else SLATE
        tag = f"{bw:.1f} GB/s" + (" (Sweet Spot)" if idx == 3 else "")
        ax2.text(idx, bw + dy, tag, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=col)

    ax2.annotate("Too small: DMA &\nloop overhead bound", xy=(0, sweep_bws[0]), xytext=(0.45, 420),
                 fontsize=8.3, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"bm={bm}\n({kb} KB)" for bm, kb in zip(bms, tile_kbs)])
    ax2.set_ylim(0, 1800)
    ax2.set_xlabel("Pallas VMEM Tile Height (block_m & Tile Size per Input)")
    ax2.set_ylabel("Effective HBM Bandwidth (GB/s)")
    ax2.set_title("B. Pallas VMEM Tile Size Sweep on TPU v6e")
    ax2.grid(True, alpha=0.7)

    plt.tight_layout(pad=1.6)
    out = FIG_DIR / "fig4_pallas_kernel_and_tile_sweep.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
