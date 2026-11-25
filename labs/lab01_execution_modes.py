"""Lab 01: TPU v6e Roofline, TorchTPU 4 Execution Modes, and XProf Trace Capture."""

import os
import pathlib
import time

os.environ.setdefault("TPU_VISIBLE_DEVICES", "0")
os.makedirs("/dev/shm/torch_tpu_cache/sprint_cache", exist_ok=True)
os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE", "sprint_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.tpu.profiler import TpuProfilerConfig


class RMSNormSwiGLUBlock(nn.Module):
    def __init__(self, hidden_dim: int = 2048, ffn_dim: int = 8192):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype)
        h = x * rms * self.weight
        activated = F.silu(self.w_gate(h)) * self.w_up(h)
        return x + self.w_down(activated)


def benchmark_fn(fn, x_in: torch.Tensor, warmup: int = 5, iters: int = 20) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(x_in)
        torch.accelerator.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(x_in)
        torch.accelerator.synchronize()
        return (time.perf_counter() - t0) / iters


def save_roofline_plot(points: dict, peak_tflops: float, peak_bw_gbs: float, dev_name: str, out_path: pathlib.Path):
    ridge = (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)
    ai_x = np.logspace(-1, 4, 500)
    roof_y = np.minimum(peak_tflops, (ai_x * peak_bw_gbs) / 1e3)

    plt.figure(figsize=(8, 5))
    plt.loglog(ai_x, roof_y, "k-", linewidth=2, label=f"{dev_name} Ceiling ({peak_tflops:.0f} TF/s, {peak_bw_gbs:.0f} GB/s)")
    plt.axvline(ridge, color="gray", linestyle="--", alpha=0.6, label=f"Ridge Point ({ridge:.0f} FLOPs/B)")

    for label, (ai, tflops, _) in points.items():
        plt.scatter([ai], [tflops], s=90, zorder=5, label=f"{label} ({tflops:.1f} TF/s)")
        plt.annotate(label, (ai, tflops), textcoords="offset points", xytext=(6, 6), fontsize=9)

    plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
    plt.ylabel("Performance (TFLOPS)")
    plt.title(f"TorchTPU 4 Execution Modes on {dev_name} Roofline")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def capture_xprof(base_dir: pathlib.Path, run_name: str, fn, x_in: torch.Tensor, eager_mode=None):
    out_dir = base_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tpu_cfg = TpuProfilerConfig(
        host_tracer_level=2,
        device_tracer_level=1,
        python_tracer_level=0,
        run_dir=out_dir,
    )
    ctx = torch.tpu.set_eager_mode(eager_mode) if eager_mode is not None else torch.no_grad()
    with ctx, torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.TPU],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            experimental_config=tpu_cfg,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
        ) as prof:
            for _ in range(10):
                _ = fn(x_in)
                torch.accelerator.synchronize()
                prof.step()
    print(f"  [XProf] Saved trace -> {out_dir}")


def main():
    device = torch.device("tpu")
    props = torch.tpu.get_device_properties(device)

    hbm_bw_map = {"v6e": 1638.0, "v5p": 2765.0, "v5e": 819.0}
    peak_bw = next((bw for k, bw in hbm_bw_map.items() if k in props.name.lower()), 1638.0)
    peak_tflops = getattr(props, "bf16_flops_per_device", 918e12) / 1e12
    hbm_gib = getattr(props, "hbm_bytes_per_device", 32 * 1024**3) / (1024**3)
    ridge = (peak_tflops * 1e12) / (peak_bw * 1e9)

    print("=" * 64)
    print(f"[Lab 01] Hardware Summary: {props.name} | PyTorch {torch.__version__}")
    print(f"  HBM Capacity      : {hbm_gib:.2f} GiB")
    print(f"  Peak BF16 Compute : {peak_tflops:.1f} TFLOPS")
    print(f"  Peak HBM Bandwidth: {peak_bw:.1f} GB/s")
    print(f"  Ridge Point       : {ridge:.1f} FLOPs/Byte")
    print("=" * 64)

    B, S, D, FFN = 16, 512, 2048, 8192
    tokens = B * S
    model = RMSNormSwiGLUBlock(D, FFN).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(B, S, D, device=device, dtype=torch.bfloat16).contiguous()

    total_flops = 3 * 2 * tokens * D * FFN
    fused_bytes = 2 * (tokens * D * 2 + 3 * D * FFN)
    unfused_bytes = fused_bytes + 2 * (tokens * D * 6 + tokens * FFN * 5)

    results = {}

    with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_NEVER):
        dt = benchmark_fn(model, x)
    results["1. Strict Eager"] = (total_flops / unfused_bytes, (total_flops / dt) / 1e12, dt * 1e3)

    with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING):
        dt = benchmark_fn(model, x, warmup=2, iters=5)
    results["2. Debug Eager"] = (total_flops / unfused_bytes, (total_flops / dt) / 1e12, dt * 1e3)

    with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
        dt = benchmark_fn(model, x)
    results["3. DeferAndFuse"] = (total_flops / ((fused_bytes + unfused_bytes) * 0.5), (total_flops / dt) / 1e12, dt * 1e3)

    compiled_model = torch.compile(model, backend="tpu")
    dt = benchmark_fn(compiled_model, x)
    results["4. torch.compile"] = (total_flops / fused_bytes, (total_flops / dt) / 1e12, dt * 1e3)

    base_ms = results["1. Strict Eager"][2]
    print(f"\n{'Execution Mode':<20} | {'Latency (ms)':>12} | {'Achieved TFLOPS':>15} | {'Speedup vs Strict':>17}")
    print("-" * 72)
    for name, (_, tflops, ms) in results.items():
        print(f"{name:<20} | {ms:>12.3f} | {tflops:>15.2f} | {base_ms / ms:>16.2f}x")

    out_dir = pathlib.Path("outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    metrics_path = out_dir / "lab01_metrics.json"
    metrics_path.write_text(json.dumps({
        "device": props.name,
        "peak_tflops_bf16": peak_tflops,
        "peak_hbm_bw_gbs": peak_bw,
        "ridge_point_flops_per_byte": ridge,
        "modes": {
            k: {"arithmetic_intensity": v[0], "achieved_tflops": v[1], "latency_ms": v[2]}
            for k, v in results.items()
        },
    }, indent=2))
    print(f"\n[JSON] Saved metrics       -> {metrics_path}")

    plot_path = out_dir / "01_roofline_execution_modes.png"
    save_roofline_plot(results, peak_tflops, peak_bw, props.name, plot_path)
    print(f"[Plot] Saved Roofline chart -> {plot_path}")

    print("\nCapturing XProf traces for Strict Eager, DeferAndFuse, and torch.compile...")
    xprof_dir = out_dir / "xprof_modes"
    capture_xprof(xprof_dir, "strict_eager", model, x, eager_mode=torch.tpu.EagerMode.DEFER_NEVER)
    capture_xprof(xprof_dir, "defer_and_fuse", model, x, eager_mode=torch.tpu.EagerMode.DEFER_AND_FUSE)
    capture_xprof(xprof_dir, "torch_compile", compiled_model, x, eager_mode=None)
    print(f"\n✅ Lab 01 Done! View traces with: xprof --logdir={xprof_dir} --port=6006")


if __name__ == "__main__":
    main()
