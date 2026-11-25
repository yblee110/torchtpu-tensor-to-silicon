"""Lab 02: Bounded Dynamism, Tensor Contiguity, v6e MXU (256x256) Alignment & Precision."""

import os
import time

os.environ.setdefault("TPU_VISIBLE_DEVICES", "0")

import torch
from torch._functorch._aot_autograd import autograd_cache


def decode_kv_step(q: torch.Tensor, kv_cache: torch.Tensor, new_kv: torch.Tensor):
    updated_kv = torch.cat([kv_cache, new_kv], dim=2)
    scores = torch.matmul(q, updated_kv.transpose(-2, -1))
    attn_out = torch.matmul(torch.softmax(scores.float(), dim=-1).to(q.dtype), updated_kv)
    return attn_out, updated_kv


def run_growing_loop(compiled_fn, device: torch.device, steps: int = 8) -> list[float]:
    q = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, 8, 16, 128, dtype=torch.bfloat16, device=device)
    new_kv = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device=device)
    torch.accelerator.synchronize()

    times_ms = []
    for _ in range(steps):
        t0 = time.perf_counter()
        _, kv = compiled_fn(q, kv, new_kv)
        torch.accelerator.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)
    return times_ms


def main():
    device = torch.device("tpu")
    print("=" * 68)
    print(f"[Lab 02] Device: {torch.tpu.get_device_properties(device).name}")
    print("=" * 68)

    # 1. Dynamic Shapes: dynamic=False vs dynamic=None (128-lane Bounded Dynamism)
    print("\n[1/4] Dynamic Shapes: Growing KV-Cache (dynamic=False vs dynamic=None)")
    torch._dynamo.reset()
    autograd_cache.AOTAutogradCache.clear()
    torch._dynamo.config.recompile_limit = 64
    static_ms = run_growing_loop(torch.compile(decode_kv_step, backend="tpu", dynamic=False), device)

    torch._dynamo.reset()
    autograd_cache.AOTAutogradCache.clear()
    dynamic_ms = run_growing_loop(torch.compile(decode_kv_step, backend="tpu", dynamic=None), device)

    for idx, (s_ms, d_ms) in enumerate(zip(static_ms, dynamic_ms), 1):
        print(f"  Step {idx:2d} (seq={16+idx:2d}) | dynamic=False: {s_ms:7.2f} ms | dynamic=None (128-bucket): {d_ms:7.2f} ms")

    # 2. Contiguity at torch.compile Boundaries
    print("\n[2/4] Tensor Contiguity vs Non-Contiguous View Input")

    @torch.compile(backend="tpu")
    def compiled_mlp(x_in: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x_in @ w)

    base = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    non_contig = base.t()
    contig = non_contig.contiguous()
    weight = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

    contiguity_results = {}
    for label, tensor_in in [("Non-contiguous view (.t())", non_contig), ("Contiguous tensor (.contiguous())", contig)]:
        for _ in range(5):
            _ = compiled_mlp(tensor_in, weight)
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = compiled_mlp(tensor_in, weight)
        torch.accelerator.synchronize()
        ms = (time.perf_counter() - t0) / 30 * 1e3
        contiguity_results[label] = ms
        print(f"  {label:<35} -> {ms:6.3f} ms")

    # 3. TPU v6e 256x256 MXU Alignment
    print("\n[3/4] TPU v6e MXU (256x256) & Register (8x128) Tile Alignment")

    @torch.compile(backend="tpu")
    def matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    shapes = [
        ("Unaligned   (3997x3999x4001)", 3997, 3999, 4001),
        ("128-Aligned (3968x3968x3968)", 3968, 3968, 3968),
        ("256-Aligned (4096x4096x4096)", 4096, 4096, 4096),
    ]
    alignment_results = {}
    for label, m, k, n in shapes:
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        for _ in range(5):
            _ = matmul_op(a, b)
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = matmul_op(a, b)
        torch.accelerator.synchronize()
        dt = (time.perf_counter() - t0) / 30
        tflops = (2 * m * k * n / dt) / 1e12
        alignment_results[label.strip()] = {"latency_ms": dt * 1e3, "effective_tflops": tflops}
        print(f"  {label} -> {dt*1e3:6.3f} ms | Effective {tflops:6.1f} TFLOPS")

    # 4. bfloat16 & MXU Multi-Pass Precision Modes
    print("\n[4/4] bfloat16 vs float32 MXU Multi-Pass Precision (4096x4096)")
    m = k = n = 4096
    a_cpu = torch.randn(m, k, dtype=torch.float32)
    b_cpu = torch.randn(k, n, dtype=torch.float32)
    ref_fp64 = (a_cpu.double() @ b_cpu.double()).float()

    a_fp32, b_fp32 = a_cpu.to(device), b_cpu.to(device)
    a_bf16, b_bf16 = a_fp32.to(torch.bfloat16), b_fp32.to(torch.bfloat16)

    precision_results = {}

    def eval_prec(run_fn, label: str):
        for _ in range(5):
            out = run_fn()
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(25):
            out = run_fn()
        torch.accelerator.synchronize()
        dt = (time.perf_counter() - t0) / 25
        max_err = (out.float().cpu() - ref_fp64).abs().max().item()
        tflops = (2 * m * k * n / dt) / 1e12
        precision_results[label.strip()] = {"latency_ms": dt * 1e3, "tflops": tflops, "max_abs_err": max_err}
        print(f"  {label:<28} | {dt*1e3:6.3f} ms | {tflops:6.1f} TFLOPS | Max Err: {max_err:.3e}")

    eval_prec(lambda: torch.matmul(a_bf16, b_bf16), "Native bfloat16 (1-pass)")
    for mode_name, prec in [
        ("FP32 Precision.DEFAULT (1p)", torch.tpu.Precision.DEFAULT),
        ("FP32 Precision.HIGH    (3p)", torch.tpu.Precision.HIGH),
        ("FP32 Precision.HIGHEST (6p)", torch.tpu.Precision.HIGHEST),
    ]:
        def _runner(p=prec):
            with torch.tpu.precision(p):
                return torch.matmul(a_fp32, b_fp32)
        eval_prec(_runner, mode_name)

    import json, pathlib
    out_dir = pathlib.Path("outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dynamic_shapes_ms": {"dynamic_false": static_ms, "dynamic_none": dynamic_ms},
        "contiguity_ms": contiguity_results,
        "mxu_alignment_tflops": alignment_results,
        "precision_modes": precision_results,
    }
    metrics_path = out_dir / "lab02_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[JSON] Saved metrics -> {metrics_path}")
    print("✅ Lab 02 Done!")


if __name__ == "__main__":
    main()
