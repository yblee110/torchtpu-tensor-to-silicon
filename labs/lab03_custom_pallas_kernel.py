"""Lab 03: Custom TPU v6e Pallas Kernel (torch.tpu.pallas.jax_op) & Tile Sweep."""

import os
import time

os.environ.setdefault("TPU_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import torch
import torch.nn as nn


def _fused_add_rmsnorm_kernel(x_ref, res_ref, w_ref, o_ref):
    x_f32 = x_ref[...].astype(jnp.float32)
    res_f32 = res_ref[...].astype(jnp.float32)
    w_f32 = w_ref[...].astype(jnp.float32)

    h = x_f32 + res_f32
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + 1e-6)
    o_ref[...] = (h * rms * w_f32).astype(o_ref.dtype)


def make_pallas_rmsnorm_op(block_m: int = 256):
    def _jax_fn(x: jax.Array, residual: jax.Array, weight_2d: jax.Array) -> jax.Array:
        num_rows, hidden_dim = x.shape
        return pl.pallas_call(
            _fused_add_rmsnorm_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid=(num_rows // block_m,),
            in_specs=[
                pl.BlockSpec((block_m, hidden_dim), lambda i: (i, 0)),
                pl.BlockSpec((block_m, hidden_dim), lambda i: (i, 0)),
                pl.BlockSpec((1, hidden_dim), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((block_m, hidden_dim), lambda i: (i, 0)),
        )(x, residual, weight_2d)

    return torch.tpu.pallas.jax_op(f"pallas::fused_add_rmsnorm_bm{block_m}", _jax_fn)


def ref_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    h = x.float() + residual.float()
    rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return (h * rms * weight.float()).to(x.dtype)


class PallasTransformerLayer(nn.Module):
    def __init__(self, norm_op, hidden_dim: int = 2048, ffn_dim: int = 8192, use_pallas: bool = True):
        super().__init__()
        self.norm_op = norm_op
        self.use_pallas = use_pallas
        self.norm_weight = nn.Parameter(torch.ones(1, hidden_dim))
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.use_pallas:
            h = self.norm_op(x, residual, self.norm_weight)
        else:
            h = ref_add_rmsnorm(x, residual, self.norm_weight)
        return self.w2(torch.nn.functional.silu(self.w1(h)))


def bench_variant(label: str, fn, x_in: torch.Tensor, r_in: torch.Tensor, bytes_moved: int, iters: int = 30) -> dict:
    with torch.no_grad():
        for _ in range(5):
            _ = fn(x_in, r_in)
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(x_in, r_in)
        torch.accelerator.synchronize()
        dt = (time.perf_counter() - t0) / iters
    eff_gbps = (bytes_moved / dt) / 1e9
    print(f"  {label:<38} | {dt*1e3:7.3f} ms | Effective BW: {eff_gbps:6.1f} GB/s")
    return {"latency_ms": dt * 1e3, "effective_bw_gbs": eff_gbps}


def main():
    import json, pathlib
    device = torch.device("tpu")
    print("=" * 72)
    print(f"[Lab 03] Custom Pallas Kernels on {torch.tpu.get_device_properties(device).name}")
    print("=" * 72)

    n_tokens, hidden = 16384, 2048
    x_tpu = torch.randn(n_tokens, hidden, device=device, dtype=torch.bfloat16)
    res_tpu = torch.randn(n_tokens, hidden, device=device, dtype=torch.bfloat16)
    w_tpu = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    w_2d = w_tpu.unsqueeze(0)

    pallas_bm256 = make_pallas_rmsnorm_op(block_m=256)

    # 1. Numerical Parity
    out_ref = ref_add_rmsnorm(x_tpu, res_tpu, w_tpu)
    out_pallas = pallas_bm256(x_tpu, res_tpu, w_2d)
    torch.accelerator.synchronize()
    max_diff = (out_ref.float() - out_pallas.float()).abs().max().item()
    print(f"\n[1/3] Numerical Parity Check -> Max Abs Diff: {max_diff:.3e}")
    assert torch.allclose(out_ref, out_pallas, atol=2e-2, rtol=2e-2), "Parity check failed!"

    metrics = {"max_abs_diff": max_diff, "microbenchmark": {}, "tile_sweep": {}, "end_to_end": {}}

    # 2. Microbenchmark: Strict Eager vs torch.compile vs Custom Pallas
    norm_bytes = 3 * n_tokens * hidden * 2
    print("\n[2/3] Fused Add + RMSNorm Microbenchmark (v6e Peak HBM ~1,638 GB/s)")
    with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_NEVER):
        metrics["microbenchmark"]["strict_eager"] = bench_variant("Strict Eager (Unfused PyTorch)", lambda a, b: ref_add_rmsnorm(a, b, w_tpu), x_tpu, res_tpu, norm_bytes)
    metrics["microbenchmark"]["torch_compile"] = bench_variant("torch.compile (XLA Fused)", torch.compile(lambda a, b: ref_add_rmsnorm(a, b, w_tpu), backend="tpu"), x_tpu, res_tpu, norm_bytes)
    metrics["microbenchmark"]["pallas_bm256"] = bench_variant("Custom Pallas Kernel (block_m=256)", lambda a, b: pallas_bm256(a, b, w_2d), x_tpu, res_tpu, norm_bytes)

    print("\n      VMEM Tile Size Sweep (block_m):")
    for bm in [8, 64, 128, 256, 512]:
        op_bm = make_pallas_rmsnorm_op(block_m=bm)
        metrics["tile_sweep"][str(bm)] = bench_variant(f"  Pallas block_m={bm:<4} ({bm*hidden*2//1024:>4} KB/tile)", lambda a, b, f=op_bm: f(a, b, w_2d), x_tpu, res_tpu, norm_bytes)

    # 3. End-to-End nn.Module + torch.compile(backend="tpu") Integration
    print("\n[3/3] End-to-End Transformer Layer with torch.compile(backend='tpu')")
    ref_layer = PallasTransformerLayer(pallas_bm256, hidden, 8192, use_pallas=False).to(device=device, dtype=torch.bfloat16).eval()
    pallas_layer = PallasTransformerLayer(pallas_bm256, hidden, 8192, use_pallas=True).to(device=device, dtype=torch.bfloat16).eval()
    pallas_layer.load_state_dict(ref_layer.state_dict())

    layer_bytes = norm_bytes + 2 * 2 * hidden * 8192
    metrics["end_to_end"]["compiled_pytorch"] = bench_variant("Compiled Block (Standard PyTorch)", torch.compile(ref_layer, backend="tpu"), x_tpu, res_tpu, layer_bytes)
    metrics["end_to_end"]["compiled_pallas"] = bench_variant("Compiled Block (with Pallas jax_op)", torch.compile(pallas_layer, backend="tpu"), x_tpu, res_tpu, layer_bytes)

    out_dir = pathlib.Path("outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "lab03_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[JSON] Saved metrics -> {metrics_path}")
    print("✅ Lab 03 Done!")


if __name__ == "__main__":
    main()
