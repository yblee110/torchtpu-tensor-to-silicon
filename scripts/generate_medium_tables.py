"""Generate publication-grade high-DPI table PNGs for Medium from measured Cloud TPU v6e benchmarks.

Requires Playwright with Chromium installed (used to render HTML/CSS at 2.5x retina DPI).
"""

import pathlib
import shutil
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path("medium/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CSS = """
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
body {
  background: #FFFFFF;
  font-family: 'Noto Sans', 'Ubuntu Sans', 'DejaVu Sans', sans-serif;
  color: #0F172A;
  padding: 20px;
  display: inline-block;
}
.card {
  width: 920px;
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-top: 4px solid #2563EB;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
}
.card-header {
  padding: 18px 22px 14px 22px;
  background: #FFFFFF;
  border-bottom: 1px solid #E2E8F0;
}
.eyebrow {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #2563EB;
  background: #EFF6FF;
  padding: 3px 9px;
  border-radius: 999px;
  margin-bottom: 7px;
}
.title {
  font-size: 18px;
  font-weight: 700;
  color: #0F172A;
  line-height: 1.3;
}
.subtitle {
  font-size: 12.5px;
  color: #64748B;
  margin-top: 4px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13.5px;
  line-height: 1.45;
}
thead th {
  background: #F8FAFC;
  color: #334155;
  font-weight: 700;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 12px 16px;
  border-bottom: 1.5px solid #CBD5E1;
  text-align: left;
  white-space: nowrap;
}
thead th.num {
  text-align: right;
}
tbody td {
  padding: 12px 16px;
  border-bottom: 1px solid #E2E8F0;
  color: #1E293B;
  vertical-align: middle;
}
tbody tr:last-child td {
  border-bottom: none;
}
tbody tr:nth-child(even):not(.winner):not(.highlight-blue):not(.section-row) {
  background: #FAFBFD;
}
tbody td.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
tbody tr.winner {
  background: #ECFDF5;
}
tbody tr.winner td {
  color: #064E3B;
  border-top: 1px solid #A7F3D0;
  border-bottom: 1px solid #A7F3D0;
}
tbody tr.winner td:first-child {
  box-shadow: inset 4px 0 0 #059669;
}
tbody tr.highlight-blue {
  background: #EFF6FF;
}
tbody tr.highlight-blue td:first-child {
  box-shadow: inset 4px 0 0 #2563EB;
}
tbody tr.section-row td {
  background: #F1F5F9;
  color: #334155;
  font-size: 11.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 8px 16px;
  border-top: 1px solid #CBD5E1;
  border-bottom: 1px solid #CBD5E1;
}
code {
  font-family: 'Noto Sans Mono', 'DejaVu Sans Mono', monospace;
  font-size: 12px;
  background: #F1F5F9;
  color: #0F172A;
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid #E2E8F0;
  white-space: nowrap;
}
thead th code {
  text-transform: none;
  letter-spacing: 0;
  font-weight: 600;
}
tr.winner code {
  background: #D1FAE5;
  border-color: #A7F3D0;
  color: #064E3B;
}
tr.highlight-blue code {
  background: #DBEAFE;
  border-color: #BFDBFE;
  color: #1E3A8A;
}
.badge {
  display: inline-block;
  font-size: 11.5px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  white-space: nowrap;
}
.badge-green {
  background: #059669;
  color: #FFFFFF;
}
.badge-blue {
  background: #2563EB;
  color: #FFFFFF;
}
.badge-amber {
  background: #FEF3C7;
  color: #92400E;
  border: 1px solid #FDE68A;
}
.badge-red {
  background: #FEE2E2;
  color: #991B1B;
}
.muted {
  color: #64748B;
  font-size: 12px;
}
.bar-wrap {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  justify-content: flex-end;
}
.bar-bg {
  width: 68px;
  height: 7px;
  background: #E2E8F0;
  border-radius: 999px;
  overflow: hidden;
  display: inline-block;
}
.bar-fill {
  display: block;
  height: 100%;
  border-radius: 999px;
}
.card-footer {
  padding: 10px 22px;
  background: #F8FAFC;
  border-top: 1px solid #E2E8F0;
  font-size: 11.5px;
  color: #64748B;
  display: flex;
  justify-content: space-between;
}
.col-pallas {
  background: rgba(239, 246, 255, 0.45);
}
thead th.col-pallas-hdr {
  background: #EFF6FF;
  color: #1D4ED8;
  border-bottom-color: #93C5FD;
}
"""

TABLES = [
    {
        "filename": "table1_execution_modes.png",
        "eyebrow": "Lab 01 · Execution Modes",
        "title": "TorchTPU Execution Modes on Cloud TPU v6e (RMSNorm + SwiGLU)",
        "subtitle": "Workload: tokens = 8,192, hidden = 4,096, intermediate = 11,008 (bfloat16) · Single-chip ct6e-standard-1t",
        "footer_left": "Measured with LD_PRELOAD=libtcmalloc_minimal.so.4 · Peak BF16 = 918 TFLOPS",
        "footer_right": "torch==2.15.0 · torch-tpu==0.3.0",
        "html": """
        <table>
          <thead>
            <tr>
              <th>Execution Mode</th>
              <th class="num">Latency (ms)</th>
              <th class="num">Achieved TFLOPS</th>
              <th class="num">Speedup vs. Strict</th>
              <th class="num">MFU (% of 918 TF/s)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Debug Eager</strong> <span class="muted">(<code>LAUNCH_BLOCKING=1</code>)</span></td>
              <td class="num">5.375</td>
              <td class="num">153.42</td>
              <td class="num"><span class="badge badge-red">0.67×</span></td>
              <td class="num">
                <div class="bar-wrap">
                  <span>16.7%</span>
                  <span class="bar-bg"><span class="bar-fill" style="width:16.7%; background:#DC2626;"></span></span>
                </div>
              </td>
            </tr>
            <tr>
              <td><strong>Strict Eager</strong> <span class="muted">(<code>DEFER_NEVER</code>)</span></td>
              <td class="num">3.616</td>
              <td class="num">228.07</td>
              <td class="num">1.00×</td>
              <td class="num">
                <div class="bar-wrap">
                  <span>24.8%</span>
                  <span class="bar-bg"><span class="bar-fill" style="width:24.8%; background:#D97706;"></span></span>
                </div>
              </td>
            </tr>
            <tr class="highlight-blue">
              <td><strong>DeferAndFuse</strong> <span class="muted">(<code>DEFER_AND_FUSE=1</code>)</span></td>
              <td class="num"><strong>1.413</strong></td>
              <td class="num"><strong>583.54</strong></td>
              <td class="num"><span class="badge badge-blue">2.56×</span></td>
              <td class="num">
                <div class="bar-wrap">
                  <strong>63.6%</strong>
                  <span class="bar-bg"><span class="bar-fill" style="width:63.6%; background:#2563EB;"></span></span>
                </div>
              </td>
            </tr>
            <tr class="winner">
              <td><strong><code>torch.compile(backend="tpu")</code></strong></td>
              <td class="num"><strong>1.248</strong></td>
              <td class="num"><strong>660.52</strong></td>
              <td class="num"><span class="badge badge-green">2.90×</span></td>
              <td class="num">
                <div class="bar-wrap">
                  <strong>72.0%</strong>
                  <span class="bar-bg"><span class="bar-fill" style="width:72.0%; background:#059669;"></span></span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        """,
    },
    {
        "filename": "table2_xprof_metrics.png",
        "eyebrow": "Lab 01 · XProf Hardware Trace",
        "title": "XProf v2.24.2 Trace Breakdown Across Execution Modes (6 Active Steps)",
        "subtitle": "Extracted directly from captured .xplane.pb device and host traces on Cloud TPU v6e",
        "footer_left": "Framework Op Placement: 39.1% Device (strict_eager) → 92.9% Device (torch_compile)",
        "footer_right": "xprof==2.24.2 · libtpu==0.0.53.1",
        "html": """
        <table>
          <thead>
            <tr>
              <th>Profiled Mode (6 Steps)</th>
              <th class="num">Total Events</th>
              <th class="num">Host <code>cpu_op</code></th>
              <th class="num">Device Ops</th>
              <th>Representative Compiled Executables</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>strict_eager</code></td>
              <td class="num">5,012</td>
              <td class="num">4,732</td>
              <td class="num">186</td>
              <td><code>tt_jit_pow</code>, <code>tt_jit_mean</code>, <code>tt_jit_add</code>, <code>tt_jit_rsqrt</code>, <code>tt_jit_linear</code> <span class="muted">(each separate!)</span></td>
            </tr>
            <tr class="highlight-blue">
              <td><code>defer_and_fuse</code></td>
              <td class="num">2,424</td>
              <td class="num"><strong>2,206</strong> <span class="badge badge-blue">-53%</span></td>
              <td class="num">120</td>
              <td>Fused <code>tt_jit_add</code> + <code>%multiply_reduce_fusion</code> + async <code>S(1)</code> VMEM <code>%copy-start</code></td>
            </tr>
            <tr class="winner">
              <td><strong><code>torch_compile</code></strong></td>
              <td class="num"><strong>1,960</strong></td>
              <td class="num"><strong>1,762</strong> <span class="badge badge-green">-63%</span></td>
              <td class="num"><strong>108</strong></td>
              <td>Single <code>lab01_execution_modes_L30_split0_fwd</code> <strong>(1 launch/step)</strong> + async <code>S(1)</code> DMA</td>
            </tr>
          </tbody>
        </table>
        """,
    },
    {
        "filename": "table3_bounded_dynamism.png",
        "eyebrow": "Lab 02 · Bounded Dynamism",
        "title": "Growing KV-Cache Decode Loop: Static Shapes vs. 128-Lane Bucketed Compilation",
        "subtitle": "Autoregressive decode steps (seq_len = 17 → 24) with torch.compile(backend='tpu') on TPU v6e",
        "footer_left": "dynamic=None pads internally to 128-lane hardware buckets — zero recompilations inside the bucket",
        "footer_right": "~240× steady-state speedup",
        "html": """
        <table>
          <thead>
            <tr>
              <th>Decode Step (<code>seq_len</code>)</th>
              <th class="num"><code>dynamic=False</code> (ms)</th>
              <th class="num"><code>dynamic=None</code> (ms)</th>
              <th>What Happened Under the Hood</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Step 1 (<code>seq=17</code>)</td>
              <td class="num">724.18</td>
              <td class="num">81.23</td>
              <td>Initial static compile</td>
            </tr>
            <tr>
              <td>Step 2 (<code>seq=18</code>)</td>
              <td class="num">126.93</td>
              <td class="num">385.37</td>
              <td>Dynamo detects changing dim → compiles <code>[17, 128]</code> bucket</td>
            </tr>
            <tr>
              <td>Step 3 (<code>seq=19</code>)</td>
              <td class="num">126.05</td>
              <td class="num">96.78</td>
              <td>Compiles steady-state XLA padded output layout</td>
            </tr>
            <tr class="winner">
              <td><strong>Steps 4–8 (<code>seq=20..24</code>)</strong></td>
              <td class="num"><span class="badge badge-red">126.5 – 129.7</span></td>
              <td class="num"><strong>0.53 – 0.55</strong></td>
              <td><span class="badge badge-green">~240× faster!</span> &nbsp;<strong>Zero recompilations</strong> inside the 128&#8209;lane bucket</td>
            </tr>
          </tbody>
        </table>
        """,
    },
    {
        "filename": "table4_mxu_alignment_precision.png",
        "eyebrow": "Lab 02 · Silicon Alignment & Precision",
        "title": "Cloud TPU v6e 256×256 MXU Alignment & Numerical Precision Trade-offs",
        "subtitle": "Benchmarking dimension alignment (bf16 GEMM) and 1-pass / 3-pass / 6-pass MXU precision (4096×4096 Matmul)",
        "footer_left": "TPU v6e upgrades MXUs from 128×128 (v5e) to 256×256 · Max Abs Error measured against FP64 reference",
        "footer_right": "Peak BF16 = 918 TFLOPS",
        "html": """
        <table>
          <thead>
            <tr>
              <th>Experiment / Mode</th>
              <th class="num">Latency (ms)</th>
              <th class="num">Effective TFLOPS</th>
              <th class="num">Max Abs Error (vs. FP64)</th>
            </tr>
          </thead>
          <tbody>
            <tr class="section-row">
              <td colspan="4">Part A · MXU Dimension Alignment (Native bfloat16 GEMM)</td>
            </tr>
            <tr>
              <td><code>3968</code> <span class="muted">(<code>128 × 31</code> — v5e-aligned, odd 128-multiple)</span></td>
              <td class="num">0.180</td>
              <td class="num">692.8 <span class="muted">(75.5% MFU)</span></td>
              <td class="num"><span class="muted">—</span></td>
            </tr>
            <tr>
              <td><code>3997 × 3999 × 4001</code> <span class="muted">(Unaligned prime dims)</span></td>
              <td class="num">0.174</td>
              <td class="num">735.0 <span class="muted">(80.1% MFU)</span></td>
              <td class="num"><span class="muted">—</span></td>
            </tr>
            <tr class="winner">
              <td><strong><code>4096</code> <span class="muted">(<code>256 × 16</code> — v6e native MXU-aligned)</span></strong></td>
              <td class="num"><strong>0.179</strong></td>
              <td class="num"><strong>766.7</strong> <span class="badge badge-green">83.5% MFU</span></td>
              <td class="num"><span class="muted">—</span></td>
            </tr>
            <tr class="section-row">
              <td colspan="4">Part B · Numerical Precision (4096×4096 Matmul · 1-Pass vs. 3-Pass vs. 6-Pass MXU)</td>
            </tr>
            <tr class="highlight-blue">
              <td><strong>Native <code>bfloat16</code></strong> <span class="muted">(1-pass BF16 MXU)</span></td>
              <td class="num"><strong>0.285</strong></td>
              <td class="num"><strong>481.4</strong></td>
              <td class="num"><code>1.44e+00</code></td>
            </tr>
            <tr>
              <td><code>FP32 Precision.DEFAULT</code> <span class="muted">(1-pass BF16 truncation)</span></td>
              <td class="num">0.665</td>
              <td class="num">206.6</td>
              <td class="num"><code>8.15e-01</code></td>
            </tr>
            <tr class="winner">
              <td><strong><code>FP32 Precision.HIGH</code></strong> <span class="muted">(3-pass BF16 accumulation)</span></td>
              <td class="num"><strong>0.757</strong></td>
              <td class="num"><strong>181.5</strong></td>
              <td class="num"><code>4.41e-03</code> <span class="badge badge-green">185× cleaner</span></td>
            </tr>
            <tr>
              <td><code>FP32 Precision.HIGHEST</code> <span class="muted">(6-pass FP32 emulation)</span></td>
              <td class="num">1.210</td>
              <td class="num">113.6</td>
              <td class="num"><code>1.37e-04</code> <span class="badge badge-amber">4.2× slower</span></td>
            </tr>
          </tbody>
        </table>
        """,
    },
    {
        "filename": "table5_triton_vs_pallas.png",
        "eyebrow": "Lab 03 · Kernel Mental Model",
        "title": "GPU Triton vs. TPU Pallas + TorchTPU: Architectural Mental Model Shift",
        "subtitle": "Translating custom GPU kernel intuitions into structured DMA + VMEM pipelining on Cloud TPU v6e",
        "footer_left": "Pallas custom ops register natively into PyTorch via torch.tpu.pallas.jax_op with zero graph breaks",
        "footer_right": "SIMT Pointer Math → Structured 2D Tile DMA",
        "html": """
        <table>
          <thead>
            <tr>
              <th style="width:17%;">Concept</th>
              <th style="width:37%;">GPU Triton (<code>@triton.jit</code>)</th>
              <th class="col-pallas-hdr" style="width:46%;">TPU Pallas + TorchTPU (<code>torch.tpu.pallas.jax_op</code>)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Execution Grid</strong></td>
              <td><code>tl.program_id(axis)</code> <span class="muted">(SIMT thread blocks)</span></td>
              <td class="col-pallas"><code>grid=(...)</code> + <code>index_map</code> <span class="muted">(TensorCore tile pipeline)</span></td>
            </tr>
            <tr>
              <td><strong>Memory Movement</strong></td>
              <td>Arbitrary 1D pointer arithmetic<br><code>tl.load(ptr + offs, mask)</code></td>
              <td class="col-pallas"><strong>Structured DMA prefetch</strong> (HBM → <strong>VMEM</strong>)<br><code>BlockSpec(block_shape, index_map)</code></td>
            </tr>
            <tr>
              <td><strong>Compute Model</strong></td>
              <td>Per-thread registers / Tensor Cores</td>
              <td class="col-pallas">Explicit 2D tile loads (<code>x_ref[...]</code>) on <strong>VPU</strong> (<code>fp32</code>) &amp; <strong>MXU</strong> (<code>256×256</code> <code>bf16</code>)</td>
            </tr>
            <tr>
              <td><strong>Alignment Rules</strong></td>
              <td>Powers of 2 recommended</td>
              <td class="col-pallas">Last two tile dims divisible by <strong><code>(8, 128)</code></strong> (or full axis); <strong><code>256</code></strong> ideal on v6e</td>
            </tr>
            <tr>
              <td><strong>PyTorch Hook</strong></td>
              <td>Direct Python wrapper</td>
              <td class="col-pallas"><code>torch.tpu.pallas.jax_op("pallas::op", fn)</code> — works in Eager &amp; <code>torch.compile</code>!</td>
            </tr>
          </tbody>
        </table>
        """,
    },
    {
        "filename": "table6_pallas_hbm_tile_sweep.png",
        "eyebrow": "Lab 03 · Custom Pallas Kernel & VMEM Tile Sweep",
        "title": "Fused Residual Add + RMSNorm on Cloud TPU v6e (16,384 × 2,048 BF16)",
        "subtitle": "HBM bandwidth saturation across execution modes and VMEM block_m tile sizes (Peak HBM BW = 1,638 GB/s)",
        "footer_left": "End-to-end compiled nn.Module (Add+RMSNorm → Linear → SiLU → Linear): 1.665 ms → 1.594 ms",
        "footer_right": "11.8× vs. Strict Eager · 1.53× vs. XLA",
        "html": """
        <table>
          <thead>
            <tr>
              <th>Variant (<code>16,384 × 2,048</code> BF16)</th>
              <th class="num">Latency (ms)</th>
              <th class="num">Effective HBM BW</th>
              <th class="num">% of v6e Peak (1,638 GB/s)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Strict Eager</strong> <span class="muted">(Unfused PyTorch)</span></td>
              <td class="num">1.789</td>
              <td class="num">112.5 GB/s</td>
              <td class="num">
                <div class="bar-wrap">
                  <span>6.9%</span>
                  <span class="bar-bg"><span class="bar-fill" style="width:6.9%; background:#DC2626;"></span></span>
                </div>
              </td>
            </tr>
            <tr class="highlight-blue">
              <td><strong><code>torch.compile</code></strong> <span class="muted">(XLA Auto-Fused)</span></td>
              <td class="num"><strong>0.231</strong></td>
              <td class="num"><strong>872.8 GB/s</strong></td>
              <td class="num">
                <div class="bar-wrap">
                  <strong>53.3%</strong>
                  <span class="bar-bg"><span class="bar-fill" style="width:53.3%; background:#2563EB;"></span></span>
                </div>
              </td>
            </tr>
            <tr>
              <td>Custom Pallas (<code>block_m=8</code>, <span class="muted">32 KB/tile</span>)</td>
              <td class="num">0.931</td>
              <td class="num">216.3 GB/s</td>
              <td class="num">
                <div class="bar-wrap">
                  <span>13.2%</span>
                  <span class="bar-bg"><span class="bar-fill" style="width:13.2%; background:#94A3B8;"></span></span>
                </div>
              </td>
            </tr>
            <tr>
              <td>Custom Pallas (<code>block_m=64</code>, <span class="muted">256 KB/tile</span>)</td>
              <td class="num">0.227</td>
              <td class="num">888.2 GB/s</td>
              <td class="num">
                <div class="bar-wrap">
                  <span>54.2%</span>
                  <span class="bar-bg"><span class="bar-fill" style="width:54.2%; background:#3B82F6;"></span></span>
                </div>
              </td>
            </tr>
            <tr>
              <td>Custom Pallas (<code>block_m=128</code>, <span class="muted">512 KB/tile</span>)</td>
              <td class="num">0.178</td>
              <td class="num">1,130.8 GB/s</td>
              <td class="num">
                <div class="bar-wrap">
                  <span>69.0%</span>
                  <span class="bar-bg"><span class="bar-fill" style="width:69.0%; background:#10B981;"></span></span>
                </div>
              </td>
            </tr>
            <tr class="winner">
              <td><strong>Custom Pallas (<code>block_m=256</code>, <span class="muted">1,024 KB/tile</span>)</strong> &nbsp;<span class="badge badge-green">1.53× vs. XLA</span></td>
              <td class="num"><strong>0.151</strong></td>
              <td class="num"><strong>1,336.1 GB/s</strong></td>
              <td class="num">
                <div class="bar-wrap">
                  <strong>81.6%</strong>
                  <span class="bar-bg"><span class="bar-fill" style="width:81.6%; background:#059669;"></span></span>
                </div>
              </td>
            </tr>
            <tr>
              <td>Custom Pallas (<code>block_m=512</code>, <span class="muted">2,048 KB/tile</span>)</td>
              <td class="num">0.159</td>
              <td class="num">1,267.0 GB/s</td>
              <td class="num">
                <div class="bar-wrap">
                  <span>77.4%</span>
                  <span class="bar-bg"><span class="bar-fill" style="width:77.4%; background:#10B981;"></span></span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        """,
    },
]


def render_html(spec: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>{BASE_CSS}</style>
</head>
<body>
  <div class="card">
    <div class="card-header">
      <div class="eyebrow">{spec['eyebrow']}</div>
      <div class="title">{spec['title']}</div>
      <div class="subtitle">{spec['subtitle']}</div>
    </div>
    {spec['html']}
    <div class="card-footer">
      <span>{spec['footer_left']}</span>
      <span>{spec['footer_right']}</span>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    chrome_bin = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    with sync_playwright() as p:
        launch_kwargs = {"executable_path": chrome_bin} if chrome_bin else {}
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2.5)
        for spec in TABLES:
            page.set_content(render_html(spec), wait_until="load")
            body = page.locator("body")
            out_path = OUT_DIR / spec["filename"]
            body.screenshot(path=str(out_path))
            print(f"[Saved] {out_path}")
        browser.close()


if __name__ == "__main__":
    main()
