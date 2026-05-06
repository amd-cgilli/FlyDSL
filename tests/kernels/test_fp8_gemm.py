"""
Download the aiter dsv3 a8w8 bpreshuffle tuned GEMM CSV, extract the
gfx950/flydsl/fp8-e4m3fn baseline rows, and produce an HTML report
comparing TFLOPS across multiple providers (baseline, FlyDSL kernel,
torch._scaled_mm, etc.).
"""

import argparse
import csv
import html as html_mod
import http.server
import io
import json
import os
import re
import subprocess
import sys
import webbrowser

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from tests.kernels.bench_fp8_gemm import bench_gemm, get_torch_scaled_mm_perf

CSV_URL = (
    "https://raw.githubusercontent.com/ROCm/aiter/main/"
    "aiter/configs/model_configs/dsv3_a8w8_bpreshuffle_tuned_gemm.csv"
)

FILTER_GFX = "gfx950"
FILTER_LIBTYPE = "flydsl"
FILTER_DTYPE = "torch.float8_e4m3fn"

COLORS = [
    ("99, 166, 255", "blue"),
    ("63, 185, 80", "green"),
    ("210, 153, 34", "orange"),
    ("188, 140, 255", "purple"),
    ("255, 107, 107", "red"),
    ("58, 211, 199", "teal"),
    ("255, 166, 201", "pink"),
    ("136, 136, 136", "gray"),
]


def download_csv(url: str) -> list[dict]:
    result = subprocess.run(
        ["curl", "-sL", url], capture_output=True, text=True, check=True,
    )
    reader = csv.DictReader(io.StringIO(result.stdout), skipinitialspace=True)
    return list(reader)


def filter_rows(
    rows: list[dict],
    gfx: str = FILTER_GFX,
    libtype: str = FILTER_LIBTYPE,
    dtype: str = FILTER_DTYPE,
) -> list[dict]:
    out = []
    for r in rows:
        if (
            r.get("gfx", "").strip() == gfx
            and r.get("libtype", "").strip() == libtype
            and r.get("q_dtype_w", "").strip() == dtype
        ):
            out.append(r)
    return out


def extract_tile(kernel_name: str) -> str | None:
    """Return the first AxBxC or AxBxCxD segment from the kernel name."""
    m = re.search(r"(\d+x\d+x\d+(?:x\d+)?)", kernel_name)
    return m.group(1) if m else None


def parse_entries(rows: list[dict]) -> list[dict]:
    entries = []
    for r in rows:
        m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
        tflops = float(r["tflops"])
        tile = extract_tile(r.get("kernelName", ""))
        tile_parts = [int(v) for v in tile.split("x")] if tile else []
        entries.append({
            "M": m, "N": n, "K": k,
            "shape": f"{m}x{n}x{k}",
            "tile": tile or "?",
            "tile_M": tile_parts[0] if len(tile_parts) > 0 else None,
            "tile_N": tile_parts[1] if len(tile_parts) > 1 else None,
            "tile_K": tile_parts[2] if len(tile_parts) > 2 else None,
            "tflops": tflops,
        })
    entries.sort(key=lambda e: (e["M"], e["N"], e["K"]))
    return entries


def _print_speedup(label: str, tflops: float, base_tf: float):
    speedup = tflops / base_tf if base_tf > 0 else float("inf")
    if speedup >= 1.0:
        arrow = f"\033[32m▲ {speedup:.2f}x\033[0m"
    else:
        arrow = f"\033[31m▼ {speedup:.2f}x\033[0m"
    print(f"  {label}: {tflops:.2f} vs {base_tf:.2f} TFLOPS {arrow}")


def run_benchmarks(baseline: list[dict]) -> dict[str, list[dict]]:
    flydsl_results = []
    torch_results = []

    for entry in baseline:
        shape = entry["shape"]
        m, n, k = entry["M"], entry["N"], entry["K"]
        base_tf = entry["tflops"]
        print(f"Benchmarking {shape}...")

        res = bench_gemm(m, n, k)
        if res:
            tflops, tile = res
            _print_speedup("FlyDSL 4-Wave", tflops, base_tf)
            flydsl_results.append({
                "M": m, "N": n, "K": k,
                "shape": shape,
                "tile": f"{tile[0]}x{tile[1]}x{tile[2]}",
                "tflops": tflops,
            })
        else:
            print("  FlyDSL 4-Wave: SKIP")

        try:
            torch_tf = get_torch_scaled_mm_perf(m, n, k)
            _print_speedup("torch.scaled_mm", torch_tf, base_tf)
            torch_results.append({
                "M": m, "N": n, "K": k,
                "shape": shape,
                "tflops": torch_tf,
            })
        except Exception as e:
            print(f"  torch.scaled_mm: SKIP ({e})")

    providers = {}
    if flydsl_results:
        providers["FlyDSL 4-Wave"] = flydsl_results
    if torch_results:
        providers["torch.scaled_mm"] = torch_results
    return providers


def generate_html(providers: dict[str, list[dict]]) -> str:
    names = list(providers.keys())
    ref_name = names[0]

    all_shapes = []
    seen = set()
    for entries in providers.values():
        for e in entries:
            if e["shape"] not in seen:
                all_shapes.append(e["shape"])
                seen.add(e["shape"])

    shape_to_m = {}
    for entries in providers.values():
        for e in entries:
            shape_to_m[e["shape"]] = e["M"]

    def sort_key(s):
        return tuple(int(x) for x in s.split("x"))

    all_shapes.sort(key=sort_key)

    shape_data = {}
    for name, entries in providers.items():
        for e in entries:
            shape_data.setdefault(e["shape"], {})[name] = e

    # --- Performance table (Shape | TFLOPS per provider | speedup vs baseline) ---
    perf_col = 0
    perf_header = f"<th data-col='{perf_col}' data-type='shape'>Shape</th>"
    perf_col += 1
    for name in names:
        perf_header += f"<th data-col='{perf_col}' data-type='num'>{html_mod.escape(name)} TFLOPS</th>"
        perf_col += 1
    for name in names[1:]:
        perf_header += f"<th data-col='{perf_col}' data-type='num'>{html_mod.escape(name)} vs {html_mod.escape(ref_name)}</th>"
        perf_col += 1

    perf_rows = []
    for shape in all_shapes:
        data = shape_data.get(shape, {})
        ref_tf = data.get(ref_name, {}).get("tflops")
        row = f"<td>{html_mod.escape(shape)}</td>"
        for name in names:
            tf = data.get(name, {}).get("tflops")
            row += f"<td class='num'>{tf:.2f}</td>" if tf is not None else "<td class='num'>—</td>"
        for name in names[1:]:
            tf = data.get(name, {}).get("tflops")
            if tf is not None and ref_tf is not None and ref_tf > 0:
                sp = tf / ref_tf
                cls = "win" if sp >= 1.0 else "loss"
                row += f"<td class='num {cls}'>{sp:.2f}x</td>"
            else:
                row += "<td class='num'>—</td>"
        perf_rows.append(f"<tr>{row}</tr>")

    # --- Tile config table (Shape | Tile + TFLOPS for each provider that has tiles) ---
    tile_names = [n for n in names if any(e.get("tile") for e in providers[n])]

    tile_header = ""
    tile_rows_html = ""
    if tile_names:
        tc = 0
        tile_header = f"<th data-col='{tc}' data-type='shape'>Shape</th>"
        tc += 1
        for name in tile_names:
            tile_header += f"<th data-col='{tc}' data-type='str'>{html_mod.escape(name)} Tile</th>"
            tc += 1
            tile_header += f"<th data-col='{tc}' data-type='num'>TFLOPS</th>"
            tc += 1

        t_rows = []
        for shape in all_shapes:
            data = shape_data.get(shape, {})
            has_any = any(data.get(n, {}).get("tile") for n in tile_names)
            if not has_any:
                continue
            row = f"<td>{html_mod.escape(shape)}</td>"
            for name in tile_names:
                e = data.get(name, {})
                tile = e.get("tile")
                tf = e.get("tflops")
                row += f"<td>{html_mod.escape(tile)}</td>" if tile else "<td>—</td>"
                row += f"<td class='num'>{tf:.2f}</td>" if tf is not None else "<td class='num'>—</td>"
            t_rows.append(f"<tr>{row}</tr>")
        tile_rows_html = "".join(t_rows)

    provider_data_js = {}
    for name in names:
        tflops_by_shape = {}
        for e in providers[name]:
            tflops_by_shape[e["shape"]] = e["tflops"]
        provider_data_js[name] = [tflops_by_shape.get(s) for s in all_shapes]

    title_parts = " vs ".join(names)
    subtitle = f"gfx950 · flydsl · float8_e4m3fn · {' / '.join(names)}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FP8 GEMM — {html_mod.escape(title_parts)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 24px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; color: #e6edf3; }}
  .subtitle {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 20px; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  canvas {{ width: 100% !important; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  th {{ background: #1c2129; color: #e6edf3; padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; position: sticky; top: 0; cursor: pointer; user-select: none; }}
  th:hover {{ background: #272d37; }}
  th.sort-asc::after {{ content: " ▲"; }}
  th.sort-desc::after {{ content: " ▼"; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1c2129; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .win {{ color: #3fb950; font-weight: 600; }}
  .loss {{ color: #f85149; font-weight: 600; }}
  .controls {{ margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls label {{ font-size: 0.82rem; color: #8b949e; }}
  .controls select, .controls input {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 4px 8px; font-size: 0.82rem; }}
</style>
</head>
<body>

<h1>FP8 GEMM Performance</h1>
<p class="subtitle">{html_mod.escape(subtitle)}</p>

<div class="controls">
  <label>Group by M: <select id="mFilter"><option value="all">All</option></select></label>
  <label>Min M: <input id="minM" type="number" value="0" style="width:80px"></label>
</div>

<div class="chart-container">
  <canvas id="barChart" height="500"></canvas>
</div>

<h2 style="margin: 24px 0 8px; font-size: 1.1rem; color: #e6edf3;">Performance</h2>
<table id="perfTable">
<thead><tr>{perf_header}</tr></thead>
<tbody>{"".join(perf_rows)}</tbody>
</table>

{"<h2 style='margin: 24px 0 8px; font-size: 1.1rem; color: #e6edf3;'>Tile Configuration</h2>" if tile_names else ""}
{"<table id='tileTable'><thead><tr>" + tile_header + "</tr></thead><tbody>" + tile_rows_html + "</tbody></table>" if tile_names else ""}

<script>
const ALL_SHAPES = {json.dumps(all_shapes)};
const PROVIDERS = {json.dumps(names)};
const PROVIDER_DATA = {json.dumps(provider_data_js)};
const COLORS = {json.dumps([c[0] for c in COLORS[:len(names)]])};

const mValues = [...new Set(ALL_SHAPES.map(l => parseInt(l.split('x')[0])))].sort((a,b) => a-b);
const mSel = document.getElementById('mFilter');
mValues.forEach(m => {{ const o = document.createElement('option'); o.value = m; o.text = 'M=' + m; mSel.appendChild(o); }});

let chart;

function buildChart() {{
  const minM = parseInt(document.getElementById('minM').value) || 0;
  const mVal = mSel.value;

  const idx = [];
  ALL_SHAPES.forEach((l, i) => {{
    const m = parseInt(l.split('x')[0]);
    if (m < minM) return;
    if (mVal !== 'all' && m !== parseInt(mVal)) return;
    idx.push(i);
  }});

  const labels = idx.map(i => ALL_SHAPES[i]);
  const datasets = PROVIDERS.map((name, pi) => ({{
    label: name,
    data: idx.map(i => PROVIDER_DATA[name][i]),
    backgroundColor: `rgba(${{COLORS[pi % COLORS.length]}}, 0.7)`,
    borderColor: `rgba(${{COLORS[pi % COLORS.length]}}, 1)`,
    borderWidth: 1,
  }}));

  const h = Math.max(400, labels.length * (8 + PROVIDERS.length * 14));
  const canvas = document.getElementById('barChart');
  canvas.style.height = h + 'px';

  if (chart) chart.destroy();
  chart = new Chart(canvas, {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ color: '#c9d1d9' }} }},
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'TFLOPS', color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: '#21262d' }},
        }},
        y: {{
          ticks: {{ color: '#c9d1d9', font: {{ size: 11 }} }},
          grid: {{ color: '#21262d' }},
        }},
      }},
    }},
  }});
}}

mSel.addEventListener('change', buildChart);
document.getElementById('minM').addEventListener('change', buildChart);
buildChart();

document.querySelectorAll('#perfTable th, #tileTable th').forEach(th => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const col = parseInt(th.dataset.col);
    const type = th.dataset.type;
    const asc = !th.classList.contains('sort-asc');

    table.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');

    rows.sort((a, b) => {{
      let va = a.children[col]?.textContent.trim() || '';
      let vb = b.children[col]?.textContent.trim() || '';
      if (type === 'num') {{
        va = parseFloat(va) || 0;
        vb = parseFloat(vb) || 0;
        return asc ? va - vb : vb - va;
      }}
      if (type === 'shape') {{
        const pa = va.split('x').map(Number);
        const pb = vb.split('x').map(Number);
        for (let i = 0; i < Math.max(pa.length, pb.length); i++) {{
          if ((pa[i]||0) !== (pb[i]||0)) return asc ? (pa[i]||0) - (pb[i]||0) : (pb[i]||0) - (pa[i]||0);
        }}
        return 0;
      }}
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>"""


def serve_html(path: str, port: int):
    directory = os.path.dirname(os.path.abspath(path))
    filename = os.path.basename(path)
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=directory, **kw)
    server = http.server.HTTPServer(("", port), handler)
    url = f"http://localhost:{port}/{filename}"
    print(f"Serving at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="FP8 GEMM baseline perf viewer")
    parser.add_argument("--gfx", default=FILTER_GFX, help=f"GPU arch filter (default: {FILTER_GFX})")
    parser.add_argument("--libtype", default=FILTER_LIBTYPE, help=f"Library type filter (default: {FILTER_LIBTYPE})")
    parser.add_argument("--dtype", default=FILTER_DTYPE, help=f"Dtype filter (default: {FILTER_DTYPE})")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output HTML path (default: fp8_gemm_perf.html)")
    parser.add_argument("--serve", action="store_true", help="Start a local HTTP server and open the report in a browser")
    parser.add_argument("--port", type=int, default=8888, help="Port for --serve (default: 8888)")
    args = parser.parse_args()

    print("Downloading baseline CSV from aiter ...")
    all_rows = download_csv(CSV_URL)
    print(f"  {len(all_rows)} total rows")

    filtered = filter_rows(all_rows, gfx=args.gfx, libtype=args.libtype, dtype=args.dtype)
    print(f"  {len(filtered)} rows matching gfx={args.gfx} libtype={args.libtype} dtype={args.dtype}")

    if not filtered:
        print("\nNo matching rows found. Available filter values:")
        gfxs = sorted(set(r.get("gfx", "").strip() for r in all_rows))
        libs = sorted(set(r.get("libtype", "").strip() for r in all_rows))
        dtypes = sorted(set(r.get("q_dtype_w", "").strip() for r in all_rows))
        print(f"  gfx:     {', '.join(gfxs)}")
        print(f"  libtype: {', '.join(libs)}")
        print(f"  dtype:   {', '.join(dtypes)}")
        sys.exit(0)

    baseline = parse_entries(filtered)

    providers = {"Baseline (aiter)": baseline}
    bench_providers = run_benchmarks(baseline)
    providers.update(bench_providers)

    out_path = args.output or os.path.join(os.path.dirname(__file__), "fp8_gemm_perf.html")
    html_content = generate_html(providers)
    with open(out_path, "w") as f:
        f.write(html_content)
    print(f"\nReport written to {out_path}")

    if args.serve:
        serve_html(out_path, args.port)


if __name__ == "__main__":
    main()
