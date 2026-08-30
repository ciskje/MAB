#!/usr/bin/env python3
# baseline.py — FASE 0: baseline di rendering + frame di riferimento per il gate
# di correttezza delle fasi successive (v5.0.0).
#
# Uso:   python baseline.py [path/mandel.py]
#        (default: mandel.py nella stessa cartella dello script)
# Output: nel repo:  baseline.txt  +  baseline/<zona>_<backend>.npy
#
# Misure (960x540, mediana di 3 run dopo 1 warmup, palette fuoco, mi=auto_mi):
#   GPU : kernel (perf_counter+sync; CuPy 14 non ha Event.elapsed_time) |
#         D2H (out.get) | PIL fromarray | totale compute_gpu
#   CPU : totale compute_cpu | costo allocazioni+prefiltro (stimato)
import importlib.util
import os
import sys
import time

import numpy as np

W, H = 960, 540
N = 3  # run per misura (warmup escluso)

ZONES = [
    # (nome, cx, cy, half)
    ("z1_init", -0.5, 0.0, 1.5),
    ("z2_med", -0.7435, 0.1314, 1e-3),
    ("z3_cusp", -0.7499302568795561, -0.015139113925433963, 5.226737155905588e-05),
]


def load_mandel(path):
    spec = importlib.util.spec_from_file_location("mandel_under_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def med(xs):
    return float(np.median(xs))


def gpu_split(m, cp, cx, cy, half, mi, prec):
    """Kernel + D2H separati (stessi parametri di mandel.compute_gpu)."""
    kernel = m._KERNEL_F64 if prec == "f64" else m._KERNEL_F32
    if kernel is None:
        return None
    fdt = np.float64 if prec == "f64" else np.float32
    need = W * H * 3
    if m._BUF is None or m._BUF.size < need:
        m._BUF = cp.empty((need,), dtype=cp.uint8)
    out = m._BUF[:need]
    bx, by = 16, 16
    grid = ((W + 2 * bx - 1) // (2 * bx), (H + by - 1) // by)
    args = (out, m._PAL_IDX[0], np.asarray(cx, fdt), np.asarray(cy, fdt),
            np.asarray(half, fdt), np.asarray(W, np.int32),
            np.asarray(H, np.int32), np.asarray(mi, np.int32))
    # CuPy 14: Event senza elapsed_time -> misuro con perf_counter + sync
    # (include l'overhead di launch, ~decine di us; coerente tra versioni).
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    kernel(grid, (bx, by), args)
    cp.cuda.Stream.null.synchronize()
    t_kernel = (time.perf_counter() - t0) * 1000.0  # ms
    t0 = time.perf_counter()
    host = out.get()
    t_d2h = (time.perf_counter() - t0) * 1000.0  # ms
    return host, t_kernel, t_d2h


def cpu_alloc_cost(cx, cy, half):
    """Stima del costo per-render di allocazioni + interior test (come in compute_cpu)."""
    def body():
        xs = cx + half * (np.arange(W) - W / 2) / (W / 2)
        ys = cy + half * (H / W) * (np.arange(H) - H / 2) / (H / 2)
        real, imag = np.meshgrid(xs, ys)
        c = real + 1j * imag
        z = np.zeros_like(c)
        diverged = np.zeros(c.shape, dtype=bool)
        it = np.zeros(c.shape, dtype=np.int32)
        w4 = 1.0 - 4.0 * c
        d1 = np.abs(w4) < 2.0 * np.sqrt(0.5 * (np.abs(w4) + np.real(w4)))
        d2 = np.abs(c + 1.0) <= 0.25
        return z, diverged, it, d1, d2
    body()  # warmup
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        body()
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    return med(ts)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    mpath = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(here, "mandel.py")
    m = load_mandel(mpath)
    outdir = os.path.dirname(mpath)
    refdir = os.path.join(outdir, "baseline")
    os.makedirs(refdir, exist_ok=True)

    cp = None
    gpu_name = "n/d"
    if m._GPU:
        import cupy
        cp = cupy
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    from PIL import Image

    L = []
    ap = L.append
    ap("BASELINE mandel v%s — %s" % (m.VERSION, time.strftime("%Y-%m-%d %H:%M:%S")))
    ap("GPU: %s | cupy %s | numpy %s | python %s" % (
        gpu_name, getattr(cp, "__version__", "n/d"), np.__version__, sys.version.split()[0]))
    ap("Risoluzione: %dx%d | run: mediana di %d (1 warmup) | palette: fuoco | mi = auto_mi(half)" % (W, H, N))
    ap("")

    for zname, cx, cy, half in ZONES:
        mi = m.auto_mi(half)
        ap("=== %s  c=(%.16g, %.16gi)  half=%.6g  mi=%d" % (zname, cx, cy, half, mi))

        # --- GPU f32 / f64 ---
        for prec in ("f32", "f64"):
            kern = (m._KERNEL_F64 if prec == "f64" else m._KERNEL_F32)
            if kern is None or cp is None:
                ap("  GPU %-3s: NON DISPONIBILE" % prec)
                continue
            gpu_split(m, cp, cx, cy, half, mi, prec)  # warmup (module load, alloc)
            tk, td, tt, tp = [], [], [], []  # NB: 4 liste distinte (non lo stesso oggetto!)
            host = None
            for _ in range(N):
                host, a, b = gpu_split(m, cp, cx, cy, half, mi, prec)
                tk.append(a); td.append(b)
                t0 = time.perf_counter()
                img = m.compute_gpu(cx, cy, half, W, H, mi, prec=prec)
                tt.append((time.perf_counter() - t0) * 1000.0)  # ms
                arr = np.asarray(img).reshape(H, W, 3)
                t0 = time.perf_counter()
                Image.fromarray(arr)
                tp.append((time.perf_counter() - t0) * 1000.0)  # ms
            np.save(os.path.join(refdir, "%s_gpu_%s.npy" % (zname, prec)),
                    np.asarray(img).reshape(H, W, 3))
            ap("  GPU %-3s: kernel %8.2f ms | D2H %7.2f ms | PIL %5.2f ms | totale %8.2f ms  (~%.1f render/s)"
               % (prec, med(tk), med(td), med(tp), med(tt), 1000.0 / med(tt)))

        # --- CPU ---
        m.compute_cpu(cx, cy, half, W, H, mi)  # warmup
        tcpu = []
        img = None
        for _ in range(N):
            t0 = time.perf_counter()
            img = m.compute_cpu(cx, cy, half, W, H, mi)
            tcpu.append((time.perf_counter() - t0) * 1000.0)  # ms
        alloc = cpu_alloc_cost(cx, cy, half)
        np.save(os.path.join(refdir, "%s_cpu.npy" % zname),
                np.asarray(img).reshape(H, W, 3))
        ap("  CPU   : totale %8.2f ms | allocazioni+prefiltro ~%6.2f ms  (~%.2f render/s)"
           % (med(tcpu), alloc, 1000.0 / med(tcpu)))
        ap("")

    ap("Reference frame salvati in: %s" % refdir)
    for f in sorted(os.listdir(refdir)):
        p = os.path.join(refdir, f)
        ap("  %s (%.2f MB)" % (f, os.path.getsize(p) / 1e6))

    report = os.path.join(outdir, "baseline.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n[baseline.txt scritto in %s]" % report)


if __name__ == "__main__":
    main()
