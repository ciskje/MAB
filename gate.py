#!/usr/bin/env python3
# gate.py — GATE DI CORRETTEZZA v5.4.0 (PORTATILE: CPU sempre; CUDA e/o Metal).
#
# Criteri (vedi STORICO di mandel.py e spec.md):
#   CPU f64     : BIT-IDENTICA a baseline/*_cpu.npy (maxdiff 0; riferimenti
#                 v5.2.0: stessa formula di coloring della GPU).
#   CPU f32     : BIT-IDENTICA a baseline/*_cpu_f32.npy (maxdiff 0;
#                 riferimenti v5.3.0: nuovo percorso f32).
#   [CUDA, se disponibile]
#     CUDA f32  : BIT-IDENTICO a baseline/*_gpu_f32.npy (maxdiff 0).
#     CUDA f64  : BIT-IDENTICO a baseline/*_gpu_f64.npy (maxdiff 0, se kernel f64).
#     CPUf64~CUDAf64 : <=2% pixel con maxdiff>8 (bordo caotico + ULP libm-vs-CUDA;
#                 misurato 0.011%-0.75% sulle 3 zone).
#   [Metal, Apple GPU, se disponibile; v5.4.0]
#     Metal f32 : BIT-IDENTICO a baseline/*_metal_f32.npy (maxdiff 0) +
#                 DETERMINISMO (2 run bit-identici). Metal e' f32-only su
#                 Apple Silicon (no 'double').
#     Metal f32 ~ CPUf32 : entro la varianza intrinseca di f32, stimata come
#                 max(2%, 1.5 x CPUf32~CUDAf32) usando il gold CUDA f32 come
#                 righello della "nube" f32. A deep-zoom le implementazioni
#                 f32 (Numba/CUDA/Metal) divergono tra loro per ~10-25% dei
#                 pixel di bordo caotico per FMA/contrazione: NON e' un difetto.
#                 (Se il gold CUDA f32 e' assente, la correttezza Metal si riduce
#                 a bit-id + determinismo.)
#   NB: il vecchio check "CPU continuita' <=1.5% vs v4.15.1" e' stato
#       sostituito in v5.2.0 (formula di coloring CPU cambiata VOLUTARIAMENTE,
#       smooth iteration, per il bug "regione lontana nera in CPU ma rossa in
#       CUDA"). I riferimenti baseline/*_cpu_v4151.npy furono rimossi in v5.2.1
#       (recuperabili dalla storia git).
# Uso: python gate.py [path/mandel.py]
import importlib.util
import os
import sys

import numpy as np

W, H = 960, 540
MAX_F64_FRAC = 0.02    # CPUf64~CUDAf64: frazione max di pixel con maxdiff>8
MAX_F32_BASE = 0.02    # bound di base per la varianza f32 (Metal~CPUf32)
F32_SPREAD_K = 1.5     # moltiplicatore sulla misura CPUf32~CUDAf32
ZONES = [
    ("z1_init", -0.5, 0.0, 1.5),
    ("z2_med", -0.7435, 0.1314, 1e-3),
    ("z3_cusp", -0.7499302568795561, -0.015139113925433963, 5.226737155905588e-05),
]


def load_mandel(path):
    spec = importlib.util.spec_from_file_location("mandel_gate", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def maxdiff(a, b):
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


def big_diff_frac(a, b, thr=8):
    """Frazione di pixel (triplette RGB) con almeno un canale che differisce di > thr."""
    p = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max(axis=-1)
    return float((p > thr).mean())


def load_ref(refdir, name):
    p = os.path.join(refdir, name)
    if not os.path.exists(p):
        return None
    return np.load(p)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    mpath = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(here, "mandel.py")
    m = load_mandel(mpath)
    refdir = os.path.join(os.path.dirname(mpath), "baseline")

    cuda_ok = (m._CUDA_OK and m._KERNEL_F32 is not None)
    metal_ok = (m._METAL_OK and m._METAL_BE is not None)
    bdesc = "CUDA" if cuda_ok else ("Metal" if metal_ok else "CPU-only")
    print("gate v5.4.0 — mandel %s — backend=%s" % (m.VERSION, bdesc))
    print("  CPU f64/f32 bit-id | CUDA: bit-id + CPUf64~CUDAf64<=%.0f%%" % (MAX_F64_FRAC * 100))
    print("  Metal: bit-id + determinismo + Metal~CPUf32 entro max(%.0f%%, %.1fx CPUf32~CUDAf32)"
          % (MAX_F32_BASE * 100, F32_SPREAD_K))

    ok = True
    for zname, cx, cy, half in ZONES:
        mi = m.auto_mi(half)

        # --- CPU (sempre): bit-id ai riferimenti ---
        ref_cpu64 = load_ref(refdir, "%s_cpu.npy" % zname)
        ref_cpu32 = load_ref(refdir, "%s_cpu_f32.npy" % zname)
        imgc64 = np.asarray(m.compute_cpu(cx, cy, half, W, H, mi, prec="f64")).reshape(H, W, 3)
        imgc32 = np.asarray(m.compute_cpu(cx, cy, half, W, H, mi, prec="f32")).reshape(H, W, 3)
        dc64 = maxdiff(imgc64, ref_cpu64) if ref_cpu64 is not None else -1
        dc32 = maxdiff(imgc32, ref_cpu32) if ref_cpu32 is not None else -1
        checks = [dc64 == 0, dc32 == 0]
        parts = ["CPUf64 md=%d" % dc64, "CPUf32 md=%d" % dc32]
        # gold CUDA f32: serve al check CUDA (se c'e' runtime) E al cross-check
        # Metal (sempre, se il file esiste: e' un confronto tra file, funziona
        # anche su una macchina senza runtime CUDA).
        cuda_gold32 = load_ref(refdir, "%s_gpu_f32.npy" % zname)

        # --- CUDA (se disponibile): bit-id + CPUf64~CUDAf64 ---
        if cuda_ok:
            ref_gpu64 = load_ref(refdir, "%s_gpu_f64.npy" % zname)
            img32 = np.asarray(m._compute_gpu_cuda(cx, cy, half, W, H, mi, prec="f32")).reshape(H, W, 3)
            d32 = maxdiff(img32, cuda_gold32) if cuda_gold32 is not None else -1
            if m._KERNEL_F64 is not None and ref_gpu64 is not None:
                img64 = np.asarray(m._compute_gpu_cuda(cx, cy, half, W, H, mi, prec="f64")).reshape(H, W, 3)
                d64 = maxdiff(img64, ref_gpu64)
                fg = big_diff_frac(imgc64, img64)
            else:
                d64, fg = -1, -1.0
            checks += [d32 == 0, d64 in (0, -1), fg <= MAX_F64_FRAC]
            parts += ["CUDAf32 md=%d" % d32, "CUDAf64 md=%d" % d64,
                      "CPUf64~CUDAf64=%.3f%%" % (fg * 100)]

        # --- Metal (Apple GPU, se disponibile): bit-id + det + ~CPUf32 ---
        if metal_ok:
            ref_metal32 = load_ref(refdir, "%s_metal_f32.npy" % zname)
            imgm = np.asarray(m._compute_gpu_metal(cx, cy, half, W, H, mi, prec="f32")).reshape(H, W, 3)
            dm = maxdiff(imgm, ref_metal32) if ref_metal32 is not None else -1
            imgm2 = np.asarray(m._compute_gpu_metal(cx, cy, half, W, H, mi, prec="f32")).reshape(H, W, 3)
            dd = maxdiff(imgm, imgm2)  # determinismo
            checks += [dm == 0, dd == 0]
            parts += ["Metal md=%d" % dm, "det md=%d" % dd]
            cross = big_diff_frac(imgm, imgc32)
            if cuda_gold32 is not None:
                base = big_diff_frac(imgc32, cuda_gold32)
                bound = max(MAX_F32_BASE, F32_SPREAD_K * base)
                okcross = cross <= bound
                parts += ["Metal~CPUf32=%.2f%%<=%.2f%%" % (cross * 100, bound * 100)]
            else:
                okcross = True  # gold CUDA assente: bit-id + determinismo bastano
                parts += ["Metal~CPUf32=%.2f%%(no gold CUDA)" % (cross * 100)]
            checks += [okcross]

        good = all(checks)
        ok = ok and good
        print(("  OK  " if good else "  FAIL ") + "%s: " % zname + " | ".join(parts))

    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
