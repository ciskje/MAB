#!/usr/bin/env python3
# gate.py — GATE DI CORRETTEZZA v5.2.0 (3 zone x GPU f32/f64/CPU).
#
# Criteri (vedi STORICO di mandel.py e spec.md):
#   GPU f32/f64 : BIT-IDENTICO a baseline/*_gpu_*.npy (maxdiff 0).
#                 (Riferimenti v4.15.1; il percorso GPU non cambia semantica
#                  -> restano validi da sempre.)
#   CPU         : BIT-IDENTICA a baseline/*_cpu.npy (maxdiff 0; riferimenti
#                 v5.2.0: stessa formula di coloring della GPU).
#   CPU vs GPUf64: stessa immagine: frazione di pixel con max diff per canale
#                 > 8 <= 2% (misurato 0.011%-0.75% sulle 3 zone: solo pixel di
#                 bordo caotico + 1-2 ULP di log2/log libm-vs-CUDA).
#   NB: il vecchio check "CPU continuita' <=1.5% pixel vs v4.15.1" e' stato
#       sostituito: in v5.2.0 la formula di coloring CPU e' cambiata
#       VOLUTARIAMENTE (smooth iteration, come la GPU) per il bug
#       "regione lontana nera in CPU ma rossa in CUDA" (v<=5.1.x:
#       rgb[it==0]=0 confondeva interiore con fuga a 1a iterazione).
#       (I riferimenti baseline/*_cpu_v4151.npy sono stati rimossi in v5.2.1,
#       recuperabili dalla storia git.)
# Uso: python gate.py [path/mandel.py]
import importlib.util
import os
import sys

import numpy as np

W, H = 960, 540
MAX_CPU_VS_GPU_FRAC = 0.02  # frazione max di pixel CPU-vs-GPUf64 con maxdiff>8
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    mpath = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(here, "mandel.py")
    m = load_mandel(mpath)
    refdir = os.path.join(os.path.dirname(mpath), "baseline")
    print("gate v5.2.0 — mandel %s — GPU=%s | GPU bit-id vs ref | CPU bit-id vs ref | CPU~GPUf64 <=%.0f%% pixel >8"
          % (m.VERSION, m._GPU, MAX_CPU_VS_GPU_FRAC * 100))
    ok = True
    for zname, cx, cy, half in ZONES:
        mi = m.auto_mi(half)
        ref_gpu32 = np.load(os.path.join(refdir, "%s_gpu_f32.npy" % zname))
        ref_gpu64 = np.load(os.path.join(refdir, "%s_gpu_f64.npy" % zname))
        ref_cpu = np.load(os.path.join(refdir, "%s_cpu.npy" % zname))
        # GPU f32 / f64: bit-identici ai riferimenti (maxdiff DEVE essere 0)
        img = np.asarray(m.compute_gpu(cx, cy, half, W, H, mi, prec="f32")).reshape(H, W, 3)
        d32 = maxdiff(img, ref_gpu32)
        if m._KERNEL_F64 is not None:
            img64 = np.asarray(m.compute_gpu(cx, cy, half, W, H, mi, prec="f64")).reshape(H, W, 3)
            d64 = maxdiff(img64, ref_gpu64)
        else:
            img64, d64 = None, -1
        # CPU: bit-identica ai riferimenti v5.2.0 (maxdiff DEVE essere 0)
        imgc = np.asarray(m.compute_cpu(cx, cy, half, W, H, mi)).reshape(H, W, 3)
        dc = maxdiff(imgc, ref_cpu)
        # CPU vs GPU f64: stessa immagine (solo bordo caotico + ULP log)
        if img64 is not None:
            fg = big_diff_frac(imgc, img64)
        else:
            fg = -1.0
        good = (d32 == 0) and (d64 in (0, -1)) and (dc == 0) and (fg <= MAX_CPU_VS_GPU_FRAC)
        ok = ok and good
        print(("  OK  " if good else "  FAIL ")
              + "%s: GPUf32 maxdiff=%d | GPUf64 maxdiff=%d | CPU maxdiff=%d | CPU~GPUf64 px>8=%.3f%%"
              % (zname, d32, d64, dc, fg * 100.0))
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
