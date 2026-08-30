#!/usr/bin/env python3
# gate.py — GATE DI CORRETTEZZA v5.0.0 (3 zone x GPU f32/f64/CPU).
#
# Criteri (vedi PIANO-REINGEGNERIZZAZIONE.md, Fase 5):
#   GPU f32/f64 : BIT-IDENTICO a baseline/*_gpu_*.npy (maxdiff 0).
#                 (Riferimenti v4.15.1; il percorso GPU non cambia semantica,
#                  solo la config di launch -> restano validi.)
#   CPU         : BIT-IDENTICA a baseline/*_cpu.npy (maxdiff 0; riferimenti
#                 v5.0.0, nuova verità) E CONTINUITA' vs v4.15.1
#                 (baseline/*_cpu_v4151.npy): <=1.5% dei pixel diversi
#                 (effetto FMA di numpy 2.4 su np.square, pixel di bordo
#                 caotico; documentato in spec/AGENTS).
# Uso: python gate.py [path/mandel.py]
import importlib.util
import os
import sys

import numpy as np

W, H = 960, 540
MAX_CPU_CONT_FRAC = 0.015  # frazione max di pixel CPU diversi vs v4.15.1
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


def diff_frac(a, b):
    """Frazione di pixel (triplette RGB) con almeno un canale diverso."""
    return float(np.any(np.asarray(a, np.uint8) != np.asarray(b, np.uint8), axis=-1).mean())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    mpath = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(here, "mandel.py")
    m = load_mandel(mpath)
    refdir = os.path.join(os.path.dirname(mpath), "baseline")
    print("gate v5 — mandel %s — GPU=%s | CPU bit-id vs v5 + continuita' <=1.5%% vs v4.15.1"
          % (m.VERSION, m._GPU))
    ok = True
    for zname, cx, cy, half in ZONES:
        mi = m.auto_mi(half)
        ref_gpu32 = np.load(os.path.join(refdir, "%s_gpu_f32.npy" % zname))
        ref_gpu64 = np.load(os.path.join(refdir, "%s_gpu_f64.npy" % zname))
        ref_cpu = np.load(os.path.join(refdir, "%s_cpu.npy" % zname))
        ref_cpu4 = np.load(os.path.join(refdir, "%s_cpu_v4151.npy" % zname))
        # GPU f32 / f64: bit-identici ai riferimenti (maxdiff DEVE essere 0)
        img = np.asarray(m.compute_gpu(cx, cy, half, W, H, mi, prec="f32")).reshape(H, W, 3)
        d32 = maxdiff(img, ref_gpu32)
        if m._KERNEL_F64 is not None:
            img64 = np.asarray(m.compute_gpu(cx, cy, half, W, H, mi, prec="f64")).reshape(H, W, 3)
            d64 = maxdiff(img64, ref_gpu64)
        else:
            d64 = -1
        # CPU: bit-identica ai riferimenti v5 (maxdiff DEVE essere 0)
        imgc = np.asarray(m.compute_cpu(cx, cy, half, W, H, mi)).reshape(H, W, 3)
        dc = maxdiff(imgc, ref_cpu)
        # ...e continuita' vs v4.15.1 (effetto FMA: <=1.5% pixel di bordo)
        fc = diff_frac(imgc, ref_cpu4)
        good = (d32 == 0) and (d64 in (0, -1)) and (dc == 0) and (fc <= MAX_CPU_CONT_FRAC)
        ok = ok and good
        print(("  OK  " if good else "  FAIL ")
              + "%s: GPUf32 maxdiff=%d | GPUf64 maxdiff=%d | CPU maxdiff=%d | CPU vs v4151 diff=%.3f%%"
              % (zname, d32, d64, dc, fc * 100.0))
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
