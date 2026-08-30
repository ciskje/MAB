#!/usr/bin/env python3
# gate.py — verifica bit-identita' delle nuove implementazioni (v4.16.0+)
# rispetto ai frame reference v4.15.1 in baseline/*.npy (maxdiff DEVE essere 0).
# Uso: python gate.py [path/mandel.py]
import importlib.util
import os
import sys

import numpy as np

W, H = 960, 540
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    mpath = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(here, "mandel.py")
    m = load_mandel(mpath)
    refdir = os.path.join(os.path.dirname(mpath), "baseline")
    print("gate vs v4.15.1 — mandel %s — GPU=%s" % (m.VERSION, m._GPU))
    ok = True
    for zname, cx, cy, half in ZONES:
        mi = m.auto_mi(half)
        ref_gpu32 = np.load(os.path.join(refdir, "%s_gpu_f32.npy" % zname))
        ref_gpu64 = np.load(os.path.join(refdir, "%s_gpu_f64.npy" % zname))
        ref_cpu = np.load(os.path.join(refdir, "%s_cpu.npy" % zname))
        # GPU f32
        img = np.asarray(m.compute_gpu(cx, cy, half, W, H, mi, prec="f32")).reshape(H, W, 3)
        d32 = int(np.abs(img.astype(np.int16) - ref_gpu32.astype(np.int16)).max())
        # GPU f64
        if m._KERNEL_F64 is not None:
            img64 = np.asarray(m.compute_gpu(cx, cy, half, W, H, mi, prec="f64")).reshape(H, W, 3)
            d64 = int(np.abs(img64.astype(np.int16) - ref_gpu64.astype(np.int16)).max())
        else:
            d64 = -1
        # CPU
        imgc = np.asarray(m.compute_cpu(cx, cy, half, W, H, mi)).reshape(H, W, 3)
        dc = int(np.abs(imgc.astype(np.int16) - ref_cpu.astype(np.int16)).max())
        line = "%s: GPUf32 maxdiff=%d | GPUf64 maxdiff=%d | CPU maxdiff=%d" % (zname, d32, d64, dc)
        good = (d32 == 0) and (d64 in (0, -1)) and (dc == 0)
        ok = ok and good
        print(("  OK  " if good else "  FAIL ") + line)
    print("GATE:", "PASS (bit-identico a v4.15.1)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
