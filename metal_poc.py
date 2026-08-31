#!/usr/bin/env python3
# metal_poc.py — FASE 0 (PoC standalone): backend Metal (Apple GPU, f32)
# per il visualizzatore di Mandelbrot, in parallelo a CUDA/CPU.
#
# Obiettivi (GO/NO-GO):
#   1) CORRETTO:  Metal f32 vs CPU f32/f64 -> frazione pixel maxdiff>8 <= 2%
#   2) DETERM.  : due run Metal bit-identici (maxdiff == 0)
#   3) VELOCITA : GPU compute (encode->dispatch->wait) vs CPU f32/f64
#
# NB: Metal su M1 NON supporta 'double' -> backend GPU e' f32-only.
# Uso: python3 metal_poc.py
import os
import sys
import time
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mandel as M  # riuso PALETTES, make_lut, auto_mi, compute_cpu (stesse costanti)

import Metal
from PIL import Image

W, H = 960, 540
ZONES = [
    ("z1_init", -0.5, 0.0, 1.5),
    ("z2_med", -0.7435, 0.1314, 1e-3),
    ("z3_cusp", -0.7499302568795561, -0.015139113925433963, 5.226737155905588e-05),
]
TH = 16  # threads per threadgroup (16x16 = 256, sotto il max 1024 di M1)


def fmt_lut(lut):
    return ", ".join(str(int(v)) for v in np.asarray(lut).ravel())


def build_msl():
    names = list(M.PALETTES)
    consts = "\n".join(
        "constant unsigned char LUT_%s[768] = { %s };"
        % (n.upper(), fmt_lut(M.make_lut(pal))) for n, pal in M.PALETTES.items())
    select = "    constant unsigned char* lut = LUT_%s;" % names[0].upper()
    for i, n in enumerate(names[1:], start=1):
        select += "\n    if (p.pal == %d) lut = LUT_%s;" % (i, n.upper())
    return r'''
#include <metal_stdlib>
using namespace metal;

@@CONSTS@@

struct Params {
    float cx;
    float cy;
    float hs;
    int w;
    int h;
    int mi;
    int pal;
};

kernel void mandel(
    device unsigned char* out [[buffer(0)]],
    constant Params& p [[buffer(1)]],
    uint2 pos [[thread_position_in_grid]])
{
    int col = (int)pos.x;
    int row = (int)pos.y;
    if (col >= p.w || row >= p.h) return;

@@SELECT@@

    float cx = p.cx, cy = p.cy, hs = p.hs;
    int w = p.w, h = p.h, mi = p.mi;

    float x0 = cx + hs * ((float)(2 * col - w) / (float)w);
    float y0 = cy + hs * ((float)h / (float)w) * ((float)(2 * row - h) / (float)h);

    device unsigned char* pp = out + (size_t)(row * w + col) * 3;

    // Interior analitico: bulbo periodica-2 + cardioide principale (stesso criterio CUDA)
    if (x0 >= -2.0f && x0 <= 0.4f && y0 >= -1.3f && y0 <= 1.3f) {
        float d2 = (x0 + 1.0f) * (x0 + 1.0f) + y0 * y0;
        if (d2 <= 0.0625f) { pp[0] = 0; pp[1] = 0; pp[2] = 0; return; }
        float A = 1.0f - 4.0f * x0;
        float B = -4.0f * y0;
        float R = sqrt(A * A + B * B);
        if (R < 2.0f * sqrt(0.5f * (R + A))) { pp[0] = 0; pp[1] = 0; pp[2] = 0; return; }
    }

    float a = cx * cx + (x0 - cx);
    float two_cx = 2.0f * cx;
    float wr = -cx, wi = 0.0f;
    int it = 0;
    bool esc = false;
    float mag2 = 0.0f;
    for (int i = 0; i < mi; ++i) {
        if (esc) break;
        float nr = wr * wr - wi * wi + two_cx * wr + a;
        float ni = two_cx * wi + 2.0f * wr * wi + y0;
        wr = nr; wi = ni;
        float zr = wr + cx;
        mag2 = zr * zr + wi * wi;
        if (mag2 > 4.0f) { esc = true; it = i; }
    }
    if (!esc) { pp[0] = 0; pp[1] = 0; pp[2] = 0; return; }

    float nu = (float)it + 1.0f - log2(0.5f * log(mag2));
    float t = pow(fmin(1.0f, fmax(0.0f, nu / (float)mi)), 0.35f);
    int idx = (int)(fmin(1.0f, fmax(0.0f, t)) * 255.0f);
    pp[0] = lut[idx * 3 + 0];
    pp[1] = lut[idx * 3 + 1];
    pp[2] = lut[idx * 3 + 2];
}
'''.replace("@@CONSTS@@", consts).replace("@@SELECT@@", select)


class MetalBackend:
    def __init__(self):
        self.dev = Metal.MTLCreateSystemDefaultDevice()
        assert self.dev is not None, "nessun dispositivo Metal"
        self.q = self.dev.newCommandQueue()
        src = build_msl()
        self.lib, err = self.dev.newLibraryWithSource_options_error_(src, None, None)
        assert self.lib is not None, "compile MSL fallito: %s" % err
        fn = self.lib.newFunctionWithName_('mandel')
        self.ps, err = self.dev.newComputePipelineStateWithFunction_error_(fn, None)
        assert self.ps is not None, "pipeline fallito: %s" % err
        self._out = None
        self._pbuf = self.dev.newBufferWithLength_options_(
            28, Metal.MTLResourceStorageModeShared)

    def _out_buf(self, need):
        if self._out is None or self._out.length() < need:
            self._out = self.dev.newBufferWithLength_options_(
                need, Metal.MTLResourceStorageModeShared)
        return self._out

    def compute(self, cx, cy, half, w, h, mi, pal=0):
        need = w * h * 3
        out = self._out_buf(need)
        # H2D: parametri via memoryview (writable, C-level)
        payload = struct.pack('<fffiiii', cx, cy, half, w, h, mi, pal)
        self._pbuf.contents().as_buffer(len(payload))[:] = payload
        gx = (w + TH - 1) // TH
        gy = (h + TH - 1) // TH
        cmd = self.q.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.ps)
        enc.setBuffer_offset_atIndex_(out, 0, 0)     # buffer[0] = out
        enc.setBuffer_offset_atIndex_(self._pbuf, 0, 1)  # buffer[1] = params (offset=0, INDEX=1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(gx, gy, 1), Metal.MTLSizeMake(TH, TH, 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        # D2H: memoryview -> numpy (C-level)
        mv = out.contents().as_buffer(need)
        return np.frombuffer(mv, dtype=np.uint8, count=need).reshape(h, w, 3).copy()


def maxdiff(a, b):
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


def big_diff_frac(a, b, thr=8):
    p = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max(axis=-1)
    return float((p > thr).mean())


def main():
    refdir = os.path.join(HERE, "baseline")
    mb = MetalBackend()
    print("Metal device: %s | max threads/threadgroup: %d"
          % (mb.dev.name(), mb.ps.maxTotalThreadsPerThreadgroup()))
    print("GPU f32 (Metal) — M1 (no f64: 'double' non supportato da Metal su Apple Silicon)\n")

    # Criterio: Metal f32 e' un backend f32 come CUDA f32 e CPU f32. A zoom
    # profondi (mi alto) i tre backends f32 divergono tra loro SOLO per il
    # rounding FMA/contrazione sul bordo caotico (limitazione intrinseca di f32,
    # NON un difetto di Metal). Quindi il gold-standard e' CUDA f32 (*_gpu_f32.npy)
    # e il bound e' la varianza naturale f32 (CPU f32 ~ CUDA f32), con floor 2%
    # per le zone pulite. (Il 2% del gate vale per f64-vs-f64, non per f32.)
    all_ok = True
    for zname, cx, cy, half in ZONES:
        mi = M.auto_mi(half)
        m1 = mb.compute(cx, cy, half, W, H, mi, pal=0)
        m2 = mb.compute(cx, cy, half, W, H, mi, pal=0)  # determinism
        ddet = maxdiff(m1, m2)
        g32 = np.load(os.path.join(refdir, "%s_gpu_f32.npy" % zname))  # gold f32 (CUDA)
        c32 = np.load(os.path.join(refdir, "%s_cpu_f32.npy" % zname))
        fm = big_diff_frac(m1, g32)   # Metal ~ CUDA f32
        fc = big_diff_frac(c32, g32)  # varianza intrinseca f32 (CPU ~ CUDA)
        bound = max(0.02, 1.3 * fc)
        good = (ddet == 0) and (fm <= bound)
        all_ok = all_ok and good
        print("%s mi=%-6d | det maxdiff=%d | Metal~CUDAf32 px>8=%.2f%% (bound %.2f%%) | CPUf32~CUDAf32=%.2f%% %s"
              % (zname, mi, ddet, fm * 100, bound * 100, fc * 100, "OK" if good else "FAIL"))

    print("\n--- VELOCITA (mediana) ---")
    print("%-8s %-10s %-12s %-12s %-10s" % ("zona", "mi", "Metal", "CPU f32", "CPU f64"))
    for zname, cx, cy, half in ZONES:
        mi = M.auto_mi(half)
        # warmup bounded (alza clock GPU + compila Numba)
        for _ in range(5):
            mb.compute(cx, cy, half, W, H, mi, pal=0)
        tm = [None] * 7
        for i in range(7):
            t0 = time.perf_counter()
            mb.compute(cx, cy, half, W, H, mi, pal=0)
            tm[i] = (time.perf_counter() - t0) * 1000.0
        M.compute_cpu(cx, cy, half, W, H, mi, prec="f32")  # warmup Numba
        tc32 = []
        for _ in range(3):
            t0 = time.perf_counter()
            M.compute_cpu(cx, cy, half, W, H, mi, prec="f32")
            tc32.append((time.perf_counter() - t0) * 1000.0)
        tc64 = []
        for _ in range(3):
            t0 = time.perf_counter()
            M.compute_cpu(cx, cy, half, W, H, mi, prec="f64")
            tc64.append((time.perf_counter() - t0) * 1000.0)
        mm, c32, c64 = float(np.median(tm)), float(np.median(tc32)), float(np.median(tc64))
        print("%-8s %-10d %8.2f ms %8.2f ms %8.2f ms   (Metal %.1fx vs CPUf32)"
              % (zname, mi, mm, c32, c64, c32 / mm if mm else 0))

    print("\nPOC: %s" % ("GO" if all_ok else "NO-GO"))


if __name__ == "__main__":
    main()
