# ============================================================================
# Insieme di Mandelbrot - visualizzatore interattivo
# VERSIONE: 7.1.1
# ----------------------------------------------------------------------------
# Shim di compatibilita': il programma vive nel pacchetto mandelbrot/
# (config/palette/state/mem/cuda/metal/vulkan/cpu/engine/app).
# STORICO completo: git log --oneline. Ultime 10 voci in mandelbrot/__init__.py
# (HISTORY, letta da Help -> Novita` recenti...).
#
# 7.1.1 - 2026-09-04
#   - Codice sicurezza copiabile (Entry + Copia); clamp valori bench in
#     config (secs 1-120, w/h 64-7680, half>0: mai piu' bench accorciati).
# 7.1.0 - 2026-09-04
#   - Benchmark Esperto (3 prove da 8 s, vale la migliore) + codice di
#     sicurezza a 64 bit (tamper-evident) + Help Verifica benchmark.
# 7.0.1 - 2026-09-04
#   - Fix split: import mancanti (struct in vulkan/metal, time+BENCH/auto_mi
#     in cuda, _fmt_lut in metal); Vulkan rende di nuovo.
# 7.0.0 - 2026-09-04
#   - Split in pacchetto mandelbrot/ (ex single-file da 3774 righe);
#     matematica e kernel invariati (CPU/CUDA bit-identici, verificato).
#     Indice palette cacheato (hot-path), import lazy anti-ciclo in mem.py.
# ============================================================================

from mandelbrot import VERSION, HISTORY, CX0, CY0, HALF0, MI0, auto_mi  # noqa: F401
from mandelbrot import apply_palette, compute_cpu  # noqa: F401 (make_icon.py)
from mandelbrot.app import MandelbrotApp, main  # noqa: F401


if __name__ == "__main__":
    main()
