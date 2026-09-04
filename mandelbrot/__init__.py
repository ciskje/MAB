"""Mandelbrot v7.0.0 - pacchetto (ex mandel.py single-file).
STORICO completo: `git log --oneline`. Qui solo le ultime 10 voci
(lette da Help -> Novita` recenti...).
"""
VERSION = "7.0.0"

HISTORY = (
    ("7.0.0", "2026-09-04", "Split in pacchetto mandelbrot/ (config/palette/state/mem/cuda/metal/vulkan/cpu/engine/app); shim mandel.py compat; indice palette cacheato."),
    ("6.2.5", "2026-09-03", "Split: calibrazione on-demand, dropdown ratio fisso, label corrente, 2a riga."),
    ("6.2.4", "2026-09-03", "Benchmark usa lo split (Entrambe) quando attivo, come il render."),
    ("6.2.3", "2026-09-03", "Fix split CUDA: output locale alla banda (row0 solo per il calcolo)."),
    ("6.2.2", "2026-09-03", "Parity split con pattern-match sulla banda 2 (indagine seam)."),
    ("6.2.1", "2026-09-03", "Split CUDA: parity automatica split-vs-single con fail-safe."),
    ("6.2", "2026-09-03", "Split CUDA su 2 GPU con rapporto auto-calibrato (gpu1/gpu2/entrambe)."),
    ("6.1.2", "2026-09-03", "Reset riporta la scala a 1x1."),
    ("6.1.1", "2026-09-03", "Benchmark in esclusiva GPU anche vs ricalcolo NxN (conteggi stabili)."),
    ("6.1", "2026-09-03", "Ricalcolo NxN: memoria misurata davvero (VRAM/RAM), via tetto statico."),
)

# Compat: make_icon.py e vecchi `import mandel` usano questi nomi.
from .config import CX0, CY0, HALF0, MI0, auto_mi  # noqa: F401
from .state import apply_palette  # noqa: F401
from .cpu import compute_cpu  # noqa: F401
