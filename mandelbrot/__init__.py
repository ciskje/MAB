"""Mandelbrot v7.2.0 - pacchetto (ex mandel.py single-file).
STORICO completo: `git log --oneline`. Qui solo le ultime 10 voci
(lette da Help -> Novita` recenti...).
"""
VERSION = "7.2.0"

HISTORY = (
    ("7.2.0", "2026-09-04", "Benchmark GPU a 4x area (2x per lato) con rps normalizzato x4 + pre-warmup pre-finestra; CPU 1x invariato; storici GPU rimesurati col metodo 4x; 'codice di sicurezza' ora 'codice di autenticità'; modalità standard/esperta memorizzata in config."),
    ("7.1.6", "2026-09-04", "Diagnostica split onesta: 'parity mai eseguita' se mai calibrato, 'in corso' solo durante la calibrazione."),
    ("7.1.5", "2026-09-04", "Cursore wait su root+canvas (su macOS il solo root non si vedeva sul canvas e restava appeso fino al movimento del mouse)."),
    ("7.1.4", "2026-09-04", "Cursore wait sempre ripristinato (anche su frame scartati); cambio motore riporta l'antialias a 1x1; nota README per il primo avvio su macOS."),
    ("7.1.3", "2026-09-04", "Storici benchmark GPU aggiornati ai valori verificati in modalita' Esperta."),
    ("7.1.1", "2026-09-04", "Codice sicurezza copiabile (Entry + Copia); clamp valori bench in config (secs/w/h/half sempre sani)."),
    ("7.1.0", "2026-09-04", "Benchmark Esperto (3x8 s, vale la migliore) + codice di sicurezza a 64 bit + Help Verifica benchmark."),
    ("7.0.1", "2026-09-04", "Fix split: import mancanti (struct in vulkan/metal, time+BENCH/auto_mi in cuda, _fmt_lut in metal); Vulkan rende di nuovo."),
    ("7.0.0", "2026-09-04", "Split in pacchetto mandelbrot/ (config/palette/state/mem/cuda/metal/vulkan/cpu/engine/app); shim mandel.py compat; indice palette cacheato."),
    ("6.2.5", "2026-09-03", "Split: calibrazione on-demand, dropdown ratio fisso, label corrente, 2a riga."),
)

# Compat: make_icon.py e vecchi `import mandel` usano questi nomi.
from .config import CX0, CY0, HALF0, MI0, auto_mi  # noqa: F401
from .state import apply_palette  # noqa: F401
from .cpu import compute_cpu  # noqa: F401
