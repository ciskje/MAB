"""Costanti immutabili: vista, rendering, benchmark, percorsi, colori UI."""
import math
import os

INIT_W, INIT_H = 1280, 720
CX0, CY0, HALF0 = -0.5, 0.0, 1.5
MI0 = 200
# Iterazioni "auto": mi = MI_AUTO_BASE * (1 + log10(HALF0/half)), clamp [50, 50000]
# (coefficiente alzato da 400 a 2000 e clamp max da 10000 a 50000: le zone
# quasi paraboliche, es. vicino al becco della cardioide, fuggono lentamente
# e con il vecchio coefficiente il coloring non convergeva).
# La STESSA formula e' usata dal benchmark (vedi v4.15.0).
MI_AUTO_BASE = 2000
MI_AUTO_MIN, MI_AUTO_MAX = 50, 50000

MIN_HALF = 1e-12  # clamp minimo per half (evita zoom infinito -> half=0, stato degenere)
MIN_DIM = 50  # larghezza/altezza canvas minimi per considerare il canvas valido

# v6.1: niente tetto statico per il ricalcolo NxN: si stimano i byte
# necessari e si confrontano con la memoria davvero disponibile (VRAM via
# CUDA/Metal, RAM host via OS). Margine + riserva sotto.
PHOTO_HEADROOM = 1.25  # margine 25% sulle stime
PHOTO_HOST_RESERVE = 1 << 30  # 1 GiB di RAM mai intaccato
PHOTO_BACKSTOP_MPX = 128.0  # rete di sicurezza SOLO se la memoria libera
# non e' determinabile (query OS/driver fallita): oltre si rifiuta comunque.

# --- Percorsi e parametri ---
# Config salvata/caricata a ogni esecuzione (vista + tutti i settaggi)
CONFIG_PATH = os.path.join(os.path.expanduser("~"), "mandelbrot", "config.json")

# Benchmark: regione + parametri. Default = regione fornita dall'utente;
# i valori sono persistiti in config.json (overridibili).
# NB: 'mi' non fa piu' parte di BENCH: e' sempre derivato dalla formula auto
# (auto_mi(bench['half'])) per mantenere il benchmark comparabile tra versioni.
BENCH = dict(cx=-0.7499302568795561, cy=-0.015139113925433963, half=5.226737155905588e-05,
             w=960, h=540, secs=8.0)

# Riferimenti storici per il grafico a barre del benchmark (rendering/s,
# stessa regione/parametri): evidenziano il salto CPU->GPU su macchine note.
BENCH_REF = (
    ("AMD 9900X (storico)", 6.62),
    ("4070 Super Vulkan (storico)", 124.0),
    ("5070 Ti Vulkan (storico)", 179.0),
    ("4070 Super CUDA (storico)", 250.0),
    ("5070 Ti CUDA (storico)", 350.0),
)

BACKEND_FG = {"cpu": "#1f6feb", "cuda": "#2ea44f",
              "metal": "#8957e5", "vulkan": "#d97706"}
ERR_FG = "#e5534b"

def auto_mi(half):
    """Iterazioni 'auto' per una data half: formula unica condivisa da eff_mi
    e benchmark (vedi costanti MI_AUTO_*)."""
    z = HALF0 / max(half, 1e-12)
    return int(max(MI_AUTO_MIN, min(MI_AUTO_MAX, MI_AUTO_BASE * (1.0 + math.log10(z)))))

