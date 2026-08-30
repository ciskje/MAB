# ============================================================================
# Insieme di Mandelbrot - visualizzatore interattivo
# VERSIONE: 4.11.1
# ----------------------------------------------------------------------------
# REGOLA: ogni modifica incrementa la versione e aggiunge una voce qui sotto
# (formato: versione - data - descrizione modifiche).
#
# STORICO:
# 4.11.1 - 2026-08-30
#   - Refactor leggibilita' (nessun cambio di comportamento, CPU bit-identica
#     maxdiff=0 su 3 viste): metodi di MandelbrotApp raggruppati per funzione
#     (Costruzione UI / Helper UI / Vista e interazione / Controlli / Pipeline /
#     File / Benchmark) con intestazioni di gruppo; intestazione "Dispatch
#     backend" e gruppi di costanti a livello modulo.
# 4.11.0 - 2026-08-29
#   - Carica zona: nuova voce menu File "Carica zona..." che legge un file JSON
#     (stesso formato del salvataggio) e ripristina vista + iterazioni; il file
#     scelto diventa il "file corrente" per "Salva zona".
#   - Titolo: mostra il nome del file corrente (quello usato da "Salva zona");
#     il file corrente e' persistito in config.json (view_file).
# 4.10.0 - 2026-08-29
#   - Benchmark: nuova regione di default (zoom profondo
#     c=(0.42663924626512445, -0.3414973874054564i), half=2.298743311298834e-06)
#     e parametri persistiti in config.json (salvati/caricati, overridibili);
#     i metodi benchmark usano self.bench invece della costante BENCH.
# 4.9.0 - 2026-08-29
#   - Nuova: salvataggio della zona attuale (vista) su file JSON testuale
#     leggibile (menu File): "Salva zona" (riscrive l'ultimo file scelto, o
#     chiede il nome al primo uso) e "Salva zona con nome...". Salva
#     cx/cy/half + iterazioni (mi, mi_auto) per riprodurre la vista.
# 4.8.0 - 2026-08-29
#   - Refactor UI: __init__ (116 righe) spezzato in _setup_fonts/_build_toolbar/
#     _build_canvas_status/_build_menu/_bind_events/_start_pipeline. Helper
#     _refresh_title e _select_palette/_select_backend/_select_precision eliminano
#     la triplice ripetizione "deselect+select" (era in set_* + load_config + reset).
#   - Perf CPU: test di fuga senza sqrt (|z|>2 -> z.re^2+z.im^2>4), output
#     bit-identico (maxdiff=0 su 5 viste incl. deep zoom). Perf GPU: indice
#     palette preallocato (_PAL_IDX), niente np.asarray per render.
#   - Robustezza: clamp half >= MIN_HALF (1e-12) in zoom_at (evita half=0).
#   - Pulizia: rimossa prec() (getter mai usato) e import time ridondante in
#     save_png; change_mi usa _update_mi_label(); tolti 'global' inutili
#     (save/load_config, reset).
# 4.7.2 - 2026-08-29
#   - UI: etichette con iniziale maiuscola (Motore/Palette/Precisione/Iter/Auto/
#     Iterazioni)
# 4.7.1 - 2026-08-29
#   - FIX: disattivando "iter: auto", il valore mostrava il fisso iniziale
#     (stale) invece del valore auto corrente. Ora self.mi viene congelato
#     sull'eff_mi() corrente al momento dello spegnimento (i pulsanti +/-
#     partono quindi dal valore corretto).
# 4.7.0 - 2026-08-29
#   - Kernel GPU: rilevamento analitico dell'interior PRIMA del loop. Bulbo
#     periodica-2 (chiuso) + cardioide principale intera via |1-sqrt(1-4c)|<1
#     (riscritta senza complessi), con prefiltro bounding-box. I pixel interni
#     (zona nera) saltano tutte le mi iterazioni: speedup misurato 1.6x a vista
#     iniziale, ~3x a zoom medio, ~9x a zoom profondo (mi=3000). Nessun falso
#     positivo (GPU/CPU identici al kernel originale).
#   - Kernel GPU compilato con --use_fast_math (log2f/powf/fminf/sqrtf piu'
#     economici; flag passato a cp.RawKernel come options=("--use_fast_math",)).
#   - Backend CPU: stesso test interior analitico (esclude cardioide/bulbo dal
#     loop) + loop in-place (np.square/z+=c, niente allocazioni): ~2.5x piu'
#     veloce, output identico (maxdiff=0 vs CPU originale).
# 4.6.0 - 2026-08-29
#   - Nuova palette "Termal": gradiente che parte col ghiaccio (blu/ciano
#     chiari), passa dal bianco gelido e finisce col fuoco (oro/arancio/rosso).
#   - Palette gestite da un registro PALETTES ordinato (fuoco/ghiaccio/termal,
#     ordine = indice passato al kernel): LUT __constant__ e selezione del
#     kernel, pulsanti UI e config sono tutti generati dal registro
#     (niente piu' if/else fuoco/ghiaccio).
# 4.5.3 - 2026-08-29
#   - Numero di versione nel titolo della finestra ("Insieme di Mandelbrot v4.5.3 - ...")
# 4.5.2 - 2026-08-29
#   - Benchmark sempre in float32 (GPU), indipendente da motore/precisione
#     selezionati nell'app (compute_gpu/compute accettano un prec esplicito)
# 4.5.1 - 2026-08-29
#   - Benchmark: prima dell'avvio e' mostrato un dialog che descrive il test
#     (regione, iterazioni, risoluzione, motore, durata) con OK (avvia) e
#     Cancel (annulla)
# 4.5.0 - 2026-08-29
#   - Barra di stato: tolto "centro" e "meta-larghezza", aggiunto il tempo di
#     rendering (misurato nel worker attorno a compute(), es. "render: 120 ms")
#   - Config persistente: tutti i settaggi (vista cx/cy/half, iterazioni, auto,
#     precisione, palette, motore) salvati e ricaricati da
#     %USERPROFILE%\mandelbrot\config.json a ogni esecuzione (salvataggio all'uscita
#     + throttled ~1s sui cambiamenti)
#   - Tasto "Reset": riporta vista + tutti i settaggi ai default, salva config
#   - Benchmark standardizzato (tasto "Benchmark"): regione fissa
#     c=(-0.74364388703, 0.13182590421i), meta=0.002, 3000 iter, 960x540,
#     ripetuta per 8 s in thread dedicato (buffer proprio, niente contesa con il
#     render normale); dialog finale con numero di ripetizioni, rate e ms/render
#   - compute_gpu/compute accettano un buffer opzionale (evita contesa su _BUF)
# 4.4.0 - 2026-08-29
#   - Auto-iterazioni raddoppiate: mi = 400 * (1 + log10(HALF0/half)),
#     clamp [50, 10000] (era 200 * (1 + log10(...)), clamp [50, 5000])
#   - Pulsanti manuali iterazioni: passo da +/-100 a +/-1000
# 4.3.0 - 2026-08-29
#   - UI: controlli, etichette, pulsanti -100/+100 e menu ingranditi (font
#     TkDefaultFont/TkTextFont/TkMenuFont a 13pt) + pady=3 sui controlli in alto
#     per area cliccabile piu ampia
# 4.2.0 - 2026-08-29
#   - Modalita "double" (precisione f64) disattivabile: kernel CUDA generati
#     in due varianti (float32/float64) dalla stessa sorgente parametrizzata;
#     radio "precisione: f32/f64" (default f32, f64 penalizzata su CUDA
#     consumer ~1:32 throughput float); la CPU resta sempre f64 (complex128).
#     Titolo e status mostrano la precisione attiva (es. "CUDA f64").
#   - Controlli e radio spostati in alto (prima: sotto il canvas)
#   - Menu File: "Salva immagine... (Ctrl+S)" + "Esci"; shortcut Ctrl+S;
#     salva l'immagine corrente del canvas via asksaveasfilename
# 4.1.0 - 2026-08-29
#   - Modalita "MI auto" (attiva di default, disattivabile: pulsante
#     "iter: auto"): le massime iterazioni crescono logaritmicamente con lo
#     zoom: mi = 200 * (1 + log10(HALF0/half)), clamp [50, 5000]. A vista
#     iniziale coincide con 200. Con auto attiva i pulsanti -100/+100 sono
#     disabilitati e l'etichetta mostra il valore calcolato.
# 4.0.0 - 2026-08-27
#   - FIX: la LUT passata come argomento device al kernel veniva sovrascritta
#     dall'output a ogni launch (colori magenta/azzurri random dal 2° frame in
#     poi). Palette ora incorporate nel kernel come array __constant__
#     generate dagli stessi dati Python; selezione palette via int (0=fuoco,
#     1=ghiaccio). Verificato: chiamate GPU consecutive identiche, diff vs CPU
#     solo su pixel di bordo.
#   - Toggle runtime CPU/CUDA (pulsanti, radio)
#   - Palette Fuoco/Ghiaccio (pulsanti, radio)
#   - LUT 256x3 condivisa CPU/GPU (colori identici sui due backend)
#   - Mappatura colore percepibile: gamma 0.35 su t=it/mi (fix schermata nera)
#   - Preview 1/4 + full render dopo 500ms anche su CPU
# 3.0.0
#   - Backend GPU CUDA (CuPy RawKernel, NVRTC): 2 px/thread, early-exit,
#     ricentramento algebrico, buffer device riutilizzato, fallback CPU
#   - Fattore d'aspetto corretto (y scale h/w)
# 2.0.0
#   - Pipeline asincrona latest-wins (worker + Condition, niente blocco UI)
#   - Zoom rotella / pan tasto sinistro, doppio click = reset vista
# 1.0.0
#   - Prima versione interattiva: app tkinter, rendering CPU numpy
# ============================================================================

import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import tkinter.font as tkfont
import threading
import queue
import math
import os
import json
import time
import numpy as np
from PIL import Image, ImageTk

# --- Costanti di vista e rendering ---
INIT_W, INIT_H = 960, 540
CX0, CY0, HALF0 = -0.5, 0.0, 1.5
MI0 = 200
MIN_HALF = 1e-12  # clamp minimo per half (evita zoom infinito -> half=0, stato degenere)
MIN_DIM = 50  # larghezza/altezza canvas minimi per considerare il canvas valido

# --- Percorsi e parametri ---
# Config salvata/caricata a ogni esecuzione (vista + tutti i settaggi)
CONFIG_PATH = os.path.join(os.path.expanduser("~"), "mandelbrot", "config.json")

# Benchmark: regione + parametri. Default = regione di zoom profondo (seme
# fornito dall'utente); i valori sono persistiti in config.json (overridibili).
BENCH = dict(cx=0.42663924626512445, cy=-0.3414973874054564, half=2.298743311298834e-06,
             mi=3000, w=960, h=540, secs=8.0)

VERSION = "4.11.1"

# ---------------- Palette (LUT 256x3 condivisa CPU/GPU) ----------------
_FIRE = (
    (0.0, 0.2, 0.45, 0.7, 0.9, 1.0),
    (0.05, 0.35, 0.85, 1.0, 1.0, 1.0),
    (0.0, 0.02, 0.2, 0.65, 0.95, 1.0),
    (0.0, 0.0, 0.02, 0.15, 0.55, 1.0),
)
_ICE = (
    (0.0, 0.25, 0.5, 0.75, 1.0),
    (0.02, 0.05, 0.30, 0.70, 1.0),
    (0.02, 0.15, 0.55, 0.85, 1.0),
    (0.10, 0.45, 0.90, 1.0, 1.0),
)
# Termal: parte col ghiaccio (blu/ciano chiari), passa dal bianco gelido e
# finisce col fuoco (oro/arancio/rosso). t=0 ghiaccio, t=1 fuoco.
_TERMAL = (
    (0.0,  0.2,  0.4,  0.55, 0.7,  0.85, 1.0),
    (0.02, 0.10, 0.55, 0.95, 1.00, 1.00, 1.00),
    (0.08, 0.45, 0.80, 0.96, 0.85, 0.55, 0.30),
    (0.28, 0.85, 0.95, 0.98, 0.45, 0.20, 0.10),
)

# Registro palette (fonte unica): l'ordine definisce l'indice passato al kernel
# (0=fuoco, 1=ghiaccio, 2=termal). UI, config e __constant__ del kernel sono
# tutti generati da questo dict.
PALETTES = {
    "fuoco": _FIRE,
    "ghiaccio": _ICE,
    "termal": _TERMAL,
}

def make_lut(pal):
    st, r, g, b = pal
    st = np.asarray(st, dtype=np.float64)
    t2 = np.linspace(0.0, 1.0, 256)
    rgb = np.stack([np.interp(t2, st, np.asarray(r)),
                    np.interp(t2, st, np.asarray(g)),
                    np.interp(t2, st, np.asarray(b))], axis=1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)

_PALETTE = "fuoco"
_LUT = make_lut(PALETTES["fuoco"])

def apply_palette(name):
    global _PALETTE, _LUT
    if name not in PALETTES:
        name = "fuoco"
    _PALETTE = name
    _LUT = make_lut(PALETTES[name])

# ---------------- GPU (CUDA) ----------------
# Tutte le palette (PALETTES) sono incorporate nel kernel come array __constant__:
# non si passano array device come argomenti (con la LUT come argomento
# il buffer di output sovrascriveva la memoria della LUT a ogni launch).
# Le dichiarazioni __constant__ e la selezione per indice pal sono generate
# dal registro PALETTES (vedi _build_kernel).
# Il kernel e' generato in due varianti di precisione (float32/float64)
# dalla stessa sorgente parametrizzata: f32 e' nativa su CUDA, f64 e'
# corretta ma penalizzata (~1:32 di throughput float) sulle GPU consumer.
def _fmt_lut(lut):
    return ", ".join(str(int(v)) for v in lut.ravel())

_KERNEL_TPL = r'''
@@CONSTS@@

__device__ __forceinline__ void pal_lut(@@T@@ t, const unsigned char* lut, unsigned char* rgb) {
    int idx = (int)(@@FMIN@@(1.0@@S@@, @@FMAX@@(0.0@@S@@, t)) * 255.0@@S@@);
    rgb[0] = lut[idx * 3 + 0];
    rgb[1] = lut[idx * 3 + 1];
    rgb[2] = lut[idx * 3 + 2];
}

__device__ __forceinline__ void process_pixel(
    int col, int row, int w, int h,
    @@T@@ cx, @@T@@ cy, @@T@@ half, int mi,
    const unsigned char* lut,
    unsigned char* __restrict__ out)
{
    if (col >= w) return;
    @@T@@ x0 = cx + half * ((@@T@@)(2 * col - w) / (@@T@@)w);
    @@T@@ y0 = cy + half * ((@@T@@)h / (@@T@@)w) * ((@@T@@)(2 * row - h) / (@@T@@)h);
    // Interior analitico: bulbo periodica-2 + cardioide principale
    // (c interno alla cardioide principale <=> |1 - sqrt(1 - 4c)| < 1,
    // riscritto senza complessi come R < 2*sqrt(0.5*(R + A)), con
    // A = 1 - 4*Re(c), R = |1 - 4c|). Se interno, il punto non diverge mai
    // -> pixel nero senza eseguire le mi iterazioni.
    // Prefiltro bounding-box dell'insieme: evita il costo della sqrt sui
    // pixel chiaramente esterni (fuggono in 1-2 iterazioni).
    if (x0 >= -2.0@@S@@ && x0 <= 0.4@@S@@ && y0 >= -1.3@@S@@ && y0 <= 1.3@@S@@) {
        unsigned char* p = out + (size_t)(row * w + col) * 3;
        @@T@@ d2 = (x0 + 1.0@@S@@) * (x0 + 1.0@@S@@) + y0 * y0;
        if (d2 <= 0.0625@@S@@) { p[0] = 0; p[1] = 0; p[2] = 0; return; }
        @@T@@ A = 1.0@@S@@ - 4.0@@S@@ * x0;
        @@T@@ B = -4.0@@S@@ * y0;
        @@T@@ R = @@SQRT@@(A * A + B * B);
        if (R < 2.0@@S@@ * @@SQRT@@(0.5@@S@@ * (R + A))) { p[0] = 0; p[1] = 0; p[2] = 0; return; }
    }
    @@T@@ a = cx * cx + (x0 - cx);
    @@T@@ two_cx = 2.0@@S@@ * cx;
    @@T@@ wr = -cx, wi = 0.0@@S@@;
    int it = 0;
    bool esc = false;
    @@T@@ mag2 = 0.0@@S@@;
    for (int i = 0; i < mi; ++i) {
        if (esc) break;
        @@T@@ nr = wr * wr - wi * wi + two_cx * wr + a;
        @@T@@ ni = two_cx * wi + 2.0@@S@@ * wr * wi + y0;
        wr = nr; wi = ni;
        @@T@@ zr = wr + cx;
        mag2 = zr * zr + wi * wi;
        if (mag2 > 4.0@@S@@) { esc = true; it = i; }
    }
    unsigned char* p = out + (size_t)(row * w + col) * 3;
    if (!esc) { p[0] = 0; p[1] = 0; p[2] = 0; return; }
    @@T@@ nu = (@@T@@)it + 1.0@@S@@ - @@LOG2@@(0.5@@S@@ * @@LOG@@(mag2));
    @@T@@ t = @@POW@@(@@FMIN@@(1.0@@S@@, @@FMAX@@(0.0@@S@@, nu / (@@T@@)mi)), 0.35@@S@@);
    unsigned char rgb[3];
    pal_lut(t, lut, rgb);
    p[0] = rgb[0]; p[1] = rgb[1]; p[2] = rgb[2];
}

extern "C" __global__ void __launch_bounds__(256) @@KNAME@@(
    unsigned char* __restrict__ out,
    int pal,
    @@T@@ cx, @@T@@ cy, @@T@@ half,
    int w, int h, int mi)
{
    @@LUTSELECT@@
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int col0 = tx * 2;
    process_pixel(col0, ty, w, h, cx, cy, half, mi, lut, out);
    process_pixel(col0 + 1, ty, w, h, cx, cy, half, mi, lut, out);
}
'''

_PRECS = {
    "f32": dict(T="float",  S="f", FMIN="fminf", FMAX="fmaxf",
                LOG2="log2f", LOG="logf", POW="powf", SQRT="sqrtf", KNAME="mandel_kernel_f32"),
    "f64": dict(T="double", S="",  FMIN="fmin",  FMAX="fmax",
                LOG2="log2",  LOG="log",  POW="pow",  SQRT="sqrt",  KNAME="mandel_kernel_f64"),
}

def _build_kernel(prec):
    d = _PRECS[prec]
    # Dichiarazioni __constant__ generate dal registro (una per palette).
    consts = "\n".join(
        "__constant__ unsigned char LUT_%s[768] = { %s };"
        % (name.upper(), _fmt_lut(make_lut(pal)))
        for name, pal in PALETTES.items())
    # Selezione LUT per indice pal (ordine PALETTES = indice).
    names = list(PALETTES)
    select = "    const unsigned char* lut = LUT_%s;" % names[0].upper()
    for i, name in enumerate(names[1:], start=1):
        select += "\n    if (pal == %d) lut = LUT_%s;" % (i, name.upper())
    src = (_KERNEL_TPL
           .replace("@@CONSTS@@", consts)
           .replace("@@LUTSELECT@@", select))
    for k, v in d.items():
        src = src.replace("@@" + k + "@@", v)
    return src, d["KNAME"]

_GPU = False
_KERNEL_F32 = None
_KERNEL_F64 = None
_BUF = None
try:
    import cupy as cp
    if cp.cuda.is_available():
        _src, _name = _build_kernel("f32")
        _KERNEL_F32 = cp.RawKernel(_src, _name, options=("--use_fast_math",))
        _GPU = True
        try:
            _src, _name = _build_kernel("f64")
            _KERNEL_F64 = cp.RawKernel(_src, _name, options=("--use_fast_math",))
        except Exception:
            _KERNEL_F64 = None
except Exception:
    _GPU = False

_PREC = "f32"

def set_prec(p):
    global _PREC
    if p == "f64" and _KERNEL_F64 is None:
        return False
    if p in ("f32", "f64"):
        _PREC = p
    return True

# Indice palette (0..N-1) preallocato come array size-1: evita np.asarray per render
_PAL_IDX = [np.asarray(i, dtype=np.int32) for i in range(len(PALETTES))]

def compute_gpu(cx, cy, half, w, h, mi, buf=None, prec=None):
    global _BUF
    need = w * h * 3
    if buf is None:
        if _BUF is None or _BUF.size < need:
            _BUF = cp.empty((need,), dtype=cp.uint8)
        buf = _BUF
    out = buf[:need]
    bx, by = 16, 16
    grid = ((w + 2 * bx - 1) // (2 * bx), (h + by - 1) // by)
    pal = list(PALETTES).index(_PALETTE)
    p = prec if prec in ("f32", "f64") else _PREC
    use64 = (p == "f64") and (_KERNEL_F64 is not None)
    kernel = _KERNEL_F64 if use64 else _KERNEL_F32
    fdt = np.float64 if use64 else np.float32
    args = (out,
            _PAL_IDX[pal],
            np.asarray(cx, dtype=fdt),
            np.asarray(cy, dtype=fdt),
            np.asarray(half, dtype=fdt),
            np.asarray(w, dtype=np.int32),
            np.asarray(h, dtype=np.int32),
            np.asarray(mi, dtype=np.int32))
    kernel(grid, (bx, by), args)
    return Image.fromarray(out.get().reshape((h, w, 3)))

# ---------------- CPU (fallback) ----------------

def compute_cpu(cx, cy, half, w, h, mi):
    xs = cx + half * (np.arange(w) - w / 2) / (w / 2)
    ys = cy + half * (h / w) * (np.arange(h) - h / 2) / (h / 2)
    real, imag = np.meshgrid(xs, ys)
    c = real + 1j * imag
    z = np.zeros_like(c)
    diverged = np.zeros(c.shape, dtype=bool)
    it = np.zeros(c.shape, dtype=np.int32)
    # Interior analitico (stesso criterio del kernel GPU): bulbo periodica-2 +
    # cardioide principale (|1 - sqrt(1 - 4c)| < 1). Questi pixel non divergono
    # mai -> restano it=0 (neri) ed esclusi dal loop: ~2.5x sulla CPU.
    w4 = 1.0 - 4.0 * c
    diverged |= (np.abs(w4) < 2.0 * np.sqrt(0.5 * (np.abs(w4) + np.real(w4))))
    diverged |= (np.abs(c + 1.0) <= 0.25)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(mi):
            if not (~diverged).any():
                break
            np.square(z, out=z)
            z += c
            m = ((z.real * z.real + z.imag * z.imag) > 4.0) & ~diverged
            if not m.any():
                continue
            diverged |= m
            it[m] = i
    t = np.power(np.clip(it / mi, 0.0, 1.0), 0.35).ravel()
    idx = (t * 255).astype(np.uint8)
    rgb = _LUT[idx].reshape((h, w, 3)).copy()
    rgb[it == 0] = 0
    return Image.fromarray(rgb)

# ---------------- Dispatch backend ----------------
_USE_GPU = _GPU

def backend():
    if _USE_GPU:
        return "CUDA " + _PREC
    return "CPU f64"

def compute(cx, cy, half, w, h, mi, buf=None, prec=None):
    if _USE_GPU:
        return compute_gpu(cx, cy, half, w, h, mi, buf=buf, prec=prec)
    return compute_cpu(cx, cy, half, w, h, mi)

def bench_engine():
    return "CUDA f32" if _GPU else "CPU f64 (GPU non disponibile)"

class MandelbrotApp:
    # ---------------- Costruzione UI ----------------
    def __init__(self, root):
        self.root = root
        self._setup_fonts()
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.mi = MI0
        self.mi_auto = True
        self.view_file = None
        self.bench = dict(BENCH)
        self._build_toolbar()
        self._build_canvas_status()
        self._build_menu()
        self._bind_events()
        self._start_pipeline()
        self._refresh_title()
        if self.load_config():
            self.request_render("config caricata")
        else:
            self.request_render("iniziale")

    def _setup_fonts(self):
        # UI piu leggibile: ingrandisce il font default di tutti i widget
        # (checkbutton, etichette, pulsanti, menu) preservando la famiglia nativa
        for _fn in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(_fn).config(size=13)
            except Exception:
                pass

    def _build_toolbar(self):
        self.canvas = tk.Canvas(self.root, width=INIT_W, height=INIT_H, bg="black", highlightthickness=0)
        # --- barra comandi in alto ---
        self.ctl = tk.Frame(self.root)
        self.ctl.pack(fill="x")
        bk = tk.Frame(self.ctl)
        bk.pack(side="left", padx=(8, 12))
        tk.Label(bk, text="Motore:").pack(side="left")
        self.cpu_btn = tk.Checkbutton(bk, text="CPU", command=lambda: self.set_backend("cpu"))
        self.cpu_btn.pack(side="left", padx=2, pady=3)
        self.cuda_btn = tk.Checkbutton(bk, text="CUDA", command=lambda: self.set_backend("cuda"))
        self.cuda_btn.pack(side="left", padx=2, pady=3)
        if _GPU:
            self.cuda_btn.select()
        else:
            self.cuda_btn.config(state="disabled")
            self.cpu_btn.select()
        pl = tk.Frame(self.ctl)
        pl.pack(side="left", padx=12)
        tk.Label(pl, text="Palette:").pack(side="left")
        # Pulsanti generati dal registro PALETTES (ordine = indice kernel).
        self.pal_btns = {}
        for _name in PALETTES:
            _b = tk.Checkbutton(pl, text=_name.capitalize(),
                                command=lambda n=_name: self.choose_palette(n))
            _b.pack(side="left", padx=2, pady=3)
            self.pal_btns[_name] = _b
        self.pal_btns["fuoco"].select()
        pc = tk.Frame(self.ctl)
        pc.pack(side="left", padx=12)
        tk.Label(pc, text="Precisione:").pack(side="left")
        self.f32_btn = tk.Checkbutton(pc, text="f32", command=lambda: self.set_precision("f32"))
        self.f32_btn.pack(side="left", padx=2, pady=3)
        self.f64_btn = tk.Checkbutton(pc, text="f64", command=lambda: self.set_precision("f64"))
        self.f64_btn.pack(side="left", padx=2, pady=3)
        self.f32_btn.select()
        if not _GPU:
            self.f32_btn.config(state="disabled")
            self.f64_btn.config(state="disabled")
        elif _KERNEL_F64 is None:
            self.f64_btn.config(state="disabled")
        mif = tk.Frame(self.ctl)
        mif.pack(side="left", padx=12)
        tk.Label(mif, text="Iter:").pack(side="left")
        self.mi_auto_var = tk.BooleanVar(value=True)
        self.auto_btn = tk.Checkbutton(mif, text="Auto", variable=self.mi_auto_var,
                                        command=self.toggle_auto_mi)
        self.auto_btn.pack(side="left", padx=2, pady=3)
        self.btns = tk.Frame(self.root)
        self.btns.pack(fill="x")
        self.mi_label = tk.Label(self.btns, text=f"Iterazioni: {self.mi}")
        self.mi_label.pack(side="left", padx=8, pady=3)
        self.mi_minus = tk.Button(self.btns, text="-1000", command=lambda: self.change_mi(-1000))
        self.mi_minus.pack(side="left", padx=2, pady=3)
        self.mi_plus = tk.Button(self.btns, text="+1000", command=lambda: self.change_mi(+1000))
        self.mi_plus.pack(side="left", padx=2, pady=3)
        self.mi_minus.config(state="disabled")
        self.mi_plus.config(state="disabled")
        self.bench_btn = tk.Button(self.btns, text="Benchmark", command=self.run_benchmark)
        self.bench_btn.pack(side="right", padx=(16, 8), pady=3)
        self.reset_btn = tk.Button(self.btns, text="Reset", command=self.reset)
        self.reset_btn.pack(side="right", padx=2, pady=3)

    def _build_canvas_status(self):
        # --- canvas al centro, status in fondo ---
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(self.root, text="render...")
        self.status.pack(fill="x")

    def _build_menu(self):
        self.menu = tk.Menu(self.root)
        self.mfile = tk.Menu(self.menu, tearoff=0)
        self.mfile.add_command(label="Salva immagine... (Ctrl+S)", command=self.save_png)
        self.mfile.add_command(label="Carica zona...", command=self.load_zone_as)
        self.mfile.add_command(label="Salva zona", command=self.save_zone)
        self.mfile.add_command(label="Salva zona con nome...", command=self.save_zone_as)
        self.mfile.add_separator()
        self.mfile.add_command(label="Esci", command=self.on_exit)
        self.menu.add_cascade(label="File", menu=self.mfile)
        self.root.config(menu=self.menu)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _bind_events(self):
        self.press_pos = None
        self.dragged = False
        self._size = (0, 0)
        self.root.bind("<Control-s>", lambda e: self.save_png())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Configure>", self.on_configure)
        self.canvas.bind("<Key-r>", lambda e: self.reset())
        self.canvas.bind("<Key-plus>", lambda e: self.zoom_center(2.0))
        self.canvas.bind("<Key-minus>", lambda e: self.zoom_center(0.5))

    # ---------------- Helper UI ----------------
    def _refresh_title(self):
        t = f"Insieme di Mandelbrot v{VERSION} - {backend()}"
        if self.view_file:
            t += " - " + os.path.basename(self.view_file)
        self.root.title(t)

    def _select_palette(self, name):
        apply_palette(name)
        name = _PALETTE
        for b in self.pal_btns.values():
            b.deselect()
        self.pal_btns[name].select()

    def _select_backend(self, be):
        global _USE_GPU
        if be == "cuda" and not _GPU:
            return False
        _USE_GPU = (be == "cuda")
        self.cpu_btn.deselect()
        self.cuda_btn.deselect()
        (self.cpu_btn if be == "cpu" else self.cuda_btn).select()
        return True

    def _select_precision(self, p):
        if p == "f64" and _KERNEL_F64 is None:
            return False
        if not set_prec(p):
            return False
        self.f32_btn.deselect()
        self.f64_btn.deselect()
        (self.f32_btn if p == "f32" else self.f64_btn).select()
        return True

    def _update_mi_label(self):
        if self.mi_auto:
            self.mi_label.config(text=f"Iterazioni: {self.eff_mi()} (auto)")
        else:
            self.mi_label.config(text=f"Iterazioni: {self.mi}")

    # ---------------- Vista e interazione (geometria + mouse/tastiera) ----------------
    def canvas_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < MIN_DIM or h < MIN_DIM:
            w, h = INIT_W, INIT_H
        return w, h

    def p2c(self, px, py):
        w, h = self.canvas_size()
        return (self.cx + (px - w / 2) / (w / 2) * self.half,
                self.cy + (py - h / 2) / (h / 2) * self.half * (h / w))

    def zoom_at(self, ux, uy, f):
        self.cx += (ux - self.cx) * (1 - 1 / f)
        self.cy += (uy - self.cy) * (1 - 1 / f)
        self.half = max(self.half / f, MIN_HALF)
        self.request_render("zoom")

    def zoom_center(self, f):
        self.zoom_at(self.cx, self.cy, f)

    def on_press(self, e):
        self.press_pos = (e.x, e.y)
        self.dragged = False

    def on_drag(self, e):
        if self.press_pos is None:
            return
        x0, y0 = self.press_pos
        if abs(e.x - x0) < 4 and abs(e.y - y0) < 4:
            return
        w, h = self.canvas_size()
        self.cx -= (e.x - x0) / (w / 2) * self.half
        self.cy -= (e.y - y0) / (h / 2) * self.half * (h / w)
        self.press_pos = (e.x, e.y)
        self.dragged = True
        self.request_render("pan")

    def on_release(self, e):
        if self.press_pos is not None and not self.dragged:
            self.zoom_at(*self.p2c(e.x, e.y), 2.0)
        self.press_pos = None

    def on_wheel(self, e):
        f = 1.25 if e.delta > 0 else 0.8
        self.zoom_at(*self.p2c(e.x, e.y), f)

    def on_configure(self, e):
        if (e.width, e.height) == self._size:
            return
        self._size = (e.width, e.height)
        if e.width < MIN_DIM or e.height < MIN_DIM:
            return
        self.request_render("ridimensionata")

    # ---------------- Controlli (MI, motore, palette, precisione, reset) ----------------
    def eff_mi(self):
        if not self.mi_auto:
            return self.mi
        z = HALF0 / max(self.half, 1e-12)
        return int(max(50, min(10000, 2 * MI0 * (1.0 + math.log10(z)))))

    def toggle_auto_mi(self):
        new_auto = self.mi_auto_var.get()
        if new_auto:
            self.mi_auto = True
        else:
            # Disattivando l'auto: congela self.mi sul valore auto corrente
            # (eff_mi calcolato con mi_auto ancora True), invece di mostrare
            # il valore fisso iniziale (stale).
            self.mi = self.eff_mi()
            self.mi_auto = False
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self._update_mi_label()
        self.request_render("MI: auto" if self.mi_auto else "MI: fissi")

    def change_mi(self, d):
        self.mi = max(50, self.mi + d)
        self._update_mi_label()
        self.request_render("iterazioni modificate")

    def set_backend(self, b):
        if not self._select_backend(b):
            return
        self._refresh_title()
        self.request_render("motore: " + b.upper())

    def choose_palette(self, name):
        self._select_palette(name)
        self.request_render("palette: " + _PALETTE)

    def set_precision(self, p):
        if not self._select_precision(p):
            return
        self._refresh_title()
        self.request_render("precisione: " + p)

    def reset(self):
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.mi = MI0
        self.mi_auto = True
        self.mi_auto_var.set(True)
        self.mi_minus.config(state="disabled")
        self.mi_plus.config(state="disabled")
        self._select_palette("fuoco")
        self._select_backend("cuda" if _GPU else "cpu")
        self._select_precision("f32")
        self._refresh_title()
        self._cfg_dirty = True
        self._update_mi_label()
        self.request_render("reset totale")

    # ---------------- Pipeline rendering (asincrona latest-wins) ----------------
    def _start_pipeline(self):
        # pipeline asincrona latest-wins (worker + Condition, niente blocco UI)
        self._cv = threading.Condition()
        self._job = None
        self._frames = queue.Queue()
        self._last_msg = ""
        self._full_timer = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._cfg_dirty = False
        self._bench_running = False
        self._bench_result = None
        self._bench_finished = False
        self.root.after(30, self._poll)
        self.root.after(1000, self._flush_config)

    def request_render(self, msg):
        self._last_msg = msg
        self._cfg_dirty = True
        self._update_mi_label()
        if self._full_timer is not None:
            self.root.after_cancel(self._full_timer)
            self._full_timer = None
        w, h = self.canvas_size()
        view = (self.cx, self.cy, self.half, self.eff_mi())
        self._submit(view, max(w // 4, 16), max(h // 4, 16))
        self._full_timer = self.root.after(500, lambda: self._maybe_full(view))

    def _maybe_full(self, view):
        self._full_timer = None
        if (self.cx, self.cy, self.half, self.eff_mi()) == view:
            w, h = self.canvas_size()
            self._submit(view, w, h)

    def _submit(self, view, w, h):
        with self._cv:
            self._job = (view, w, h)
            self._cv.notify()

    def _worker_loop(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job = self._job
                self._job = None
            view, w, h = job
            try:
                t0 = time.perf_counter()
                img = compute(view[0], view[1], view[2], w, h, view[3])
                rt = time.perf_counter() - t0
            except Exception:
                continue
            self._frames.put((img, self._last_msg, rt))

    def _poll(self):
        frame = None
        try:
            while True:
                frame = self._frames.get_nowait()
        except queue.Empty:
            pass
        if frame is not None:
            self._show(frame[0], frame[1], frame[2])
        if self._bench_finished:
            self._bench_finished = False
            count, secs, err = self._bench_result
            self._bench_done(count, secs, err)
        self.root.after(30, self._poll)

    def _show(self, img, msg, rt=0.0):
        w, h = self.canvas_size()
        if img.size != (w, h):
            img = img.resize((w, h), Image.NEAREST)
        self.pil = img
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.photo)
        self.status.config(text=f"{msg} | {backend()} | palette: {_PALETTE} | render: {rt*1000:.0f} ms")

    # ---------------- File: PNG, zona (JSON), config ----------------
    def save_png(self):
        if getattr(self, "pil", None) is None:
            self.status.config(text="niente immagine da salvare")
            return
        default = "mandelbrot_" + time.strftime("%Y%m%d_%H%M%S") + ".png"
        path = tk.filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".png", initialfile=default,
            filetypes=[("Immagini PNG", "*.png")])
        if not path:
            return
        try:
            self.pil.save(path, "PNG")
            self.status.config(text="salvata: " + path)
        except Exception as ex:
            self.status.config(text="errore salvataggio: " + str(ex))

    def save_zone(self):
        if self.view_file:
            self._save_zone_to(self.view_file)
        else:
            self.save_zone_as()

    def save_zone_as(self):
        default = "mandelbrot_" + time.strftime("%Y%m%d_%H%M%S") + ".json"
        path = tk.filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".json", initialfile=default,
            filetypes=[("File JSON", "*.json"), ("Tutti i file", "*.*")])
        if not path:
            return
        self._save_zone_to(path)

    def _save_zone_to(self, path):
        c = {
            "app": "mandelbrot",
            "versione": VERSION,
            "cx": self.cx,
            "cy": self.cy,
            "half": self.half,
            "mi": self.mi,
            "mi_auto": self.mi_auto,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=2, ensure_ascii=False)
                f.write("\n")
            self.view_file = path
            self._refresh_title()
            self.status.config(text="zona salvata: " + path)
        except Exception as ex:
            self.status.config(text="errore salvataggio zona: " + str(ex))

    def load_zone_as(self):
        path = tk.filedialog.askopenfilename(
            parent=self.root, title="Carica zona",
            filetypes=[("File JSON", "*.json"), ("Tutti i file", "*.*")])
        if not path:
            return
        self._load_zone_from(path)

    def _load_zone_from(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                c = json.load(f)
            self.cx = float(c["cx"])
            self.cy = float(c["cy"])
            self.half = max(float(c["half"]), MIN_HALF)
            self.mi = int(c.get("mi", self.mi))
            self.mi_auto = bool(c.get("mi_auto", self.mi_auto))
        except Exception as ex:
            self.status.config(text="errore caricamento zona: " + str(ex))
            return
        self.mi_auto_var.set(self.mi_auto)
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self.view_file = path
        self._refresh_title()
        self.request_render("zona caricata: " + os.path.basename(path))

    def save_config(self):
        c = dict(cx=self.cx, cy=self.cy, half=self.half,
                 mi=self.mi, mi_auto=bool(self.mi_auto),
                 precision=_PREC, palette=_PALETTE,
                 backend=("cuda" if _USE_GPU else "cpu"),
                 bench=dict(self.bench),
                 view_file=self.view_file)
        try:
            d = os.path.dirname(CONFIG_PATH)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=2)
        except Exception:
            pass

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return False
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                c = json.load(f)
        except Exception:
            return False
        self.cx = float(c.get("cx", self.cx))
        self.cy = float(c.get("cy", self.cy))
        self.half = float(c.get("half", self.half))
        self.mi = int(c.get("mi", self.mi))
        self.mi_auto = bool(c.get("mi_auto", self.mi_auto))
        self._load_bench(c.get("bench"))
        vf = c.get("view_file")
        self.view_file = vf if (isinstance(vf, str) and vf) else None
        self.mi_auto_var.set(self.mi_auto)
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self._select_precision(c.get("precision", "f32"))
        self._select_palette(c.get("palette", "fuoco"))
        be = c.get("backend", "cuda" if _GPU else "cpu")
        if be == "cuda" and not _GPU:
            be = "cpu"
        self._select_backend(be)
        self._refresh_title()
        self._update_mi_label()
        return True

    def _load_bench(self, b):
        if not isinstance(b, dict):
            return
        self.bench = dict(BENCH)
        self.bench["cx"] = float(b.get("cx", BENCH["cx"]))
        self.bench["cy"] = float(b.get("cy", BENCH["cy"]))
        self.bench["half"] = float(b.get("half", BENCH["half"]))
        self.bench["mi"] = int(b.get("mi", BENCH["mi"]))
        self.bench["w"] = int(b.get("w", BENCH["w"]))
        self.bench["h"] = int(b.get("h", BENCH["h"]))
        self.bench["secs"] = float(b.get("secs", BENCH["secs"]))

    def _flush_config(self):
        if self._cfg_dirty:
            self._cfg_dirty = False
            self.save_config()
        self.root.after(1000, self._flush_config)

    def on_exit(self):
        self._cfg_dirty = False
        try:
            self.save_config()
        except Exception:
            pass
        self.root.destroy()

    # ---------------- Benchmark ----------------
    def run_benchmark(self):
        if getattr(self, "_bench_running", False):
            self.status.config(text="benchmark gia' in corso")
            return
        b = self.bench
        msg = (
            "Benchmark standardizzato\n"
            f"  Regione: c=({b['cx']}, {b['cy']}i), meta={b['half']}\n"
            f"  Iterazioni: {b['mi']}   |   Risoluzione: {b['w']}x{b['h']}\n"
            f"  Motore: {bench_engine()} (sempre float32, indipendente dai settaggi)\n"
            f"  Durata: {b['secs']:.0f} s (loop continuo, poi report)\n\n"
            "Avviare il benchmark?"
        )
        if not tk.messagebox.askokcancel("Benchmark Mandelbrot", msg):
            self.status.config(text="benchmark annullato")
            return
        self._bench_running = True
        self.status.config(text="benchmark in corso (8 s)...")
        threading.Thread(target=self._bench_worker, daemon=True).start()

    def _bench_worker(self):
        b = self.bench
        need = b["w"] * b["h"] * 3
        bench_buf = None
        if _GPU:
            try:
                import cupy as cp
                bench_buf = cp.empty((need,), dtype=cp.uint8)
            except Exception:
                bench_buf = None
        def render():
            if _GPU:
                # benchmark sempre in float32, indipendentemente dai settaggi
                return compute_gpu(b["cx"], b["cy"], b["half"], b["w"], b["h"],
                                   b["mi"], buf=bench_buf, prec="f32")
            return compute_cpu(b["cx"], b["cy"], b["half"], b["w"], b["h"], b["mi"])
        t_end = time.perf_counter() + b["secs"]
        count = 0
        err = None
        while time.perf_counter() < t_end:
            try:
                render()
                count += 1
            except Exception as ex:
                err = str(ex)
                break
        # thread-safe: il thread principale (in _poll) rileva il flag e mostra il risultato
        self._bench_result = (count, b["secs"], err)
        self._bench_finished = True

    def _bench_done(self, count, secs, err):
        self._bench_running = False
        eng = bench_engine()
        b = self.bench
        if count > 0:
            msg = (f"Completati {count} rendering in {secs:.1f} s\n"
                   f"  {count/secs:.2f} rendering/s   |   {secs/count*1000:.0f} ms ciascuno\n\n"
                   f"Regione standard: c=({b['cx']}, {b['cy']}i), meta={b['half']}\n"
                   f"Iterazioni: {b['mi']}   |   Risoluzione: {b['w']}x{b['h']}\n"
                   f"Motore: {eng}")
        else:
            msg = (f"Benchmark fallito: {err}\n\n"
                   f"Regione standard: c=({b['cx']}, {b['cy']}i), meta={b['half']}, "
                   f"{b['mi']} iter, {b['w']}x{b['h']}")
        self.status.config(text="benchmark completato")
        tk.messagebox.showinfo("Benchmark Mandelbrot", msg)


def main():
    root = tk.Tk()
    MandelbrotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


