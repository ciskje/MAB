# ============================================================================
# Insieme di Mandelbrot - visualizzatore interattivo
# VERSIONE: 4.5.0
# ----------------------------------------------------------------------------
# REGOLA: ogni modifica incrementa la versione e aggiunge una voce qui sotto
# (formato: versione - data - descrizione modifiche).
#
# STORICO:
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

INIT_W, INIT_H = 960, 540
CX0, CY0, HALF0 = -0.5, 0.0, 1.5
MI0 = 200

# Config salvata/caricata a ogni esecuzione (vista + tutti i settaggi)
CONFIG_PATH = os.path.join(os.path.expanduser("~"), "mandelbrot", "config.json")

# Benchmark standardizzato: regione fissa, >=3000 iterazioni, risoluzione fissa
BENCH = dict(cx=-0.74364388703, cy=0.13182590421, half=0.002,
             mi=3000, w=960, h=540, secs=8.0)

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

def make_lut(pal):
    st, r, g, b = pal
    st = np.asarray(st, dtype=np.float64)
    t2 = np.linspace(0.0, 1.0, 256)
    rgb = np.stack([np.interp(t2, st, np.asarray(r)),
                    np.interp(t2, st, np.asarray(g)),
                    np.interp(t2, st, np.asarray(b))], axis=1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)

_PALETTE = "fuoco"
_LUT = make_lut(_FIRE)

def apply_palette(name):
    global _PALETTE, _LUT
    _PALETTE = name
    _LUT = make_lut(_FIRE if name == "fuoco" else _ICE)

# ---------------- GPU (CUDA) ----------------
# Le due palette sono incorporate nel kernel come array __constant__:
# non si passano array device come argomenti (con la LUT come argomento
# il buffer di output sovrascriveva la memoria della LUT a ogni launch).
# Il kernel e' generato in due varianti di precisione (float32/float64)
# dalla stessa sorgente parametrizzata: f32 e' nativa su CUDA, f64 e'
# corretta ma penalizzata (~1:32 di throughput float) sulle GPU consumer.
def _fmt_lut(lut):
    return ", ".join(str(int(v)) for v in lut.ravel())

_KERNEL_TPL = r'''
__constant__ unsigned char LUT_FIRE[768] = { @@FIRE@@ };
__constant__ unsigned char LUT_ICE[768] = { @@ICE@@ };

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
    const unsigned char* lut = (pal == 1) ? LUT_ICE : LUT_FIRE;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int col0 = tx * 2;
    process_pixel(col0, ty, w, h, cx, cy, half, mi, lut, out);
    process_pixel(col0 + 1, ty, w, h, cx, cy, half, mi, lut, out);
}
'''

_PRECS = {
    "f32": dict(T="float",  S="f", FMIN="fminf", FMAX="fmaxf",
                LOG2="log2f", LOG="logf", POW="powf", KNAME="mandel_kernel_f32"),
    "f64": dict(T="double", S="",  FMIN="fmin",  FMAX="fmax",
                LOG2="log2",  LOG="log",  POW="pow",  KNAME="mandel_kernel_f64"),
}

def _build_kernel(prec):
    d = _PRECS[prec]
    src = (_KERNEL_TPL
           .replace("@@FIRE@@", _fmt_lut(make_lut(_FIRE)))
           .replace("@@ICE@@", _fmt_lut(make_lut(_ICE))))
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
        _KERNEL_F32 = cp.RawKernel(_src, _name)
        _GPU = True
        try:
            _src, _name = _build_kernel("f64")
            _KERNEL_F64 = cp.RawKernel(_src, _name)
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

def prec():
    return _PREC

def compute_gpu(cx, cy, half, w, h, mi, buf=None):
    global _BUF
    need = w * h * 3
    if buf is None:
        if _BUF is None or _BUF.size < need:
            _BUF = cp.empty((need,), dtype=cp.uint8)
        buf = _BUF
    out = buf[:need]
    bx, by = 16, 16
    grid = ((w + 2 * bx - 1) // (2 * bx), (h + by - 1) // by)
    pal = 1 if _PALETTE == "ghiaccio" else 0
    use64 = (_PREC == "f64") and (_KERNEL_F64 is not None)
    kernel = _KERNEL_F64 if use64 else _KERNEL_F32
    fdt = np.float64 if use64 else np.float32
    args = (out,
            np.asarray(pal, dtype=np.int32),
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
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(mi):
            z = z * z + c
            m = (np.abs(z) > 2) & ~diverged
            if not m.any():
                if diverged.all():
                    break
                continue
            diverged |= m
            it[m] = i
    t = np.power(np.clip(it / mi, 0.0, 1.0), 0.35).ravel()
    idx = (t * 255).astype(np.uint8)
    rgb = _LUT[idx].reshape((h, w, 3)).copy()
    rgb[it == 0] = 0
    return Image.fromarray(rgb)

_USE_GPU = _GPU

def backend():
    if _USE_GPU:
        return "CUDA " + _PREC
    return "CPU f64"

def compute(cx, cy, half, w, h, mi, buf=None):
    if _USE_GPU:
        return compute_gpu(cx, cy, half, w, h, mi, buf=buf)
    return compute_cpu(cx, cy, half, w, h, mi)

class MandelbrotApp:
    def __init__(self, root):
        self.root = root
        # UI piu leggibile: ingrandisce font default di tutti i widget (checkbutton,
        # etichette, pulsanti, menu) preservando la famiglia nativa
        for _fn in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(_fn).config(size=13)
            except Exception:
                pass
        root.title(f"Insieme di Mandelbrot - {backend()}")
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.mi = MI0
        self.mi_auto = True
        self.canvas = tk.Canvas(root, width=INIT_W, height=INIT_H, bg="black", highlightthickness=0)
        # --- barra comandi in alto ---
        self.ctl = tk.Frame(root)
        self.ctl.pack(fill="x")
        bk = tk.Frame(self.ctl)
        bk.pack(side="left", padx=(8, 12))
        tk.Label(bk, text="motore:").pack(side="left")
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
        tk.Label(pl, text="palette:").pack(side="left")
        self.fire_btn = tk.Checkbutton(pl, text="Fuoco", command=lambda: self.choose_palette("fuoco"))
        self.fire_btn.pack(side="left", padx=2, pady=3)
        self.ice_btn = tk.Checkbutton(pl, text="Ghiaccio", command=lambda: self.choose_palette("ghiaccio"))
        self.ice_btn.pack(side="left", padx=2, pady=3)
        self.fire_btn.select()
        pc = tk.Frame(self.ctl)
        pc.pack(side="left", padx=12)
        tk.Label(pc, text="precisione:").pack(side="left")
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
        tk.Label(mif, text="iter:").pack(side="left")
        self.mi_auto_var = tk.BooleanVar(value=True)
        self.auto_btn = tk.Checkbutton(mif, text="auto", variable=self.mi_auto_var,
                                        command=self.toggle_auto_mi)
        self.auto_btn.pack(side="left", padx=2, pady=3)
        self.btns = tk.Frame(root)
        self.btns.pack(fill="x")
        self.mi_label = tk.Label(self.btns, text=f"iterazioni: {self.mi}")
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
        # --- canvas al centro, status in fondo ---
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(root, text="render...")
        self.status.pack(fill="x")
        # --- menu File ---
        self.menu = tk.Menu(root)
        self.mfile = tk.Menu(self.menu, tearoff=0)
        self.mfile.add_command(label="Salva immagine... (Ctrl+S)", command=self.save_png)
        self.mfile.add_separator()
        self.mfile.add_command(label="Esci", command=self.on_exit)
        self.menu.add_cascade(label="File", menu=self.mfile)
        root.config(menu=self.menu)
        root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.root.bind("<Control-s>", lambda e: self.save_png())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Configure>", self.on_configure)
        self.canvas.bind("<Key-r>", lambda e: self.reset())
        self.canvas.bind("<Key-plus>", lambda e: self.zoom_center(2.0))
        self.canvas.bind("<Key-minus>", lambda e: self.zoom_center(0.5))
        self.press_pos = None
        self.dragged = False
        self._size = (0, 0)
        # pipeline asincrona latest-wins
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
        if self.load_config():
            self.request_render("config caricata")
        else:
            self.request_render("iniziale")

    def canvas_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50 or h < 50:
            w, h = INIT_W, INIT_H
        return w, h

    def p2c(self, px, py):
        w, h = self.canvas_size()
        return (self.cx + (px - w / 2) / (w / 2) * self.half,
                self.cy + (py - h / 2) / (h / 2) * self.half * (h / w))

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

    def on_configure(self, e):
        if (e.width, e.height) == self._size:
            return
        self._size = (e.width, e.height)
        if e.width < 50 or e.height < 50:
            return
        self.request_render("ridimensionata")

    def eff_mi(self):
        if not self.mi_auto:
            return self.mi
        z = HALF0 / max(self.half, 1e-12)
        return int(max(50, min(10000, 2 * MI0 * (1.0 + math.log10(z)))))

    def _update_mi_label(self):
        if self.mi_auto:
            self.mi_label.config(text=f"iterazioni: {self.eff_mi()} (auto)")
        else:
            self.mi_label.config(text=f"iterazioni: {self.mi}")

    def toggle_auto_mi(self):
        self.mi_auto = self.mi_auto_var.get()
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self._update_mi_label()
        self.request_render("MI: auto" if self.mi_auto else "MI: fissi")

    def change_mi(self, d):
        self.mi = max(50, self.mi + d)
        self.mi_label.config(text=f"iterazioni: {self.mi}")
        self.request_render("iterazioni modificate")

    def set_backend(self, b):
        global _USE_GPU
        if b == "cuda" and not _GPU:
            return
        _USE_GPU = (b == "cuda")
        self.cpu_btn.deselect()
        self.cuda_btn.deselect()
        (self.cpu_btn if b == "cpu" else self.cuda_btn).select()
        self.root.title(f"Insieme di Mandelbrot - {backend()}")
        self.request_render("motore: " + b.upper())

    def choose_palette(self, name):
        apply_palette(name)
        self.fire_btn.deselect()
        self.ice_btn.deselect()
        (self.fire_btn if name == "fuoco" else self.ice_btn).select()
        self.request_render("palette: " + name)

    def set_precision(self, p):
        if p == "f64" and _KERNEL_F64 is None:
            return
        if set_prec(p):
            self.f32_btn.deselect()
            self.f64_btn.deselect()
            (self.f32_btn if p == "f32" else self.f64_btn).select()
        self.root.title(f"Insieme di Mandelbrot - {backend()}")
        self.request_render("precisione: " + p)

    def save_png(self):
        import time as _t
        if getattr(self, "pil", None) is None:
            self.status.config(text="niente immagine da salvare")
            return
        default = "mandelbrot_" + _t.strftime("%Y%m%d_%H%M%S") + ".png"
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

    # ---------------- config.json (salva/carica tutti i settaggi) ----------------
    def save_config(self):
        global _PREC, _PALETTE, _USE_GPU
        c = dict(cx=self.cx, cy=self.cy, half=self.half,
                 mi=self.mi, mi_auto=bool(self.mi_auto),
                 precision=_PREC, palette=_PALETTE,
                 backend=("cuda" if _USE_GPU else "cpu"))
        try:
            d = os.path.dirname(CONFIG_PATH)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=2)
        except Exception:
            pass

    def load_config(self):
        global _PREC, _PALETTE, _USE_GPU
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
        self.mi_auto_var.set(self.mi_auto)
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        if set_prec(c.get("precision", "f32")):
            self.f32_btn.deselect()
            self.f64_btn.deselect()
            (self.f32_btn if _PREC == "f32" else self.f64_btn).select()
        pal = c.get("palette", "fuoco")
        apply_palette(pal)
        self.fire_btn.deselect()
        self.ice_btn.deselect()
        (self.fire_btn if pal == "fuoco" else self.ice_btn).select()
        be = c.get("backend", "cuda" if _GPU else "cpu")
        if be == "cuda" and not _GPU:
            be = "cpu"
        _USE_GPU = (be == "cuda")
        self.cpu_btn.deselect()
        self.cuda_btn.deselect()
        (self.cpu_btn if be == "cpu" else self.cuda_btn).select()
        self.root.title(f"Insieme di Mandelbrot - {backend()}")
        self._update_mi_label()
        return True

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

    # ---------------- benchmark standardizzato ----------------
    def run_benchmark(self):
        if getattr(self, "_bench_running", False):
            self.status.config(text="benchmark gia' in corso")
            return
        self._bench_running = True
        self.status.config(text="benchmark in corso (8 s)...")
        threading.Thread(target=self._bench_worker, daemon=True).start()

    def _bench_worker(self):
        b = BENCH
        need = b["w"] * b["h"] * 3
        bench_buf = None
        if _USE_GPU and _GPU:
            try:
                import cupy as cp
                bench_buf = cp.empty((need,), dtype=cp.uint8)
            except Exception:
                bench_buf = None
        t_end = time.perf_counter() + b["secs"]
        count = 0
        err = None
        while time.perf_counter() < t_end:
            try:
                compute(b["cx"], b["cy"], b["half"], b["w"], b["h"], b["mi"], buf=bench_buf)
                count += 1
            except Exception as ex:
                err = str(ex)
                break
        # thread-safe: il thread principale (in _poll) rileva il flag e mostra il risultato
        self._bench_result = (count, b["secs"], err)
        self._bench_finished = True

    def _bench_done(self, count, secs, err):
        self._bench_running = False
        eng = backend()
        if count > 0:
            msg = (f"Completati {count} rendering in {secs:.1f} s\n"
                   f"  {count/secs:.2f} rendering/s   |   {secs/count*1000:.0f} ms ciascuno\n\n"
                   f"Regione standard: c=({BENCH['cx']}, {BENCH['cy']}i), meta={BENCH['half']}\n"
                   f"Iterazioni: {BENCH['mi']}   |   Risoluzione: {BENCH['w']}x{BENCH['h']}\n"
                   f"Motore: {eng}")
        else:
            msg = (f"Benchmark fallito: {err}\n\n"
                   f"Regione standard: c=({BENCH['cx']}, {BENCH['cy']}i), meta={BENCH['half']}, "
                   f"{BENCH['mi']} iter, {BENCH['w']}x{BENCH['h']}")
        self.status.config(text="benchmark completato")
        tk.messagebox.showinfo("Benchmark Mandelbrot", msg)

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

    def zoom_at(self, ux, uy, f):
        self.cx += (ux - self.cx) * (1 - 1 / f)
        self.cy += (uy - self.cy) * (1 - 1 / f)
        self.half /= f
        self.request_render("zoom")

    def zoom_center(self, f):
        self.zoom_at(self.cx, self.cy, f)

    def reset(self):
        global _PREC, _PALETTE, _USE_GPU
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.mi = MI0
        self.mi_auto = True
        self.mi_auto_var.set(True)
        self.mi_minus.config(state="disabled")
        self.mi_plus.config(state="disabled")
        apply_palette("fuoco")
        self.fire_btn.deselect()
        self.ice_btn.deselect()
        self.fire_btn.select()
        _USE_GPU = _GPU
        self.cpu_btn.deselect()
        self.cuda_btn.deselect()
        (self.cuda_btn if _GPU else self.cpu_btn).select()
        if _GPU:
            set_prec("f32")
            self.f32_btn.deselect()
            self.f64_btn.deselect()
            self.f32_btn.select()
        self.root.title(f"Insieme di Mandelbrot - {backend()}")
        self._cfg_dirty = True
        self._update_mi_label()
        self.request_render("reset totale")

def main():
    root = tk.Tk()
    MandelbrotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


