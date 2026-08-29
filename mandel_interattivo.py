# ============================================================================
# Insieme di Mandelbrot - visualizzatore interattivo
# VERSIONE: 4.0.0
# ----------------------------------------------------------------------------
# REGOLA: ogni modifica incrementa la versione e aggiunge una voce qui sotto
# (formato: versione - data - descrizione modifiche).
#
# STORICO:
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
import threading
import queue
import numpy as np
from PIL import Image, ImageTk

INIT_W, INIT_H = 960, 540
CX0, CY0, HALF0 = -0.5, 0.0, 1.5
MI0 = 200

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
def _fmt_lut(lut):
    return ", ".join(str(int(v)) for v in lut.ravel())

_KERNEL_SRC = r'''
__constant__ unsigned char LUT_FIRE[768] = { @@FIRE@@ };
__constant__ unsigned char LUT_ICE[768] = { @@ICE@@ };

__device__ __forceinline__ void pal_lut(float t, const unsigned char* lut, unsigned char* rgb) {
    int idx = (int)(fminf(1.0f, fmaxf(0.0f, t)) * 255.0f);
    rgb[0] = lut[idx * 3 + 0];
    rgb[1] = lut[idx * 3 + 1];
    rgb[2] = lut[idx * 3 + 2];
}

__device__ __forceinline__ void process_pixel(
    int col, int row, int w, int h,
    float cx, float cy, float half, int mi,
    const unsigned char* lut,
    unsigned char* __restrict__ out)
{
    if (col >= w) return;
    float x0 = cx + half * ((float)(2 * col - w) / (float)w);
    float y0 = cy + half * ((float)h / (float)w) * ((float)(2 * row - h) / (float)h);
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
    unsigned char* p = out + (size_t)(row * w + col) * 3;
    if (!esc) { p[0] = 0; p[1] = 0; p[2] = 0; return; }
    float nu = (float)it + 1.0f - log2f(0.5f * logf(mag2));
    float t = powf(fminf(1.0f, fmaxf(0.0f, nu / (float)mi)), 0.35f);
    unsigned char rgb[3];
    pal_lut(t, lut, rgb);
    p[0] = rgb[0]; p[1] = rgb[1]; p[2] = rgb[2];
}

extern "C" __global__ void __launch_bounds__(256) mandel_kernel(
    unsigned char* __restrict__ out,
    int pal,
    float cx, float cy, float half,
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

_GPU = False
_KERNEL = None
_BUF = None
try:
    import cupy as cp
    if cp.cuda.is_available():
        _src = (_KERNEL_SRC
                .replace("@@FIRE@@", _fmt_lut(make_lut(_FIRE)))
                .replace("@@ICE@@", _fmt_lut(make_lut(_ICE))))
        _KERNEL = cp.RawKernel(_src, "mandel_kernel")
        _GPU = True
except Exception:
    _GPU = False

def compute_gpu(cx, cy, half, w, h, mi):
    global _BUF
    need = w * h * 3
    if _BUF is None or _BUF.size < need:
        _BUF = cp.empty((need,), dtype=cp.uint8)
    out = _BUF[:need]
    bx, by = 16, 16
    grid = ((w + 2 * bx - 1) // (2 * bx), (h + by - 1) // by)
    pal = 1 if _PALETTE == "ghiaccio" else 0
    args = (out,
            np.asarray(pal, dtype=np.int32),
            np.asarray(cx, dtype=np.float32),
            np.asarray(cy, dtype=np.float32),
            np.asarray(half, dtype=np.float32),
            np.asarray(w, dtype=np.int32),
            np.asarray(h, dtype=np.int32),
            np.asarray(mi, dtype=np.int32))
    _KERNEL(grid, (bx, by), args)
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
    return "CUDA" if _USE_GPU else "CPU"

def compute(cx, cy, half, w, h, mi):
    if _USE_GPU:
        return compute_gpu(cx, cy, half, w, h, mi)
    return compute_cpu(cx, cy, half, w, h, mi)

class MandelbrotApp:
    def __init__(self, root):
        self.root = root
        root.title(f"Insieme di Mandelbrot - {backend()}")
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.mi = MI0
        self.canvas = tk.Canvas(root, width=INIT_W, height=INIT_H, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.btns = tk.Frame(root)
        self.btns.pack(fill="x")
        self.mi_label = tk.Label(self.btns, text=f"iterazioni: {self.mi}")
        self.mi_label.pack(side="left", padx=8)
        tk.Button(self.btns, text="-100", command=lambda: self.change_mi(-100)).pack(side="left", padx=2)
        tk.Button(self.btns, text="+100", command=lambda: self.change_mi(+100)).pack(side="left", padx=2)
        self.ctl = tk.Frame(root)
        self.ctl.pack(fill="x")
        bk = tk.Frame(self.ctl)
        bk.pack(side="left", padx=(8, 12))
        tk.Label(bk, text="motore:").pack(side="left")
        self.cpu_btn = tk.Checkbutton(bk, text="CPU", command=lambda: self.set_backend("cpu"))
        self.cpu_btn.pack(side="left", padx=2)
        self.cuda_btn = tk.Checkbutton(bk, text="CUDA", command=lambda: self.set_backend("cuda"))
        self.cuda_btn.pack(side="left", padx=2)
        if _GPU:
            self.cuda_btn.select()
        else:
            self.cuda_btn.config(state="disabled")
            self.cpu_btn.select()
        pl = tk.Frame(self.ctl)
        pl.pack(side="left", padx=12)
        tk.Label(pl, text="palette:").pack(side="left")
        self.fire_btn = tk.Checkbutton(pl, text="Fuoco", command=lambda: self.choose_palette("fuoco"))
        self.fire_btn.pack(side="left", padx=2)
        self.ice_btn = tk.Checkbutton(pl, text="Ghiaccio", command=lambda: self.choose_palette("ghiaccio"))
        self.ice_btn.pack(side="left", padx=2)
        self.fire_btn.select()
        self.status = tk.Label(root, text="render...")
        self.status.pack(fill="x")
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
        self.root.after(30, self._poll)
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
        if self._full_timer is not None:
            self.root.after_cancel(self._full_timer)
            self._full_timer = None
        w, h = self.canvas_size()
        view = (self.cx, self.cy, self.half, self.mi)
        self._submit(view, max(w // 4, 16), max(h // 4, 16))
        self._full_timer = self.root.after(500, lambda: self._maybe_full(view))

    def _maybe_full(self, view):
        self._full_timer = None
        if (self.cx, self.cy, self.half, self.mi) == view:
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
                img = compute(view[0], view[1], view[2], w, h, view[3])
            except Exception:
                continue
            self._frames.put((img, self._last_msg))

    def _poll(self):
        frame = None
        try:
            while True:
                frame = self._frames.get_nowait()
        except queue.Empty:
            pass
        if frame is not None:
            self._show(frame[0], frame[1])
        self.root.after(30, self._poll)

    def _show(self, img, msg):
        w, h = self.canvas_size()
        if img.size != (w, h):
            img = img.resize((w, h), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.photo)
        self.status.config(text=f"{msg} | {backend()} | palette: {_PALETTE} | centro: ({self.cx:.12f}, {self.cy:.12f}i) | meta-larghezza: {self.half:.3e}")

    def on_configure(self, e):
        if (e.width, e.height) == self._size:
            return
        self._size = (e.width, e.height)
        if e.width < 50 or e.height < 50:
            return
        self.request_render("ridimensionata")

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
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.request_render("reset")

def main():
    root = tk.Tk()
    MandelbrotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


