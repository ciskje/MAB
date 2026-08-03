#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esploratore interattivo dell'insieme di Mandelbrot con palette fuoco.

  * palette "fuoco" (nero -> rosso -> arancio -> giallo -> bianco)
  * controlli iterazioni: spinbox, pulsanti +/-, modalità automatica
  * mouse:
      - rotella           -> zoom+/dezoom centrato sul cursore
      - clic sinistro     -> zoom x2 sul punto
      - clic destro       -> dezoom x2 sul punto
      - trascina col sx   -> zoom sull'area selezionata
      - tasto R           -> reimposta la vista
  * barra di stato con zona visualizzata, cursore e tempi di rendering

Dipendenze: numpy  (pip install numpy)
"""

import math
import time
import tkinter as tk

import numpy as np


# ---------------------------------------------------------------------------
# Palette fuoco
# ---------------------------------------------------------------------------
def build_fire_palette(n=256):
    """LUT che va dal nero (insieme) al bianco caldo passando per il fuoco."""
    stops = [
        (0.00, (0, 0, 0)),
        (0.12, (35, 0, 5)),
        (0.30, (115, 10, 0)),
        (0.50, (195, 45, 0)),
        (0.70, (255, 110, 0)),
        (0.86, (255, 195, 40)),
        (1.00, (255, 255, 210)),
    ]
    t = np.linspace(0.0, 1.0, n)
    xs = [s[0] for s in stops]
    channels = [np.interp(t, xs, [s[1][ch] for s in stops]) for ch in range(3)]
    return np.stack(channels, axis=1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Calcolo dell'insieme (vettoriale, con colorazione continua "smooth")
# ---------------------------------------------------------------------------
def compute_mandelbrot(width, height, center_re, center_im, span_re, max_iter):
    span_im = span_re * height / width
    re = np.linspace(center_re - span_re / 2, center_re + span_re / 2, width)
    im = np.linspace(center_im + span_im / 2, center_im - span_im / 2, height)  # riga 0 in alto
    c = re[np.newaxis, :] + 1j * im[:, np.newaxis]

    z = np.zeros_like(c)
    smooth = np.full(c.shape, float(max_iter))   # i punti interni restano a max_iter
    active = np.ones(c.shape, dtype=bool)

    zr = z.ravel()
    cr = c.ravel()
    smooth_r = smooth.ravel()
    active_r = active.ravel()

    for i in range(1, max_iter + 1):
        idx = np.flatnonzero(active_r)
        if idx.size == 0:
            break
        zn = zr[idx] * zr[idx] + cr[idx]
        m2 = zn.real * zn.real + zn.imag * zn.imag
        zr[idx] = zn
        div = m2 > 4.0
        if div.any():
            div_idx = idx[div]
            # valore continuo per la sfumatura: n + 1 - log2(log2|z|)
            smooth_r[div_idx] = i + 1.0 - np.log2(0.5 * np.log2(m2[div]))
            active_r[div_idx] = False

    return smooth


# ---------------------------------------------------------------------------
# Applicazione
# ---------------------------------------------------------------------------
class MandelbrotExplorer:
    WHEEL_FACTOR = 1.4       # zoom per ogni "tacca" della rotella
    CLICK_FACTOR = 2.0       # zoom per clic sinistro/destro
    MIN_SPAN = 1e-13         # limite pratico della precisione float64
    MAX_SPAN = 20.0
    INIT_CENTER = (-0.6, 0.0)
    INIT_SPAN = 3.2

    def __init__(self, root):
        self.root = root
        root.title("Insieme di Mandelbrot – palette fuoco")

        self.lut = build_fire_palette()
        self.center_re, self.center_im = self.INIT_CENTER
        self.span_re = self.INIT_SPAN
        self.width = None
        self.height = None
        self.photo = None
        self.img_id = None
        self._drag_start = None
        self._resize_job = None
        self.render_time = 0.0

        self.max_iter = tk.IntVar(value=200)
        self.auto_iter = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_events()

    # ------------------------- interfaccia -------------------------
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        tk.Label(top, text="Iterazioni:").pack(side=tk.LEFT)
        self.spin = tk.Spinbox(top, from_=20, to=10000, increment=50,
                               textvariable=self.max_iter, width=7,
                               command=self.render)
        self.spin.pack(side=tk.LEFT, padx=(4, 8))
        self.spin.bind("<Return>", lambda e: self.render())

        self.btn_minus = tk.Button(top, text="- 100", command=lambda: self._change_iter(-100))
        self.btn_plus = tk.Button(top, text="+ 100", command=lambda: self._change_iter(+100))
        self.btn_minus.pack(side=tk.LEFT)
        self.btn_plus.pack(side=tk.LEFT, padx=(2, 12))

        tk.Checkbutton(top, text="Iterazioni automatiche (in base allo zoom)",
                       variable=self.auto_iter,
                       command=self._on_auto_iter).pack(side=tk.LEFT)

        tk.Button(top, text="Reimposta vista (R)", command=self.reset_view).pack(side=tk.RIGHT)

        hint = ("Rotella: zoom/dezoom  •  Trascina col sinistro: zoom su un'area  •  "
                "Clic sinistro: zoom x2  •  Clic destro: dezoom x2  •  R: reset")
        tk.Label(self.root, text=hint, anchor="w", fg="#555555").pack(side=tk.TOP, fill=tk.X, padx=6)

        # barra di stato (creata PRIMA del canvas così resta sempre visibile)
        bottom = tk.Frame(self.root, bg="#f0f0f0")
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_left = tk.Label(bottom, anchor="w", bg="#f0f0f0")
        self.status_left.pack(side=tk.LEFT, padx=6, pady=2)
        self.status_right = tk.Label(bottom, anchor="e", bg="#f0f0f0")
        self.status_right.pack(side=tk.RIGHT, padx=6, pady=2)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._update_status()

    def _bind_events(self):
        c = self.canvas
        c.bind("<Configure>", self._on_configure)
        c.bind("<Motion>", self._on_motion)
        c.bind("<MouseWheel>", self._on_wheel)    # Windows / macOS
        c.bind("<Button-4>", self._on_wheel)      # Linux: scroll su
        c.bind("<Button-5>", self._on_wheel)      # Linux: scroll giù
        c.bind("<ButtonPress-1>", self._on_press_left)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release_left)
        c.bind("<ButtonPress-3>", self._on_right_click)
        self.root.bind("<Key>", self._on_key)

    # ------------------------- rendering -------------------------
    def render(self):
        if not self.width or not self.height:
            return
        self._apply_auto_iter()
        iters = self.max_iter.get()

        t0 = time.perf_counter()
        smooth = compute_mandelbrot(self.width, self.height,
                                    self.center_re, self.center_im,
                                    self.span_re, iters)
        rgb = self._colorize(smooth, iters)
        self.photo = self._to_photoimage(rgb)

        if self.img_id is None:
            self.img_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.coords(self.img_id, 0, 0)
            self.canvas.itemconfig(self.img_id, image=self.photo)

        self.render_time = time.perf_counter() - t0
        self._update_status()

    def _colorize(self, smooth, max_iter):
        inside = smooth >= max_iter
        t = np.sqrt(np.clip(smooth / max_iter, 0.0, 1.0))
        idx = np.minimum((t * len(self.lut)).astype(int), len(self.lut) - 1)
        rgb = self.lut[idx]
        rgb[inside] = (0, 0, 0)      # l'interno dell'insieme resta nero
        return rgb

    @staticmethod
    def _to_photoimage(rgb):
        h, w, _ = rgb.shape
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        return tk.PhotoImage(data=header + np.ascontiguousarray(rgb).tobytes())

    # ------------------------- geometria -------------------------
    def _pixel_to_complex(self, px, py):
        s = self.span_re / self.width            # unità complesse per pixel
        return (self.center_re + (px - self.width / 2.0) * s,
                self.center_im - (py - self.height / 2.0) * s)

    def _zoom_at(self, px, py, factor):
        new_span = max(self.MIN_SPAN, min(self.MAX_SPAN, self.span_re / factor))
        factor = self.span_re / new_span
        if factor == 1.0:
            return
        cre, cim = self._pixel_to_complex(px, py)
        fx, fy = px / self.width, py / self.height
        span_y = new_span * self.height / self.width
        self.center_re = (cre - fx * new_span) + new_span / 2
        self.center_im = (cim + fy * span_y) - span_y / 2
        self.span_re = new_span
        self.render()

    # ------------------------- eventi mouse -------------------------
    def _on_wheel(self, event):
        up = (getattr(event, "num", None) == 4) or (event.delta > 0)
        self._zoom_at(event.x, event.y,
                      self.WHEEL_FACTOR if up else 1.0 / self.WHEEL_FACTOR)

    def _on_press_left(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        self.canvas.delete("rubberband")
        self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                     outline="#ffdd55", dash=(4, 3), tags="rubberband")
        self._update_status(mouse=(event.x, event.y))

    def _on_release_left(self, event):
        self.canvas.delete("rubberband")
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        w_px, h_px = abs(event.x - x0), abs(event.y - y0)

        if w_px < 6 or h_px < 6:                 # clic semplice: zoom x2 sul punto
            self._zoom_at(event.x, event.y, self.CLICK_FACTOR)
            return

        cre, cim = self._pixel_to_complex((x0 + event.x) / 2, (y0 + event.y) / 2)
        s = self.span_re / self.width
        new_span = max(self.MIN_SPAN, s * max(w_px, h_px * self.width / self.height))
        self.center_re, self.center_im, self.span_re = cre, cim, new_span
        self.render()

    def _on_right_click(self, event):
        self._zoom_at(event.x, event.y, 1.0 / self.CLICK_FACTOR)

    def _on_motion(self, event):
        self._update_status(mouse=(event.x, event.y))

    def _on_configure(self, event):
        self.width, self.height = event.width, event.height
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, self.render)

    # ------------------------- iterazioni -------------------------
    def _change_iter(self, delta):
        if self.auto_iter.get():
            return
        val = max(20, min(10000, self.max_iter.get() + delta))
        self.max_iter.set(val)
        self.render()

    def _on_auto_iter(self):
        state = "disabled" if self.auto_iter.get() else "normal"
        self.spin.config(state=state)
        self.btn_minus.config(state=state)
        self.btn_plus.config(state=state)
        self.render()

    def _apply_auto_iter(self):
        if not self.auto_iter.get():
            return
        zoom = self.INIT_SPAN / self.span_re
        iters = int(100 + 120 * math.log10(max(zoom, 1.0)))
        self.max_iter.set(max(60, min(8000, iters)))

    # ------------------------- varie -------------------------
    def reset_view(self):
        self.center_re, self.center_im = self.INIT_CENTER
        self.span_re = self.INIT_SPAN
        self.render()

    def _on_key(self, event):
        if self.root.focus_get() is self.spin:
            return
        if event.char in ("+", "="):
            self._change_iter(+100)
        elif event.char == "-":
            self._change_iter(-100)
        elif event.char in ("r", "R"):
            self.reset_view()

    def _update_status(self, mouse=None):
        zoom = self.INIT_SPAN / self.span_re
        right = (f"Centro: ({self.center_re:+.10g}, {self.center_im:+.10g})   "
                 f"Δre: {self.span_re:.4g}   Zoom: x{zoom:.6g}   "
                 f"Iterazioni: {self.max_iter.get()}   "
                 f"Render: {self.render_time * 1000:.0f} ms")
        if mouse is not None:
            cre, cim = self._pixel_to_complex(*mouse)
            left = (f"c = {cre:+.12g} {cim:+.12g}i    |c| = {math.hypot(cre, cim):.6g}    "
                    f"pixel: ({mouse[0]}, {mouse[1]})")
        else:
            left = "Muovi il mouse sul frattale per leggere le coordinate del punto"
        self.status_left.config(text=left)
        self.status_right.config(text=right)


def main():
    root = tk.Tk()
    root.geometry("1024x720")
    MandelbrotExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
