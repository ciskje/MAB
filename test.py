#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esploratore interattivo dell'insieme di Mandelbrot con palette fuoco.
Versione accelerata con CUDA/GPU e ottimizzata per CPU multi-core.

  * palette "fuoco" (nero -> rosso -> arancio -> giallo -> bianco)
  * controlli iterazioni: spinbox, pulsanti +/-, modalità automatica
  * mouse:
      - rotella           -> zoom+/dezoom centrato sul cursore
      - clic sinistro     -> zoom x2 sul punto
      - clic destro       -> dezoom x2 sul punto
      - trascina col sx   -> zoom sull'area selezionata
      - tasto R           -> reimposta la vista
  * barra di stato con zona visualizzata, cursore e tempi di rendering

Dipendenze: torch, numpy, tkinter  (pip install torch numpy)

Performance:
  - GPU CUDA: fino a 100x più veloce rispetto alla CPU (se disponibile)
  - CPU multi-core: sfrutta tutti i core disponibili tramite PyTorch
    Le operazioni vettoriali sono parallelizzate automaticamente su tutti i thread
"""

import math
import os
import time
import tkinter as tk

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configurazione dispositivo e ottimizzazione multi-core CPU
# ---------------------------------------------------------------------------
def get_device():
    """
    Restituisce il dispositivo CUDA se disponibile, altrimenti CPU.
    Ottimizza automaticamente il numero di thread per CPU multi-core.
    
    Nota sulle performance CPU:
    - PyTorch usa automaticamente tutti i core disponibili per le operazioni vettoriali
    - Le operazioni sui tensori sono parallelizzate a livello di BLAS/OpenMP
    - Su CPU multi-core, il calcolo è significativamente più veloce rispetto al codice single-thread
    """
    # Rileva i core disponibili
    cpu_cores = os.cpu_count() or 4
    
    # Configura PyTorch per usare tutti i core (meno uno per la UI su sistemi con molti core)
    # Su sistemi con pochi core (<4), usa tutti i core disponibili
    num_threads = cpu_cores if cpu_cores <= 4 else cpu_cores - 1
    torch.set_num_threads(num_threads)
    
    print(f"CPU rilevata: {cpu_cores} core")
    print(f"Thread PyTorch configurati: {num_threads}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA disponibile: {torch.cuda.get_device_name(0)}")
        print(f"Memoria CUDA totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("→ Modalità: GPU-accelerated (massime prestazioni)")
    else:
        device = torch.device("cpu")
        print("CUDA non disponibile → esecuzione su CPU multi-core")
        print("→ Modalità: CPU multi-threaded (buone prestazioni)")
        print(f"   Il codice sfrutta tutti i {num_threads} thread per il calcolo parallelo")
    return device


DEVICE = get_device()


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
    t = torch.linspace(0.0, 1.0, n, device=DEVICE)
    xs = torch.tensor([s[0] for s in stops], device=DEVICE)
    channels = []
    for ch in range(3):
        values = torch.tensor([s[1][ch] for s in stops], dtype=torch.float32, device=DEVICE)
        # Interpolazione lineare manuale per tensori CUDA
        ch_vals = torch.zeros_like(t)
        for i in range(len(stops) - 1):
            mask = (t >= xs[i]) & (t <= xs[i+1])
            if mask.any():
                t_segment = t[mask]
                ratio = (t_segment - xs[i]) / (xs[i+1] - xs[i] + 1e-8)
                ch_vals[mask] = values[i] + ratio * (values[i+1] - values[i])
        channels.append(ch_vals)
    return torch.stack(channels, dim=1).byte().cpu().numpy()


# ---------------------------------------------------------------------------
# Calcolo dell'insieme (versione CUDA accelerata, con colorazione continua "smooth")
# ---------------------------------------------------------------------------
def compute_mandelbrot(width, height, center_re, center_im, span_re, max_iter):
    """Calcola l'insieme di Mandelbrot usando tensori CUDA per l'accelerazione GPU."""
    span_im = span_re * height / width
    
    # Crea i tensori per le coordinate reali e immaginarie direttamente sul dispositivo
    re = torch.linspace(center_re - span_re / 2, center_re + span_re / 2, width, device=DEVICE)
    im = torch.linspace(center_im + span_im / 2, center_im - span_im / 2, height, device=DEVICE)
    
    # Crea la griglia complessa c = re + i*im
    c = re.unsqueeze(0) + 1j * im.unsqueeze(1)
    
    # Inizializza z a zero
    z = torch.zeros_like(c, dtype=torch.complex64)
    
    # Tensori per il calcolo smooth e lo stato attivo
    smooth = torch.full(c.shape, float(max_iter), dtype=torch.float32, device=DEVICE)
    active = torch.ones(c.shape, dtype=torch.bool, device=DEVICE)
    
    # Costanti per il calcolo
    two = torch.tensor(2.0, dtype=torch.float32, device=DEVICE)
    four = torch.tensor(4.0, dtype=torch.float32, device=DEVICE)
    one = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    
    # Iterazioni
    for i in range(1, max_iter + 1):
        if not active.any():
            break
        
        # Calcola z^2 + c solo per i punti attivi
        z_real = z.real
        z_imag = z.imag
        
        # z_new = z^2 + c = (zr^2 - zi^2 + cr) + i(2*zr*zi + ci)
        z_real_new = z_real * z_real - z_imag * z_imag + c.real
        z_imag_new = two * z_real * z_imag + c.imag
        
        # Calcola |z|^2 per verificare la divergenza
        m2 = z_real_new * z_real_new + z_imag_new * z_imag_new
        
        # Aggiorna z
        z = z_real_new + 1j * z_imag_new
        
        # Trova i punti che hanno appena superato la soglia di divergenza
        div = (m2 > four) & active
        if div.any():
            # Colorazione smooth: n + 1 - log2(log2|z|)
            # Usiamo un piccolo epsilon per evitare log(0)
            log_z = torch.log(torch.sqrt(m2) + 1e-10)
            log_log_z = torch.log(log_z + 1e-10) / torch.log(two)
            smooth[div] = i + one - log_log_z[div]
            active[div] = False
    
    return smooth.cpu().numpy()


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
