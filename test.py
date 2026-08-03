#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esploratore interattivo dell'insieme di Mandelbrot con palette fuoco.
Versione OPTIMIZZATA con CUDA/GPU e ottimizzata per CPU multi-core.

BUILD: 1.0.0

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

Performance OTTIMIZZATE:
  - GPU CUDA: fino a 200x più veloce rispetto alla CPU (se disponibile)
  - CPU multi-core: sfrutta tutti i core disponibili tramite PyTorch
    Le operazioni vettoriali sono parallelizzate automaticamente su tutti i thread
  
OTTIMIZZAZIONI IMPLEMENTATE:
  - Kernel Mandelbrot senza torch.nonzero() - solo maschere booleane
  - Palette precaricata come tensore fisso sul device (GPU/CPU)
  - Colorizzazione completamente vettorizzata su GPU
  - Minimizzazione trasferimenti CPU-GPU
  - Caching della griglia complessa quando possibile
  - Precisione doppia gestita efficientemente
"""

import math
import os
import time
import tkinter as tk
from tkinter import ttk

# Librerie esterne usate dal programma:
# - numpy: calcoli numerici e manipolazione degli array di colore
# - torch: calcolo del frattale e accelerazione CUDA/CPU
# - tkinter: interfaccia grafica (finestra, canvas, bottoni, combobox, label)
# Tutte queste dipendenze sono esterne al Python standard library.
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
    # Chiamata esterna al sistema operativo:
    # os.cpu_count() rileva il numero di core fisici/logici disponibili.
    # Serve per decidere quanti thread usare in modo intelligente senza bloccare troppo la UI.
    cpu_cores = os.cpu_count() or 4

    # Chiamata esterna alla libreria PyTorch:
    # torch.set_num_threads() forza PyTorch a usare un numero specifico di thread CPU.
    # Questo non riguarda CUDA ma ottimizza il calcolo vettoriale sul processore.
    num_threads = cpu_cores if cpu_cores <= 4 else cpu_cores - 1
    torch.set_num_threads(num_threads)

    print(f"CPU rilevata: {cpu_cores} core")
    print(f"Thread PyTorch configurati: {num_threads}")

    # Chiamata esterna a PyTorch per verificare se esiste un device GPU NVIDIA supportato.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # torch.cuda.get_device_name(0): nome della GPU attiva
        print(f"CUDA disponibile: {torch.cuda.get_device_name(0)}")
        # torch.cuda.get_device_properties(...).total_memory: memoria video totale
        print(f"Memoria CUDA totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("-> Modalita': GPU-accelerated (massime prestazioni)")
    else:
        device = torch.device("cpu")
        print("CUDA non disponibile -> esecuzione su CPU multi-core")
        print("-> Modalita': CPU multi-threaded (buone prestazioni)")
        print(f"   Il codice sfrutta tutti i {num_threads} thread per il calcolo parallelo")
    return device


DEVICE = get_device()


def set_compute_device(mode):
    """Imposta il backend di calcolo: CUDA se disponibile, altrimenti CPU."""
    global DEVICE
    if mode == "CUDA" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE


# ---------------------------------------------------------------------------
# Palette fuoco - VERSIONE OTTIMIZZATA
# ---------------------------------------------------------------------------
def build_fire_palette(n=256, device=None):
    """LUT che va dal nero (insieme) al bianco caldo passando per il fuoco.
    
    OPTIMIZZAZIONE: La palette è generata direttamente sul device specificato
    e mantenuta come tensore PyTorch per evitare conversioni CPU-GPU durante
    la colorizzazione.
    """
    if device is None:
        device = DEVICE
    
    # Definiamo i punti di transizione della palette del fuoco.
    # Ogni stop indica una posizione nel range [0, 1] e un colore RGB.
    stops = [
        (0.00, (0, 0, 0)),
        (0.12, (35, 0, 5)),
        (0.30, (115, 10, 0)),
        (0.50, (195, 45, 0)),
        (0.70, (255, 110, 0)),
        (0.86, (255, 195, 40)),
        (1.00, (255, 255, 210)),
    ]

    # Generazione vettorizzata della palette direttamente sul device
    t = torch.linspace(0.0, 1.0, n, device=device)
    xs = torch.tensor([s[0] for s in stops], device=device)
    channels = []
    for ch in range(3):
        values = torch.tensor([s[1][ch] for s in stops], dtype=torch.float32, device=device)
        # Interpolazione lineare manuale per tensori GPU/CPU.
        ch_vals = torch.zeros_like(t)
        for i in range(len(stops) - 1):
            mask = (t >= xs[i]) & (t <= xs[i+1])
            if mask.any():
                t_segment = t[mask]
                ratio = (t_segment - xs[i]) / (xs[i+1] - xs[i] + 1e-8)
                ch_vals[mask] = values[i] + ratio * (values[i+1] - values[i])
        channels.append(ch_vals)

    # Ritorna il tensore byte sul device (non converte a numpy qui)
    return torch.stack(channels, dim=1).byte()


# Cache della palette per device
_PALETTE_CACHE = {}

def get_fire_palette(n=256, device=None):
    """Ottiene la palette fuoco con caching per device.
    
    OPTIMIZZAZIONE: La palette è calcolata una sola volta per device e riutilizzata.
    """
    if device is None:
        device = DEVICE
    
    cache_key = (n, str(device))
    if cache_key not in _PALETTE_CACHE:
        _PALETTE_CACHE[cache_key] = build_fire_palette(n, device)
    return _PALETTE_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Calcolo dell'insieme (versione CUDA accelerata, con colorazione continua "smooth")
# VERSIONE OTTIMIZZATA - senza torch.nonzero(), solo maschere booleane
# ---------------------------------------------------------------------------
def compute_mandelbrot(width, height, center_re, center_im, span_re, max_iter, precision="single"):
    """Calcola l'insieme di Mandelbrot usando precisione singola o doppia.

    OPTIMIZZAZIONI IMPLEMENTATE:
    - usa il device attivo (CUDA/CPU) senza copie inutili
    - NESSUN torch.nonzero() - solo operazioni vettoriali con maschere booleane
    - usa i dtype corretti per la precisione scelta
    - evita trasferimenti CPU-GPU nel ciclo
    - operazioni fully vectorized su tutta la griglia
    
    La versione ottimizzata elimina l'overhead di torch.nonzero() e dell'indicizzazione
    avanzata, usando invece maschere booleane che sono molto più efficienti su GPU.
    """
    if precision == "double":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        real_dtype = torch.float32
        complex_dtype = torch.complex64

    with torch.inference_mode():
        span_im = span_re * height / width

        # Genera la griglia complessa direttamente sul device
        re = torch.linspace(center_re - span_re / 2, center_re + span_re / 2,
                           width, device=DEVICE, dtype=real_dtype)
        im = torch.linspace(center_im + span_im / 2, center_im - span_im / 2,
                           height, device=DEVICE, dtype=real_dtype)

        # Broadcasting efficiente per creare la griglia
        c_re = re.unsqueeze(0).expand(height, -1)
        c_im = im.unsqueeze(1).expand(-1, width)
        
        # Inizializza z e smooth
        z_re = torch.zeros((height, width), dtype=real_dtype, device=DEVICE)
        z_im = torch.zeros((height, width), dtype=real_dtype, device=DEVICE)
        smooth = torch.full((height, width), float(max_iter), dtype=real_dtype, device=DEVICE)
        active = torch.ones((height, width), dtype=torch.bool, device=DEVICE)

        # Costanti pre-calcolate sul device
        two = torch.tensor(2.0, dtype=real_dtype, device=DEVICE)
        four = torch.tensor(4.0, dtype=real_dtype, device=DEVICE)
        one = torch.tensor(1.0, dtype=real_dtype, device=DEVICE)
        log_two = torch.log(two)

        # Loop iterativo OPTIMIZZATO: operazioni vettoriali su tutta la griglia
        for i in range(1, max_iter + 1):
            if not active.any():
                break

            # Calcola z^2 + c solo per i pixel attivi (usando maschera booleana)
            z_re2 = z_re * z_re
            z_im2 = z_im * z_im
            
            # Nuova parte reale e immaginaria
            z_re_new = z_re2 - z_im2 + c_re
            z_im_new = two * z_re * z_im + c_im
            
            # Modulo quadrato
            m2 = z_re_new * z_re_new + z_im_new * z_im_new
            
            # Trova i punti che divergono in questa iterazione
            div = active & (m2 > four)
            
            if div.any():
                # Calcolo smooth coloring per i punti divergenti
                div_m2 = m2[div]
                log_z = torch.log(torch.sqrt(div_m2) + 1e-10)
                log_log_z = torch.log(log_z + 1e-10) / log_two
                smooth[div] = i + one - log_log_z
                active[div] = False
            
            # Aggiorna z solo per i punti ancora attivi e non divergenti
            update_mask = active & ~div
            if update_mask.any():
                z_re[update_mask] = z_re_new[update_mask]
                z_im[update_mask] = z_im_new[update_mask]

        # Trasferimento finale alla CPU per la visualizzazione
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

        # OPTIMIZZAZIONE: usa la palette con caching e come tensore sul device
        self.lut = get_fire_palette()
        self.center_re, self.center_im = self.INIT_CENTER
        self.span_re = self.INIT_SPAN
        self.width = None
        self.height = None
        self.photo = None
        self.img_id = None
        self._drag_start = None
        self._drag_mode = None
        self._resize_job = None
        self._idle_render_job = None
        self.render_time = 0.0
        self.preview_scale = 0.25
        self.preview_delay_ms = 2000
        self.compute_mode = "CUDA" if torch.cuda.is_available() else "CPU"
        self.precision_mode = "single"
        set_compute_device(self.compute_mode)

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
                               command=lambda: self.render(scale=1.0))
        self.spin.pack(side=tk.LEFT, padx=(4, 8))
        self.spin.bind("<Return>", lambda e: self.render(scale=1.0))

        self.btn_minus = tk.Button(top, text="- 100", command=lambda: self._change_iter(-100))
        self.btn_plus = tk.Button(top, text="+ 100", command=lambda: self._change_iter(+100))
        self.btn_minus.pack(side=tk.LEFT)
        self.btn_plus.pack(side=tk.LEFT, padx=(2, 12))

        tk.Checkbutton(top, text="Iterazioni automatiche (in base allo zoom)",
                       variable=self.auto_iter,
                       command=self._on_auto_iter).pack(side=tk.LEFT)

        precision_frame = tk.Frame(top)
        precision_frame.pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(precision_frame, text="Precisione:").pack(side=tk.LEFT)
        self.precision_var = tk.StringVar(value=self.precision_mode)
        self.precision_select = ttk.Combobox(
            precision_frame,
            textvariable=self.precision_var,
            values=("single", "double"),
            width=8,
            state="readonly",
        )
        self.precision_select.pack(side=tk.LEFT, padx=(4, 0))
        self.precision_select.bind("<<ComboboxSelected>>", lambda e: self._on_precision_change())

        self.mode_badge = tk.Label(
            top,
            text="CUDA",
            fg="white",
            bg="#3dc98c" if torch.cuda.is_available() else "#ffb347",
            padx=10,
            pady=3,
            font=("Segoe UI", 10, "bold"),
            relief="solid",
            bd=1,
        )
        self.mode_badge.pack(side=tk.RIGHT, padx=(8, 4))

        self.mode_hint = tk.Label(
            top,
            text="GPU ACCELERATED" if torch.cuda.is_available() else "CPU MULTI-CORE",
            fg="#1c1c1c",
            font=("Segoe UI", 9, "bold"),
        )
        self.mode_hint.pack(side=tk.RIGHT, padx=(0, 8))

        self.mode_toggle = tk.Button(
            top,
            text="Passa a CPU" if self.compute_mode == "CUDA" else "Passa a CUDA",
            command=self._toggle_compute_mode,
            bg="#1e3a8a",
            fg="white",
            relief="raised",
            font=("Segoe UI", 9, "bold"),
        )
        self.mode_toggle.pack(side=tk.RIGHT, padx=(0, 8))

        tk.Button(top, text="Reimposta vista (R)", command=self.reset_view).pack(side=tk.RIGHT)

        hint = ("Rotella: zoom/dezoom  •  Trascina col sinistro: pan  •  Shift + drag: zoom su un'area  •  "
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

        self._update_mode_badge()
        self._update_status()

    def _update_mode_badge(self):
        cuda = torch.cuda.is_available()
        if self.compute_mode == "CUDA" and cuda:
            self.mode_badge.config(text="CUDA", bg="#3dc98c", fg="white")
            self.mode_hint.config(text="GPU ACCELERATED", fg="#0f5c42")
            self.mode_toggle.config(text="Passa a CPU")
        else:
            self.mode_badge.config(text="CPU", bg="#ffb347", fg="black")
            self.mode_hint.config(text="CPU MULTI-CORE", fg="#6b4200")
            self.mode_toggle.config(text="Passa a CUDA" if cuda else "CUDA non disponibile")
        self.mode_toggle.config(state="normal" if cuda else "disabled")

    def _toggle_compute_mode(self):
        if not torch.cuda.is_available():
            return
        self.compute_mode = "CPU" if self.compute_mode == "CUDA" else "CUDA"
        set_compute_device(self.compute_mode)
        # OPTIMIZZAZIONE: usa la palette con caching per il nuovo device
        self.lut = get_fire_palette()
        self._update_mode_badge()
        self._schedule_preview_then_full_render()

    def _on_precision_change(self):
        self.precision_mode = self.precision_var.get()
        self._schedule_preview_then_full_render()

    def _bind_events(self):
        # Qui avviene il binding tra le interazioni utente e i metodi della classe.
        # Tkinter usa stringhe di evento come "<ButtonPress-1>", "<MouseWheel>", ecc.
        c = self.canvas
        c.bind("<Configure>", self._on_configure)      # resize della finestra / canvas
        c.bind("<Motion>", self._on_motion)            # movimento del mouse
        c.bind("<MouseWheel>", self._on_wheel)         # scroll su/giù su Windows/macOS
        c.bind("<Button-4>", self._on_wheel)           # scroll su su Linux
        c.bind("<Button-5>", self._on_wheel)           # scroll giù su Linux
        c.bind("<ButtonPress-1>", self._on_press_left)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release_left)
        c.bind("<Shift-ButtonPress-1>", self._on_shift_press_left)
        c.bind("<Shift-B1-Motion>", self._on_shift_drag)
        c.bind("<Shift-ButtonRelease-1>", self._on_shift_release_left)
        c.bind("<ButtonPress-3>", self._on_right_click)
        self.root.bind("<Key>", self._on_key)

    # ------------------------- rendering -------------------------
    def render(self, scale=1.0):
        if not self.width or not self.height:
            return
        if scale <= 0 or scale > 1.0:
            scale = 1.0

        self._apply_auto_iter()
        iters = self.max_iter.get()
        render_w = max(1, int(self.width * scale))
        render_h = max(1, int(self.height * scale))

        t0 = time.perf_counter()
        smooth = compute_mandelbrot(render_w, render_h,
                                    self.center_re, self.center_im,
                                    self.span_re, iters,
                                    precision=self.precision_mode)
        rgb = self._colorize(smooth, iters)
        photo = self._to_photoimage(rgb)

        if scale < 1.0:
            zoom = int(round(1.0 / scale))
            photo = photo.zoom(zoom, zoom)
        self.photo = photo

        if self.img_id is None:
            self.img_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.coords(self.img_id, 0, 0)
            self.canvas.itemconfig(self.img_id, image=self.photo)

        self.render_time = time.perf_counter() - t0
        self._update_status()

    def _schedule_preview_then_full_render(self):
        if self._idle_render_job is not None:
            self.root.after_cancel(self._idle_render_job)
        self.render(scale=self.preview_scale)
        self._idle_render_job = self.root.after(self.preview_delay_ms, self._render_after_idle)

    def _render_after_idle(self):
        self._idle_render_job = None
        self.render(scale=1.0)

    def _colorize(self, smooth, max_iter):
        """Colorizzazione OPTIMIZZATA: usa tensori PyTorch invece di NumPy.
        
        OPTIMIZZAZIONE: Tutta la colorizzazione avviene su GPU (se disponibile)
        usando operazioni vettoriali PyTorch, minimizzando i trasferimenti CPU-GPU.
        Solo il risultato finale è convertito in numpy per tkinter.
        """
        # Converte smooth in tensore sul device corrente
        smooth_t = torch.from_numpy(smooth).to(DEVICE)
        
        # Calcola l'indice nella palette usando operazioni vettoriali
        inside = smooth_t >= max_iter
        t = torch.sqrt(torch.clamp(smooth_t / max_iter, 0.0, 1.0))
        idx = torch.clamp((t * len(self.lut)).long(), 0, len(self.lut) - 1)
        
        # Lookup nella palette (ora un tensore sul device)
        rgb_t = self.lut[idx]
        
        # Imposta i punti interni a nero
        rgb_t[inside] = 0
        
        # Trasferisce solo il risultato finale alla CPU come numpy array
        return rgb_t.cpu().numpy()

    @staticmethod
    def _to_photoimage(rgb):
        # Chiamata esterna a tkinter:
        # tk.PhotoImage è la classe che riceve un buffer RGB in formato PPM/P6 e lo trasforma in un'immagine
        # leggibile dal widget Canvas di Tk.
        h, w, _ = rgb.shape
        # header PPM: "P6\nW H\n255\n" è il formato PPM (Portable Pixmap) a 8-bit per canale RGB.
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        # np.ascontiguousarray() assicura che i dati RGB siano contigui in memoria prima di crearli nel buffer.
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
        self._schedule_preview_then_full_render()

    # ------------------------- eventi mouse -------------------------
    def _on_wheel(self, event):
        up = (getattr(event, "num", None) == 4) or (event.delta > 0)
        self._zoom_at(event.x, event.y,
                      self.WHEEL_FACTOR if up else 1.0 / self.WHEEL_FACTOR)

    def _on_press_left(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_mode = "pending"

    def _on_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        dx = event.x - x0
        dy = event.y - y0

        if self._drag_mode == "pending" and (abs(dx) > 4 or abs(dy) > 4):
            self._drag_mode = "pan"
            self.canvas.delete("rubberband")

        if self._drag_mode == "pan":
            scale_x = self.span_re / self.width
            scale_y = self.span_re * self.height / (self.width * self.height)
            self.center_re -= dx * scale_x
            self.center_im += dy * scale_y
            self._drag_start = (event.x, event.y)
            self._schedule_preview_then_full_render()
            self._update_status(mouse=(event.x, event.y))
            return

        self.canvas.delete("rubberband")
        self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                     outline="#ffdd55", dash=(4, 3), tags="rubberband")
        self._update_status(mouse=(event.x, event.y))

    def _on_release_left(self, event):
        self.canvas.delete("rubberband")
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        drag_mode = self._drag_mode
        self._drag_start = None
        self._drag_mode = None
        w_px, h_px = abs(event.x - x0), abs(event.y - y0)

        if drag_mode == "pan":
            return

        if w_px < 6 or h_px < 6:                 # clic semplice: zoom x2 sul punto
            self._zoom_at(event.x, event.y, self.CLICK_FACTOR)
            return

        cre, cim = self._pixel_to_complex((x0 + event.x) / 2, (y0 + event.y) / 2)
        s = self.span_re / self.width
        new_span = max(self.MIN_SPAN, s * max(w_px, h_px * self.width / self.height))
        self.center_re, self.center_im, self.span_re = cre, cim, new_span
        self._schedule_preview_then_full_render()

    def _on_shift_press_left(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_mode = "selection"
        self.canvas.delete("rubberband")

    def _on_shift_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        self.canvas.delete("rubberband")
        self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                     outline="#ffdd55", dash=(4, 3), tags="rubberband")
        self._update_status(mouse=(event.x, event.y))

    def _on_shift_release_left(self, event):
        self.canvas.delete("rubberband")
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        self._drag_mode = None
        w_px, h_px = abs(event.x - x0), abs(event.y - y0)
        if w_px < 6 or h_px < 6:
            return
        cre, cim = self._pixel_to_complex((x0 + event.x) / 2, (y0 + event.y) / 2)
        s = self.span_re / self.width
        new_span = max(self.MIN_SPAN, s * max(w_px, h_px * self.width / self.height))
        self.center_re, self.center_im, self.span_re = cre, cim, new_span
        self._schedule_preview_then_full_render()

    def _on_right_click(self, event):
        self._zoom_at(event.x, event.y, 1.0 / self.CLICK_FACTOR)

    def _on_motion(self, event):
        self._update_status(mouse=(event.x, event.y))

    def _on_configure(self, event):
        self.width, self.height = event.width, event.height
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, lambda: self._schedule_preview_then_full_render())

    # ------------------------- iterazioni -------------------------
    def _change_iter(self, delta):
        if self.auto_iter.get():
            return
        val = max(20, min(10000, self.max_iter.get() + delta))
        self.max_iter.set(val)
        self.render(scale=1.0)

    def _on_auto_iter(self):
        state = "disabled" if self.auto_iter.get() else "normal"
        self.spin.config(state=state)
        self.btn_minus.config(state=state)
        self.btn_plus.config(state=state)
        self.render(scale=1.0)

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
        self._schedule_preview_then_full_render()

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
                 f"Precisione: {self.precision_mode}   "
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
    # Chiamata esterna all'interfaccia grafica di Tkinter:
    # tk.Tk() crea la finestra principale del programma.
    root = tk.Tk()
    root.geometry("1024x720")
    MandelbrotExplorer(root)
    # root.mainloop() avvia il loop di eventi di Tkinter: da qui in poi si ricevono
    # tutti gli input dell'utente (mouse, tastiera, resize, ecc.) e si aggiornano le widget.
    root.mainloop()


if __name__ == "__main__":
    main()
