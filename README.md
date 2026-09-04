# Insieme di Mandelbrot — visualizzatore interattivo

Visualizzatore interattivo dell'insieme di Mandelbrot (Python + Tkinter)
con 4 backend di rendering selezionabili a runtime e fallback automatico
sulla CPU.

![Python](https://img.shields.io/badge/python-3.14-blue)
![Licenza](https://img.shields.io/badge/licenza-vedi_riga_sotto-lightgrey)

## Funzioni

- **Navigazione fluida**: zoom con rotella al cursore (×1.25/×0.8), click
  ×2, click destro ×0.5, trascinamento per il pan, tasti `+`/`-`/`r`.
- **4 motori**: CPU (numpy/Numba multi-core), CUDA (CuPy), Metal (Apple
  Silicon), Vulkan (wgpu, funziona out-of-the-box su AMD/NVIDIA/Intel).
  I backend non disponibili restano visibili ma disabilitati.
- **Multi-GPU**: dropdown GPU per device CUDA e adapter Vulkan; su 2 CUDA
  c'è lo split "Entrambe" in bande orizzontali con rapporto calibrabile.
- **Precisione** f32/f64 (f64 dove il backend la supporta), **palette**
  fuoco/ghiaccio/termal, **iterazioni** auto (logaritmiche sullo zoom) o
  manuali.
- **Ricalcolo antialiasing** 1x1/2x2/4x4/8x8, salvataggio PNG (`Ctrl+S`) e
  zone JSON (vista + iterazioni, ricaricabili).
- **Benchmark standardizzato** (8 s) ed **Esperto** (3×8 s, vale la
  migliore) con grafico dei riferimenti storici e **codice di sicurezza a
  64 bit** anti-ritocco, verificabile in `Help > Verifica benchmark...`.

## Requisiti

- Python 3.14, `numpy`, `Pillow` (sempre); `numba` consigliato (altrimenti
  fallback numpy single-core).
- GPU: CUDA richiede driver NVIDIA + runtime (rilevati via
  cuda-pathfinder); Vulkan è incluso nel pacchetto; Metal solo su macOS.

## Avvio

```bash
python mandel.py
```

## Build dell'app

Solo su richiesta esplicita (mai automatica):

```bash
python build_app.py
```

Produce l'app self-contained (Windows: EXE + zip in `dist/`; macOS: .app
firmata ad-hoc + .dmg). Dettagli in `spec.md` §11.

## Struttura

| file | ruolo |
|---|---|
| `mandelbrot/` | il programma (`config`, `palette`, `state`, `mem`, `cpu`, `cuda`, `metal`, `vulkan`, `engine`, `app`, `expert`) |
| `mandel.py` | entry-point + compatibilità `import mandel` |
| `spec.md` | specifica tecnica (sempre in sync col sorgente) |
| `build_app.py` / `mandelbrot.spec` / `hook_dlldir.py` | build PyInstaller |
| `make_icon.py` | genera l'icona dell'app dal motore CPU |

Versione corrente: vedi `VERSION` in `mandelbrot/__init__.py`
(storico completo con `git log --oneline`).
