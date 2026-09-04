# MAB - Mandelbrot Ai benchmark per LLM agentica — v7.1.3

Questo repo è un **benchmark per testare le capacità agentiche e di
programmazione di una LLM** su un progetto software reale e vivo: un
visualizzatore interattivo dell'insieme di Mandelbrot (Python + Tkinter)
con 4 motori di rendering selezionabili a runtime (CPU, CUDA, Vulkan, Metal) e fallback automatico
sulla CPU. La LLM lavora qui come sviluppatore: implementa funzioni,
corregge bug, fa refactor e rilascia versioni — il tutto verificato
bit-identico prima di ogni commit (vedi sotto).

![Demo (palette termal, antialias 2x2)](demo.png)

## Download

| Piattaforma | File |
|---|---|
| Windows 64-bit (v7.1.3) | [Mandelbrot-v7.1.3-win64.zip](https://github.com/ciskje/MandelbrotTest/releases/download/v7.1.3/Mandelbrot-v7.1.3-win64.zip) |
| macOS (v5.10.2) | [Mandelbrot-v5.10.2-macos.dmg](https://github.com/ciskje/MandelbrotTest/releases/download/v5.10.2/Mandelbrot-v5.10.2-macos.dmg) |

Vulkan incluso ovunque; CUDA richiede driver NVIDIA + runtime utente.
Altre versioni nella pagina [Releases](https://github.com/ciskje/MandelbrotTest/releases).

## Motivazioni

Il progetto nasce per confrontare dal vero CPU e GPU sullo stesso calcolo
(escape-time con interior analitico e coloring smooth identici su tutti i
backend, a meno di 1–2 ULP): quanto rende Numba contro CuPy, Metal e Vulkan
sulla stessa immagine, e dove la precisione f32 inizia a divergere nei
deep-zoom. In più è un banco di prova per lo sviluppo assistito da AI
locale (vedi sotto): specifica e codice evolvono insieme, ogni versione è
verificata bit-identica prima del commit.

![Python](https://img.shields.io/badge/python-3.14-blue)
[![Licenza: MIT](https://img.shields.io/badge/Licenza-MIT-yellow.svg)](LICENSE)

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

## Benchmark per LLM

Il progetto è sviluppato anche e soprattutto con una AI locale (Qwen3.8
27B). `spec.md` (specifica tecnica sempre in sync col sorgente) e
`AGENTS.md` (convenzioni di lavoro: versionamento, build, note operative)
sono tenuti aggiornati proprio per questo: dati in pasto a una LLM,
permettono di testarne le capacità agentiche e di programmazione su un
progetto reale (refactor, bugfix verificati, bump di versione disciplinati).

## Struttura

| file | ruolo |
|---|---|
| `mandelbrot/` | il programma (`config`, `palette`, `state`, `mem`, `cpu`, `cuda`, `metal`, `vulkan`, `engine`, `app`, `expert`) |
| `mandel.py` | entry-point + compatibilità `import mandel` |
| `spec.md` | specifica tecnica (sempre in sync col sorgente) |
| `build_app.py` / `mandelbrot.spec` / `hook_dlldir.py` | build PyInstaller |
| `make_icon.py` | genera l'icona dell'app dal motore CPU |

Versione corrente: vedi titolo in cima (fonte di verità: `VERSION` in
`mandelbrot/__init__.py`).
