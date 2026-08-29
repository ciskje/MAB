# Visualizzatore Mandelbrot interattivo — spec di ricreazione

> Obiettivo: descrizione concisa ma sufficiente perché un altro LLM (o sviluppatore)
> possa ricreare il programma da zero. Riferimento: `mandel.py` (un solo file).

## Panoramica
Viewer interattivo dell'insieme di Mandelbrot in **un solo file Python**.
- GUI: **tkinter**
- Rendering: **GPU (CUDA via CuPy `RawKernel`)** + **fallback CPU (numpy)**
- Esecuzione: **asincrona** (thread worker, UI sempre responsive)

## Stack
Python 3.12, `tkinter`, `numpy`, CuPy (`RawKernel` + NVRTC), Pillow (`PIL`).

## Vista e interazione
- Vista iniziale: centro `(-0.5, 0)`, half-width `1.5`, canvas `960×540`.
- Fattore d'aspetto corretto: asse y scalato di `h/w`.
- **Zoom**: rotella (`×1.25` / `×0.8`) al cursore; **click** (release senza drag) = `×2` al cursore; tasto `+` / `−` = `×2` / `×0.5` al centro.
- **Pan**: trascinamento tasto sinistro.
- **Reset**: pulsante o tasto `R` → vista + tutti i settaggi ai default.
- **Salva PNG**: `Ctrl+S` / menu File.
- Barra di stato: `messaggio | backend | palette | render: <ms>`.
- Titolo finestra: `Insieme di Mandelbrot v<VERSIONE> - <backend>`.

## Rendering (algoritmo)
- Mandelbrot standard: `z = z² + c`, `z₀ = 0`, escape quando `|z|² > 4`.
- **GPU**: usa coordinata **spostata** `w = z − (cx, 0)` (solo parte reale) per stabilità numerica ("ricentramento algebrico").
- **Coloring (GPU)**: `nu = it + 1 − log2(0.5·ln(|z|²))`; `t = (nu/mi)^0.35`; colore = `LUT[round(t·255)]`.
- **Coloring (CPU)**: forma più semplice `t = (it/mi)^0.35`.
- Punti interni (non divergenti) = **nero**.
- **LUT 256×3** condivisa CPU/GPU: interpolazione lineare (`np.interp`) delle stop su 256 punti, `×255`, `uint8`.

## Palette
Registro ordinato `{nome: stop}` — l'ordine = indice passato al kernel.
Le LUT `__constant__` del kernel, i pulsanti UI e la config sono generati da questo registro.

Stop: tuple `(t, R, G, B)`.

- **fuoco**
  - `t = (0, .2, .45, .7, .9, 1)`
  - `R = (.05, .35, .85, 1, 1, 1)`
  - `G = (0, .02, .2, .65, .95, 1)`
  - `B = (0, 0, .02, .15, .55, 1)`
- **ghiaccio**
  - `t = (0, .25, .5, .75, 1)`
  - `R = (.02, .05, .30, .70, 1)`
  - `G = (.02, .15, .55, .85, 1)`
  - `B = (.10, .45, .90, 1, 1)`
- **termal** (ghiaccio → fuoco)
  - `t = (0, .2, .4, .55, .7, .85, 1)`
  - `R = (.02, .10, .55, .95, 1, 1, 1)`
  - `G = (.08, .45, .80, .96, .85, .55, .30)`
  - `B = (.28, .85, .95, .98, .45, .20, .10)`

## Iterazioni (MI)
- **Auto** (default ON, disattivabile): `mi = 400·(1 + log10(1.5/half))`, clamp `[50, 10000]` — cresce con lo zoom.
- **Manual**: pulsanti `±1000` (disabilitati in auto).

## Backend e precisione
- **CPU**: numpy, sempre **f64** (`complex128`).
- **GPU**: CUDA, precisione **f32** (default) o **f64** (più lenta su GPU consumer, ~1:32). Kernel generato in **2 varianti di precisione** dalla stessa sorgente parametrizzata.
- Toggle runtime **CPU / CUDA** (pulsanti).

## Architettura (pipeline asincrona latest-wins)
- **Thread worker** (daemon) + `threading.Condition` + **slot job singolo**: i request in coda collassano sull'ultimo (*latest-wins*).
- `request_render`: invia subito una **preview 1/4**, poi il **render full dopo 500 ms** (solo se la vista non è cambiata).
- Thread principale: ogni **30 ms** (`tkinter.after`) spolia una coda e mostra l'ultimo frame (ridimensionato al canvas).
- **UI sempre responsive** (render su thread separato); **nessuna cancellazione** del render in corso.

## Vincoli GPU (imparati a caro prezzo)
- Le palette **devono** essere array `__constant__` **incorporati nel kernel** (uno per palette), **non** passati come buffer device: CuPy sovrascriveva la LUT con l'output a ogni launch → colori corrotti.
- Kernel: **2 px/thread** (orizzontali), early-exit, block `16×16`, grid `(ceil(w/32), ceil(h/16))`.
- Argomenti scalari CuPy = **array numpy size-1** (non scalari Python).
- NVRTC compila **lazy** alla prima chiamata; non usare `-O3` / `--opt-level`.
- `np.asarray(array_cupy)` **non** è permesso → usare `.get()`.
- PowerShell 5.1: non usare `Start-Process -RedirectStandardOutput` su path UNC (timeout).

## Config (persistente)
- Salva/carica `~\mandelbrot\config.json` (JSON): `cx, cy, half, mi, mi_auto, precision, palette, backend`.
- Salvataggio all'uscita + **throttled ~1 s** sui cambiamenti. Reset riporta i default e salva.

## Benchmark
- Pulsante "Benchmark": dialog di conferma (regione / iterazioni / risoluzione / motore / durata), poi **thread dedicato 8 s**:
  - regione fissa `c = (-0.74364388703, 0.13182590421i)`, `half = 0.002`, `mi = 3000`, `960×540`
  - **sempre f32**, buffer proprio (no contesa col render normale)
- Report: n. ripetizioni, rendering/s, ms/render.

## Convenzione di versionamento
- Header commentato con `VERSIONE` + `STORICO`.
- Ogni modifica: bump versione + voce `versione - AAAA-MM-GG - descrizione`.
  - patch = bugfix / modifica minima
  - minor = modifica funzionale
  - major = riscrittura / architettura
