# Visualizzatore Mandelbrot — spec di ricreazione

Descrizione concisa ma sufficiente perché un altro LLM (o sviluppatore) ricrei il
programma da zero. Riferimento: `mandel.py` (un solo file), sincronizzata alla **v4.15.1**.
Ogni modifica al sorgente DEVE aggiornare anche questa spec (vedi AGENTS.md).

## Panoramica
- Un solo file **Python 3.12**: GUI **tkinter**, rendering **CUDA (CuPy `RawKernel`)**
  con fallback **CPU (numpy)**, Pillow per il PNG.
- Esecuzione asincrona: il render gira su thread separato, l'UI non si blocca mai.
- Metodi di `MandelbrotApp` raggruppati per funzione: UI, vista, controlli, pipeline, file, benchmark.

## UI e interazione
- Vista iniziale: centro `(-0.5, 0)`, `half = 1.5`, canvas 960×540; asse y scalato di `h/w`; clamp `half ≥ 1e-12`.
- **Zoom**: rotella ×1.25/×0.8 al cursore; click ×2 al cursore; `+`/`-` ×2/×0.5 al centro.
  **Pan**: trascinamento.
- `R` = reset (vista + tutti i settaggi ai default); `Ctrl+S` = salva PNG.
- Layout: toolbar (Motore / Palette / Precisione / Iter), riga pulsanti (campo iterazioni +
  `−1000`/`+1000`, Benchmark, Reset), canvas, barra di stato `messaggio | backend | palette | render: N ms`.
  Font 13 pt. Menu File: Salva immagine…, Carica zona…, Salva zona, Salva zona con nome…, Esci.
- Titolo: `Insieme di Mandelbrot v<VER> - <backend>` + ` - <file zona corrente>` se presente.

## Rendering
- `z = z² + c`, escape `|z|² > 4`; punti interni = **nero** (early-exit nel loop).
- **Interior analitico** (GPU e CPU, prima del loop): bulbo periodica-2 `|c+1| ≤ 0.25` e
  cardioide principale `|1 − √(1−4c)| < 1` (riscritta senza complessi: `R < 2·√(0.5·(R+A))`,
  `A = 1−4·Re c`, `R = |1−4c|`, con prefiltro bounding-box) → subito neri, saltano le `mi`
  iterazioni. Bit-identico al kernel senza test.
- **GPU**: iterazione in coordinata spostata `w = z − cx` (solo parte reale, stabilità
  numerica); coloring continuo `nu = it + 1 − log2(0.5·ln|z|²)`, `t = (nu/mi)^0.35`.
  **CPU**: `t = (it/mi)^0.35`.
- **LUT 256×3** condivisa CPU/GPU: `np.interp` delle stop su 256 punti, ×255, clip, uint8;
  colore = `LUT[round(t·255)]`.
- **Kernel GPU**: 2 px/thread, `__launch_bounds__(256)`, block 16×16, grid `(⌈w/32⌉, ⌈h/16⌉)`,
  `--use_fast_math`; 2 varianti (f32/f64) dalla stessa sorgente parametrizzata; NVRTC lazy.

## Palette
Registro ordinato `{nome: (t, R, G, B)}` — l'ordine = indice passato al kernel
(0=fuoco, 1=ghiaccio, 2=termal). UI, config e LUT `__constant__` del kernel sono generate
da questo registro.

- **fuoco**: `t=(0,.2,.45,.7,.9,1)` `R=(.05,.35,.85,1,1,1)` `G=(0,.02,.2,.65,.95,1)` `B=(0,0,.02,.15,.55,1)`
- **ghiaccio**: `t=(0,.25,.5,.75,1)` `R=(.02,.05,.30,.70,1)` `G=(.02,.15,.55,.85,1)` `B=(.10,.45,.90,1,1)`
- **termal** (ghiaccio→fuoco): `t=(0,.2,.4,.55,.7,.85,1)` `R=(.02,.10,.55,.95,1,1,1)` `G=(.08,.45,.80,.96,.85,.55,.30)` `B=(.28,.85,.95,.98,.45,.20,.10)`

## Iterazioni (MI)
- **Auto** (default): `auto_mi(half) = 2000·(1 + log10(1.5/half))`, clamp `[50, 50000]` —
  funzione di modulo unica, **condivisa col benchmark**.
- **Manual**: campo editabile (commit su Invio/FocusOut; validazione intero **50–100000**,
  valore non valido → messaggio nella barra di stato + ripristino del valore precedente)
  e pulsanti `±1000` (disabilitati in auto). In auto il campo è disabilitato ma mostra il
  valore corrente (etichetta `Iterazioni (auto):`); disattivando l'auto, `mi` si **congela
  sul valore auto corrente**.

## Backend e precisione
- CPU: numpy, sempre **f64**. GPU: **f32** (default) o **f64** (~32× più lenta su GPU consumer).
- Toggle runtime CPU/CUDA; senza CUDA i controlli CUDA/f64 sono disabilitati. Il kernel f64
  è compilato alla partenza: se fallisce, resta disabilitato (si resta in f32).

## Pipeline (asincrona, latest-wins)
- Thread worker (daemon) + `threading.Condition` + **slot job singolo**: i request in coda
  collassano sull'ultimo (*latest-wins*); nessun render in corso viene cancellato.
- `request_render`: **preview ¼** (min 16 px) immediata, poi render full dopo **500 ms**
  (solo se la vista non è cambiata).
- UI: poll ogni **30 ms** (`tkinter.after`) che mostra l'ultimo frame (ridimensionato al canvas).

## File
- **Zona** (JSON della vista, indentato): `{"app", "versione", "cx", "cy", "half", "mi", "mi_auto"}`.
  "Salva zona" riscrive il **file corrente** (`view_file`), altrimenti chiede il nome
  (default `mandelbrot_<AAAAmmgg_HHMMSS>.json`); "Carica zona…" ripristina vista + MI
  (clamp `half ≥ 1e-12`) e rende il file quello corrente (mostrato nel titolo).
- **Config**: `~\mandelbrot\config.json` con
  `cx, cy, half, mi, mi_auto, precision, palette, backend, bench, view_file`;
  salvata all'uscita e **throttled ~1 s** sui cambiamenti; reset riporta i default.

## Benchmark
- Dialog di conferma custom (`Toplevel` modale centrato, griglia parametri, Avvia/Annulla,
  `Return`/`Esc`); poi thread dedicato per la durata (default 8 s).
- Default: `c = (-0.7499302568795561, -0.015139113925433963i)`, `half = 5.226737155905588e-05`,
  960×540, `secs = 8.0`.
- **`mi` non è un parametro**: sempre `auto_mi(bench['half'])` (~10 915 a default) →
  benchmark comparabile anche se la formula auto cambia.
- Parametri in `config.json` (chiave `bench`, overridibile; una vecchia `bench.mi` è ignorata).
- GPU: **sempre f32**, buffer proprio (no contesa col render normale); senza GPU: CPU f64.
- Report: dialog con **rendering/s in grande** (il vero risultato, ~42pt verde), statistiche
  (n. rendering, ms/render) e griglia dei parametri; errore → "BENCHMARK FALLITO" + dettaglio.

## Note tecniche (imparati a caro prezzo)
- Le LUT **devono** essere `__constant__` **incorporate nel kernel** (una per palette):
  passate come buffer device, CuPy le sovrascriveva con l'output a ogni launch.
- Argomenti scalari CuPy = **array numpy size-1**; indice palette preallocato (`_PAL_IDX`).
- `np.asarray(array_cupy)` **non** è permesso → `.get()`.
- NVRTC: niente `-O3`/`--opt-level` (la compilazione è lazy alla prima chiamata).
- Tkinter: `delete`/`insert` su `Entry` disabilitato sono **no-op** → prima `state="normal"`.
- Note operative (PowerShell su UNC, tool, workflow di versionamento): vedi **AGENTS.md**.
