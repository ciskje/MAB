# Visualizzatore Mandelbrot — spec di ricreazione

Descrizione concisa ma sufficiente perché un altro LLM (o sviluppatore) ricrei il
programma da zero. Riferimento: `mandel.py` (un solo file), sincronizzata alla **v5.2.0**.
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
  **CPU** (v5.2.0): **stesso** coloring continuo della GPU `nu = it + 1 − log2(0.5·ln|z|²)`,
  `t = (nu/mi)^0.35` (il kernel Numba/fallback esportano anche `mag = |z|²` alla fuga);
  solo i pixel mai fuggiti (`mag==0`) restano neri. CPU e CUDA producono lo stesso colore
  (a parte 1-2 ULP di `log2`/`log` libm-vs-CUDA → ≤1 entry di LUT; verificato dal gate).
  (v≤5.1.x: `t=(it/mi)^0.35` + `rgb[it==0]=0`, che anneriva anche i punti "lontani" fuggiti
  alla 1ª iterazione — bug CPU-nero/CUDA-rosso, corretto in v5.2.0).
- **LUT 256×3** condivisa CPU/GPU: `np.interp` delle stop su 256 punti, ×255, clip, uint8;
  colore = `LUT[round(t·255)]`.
- **Kernel GPU**: 1 px/thread (v5.0.0: micro-benchmark A/B, 1.8× su z1 / 1.1× su z3
  vs 2 px/thread; le varianti sono bit-identiche tra loro), `__launch_bounds__(256)`,
  block 16×16, grid `(⌈w/16⌉, ⌈h/16⌉)`, `--use_fast_math`; 2 varianti (f32/f64) dalla
  stessa sorgente parametrizzata; NVRTC lazy.
- **D2H pinned (v4.16.0)**: buffer host pinned cacheato per dimensione
  (`cp.cuda.PinnedMemory` + `cp.cuda.runtime.memcpy(..., memcpyDeviceToHost)`, DMA ~1,7×
  del `.get()` pageable), con fallback `.get()`. La vista numpy: `np.frombuffer` su
  `ctypes` array all'indirizzo `pm.ptr`.
- **Warmup GPU all'avvio (v4.16.0)**: thread daemon che all'import fa un render 64×64
  (f32 e f64) → init context CUDA, module load e prime allocazioni (device + pinned)
  pagati FUORI dal primo render reale.
- **CPU (v4.16.0)**: array di lavoro cacheati per (w,h) (offset di griglia +
  real/imag/c/w4 + work arrays riutilizzati, max 3 dimensioni): nessuna allocazione per
  render.
- **CPU (v5.0.0)**: escape loop in **Numba** `@njit(parallel=True)` (dipendenza
  opzionale): parallelo sulle righe (`prange`), early-exit per pixel, f64; geometry,
  interior analitico e coloring restano in numpy (il kernel Numba produce `it[]` e,
  da v5.2.0, `mag[] = |z|²` alla fuga, usato dal coloring smooth).
  **Self-test di bit-identità** all'avvio (thread, 4×4 con casi limite, `it[]` **e** `mag[]`,
  vs il loop numpy di riferimento) → se fallisce o Numba è assente, **fallback numpy automatico** (stesso loop,
  stesso gate). Warmup/compilazione Numba in thread all'avvio (fuori dal primo render).
  **Cancellazione cooperativa**: ogni nuovo job incrementa il contatore di
  "generazione" (`_GEN`); il percorso Numba divide le righe in **16 bande** e
  controlla il contatore **a livello Python tra le bande** (un check in-kernel
  NON è affidabile — vedi note tecniche), il fallback a ogni iterazione; il
  worker scarta comunque il frame se la generazione è cambiata.
  `my_gen=0` disattiva la cancellazione (benchmark/CLI).
  **Attenzione FMA**: il quadrato complesso va fatto in parti esplicite (`a*a-b*b`,
  `2*a*b`, ufunc singoli) e NON con `np.square` — vedi note tecniche.

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
  collassano sull'ultimo (*latest-wins*).
- **Cancellazione cooperativa (v5.0.0)**: `_submit` incrementa `_GEN` (list `[int]`,
  atomoico); il render CPU in corso si ferma se la generazione cambia (Numba: ogni
  banda di 16 righe, check a livello Python; fallback: iterazione per iterazione)
  e il worker scarta il frame obsoleto prima di `_show`. GPU: non cancellabile
  (render <160 ms), solo scarto post-render.
- `request_render`: **preview ¼** (min 16 px) immediata, poi render full dopo **500 ms**
  (solo se la vista non è cambiata).
- UI: poll ogni **30 ms** (`tkinter.after`) che mostra l'ultimo frame (ridimensionato al canvas).

## File
- **Zona** (JSON della vista, indentato): `{"app", "versione", "cx", "cy", "half", "mi", "mi_auto"}`.
  "Salva zona" riscrive il **file corrente** (`view_file`) se c'è, altrimenti chiede il
  nome (default `mandelbrot_<AAAAmmgg_HHMMSS>.json`); "Carica zona…" ripristina vista + MI
  (clamp `half ≥ 1e-12`) e rende il file quello corrente (mostrato nel titolo).
  **All'avvio il programma parte sempre SENZA file corrente** (v5.1.1: `view_file`
  non viene più ripristinato dalla config) → "Salva zona" chiede sempre il nome finché
  l'utente non carica/salva esplicitamente una zona in quella sessione.
- **Config**: `~\mandelbrot\config.json` con
  `precision, palette, backend, bench` (la **vista** `cx, cy, half, mi, mi_auto` e
  `view_file` non sono più persistite: v5.1.1/v5.1.2, eventuali vecchi valori in
  config esistenti sono ignorati); salvata all'uscita e **throttled ~1 s** sui
  cambiamenti; reset riporta i default.
- **Avvio (v5.1.2)**: il programma parte SEMPRE con la configurazione di default —
  vista sull'intero insieme di Mandelbrot (`cx=-0.5, cy=0, half=1.5`) + MI auto —
  come la prima volta; la vista precedente si recupera solo con "Carica zona…".

## Benchmark
- Dialog di conferma custom (`Toplevel` modale centrato, griglia parametri, Avvia/Annulla,
  `Return`/`Esc`); poi thread dedicato per la durata (default 8 s).
- Default: `c = (-0.7499302568795561, -0.015139113925433963i)`, `half = 5.226737155905588e-05`,
  960×540, `secs = 8.0`.
- **`mi` non è un parametro**: sempre `auto_mi(bench['half'])` (~10 915 a default) →
  benchmark comparabile anche se la formula auto cambia.
- Parametri in `config.json` (chiave `bench`, overridibile; una vecchia `bench.mi` è ignorata).
- Esegue nella **modalità corrente** dell'app (v5.1.0, prima era sempre CUDA f32):
  motore CPU/CUDA + precisione f32/f64 come selezionati in toolbar; su CUDA usa un
  buffer proprio (no contesa col render normale). Per confrontare versioni,
  usarlo nella stessa modalità.
- Report: dialog con **rendering/s in grande** (il vero risultato, ~42pt verde), statistiche
  (n. rendering, ms/render) e griglia dei parametri; errore → "BENCHMARK FALLITO" + dettaglio.

## Note tecniche (imparati a caro prezzo)
- Le LUT **devono** essere `__constant__` **incorporate nel kernel** (una per palette):
  passate come buffer device, CuPy le sovrascriveva con l'output a ogni launch.
- Argomenti scalari CuPy = **array numpy size-1**; indice palette preallocato (`_PAL_IDX`).
- `np.asarray(array_cupy)` **non** è permesso → `.get()`.
- NVRTC: niente `-O3`/`--opt-level` (la compilazione è lazy alla prima chiamata).
- **CuPy 14**: `cp.cuda.MemoryHost` e `Event.elapsed_time` non esistono più → memoria
  host pinned con `cp.cuda.PinnedMemory(size)` (+`.ptr`) e `cp.cuda.runtime.memcpy(dst,
  src, n, kind)`; timing con `perf_counter`+sync.
- **FMA numpy (scoperta v5.0.0, critica per la bit-identità)**: `np.square` su array
  complessi è compilato da numpy 2.4 **con FMA** (`re*re - im*im` contratto in
  `fma(re, re, -(im*im))`, dipende dal build SIMD), mentre Numba senza fastmath NON
  contrae: 1 ULP di differenza su orbite caotiche → escape time diverso su una
  piccolissima frazione di pixel di bordo (0.01–0.8% a seconda dello zoom). Regola:
  in tutto il codice CPU il quadrato complesso va fatto in parti esplicite
  (`a*a-b*b`, `2*a*b` con ufunc singoli) → percorso Numba e fallback numpy restano
  bit-identici tra loro. Gate v5.2.0: GPU f32/f64 e CPU **bit-identici** ai loro
  riferimenti `baseline/*.npy`; CPU e GPUf64 devono rendere la **stessa immagine**
  (≤2% di pixel con max diff per canale > 8 — solo bordo caotico + 1-2 ULP di
  `log2`/`log` libm-vs-CUDA). (v5.0.0: il gate CPU era bit-id vs `*_cpu.npy` +
  continuità ≤1.5% vs `*_cpu_v4151.npy`, riferimento storico ancora in `baseline/`.)
- **Modello di memoria Numba (scoperta v5.0.0)**: dentro `prange` la lettura di una
  memoria scritta da un altro thread **non è affidabile**: Numba/LLVM fa hoisting del
  load fuori dal loop (semantica single-thread), quindi un bump cross-thread non è
  visto e il lavoro viene comunque eseguito (verificato sperimentalmente). La
  cancellazione cooperativa va fatta a livello Python (tra bande/segmenti), non
  dentro il kernel.
- Tkinter: `delete`/`insert` su `Entry` disabilitato sono **no-op** → prima `state="normal"`.
- Note operative (PowerShell su UNC, tool, workflow di versionamento): vedi **AGENTS.md**.
