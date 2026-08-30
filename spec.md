# Visualizzatore Mandelbrot interattivo — spec di ricreazione

> Obiettivo: descrizione concisa ma sufficiente perché un altro LLM (o sviluppatore)
> possa ricreare il programma da zero. Riferimento: `mandel.py` (un solo file), aggiornata alla **v4.15.1** (2026-08-30).
> Regola: ogni modifica al sorgente DEVE essere seguita dall'aggiornamento di questa spec (vedi AGENTS.md).

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
- Clamp: `half ≥ 1e-12` (evita zoom infinito → stato degenere).
- **Reset**: pulsante o tasto `R` → vista + tutti i settaggi ai default.
- **Salva PNG**: `Ctrl+S` / menu File.
- **Zona**: menu File "Carica zona...", "Salva zona", "Salva zona con nome..." (file JSON della vista, vedi sezione *Zona*).
- Barra di stato: `messaggio | backend | palette | render: <ms>`.
- Titolo finestra: `Insieme di Mandelbrot v<VERSIONE> - <backend>`, più ` - <nome file corrente>` se c'è una zona salvata/caricata (`view_file`).
- **UI**: font di tutti i widget a 13 pt (`TkDefaultFont`/`TkTextFont`/`TkMenuFont`); layout: controlli in alto (Motore/Palette/Precisione/Iter), riga pulsanti (etichetta + campo numerico iterazioni, `−1000`/`+1000`, `Benchmark`, `Reset`), canvas al centro, status in fondo. Menu: File → Salva immagine…(Ctrl+S), Carica zona…, Salva zona, Salva zona con nome…, Esci.

## Rendering (algoritmo)
- Mandelbrot standard: `z = z² + c`, `z₀ = 0`, escape quando `|z|² > 4`.
- **GPU**: usa coordinata **spostata** `w = z − (cx, 0)` (solo parte reale) per stabilità numerica ("ricentramento algebrico").
- **Coloring (GPU)**: `nu = it + 1 − log2(0.5·ln(|z|²))`; `t = (nu/mi)^0.35`; colore = `LUT[round(t·255)]`.
- **Coloring (CPU)**: forma più semplice `t = (it/mi)^0.35`.
- **Interior analitico (GPU e CPU)**: PRIMA del loop, i pixel interni al bulbo periodica-2 (`|c+1| ≤ 0.25`) e alla cardioide principale (`|1 − sqrt(1 − 4c)| < 1`, riscritta senza complessi come `R < 2·sqrt(0.5·(R+A))` con `A = 1 − 4·Re(c)`, `R = |1 − 4c|`) sono subito neri, saltando le `mi` iterazioni (con prefiltro bounding-box per evitare le `sqrt` sui pixel chiaramente esterni). Speedup misurato ~1.6x a vista iniziale, ~3x zoom medio, ~9x deep zoom (mi=3000); nessun falso positivo (bit-identico al kernel senza test).
- **Kernel GPU** compilato con `--use_fast_math` (passato a `cp.RawKernel` come `options=("--use_fast_math",)`).
- Punti interni (non divergenti) = **nero** (GPU: early-exit nel loop, `esc` → break).
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
- **Auto** (default ON, disattivabile): `mi = 2000·(1 + log10(1.5/half))`, clamp `[50, 50000]` — cresce con lo zoom. Funzione di modulo unica `auto_mi(half)` (costanti `MI_AUTO_BASE=2000`, `MI_AUTO_MIN/MAX=50/50000`), **condivisa col benchmark**.
- **Manual**: pulsanti `±1000` (disabilitati in auto) **e immissione diretta del valore** nel campo (attivo in manual; in auto è disabilitato ma mostra comunque il valore auto corrente, e l'etichetta diventa `Iterazioni (auto):`); commit su Invio o perdita del focus; `mi` mai sotto 50.
- **Validazione immissione**: intero tra **50 e 100000**; valore non valido → messaggio nella barra di stato e ripristino del valore precedente nel campo (nessun render).
- Disattivando l'auto, `mi` viene **congelato sul valore auto corrente** (non sul default 200), così i pulsanti ± e il campo partono dal punto giusto.

## Backend e precisione
- **CPU**: numpy, sempre **f64** (`complex128`).
- **GPU**: CUDA, precisione **f32** (default) o **f64** (più lenta su GPU consumer, ~1:32). Kernel generato in **2 varianti di precisione** dalla stessa sorgente parametrizzata.
- Toggle runtime **CPU / CUDA** (pulsanti); se CUDA non è disponibile il pulsante `CUDA` è disabilitato (stesso per i radio precisione).
- Precisione f64: kernel generato separatamente alla partenza; se la compilazione fallisce il radio `f64` resta disabilitato (si resta in f32).

## Vincoli GPU (imparati a caro prezzo)
- Le palette **devono** essere array `__constant__` **incorporati nel kernel** (uno per palette), **non** passati come buffer device: CuPy sovrascriveva la LUT con l'output a ogni launch → colori corrotti.
- Kernel: **2 px/thread** (orizzontali), early-exit, `__launch_bounds__(256)`, block `16×16`, grid `(ceil(w/32), ceil(h/16))`.
- Argomenti scalari CuPy = **array numpy size-1** (non scalari Python); l'indice palette è preallocato al boot (`_PAL_IDX`), non creato a ogni render.
- NVRTC compila **lazy** alla prima chiamata; non usare `-O3` / `--opt-level`; `--use_fast_math` sì (flag `options=` di `cp.RawKernel`).
- `np.asarray(array_cupy)` **non** è permesso → usare `.get()`.
- PowerShell 5.1: non usare `Start-Process -RedirectStandardOutput` su path UNC (timeout).

## Architettura (pipeline asincrona latest-wins)
- **Thread worker** (daemon) + `threading.Condition` + **slot job singolo**: i request in coda collassano sull'ultimo (*latest-wins*).
- `request_render`: invia subito una **preview 1/4** (min 16 px), poi il **render full dopo 500 ms** (solo se la vista non è cambiata).
- Thread principale: ogni **30 ms** (`tkinter.after`) spolia una coda e mostra l'ultimo frame (ridimensionato al canvas).
- **UI sempre responsive** (render su thread separato); **nessuna cancellazione** del render in corso.
- Metodi della classe raggruppati per funzione (in quest'ordine): Costruzione UI, Helper UI, Vista e interazione, Controlli, Pipeline rendering, File (PNG/zona/config), Benchmark.

## Zona (file JSON della vista)
- "Salva zona": riscrive l'ultimo file usato (`view_file`), altrimenti chiede il nome; "Salva zona con nome..." chiede sempre. Nome default `mandelbrot_<AAAAmmgg_HHMMSS>.json`.
- "Carica zona...": apre un JSON e ripristina `cx, cy, half, mi, mi_auto` (con clamp `half ≥ 1e-12`); il file scelto diventa il **file corrente** (`view_file`, persistito in config e mostrato nel titolo).
- Formato file: `{"app", "versione", "cx", "cy", "half", "mi", "mi_auto"}` (JSON indentato, leggibile).

## Config (persistente)
- Salva/carica `~\mandelbrot\config.json` (JSON): `cx, cy, half, mi, mi_auto, precision, palette, backend, bench, view_file`.
- Salvataggio all'uscita + **throttled ~1 s** sui cambiamenti. Reset riporta i default e salva.

## Benchmark
- Pulsante "Benchmark": **dialog di conferma custom** (`Toplevel` modale centrato, non `messagebox`): titolo, parametri in griglia etichetta+valore (font monospace), pulsanti **Avvia/Annulla** (`Return` = Avvia, `Esc`/X = Annulla); poi **thread dedicato** per la durata in config (default 8 s):
  - default: `c = (-0.7499302568795561, -0.015139113925433963i)`, `half = 5.226737155905588e-05`, `960×540`, `secs = 8.0`; nella zona di default `mi ≈ 10916`
  - **`mi` NON è più un parametro fisso**: viene SEMPRE calcolato con la stessa formula auto, `mi = auto_mi(bench['half'])` (dialog e report mostrano il valore derivato); così il benchmark resta comparabile anche se la formula auto cambia in futuro
  - gli altri parametri sono **persistiti in config.json** (chiave `bench`, overridibili); il metodo benchmark usa sempre `self.bench`, non la costante (una vecchia chiave `bench.mi` in config viene ignorata)
  - GPU: **sempre f32** (indipendente da motore/precisione selezionati), buffer proprio (no contesa col render normale); senza GPU: CPU f64
- Report: **dialog risultato custom** in cui il **rendering/s è il protagonista** (numeron grande ~42pt verde + etichetta "rendering / secondo"); sotto, statistiche secondarie (n. ripetizioni, ms/render) e griglia dei parametri del test. In caso di errore: "BENCHMARK FALLITO" in rosso + dettaglio. Lo status bar mostra comunque "benchmark completato".

## Convenzioni
- **Ogni modifica al sorgente deve essere seguita dall'aggiornamento di questa spec** (spec sempre in sincronia con `mandel.py`).
- Versionamento: header di `mandel.py` con `VERSIONE` + `STORICO`, bump + voce per ogni modifica — regole complete in AGENTS.md.
