# Visualizzatore Mandelbrot — spec di ricreazione

Descrizione concisa ma sufficiente perché un altro LLM (o sviluppatore) ricrei il
programma da zero. Riferimento: `mandel.py` (un solo file).
Ogni modifica al sorgente DEVE aggiornare anche questa spec (vedi `AGENTS.md`).

Convenzioni: `cpu|cuda|metal|vulkan` = valori di `_ACTIVE`; `Cpu/Cuda/...`
capitalizzato = label toolbar; `CUDA/METAL/VULKAN` upper-case = prefisso di
`backend()` (CPU resta `CPU`); `f32/f64` sempre minuscoli; `FMA`/`ULP` sempre
maiuscoli; decimali col punto; unità `ms/s/px/pt`.

Struttura in ordine di costruzione (§1→§12): stack → matematica → palette → MI →
backend → implementazione (§6.1–§6.4) → pipeline → UI → file → benchmark →
build → gotchas (§12.1–§12.6).

## 1. Stack e architettura
- Un solo file Python 3.14: GUI `tkinter`, rendering su 1 di 4 backend
  selezionabili — CPU (numpy/Numba), CUDA (CuPy `RawKernel`), Metal (pyobjc),
  Vulkan (wgpu/wgpu-native) — con fallback CPU; Pillow per PNG.
  Backend non disponibili restano visibili in toolbar ma disabilitati (grigi).
- Render su thread worker, UI mai bloccata (§7).
- Metodi di `MandelbrotApp` raggruppati per funzione: UI, vista, controlli,
  pipeline, file, foto, benchmark.
- Versione in due punti che devono coincidere: header `# VERSIONE: X.Y.Z`
  (letto da `mandelbrot.spec`/`build_app.py` per EXE/.app/zip) e costante
  runtime `VERSION = "X.Y.Z"` (titolo + JSON zona). Incremento/STORICO: vedi
  `AGENTS.md`.
- Dipendenze: numpy (sempre); Numba (opzionale, altrimenti fallback numpy
  single-core); CuPy (CUDA); pyobjc (Metal); wgpu (Vulkan); Pillow (PNG, icona).

## 2. Matematica del rendering
Comune alle 4 vie (differenze solo di 1–2 ULP, vedi §12.2–§12.3).
- Escape `z = z*z + c`, `|z|^2 > 4`; punti mai fuggiti = nero.
- Interior analitico prima del loop (stesso su CPU/GPU, salta le `mi`
  iterazioni): bulbo periodo-2 `|c+1| <= 0.25` (`d2 = (x0+1)^2 + y0^2 <= 0.0625`)
  e cardioide `|1 - sqrt(1-4c)| < 1` (forma senza complessi
  `R < 2*sqrt(0.5*(R+A))`, `A = 1-4*Re(c)`, `R = |1-4c|`, con prefiltro
  bounding-box).
- GPU: iterazione in coordinata spostata `w = z - cx` (solo parte reale).
- Coloring smooth identico CPU/GPU: `nu = it + 1 - log2(0.5*ln(mag))`
  (`mag = |z|^2` alla fuga, `> 4`), `t = clip(nu/mi,0,1)^0.35`.
  Solo `mag == 0` resta nero (distingue interiore da fuga a `it == 0`).
- LUT 256x3 condivisa: `make_lut` = `linspace(0,1,256)` float64,
  `np.interp` per canale, `*255`, clip, `uint8`. Indice =
  `(t*255).astype(uint8)` = troncamento (non `round`).
  `log2/log` libm-vs-device differiscono di 1–2 ULP → al massimo 1 entry LUT.
- LUT nel kernel: CUDA/Metal una `constant` per palette incorporata nel
  sorgente; Vulkan un unico buffer `storage` con tutte le palette concatenate
  (`lut[pal*256+idx]`, colori packed `0x00RRGGBB` perché WGSL non ha `u8`).

## 3. Palette
Registro `PALETTES` (fonte unica): ordine = indice kernel (0 fuoco, 1 ghiaccio,
2 termal). UI, config e LUT kernel sono generati da qui. Default `fuoco`.

| palette  | n.stop | t | R | G | B |
|---|---|---|---|---|---|
| fuoco | 6 | 0,.2,.45,.7,.9,1 | .05,.35,.85,1,1,1 | 0,.02,.2,.65,.95,1 | 0,0,.02,.15,.55,1 |
| ghiaccio | 5 | 0,.25,.5,.75,1 | .02,.05,.30,.70,1 | .02,.15,.55,.85,1 | .10,.45,.90,1,1 |
| termal | 7 | 0,.2,.4,.55,.7,.85,1 | .02,.10,.55,.95,1,1,1 | .08,.45,.80,.96,.85,.55,.30 | .28,.85,.95,.98,.45,.20,.10 |

## 4. Iterazioni (MI)
- Auto (default, formula unica condivisa col benchmark):
  `auto_mi(half) = int(max(50, min(50000, 2000*(1 + log10(1.5/max(half,1e-12))))))`.
  Vista iniziale (`half = 1.5`) → 2000; default benchmark
  (`half = 5.226737155905588e-05`) → 10915; `half = 5.2e-05` → 10920.
- Manuale: entry (commit su Invio/FocusOut; intero 50–100000, altrimenti
  messaggio in status bar + ripristino) e pulsanti `±1000`
  (`change_mi`: `max(50, mi+d)`, senza clamp superiore — può superare 100000).
  I pulsanti sono disabilitati in auto; in auto il campo è disabilitato ma
  mostra il valore corrente; uscendo dall'auto, `mi` si congela sul valore
  auto corrente. Reset ripristina `mi = MI0 = 200` + auto.
- Canvas sotto `MIN_DIM = 50` px è invalido → si usa 1280x720.

## 5. Backend e precisione
- Rilevamento all'avvio in ordine di preferenza: CUDA > Metal > Vulkan > CPU
  (`_BACKENDS_OK`, `_ACTIVE` stringa `cpu|cuda|metal|vulkan`); default = primo
  disponibile (`_default_backend()`); `compute` dispatcha su `_ACTIVE`.
- Toolbar "Motore": 4 `Radiobutton` nello stesso `Frame`, senza `variable`
  (Tk non li raggruppa da solo; la mutua esclusione è manuale via
  `.select()`/`.deselect()`); non disponibili = `state="disabled"`.
- `backend()`: GPU → `_ACTIVE.upper() + " " + _PREC` (es. `CUDA f32`);
  CPU → `"CPU " + _PREC + " multi-core"` col kernel Numba parallelo,
  `"CPU " + _PREC + " single-core (numpy)"` col fallback. La scelta segue
  `_NUMBA_OK[_PREC]` (per precisione, come `compute_cpu`), non il solo import;
  prima del warmup i render usano davvero il fallback → `single-core`.

| backend | f32 | f64 (condizione) |
|---|---|---|
| cpu | sì | sì, sempre (anche fallback numpy, senza check Numba) |
| cuda | sì | sì, solo se il kernel f64 compila (`_KERNEL_F64 is not None`) |
| metal | sì | mai (f32-only) |
| vulkan | sì | mai (f32-only) |

- GPU multipla: dropdown `GPU:` con contenuto per motore attivo (nascosto se
  il motore ha <= 1 GPU). CUDA (v5.9.0): device da `_CUDA_DEVICES`, default 0;
  render/benchmark/warmup sul device scelto via `with cp.cuda.Device(...)`
  (current-device CuPy per-thread); cambio invalida `_BUF`. Vulkan (v5.9.2):
  adapter fisici backend Vulkan da `_VULKAN_ADAPTERS` (deduplicati per
  vendor/device, default = ex high-performance); cambio via
  `VulkanBackend.select_adapter()` (ricrea risorse device-bound sotto lock).
  Titolo/stato seguono la scelta (`hw_name`); warmup 64x64 in background sul
  nuovo device/adapter se il motore è attivo. Selezioni persistite
  (`cuda_device`, `vulkan_adapter`, clampate).
- Precisione = settaggio globale unico (default f32). Bottone f64 dinamico
  (`_sync_precision_buttons`, chiamato in `_build_toolbar`/`set_backend`/`reset`;
  `load_config` passa da `_select_backend`/`_select_precision` + sync):
  abilitato solo se il backend corrente supporta f64; passando a backend
  f32-only con f64 attiva si torna a f32; `set_prec` rifiuta f64 non
  supportata. La precisione è applicata dopo il motore al load (vincolo
  d'ordine).
- Nota misurata: f32 non accelera il loop di escape (latency-bound); guadagno
  solo sui passi memory-bound (geometria/prefiltro/coloring): 1.6–1.7x
  esterno, 1.07x deep-zoom.

## 6. Implementazione per backend
Comune: matematica §2, 1 px/thread, griglia 16x16 (256 thread), LUT come §2.
Sotto solo scarti + gotcha API.

### 6.1 CPU (numpy/Numba)
- Array di lavoro cacheati per `(w,h,prec)` (offset griglia + real/imag/c/w4 +
  work array, max 6) → nessuna allocazione per render.
- Escape in Numba `@njit(parallel=True)`, `prange` sulle righe, early-exit;
  geometria/interior/coloring in numpy; produce `it[]` + `mag[]`.
- Senza Numba (o self-test fallito per quella precisione): fallback numpy
  single-core, `backend()` = `CPU f32/f64 single-core (numpy)`,
  flag `_NUMBA_AVAILABLE = (njit is not None)`, stato per precisione
  `_NUMBA_OK = {"f32": bool, "f64": bool}`.
- f32/f64: array nel dtype target (X/Y restano f64, prodotto con `half` in f64
  poi arrotondato); kernel Numba auto-specializzato `complex64/128`;
  fallback = stessa sequenza di ufunc in parti esplicite (no FMA) anche in f32.
- Self-test all'avvio in thread (4x4 con casi limite vs loop numpy), una
  precisione per volta → fallback numpy automatico per la precisione che
  fallisce. Criterio: f64 bit-identità di `it[]` e `mag[]`; f32 `it[]` esatto +
  `mag[]` entro 1e-2 relativo (FMA Numba-f32 contesto-dipendente, §12.2).
- Warmup Numba di entrambe le precisioni in thread all'avvio.
- Cancellazione cooperativa: vedi §7.

### 6.2 CUDA (CuPy `RawKernel`)
- 1 px/thread, `__launch_bounds__(256)`, block 16x16, grid
  `(ceil(w/16), ceil(h/16))`, `options=("--use_fast_math",)`; 2 varianti
  f32/f64 dalla stessa sorgente; NVRTC lazy (no `-O3`).
- Scalari = array numpy size-1; indice palette preallocato (`_PAL_IDX`);
  `np.asarray` su array CuPy vietato → `.get()`.
- D2H pinned: `PinnedMemory(size)` + `runtime.memcpy(..., memcpyDeviceToHost)`
  cacheato per dimensione (DMA 1.7x di `.get()` pageable), fallback `.get()`;
  vista numpy via `np.frombuffer` su `ctypes` all'indirizzo `pm.ptr`.
- CuPy 14: niente `MemoryHost` / `Event.elapsed_time` → `PinnedMemory` + `ptr`,
  timing `perf_counter` + sync.
- Warmup all'avvio in daemon: render 64x64 f32 (+ f64 se supportata) → context,
  module load e prime allocazioni fuori dal primo render.

### 6.3 Metal (pyobjc, f32-only)
- Kernel MSL `mandel`, threadgroup 16x16 (256, sotto max 1024 di M1),
  grid `(ceil(w/16), ceil(h/16))`.
- Parametri `(cx,cy,half,w,h,mi,pal)` in uno `struct` via `[[buffer(1)]]`
  (né scalari nudi né singoli float); output RGB888 in `[[buffer(0)]]`.
- H2D/D2H: `buf.contents().as_buffer(N)` (memoryview C-level, circa 0.01 ms;
  la varlist non è bytes-like).
- Deterministico (2 run bit-identici). Misurato 12–30x la CPU su M1.
- Gotcha: `fmin/fmax/pow/sqrt/log2/log` (senza `f`); `half` è tipo riservato →
  campo `hs`; puntatori con address space esplicito (`device`/`constant`);
  `setBuffer:offset:atIndex:` = `(buffer, offset, index)` → `(pbuf, 0, 1)`;
  compute sincrono (`commit` + `waitUntilCompleted` + read) + `Lock`.

### 6.4 Vulkan (wgpu 0.32, f32-only)
- Shader WGSL `main`, workgroup 16x16 (256), dispatch
  `(ceil(w/16), ceil(h/16))`. Lib nativa nella wheel, nessuna runtime esterna;
  funziona su AMD/NVIDIA/Intel.
- Vincoli WGSL: output e LUT = `array<u32>` packed `0x00RRGGBB` (unpack numpy
  `(h,w,3)` via shift al readback); parametri in 2 uniform
  (`vec4<f32>` cx,cy,half + `vec4<i32>` w,h,mi,pal); LUT storage concatenata;
  built-in senza `f`; `var`/`let`; `break` ultimo del blocco.
- Deterministico (2 run bit-identici, verificato su AMD 780M).
- H2D `queue.write_buffer` (buffer persistenti `UNIFORM|COPY_DST`); D2H
  `queue.read_buffer` (buffer `STORAGE|COPY_SRC`); `queue.submit([enc.finish()])`
  + `Lock` (sincrono).
- Selezione adapter (v5.9.2): enumera gli adapter, tiene uno per GPU fisica
  con `backend_type == "Vulkan"` (scarta dupe D3D12/OpenGL/WARP); default =
  ex high-performance; `select_adapter()` ricrea device/pipeline/buffer
  sotto il lock esistente (mai in gara con `compute`).
- API "WebGPU-style": `gpu.enumerate_adapters_sync()`,
  `gpu.request_adapter_sync(power_preference="high-performance")` (solo per
  individuare il default),
  `adapter.request_device_sync()`, `create_shader_module(code=...)`,
  `create_compute_pipeline(layout="auto", compute=ProgrammableStage(...))`,
  `create_buffer(size=, usage=)` / `create_buffer_with_data(data=, usage=)`
  (keyword, non descrittore), `create_bind_group(layout=, entries=
  [BindGroupEntry(binding=N, resource=buf)])`, layout da
  `pipe.get_bind_group_layout(0)`; bundle via `collect_all(wgpu/cffi)` +
  `hiddenimports wgpu.backends.wgpu_native`.

## 7. Pipeline (asincrona, latest-wins)
- Worker daemon + `Condition` + slot job singolo; request in coda collassano
  sull'ultimo. Durante il benchmark il worker scarta i job (render
  interattivi sospesi: niente contesa col thread bench sullo stesso device,
  su GPU display evita TDR/reset).
- `_submit`: `self._gen += 1; _GEN[0] = self._gen` (`_GEN` =
  `np.zeros(1, int32)`); worker scarta il frame se `_GEN[0] != gen` prima di
  `frames.put` (mostrato poi da `_poll`).
- Cancellazione cooperativa solo CPU: Numba ogni banda (16 bande totali,
  `band = ceil(h/16)`; check Python tra bande — check in-kernel inaffidabile
  per hoisting in `prange`, §12.4); fallback numpy ogni iterazione.
  `my_gen = 0` disattiva il check (uso interno benchmark).
  GPU non cancellabile (render tipico sotto 160 ms), solo scarto post-render.
- `request_render`: preview immediata `max(w//div,16)` con `div = 8` se CPU
  (draft leggero durante l'interazione), `div = 4` su GPU; poi full dopo
  500 ms solo se vista invariata. Poll UI ogni 30 ms (`after`), resize `NEAREST`.

## 8. UI e interazione
### 8.1 Vista e navigazione
- Avvio: centro `(-0.5, 0)`, `half = 1.5`, canvas 1280x720 (16:9); asse y scalato `h/w`;
  clamp `half >= MIN_HALF = 1e-12`.
- Rotella x1.25/x0.8 al cursore; click x2 al cursore; click destro x0.5 al
  cursore (due dita sul trackpad; bind su `ButtonPress-2/3`); `+`/`-` x2/x0.5
  al centro e `r` = reset legati alla finestra (`root.bind` con guardia che
  li ignora col focus su una `Entry`; il click sul canvas prende il focus).
- `r` = reset; `Ctrl+S` (bind su root) = salva PNG.

### 8.2 Layout, titolo, status
- Toolbar: `Motore:` (4 radio + dropdown `GPU:` con device/adapter del
  motore attivo, se > 1) / `Palette:` (check dal registro) / `Precisione:`
  (`f32`/`f64`) / `Iter:` + `Auto`. Seconda riga: label
  `Iterazioni:` / `Iterazioni (auto):`, entry (largh. 7, allineata a destra),
  `-1000`/`+1000`, `Ricalcola 2x2`, `Ricalcola 4x4`, `Ricalcola`, `Benchmark`,
  `Reset`. Poi canvas, poi status bar.
  Font `TkDefaultFont/TkTextFont/TkMenuFont` 13 pt.
   Menu File: `Salva immagine... (Ctrl+S)`, `Carica zona...`, `Salva zona`,
   `Salva zona con nome...`, separatore, `Esci`.
   Menu Help: `Istruzioni...` (guida rapida: navigazione, motore/precisione,
   iterazioni, palette, file, foto, benchmark), `Novità recenti...` (ultime 10
   modifiche da tupla embedded `HISTORY`, righe versione/data/descrizione),
   separatore, `Informazioni...` (versione, backend/hardware attivi, autore
   Francesco Ferrara `<occhiobello@gmail.com>`, diagnostica CPU/Numba da
   `_numba_diag()`). Dialog `Toplevel` modali centrati via `_modal`. `HISTORY` è embedded (non STORICO commentato, che
   PyInstaller scarta): va aggiornata a ogni bump (voce in testa, max 10).
- Titolo: `Insieme di Mandelbrot v<VER> - <backend()> (<hw>)`
  + ` - <basename zona>` se presente; ricalcolato a ogni frame in `_show`
  così segue la transizione warmup `single-core` → `multi-core`.
- Status: `<msg> | <backend> · <hw> | <palette> | render: <N> ms`
  (+ ` · single-core: motivo in Help > Informazioni` in rosso se la CPU è
  sul fallback numpy; si auto-cancella al multi-core). Esito warmup in
  `_NUMBA_STATUS` (import/self-test/compilazione/tempo per precisione).
- `Entry` disabilitata: `delete`/`insert` sono no-op → riabilitare prima (§12.6).

### 8.3 Hardware, cursore, tema
- `hw_name()` in cache per backend, senza nuove dipendenze; fallback
  `CPU`/`GPU` su errore o nome vuoto: CPU da registro Windows
  (`ProcessorNameString`) / `sysctl machdep.cpu.brand_string` su macOS /
   `platform.processor()` altrove; CUDA da `getDeviceProperties()` sul device
   selezionato (dropdown GPU, default 0; con più GPU l'ordine può differire
   da `nvidia-smi`); Metal da
   `dev.name` (se callable va invocato); Vulkan dall'adapter selezionato
   (`VulkanBackend.name`, default ex high-performance).
- Benchmark: cursore `watch` in `run_benchmark()`, ripristino in
  `_bench_done()` (unico punto di uscita, anche su errore).
- Ricalcola NxN (`Ricalcola 2x2`/`Ricalcola 4x4`): ricalcola la vista a NxN
  per lato (N=2/N=4; 4x4 = 16x pixel, molto piu' lento e pesante in memoria)
  in thread dedicato e mostra la media NxN su RGB (`take_photo(n)`/`_photo_worker`/`_photo_done`;
  box-filter sull'output, unico code-path per tutti i backend dato che la
  GPU colora in-kernel); cursore `watch` durante il calcolo, ripristino in
  `_photo_done()` (unico punto di uscita, anche su errore); all'avvio
  invalida i render interattivi in volo/pendenti (bump `_GEN` + cancella
  il full-timer ritardato) e il risultato e' scartato se vista, palette,
  motore o precisione cambiano nel frattempo (snapshot esteso, non
  generazione); `self.pil` aggiornata così `Ctrl+S` salva la versione
  antialiased. Un ricalcolo alla volta (secondo click: `ricalcolo gia' in corso`).
- Ricalcola: rifa il rendering della vista corrente (`request_render`).
- Tema: Button/Checkbutton/Radiobutton nativi (mai colorati: su macOS
  degradano a flat illeggibili, lezione 5.4.2); colore solo su Label/Frame:
  status bar e barra accento (3px sopra il canvas) nel colore del motore
  (`BACKEND_FG`: CPU `#1f6feb`, CUDA `#2ea44f`, Metal `#8957e5`, Vulkan
  `#d97706`), titoli dialog Help colorati, errori in rosso `#e5534b`.
  Dialog benchmark: testo di sistema; accenti OK/KO `#2ea44f`/`#e5534b`,
  divisore `#8a8a8a`; grafico: griglia `#a0a0a0`, assi/divisore `#8a8a8a`.
  Solo UI, nessun effetto sul rendering.
- Reset: vista + `mi = 200` + auto + palette `fuoco` + backend default +
  device CUDA 0 + adapter Vulkan 0 + f32. Non tocca benchmark, `view_file`,
  cache hardware.

## 9. File I/O
| file | path | chiavi | note |
|---|---|---|---|
| zona (JSON indentato) | a scelta, default `mandelbrot_AAAAMMGG_HHMMSS.json` | `app, versione, cx, cy, half, mi, mi_auto` | `Carica` ripristina vista+MI (clamp `half`), rende file corrente (titolo). `Salva zona` voce disabilitata se nessun file corrente; `save_zone()` senza file ricade su `save_as`. PNG usa stesso pattern nome |
| config | `~/mandelbrot/config.json` | `precision, palette, backend, cuda_device, vulkan_adapter, bench` | vista e `view_file` mai persistiti (vecchi valori ignorati); backend legacy `"gpu"` → default, ignoto → default; `cuda_device`/`vulkan_adapter` clampati al range (default 0); salvataggio all'uscita + throttle 1 s (`after(1000)`; `request_render` marca dirty anche per sola vista) |

- Avvio sempre da default (insieme intero + MI auto); vista precedente solo via
  `Carica zona...`.

## 10. Benchmark
- Dialog conferma custom (`Toplevel` modale centrato, Avvia/Annulla,
  `Return`/`Esc`); poi thread dedicato per `secs`.
- Default (in `config.json` chiave `bench`, overridibile; vecchia `bench.mi`
  ignorata):

| par | cx | cy | half | w x h | secs | mi |
|---|---|---|---|---|---|---|
| val | -0.7499302568795561 | -0.015139113925433963 | 5.226737155905588e-05 | 960x540 | 8.0 | sempre `auto_mi(half)` = 10915 |

- Esegue nel modo corrente (motore + f32/f64 di toolbar, GPU/adapter
  selezionato); CUDA usa buffer proprio (allocato sul device scelto),
  Metal/Vulkan output proprio, CPU memoria numpy. Per gli 8 s ha la GPU in
  esclusiva (worker in pausa, §7); se il device è occupato/in reset il dialog
  FALLITO lo segnala con hint (riprovare o riavviare).
- Report: dialog con `rendering/s` grande (42 pt, `#2ea44f`), statistiche
  (n. rendering, ms/render), grafico + griglia parametri (riga `Hardware` da
  `hw_name()`); errore → `BENCHMARK FALLITO` + dettaglio (`#e5534b`).

| barra | valore (rendering/s) | stile |
|---|---|---|
| `AMD 9900X (storico)` | 6.62 | grigia |
| `4070 Super Vulkan (storico)` | 124.0 | grigia |
| `5070 Ti Vulkan (storico)` | 179.0 | grigia |
| `4070 Super CUDA (storico)` | 250.0 | grigia |
| `5070 Ti CUDA (storico)` | 350.0 | grigia |
| `<backend()> — questa run` | misurato | verde (solo metodo, no hw) |

  Grafico `tk.Canvas` 590xN (margini 230,60,8,26; barre 20 px, gap 10 px;
  altezza N = 8 + n*20 + (n-1)*10 + 26 dal numero di barre):
  scala lineare 0→max, step nice 1/2/5×10^n, valore a fine barra.

## 11. Build dell'app (multipiattaforma)
One-dir self-contained, nessuna dipendenza per l'utente tranne driver/runtime
CUDA (solo per CUDA; Vulkan bundled).

| piattaforma | app (disco di sistema, temp) | distributivo (progetto `dist/`) | GPU |
|---|---|---|---|
| Windows | `mandelbrot_dist/Mandelbrot/Mandelbrot.exe` + `_internal/` | `dist/Mandelbrot-v<ver>-win64.zip` | Vulkan bundled; CUDA solo con driver+runtime utente (cuda-pathfinder: `CUDA_PATH`/PATH/Program Files) |
| macOS | `mandelbrot_dist/Mandelbrot.app` (firma ad-hoc) | `dist/Mandelbrot-v<ver>-macos.dmg` (staging `mandelbrot_dmg` + link `/Applications`, `hdiutil UDZO`) | Metal + Vulkan bundled |

- Script unico `build_app.py`: icona → PyInstaller (`mandelbrot.spec`,
  `--workpath mandelbrot_build` + `--distpath mandelbrot_dist` in temp) →
  post (verifica + zip/dmg) → cleanup. Intermedie `build/` mai nel progetto
  (lento su share di rete).
- Ricetta `mandelbrot.spec`: comune `collect_all(numba/llvmlite)`,
  `PIL._tkinter_finder`, escluse `torch/matplotlib/IPython/pytest`;
  win32 `collect_all(cupy)` (solo moduli, no DLL CUDA) +
  `collect_all(wgpu/cffi)` + `hiddenimports wgpu.backends.wgpu_native`, icona
  `.ico`, versioninfo, hook `hook_dlldir.py` (`os.add_dll_directory` su EXE +
  `_internal`); darwin `collect_all(objc/Foundation/Metal)` +
  `collect_all(wgpu/cffi)`, `libomp.dylib` da `torch/lib`, `BUNDLE` (icona
  `.icns`, bundle_id, plist).
- Ritenzione: `KEEP_N = 3` versioni per piattaforma in `dist/` (zip e dmg
  separatamente; rimozione automatica delle più vecchie per X.Y.Z).
- Icona `make_icon.py`: render 1024x1024 (CPU f64, `termal`, `mi = 800`) →
  `icon_src.png` + `mandelbrot.ico` (16,24,32,48,64,128,256) + su macOS
  `mandelbrot.icns` via Pillow. Versione distributivi dall'header `# VERSIONE`
  (§1). Processo/ritenzione/gitignore: vedi `AGENTS.md`.

## 12. Note tecniche (gotchas)
### 12.1 LUT come costanti
Passate come buffer device, CuPy le sovrascriveva con l'output → incorporate
nel sorgente (CUDA/Metal) o storage concatenato (Vulkan, §2).

### 12.2 FMA numpy (critica per bit-identità)
`np.square` su complessi è compilato con FMA (build SIMD-dipendente), Numba
senza fastmath no → 1 ULP su orbite caotiche, escape diverso su pochi pixel di
bordo. Regola: quadrato in parti esplicite (`a*a-b*b`, `2*a*b`, ufunc singoli)
→ f64 bit-identico. Eccezione f32: Numba/LLVM contrae in FMA in modo
contesto-dipendente (non riproducibile in numpy) → `it[]` uguale, al massimo 1
entry LUT; per questo il self-test f32 usa `it[]` esatto + `mag[]` entro 1e-2.

### 12.3 Associatività IEEE
Riordinare `*`/`/` cambia l'arrotondamento: `cx + half*X/s` non è bit-identico
a `cx + half*(X/s)`. Conservare l'ordine e verificare col gate multi-zona.

### 12.4 Memoria Numba + cancellazione
In `prange` la lettura cross-thread non è affidabile (hoisting) → check di
cancellazione solo a livello Python tra bande (§7). Inoltre: niente
`break`/`continue` in `prange` (uccide parallelismo); `cache=True` inutile
(modulo rinominato a ogni load; paga il warmup); nessun ufunc `fma` in Numba.

### 12.5 Gate di correttezza (soglie per coppia)
| coppia | criterio |
|---|---|
| CPU f64/f32 vs riferimento (`<zona>_cpu.npy` / `<zona>_cpu_f32.npy`) | bit-identico |
| CUDA f32/f64 vs `<zona>_gpu_f32/_gpu_f64.npy` | bit-identico |
| CPUf64 vs CUDAf64 | al massimo 2% pixel con diff > 8 (bordo + ULP libm-vs-CUDA) |
| Metal f32 | bit-identico a `<zona>_metal_f32.npy` + 2 run identiche + scarto vs CPUf32 entro `max(2%, 1.5x CPUf32-vs-CUDAf32)` (gold CUDAf32 = righello nube f32) |
| Deep-zoom f32 (Numba/CUDA/Metal/Vulkan) | 10–25% pixel di bordo divergono per FMA: atteso, non difetto |

### 12.6 Tkinter
`delete`/`insert` su `Entry` disabilitata sono no-op → `state="normal"` prima.
- Note operative (versionamento, GPU condivisa con llama.cpp, benchmark
  bounded/A-B, warmup clock dopo pause > 10 s): vedi `AGENTS.md`.
