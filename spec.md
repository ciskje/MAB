# Visualizzatore Mandelbrot — spec di ricreazione

Descrizione concisa ma sufficiente perché un altro LLM (o sviluppatore) ricrei il
 programma da zero. Riferimento: `mandel.py` (un solo file).
Ogni modifica al sorgente DEVE aggiornare anche questa spec (vedi AGENTS.md).

## Panoramica
- Un solo file **Python 3.12**: GUI **tkinter**, rendering su uno di **4 backend
  selezionabili** — **CPU (numpy/Numba)**, **CUDA (CuPy `RawKernel`)** su NVIDIA,
  **Metal (pyobjc)** su Apple Silicon, **Vulkan (wgpu/wgpu-native)** cross-platform
  (AMD/NVIDIA/Intel) — con fallback **CPU (numpy/Numba)**, Pillow per il PNG.
  I backend non disponibili restano visibili in toolbar ma disabilitati.
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
- **Tema UI**: l'UI usa i **widget nativi macOS** (nessun colore forzato) che
  seguono il tema del **sistema** — in Dark Mode i bottoni/checkbox/radio restano
  aqua scuri con testo chiaro, senza la cornice nera che si otteneva forzando un
  `background` esplicito (che li degradava a widget *flat*, testo illeggibile).
  I dialog di benchmark usano il **colore testo di sistema** (dinamico, sempre
  leggibile su chiaro e scuro); solo gli accenti OK/KO sono fissi in toni medi che
  leggono su entrambi (`#2ea44f`/`#e5534b`) e il divisore è grigio medio (`#8a8a8a`).
  **Solo colori UI, nessun effetto sul rendering.**

## Rendering
- `z = z² + c`, escape `|z|² > 4`; punti interni = **nero** (early-exit nel loop).
- **Interior analitico** (GPU e CPU, prima del loop): bulbo periodica-2 `|c+1| ≤ 0.25` e
  cardioide principale `|1 − √(1−4c)| < 1` (riscritta senza complessi: `R < 2·√(0.5·(R+A))`,
  `A = 1−4·Re c`, `R = |1−4c|`, con prefiltro bounding-box) → subito neri, saltano le `mi`
  iterazioni. Bit-identico al kernel senza test.
- **GPU**: iterazione in coordinata spostata `w = z − cx` (solo parte reale, stabilità
  numerica); coloring continuo `nu = it + 1 − log2(0.5·ln|z|²)`, `t = (nu/mi)^0.35`.
   **CPU**: **stesso** coloring continuo della GPU `nu = it + 1 − log2(0.5·ln|z|²)`,
  `t = (nu/mi)^0.35` (il kernel Numba/fallback esportano anche `mag = |z|²` alla fuga);
   solo i pixel mai fuggiti (`mag==0`) restano neri. CPU e GPU (CUDA/Metal/Vulkan)
   producono lo stesso colore (a parte 1-2 ULP di `log2`/`log` libm-vs-kernel →
   ≤1 entry di LUT; verificato dal gate).
- **LUT 256×3** condivisa CPU/GPU: `np.interp` delle stop su 256 punti, ×255, clip, uint8;
  colore = `LUT[round(t·255)]`.
- **Kernel GPU (CUDA)**: 1 px/thread (micro-benchmark A/B, 1.8× su z1 / 1.1× su z3
  vs 2 px/thread; le varianti sono bit-identiche tra loro), `__launch_bounds__(256)`,
  block 16×16, grid `(⌈w/16⌉, ⌈h/16⌉)`, `--use_fast_math`; 2 varianti (f32/f64) dalla
  stessa sorgente parametrizzata; NVRTC lazy.
- **Kernel GPU (Metal, Apple Silicon)**: kernel MSL `mandel`, 1 px/thread,
  threadgroup 16×16 (256), grid `(⌈w/16⌉, ⌈h/16⌉)`; **stesso** criterio di escape,
  interior analitico e coloring smooth del kernel CUDA e della CPU (le tre vie producono
  lo stesso colore, a parte 1-qualche ULP di FMA/contrazione sul bordo caotico).
  Parametri `(cx,cy,half,w,h,mi,pal)` in uno **struct** passato via buffer `[[buffer(1)]]`
  (ne' scalari nudi ne' `[[buffer]]` su singoli float); output RGB888 in buffer
  `[[buffer(0)]]`; LUT incornicate nel sorgente MSL (una `constant` per palette, come
  CUDA). **f32-only**: su Apple Silicon `double` non è supportato → il backend Metal non
  fa f64 (f64 resta solo CPU/CUDA). Deterministico (2 run bit-identici). H2D/D2H via
   `buf.contents().as_buffer(N)` (memoryview scrivibile, C-level, ~0.01 ms). 12–30× più
    veloce della CPU (misurato su M1).
- **Kernel GPU (Vulkan, wgpu/wgpu-native, cross-platform)**: compute shader WGSL
   `main`, 1 px/thread, workgroup 16×16 (256), dispatch `(⌈w/16⌉, ⌈h/16⌉)`; **stesso**
   criterio di escape, interior analitico e coloring smooth di CUDA/MSL/CPU.
   **Vincoli WGSL** (vs MSL/CUDA): nessun tipo `u8` → **LUT e output sono `array<u32>`
   con colori packed `0x00RRGGBB`** (1 px = 1 `u32`); al readback si unpacka in numpy
   `(h,w,3)` (shift R/G/B). Parametri in **due buffer uniform**: `vec4<f32>`
   (cx,cy,half) + `vec4<i32>` (w,h,mi,pal); LUT in un buffer `storage` con tutte le
   palette concatenate (selezione `lut[pal*256+idx]`, come CUDA/MSL); built-in
   `min/max/pow/sqrt/log/log2` (senza suffisso `f`, a differenza di MSL). **f32-only**
   (come Metal). Deterministico (2 run bit-identici, verificato su 780M). H2D via
   `queue.write_buffer` su buffer persistenti (uso `UNIFORM|COPY_DST`), D2H via
   `queue.read_buffer` su buffer `STORAGE|COPY_SRC`; un `threading.Lock` serializza
   (compute sincrono: `submit` + `read_buffer`). Funziona out-of-the-box su
   AMD/NVIDIA/Intel (lib nativa nella wheel, no runtime esterno).
- **D2H pinned**: buffer host pinned cacheato per dimensione
  (`cp.cuda.PinnedMemory` + `cp.cuda.runtime.memcpy(..., memcpyDeviceToHost)`, DMA ~1,7×
  del `.get()` pageable), con fallback `.get()`. La vista numpy: `np.frombuffer` su
  `ctypes` array all'indirizzo `pm.ptr`.
- **Warmup GPU all'avvio**: thread daemon che all'import fa un render 64×64
  (f32 e f64) → init context CUDA, module load e prime allocazioni (device + pinned)
  pagati FUORI dal primo render reale.
- **CPU**: array di lavoro cacheati per (w,h,prec) (per precisione,
  offset di griglia + real/imag/c/w4 + work arrays riutilizzati, max 6 dimensioni):
  nessuna allocazione per render.
- **CPU**: escape loop in **Numba** `@njit(parallel=True)` (dipendenza
  opzionale): parallelo sulle righe (`prange`), early-exit per pixel; geometry,
  interior analitico e coloring restano in numpy (il kernel Numba produce `it[]` e,
   `mag[] = |z|²` alla fuga, usato dal coloring smooth).
   **CPU f32/f64**: la precisione (f32/f64) vale anche per il motore CPU
  (prima sempre f64): tutti gli array di lavoro sono del dtype corrispondente
  (X/Y restano f64: il prodotto con `half` si calcola in f64 e si arrotonda al
  dtype target), il kernel Numba si **auto-specializza** su `complex64`/`complex128`,
  il fallback numpy usa la stessa sequenza di ufunc in parti esplicite (no FMA)
  anche in f32.
   **Self-test** all'avvio (thread, 4×4 con casi limite, vs il loop numpy di
   riferimento),     **una precisione per volta** → se fallisce per una
   precisione (o Numba è assente), **fallback numpy automatico per quella
   precisione** (stesso loop, stesso gate). Criterio per precisione: **f64 =
   bit-identità** di `it[]` **e** `mag[]` (Numba f64 e numpy f64 usano la stessa
   aritmetica); **f32 = `it[]` (decisioni di fuga) esatto + `mag[]` entro 1e-2
   relativo** — in f32 Numba/LLVM contrae in FMA / riassocia in modo *dipendente
   dal contesto del loop* (comportamento intrinseco, non riproducibile in modo
   affidabile con numpy), la differenza è 1-qualche ULP su una piccolissima
   frazione di pixel di bordo caotico (al più 1 entry di LUT nel coloring).
   Warmup/compilazione Numba di entrambe le precisioni in thread all'avvio
   (fuori dal primo render).
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
- **CPU**: numpy/Numba, **f32** o **f64** (prima sempre f64). Sempre disponibile.
- **Selezione 4 backend (v5.6.0)**: la toolbar "Motore" ha 4 **radio** —
  `CPU` / `CUDA` / `Metal` / `Vulkan` (stesso frame = gruppo radio unico). I backend
  **non disponibili restano visibili ma disabilitati** (grigi). Rilevamento all'avvio:
  `_CUDA_OK`/`_METAL_OK`/`_VULKAN_OK` per ciascuno; `_BACKENDS_OK` è l'elenco di quelli
  disponibili in ordine di preferenza (CUDA > Metal > Vulkan > CPU); `_ACTIVE` è il
  backend corrente (stringa `cpu|cuda|metal|vulkan`); il **default** è il primo GPU
  disponibile, altrimenti CPU (`_default_backend`). `compute` dispatcha su `_ACTIVE`;
  `compute_gpu` è un dispatcher (`_compute_gpu_cuda` / `_compute_gpu_metal` /
  `_compute_gpu_vulkan`). `backend()` mostra "CUDA f32|f64" / "Metal f32" /
  "VULKAN f32" / "CPU f32|f64".
- **f64**: disponibile su CPU (Numba) e su CUDA (se il kernel f64 si compila).
  **Metal e Vulkan sono f32-only** (Apple Silicon non supporta `double`; su Vulkan la
  scelta è la stessa di Metal). Di conseguenza:
  - la precisione è un unico settaggio globale (default f32), ma il **bottone f64 ha
    stato dinamico**: abilitato solo se il backend corrente lo fa — CPU sempre, CUDA con
    kernel f64, **Metal e Vulkan mai** (`_sync_precision_buttons`, chiamato a ogni cambio
    motore, in `set_backend`/`reset`/`load_config`);
  - passando a un backend f32-only con f64 attiva, si torna a f32;
  - `set_prec`/`_select_precision` rifiutano f64 se il backend corrente non lo fa.
- Senza GPU il default resta CPU (i radio GPU restano grigi). Con Metal/Vulkan attivo
  il bottone f64 è disabilitato; tornando a CPU f32/f64 sono di nuovo selezionabili.
- **Nota (misurata)**: f32 NON accelera il loop di escape (catena dipendente
  seriale, latency-bound: scalare f32 ≈ scalare f64, in deep zoom leggermente più
  lento); il guadagno è solo sui passi memory-bound (geometria/prefiltro/coloring):
  ~1.6–1.7× sulle regioni esterne, ~1.07× in deep zoom (regione del benchmark).

## Pipeline (asincrona, latest-wins)
- Thread worker (daemon) + `threading.Condition` + **slot job singolo**: i request in coda
  collassano sull'ultimo (*latest-wins*).
- **Cancellazione cooperativa**: `_submit` incrementa `_GEN` (list `[int]`,
  atomoico); il render CPU in corso si ferma se la generazione cambia (Numba: ogni
  banda di 16 righe, check a livello Python; fallback: iterazione per iterazione)
  e il worker scarta il frame obsoleto prima di `_show`. GPU: non cancellabile
  (render <160 ms), solo scarto post-render.
- `request_render`: **preview ¼** (min 16 px) immediata, poi render full dopo **500 ms**
  (solo se la vista non è cambiata).
- UI: poll ogni **30 ms** (`tkinter.after`) che mostra l'ultimo frame (ridimensionato al canvas).

## File
- **Zona** (JSON della vista, indentato): `{"app", "versione", "cx", "cy", "half", "mi", "mi_auto"}`.
  **"Salva zona" è DISABILITATO finché non esiste un file di zona corrente** (`view_file=None`,
   es. all'avvio); si abilita dopo "Salva zona con nome…" o "Carica zona…".
  "Salva zona" riscrive il **file corrente** (`view_file`); "Salva zona con nome…" chiede il
  nome (default `mandelbrot_<AAAAmmgg_HHMMSS>.json`) e lo rende file corrente; "Carica zona…"
  ripristina vista + MI (clamp `half ≥ 1e-12`) e rende il file quello corrente (mostrato nel titolo).
   **All'avvio il programma parte sempre SENZA file corrente** (`view_file`
  non viene più ripristinato dalla config) → "Salva zona" resta disabilitato finché
  l'utente non definisce un nome con "Salva zona con nome…" o non carica una zona.
- **Config**: `~/mandelbrot/config.json` con
   `precision, palette, backend, bench` — `backend` = il backend **attivo** per nome
   ∈ {`"cpu"`, `"cuda"`, `"metal"`, `"vulkan"`} (v5.6.0: prima era `"cpu"`/`"gpu"`; i
  vecchi valori `"gpu"`/`"cuda"` sono migrati al default GPU / a `"cuda"` alla
  lettura, e un backend non disponibile resta sul default di avvio) — (la **vista**
   `cx, cy, half, mi, mi_auto` e `view_file` non sono più persistite: eventuali
   vecchi valori in config esistenti sono ignorati); salvata all'uscita e **throttled
   ~1 s** sui cambiamenti; reset riporta i default.
- **Avvio**: il programma parte SEMPRE con la configurazione di default —
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
- Esegue nella **modalità corrente** dell'app (prima era sempre CUDA f32):
  motore (CPU/CUDA/Metal/Vulkan) + precisione f32/f64 come selezionati in toolbar;
  su CUDA usa un buffer proprio (no contesa col render normale), Metal e Vulkan
  usano il proprio buffer di output, CPU la memoria numpy. Per confrontare versioni,
  usarlo nella stessa modalità.
- Report: dialog con **rendering/s in grande** (il vero risultato, ~42pt verde), statistiche
  (n. rendering, ms/render) e griglia dei parametri; errore → "BENCHMARK FALLITO" + dettaglio.

## Build dell'app (multipiattaforma)
- **one-dir self-contained**, senza dipendenze di Python/librerie per l'utente:
  la CPU è sempre disponibile (Numba/numpy bundled); la **GPU Vulkan (wgpu) è
  bundled e funziona out-of-the-box** (lib nativa nella wheel, AMD/NVIDIA/Intel);
  la GPU CUDA richiede solo che l'utente abbia driver + runtime NVIDIA:
  - **Windows** → `dist/Mandelbrot/Mandelbrot.exe` + `_internal/` + zip
    `dist/Mandelbrot-v<ver>-win64.zip`. GPU CuPy incluso ma **runtime CUDA NON
    bundled**: la GPU CUDA funziona solo se l'utente installa driver NVIDIA + runtime
    CUDA (toolkit o pacchetti pip `nvidia/*`), che CuPy trova a runtime via
    cuda-pathfinder da `CUDA_PATH`/PATH/Program Files. GPU Vulkan (wgpu) invece
    bundled e subito disponibile. Senza alcuna GPU l'app degrada su CPU.
  - **macOS** → `dist/Mandelbrot.app` (GPU Metal/pyobjc + Vulkan/wgpu, firma ad-hoc).
- **Script**: `build_app.py` (unico script, ramificato su `sys.platform`:
  icona → PyInstaller → post). `build_app.sh` lo redirect (compatibilità); su
  Windows si usa `build_app.ps1` (o `python build_app.py`).
- **Ricetta**: `mandelbrot.spec` (PyInstaller), multipiattaforma:
  - comune: `collect_all(numba/llvmlite)` (dylib/dll di llvmlite + dati JIT),
    `PIL._tkinter_finder`, esclude `torch/matplotlib/IPython/pytest`;
  - **win32**: `collect_all(cupy)` (solo moduli, **NO DLL CUDA**: il runtime
    NVIDIA non è bundle, l'utente lo installa e CuPy lo trova via
    cuda-pathfinder), `collect_all(wgpu/cffi)` (GPU Vulkan, lib nativa
    `wgpu/resources/*.dll` bundled + hook `hook-wgpu.py` di wgpu), icona
    `mandelbrot.ico`, versione EXE (versioninfo),
    runtime hook `hook_dlldir.py` (`os.add_dll_directory` su cartella EXE +
    `_internal`, difensivo);
  - **darwin**: `collect_all(objc/Foundation/Metal)` + submoduli,
    `collect_all(wgpu/cffi)` (GPU Vulkan), `libomp.dylib`
    da `torch/lib` (il rpath di `omppool` è hardcoded a una dir CI inesistente),
    `BUNDLE` → `.app` (icon `.icns`, bundle_id, plist).
- **Icona**: `make_icon.py` renderizza 1024×1024 con l'app (CPU f64, palette
  'termal') → `icon_src.png` (per il `.icns` su macOS via sips/iconutil) +
  `mandelbrot.ico` multi-size 16–256 (Pillow).
- **Versione unica di verità**: `VERSIONE` nell'header di `mandel.py`; la spec
  PyInstaller la legge da lì (consistente tra titolo EXE, plist del `.app` e
  `VERSION` del sorgente).
- **Ordine** (obbligatorio, vedi AGENTS.md): modifica sorgente → `build_app.py`
  → verifica → commit. `dist/`, `build/`, icone ed EXE sono gitignorati (l'app
  NON va committata, ma va rigenerata prima del commit per lasciarla aggiornata).

## Note tecniche (imparati a caro prezzo)
- Le LUT **devono** essere `__constant__` **incorporate nel kernel** (una per palette):
  passate come buffer device, CuPy le sovrascriveva con l'output a ogni launch.
- Argomenti scalari CuPy = **array numpy size-1**; indice palette preallocato (`_PAL_IDX`).
- `np.asarray(array_cupy)` **non** è permesso → `.get()`.
- NVRTC: niente `-O3`/`--opt-level` (la compilazione è lazy alla prima chiamata).
- **CuPy 14**: `cp.cuda.MemoryHost` e `Event.elapsed_time` non esistono più → memoria
  host pinned con `cp.cuda.PinnedMemory(size)` (+`.ptr`) e `cp.cuda.runtime.memcpy(dst,
  src, n, kind)`; timing con `perf_counter`+sync.
- **Metal / pyobjc (scoperti a caro prezzo)**:
  - gli argomenti **scalarmi** vanno in uno `struct` passato via buffer
    `[[buffer(1)]]` (né scalari nudi né `[[buffer]]` su singoli `float` sono accettati
    in MSL);
  - MSL usa `fmin/fmax/pow/sqrt/log2/log` (senza suffisso `f`) e `half` è un tipo
    riservato → il campo dello struct si chiama `hs`, non `half`;
  - i puntatori richiedono l'address space esplicito (`device`/`constant`);
  - H2D/D2H: `buf.contents().as_buffer(N)` restituisce una **memoryview scrivibile**
    (C-level, ~0.01 ms) — è il ponte rapido (la varlist NON è bytes-like);
  - `setBuffer:offset:atIndex:` ha firma `(buffer, **offset**, **INDEX**)` →
    `(pbuf, 0, 1)`, non `(pbuf, 1, 0)`;
   - su Apple Silicon **`double` non è supportato** → il kernel è f32-only;
   - il compute è sincrono (`commit` + `waitUntilCompleted` + read); un `threading.Lock`
     serializza le chiamate; 2 run sono **bit-identici** (deterministico).
 - **Vulkan / wgpu (scoperti a caro prezzo)**:
   - l'API wgpu 0.32 usa nomi "WebGPU-style" omonimi: `wgpu.gpu.enumerate_adapters_sync()`,
     `wgpu.gpu.request_adapter_sync(power_preference=...)`, `adapter.request_device_sync()`,
     `device.create_shader_module(code=...)`, `device.create_compute_pipeline(layout="auto",
     compute=wgpu.ProgrammableStage(module=..., entry_point="main"))`;
   - **buffer**: `create_buffer(size=, usage=)` / `create_buffer_with_data(data=, usage=)`
     (argomenti keyword, NON un descrittore); `create_bind_group(layout=, entries=
     [wgpu.BindGroupEntry(binding=N, resource=buf)])`; layout via
     `pipe.get_bind_group_layout(0)`;
   - **niente tipo `u8`** in WGSL → LUT e output sono `array<u32>` packed `0x00RRGGBB`
     (1 px = 1 u32), unpackati in numpy `(h,w,3)` al readback; parametri in due buffer
     uniform (`vec4<f32>` + `vec4<i32>`);
   - **H2D/D2H**: `queue.write_buffer` richiede `COPY_DST` sul buffer di destinazione
     (usare `UNIFORM|COPY_DST`); `queue.read_buffer` richiede `COPY_SRC` sul buffer
     sorgente (usare `STORAGE|COPY_SRC`); `queue.submit([enc.finish()])`;
   - built-in WGSL `min/max/pow/sqrt/log/log2` (senza suffisso `f`); `var` per
     variabili mutabili, `let` per costanti; `break` deve essere l'ultima istruzione
     del blocco;
   - il compute è sincrono (`submit` + `read_buffer` blocca); un `threading.Lock`
     serializza le chiamate; 2 run sono **bit-identici** (deterministico, verificato
     su AMD 780M); la lib nativa `wgpu/resources/*.dll` è nella wheel (bundled via
     `collect_all(wgpu)` + hook `hook-wgpu.py`).
- **FMA numpy (critica per la bit-identità)**: `np.square` su array
  complessi è compilato da numpy 2.4 **con FMA** (`re*re - im*im` contratto in
  `fma(re, re, -(im*im))`, dipende dal build SIMD), mentre Numba senza fastmath NON
  contrae: 1 ULP di differenza su orbite caotiche → escape time diverso su una
   piccolissima frazione di pixel di bordo (0.01–0.8% a seconda dello zoom). Regola:
   in tutto il codice CPU il quadrato complesso va fatto in parti esplicite
   (`a*a-b*b`, `2*a*b` con ufunc singoli) → in **f64** il percorso Numba e il
    fallback numpy restano bit-identici tra loro. **Eccezione f32**: in f32
   Numba/LLVM contrae in FMA / riassocia in modo *dipendente dal contesto del
   loop* (intrinseco, non riproducibile in modo affidabile con numpy) → Numba-f32
   e numpy-f32 differiscono di 1-qualche ULP su una piccolissima frazione di pixel
   di bordo caotico (`it[]` sempre uguale; al più 1 entry di LUT). Per questo il
    self-test f32 usa `it[]` esatto + `mag[]` entro tolleranza, e il gate confronta
     la CPU-f32 col *suo* riferimento (entrambi Numba → bit-identici). Gate
    (portatile per piattaforma): CPU f64/f32 **bit-identiche** ai riferimenti
    (`<zona>_cpu.npy` = f64, `<zona>_cpu_f32.npy` = f32); **se CUDA**: CUDA f32/f64
    bit-id a `<zona>_gpu_f32/_gpu_f64.npy` + CPUf64~CUDAf64 ≤2% pixel >8 (bordo
    caotico + ULP libm-vs-CUDA); **se Metal**: Metal f32 bit-id a
    `<zona>_metal_f32.npy` + determinismo (2 run bit-identici) + Metal~CPUf32 entro la
    varianza f32 intrinseca, stimata `max(2%, 1.5× CPUf32~CUDAf32)` usando il gold
    CUDA f32 come righello della "nube" f32 — a deep-zoom le implementazioni f32
    (Numba/CUDA/Metal/Vulkan) divergono tra loro per il 10-25% dei pixel di bordo
    caotico per FMA/contrazione: NON è un difetto (Metal/Vulkan f32 sono all'altezza
    di CUDA f32 e CPU f32).
- **Associatività IEEE (bit-identità)**: riscrivere un'espressione numpy può
  cambiare gli arrotondamenti — `cx + half*X/s` NON è bit-identico a
  `cx + half*(X/s)` (l'ordine di `*` e `/` conta). Per restare bit-identici
  conservare l'ordine originale delle operazioni e verificare con il gate
  multi-zona (poteva passare in una zona e rompersi in un'altra).
- **Modello di memoria Numba**: dentro `prange` la lettura di una
  memoria scritta da un altro thread **non è affidabile**: Numba/LLVM fa hoisting del
  load fuori dal loop (semantica single-thread), quindi un bump cross-thread non è
  visto e il lavoro viene comunque eseguito (verificato sperimentalmente). La
  cancellazione cooperativa va fatta a livello Python (tra bande/segmenti), non
  dentro il kernel.
  Altri vincoli Numba: `break`/`continue` nel corpo di `prange` impediscono la
  parallelizzazione; `cache=True` NON è usato (mandel.py cambia nome di modulo a ogni
  load → cache inutilizzabile, la compilazione è pagata dal warmup thread all'avvio);
  nessun ufunc `fma` disponibile in Numba.
- Tkinter: `delete`/`insert` su `Entry` disabilitato sono **no-op** → prima `state="normal"`.
- Note operative (versionamento, ambiente GPU condiviso, benchmarking): vedi **AGENTS.md**.
