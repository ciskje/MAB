# Visualizzatore Mandelbrot — spec di ricreazione

Descrizione concisa ma sufficiente perché un altro LLM (o sviluppatore) ricrei il
 programma da zero. Riferimento: `mandel.py` (un solo file).
 Ogni modifica al sorgente DEVE aggiornare anche questa spec (vedi AGENTS.md).

Struttura in **ordine di costruzione**: stack → matematica → palette → MI →
 backend → implementazione per backend → pipeline → UI → file → benchmark → build → gotchas.

## 1. Stack e architettura
- Un solo file **Python 3.14**: GUI **tkinter**, rendering su uno di **4 backend
  selezionabili** — **CPU (numpy/Numba)**, **CUDA (CuPy `RawKernel`)** su NVIDIA,
  **Metal (pyobjc)** su Apple Silicon, **Vulkan (wgpu/wgpu-native)** cross-platform
  (AMD/NVIDIA/Intel) — con fallback **CPU (numpy/Numba)**; **Pillow** per il PNG.
  I backend non disponibili restano visibili in toolbar ma disabilitati.
- Esecuzione asincrona: il render gira su thread separato, l'UI non si blocca mai.
- Metodi di `MandelbrotApp` raggruppati per funzione: UI, vista, controlli, pipeline, file, benchmark.
- **Versione** in due punti che devono coincidere: l'header `# VERSIONE: X.Y.Z` nel
  commento iniziale (letto per EXE/.app/zip) e la costante runtime `VERSION = "X.Y.Z"`
  (titolo finestra + JSON zona). Regole di incremento/STORICO: vedi AGENTS.md.
- **Dipendenze**: numpy (sempre); Numba (opzionale, altrimenti fallback numpy);
  CuPy (CUDA); pyobjc (Metal); wgpu (Vulkan); Pillow (PNG, icona).

## 2. Matematica del rendering
- `z = z² + c`, escape `|z|² > 4`; punti interni = **nero** (early-exit nel loop).
- **Interior analitico** (GPU e CPU, prima del loop): bulbo periodica-2 `|c+1| ≤ 0.25`
  (nel kernel: `d2 = (x0+1)² + y0²`, subito nero se `d2 ≤ 0.0625`) e cardioide
  principale `|1 − √(1−4c)| < 1` (riscritta senza complessi: `R < 2·√(0.5·(R+A))`,
  `A = 1−4·Re c`, `R = |1−4c|`, con prefiltro bounding-box) → subito neri, saltano
  le `mi` iterazioni. Bit-identico al kernel senza test.
- **GPU**: iterazione in coordinata spostata `w = z − cx` (solo parte reale, stabilità
  numerica).
- **Coloring continuo** (stesso per CPU e GPU): `nu = it + 1 − log2(0.5·ln|z|²)`,
  `t = (nu/mi)^0.35`. Il kernel esporta `it[]` e, alla fuga, `mag = |z|²` (usato dal
  coloring); solo i pixel mai fuggiti (`mag == 0`) restano neri.
- **LUT 256×3 condivisa CPU/GPU**: `np.interp` delle stop su 256 punti, ×255, clip,
  uint8; colore = `LUT[round(t·255)]`. CPU e GPU producono lo stesso colore (a parte
  1-2 ULP di `log2`/`log` libm-vs-kernel → ≤1 entry di LUT; verificato dal gate).
  Le LUT sono **incorporate nel kernel** (una `constant` per palette), NON passate
  come buffer device (vedi gotchas §12).

## 3. Palette
- **Registro ordinato** `{nome: (t, R, G, B)}` — l'ordine = indice passato al kernel
  (0=fuoco, 1=ghiaccio, 2=termal). UI, config e LUT `__constant__` del kernel sono
  tutte generate da questo registro. **Default: fuoco**.
- `make_lut(pal)`: `t2 = np.linspace(0, 1, 256)`, `np.interp` per canale →
  `(rgb*255).clip(0,255).astype(uint8)`.
- **fuoco**: `t=(0,.2,.45,.7,.9,1)` `R=(.05,.35,.85,1,1,1)` `G=(0,.02,.2,.65,.95,1)` `B=(0,0,.02,.15,.55,1)`
- **ghiaccio**: `t=(0,.25,.5,.75,1)` `R=(.02,.05,.30,.70,1)` `G=(.02,.15,.55,.85,1)` `B=(.10,.45,.90,1,1)`
- **termal** (ghiaccio→fuoco): `t=(0,.2,.4,.55,.7,.85,1)` `R=(.02,.10,.55,.95,1,1,1)` `G=(.08,.45,.80,.96,.85,.55,.30)` `B=(.28,.85,.95,.98,.45,.20,.10)`

## 4. Iterazioni (MI)
- **Auto** (default): `auto_mi(half) = 2000·(1 + log10(1.5/half))`, clamp `[50, 50000]`
  — funzione di modulo unica, **condivisa col benchmark**. A vista iniziale ~2000;
  nella zona del becco (`half~5.2e-5`) ~10 916.
- **Manual**: campo editabile (commit su Invio/FocusOut; validazione intero **50–100000**,
  valore non valido → messaggio nella barra di stato + ripristino del valore precedente)
  e pulsanti `±1000` (disabilitati in auto). In auto il campo è disabilitato ma mostra il
  valore corrente (etichetta `Iterazioni (auto):`); disattivando l'auto, `mi` si **congela
  sul valore auto corrente**.

## 5. Backend e precisione
- **CPU**: numpy/Numba, **f32** o **f64**. Sempre disponibile.
- **Selezione 4 backend**: la toolbar "Motore" ha 4 **radio** — `CPU` / `CUDA` /
  `Metal` / `Vulkan` (stesso frame = gruppo radio unico). I backend **non disponibili
  restano visibili ma disabilitati** (grigi).
- **Rilevamento all'avvio**: `_CUDA_OK`/`_METAL_OK`/`_VULKAN_OK` per ciascuno;
  `_BACKENDS_OK` = elenco di quelli disponibili in ordine di preferenza
  (CUDA > Metal > Vulkan > CPU); `_ACTIVE` = backend corrente (stringa
  `cpu|cuda|metal|vulkan`); il **default** è il primo GPU disponibile, altrimenti CPU
  (`_default_backend`). `compute` dispatcha su `_ACTIVE`; `compute_gpu` è un dispatcher
  (`_compute_gpu_cuda` / `_compute_gpu_metal` / `_compute_gpu_vulkan`).
- **`backend()`** mostra "CUDA f32|f64" / "METAL f32" / "VULKAN f32" /
  "CPU f32|f64" (nomi GPU upper-case, CPU no; senza Numba la CPU mostra
  "CPU f32|f64 (numpy)").
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
- **Nota (misurata)**: f32 non accelera il loop di escape (latency-bound, catena seriale);
  guadagno solo sui passi memory-bound (geometria/prefiltro/coloring): ~1.6–1.7× esterno,
  ~1.07× deep zoom.

## 6. Implementazione per backend

Comune a tutti i backend: matematica §2 (escape, interior analitico, coloring smooth — le 4
vie producono lo stesso colore a parte 1-qualche ULP di FMA sul bordo caotico), 1 px/thread,
block/workgroup/threadgroup 16×16, LUT incorporate come costanti (§12). Di seguito solo scarti
+ gotcha API per linguaggio.

### 6.1 CPU (numpy/Numba)
- **Array di lavoro cacheati** per `(w,h,prec)`: offset di griglia + real/imag/c/w4 +
  work arrays riutilizzati (max 6 dimensioni) → nessuna allocazione per render.
- **Escape loop** in **Numba** `@njit(parallel=True)` (dipendenza opzionale): parallelo
  sulle righe (`prange`), early-exit per pixel. Geometry, interior analitico e coloring
  restano in numpy; il kernel Numba produce `it[]` e, alla fuga, `mag[] = |z|²` (usato
  dal coloring smooth).
- **Numba assente**: la CPU gira sul **fallback numpy single-core** e `backend()` mostra
  `CPU f32/f64 (numpy)` in titolo + barra di stato; flag `_NUMBA_AVAILABLE = (njit is not None)`.
- **f32/f64** (§5): tutti gli array di lavoro nel dtype corrispondente (X/Y restano f64:
  il prodotto con `half` si calcola in f64 e si arrotonda al target); il kernel Numba si
  **auto-specializza** su `complex64`/`complex128`; il fallback numpy usa la stessa sequenza
  di ufunc in parti esplicite (no FMA) anche in f32.
- **Self-test** all'avvio (thread, 4×4 con casi limite, vs il loop numpy di riferimento),
  **una precisione per volta** → se fallisce (o Numba assente) per una precisione,
  **fallback numpy automatico per quella precisione** (stesso loop, stesso gate).
  Criterio: **f64 = bit-identità** di `it[]` **e** `mag[]`; **f32 = `it[]` esatto +
  `mag[]` entro 1e-2 relativo** (FMA Numba-f32 contesto-dipendente, vedi §12).
- **Warmup/compilazione Numba** di entrambe le precisioni in thread all'avvio (fuori dal
  primo render).
- **Cancellazione cooperativa**: vedi §7.
- **Attenzione FMA**: quadrato complesso in parti esplicite (`a*a-b*b`, `2*a*b`, ufunc
  singoli), NON `np.square` — vedi §12.

### 6.2 CUDA (CuPy `RawKernel`)
- **Kernel**: 1 px/thread (micro-benchmark A/B: 1.8× su z1 / 1.1× su z3 vs 2 px/thread;
  le varianti sono bit-identiche tra loro), `__launch_bounds__(256)`, block 16×16,
  grid `(⌈w/16⌉, ⌈h/16⌉)`, `--use_fast_math`; 2 varianti (f32/f64) dalla stessa sorgente
  parametrizzata; **NVRTC lazy** (compilazione alla prima chiamata, no `-O3`/`--opt-level`).
- **Parametri scalari** = **array numpy size-1**; indice palette preallocato (`_PAL_IDX`).
- **LUT** `__constant__` (una per palette), non passate come buffer device — vedi §12.
- **D2H pinned**: buffer host pinned cacheato per dimensione
  (`cp.cuda.PinnedMemory(size)` + `cp.cuda.runtime.memcpy(dst, src, n, memcpyDeviceToHost)`,
  DMA ~1,7× del `.get()` pageable), con fallback `.get()`. Vista numpy: `np.frombuffer`
  su `ctypes` array all'indirizzo `pm.ptr`.
- `np.asarray(array_cupy)` **non** è permesso → `.get()`.
- **CuPy 14**: `cp.cuda.MemoryHost` e `Event.elapsed_time` non esistono più → memoria
  host pinned con `cp.cuda.PinnedMemory(size)` (+`.ptr`) e `cp.cuda.runtime.memcpy(dst,
  src, n, kind)`; timing con `perf_counter` + sync.
- **Warmup GPU all'avvio**: thread daemon che all'import fa un render 64×64 (f32 e f64) →
  init context CUDA, module load e prime allocazioni (device + pinned) pagati FUORI dal
  primo render reale.

### 6.3 Metal (pyobjc, Apple Silicon)
- **Kernel MSL** `mandel`: threadgroup 16×16 (256, sotto il max 1024 di M1),
  grid `(⌈w/16⌉, ⌈h/16⌉)`; f32-only (vedi §5).
- **Parametri** `(cx,cy,half,w,h,mi,pal)` in uno **struct** via buffer `[[buffer(1)]]`
  (né scalari nudi né `[[buffer]]` su singoli float); output RGB888 in `[[buffer(0)]]`;
  LUT `constant` (una per palette, §12).
- **H2D/D2H**: `buf.contents().as_buffer(N)` (memoryview scrivibile, C-level, ~0.01 ms) —
  è il ponte rapido (la varlist NON è bytes-like).
- **Deterministico** (2 run bit-identici). 12–30× più veloce della CPU (misurato su M1).
- **Gotchas MSL/pyobjc**: `fmin/fmax/pow/sqrt/log2/log` (senza suffisso `f`); `half` è un
  tipo riservato → il campo dello struct si chiama `hs`, non `half`; i puntatori
  richiedono l'address space esplicito (`device`/`constant`); `setBuffer:offset:atIndex:`
  ha firma `(buffer, **offset**, **INDEX**)` → `(pbuf, 0, 1)`, non `(pbuf, 1, 0)`; il
  compute è sincrono (`commit` + `waitUntilCompleted` + read); un `threading.Lock`
  serializza le chiamate.

### 6.4 Vulkan (wgpu/wgpu-native, cross-platform)
- **Compute shader WGSL** `main`: workgroup 16×16 (256, sotto il max AMD/NVIDIA/Intel),
  dispatch `(⌈w/16⌉, ⌈h/16⌉)`; f32-only (vedi §5).
- **Vincoli WGSL** (vs MSL/CUDA): nessun tipo `u8` → **LUT e output sono `array<u32>`
  con colori packed `0x00RRGGBB`** (1 px = 1 `u32`); al readback si unpacka in numpy
  `(h,w,3)` (shift R/G/B). Parametri in **due buffer uniform**: `vec4<f32>` (cx,cy,half) +
  `vec4<i32>` (w,h,mi,pal); LUT in un buffer `storage` con tutte le palette concatenate
  (selezione `lut[pal*256+idx]`, come CUDA/MSL); built-in `min/max/pow/sqrt/log/log2`
  (senza suffisso `f`); `var` per variabili mutabili, `let` per costanti; `break` deve
  essere l'ultima istruzione del blocco.
- **Deterministico** (2 run bit-identici, verificato su AMD 780M).
- **H2D**: `queue.write_buffer` su buffer persistenti (uso `UNIFORM|COPY_DST`); **D2H**:
  `queue.read_buffer` su buffer `STORAGE|COPY_SRC`; `queue.submit([enc.finish()])`; un
  `threading.Lock` serializza (compute sincrono: `submit` + `read_buffer` blocca).
- Funziona **out-of-the-box** su AMD/NVIDIA/Intel (lib nativa nella wheel, no runtime
  esterno).
- **Gotchas API wgpu 0.32** (nomi "WebGPU-style"): `wgpu.gpu.enumerate_adapters_sync()`,
  `wgpu.gpu.request_adapter_sync(power_preference=...)`, `adapter.request_device_sync()`,
  `device.create_shader_module(code=...)`, `device.create_compute_pipeline(layout="auto",
  compute=wgpu.ProgrammableStage(module=..., entry_point="main"))`; buffer:
  `create_buffer(size=, usage=)` / `create_buffer_with_data(data=, usage=)` (argomenti
  keyword, NON un descrittore); `create_bind_group(layout=, entries=
  [wgpu.BindGroupEntry(binding=N, resource=buf)])`; layout via `pipe.get_bind_group_layout(0)`;
  la lib nativa `wgpu/resources/*.dll` è nella wheel (bundled via `collect_all(wgpu)` +
  hook `hook-wgpu.py`).

## 7. Pipeline (asincrona, latest-wins)
- **Thread worker** (daemon) + `threading.Condition` + **slot job singolo**: i request in
  coda collassano sull'ultimo (*latest-wins*).
- **Cancellazione cooperativa**: `_submit` incrementa `_GEN` (list `[int]`, atomoico); il
  render CPU in corso si ferma se la generazione cambia (Numba: ogni **banda di 16 righe**,
  check a livello **Python** tra le bande — un check in-kernel NON è affidabile, vedi §12;
  fallback: iterazione per iterazione) e il worker scarta il frame obsoleto prima di
  `_show`. **GPU: non cancellabile** (render <160 ms), solo scarto post-render.
  `my_gen=0` disattiva la cancellazione (benchmark/CLI).
- **`request_render`**: **preview ¼** (min 16 px) immediata, poi render full dopo **500 ms**
  (solo se la vista non è cambiata).
- **UI**: poll ogni **30 ms** (`tkinter.after`) che mostra l'ultimo frame (ridimensionato al
  canvas).

## 8. UI e interazione
- **Vista iniziale**: centro `(-0.5, 0)`, `half = 1.5`, canvas 960×540; asse y scalato di
  `h/w`; clamp `half ≥ 1e-12` (`MIN_HALF`, evita `half=0`).
- **Zoom**: rotella ×1.25/×0.8 al cursore; click ×2 al cursore; `+`/`-` ×2/×0.5 al centro.
  **Pan**: trascinamento.
- `R` = reset (vista + tutti i settaggi ai default); `Ctrl+S` = salva PNG.
- **Layout**: toolbar (Motore / Palette / Precisione / Iter), riga pulsanti (campo
  iterazioni + `−1000`/`+1000`, Benchmark, Reset), canvas, barra di stato
  `messaggio | backend · hardware | palette | render: N ms`. Font 13 pt.
  Menu File: Salva immagine…, Carica zona…, Salva zona, Salva zona con nome…, Esci.
- **Titolo**: `Insieme di Mandelbrot v<VER> - <backend> (<hardware>)` +
  ` - <file zona corrente>` se presente.
- **Nome hardware** (`hw_name()`, in cache per backend): CPU dal registro Windows
  (`ProcessorNameString`) o `sysctl machdep.cpu.brand_string` su macOS; GPU da CuPy
  (`runtime.getDeviceProperties(0)["name"]`, il device 0 CUDA che CuPy usa di default —
  con più GPU l'ordine CUDA può differire da nvidia-smi), pyobjc (`MTLDevice.name`) o wgpu
  (`adapter.info["device"]`, preferenza high-performance); nessuna dipendenza nuova, in caso
  di errore fallback generico "CPU"/"GPU".
- **Cursore occupato**: durante il benchmark il cursore della finestra principale passa a
  `watch` (sabbia su Windows / pallina su macOS = "occupato", ereditato dai figli
  canvas/pulsanti) e torna alla freccia a terminazione. Impostato in `run_benchmark()`,
  ripristinato in `_bench_done()` (unico punto di uscita, anche su errore).
- **Tema UI**: **widget nativi macOS** (nessun colore forzato), seguono il tema del
  **sistema** (in Dark Mode restano aqua scuri; forzare un `background` esplicito li
  degradava a widget *flat* illeggibili). Dialog benchmark: **colore testo di sistema**
  (dinamico); accenti OK/KO fissi in toni medi leggibili su entrambi (`#2ea44f`/`#e5534b`),
  divisore grigio medio (`#8a8a8a`). Solo colori UI, nessun effetto sul rendering.

## 9. File I/O
- **Zona** (JSON della vista, indentato): `{"app", "versione", "cx", "cy", "half", "mi", "mi_auto"}`.
  - **"Salva zona"** è DISABILITATO finché non esiste un file corrente (`view_file=None`; all'avvio
    si parte SENZA file corrente, non ripristinato dalla config) e riscrive il file corrente
    (`view_file`). Si abilita dopo **"Salva zona con nome…"** (default
    `mandelbrot_<AAAAmmgg_HHMMSS>.json`) o **"Carica zona…"** (ripristina vista + MI, clampa
    `half ≥ 1e-12`), che lo rendono file corrente (mostrato nel titolo).
- **Config**: `~/mandelbrot/config.json` con `precision, palette, backend, bench` — `backend`
  = il backend **attivo** per nome ∈ {`"cpu"`, `"cuda"`, `"metal"`, `"vulkan"`} (i vecchi
  valori `"gpu"`/`"cuda"` sono migrati al default GPU / a `"cuda"` alla lettura, e un backend
  non disponibile resta sul default di avvio) — (la **vista** `cx, cy, half, mi, mi_auto` e
  `view_file` non sono persistite: eventuali vecchi valori in config esistenti sono ignorati);
  salvata all'uscita e **throttled ~1 s** sui cambiamenti; reset riporta i default.
- **Avvio**: il programma parte SEMPRE con la configurazione di default — vista sull'intero
  insieme di Mandelbrot (`cx=-0.5, cy=0, half=1.5`) + MI auto — come la prima volta; la vista
  precedente si recupera solo con "Carica zona…".

## 10. Benchmark
- **Dialog di conferma** custom (`Toplevel` modale centrato, griglia parametri, Avvia/Annulla,
  `Return`/`Esc`); poi thread dedicato per la durata (default 8 s).
- **Cursore occupato**: vedi §8.
- **Default**: `c = (-0.7499302568795561, -0.015139113925433963i)`,
  `half = 5.226737155905588e-05`, 960×540, `secs = 8.0`.
- **`mi` non è un parametro**: sempre `auto_mi(bench['half'])` (~10 915 a default) →
  benchmark comparabile anche se la formula auto cambia.
- Parametri in `config.json` (chiave `bench`, overridibile; una vecchia `bench.mi` è ignorata).
- **Esegue nella modalità corrente dell'app**: motore (CPU/CUDA/Metal/Vulkan) + precisione
  f32/f64 come selezionati in toolbar; su CUDA usa un buffer proprio (no contesa col render
  normale), Metal e Vulkan usano il proprio buffer di output, CPU la memoria numpy. Per
  confrontare versioni, usarlo nella stessa modalità.
- **Report**: dialog con **rendering/s in grande** (il vero risultato, ~42pt verde),
  statistiche (n. rendering, ms/render), **grafico a barre orizzontali** e griglia dei
  parametri; errore → "BENCHMARK FALLITO" + dettaglio.
- **Grafico a barre** (`tk.Canvas`, no dipendenza nuova): i 3 riferimenti storici in `BENCH_REF`
  (6.62 9900X CPU, 177 5070 Ti Vulkan, 250 5070 Ti CUDA, rendering/s) grigie col proprio nome
  hardware; la run corrente verde riporta in etichetta **solo il metodo attivo** (es. "CUDA f32
  — questa run"), nome hardware nella griglia parametri (riga Hardware, via `hw_name()`).
  Scala lineare automatica 0→max, step "nice" (1/2/5×10ⁿ) + griglia verticale, valore a fine barra.

## 11. Build dell'app (multipiattaforma)
- **one-dir self-contained**, no dipendenze per l'utente: CPU sempre (Numba/numpy bundled);
  **GPU Vulkan (wgpu) bundled out-of-the-box** (lib nativa nella wheel, AMD/NVIDIA/Intel);
  **GPU CUDA** richiede solo driver + runtime NVIDIA installati dall'utente.
  - **Windows** → `dist/Mandelbrot/Mandelbrot.exe` + `_internal/` + zip
    `dist/Mandelbrot-v<ver>-win64.zip`. CuPy incluso ma **runtime CUDA NON bundled**: CuPy lo
    trova a runtime via cuda-pathfinder da `CUDA_PATH`/PATH/Program Files. Senza GPU degrada su CPU.
  - **macOS** → `dist/Mandelbrot.app` (GPU Metal/pyobjc + Vulkan/wgpu, firma ad-hoc).
- **Script**: `build_app.py` (unico script, ramificato su `sys.platform`: icona → PyInstaller
  → post → zip su Windows). `build_app.sh` lo redirect (compatibilità); su Windows si usa
  `build_app.ps1` (o `python build_app.py`).
- **Ricetta**: `mandelbrot.spec` (PyInstaller), multipiattaforma:
  - comune: `collect_all(numba/llvmlite)` (dylib/dll di llvmlite + dati JIT),
    `PIL._tkinter_finder`, esclude `torch/matplotlib/IPython/pytest`;
  - **win32**: `collect_all(cupy)` (solo moduli, **NO DLL CUDA**: il runtime NVIDIA non è
    bundle, l'utente lo installa e CuPy lo trova via cuda-pathfinder), `collect_all(wgpu/cffi)`
    (GPU Vulkan, lib nativa `wgpu/resources/*.dll` bundled + hook `hook-wgpu.py` di wgpu),
    icona `mandelbrot.ico`, versione EXE (versioninfo), runtime hook `hook_dlldir.py`
    (`os.add_dll_directory` su cartella EXE + `_internal`, difensivo);
  - **darwin**: `collect_all(objc/Foundation/Metal)` + submoduli, `collect_all(wgpu/cffi)`
    (GPU Vulkan), `libomp.dylib` da `torch/lib` (il rpath di `omppool` è hardcoded a una dir
    CI inesistente), `BUNDLE` → `.app` (icon `.icns`, bundle_id, plist).
- **Percorsi build**: workpath intermedio `build/` E output app `dist/` (PyInstaller) vanno
  sul **disco di sistema** (temp utente, `tempfile.gettempdir()/mandelbrot_{build,dist}` via
  `--workpath`+`--distpath` in `build_app.py`) e NON nel progetto (che può stare su una share
  di rete, dove la build è lenta). Sul progetto/NAS resta **solo l'artefatto distributivo**:
  su Windows la zip `dist/Mandelbrot-v<ver>-win64.zip` (generata da C:), su macOS il
  `dist/Mandelbrot-v<ver>-macos.dmg` (staging app + link `/Applications` in temp, `hdiutil UDZO`).
- **Icona**: `make_icon.py` renderizza 1024×1024 con l'app (CPU f64, palette 'termal') →
  `icon_src.png` (per il `.icns` su macOS via sips/iconutil) + `mandelbrot.ico` multi-size
  16–256 (Pillow).
- **Versione** letta dall'header di `mandel.py` (consistente tra EXE/.app/zip e sorgente, vedi
  §1). **Regole di processo** (ordine build, gitignore, rigenerazione pre-commit):
  vedi AGENTS.md.

## 12. Note tecniche (gotchas, imparati a caro prezzo)
- **LUT come `__constant__` incorporate nel kernel** (una per palette): passate come buffer
  device, CuPy le sovrascriveva con l'output a ogni launch.
- **FMA numpy (critica per la bit-identità)**: `np.square` su array complessi è compilato da
  numpy 2.4 **con FMA** (dipende dal build SIMD), Numba senza fastmath NON contrae → 1 ULP su
  orbite caotiche, escape time diverso su una piccolissima frazione di bordo. Regola: quadrato
  complesso in parti esplicite (`a*a-b*b`, `2*a*b`, ufunc singoli) → in **f64** Numba e numpy
  bit-identici. **Eccezione f32**: Numba/LLVM contrae in FMA *contesto-dipendente* (intrinseco,
  non riproducibile con numpy) → Numba-f32 e numpy-f32 differiscono di 1-qualche ULP sul bordo
  caotico (`it[]` sempre uguale, al più 1 entry LUT); per questo self-test f32 usa `it[]` esatto
  + `mag[]` in tolleranza e il gate confronta CPU-f32 col suo riferimento Numba (bit-identici).
- **Associatività IEEE (bit-identità)**: riscrivere un'espressione numpy può cambiare gli
  arrotondamenti — `cx + half*X/s` NON è bit-identico a `cx + half*(X/s)` (l'ordine di `*` e
  `/` conta). Per restare bit-identici conservare l'ordine originale delle operazioni e
  verificare con il gate multi-zona (poteva passare in una zona e rompersi in un'altra).
- **Modello di memoria Numba**: dentro `prange` la lettura di memoria scritta da un altro thread
  **non è affidabile** (hoisting del load fuori dal loop, semantica single-thread) → un bump
  cross-thread non è visto; la cancellazione va fatta a livello Python (tra bande, vedi §7).
  Altri vincoli: `break`/`continue` nel corpo di `prange` impediscono la parallelizzazione;
  `cache=True` inutilizzabile (mandel.py cambia nome modulo a ogni load, compilazione pagata
  dal warmup); nessun ufunc `fma` in Numba.
- **Gate (portatile per piattaforma)**: CPU f64/f32 **bit-identiche** ai riferimenti
  (`<zona>_cpu.npy` = f64, `<zona>_cpu_f32.npy` = f32); **se CUDA**: CUDA f32/f64 bit-id a
  `<zona>_gpu_f32/_gpu_f64.npy` + CPUf64~CUDAf64 ≤2% pixel >8 (bordo caotico + ULP
  libm-vs-CUDA); **se Metal**: Metal f32 bit-id a `<zona>_metal_f32.npy` + determinismo (2 run
  bit-identici) + Metal~CPUf32 entro la varianza f32 intrinseca, stimata
  `max(2%, 1.5× CPUf32~CUDAf32)` usando il gold CUDA f32 come righello della "nube" f32 — a
  deep-zoom le implementazioni f32 (Numba/CUDA/Metal/Vulkan) divergono tra loro per il 10-25%
  dei pixel di bordo caotico per FMA/contrazione: NON è un difetto (Metal/Vulkan f32 sono
  all'altezza di CUDA f32 e CPU f32).
- **Tkinter**: `delete`/`insert` su `Entry` disabilitato sono **no-op** → prima `state="normal"`.
- **Note operative** (versionamento, ambiente GPU condiviso, benchmarking): vedi **AGENTS.md**.
