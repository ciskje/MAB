# Convenzioni del progetto

## Versionamento sorgente (OBBLIGATORIO)
Ogni modifica a `mandel.py` (o ad altri sorgenti Python del
progetto) DEVE:
1. Incrementare la versione nel blocco di commento iniziale del sorgente
    (`mandel.py`).
    - bugfix = patch (x.y.Z), modifica minima (es. un dialog in piu, un testo,
      un opzione) = patch (x.y.Z), modifica funzionale = minor (x.Y.0),
      riscrittura/architettura = major (X.0.0)
2. Aggiungere una voce allo STORICO nello stesso blocco, formato:
   `VERSIONE - AAAA-MM-GG - descrizione delle modifiche`.
3. Aggiornare `spec.md` in modo che rispecchi il nuovo stato del programma
   (la spec DEVE restare sempre in sincronia col sorgente).
Il blocco è intitolato "Insieme di Mandelbrot - visualizzatore interattivo"
e si trova in cima al file.

## Note tecniche (imparate a caro prezzo)
- PowerShell 5.1: NON usare `Start-Process` con `-RedirectStandardOutput`
  verso path UNC (va in timeout). Usare `Start-Process -PassThru`.
- Tkinter: `tk.Checkbutton` (ortografia americana), mai `tk.CheckButton`.
- Tkinter Entry: `delete`/`insert` su un Entry con `state="disabled"` sono
  **no-op silenziosi**. Per aggiornarne il testo: prima `state="normal"`,
  poi `delete`/`insert`, poi (se serve) `state="disabled"`. Attenzione che il
  widget arriva già disabilitato dalla chiamata precedente (vedi v4.13.3).
- CuPy: `np.asarray(array_cupy)` NON è permesso (TypeError); usare `.get()`.
- CuPy RawKernel: NON passare la palette LUT come argomento array device
  (l'output del kernel la sovrascriveva a ogni launch). Le palette sono
  incorporate nel kernel come `__constant__` (vedi v4.0.0).
- Il tool `write` NON funziona su path UNC; usare `edit` (che funziona).
- NVRTC: niente flag `-O3`/`--opt-level`; la compilazione è lazy alla
  prima chiamata del kernel.
- Argomenti scalari dei RawKernel: array numpy size-1, non scalari Python.
- CuPy 14: `cp.cuda.MemoryHost` e `Event.elapsed_time` sono stati rimossi. Memoria
  host pinned: `cp.cuda.PinnedMemory(size)` (+ `.ptr`), copia D2H con
  `cp.cuda.runtime.memcpy(dst, src, n, cp.cuda.runtime.memcpyDeviceToHost)`; la vista
  numpy si fa con `np.frombuffer((ctypes.c_ubyte*n).from_address(ptr), dtype=np.uint8)`.
  Timing CUDA: `perf_counter` + `Stream.null.synchronize()`.
- Attenzione all'associatività IEEE quando si riscrivono espressioni numpy:
  `cx + half*X/s` NON è bit-identico a `cx + half*(X/s)` (l'ordine di `*` e `/` cambia
  gli arrotondamenti). Per restare bit-identici conservare l'ordine originale e
  verificare con il gate multi-zona (poteva passare in una zona e rompersi in un'altra).
- **FMA di numpy (critico, scoperto in v5.0.0)**: `np.square` (e gli ufunc aritmetici
  su complessi) è compilato da numpy 2.4 **con FMA** nel build SIMD (`re*re - im*im`
  → `fma(re, re, -(im*im))`); Numba senza fastmath NON contrae. 1 ULP su orbite
  caotiche cambia l'escape time di una piccolissima frazione di pixel di bordo.
  **Regola**: in codice CPU che debba essere bit-riproducibile (e confrontato con
  Numba) il quadrato complesso va fatto in parti esplicite (`a*a-b*b`, `2*a*b` con
  ufunc singoli) — mai `np.square`/`np.multiply` su complessi.
- **Numba**: `cache=True` NON usato (mandel.py cambia nome di modulo a ogni load →
  cache inutilizzabile); compilazione pagata dal warmup thread all'avvio.
  `break`/`continue` nel corpo di `prange` impediscono la parallelizzazione.
  **Memoria cross-thread NON affidabile dentro `prange`** (verificato
  sperimentalmente): Numba/LLVM fa hoisting del load fuori dal loop (semantica
  single-thread), quindi un bump di un contatore scritto da un altro thread NON
  e' visto (il check in-kernel `ncol=0` non scattava mai). La cancellazione
  cooperativa va fatta a livello Python: bande di righe + check tra le bande.
  Nessun `fma` disponibile in numba (verificato).
- **Gate correttezza (gate.py, v5)**: GPU bit-identico a `baseline/*_gpu_*.npy`
  (riferimenti v4.15.1); CPU bit-identica a `baseline/*_cpu.npy` (riferimenti v5.0.0,
  nuova verità) E continuità ≤1,5% dei pixel vs `baseline/*_cpu_v4151.npy` (effetto
  FMA documentato). Rigenerare i riferimenti CPU con `baseline.py` dopo ogni cambio
  semantico del percorso CPU; i riferimenti GPU restano validi finché il kernel CUDA
  non cambia.
- Ambiente: la GPU è condivisa con il server LLM locale (llama.cpp) che NON va
  fermato (l'agente gira su quello stesso server). I benchmark GPU hanno rumore di
  fondo: usare workload bounded (n launch fissi, non loop a tempo) e confrontare
  versioni a parità di condizioni (o meglio: A/B nello stesso processo alternato).
- Dopo pause lunghe (>~10 s, es. render CPU da 3–18 s) il GPU torna a clock idle: la
  prima misura GPU è 10–15× più lenta del valore a regime. Scalare il clock con un
  breve burst prima di misurare.
