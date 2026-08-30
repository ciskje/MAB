# Piano reingegnerizzazione — velocità di rendering (CPU + CUDA)

> Basato su `mandel.py` v4.15.1. Obiettivo: render full più veloci e interazione più
> reattiva, senza cambiare l'interfaccia, l'architettura "un solo file" né il formato
> dei file (zona/config). Versione target: **v5.0.0** (rewrite).
>
> Stato attuale (misurato/osservato):
> - **CPU** (collo di bottiglia principale): loop `for i in range(mi)` a livello Python,
>   ogni passo = 3–4 ufunc full-array; a `mi≈10⁴` e 960×540 sono secondi di CPU.
> - **CPU**: tutte le matrici di lavoro (`xs, ys, real, imag, c, z, diverged, it, w4`)
>   vengono riallocate a ogni render.
> - **GPU**: `out.get()` è una copia D2H **pageable** (no DMA pinned); il primo render
>   paga context init + module load + allocation (hitch percettibile); launch config
>   (2 px/thread, 16×16) mai ottimizzata empiricamente.
> - **Pipeline**: nessun meccanismo di cancellazione — un full render avviato ma ormai
>   obsoleto (la vista è cambiata) va comunque completato; su CPU è il lavoro sprecato
>   peggiore durante pan/zoom rapidi.

## Decisioni (CONFERMATE 2026-08-30 dall'utente)
1. **Numba** per la CPU (dipendenza nuova, wheel pura, ~10–50×), con **fallback numpy**
   (l'attuale percorso resta come riferimento per il gate).
2. **Render progressivo**: opzionale, solo se utile (Fase 3.11), dietro flag.
3. **CPU resta solo f64** (no f32): la GPU f32 già copre il caso "veloce".

---

## Fase 0 — Baseline e instrumentation (precondizione, ~0,5 g) — **FATTA 2026-08-30**
- [x] `baseline.py` (nel repo): misure per-stadio GPU (kernel / D2H / PIL) e CPU
      (totale + allocazioni), mediana di 3 run + warmup, 960×540, mi = `auto_mi(half)`.
      NB: CuPy 14 ha rimosso `Event.elapsed_time` → timing con `perf_counter`+sync.
- [x] `baseline.txt` + 9 frame reference in `baseline/*.npy` (gate correttezza per-backend:
      nuovo-CPU vs CPU-v4.15.1, nuovo-GPU vs GPU-v4.15.1; CPU e GPU usano formule di
      coloring diverse per design → maxdiff CPU-vs-GPU non è un gate).

### Risultati misurati (v4.15.1, RTX 4070 SUPER, 960×540)
| Zona | mi | GPU f32 | GPU f64 | CPU f64 |
|---|---|---|---|---|
| z1 vista iniziale | 2000 | **1,03 ms** (kernel 0,29 / D2H 0,36 / PIL 0,35) | 9,55 ms (kernel 9,23) | **3305 ms** |
| z2 zoom medio | 8352 | **9,19 ms** (kernel 7,10 / D2H 2,01 / PIL 0,34) | 11,72 ms | **13592 ms** |
| z3 cuspo (bench) | 10915 | **49,95 ms** (kernel 48,74 / D2H 2,04 / PIL 0,34) | 163,38 ms | **18198 ms** |

Allocazioni+prefiltro CPU: ~10,5 ms/render (minimo; il loop è il 99%).
**Conferme**: CPU 300–400× più lenta della GPU f32 → rewrite CPU è il guadagno principale;
GPU f64 ≈ 32× più lenta di f32 (nota spec confermata); sulla GPU il kernel domina
(D2H/PIL <10%); zone quasi paraboliche (z3) costano 50× più di z1 a parità di risoluzione.

## Fase 1 — Quick wins a basso rischio (~1 g) — **FATTA 2026-08-30 (v4.16.0)**
- [x] 1. **Warmup GPU in background all'avvio**: thread daemon, render 64×64 f32+f64
      all'import → context init, module load, prime allocazioni (device + pinned)
      pagati fuori dal primo render reale.
- [x] 2. **D2H pinned**: buffer host pinned cacheato per dimensione (max 3 slot) +
      `runtime.memcpy(..., memcpyDeviceToHost)`, fallback `.get()`. **Copia D2H
      misurata 0,24 vs 0,42 ms (1,7×)** a 960×540.
      ⚠ **CuPy 14**: `cp.cuda.MemoryHost` NON esiste più → `cp.cuda.PinnedMemory(size)`
      (+`.ptr`) e `cp.cuda.runtime.memcpy(dst, src, n, kind)`; la vista numpy si fa
      con `np.frombuffer((ctypes.c_ubyte*n).from_address(ptr), dtype=np.uint8)`.
      ⚠ Un `except Exception` silenzioso sul percorso pinned ha mascherato l'errore di
      API (fallback `.get()` funzionante): verificare che il percorso nuovo sia
      EFFETTIVAMENTE usato, non solo che non errori.
- [x] 3. **Cache array CPU per (w,h)**: offset di griglia + real/imag/c/w4/z/diverged/it
      riutilizzati (max 3 dimensioni). **Gate bit-identico PASS** (stesso ordine delle
      operazioni!): `cx + half*X/(w/2)` NON è `cx + (half*(X/(w/2)))` (associatività
      IEEE) — il primo attempt rompeva la bit-identità in z1 (maxdiff 255) e passava
      in z2/z3 per fortuna → il gate multi-zona è indispensabile.
- [ ] 4. **Micro-benchmark launch config GPU**: rinviato alla Fase 3 (il kernel in z3
      è dominato dall'algebra, non dal launch; valutare lì con config reali).
- [x] 5. Preallocazione argomenti scalari: già presente (`_PAL_IDX`); l'ulteriore scartato.

### Verifiche (v4.16.0)
- **Gate correttezza**: bit-identico a v4.15.1 su 3 zone × (GPU f32, GPU f64, CPU) — PASS.
- **A/B CPU stesso processo** (v4.15.1 reale da git vs v4.16.0, 3 round alternati):
  z1 3360→3349 ms (−0,3%), z3 18514→18557 ms (+0,2%) → **pari nel rumore**
  (la cache toglie ~10 ms di allocazioni, sotto la soglia di misura qui).
- **Baseline cross-run** (GPU 4070 SUPER CONDIVISA con llama-server, ~90% util):
  GPU f32 z1 1,02→0,95 ms, z2 1,40→1,21 ms, z3 4,36→4,37 ms; CPU cross-run non
  affidabile (±30% di rumore per carico di fondo; l'A/B in-process è la misura valida).
- ⚠ **Ambiente**: la GPU è condivisa con il server LLM locale (llama.cpp) che DEVE
  restare attivo (l'agente ci gira sopra): benchmark con workload GPU bounded
  (spin a 20 launch fissi, non a tempo) e misure confrontate a parità di condizioni.
- ⚠ **Artefatto clock**: dopo pause lunghe (run CPU 3–18 s) il GPU è a clock idle e la
  prima misura è 10–15× più lenta (z3: 49 ms invece di 4,4). `baseline.py` ora scalda
  il clock prima di ogni misura.

## Fase 2 — Rewrite CPU (dove sta il guadagno, ~2 g) — **IN CORSO (v5.0.0)**
6. **Escape loop Numba** (`@njit(parallel=True)`) — FATTA (in attesa verifiche finali):
   - **Scelta architetturale (opzione B)**: il kernel Numba fa SOLO l'escape loop
     (produzione di `it[]`); geometry, interior analitico e coloring restano in numpy.
     Motivo: massima superficie di bit-identità e semplicità del gate.
     *Deviazione dal piano*: il coloring resta la formula CPU storica
     `t=(it/mi)^0.35` (NON la smooth del kernel GPU) — la parità CPU/GPU del coloring
     NON era mai vera (le formule differiscono per design) e i riferimenti v4.15.1 CPU
     usano la formula semplice: cambiarla avrebbe rotto il gate.
   - Loop parallelo sulle righe (`prange`), interno sulle `mi` iterazioni con
     early-exit (`z_re²+z_im²>4`); f64 nativo.
   - **Self-test di bit-identità all'avvio** (thread, 4×4 con casi limite vs loop numpy
     di riferimento con la STESSA aritmetica) → se fallisce o Numba assente: **fallback
     numpy automatico** (stesso loop, stesso gate).
   - Warmup/compilazione Numba in thread all'avvio (fuori dal primo render).
   - Attesa: **10–50×** sulla CPU (mi≈10⁴, 960×540: secondi → <1 s).
7. **Cancellazione cooperativa CPU** — FATTA: contatore di generazione `_GEN`
   (array numpy int32, size-1); `_submit` lo incrementa a ogni nuovo job. Verifica:
   - fallback numpy: a ogni iterazione del loop (break);
   - **Numba: a livello PYTHON tra 16 bande di righe** (il kernel riceve
     `row0/row1` e il chiamante fa il check tra le bande).
   Il worker scarta comunque il frame obsoleto. `my_gen=0` = nessuna cancellazione
   (CLI/bench).
   ⚠ **IL CHECK IN-KERNEL NON FUNZIONAVA (verificato sperimentalmente)**:
   la versione "ncol=0 se obsoleto" dentro `prange` NON vedeva il bump del
   contatore scritto da un altro thread (test A/B isolati: 100% delle righe
   elaborate nonostante il bump; con `continue` e stato iniziale coerente
   idem). Causa: Numba/LLVM fa hoisting del load fuori dal loop (semantica di
   memoria single-thread; i write cross-thread non sono nella sua analisi).
   **Regola**: con Numba la sincronizzazione cross-thread si fa a livello
   Python, mai dentro il kernel. Granularità: 1/16 del render (~10 ms a z3).
   (Nota: `break`/`continue` nel corpo `prange` restano comunque da evitare:
   Numba non parallelizzerebbe.)
8. **Variante 2B (tiling ThreadPool)**: NON SERVE (Numba adottata e funzionante).

### ⚠⚠ LESSONE CRITICA — FMA di numpy (scoperta durante la Fase 2)
- **numpy 2.4 compila `np.square` su complessi con FMA** (build SIMD):
  `re*re - im*im` → `fma(re, re, -(im*im))`. Numba (senza fastmath) NON contrae.
- Conseguenza: il kernel Numba "naive" (`z_re*z_re - z_im*z_im`) NON era
  bit-identico al vecchio loop numpy: 85/518 400 pixel diversi su z1 (escape time
  diverso di 1–600 per 1 ULP di `re(z²)` su orbite caotiche; z3: 0,8%).
- **Soluzione**: la CPU (numba + fallback) usa il quadrato in parti esplicite
  (`a*a-b*b`, `2*a*b` con ufunc singoli, ordine delle operazioni identico) → i due
  percorsi CPU sono bit-identici tra loro e deterministici; contro i frame v4.15.1
  restano differenze solo su pixel di bordo caotico (0,013% z1 / 0,133% z2 / 0,811% z3).
- **Gate CPU aggiornato (gate.py v5)**: (a) bit-identico ai riferimenti v5.0.0
  (nuova verità, `baseline/*_cpu.npy` rigenerati); (b) continuità ≤1,5% pixel vs
  riferimenti v4.15.1 (`baseline/*_cpu_v4151.npy`, conservati). GPU: bit-identico ai
  riferimenti v4.15.1 (percorso GPU intoccato).
- Regola permanente (spec + AGENTS): **mai `np.square`/`np.multiply` su complessi in
  codice CPU che debba essere bit-riproducibile**; sempre parti esplicite.
- NB: `cache=True` di Numba NON usato (mandel.py cambia nome di modulo a ogni load:
  `__main__`/`mandel`/dynamic → il cache sarebbe inutilizzabile); il warmup thread
  copre il costo.

### Risultati finali Fase 2 (A/B in-process, mediana 3 round, 960×540)
| Zona | mi | v4.15.1 (numpy) | v5.0.0 (Numba, bande) | Speedup |
|---|---|---|---|---|
| z1 vista iniziale | 2000 | 3315 ms | **29 ms** | **113×** |
| z3 cuspo (bench) | 10915 | 18608 ms | **167 ms** | **111×** |

- **Bit-identità**: Numba == fallback numpy su z1 e z3 (0 pixel diversi);
  self-test all'avvio OK. vs v4.15.1: differenze solo su bordo caotico
  (0,013% z1 / 0,133% z2 / 0,811% z3 — effetto FMA, gate ≤1,5%).
- **Cancellazione**: bump a metà render → termina alla banda successiva
  (misurato: 37% prima a z3) + scarto del frame dal worker (doppia garanzia).
- **Osservazione**: con 29–167 ms il full render CPU è ora PIU' VELOCE della
  finestra di 500 ms della pipeline preview→full: la cancellazione resta utile
  ma non è più critica (prima era 3–18 s di lavoro sprecato).

### Stato Fase 2 (2026-08-30)
- [x] Codice + self-test + warmup + fallback + docs (mandel.py v5.0.0, spec, AGENTS)
- [x] A/B velocità + bit-identità + cancellazione (sopra; test su copia locale
      del sorgente perché la share era inaccessibile dall'OS)
- [x] **FATTA 2026-08-30 (share tornata)**: rename `baseline/*_cpu.npy`
      → `*_cpu_v4151.npy`, rigenerazione riferimenti CPU v5 (baseline.py, grid GPU
      aggiornata a 1px/thread), gate v5 installato come `gate.py` e PASS, commit.

## Fase 3 — Polishing GPU (~1 g) — **FATTA 2026-08-30 (v5.0.0)**
9. **Launch config GPU — FATTA (micro-benchmark A/B in-process, rinvio da 1.4)**:
   - 8 varianti (px/thread 1/2/4 × block 16×16/32×8/8×32/32×16/8×16/16×8), f32,
     `--use_fast_math`; GPU condivisa (llama-server ~90%) → solo misure relative:
     warmup bounded (20 launch fissi per config), mediana di 15–30 round alternati.
   - **Tutte le varianti BIT-IDENTICHE tra loro** (z3, verifica esplicita): ogni
     pixel è calcolato in modo indipendente → la config di launch non cambia il
     risultato, solo la ripartizione del lavoro (nessun rischio per il gate).
   - Risultati (z1 mi=2000 / z3 mi=10915):
     | config | z1 | z3 |
     |---|---|---|
     | **1px 16×16 (vincente)** | **0.169 ms** | **3.30 ms** |
     | 1px 16×8 | 0.169 ms | 3.35 ms |
     | 2px 16×16 (attuale) | 0.278–0.310 ms | 3.68–3.71 ms |
     | 4px 16×16 | 0.44 ms | 4.44 ms |
     → **1px 16×16 adottata** (1.8× su z1, 1.1× su z3; 16×8 statisticamente
     equivalente: 0.5% di differenza, nel rumore della GPU condivisa).
   - Applicata a `mandel.py` (template kernel + grid `ceil(w/16)×ceil(h/16)`,
     vale per f32 e f64).
10. **Stream di copia separato — SCARTATA (decisione documentata)**:
    guadagno atteso 1–2 ms/frame (D2H 0.3–0.4 ms) solo sotto drag continuo,
    contro un cambio architetturale del worker (render concorrenti, gestione
    stream, ordinamento frame). Con render GPU di 0.2–4 ms non vale la
    complessità; la pipeline latest-wins resta.
11. **Render progressivo a strisce — SCARTATA (decisione documentata)**:
    il suo scopo era ridurre il tempo percepito dei render lenti; dopo le Fasi
    2–3 i full render sono 0.2–4 ms (GPU) e 29–167 ms (CPU), entrambi sotto o
    vicino alla finestra di 500 ms della pipeline: il full "appare" già quasi
    subito. Complessità media (strisce + D2H multipli + invalidazione) per un
    beneficio ormai impercettibile.
12. **`--use_fast_math` in zona cuspo — OK**: coperto dal gate (z3 bit-identico
    ai riferimenti v4.15.1, che già usava fastmath; il percorso GPU non cambia
    semantica, solo la config di launch).
13. **Stretch (fuori scope v5, solo nota)**: blit GPU→schermo via OpenGL (niente D2H).
    Cambio architetturale + dipendenza PyOpenGL; valutare solo se dopo v5 la D2H resta
    un collo di bottiglia visibile (oggi 0.3–0.4 ms: no).

## Fase 4 — Pipeline e benchmark (~0,5 g) — **FATTA 2026-08-30 (verifiche + decisioni)**
14. **Generazione job condivisa — GIÀ FATTA (Fase 2)**: `_submit` incrementa `_GEN`,
    il render CPU in corso (Numba bande / fallback) lo verifica e si ferma, il
    worker scarta il frame obsoleto. Latest-wins invariato; il lavoro in corso
    (CPU) è abortibile.
15. **Benchmark esteso — SCARTATA (decisione documentata)**: lo scopo del tasto
    "Benchmark" resta la COMPARABILITÀ tra versioni (regione+parametri fissi,
    render/s); la breakdown per stadio (kernel/D2H/CPU) è già coperta da
    `baseline.py` (diagnostica da riga di comando), e il confronto
    "CPU vecchia vs nuova" non ha più senso in v5 (il percorso vecchio non
    esiste più; il fallback numpy è solo rete di sicurezza, non prodotto).
16. **Timeout worker — NON SERVE (verificato)**: i carichi sono bounded:
    `mi` ≤ 50000 (auto) / ≤ 100000 (manuale, validato in UI), `half` ≥ 1e-12,
    `mi=100000` + zoom cuspo peggior caso ≈ 1,5 s su CPU (estrapolazione da
    z3 mi=10915 @ 167 ms). Nessun percorso non terminante (loop Numba bounded
    da `mi`, fallback bounded da `mi` + early-exit). Un timeout aggiungerebbe
    complessità (thread join, stati) per un rischio nullo.

## Fase 5 — Validazione e documentazione (~0,5 g) — **FATTA 2026-08-30 (share tornata)**
17. Gate di correttezza completo: 3 zone × (GPU f32, GPU f64, CPU) — maxdiff vs
    reference Fase 0 (GPU: 0, bit-identico; CPU: bit-identico ai riferimenti v5 +
    continuità ≤1,5% vs v4.15.1, effetto FMA documentato). **FATTA 2026-08-30**
    (share tornata): rename `*_cpu.npy` → `*_cpu_v4151.npy`; `baseline.py`
    rigenera riferimenti CPU v5 + baseline.txt (grid GPU aggiornata a 1px/thread);
    `gate.py` = gate v5 (GPU bit-id vs `*_gpu_*.npy`, CPU bit-id vs `*_cpu.npy`
    + continuità vs `*_cpu_v4151.npy`). **PASS**: z1 0.013% | z2 0.133% |
    z3 0.811% (tutte ≤1,5%), tutti gli altri maxdiff=0.
18. Spec.md/AGENTS/STORICO aggiornati durante le fasi (CPU Numba, FMA, modello di
    memoria Numba, launch config, pinned, benchmark); **VERSION = 5.0.0**. — FATTA.
19. Commit: Fasi 0-1 già committate (`9b15cc0`, `f03ebc0`); **commit v5.0.0
    (Fasi 2-5) FATTO 2026-08-30** (share tornata).

---

## Rischi e mitigazioni
| Rischio | Mitigazione |
|---|---|
| Numba fastmath cambia gli ultimi ulp → escape diverso vicino al bordo | gate maxdiff; flag `fastmath` disattivabile; fallback numpy esatto |
| Primo avvio più lento (JIT Numba 2–5 s) | compilazione in thread all'avvio (parallel warmup col GPU), cache persistente |
| Numba assente su altre macchine | fallback numpy attuale già funzionante (scelto come reference) |
| Pinned memory: dimensioni variabili | cache per (w,h), max 3 slot; fallback `.get()` classico |
| Threading CPU + GIL | ufunc rilasciano il GIL; con Numba il GIL non è in gioco (parallel=true) |
| Cancellazione a metà → frame parziali | solo frame completi entrano in coda; le strisce progressivo sono atomiche |
| Regressione visiva (coloring) | gate di correttezza su 3 zone a ogni fase, reference salvati in Fase 0 |

## Post-v5 — modifiche successive
- **v5.1.0 (2026-08-30)**: il benchmark ora esegue nella **modalità corrente**
  dell'app (motore CPU/CUDA + precisione f32/f64 selezionati in toolbar), invece
  che sempre in CUDA f32 (design v4.x). `bench_engine()` rimossa (display via
  `backend()`); il render del benchmark passa per `compute()` (dispatch
  `_USE_GPU` + `_PREC`, buffer proprio su CUDA, `my_gen=0`). Verificato:
  CPU/CUDA-f32/CUDA-f64 tutti bit-identici ai riferimenti baseline.

## Riepilogo risultati v5 (misurati)
| Parte | v4.15.1 | v5.0.0 | Guadagno |
|---|---|---|---|
| CPU full render (mi≈10⁴, z3) | 18.5 s (loop Python) | **167 ms** (Numba) | **~111×** |
| CPU full render (z1, mi=2000) | 3.3 s | **29 ms** | **~113×** |
| GPU primo render | hitch (init+load) | identico al secondo (warmup) | — |
| GPU D2H (960×540) | 0.42 ms pageable | **0.24 ms** pinned/DMA | 1.7× |
| GPU kernel (z3, f32) | 3.71 ms (2px 16×16) | **3.30 ms** (1px 16×16) | 1.1× |
| GPU kernel (z1, f32) | 0.31 ms | **0.17 ms** | 1.8× |
| Pan/zoom rapidi (CPU) | lavoro sprecato (3–18 s) | cancellazione cooperativa (bande) + scarto frame | — |
| UX progressivo | — | scartato (render già <170 ms) | — |
