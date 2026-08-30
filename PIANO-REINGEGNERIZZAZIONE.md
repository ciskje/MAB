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

## Fase 2 — Rewrite CPU (dove sta il guadagno, ~2 g)
6. **Escape loop Numba** (`@njit(parallel=True)`, `fastmath` come flag testabile):
   - una singola funzione compilata: loop parallelo sulle righe (`prange`), dentro il
     loop sulle `mi` iterazioni, con early-exit, interior test analitico inline,
     coloring `nu = it + 1 − log2(0.5·ln|z|²)` identico al kernel GPU (parità CPU/GPU),
     LUT passata come array (256×3) → `t` identica, output uint8 RGB.
   - `mi` clamp interno; f64 nativo.
   - **Fallback**: se `import numba` fallisce → percorso numpy attuale (restante anche
     come **riferimento** per il gate di correttezza).
   - Attesa: **10–50×** sulla CPU (mi≈10⁴, 960×540: secondi → <1 s).
7. **Cancellazione cooperativa CPU**: contatore di generazione; il loop Numba (o le
   bande del fallback) controlla il flag ogni banda/ogni N iterazioni e si ferma se la
   vista è cambiata. Elimina il lavoro sprecato durante pan/zoom rapidi.
8. **(Se scartata Numba — variante 2B)** tiling verticale in `min(cpu, 8)` bande +
   `ThreadPoolExecutor`, ciascuna banda con array propri e lo stesso loop numpy attuale
   (i ufunc rilasciano il GIL); stesso meccanismo di cancellazione per banda. Attesa:
   **4–8×**.

## Fase 3 — Polishing GPU (~1 g)
9. Adottare la launch config vincente della Fase 1.4 (costante per precisione).
10. **Stream di copia separato**: durante la D2H del frame corrente, se è già in coda
    il job successivo, lanciali sullo stream di compute → overlap copia/compute
    (utile sotto carichi sostenuti: drag continuo).
11. **Opzionale — render progressivo**: dividere il full render in 4–8 strisce
    orizzontali, ciascuna un launch + una D2H; l'UI mostra le strisce man mano
    (solo se la vista non cambia; un cambio vista scarta le restanti). UX: il full
    "appare" in 1/4 del tempo percepito. Complessità media; dietro flag.
12. Verifica che `--use_fast_math` non degradi il coloring in zona cuspo (già coperto
    dal gate Fase 0).
13. **Stretch (fuori scope v5, solo nota)**: blit GPU→schermo via OpenGL (niente D2H).
    Cambio architetturale + dipendenza PyOpenGL; valutare solo se dopo v5 la D2H resta
    un collo di bottiglia visibile.

## Fase 4 — Pipeline e benchmark (~0,5 g)
14. Generazione job unica condivisa tra worker e cancellazione (Fase 2.7); il
    latest-wins resta, ma ora anche il lavoro **in corso** (CPU) è abortibile.
15. Benchmark esteso: mostrare **kernel/D2H/CPU** separati, ripetizioni, e (se Numba)
    confronto CPU-vecchia vs CPU-nuova in un run.
16. Timeout di sicurezza sul worker (evita hang CPU in casi degeneri: `mi` max,
    `half` min già clampati — verifica).

## Fase 5 — Validazione e documentazione (~0,5 g)
17. Gate di correttezza completo: 3 zone × (GPU f32, GPU f64, CPU) — maxdiff vs
    reference Fase 0 (CPU: 0; GPU: 0; eventuali varianti fastmath: ≤1 in LUT-idx).
18. Rielaborazione spec.md (sezioni Rendering/CPU/Pipeline), note AGENTS.md
    (Numba: primo JIT ~2–5 s in background all'avvio, cache in `__pycache__`; pinned
    memory; regole di threading), STORICO completo, **VERSION = 5.0.0**.
19. Commit per fase (0→1→2→3→4→5), working tree pulita a ogni step.

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

## Riepilogo atteso
| Parte | Oggi | Target v5 |
|---|---|---|
| CPU full render (mi≈10⁴) | secondi (loop Python) | **<1 s** (Numba) / ~4–8× (tiling) |
| GPU primo render | hitch (init+load) | identico al secondo (warmup) |
| GPU D2H | pageable | pinned/DMA, ~1,5–2× |
| GPU launch | config statica | tuning misurato |
| Pan/zoom rapidi | lavoro sprecato in CPU | cancellazione cooperativa |
| UX (opz.) | preview→full | progressivo a strisce |
