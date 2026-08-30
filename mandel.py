# ============================================================================
# Insieme di Mandelbrot - visualizzatore interattivo
# VERSIONE: 5.1.1
# ----------------------------------------------------------------------------
# REGOLA: ogni modifica incrementa la versione e aggiunge una voce qui sotto
# (formato: versione - data - descrizione modifiche).
#
# STORICO:
# 5.1.1 - 2026-08-30
#   - All'avvio NON viene piu' ripristinato il file di zona corrente (view_file)
#     da config.json: il programma parte SENZA file corrente, quindi 'Salva zona'
#     chiede SEMPRE il nome (dialog Salva con nome...) finche' l'utente non
#     carica o salva esplicitamente una zona. view_file rimosso anche dalla
#     config salvata (dato morto: non viene piu' letto).
# 5.1.0 - 2026-08-30
#   - Benchmark: ora eseguito nella MODALITA' CORRENTE dell'app (motore
#     CPU/CUDA + precisione f32/f64 selezionati), invece che sempre in CUDA f32
#     (design v4.x 'sempre f32, indipendente dai settaggi'). Il dialog di
#     conferma mostra la modalita' corrente ('Motore: ... (corrente)').
#     Nota: per confrontare versioni va usato nella stessa modalita'.
#   - Rimossa la funzione bench_engine() (sostituita da backend() per il
#     display della modalita' corrente).
# 5.0.0 - 2026-08-30
#   - (Fase 2 reingegnerizzazione v5, vedi PIANO-REINGEGNERIZZAZIONE.md)
#     CPU: escape loop riscritto in Numba (@njit(parallel=True), dipendenza
#     opzionale): parallelo sulle righe, early-exit per pixel, f64. Attesa
#     10-50x sul collo di bottiglia CPU (mi~10^4: secondi -> <1 s).
#   - CPU: self-test di BIT-IDENTICITA' del kernel Numba contro il loop numpy
#     di riferimento all'avvio (thread, 16x16 con casi limite); se non tiene o
#     Numba e' assente -> fallback numpy automatico (stesso loop, stesso gate).
#   - CPU: cancellazione cooperativa: ogni nuovo job incrementa una
#     "generazione" condivisa; il render CPU in corso (Numba o fallback)
#     verifica il flag e si ferma se la vista e' cambiata; il worker scarta
#     il frame obsoleto. my_gen=0 = nessuna cancellazione (benchmark/riga di
#     comando). Il check Numba e' a livello PYTHON tra 16 bande di righe:
#     un check in-kernel NON e' affidabile (Numba/LLVM fa hoisting dei
#     letture di memoria cross-thread dentro prange: il bump non era visto,
#     verificato sperimentalmente).
#   - Warmup Numba in thread all'avvio: la compilazione (1-2 s) e' pagata
#     FUORI dal primo render CPU.
#   - Nota: GPU non cancellabile (render GPU <160 ms; non vale la
#     complessita').
#   - (Fase 3 reingegnerizzazione v5, vedi PIANO-REINGEGNERIZZAZIONE.md)
#     GPU: launch config ottimizzata con micro-benchmark A/B in-process
#     (8 varianti px/thread x block, GPU condivisa con llama-server: misure
#     solo relative, warmup bounded 20 launch, mediana). Vincente: 1 px/thread
#     block 16x16 (era 2 px/thread): z1 0.17 vs 0.31 ms (1.8x), z3 3.30 vs
#     3.71 ms (1.1x). Tutte le varianti verificate BIT-IDENTICHE tra loro
#     (ogni pixel e' calcolato in modo indipendente: la config di launch non
#     cambia il risultato, solo la ripartizione del lavoro).
#   - CORRETTEZZA (importante, scoperta durante questa fase): numpy 2.4
#     compila l'ufunc np.square su complessi con FMA (re*re - im*im contratto
#     in fma, dipendente dal build) mentre Numba (senza fastmath) NON
#     contrae: i due bit differivano su orbite caotiche (escape time
#     diverso per una piccolissima frazione di pixel di bordo: 0.01-0.8%
#     della immagine a seconda dello zoom). Soluzione: la CPU (numba e
#     fallback numpy) usa ORA il quadrato in parti esplicite (a*a-b*b,
#     2*a*b con ufunc singoli, senza FMA) -> i due percorsi CPU sono
#     bit-identici tra loro e deterministici. Contro i frame v4.15.1 CPU
#     (calcolati con np.square FMA) restano differenze solo su pixel di
#     bordo caotico (<=0.8% a z3): il gate CPU e' ora (a) bit-identico ai
#     riferimenti v5.0.0 e (b) continuita' <=1.5% dei pixel vs v4.15.1.
#     GPU invariato e bit-identico ai riferimenti v4.15.1.
# 4.16.0 - 2026-08-30
#   - (Fase 1 reingegnerizzazione v5, vedi PIANO-REINGEGNERIZZAZIONE.md)
#     Rendering: D2H su buffer host PINNED (CuPy 14: cp.cuda.PinnedMemory +
#     cp.cuda.runtime.memcpy, cache per dimensione, DMA ~1,7x del .get()
#     pageable; fallback .get()).
#   - CPU: array di lavoro cacheati per (w,h) (offset di griglia +
#     real/imag/c/w4/z/diverged/it riutilizzati: nessuna allocazione per
#     render; max 3 dimensioni cacheate). Bit-identico al codice precedente
#     (stesso ordine delle operazioni, verificato con gate).
#   - Warmup GPU in thread all'avvio (64x64, f32+f64): init context CUDA,
#     module load e prime allocazioni pagati FUORI dal primo render reale.
# 4.15.1 - 2026-08-30
#   - Grafica benchmark: dialog messagebox sostituiti da dialog Toplevel
#     custom (modali, centrati). Conferma: titolo, riga parametri in griglia
#     (etichetta + valore monospace), pulsanti Avvia/Annulla (Return/Esc
#     funzionanti). Risultato: il rendering/s e' il protagonista (42pt verde,
#     etichetta 'rendering / secondo'), sotto le statistiche secondarie e la
#     griglia dei parametri; in caso di errore 'BENCHMARK FALLITO' rosso +
#     dettaglio. Fix: status 'benchmark in corso' non piu' hardcodato a 8 s.
# 4.15.0 - 2026-08-30
#   - Formula auto MI aggiornata: mi = 2000 * (1 + log10(1.5/half)),
#     clamp [50, 50000] (era 400 * ..., clamp [50, 10000]); a vista iniziale
#     2000, nella zona del becco (half~5.2e-5) ~10916. Funzione di modulo
#     unica auto_mi(half) condivisa con il benchmark.
#   - Benchmark: 'mi' rimosso da BENCH/config e non e' piu' un parametro
#     fisso: viene SEMPRE calcolato con auto_mi(bench['half']), cosi' il
#     benchmark resta comparabile anche se cambia la formula auto. Dialog e
#     report mostrano il valore derivato.
# 4.14.0 - 2026-08-30
#   - Benchmark: regione di default cambiata in c=(-0.7499302568795561,
#     -0.015139113925433963i), half=5.226737155905588e-05 (mi=3000 invariato).
#     Aggiornata anche la 'bench' persistita in config.json (che sovrascrive
#     il default).
# 4.13.3 - 2026-08-30
#   - FIX (il 4.13.2 non bastava): _update_mi_label scriveva delete/insert
#     con l'Entry GIA' disabilitato dalla chiamata precedente (no-op silenziosi
#     di Tkinter) -> in auto il campo mostrava il valore del primo update e
#     poi non si aggiornava piu'. Ora: state=normal -> delete/insert ->
#     state=disabled (se auto). Verificato con test su widget reali.
# 4.13.2 - 2026-08-30
#   - FIX: in auto il campo MI era vuoto: in Tkinter delete/insert su Entry
#     disabilitato sono no-op silenziosi e lo stato veniva impostato PRIMA di
#     scrivere il testo. Ora il testo e' aggiornato per primo, poi lo stato.
#     (Verificato con test automatizzato su widget reali.)
# 4.13.1 - 2026-08-30
#   - FIX: con l'auto attivo nel campo MI non si vedeva il valore (il testo
#     "N (auto)" era piu' lungo del campo e, con l'allineamento a destra,
#     tagliava i numeri). Il campo ora mostra SEMPRE solo il numero (anche in
#     auto, disabilitato); l'indicatore auto passa sull'etichetta
#     ("Iterazioni (auto):").
# 4.13.0 - 2026-08-30
#   - Iterazioni: con l'auto disattivato il valore si puo' immettere
#     direttamente nel campo (prima etichetta sola, modificabile solo con
#     +/-1000); commit su Invio o perdita del focus; validazione: intero tra
#     50 e 100000, altrimenti messaggio di errore e ripristino del valore
#     precedente. Con l'auto attivo il campo e' disabilitato e mostra il
#     valore auto corrente ("N (auto)").
# 4.12.0 - 2026-08-30
#   - Benchmark: regione di default cambiata in c=(-0.7495463271154293,
#     -0.04276767920924388i), half=4.8677783048763816e-04 (mi=3000 invariato).
#     Aggiornata anche la 'bench' persistita in config.json (che sovrascrive
#     il default).
# 4.11.1 - 2026-08-30
#   - Refactor leggibilita' (nessun cambio di comportamento, CPU bit-identica
#     maxdiff=0 su 3 viste): metodi di MandelbrotApp raggruppati per funzione
#     (Costruzione UI / Helper UI / Vista e interazione / Controlli / Pipeline /
#     File / Benchmark) con intestazioni di gruppo; intestazione "Dispatch
#     backend" e gruppi di costanti a livello modulo.
# 4.11.0 - 2026-08-29
#   - Carica zona: nuova voce menu File "Carica zona..." che legge un file JSON
#     (stesso formato del salvataggio) e ripristina vista + iterazioni; il file
#     scelto diventa il "file corrente" per "Salva zona".
#   - Titolo: mostra il nome del file corrente (quello usato da "Salva zona");
#     il file corrente e' persistito in config.json (view_file).
# 4.10.0 - 2026-08-29
#   - Benchmark: nuova regione di default (zoom profondo
#     c=(0.42663924626512445, -0.3414973874054564i), half=2.298743311298834e-06)
#     e parametri persistiti in config.json (salvati/caricati, overridibili);
#     i metodi benchmark usano self.bench invece della costante BENCH.
# 4.9.0 - 2026-08-29
#   - Nuova: salvataggio della zona attuale (vista) su file JSON testuale
#     leggibile (menu File): "Salva zona" (riscrive l'ultimo file scelto, o
#     chiede il nome al primo uso) e "Salva zona con nome...". Salva
#     cx/cy/half + iterazioni (mi, mi_auto) per riprodurre la vista.
# 4.8.0 - 2026-08-29
#   - Refactor UI: __init__ (116 righe) spezzato in _setup_fonts/_build_toolbar/
#     _build_canvas_status/_build_menu/_bind_events/_start_pipeline. Helper
#     _refresh_title e _select_palette/_select_backend/_select_precision eliminano
#     la triplice ripetizione "deselect+select" (era in set_* + load_config + reset).
#   - Perf CPU: test di fuga senza sqrt (|z|>2 -> z.re^2+z.im^2>4), output
#     bit-identico (maxdiff=0 su 5 viste incl. deep zoom). Perf GPU: indice
#     palette preallocato (_PAL_IDX), niente np.asarray per render.
#   - Robustezza: clamp half >= MIN_HALF (1e-12) in zoom_at (evita half=0).
#   - Pulizia: rimossa prec() (getter mai usato) e import time ridondante in
#     save_png; change_mi usa _update_mi_label(); tolti 'global' inutili
#     (save/load_config, reset).
# 4.7.2 - 2026-08-29
#   - UI: etichette con iniziale maiuscola (Motore/Palette/Precisione/Iter/Auto/
#     Iterazioni)
# 4.7.1 - 2026-08-29
#   - FIX: disattivando "iter: auto", il valore mostrava il fisso iniziale
#     (stale) invece del valore auto corrente. Ora self.mi viene congelato
#     sull'eff_mi() corrente al momento dello spegnimento (i pulsanti +/-
#     partono quindi dal valore corretto).
# 4.7.0 - 2026-08-29
#   - Kernel GPU: rilevamento analitico dell'interior PRIMA del loop. Bulbo
#     periodica-2 (chiuso) + cardioide principale intera via |1-sqrt(1-4c)|<1
#     (riscritta senza complessi), con prefiltro bounding-box. I pixel interni
#     (zona nera) saltano tutte le mi iterazioni: speedup misurato 1.6x a vista
#     iniziale, ~3x a zoom medio, ~9x a zoom profondo (mi=3000). Nessun falso
#     positivo (GPU/CPU identici al kernel originale).
#   - Kernel GPU compilato con --use_fast_math (log2f/powf/fminf/sqrtf piu'
#     economici; flag passato a cp.RawKernel come options=("--use_fast_math",)).
#   - Backend CPU: stesso test interior analitico (esclude cardioide/bulbo dal
#     loop) + loop in-place (np.square/z+=c, niente allocazioni): ~2.5x piu'
#     veloce, output identico (maxdiff=0 vs CPU originale).
# 4.6.0 - 2026-08-29
#   - Nuova palette "Termal": gradiente che parte col ghiaccio (blu/ciano
#     chiari), passa dal bianco gelido e finisce col fuoco (oro/arancio/rosso).
#   - Palette gestite da un registro PALETTES ordinato (fuoco/ghiaccio/termal,
#     ordine = indice passato al kernel): LUT __constant__ e selezione del
#     kernel, pulsanti UI e config sono tutti generati dal registro
#     (niente piu' if/else fuoco/ghiaccio).
# 4.5.3 - 2026-08-29
#   - Numero di versione nel titolo della finestra ("Insieme di Mandelbrot v4.5.3 - ...")
# 4.5.2 - 2026-08-29
#   - Benchmark sempre in float32 (GPU), indipendente da motore/precisione
#     selezionati nell'app (compute_gpu/compute accettano un prec esplicito)
# 4.5.1 - 2026-08-29
#   - Benchmark: prima dell'avvio e' mostrato un dialog che descrive il test
#     (regione, iterazioni, risoluzione, motore, durata) con OK (avvia) e
#     Cancel (annulla)
# 4.5.0 - 2026-08-29
#   - Barra di stato: tolto "centro" e "meta-larghezza", aggiunto il tempo di
#     rendering (misurato nel worker attorno a compute(), es. "render: 120 ms")
#   - Config persistente: tutti i settaggi (vista cx/cy/half, iterazioni, auto,
#     precisione, palette, motore) salvati e ricaricati da
#     %USERPROFILE%\mandelbrot\config.json a ogni esecuzione (salvataggio all'uscita
#     + throttled ~1s sui cambiamenti)
#   - Tasto "Reset": riporta vista + tutti i settaggi ai default, salva config
#   - Benchmark standardizzato (tasto "Benchmark"): regione fissa
#     c=(-0.74364388703, 0.13182590421i), meta=0.002, 3000 iter, 960x540,
#     ripetuta per 8 s in thread dedicato (buffer proprio, niente contesa con il
#     render normale); dialog finale con numero di ripetizioni, rate e ms/render
#   - compute_gpu/compute accettano un buffer opzionale (evita contesa su _BUF)
# 4.4.0 - 2026-08-29
#   - Auto-iterazioni raddoppiate: mi = 400 * (1 + log10(HALF0/half)),
#     clamp [50, 10000] (era 200 * (1 + log10(...)), clamp [50, 5000])
#   - Pulsanti manuali iterazioni: passo da +/-100 a +/-1000
# 4.3.0 - 2026-08-29
#   - UI: controlli, etichette, pulsanti -100/+100 e menu ingranditi (font
#     TkDefaultFont/TkTextFont/TkMenuFont a 13pt) + pady=3 sui controlli in alto
#     per area cliccabile piu ampia
# 4.2.0 - 2026-08-29
#   - Modalita "double" (precisione f64) disattivabile: kernel CUDA generati
#     in due varianti (float32/float64) dalla stessa sorgente parametrizzata;
#     radio "precisione: f32/f64" (default f32, f64 penalizzata su CUDA
#     consumer ~1:32 throughput float); la CPU resta sempre f64 (complex128).
#     Titolo e status mostrano la precisione attiva (es. "CUDA f64").
#   - Controlli e radio spostati in alto (prima: sotto il canvas)
#   - Menu File: "Salva immagine... (Ctrl+S)" + "Esci"; shortcut Ctrl+S;
#     salva l'immagine corrente del canvas via asksaveasfilename
# 4.1.0 - 2026-08-29
#   - Modalita "MI auto" (attiva di default, disattivabile: pulsante
#     "iter: auto"): le massime iterazioni crescono logaritmicamente con lo
#     zoom: mi = 200 * (1 + log10(HALF0/half)), clamp [50, 5000]. A vista
#     iniziale coincide con 200. Con auto attiva i pulsanti -100/+100 sono
#     disabilitati e l'etichetta mostra il valore calcolato.
# 4.0.0 - 2026-08-27
#   - FIX: la LUT passata come argomento device al kernel veniva sovrascritta
#     dall'output a ogni launch (colori magenta/azzurri random dal 2° frame in
#     poi). Palette ora incorporate nel kernel come array __constant__
#     generate dagli stessi dati Python; selezione palette via int (0=fuoco,
#     1=ghiaccio). Verificato: chiamate GPU consecutive identiche, diff vs CPU
#     solo su pixel di bordo.
#   - Toggle runtime CPU/CUDA (pulsanti, radio)
#   - Palette Fuoco/Ghiaccio (pulsanti, radio)
#   - LUT 256x3 condivisa CPU/GPU (colori identici sui due backend)
#   - Mappatura colore percepibile: gamma 0.35 su t=it/mi (fix schermata nera)
#   - Preview 1/4 + full render dopo 500ms anche su CPU
# 3.0.0
#   - Backend GPU CUDA (CuPy RawKernel, NVRTC): 2 px/thread, early-exit,
#     ricentramento algebrico, buffer device riutilizzato, fallback CPU
#   - Fattore d'aspetto corretto (y scale h/w)
# 2.0.0
#   - Pipeline asincrona latest-wins (worker + Condition, niente blocco UI)
#   - Zoom rotella / pan tasto sinistro, doppio click = reset vista
# 1.0.0
#   - Prima versione interattiva: app tkinter, rendering CPU numpy
# ============================================================================

import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import tkinter.font as tkfont
import threading
import queue
import math
import os
import json
import time
import ctypes
import numpy as np
from PIL import Image, ImageTk

# --- Costanti di vista e rendering ---
INIT_W, INIT_H = 960, 540
CX0, CY0, HALF0 = -0.5, 0.0, 1.5
MI0 = 200
# Iterazioni "auto": mi = MI_AUTO_BASE * (1 + log10(HALF0/half)), clamp [50, 50000]
# (coefficiente alzato da 400 a 2000 e clamp max da 10000 a 50000: le zone
# quasi paraboliche, es. vicino al becco della cardioide, fuggono lentamente
# e con il vecchio coefficiente il coloring non convergeva).
# La STESSA formula e' usata dal benchmark (vedi v4.15.0).
MI_AUTO_BASE = 2000
MI_AUTO_MIN, MI_AUTO_MAX = 50, 50000

MIN_HALF = 1e-12  # clamp minimo per half (evita zoom infinito -> half=0, stato degenere)
MIN_DIM = 50  # larghezza/altezza canvas minimi per considerare il canvas valido

def auto_mi(half):
    """Iterazioni 'auto' per una data half: formula unica condivisa da eff_mi
    e benchmark (vedi costanti MI_AUTO_*)."""
    z = HALF0 / max(half, 1e-12)
    return int(max(MI_AUTO_MIN, min(MI_AUTO_MAX, MI_AUTO_BASE * (1.0 + math.log10(z)))))

# --- Percorsi e parametri ---
# Config salvata/caricata a ogni esecuzione (vista + tutti i settaggi)
CONFIG_PATH = os.path.join(os.path.expanduser("~"), "mandelbrot", "config.json")

# Benchmark: regione + parametri. Default = regione fornita dall'utente;
# i valori sono persistiti in config.json (overridibili).
# NB: 'mi' non fa piu' parte di BENCH: e' sempre derivato dalla formula auto
# (auto_mi(bench['half'])) per mantenere il benchmark comparabile tra versioni.
BENCH = dict(cx=-0.7499302568795561, cy=-0.015139113925433963, half=5.226737155905588e-05,
             w=960, h=540, secs=8.0)

VERSION = "5.1.1"

# ---------------- Palette (LUT 256x3 condivisa CPU/GPU) ----------------
_FIRE = (
    (0.0, 0.2, 0.45, 0.7, 0.9, 1.0),
    (0.05, 0.35, 0.85, 1.0, 1.0, 1.0),
    (0.0, 0.02, 0.2, 0.65, 0.95, 1.0),
    (0.0, 0.0, 0.02, 0.15, 0.55, 1.0),
)
_ICE = (
    (0.0, 0.25, 0.5, 0.75, 1.0),
    (0.02, 0.05, 0.30, 0.70, 1.0),
    (0.02, 0.15, 0.55, 0.85, 1.0),
    (0.10, 0.45, 0.90, 1.0, 1.0),
)
# Termal: parte col ghiaccio (blu/ciano chiari), passa dal bianco gelido e
# finisce col fuoco (oro/arancio/rosso). t=0 ghiaccio, t=1 fuoco.
_TERMAL = (
    (0.0,  0.2,  0.4,  0.55, 0.7,  0.85, 1.0),
    (0.02, 0.10, 0.55, 0.95, 1.00, 1.00, 1.00),
    (0.08, 0.45, 0.80, 0.96, 0.85, 0.55, 0.30),
    (0.28, 0.85, 0.95, 0.98, 0.45, 0.20, 0.10),
)

# Registro palette (fonte unica): l'ordine definisce l'indice passato al kernel
# (0=fuoco, 1=ghiaccio, 2=termal). UI, config e __constant__ del kernel sono
# tutti generati da questo dict.
PALETTES = {
    "fuoco": _FIRE,
    "ghiaccio": _ICE,
    "termal": _TERMAL,
}

def make_lut(pal):
    st, r, g, b = pal
    st = np.asarray(st, dtype=np.float64)
    t2 = np.linspace(0.0, 1.0, 256)
    rgb = np.stack([np.interp(t2, st, np.asarray(r)),
                    np.interp(t2, st, np.asarray(g)),
                    np.interp(t2, st, np.asarray(b))], axis=1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)

_PALETTE = "fuoco"
_LUT = make_lut(PALETTES["fuoco"])

def apply_palette(name):
    global _PALETTE, _LUT
    if name not in PALETTES:
        name = "fuoco"
    _PALETTE = name
    _LUT = make_lut(PALETTES[name])

# ---------------- GPU (CUDA) ----------------
# Tutte le palette (PALETTES) sono incorporate nel kernel come array __constant__:
# non si passano array device come argomenti (con la LUT come argomento
# il buffer di output sovrascriveva la memoria della LUT a ogni launch).
# Le dichiarazioni __constant__ e la selezione per indice pal sono generate
# dal registro PALETTES (vedi _build_kernel).
# Il kernel e' generato in due varianti di precisione (float32/float64)
# dalla stessa sorgente parametrizzata: f32 e' nativa su CUDA, f64 e'
# corretta ma penalizzata (~1:32 di throughput float) sulle GPU consumer.
def _fmt_lut(lut):
    return ", ".join(str(int(v)) for v in lut.ravel())

_KERNEL_TPL = r'''
@@CONSTS@@

__device__ __forceinline__ void pal_lut(@@T@@ t, const unsigned char* lut, unsigned char* rgb) {
    int idx = (int)(@@FMIN@@(1.0@@S@@, @@FMAX@@(0.0@@S@@, t)) * 255.0@@S@@);
    rgb[0] = lut[idx * 3 + 0];
    rgb[1] = lut[idx * 3 + 1];
    rgb[2] = lut[idx * 3 + 2];
}

__device__ __forceinline__ void process_pixel(
    int col, int row, int w, int h,
    @@T@@ cx, @@T@@ cy, @@T@@ half, int mi,
    const unsigned char* lut,
    unsigned char* __restrict__ out)
{
    if (col >= w) return;
    @@T@@ x0 = cx + half * ((@@T@@)(2 * col - w) / (@@T@@)w);
    @@T@@ y0 = cy + half * ((@@T@@)h / (@@T@@)w) * ((@@T@@)(2 * row - h) / (@@T@@)h);
    // Interior analitico: bulbo periodica-2 + cardioide principale
    // (c interno alla cardioide principale <=> |1 - sqrt(1 - 4c)| < 1,
    // riscritto senza complessi come R < 2*sqrt(0.5*(R + A)), con
    // A = 1 - 4*Re(c), R = |1 - 4c|). Se interno, il punto non diverge mai
    // -> pixel nero senza eseguire le mi iterazioni.
    // Prefiltro bounding-box dell'insieme: evita il costo della sqrt sui
    // pixel chiaramente esterni (fuggono in 1-2 iterazioni).
    if (x0 >= -2.0@@S@@ && x0 <= 0.4@@S@@ && y0 >= -1.3@@S@@ && y0 <= 1.3@@S@@) {
        unsigned char* p = out + (size_t)(row * w + col) * 3;
        @@T@@ d2 = (x0 + 1.0@@S@@) * (x0 + 1.0@@S@@) + y0 * y0;
        if (d2 <= 0.0625@@S@@) { p[0] = 0; p[1] = 0; p[2] = 0; return; }
        @@T@@ A = 1.0@@S@@ - 4.0@@S@@ * x0;
        @@T@@ B = -4.0@@S@@ * y0;
        @@T@@ R = @@SQRT@@(A * A + B * B);
        if (R < 2.0@@S@@ * @@SQRT@@(0.5@@S@@ * (R + A))) { p[0] = 0; p[1] = 0; p[2] = 0; return; }
    }
    @@T@@ a = cx * cx + (x0 - cx);
    @@T@@ two_cx = 2.0@@S@@ * cx;
    @@T@@ wr = -cx, wi = 0.0@@S@@;
    int it = 0;
    bool esc = false;
    @@T@@ mag2 = 0.0@@S@@;
    for (int i = 0; i < mi; ++i) {
        if (esc) break;
        @@T@@ nr = wr * wr - wi * wi + two_cx * wr + a;
        @@T@@ ni = two_cx * wi + 2.0@@S@@ * wr * wi + y0;
        wr = nr; wi = ni;
        @@T@@ zr = wr + cx;
        mag2 = zr * zr + wi * wi;
        if (mag2 > 4.0@@S@@) { esc = true; it = i; }
    }
    unsigned char* p = out + (size_t)(row * w + col) * 3;
    if (!esc) { p[0] = 0; p[1] = 0; p[2] = 0; return; }
    @@T@@ nu = (@@T@@)it + 1.0@@S@@ - @@LOG2@@(0.5@@S@@ * @@LOG@@(mag2));
    @@T@@ t = @@POW@@(@@FMIN@@(1.0@@S@@, @@FMAX@@(0.0@@S@@, nu / (@@T@@)mi)), 0.35@@S@@);
    unsigned char rgb[3];
    pal_lut(t, lut, rgb);
    p[0] = rgb[0]; p[1] = rgb[1]; p[2] = rgb[2];
}

extern "C" __global__ void __launch_bounds__(256) @@KNAME@@(
    unsigned char* __restrict__ out,
    int pal,
    @@T@@ cx, @@T@@ cy, @@T@@ half,
    int w, int h, int mi)
{
    @@LUTSELECT@@
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    // v5.0.0 (Fase 3): 1 px/thread (era 2): micro-benchmark A/B in-process
    // su 8 varianti (px 1/2/4 x block 16x16/32x8/8x32/32x16/8x16/16x8)
    // -> 1px 16x16 vincente su entrambe le zone (z1 1.8x, z3 1.1x); tutte le
    // varianti bit-identiche tra loro (ogni pixel e' calcolato in modo
    // indipendente, la config non cambia il risultato).
    int col0 = tx;
    process_pixel(col0, ty, w, h, cx, cy, half, mi, lut, out);
}
'''

_PRECS = {
    "f32": dict(T="float",  S="f", FMIN="fminf", FMAX="fmaxf",
                LOG2="log2f", LOG="logf", POW="powf", SQRT="sqrtf", KNAME="mandel_kernel_f32"),
    "f64": dict(T="double", S="",  FMIN="fmin",  FMAX="fmax",
                LOG2="log2",  LOG="log",  POW="pow",  SQRT="sqrt",  KNAME="mandel_kernel_f64"),
}

def _build_kernel(prec):
    d = _PRECS[prec]
    # Dichiarazioni __constant__ generate dal registro (una per palette).
    consts = "\n".join(
        "__constant__ unsigned char LUT_%s[768] = { %s };"
        % (name.upper(), _fmt_lut(make_lut(pal)))
        for name, pal in PALETTES.items())
    # Selezione LUT per indice pal (ordine PALETTES = indice).
    names = list(PALETTES)
    select = "    const unsigned char* lut = LUT_%s;" % names[0].upper()
    for i, name in enumerate(names[1:], start=1):
        select += "\n    if (pal == %d) lut = LUT_%s;" % (i, name.upper())
    src = (_KERNEL_TPL
           .replace("@@CONSTS@@", consts)
           .replace("@@LUTSELECT@@", select))
    for k, v in d.items():
        src = src.replace("@@" + k + "@@", v)
    return src, d["KNAME"]

_GPU = False
_KERNEL_F32 = None
_KERNEL_F64 = None
_BUF = None
try:
    import cupy as cp
    if cp.cuda.is_available():
        _src, _name = _build_kernel("f32")
        _KERNEL_F32 = cp.RawKernel(_src, _name, options=("--use_fast_math",))
        _GPU = True
        try:
            _src, _name = _build_kernel("f64")
            _KERNEL_F64 = cp.RawKernel(_src, _name, options=("--use_fast_math",))
        except Exception:
            _KERNEL_F64 = None
except Exception:
    _GPU = False

_PREC = "f32"

def set_prec(p):
    global _PREC
    if p == "f64" and _KERNEL_F64 is None:
        return False
    if p in ("f32", "f64"):
        _PREC = p
    return True

# Indice palette (0..N-1) preallocato come array size-1: evita np.asarray per render
_PAL_IDX = [np.asarray(i, dtype=np.int32) for i in range(len(PALETTES))]

# v4.16.0: buffer host pinned per il D2H (DMA, ~1,7x piu' veloce di .get()
# pageable su frame grandi). Cache per dimensione (max 3 slot).
# NB CuPy 14: MemoryHost rimosso -> cp.cuda.PinnedMemory (+ ptr).
_PINNED = {}

def _pinned_view(w, h):
    """(PinnedMemory, vista numpy uint8 h*w*3) su memoria host pinned."""
    key = (w, h)
    ent = _PINNED.get(key)
    if ent is None:
        size = w * h * 3
        pm = cp.cuda.PinnedMemory(size)
        host = np.frombuffer((ctypes.c_ubyte * size).from_address(pm.ptr),
                             dtype=np.uint8)
        ent = (pm, host)
        if len(_PINNED) >= 3:
            _PINNED.pop(next(iter(_PINNED)))
        _PINNED[key] = ent
    return ent

def compute_gpu(cx, cy, half, w, h, mi, buf=None, prec=None):
    global _BUF
    need = w * h * 3
    if buf is None:
        if _BUF is None or _BUF.size < need:
            _BUF = cp.empty((need,), dtype=cp.uint8)
        buf = _BUF
    out = buf[:need]
    bx, by = 16, 16
    # v5.0.0 (Fase 3): 1 px/thread -> grid = ceil(w/bx) x ceil(h/by)
    grid = ((w + bx - 1) // bx, (h + by - 1) // by)
    pal = list(PALETTES).index(_PALETTE)
    p = prec if prec in ("f32", "f64") else _PREC
    use64 = (p == "f64") and (_KERNEL_F64 is not None)
    kernel = _KERNEL_F64 if use64 else _KERNEL_F32
    fdt = np.float64 if use64 else np.float32
    args = (out,
            _PAL_IDX[pal],
            np.asarray(cx, dtype=fdt),
            np.asarray(cy, dtype=fdt),
            np.asarray(half, dtype=fdt),
            np.asarray(w, dtype=np.int32),
            np.asarray(h, dtype=np.int32),
            np.asarray(mi, dtype=np.int32))
    kernel(grid, (bx, by), args)
    # v4.16.0: D2H pinned (DMA, memcpyDeviceToHost) con fallback .get().
    try:
        pm, host = _pinned_view(w, h)
        cp.cuda.runtime.memcpy(pm.ptr, out.data.ptr, w * h * 3,
                               cp.cuda.runtime.memcpyDeviceToHost)
        return Image.fromarray(host.reshape((h, w, 3)))
    except Exception:
        return Image.fromarray(out.get().reshape((h, w, 3)))

# ---------------- CPU (fallback) ----------------

# v4.16.0: array di lavoro CPU cacheati per (w,h) (nessuna allocazione per
# render: offset di griglia + real/imag/c/w4/z/diverged/it riutilizzati).
# v5.0.0: escape loop Numba (opzionale, con fallback numpy bit-identico).
# Il kernel fa SOLO il loop di escape (stessa semantica del loop numpy:
# it = indice 0-based dell'iterazione in cui |z|^2 > 4, altrimenti 0);
# geometria, prefiltro interior e coloring restano in numpy (invariati).
# Cancellazione cooperativa (my_gen = generazione del job; 0 = nessuna
# cancellazione, es. benchmark/CLI): le righe sono divise in 16 bande e il
# contatore _GEN[0] e' controllato a livello PYTHON tra le bande (affidabile;
# un check in-kernel NON lo e': Numba/LLVM fa hoisting dei letture di
# memoria cross-thread dentro prange, verificato sperimentalmente).
# Se il job diventa obsoleto, le bande rimanenti vengono saltate e il frame
# e' comunque scartato dal worker (doppia garanzia).
_NUMBA_OK = False
_GEN = np.zeros(1, dtype=np.int32)
try:
    from numba import njit, prange

    # NB: NO cache=True: il cache di numba e' legato al nome del modulo e
    # mandel.py e' eseguito/importato con nomi diversi (__main__ vs mandel vs
    # loader dinamico) -> il load della cache falliva. La compilazione (~1-2 s)
    # e' pagata dal thread di warmup all'avvio, prima del primo render CPU.
    @njit(parallel=True)
    def _mandel_escape(c, diverged, it, mi, row0, row1):
        # Solo il loop di fuga sulle righe [row0, row1). La cancellazione
        # cooperativa e' gestita dal chiamante (compute_cpu) a livello
        # PYTHON, tra bande di righe: un check in-kernel di un contatore
        # scritto da un altro thread NON e' affidabile con Numba/LLVM
        # (verificato sperimentalmente: il load di gen_arr[0] viene fatto
        # una sola volta / hoistato fuori dal loop prange, quindi il bump
        # cross-thread non e' visto e le righe vengono comunque elaborate).
        w = c.shape[1]
        for row in prange(row0, row1):
            for col in range(w):
                if diverged[row, col]:
                    continue
                vre = c[row, col].real
                vim = c[row, col].imag
                z_re = 0.0
                z_im = 0.0
                for i in range(mi):
                    # z = z^2 + c (stesso ordine di operazioni del numpy)
                    z_re, z_im = z_re * z_re - z_im * z_im, 2.0 * z_re * z_im
                    z_re += vre
                    z_im += vim
                    if z_re * z_re + z_im * z_im > 4.0:
                        it[row, col] = i
                        break
except Exception:
    njit = None
    prange = None

def _numba_warmup():
    """Compila il kernel in background all'avvio e fa un self-test di
    bit-identita' contro il loop numpy di riferimento; se la bit-identita'
    non tiene (o la compilazione fallisce) resta il fallback numpy.
    """
    global _NUMBA_OK
    if njit is None:
        return
    try:
        # self-test: 4x4 con casi limite (interiori, fuga rapida/lenta,
        # pre-diverged, overflow) — il risultato DEVE essere bit-identico
        # al loop numpy di compute_cpu.
        c = np.array([
            -0.5 + 0.5j, 0.7 + 0.1j, -1.7 + 0j, 0.3 + 0.6j,
            -0.1 - 0.9j, 2.5j, 0.0, 1.0,
            -0.749 - 0.001j, 0.1 - 0.7j, -1.2 - 0.5j, 0.9 + 0.9j,
            -0.28 + 0.01j, -2.0j, 0.5 - 1.9j, -0.75 + 0.12j,
        ], dtype=np.complex128).reshape(4, 4)
        d = np.zeros((4, 4), dtype=bool)
        d[0, 0] = True  # pre-diverged (come il prefiltro interior)
        it_numba = np.zeros((4, 4), dtype=np.int32)
        _mandel_escape(c, d, it_numba, 64, 0, 4)
        # riferimento numpy con la STESSA aritmetica del fallback in
        # compute_cpu (parti esplicite, no FMA) — il risultato DEVE essere
        # bit-identico al kernel Numba.
        zr = np.zeros((4, 4))
        zi = np.zeros((4, 4))
        cr = c.real
        ci = c.imag
        div = d.copy()
        it_ref = np.zeros((4, 4), dtype=np.int32)
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(64):
                if not (~div).any():
                    break
                zr2 = zr * zr - zi * zi
                zi2 = 2.0 * zr * zi
                zr = zr2 + cr
                zi = zi2 + ci
                m = ((zr * zr + zi * zi) > 4.0) & ~div
                if not m.any():
                    continue
                div |= m
                it_ref[m] = i
        if not np.array_equal(it_numba, it_ref):
            return  # bit-identita' violata: resta il fallback numpy
        # warmup: compila il percorso parallelo a dimensioni realistiche
        c2 = np.zeros((64, 64), dtype=np.complex128)
        d2 = np.zeros((64, 64), dtype=bool)
        it2 = np.zeros((64, 64), dtype=np.int32)
        _mandel_escape(c2, d2, it2, 32, 0, 64)
        _NUMBA_OK = True
    except Exception:
        pass

_CPU_WS = {}

def _cpu_ws(w, h):
    key = (w, h)
    ws = _CPU_WS.get(key)
    if ws is None:
        ws = {
            "X": np.arange(w) - w / 2,
            "Y": np.arange(h) - h / 2,
            "tx": np.empty(w),
            "ty": np.empty(h),
            "real": np.empty((h, w)),
            "imag": np.empty((h, w)),
            "c": np.empty((h, w), dtype=np.complex128),
            "w4": np.empty((h, w), dtype=np.complex128),
            # v5.0.0: z in parti reali/immaginarie (NO np.square: quel ufunc e'
            # compilato da numpy con FMA -> bit diversi dal percorso Numba).
            "zr": np.empty((h, w)),
            "zi": np.empty((h, w)),
            "tr": np.empty((h, w)),
            "ti": np.empty((h, w)),
            "diverged": np.empty((h, w), dtype=bool),
            "it": np.empty((h, w), dtype=np.int32),
        }
        if len(_CPU_WS) >= 3:
            _CPU_WS.pop(next(iter(_CPU_WS)))
        _CPU_WS[key] = ws
    return ws

def compute_cpu(cx, cy, half, w, h, mi, my_gen=0):
    ws = _cpu_ws(w, h)
    real, imag = ws["real"], ws["imag"]
    # Ordine delle operazioni IDENTICO al codice originale (bit-identita'
    # garantita): xs = cx + (half*X)/(w/2); ys = cy + ((half*h/w)*Y)/(h/2).
    np.multiply(ws["X"], half, out=ws["tx"])
    ws["tx"] /= (w / 2)
    ws["tx"] += cx
    np.multiply(ws["Y"], half * (h / w), out=ws["ty"])
    ws["ty"] /= (h / 2)
    ws["ty"] += cy
    np.copyto(real, ws["tx"][None, :])
    np.copyto(imag, ws["ty"][:, None])
    c = ws["c"]
    np.multiply(imag, 1j, out=c)
    c += real
    zr = ws["zr"]
    zi = ws["zi"]
    zr.fill(0)
    zi.fill(0)
    diverged = ws["diverged"]
    diverged.fill(False)
    it = ws["it"]
    it.fill(0)
    # Interior analitico (stesso criterio del kernel GPU): bulbo periodica-2 +
    # cardioide principale (|1 - sqrt(1 - 4c)| < 1). Questi pixel non divergono
    # mai -> restano it=0 (neri) ed esclusi dal loop: ~2.5x sulla CPU.
    w4 = ws["w4"]
    np.multiply(c, -4.0, out=w4)
    w4 += 1.0
    diverged |= (np.abs(w4) < 2.0 * np.sqrt(0.5 * (np.abs(w4) + np.real(w4))))
    diverged |= (np.abs(c + 1.0) <= 0.25)
    if _NUMBA_OK:
        # v5.0.0: escape loop parallelo Numba (bit-identico, verificato a
        # runtime dal self-test di _numba_warmup).
        # Cancellazione cooperativa a livello PYTHON (affidabile): le righe
        # sono divise in 16 bande e il contatore di generazione viene
        # controllato tra una banda e l'altra. In-kernel non e' affidabile:
        # Numba/LLVM fa hoisting dei letture di memoria cross-thread dentro
        # prange, quindi un check in-kernel non vedrebbe il bump.
        BANDS = 16
        band = (h + BANDS - 1) // BANDS
        for b0 in range(0, h, band):
            if my_gen != 0 and _GEN[0] != my_gen:
                break  # job obsoleto: stop (il worker scarta il frame)
            _mandel_escape(c, diverged, it, mi, b0, min(b0 + band, h))
    else:
        # Fallback numpy (e riferimento del gate di correttezza).
        # v5.0.0: quadrato complesso in parti esplicite (a*a-b*b, 2*a*b con
        # ufunc singoli, NO FMA) = bit-identico al kernel Numba. NB: np.square
        # su complessi e' compilato da numpy con FMA (re*re - im*im contratto
        # in fma) e darebbe bit diversi (1 ULP su orbite caotiche -> escape
        # time diverso su una piccolissima frazione di pixel di bordo).
        cr = c.real
        ci = c.imag
        tr = ws["tr"]
        ti = ws["ti"]
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(mi):
                if my_gen != 0 and _GEN[0] != my_gen:
                    break  # vista cambiata: il frame verra' scartato
                if not (~diverged).any():
                    break
                # z = z^2 + c, in parti (stesse operazioni e ordine del Numba)
                np.multiply(zr, zr, out=tr)   # tr = zr^2
                np.multiply(zi, zi, out=ti)   # ti = zi^2
                tr -= ti                      # tr = zr^2 - zi^2
                np.multiply(zr, zi, out=ti)   # ti = zr*zi (zr,zi ancora vecchi)
                ti *= 2.0                     # ti = 2*zr*zi
                np.add(tr, cr, out=zr)        # zr = tr + cr
                np.add(ti, ci, out=zi)        # zi = ti + ci
                m = ((zr * zr + zi * zi) > 4.0) & ~diverged
                if not m.any():
                    continue
                diverged |= m
                it[m] = i
    t = np.power(np.clip(it / mi, 0.0, 1.0), 0.35).ravel()
    idx = (t * 255).astype(np.uint8)
    rgb = _LUT[idx].reshape((h, w, 3)).copy()
    rgb[it == 0] = 0
    return Image.fromarray(rgb)

# ---------------- Dispatch backend ----------------
_USE_GPU = _GPU

def backend():
    if _USE_GPU:
        return "CUDA " + _PREC
    return "CPU f64"

def compute(cx, cy, half, w, h, mi, buf=None, prec=None, my_gen=0):
    if _USE_GPU:
        return compute_gpu(cx, cy, half, w, h, mi, buf=buf, prec=prec)
    return compute_cpu(cx, cy, half, w, h, mi, my_gen=my_gen)

# v4.16.0: warmup GPU in background all'avvio: init context CUDA, module load
# e prime allocazioni (buffer device + pinned) pagati FUORI dal primo render
# reale (che senza warmup e' 8-17 ms piu' lento).
def _gpu_warmup():
    try:
        compute_gpu(CX0, CY0, HALF0, 64, 64, 64, prec="f32")
        if _KERNEL_F64 is not None:
            compute_gpu(CX0, CY0, HALF0, 64, 64, 64, prec="f64")
    except Exception:
        pass

if _GPU:
    threading.Thread(target=_gpu_warmup, daemon=True).start()
threading.Thread(target=_numba_warmup, daemon=True).start()

# v5.1.0: il benchmark segue la modalita' corrente (motore+precisione
# selezionati nell'app); per mostrarla nel dialog si usa backend().

class MandelbrotApp:
    # ---------------- Costruzione UI ----------------
    def __init__(self, root):
        self.root = root
        self._setup_fonts()
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.mi = MI0
        self.mi_auto = True
        self.view_file = None
        self.bench = dict(BENCH)
        self._build_toolbar()
        self._build_canvas_status()
        self._build_menu()
        self._bind_events()
        self._start_pipeline()
        self._refresh_title()
        if self.load_config():
            self.request_render("config caricata")
        else:
            self.request_render("iniziale")

    def _setup_fonts(self):
        # UI piu leggibile: ingrandisce il font default di tutti i widget
        # (checkbutton, etichette, pulsanti, menu) preservando la famiglia nativa
        for _fn in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(_fn).config(size=13)
            except Exception:
                pass

    def _build_toolbar(self):
        self.canvas = tk.Canvas(self.root, width=INIT_W, height=INIT_H, bg="black", highlightthickness=0)
        # --- barra comandi in alto ---
        self.ctl = tk.Frame(self.root)
        self.ctl.pack(fill="x")
        bk = tk.Frame(self.ctl)
        bk.pack(side="left", padx=(8, 12))
        tk.Label(bk, text="Motore:").pack(side="left")
        self.cpu_btn = tk.Checkbutton(bk, text="CPU", command=lambda: self.set_backend("cpu"))
        self.cpu_btn.pack(side="left", padx=2, pady=3)
        self.cuda_btn = tk.Checkbutton(bk, text="CUDA", command=lambda: self.set_backend("cuda"))
        self.cuda_btn.pack(side="left", padx=2, pady=3)
        if _GPU:
            self.cuda_btn.select()
        else:
            self.cuda_btn.config(state="disabled")
            self.cpu_btn.select()
        pl = tk.Frame(self.ctl)
        pl.pack(side="left", padx=12)
        tk.Label(pl, text="Palette:").pack(side="left")
        # Pulsanti generati dal registro PALETTES (ordine = indice kernel).
        self.pal_btns = {}
        for _name in PALETTES:
            _b = tk.Checkbutton(pl, text=_name.capitalize(),
                                command=lambda n=_name: self.choose_palette(n))
            _b.pack(side="left", padx=2, pady=3)
            self.pal_btns[_name] = _b
        self.pal_btns["fuoco"].select()
        pc = tk.Frame(self.ctl)
        pc.pack(side="left", padx=12)
        tk.Label(pc, text="Precisione:").pack(side="left")
        self.f32_btn = tk.Checkbutton(pc, text="f32", command=lambda: self.set_precision("f32"))
        self.f32_btn.pack(side="left", padx=2, pady=3)
        self.f64_btn = tk.Checkbutton(pc, text="f64", command=lambda: self.set_precision("f64"))
        self.f64_btn.pack(side="left", padx=2, pady=3)
        self.f32_btn.select()
        if not _GPU:
            self.f32_btn.config(state="disabled")
            self.f64_btn.config(state="disabled")
        elif _KERNEL_F64 is None:
            self.f64_btn.config(state="disabled")
        mif = tk.Frame(self.ctl)
        mif.pack(side="left", padx=12)
        tk.Label(mif, text="Iter:").pack(side="left")
        self.mi_auto_var = tk.BooleanVar(value=True)
        self.auto_btn = tk.Checkbutton(mif, text="Auto", variable=self.mi_auto_var,
                                        command=self.toggle_auto_mi)
        self.auto_btn.pack(side="left", padx=2, pady=3)
        self.btns = tk.Frame(self.root)
        self.btns.pack(fill="x")
        self.mi_caption = tk.Label(self.btns, text="Iterazioni:")
        self.mi_caption.pack(side="left", padx=(8, 2), pady=3)
        self.mi_entry = tk.Entry(self.btns, width=7, justify="right")
        self.mi_entry.pack(side="left", padx=2, pady=3)
        self.mi_entry.bind("<Return>", self._commit_mi_entry)
        self.mi_entry.bind("<FocusOut>", self._commit_mi_entry)
        self._update_mi_label()
        self.mi_minus = tk.Button(self.btns, text="-1000", command=lambda: self.change_mi(-1000))
        self.mi_minus.pack(side="left", padx=2, pady=3)
        self.mi_plus = tk.Button(self.btns, text="+1000", command=lambda: self.change_mi(+1000))
        self.mi_plus.pack(side="left", padx=2, pady=3)
        self.mi_minus.config(state="disabled")
        self.mi_plus.config(state="disabled")
        self.bench_btn = tk.Button(self.btns, text="Benchmark", command=self.run_benchmark)
        self.bench_btn.pack(side="right", padx=(16, 8), pady=3)
        self.reset_btn = tk.Button(self.btns, text="Reset", command=self.reset)
        self.reset_btn.pack(side="right", padx=2, pady=3)

    def _build_canvas_status(self):
        # --- canvas al centro, status in fondo ---
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(self.root, text="render...")
        self.status.pack(fill="x")

    def _build_menu(self):
        self.menu = tk.Menu(self.root)
        self.mfile = tk.Menu(self.menu, tearoff=0)
        self.mfile.add_command(label="Salva immagine... (Ctrl+S)", command=self.save_png)
        self.mfile.add_command(label="Carica zona...", command=self.load_zone_as)
        self.mfile.add_command(label="Salva zona", command=self.save_zone)
        self.mfile.add_command(label="Salva zona con nome...", command=self.save_zone_as)
        self.mfile.add_separator()
        self.mfile.add_command(label="Esci", command=self.on_exit)
        self.menu.add_cascade(label="File", menu=self.mfile)
        self.root.config(menu=self.menu)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _bind_events(self):
        self.press_pos = None
        self.dragged = False
        self._size = (0, 0)
        self.root.bind("<Control-s>", lambda e: self.save_png())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Configure>", self.on_configure)
        self.canvas.bind("<Key-r>", lambda e: self.reset())
        self.canvas.bind("<Key-plus>", lambda e: self.zoom_center(2.0))
        self.canvas.bind("<Key-minus>", lambda e: self.zoom_center(0.5))

    # ---------------- Helper UI ----------------
    def _refresh_title(self):
        t = f"Insieme di Mandelbrot v{VERSION} - {backend()}"
        if self.view_file:
            t += " - " + os.path.basename(self.view_file)
        self.root.title(t)

    def _select_palette(self, name):
        apply_palette(name)
        name = _PALETTE
        for b in self.pal_btns.values():
            b.deselect()
        self.pal_btns[name].select()

    def _select_backend(self, be):
        global _USE_GPU
        if be == "cuda" and not _GPU:
            return False
        _USE_GPU = (be == "cuda")
        self.cpu_btn.deselect()
        self.cuda_btn.deselect()
        (self.cpu_btn if be == "cpu" else self.cuda_btn).select()
        return True

    def _select_precision(self, p):
        if p == "f64" and _KERNEL_F64 is None:
            return False
        if not set_prec(p):
            return False
        self.f32_btn.deselect()
        self.f64_btn.deselect()
        (self.f32_btn if p == "f32" else self.f64_btn).select()
        return True

    def _update_mi_label(self):
        self.mi_caption.config(text="Iterazioni (auto):" if self.mi_auto else "Iterazioni:")
        # ATTENZIONE: in Tkinter delete/insert su Entry disabilitato sono no-op
        # silenziosi -> riabilitare PRIMA di scrivere, poi disabilitare di nuovo
        # (il widget arriva gia' disabilitato dalla chiamata precedente).
        self.mi_entry.config(state="normal")
        self.mi_entry.delete(0, "end")
        self.mi_entry.insert(0, str(self.eff_mi()))
        if self.mi_auto:
            self.mi_entry.config(state="disabled")

    # ---------------- Vista e interazione (geometria + mouse/tastiera) ----------------
    def canvas_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < MIN_DIM or h < MIN_DIM:
            w, h = INIT_W, INIT_H
        return w, h

    def p2c(self, px, py):
        w, h = self.canvas_size()
        return (self.cx + (px - w / 2) / (w / 2) * self.half,
                self.cy + (py - h / 2) / (h / 2) * self.half * (h / w))

    def zoom_at(self, ux, uy, f):
        self.cx += (ux - self.cx) * (1 - 1 / f)
        self.cy += (uy - self.cy) * (1 - 1 / f)
        self.half = max(self.half / f, MIN_HALF)
        self.request_render("zoom")

    def zoom_center(self, f):
        self.zoom_at(self.cx, self.cy, f)

    def on_press(self, e):
        self.press_pos = (e.x, e.y)
        self.dragged = False

    def on_drag(self, e):
        if self.press_pos is None:
            return
        x0, y0 = self.press_pos
        if abs(e.x - x0) < 4 and abs(e.y - y0) < 4:
            return
        w, h = self.canvas_size()
        self.cx -= (e.x - x0) / (w / 2) * self.half
        self.cy -= (e.y - y0) / (h / 2) * self.half * (h / w)
        self.press_pos = (e.x, e.y)
        self.dragged = True
        self.request_render("pan")

    def on_release(self, e):
        if self.press_pos is not None and not self.dragged:
            self.zoom_at(*self.p2c(e.x, e.y), 2.0)
        self.press_pos = None

    def on_wheel(self, e):
        f = 1.25 if e.delta > 0 else 0.8
        self.zoom_at(*self.p2c(e.x, e.y), f)

    def on_configure(self, e):
        if (e.width, e.height) == self._size:
            return
        self._size = (e.width, e.height)
        if e.width < MIN_DIM or e.height < MIN_DIM:
            return
        self.request_render("ridimensionata")

    # ---------------- Controlli (MI, motore, palette, precisione, reset) ----------------
    def eff_mi(self):
        if not self.mi_auto:
            return self.mi
        return auto_mi(self.half)

    def toggle_auto_mi(self):
        new_auto = self.mi_auto_var.get()
        if new_auto:
            self.mi_auto = True
        else:
            # Disattivando l'auto: congela self.mi sul valore auto corrente
            # (eff_mi calcolato con mi_auto ancora True), invece di mostrare
            # il valore fisso iniziale (stale).
            self.mi = self.eff_mi()
            self.mi_auto = False
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self._update_mi_label()
        self.request_render("MI: auto" if self.mi_auto else "MI: fissi")

    def change_mi(self, d):
        self.mi = max(50, self.mi + d)
        self._update_mi_label()
        self.request_render("iterazioni modificate")

    def _commit_mi_entry(self, e=None):
        if self.mi_auto:
            return
        txt = self.mi_entry.get().strip()
        try:
            v = int(txt)
        except ValueError:
            v = 0
        if v < 50 or v > 100000:
            self.status.config(text="MI invalidi: intero tra 50 e 100000")
            self.mi_entry.delete(0, "end")
            self.mi_entry.insert(0, str(self.mi))
            return
        if v == self.mi:
            return
        self.mi = v
        self.request_render("iterazioni modificate")

    def set_backend(self, b):
        if not self._select_backend(b):
            return
        self._refresh_title()
        self.request_render("motore: " + b.upper())

    def choose_palette(self, name):
        self._select_palette(name)
        self.request_render("palette: " + _PALETTE)

    def set_precision(self, p):
        if not self._select_precision(p):
            return
        self._refresh_title()
        self.request_render("precisione: " + p)

    def reset(self):
        self.cx, self.cy, self.half = CX0, CY0, HALF0
        self.mi = MI0
        self.mi_auto = True
        self.mi_auto_var.set(True)
        self.mi_minus.config(state="disabled")
        self.mi_plus.config(state="disabled")
        self._select_palette("fuoco")
        self._select_backend("cuda" if _GPU else "cpu")
        self._select_precision("f32")
        self._refresh_title()
        self._cfg_dirty = True
        self._update_mi_label()
        self.request_render("reset totale")

    # ---------------- Pipeline rendering (asincrona latest-wins) ----------------
    def _start_pipeline(self):
        # pipeline asincrona latest-wins (worker + Condition, niente blocco UI)
        self._cv = threading.Condition()
        self._job = None
        self._gen = 0
        self._frames = queue.Queue()
        self._last_msg = ""
        self._full_timer = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._cfg_dirty = False
        self._bench_running = False
        self._bench_result = None
        self._bench_finished = False
        self.root.after(30, self._poll)
        self.root.after(1000, self._flush_config)

    def request_render(self, msg):
        self._last_msg = msg
        self._cfg_dirty = True
        self._update_mi_label()
        if self._full_timer is not None:
            self.root.after_cancel(self._full_timer)
            self._full_timer = None
        w, h = self.canvas_size()
        view = (self.cx, self.cy, self.half, self.eff_mi())
        self._submit(view, max(w // 4, 16), max(h // 4, 16))
        self._full_timer = self.root.after(500, lambda: self._maybe_full(view))

    def _maybe_full(self, view):
        self._full_timer = None
        if (self.cx, self.cy, self.half, self.eff_mi()) == view:
            w, h = self.canvas_size()
            self._submit(view, w, h)

    def _submit(self, view, w, h):
        # v5.0.0: ogni nuovo job e' una nuova "generazione"; il render CPU in
        # corso (se obsoleto) se ne accorge a fine riga e si ferma (il frame
        # verra' scartato dal worker).
        self._gen += 1
        _GEN[0] = self._gen
        with self._cv:
            self._job = (view, w, h, self._gen)
            self._cv.notify()

    def _worker_loop(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job = self._job
                self._job = None
            view, w, h, gen = job
            try:
                t0 = time.perf_counter()
                img = compute(view[0], view[1], view[2], w, h, view[3], my_gen=gen)
                rt = time.perf_counter() - t0
            except Exception:
                continue
            if _GEN[0] != gen:
                continue  # obsoleto (vista cambiata): scarto il frame parziale
            self._frames.put((img, self._last_msg, rt))

    def _poll(self):
        frame = None
        try:
            while True:
                frame = self._frames.get_nowait()
        except queue.Empty:
            pass
        if frame is not None:
            self._show(frame[0], frame[1], frame[2])
        if self._bench_finished:
            self._bench_finished = False
            count, secs, err = self._bench_result
            self._bench_done(count, secs, err)
        self.root.after(30, self._poll)

    def _show(self, img, msg, rt=0.0):
        w, h = self.canvas_size()
        if img.size != (w, h):
            img = img.resize((w, h), Image.NEAREST)
        self.pil = img
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.photo)
        self.status.config(text=f"{msg} | {backend()} | palette: {_PALETTE} | render: {rt*1000:.0f} ms")

    # ---------------- File: PNG, zona (JSON), config ----------------
    def save_png(self):
        if getattr(self, "pil", None) is None:
            self.status.config(text="niente immagine da salvare")
            return
        default = "mandelbrot_" + time.strftime("%Y%m%d_%H%M%S") + ".png"
        path = tk.filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".png", initialfile=default,
            filetypes=[("Immagini PNG", "*.png")])
        if not path:
            return
        try:
            self.pil.save(path, "PNG")
            self.status.config(text="salvata: " + path)
        except Exception as ex:
            self.status.config(text="errore salvataggio: " + str(ex))

    def save_zone(self):
        if self.view_file:
            self._save_zone_to(self.view_file)
        else:
            self.save_zone_as()

    def save_zone_as(self):
        default = "mandelbrot_" + time.strftime("%Y%m%d_%H%M%S") + ".json"
        path = tk.filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".json", initialfile=default,
            filetypes=[("File JSON", "*.json"), ("Tutti i file", "*.*")])
        if not path:
            return
        self._save_zone_to(path)

    def _save_zone_to(self, path):
        c = {
            "app": "mandelbrot",
            "versione": VERSION,
            "cx": self.cx,
            "cy": self.cy,
            "half": self.half,
            "mi": self.mi,
            "mi_auto": self.mi_auto,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=2, ensure_ascii=False)
                f.write("\n")
            self.view_file = path
            self._refresh_title()
            self.status.config(text="zona salvata: " + path)
        except Exception as ex:
            self.status.config(text="errore salvataggio zona: " + str(ex))

    def load_zone_as(self):
        path = tk.filedialog.askopenfilename(
            parent=self.root, title="Carica zona",
            filetypes=[("File JSON", "*.json"), ("Tutti i file", "*.*")])
        if not path:
            return
        self._load_zone_from(path)

    def _load_zone_from(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                c = json.load(f)
            self.cx = float(c["cx"])
            self.cy = float(c["cy"])
            self.half = max(float(c["half"]), MIN_HALF)
            self.mi = int(c.get("mi", self.mi))
            self.mi_auto = bool(c.get("mi_auto", self.mi_auto))
        except Exception as ex:
            self.status.config(text="errore caricamento zona: " + str(ex))
            return
        self.mi_auto_var.set(self.mi_auto)
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self.view_file = path
        self._refresh_title()
        self.request_render("zona caricata: " + os.path.basename(path))

    def save_config(self):
        c = dict(cx=self.cx, cy=self.cy, half=self.half,
                 mi=self.mi, mi_auto=bool(self.mi_auto),
                 precision=_PREC, palette=_PALETTE,
                 backend=("cuda" if _USE_GPU else "cpu"),
                 bench=dict(self.bench))
        # NB: view_file NON e' persistito (v5.1.1): all'avvio il programma deve
        # partire senza file corrente, cosi' 'Salva zona' chiede sempre il nome.
        try:
            d = os.path.dirname(CONFIG_PATH)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=2)
        except Exception:
            pass

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return False
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                c = json.load(f)
        except Exception:
            return False
        self.cx = float(c.get("cx", self.cx))
        self.cy = float(c.get("cy", self.cy))
        self.half = float(c.get("half", self.half))
        self.mi = int(c.get("mi", self.mi))
        self.mi_auto = bool(c.get("mi_auto", self.mi_auto))
        self._load_bench(c.get("bench"))
        # NB: view_file NON viene ripristinato (v5.1.1): all'avvio il programma
        # parte senza file corrente (eventuali vecchi 'view_file' nella config
        # esistente sono ignorati); 'Salva zona' chiede sempre il nome finche'
        # l'utente non carica/salva esplicitamente una zona in questa sessione.
        self.mi_auto_var.set(self.mi_auto)
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self._select_precision(c.get("precision", "f32"))
        self._select_palette(c.get("palette", "fuoco"))
        be = c.get("backend", "cuda" if _GPU else "cpu")
        if be == "cuda" and not _GPU:
            be = "cpu"
        self._select_backend(be)
        self._refresh_title()
        self._update_mi_label()
        return True

    def _load_bench(self, b):
        if not isinstance(b, dict):
            return
        self.bench = dict(BENCH)
        self.bench["cx"] = float(b.get("cx", BENCH["cx"]))
        self.bench["cy"] = float(b.get("cy", BENCH["cy"]))
        self.bench["half"] = float(b.get("half", BENCH["half"]))
        # 'mi' non si carica piu': e' derivato da auto_mi(bench['half'])
        self.bench["w"] = int(b.get("w", BENCH["w"]))
        self.bench["h"] = int(b.get("h", BENCH["h"]))
        self.bench["secs"] = float(b.get("secs", BENCH["secs"]))

    def _flush_config(self):
        if self._cfg_dirty:
            self._cfg_dirty = False
            self.save_config()
        self.root.after(1000, self._flush_config)

    def on_exit(self):
        self._cfg_dirty = False
        try:
            self.save_config()
        except Exception:
            pass
        self.root.destroy()

    # ---------------- Benchmark ----------------
    def _bench_rows(self):
        b = self.bench
        mi = auto_mi(b["half"])
        return [
            ("Regione", f"c = ({b['cx']}, {b['cy']}i)"),
            ("Met\u00e0 lato", f"{b['half']:.3e}"),
            ("Iterazioni", f"{mi}\u00a0 (formula auto)"),
            ("Risoluzione", f"{b['w']} \u00d7 {b['h']} px"),
            ("Motore", f"{backend()} (corrente)"),
            ("Durata", f"{b['secs']:.0f} s"),
        ]

    def _modal(self, win):
        """Rende 'win' modale e centrato su self.root; blocca fino a chiusura.
        Ritorna True se la finestra e' stata chiusa normalmente (distrutta),
        False se annullata (X, Esc o pulsante 'Annulla')."""
        def cancel(_e=None):
            win._annullato = True
            win.destroy()
        win._annullato = False
        win.protocol("WM_DELETE_WINDOW", cancel)
        win.bind("<Escape>", cancel)
        win.transient(self.root)
        win.grab_set()
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - w) // 2
        y = self.root.winfo_rooty() + max((self.root.winfo_height() - h) // 3, 0)
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.root.wait_window(win)
        self.root.focus_force()
        return not win._annullato

    def _bench_ask(self):
        """Dialog di conferma personalizzato: True se avviare."""
        win = tk.Toplevel(self.root)
        win.title("Benchmark Mandelbrot")
        win.resizable(False, False)
        body = tk.Frame(win, padx=26, pady=20)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Benchmark standardizzato",
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(body, text="Parametri del test (regione e durata comparabili tra versioni):",
                 foreground="#555").pack(anchor="w", pady=(2, 12))
        rows = tk.Frame(body)
        rows.pack(fill="x")
        for i, (k, v) in enumerate(self._bench_rows()):
            tk.Label(rows, text=k, width=12, anchor="w",
                     foreground="#444").grid(row=i, column=0, sticky="w", pady=3)
            tk.Label(rows, text=v, anchor="e",
                     font=("Consolas", 12)).grid(row=i, column=1, sticky="e",
                                                  padx=(18, 0), pady=3)
        btns = tk.Frame(win)
        btns.pack(fill="x", padx=26, pady=(16, 18))

        def go(_e=None):
            win._annullato = False
            win.destroy()
        annulla = tk.Button(btns, text="Annulla")
        annulla.pack(side="right")
        avvia = tk.Button(btns, text="Avvia", command=go)
        avvia.pack(side="right", padx=(0, 10))
        win.bind("<Return>", go)
        avvia.focus_set()
        return self._modal(win)

    def _bench_result_dialog(self, count, secs, err):
        """Dialog risultato: rendering/s grande e evidente come vero risultato."""
        win = tk.Toplevel(self.root)
        win.title("Benchmark \u2014 risultato")
        win.resizable(False, False)
        body = tk.Frame(win, padx=30, pady=22)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Risultato",
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        if count > 0:
            tk.Frame(body, bg="#e6f4ea", height=1).pack(fill="x", pady=(10, 0))
            tk.Label(body, text=f"{count/secs:.2f}",
                     font=("Segoe UI", 42, "bold"),
                     foreground="#0a7d33").pack(pady=(16, 0))
            tk.Label(body, text="rendering / secondo",
                     font=("Segoe UI", 13, "bold"),
                     foreground="#0a7d33").pack(pady=(0, 8))
            tk.Label(body, text=f"{count} rendering in {secs:.1f} s   \u00b7   {secs/count*1000:.0f} ms ciascuno",
                     foreground="#555").pack(pady=(0, 16))
        else:
            tk.Label(body, text="BENCHMARK FALLITO",
                     font=("Segoe UI", 20, "bold"),
                     foreground="#b00020").pack(pady=(16, 6))
            tk.Label(body, text=str(err), foreground="#b00020",
                     wraplength=420, justify="left").pack(anchor="w", pady=(0, 16))
        tk.Label(body, text="Parametri del test:",
                 foreground="#555").pack(anchor="w", pady=(0, 4))
        rows = tk.Frame(body)
        rows.pack(fill="x", anchor="w")
        for i, (k, v) in enumerate(self._bench_rows()):
            tk.Label(rows, text=k, width=12, anchor="w",
                     foreground="#444").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(rows, text=v, anchor="e",
                     font=("Consolas", 11)).grid(row=i, column=1, sticky="e",
                                                  padx=(18, 0), pady=2)
        def chiudi(_e=None):
            win._annullato = False
            win.destroy()
        tk.Button(win, text="Chiudi", command=chiudi).pack(pady=(18, 20))
        win.bind("<Return>", chiudi)
        self._modal(win)

    def run_benchmark(self):
        if getattr(self, "_bench_running", False):
            self.status.config(text="benchmark gia' in corso")
            return
        if not self._bench_ask():
            self.status.config(text="benchmark annullato")
            return
        self._bench_running = True
        self.status.config(text=f"benchmark in corso ({self.bench['secs']:.0f} s)...")
        threading.Thread(target=self._bench_worker, daemon=True).start()

    def _bench_worker(self):
        b = self.bench
        mi = auto_mi(b["half"])
        need = b["w"] * b["h"] * 3
        bench_buf = None
        if _USE_GPU:
            try:
                import cupy as cp
                bench_buf = cp.empty((need,), dtype=cp.uint8)
            except Exception:
                bench_buf = None
        def render():
            # v5.1.0: modalita' CORRENTE (motore+precisione selezionati), non
            # piu' CUDA f32 fisso. compute() dispatcha su _USE_GPU e usa
            # _PREC (GPU) / f64 (CPU); my_gen=0 -> nessuna cancellazione.
            return compute(b["cx"], b["cy"], b["half"], b["w"], b["h"], mi,
                           buf=bench_buf)
        t_end = time.perf_counter() + b["secs"]
        count = 0
        err = None
        while time.perf_counter() < t_end:
            try:
                render()
                count += 1
            except Exception as ex:
                err = str(ex)
                break
        # thread-safe: il thread principale (in _poll) rileva il flag e mostra il risultato
        self._bench_result = (count, b["secs"], err)
        self._bench_finished = True

    def _bench_done(self, count, secs, err):
        self._bench_running = False
        self.status.config(text="benchmark completato")
        self._bench_result_dialog(count, secs, err)


def main():
    root = tk.Tk()
    MandelbrotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


