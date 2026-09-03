# ============================================================================
# Insieme di Mandelbrot - visualizzatore interattivo
# VERSIONE: 5.10.0
# ----------------------------------------------------------------------------
# REGOLA: ogni modifica incrementa la versione e aggiunge una voce qui sotto
# (formato: versione - data - descrizione modifiche).
#
# STORICO:
# 5.10.0 - 2026-09-03
#   - Foto sdoppiata: "Foto 2x2" e "Foto 4x4" (fattore N parametrico in
#     take_photo/_photo_worker/_photo_done; 4x4 = 16x pixel, molto piu'
#     lento). Nuovo pulsante "Ricalcola" che rifa il rendering della
#     vista corrente.
# 5.9.8 - 2026-09-03
#   - Zoom-out su macOS: tasti + / - / r legati alla finestra (prima solo
#     col focus sul canvas, dopo un click in toolbar sembravano morti;
#     guardia per non rubarli alle Entry) e click destro (due dita sul
#     trackpad) = zoom x0.5 al cursore, senza dipendere dalla rotella.
# 5.9.7 - 2026-09-03
#   - Bugfix Foto: typo self._photo_btn (attributo inesistente) faceva
#     fallire take_photo prima dell'hourglass: il pulsante non faceva
#     nulla. Rinominato in self.photo_btn (creato in _build_toolbar).
#     Inoltre la foto ora invalida i render interattivi in volo/pendenti
#     (bump _GEN + cancella full-timer) e lo scarto stantio confronta
#     vista+palette+motore+precisione invece della generazione.
# 5.9.6 - 2026-09-03
#   - Pulsante Foto: ricalcola la vista a 2x per lato (4x pixel) e la
#     mostra con antialiasing (media 2x2 su RGB, uguale per tutti i
#     backend, in background con cursore hourglass); foto scartata se
#     la vista cambia durante il calcolo. Ctrl+S la salva.
# 5.9.5 - 2026-09-03
#   - Diagnostica single-core: se la CPU non va in multi-core, ORA SI VEDE
#     IL PERCHE'. _numba_warmup() registra esito/motivo/tempo in
#     _NUMBA_STATUS (prima gli except erano muti); status bar col suffisso
#     "single-core: motivo in Help > Informazioni" in rosso finche' resta
#     il fallback; dialog Informazioni con blocco CPU/Numba dettagliato.
# 5.9.4 - 2026-09-03
#   - UI: finestra default allargata a 1280x720 (sempre 16:9): la toolbar
#     cresciuta (dropdown GPU) forzava la finestra oltre i 960px e il canvas
#     si stirava perdendo il 16:9. Benchmark invariato a 960x540.
# 5.9.3 - 2026-09-03
#   - Benchmark: ref 4070 Super Vulkan corretto 177.0 -> 124 rendering/s
#     (misurato in-app sul vero adapter; 177 era col bug pre-5.9.2 che
#     girava tutto sulla 5070 Ti).
# 5.9.2 - 2026-09-03
#   - Bugfix: il dropdown GPU pilotava solo CUDA; Vulkan usava sempre
#     l'adapter high-performance (stesso per 4070/5070 Ti -> benchmark
#     quasi uguali). Ora enumera gli adapter fisici (backend Vulkan) e il
#     dropdown mostra quelli attivi per il motore corrente (CUDA/Vulkan);
#     VulkanBackend.select_adapter() ricrea le risorse device-bound sotto
#     lock. Selezione Vulkan persistita in config ("vulkan_adapter").
# 5.9.1 - 2026-09-03
#   - Bugfix benchmark su GPU display (es. 5070 Ti che pilota il desktop):
#     durante gli 8 s il worker sospende i render interattivi (niente
#     contesa col thread bench sullo stesso device: un kernel pesante in
#     vista + benchmark rischiava TDR/reset -> cudaErrorDevicesUnavailable
#     al primo render). Se il device risulta comunque occupato/in reset,
#     il dialog FALLITO aggiunge il suggerimento di riprovare o riavviare.
#   - Riferimenti storici ridefiniti sulle GPU locali (5 voci): AMD 9900X
#     6.62, 4070 Super Vulkan 177.0, 5070 Ti Vulkan 179.0,
#     4070 Super CUDA 250.0, 5070 Ti CUDA 350.0 rendering/s; canvas del
#     grafico ad altezza dinamica (150px fissi insufficienti per 6 barre).
# 5.9.0 - 2026-09-03
#   - GPU multipla: se CuPy rileva > 1 GPU CUDA, la toolbar mostra un dropdown
#     "GPU:" (indice + nome) per scegliere il device di render/benchmark.
#     Selezione persistita in config ("cuda_device"), titolo/stato seguono
#     il device scelto (hw_name sul device attivo). Il device CuPy e'
#     per-thread: render, benchmark e warmup usano
#     'with cp.cuda.Device(_CUDA_DEV)'; cambio device invalida _BUF e scalda
#     il nuovo device in background (64x64) se CUDA e' attivo.
# 5.8.13 - 2026-09-03
#   - UI: un po' di colore (toni medi, leggibili su chiaro/scuro): status bar
#     nel colore del motore attivo, barra accento da 3px sopra il canvas,
#     titoli dei dialog Help colorati, errori in rosso. SOLO Label/Frame:
#     Button/Checkbutton/Radiobutton restano nativi (lezione 5.4.2: forzarne
#     i colori su macOS li degrada a flat illeggibili).
# 5.8.12 - 2026-09-03
#   - Menu Help: nuova voce "Novità recenti..." che mostra le ultime 10
#     modifiche di versione (da HISTORY, tupla embedded: i commenti STORICO
#     non sopravvivono alla build PyInstaller). Ad ogni bump, HISTORY va
#     aggiornata insieme allo STORICO (voce nuova in testa, max 10).
# 5.8.11 - 2026-09-03
#   - Menu Help con due voci separate: "Istruzioni..." (guida rapida a
#     navigazione, motore/precisione, iterazioni, palette, zone, benchmark)
#     e "Informazioni..." (versione, backend/hardware attivi, autore
#     Francesco Ferrara <occhiobello@gmail.com>). Dialog Toplevel modali
#     centrati sullo stile di quelli del benchmark (via _modal).
# 5.8.10 - 2026-09-04
#   - UI: backend() CPU mostra sempre single/multi-core per la precisione
#     corrente ("CPU f32 multi-core" col kernel Numba parallelo,
#     "CPU f32 single-core (numpy)" col fallback). Prima mostrava "(numpy)"
#     solo a import Numba fallito, ma il fallback scatta per precisione
#     (_NUMBA_OK[prec]) e l'etichetta poteva mentire. Il titolo si aggiorna
#     a ogni frame (_show) cosi' segue il warmup single->multi.
# 5.8.9 - 2026-09-03
#   - Build: regola di ritenzione dist/ ora tiene le ultime KEEP_N versioni
#     PER PIATTAFORMA (win64.zip e macos.dmg separatamente, max 3 ciascuna).
#     Prima teneva 3 totali con vincolo min-1/piattaforma.
# 5.8.8 - 2026-09-02
#   - Bugfix: hw_name() su Metal restituiva il repr del native-selector pyobjc
#     ("<native-selector name of <AGXG13GDevice...>...") invece del nome GPU
#     (es. "Apple M1"). Ora se dev.name e' callable (selector) viene invocato.
# 5.8.7 - 2026-09-02
#   - Build: regola di ritenzione dist/ garantisce ora SEMPRE almeno un artefatto
#     per piattaforma (win64.zip + macos.dmg), anche se supera KEEP_N. Prima
#     keep_last_n_dist(3) poteva azzerare una piattaforma se l'altra aveva >= 3
#     versioni. Documentata in AGENTS.md/spec.md.
# 5.8.6 - 2026-09-02
#   - Build macOS: make_icon.py genera ora mandelbrot.icns via Pillow (formato ICNS,
#     multi-size 16-1024) al posto di sips, che fallisce su macOS 26.x (error 13).
# 5.8.5 - 2026-09-02
#   - Build: regola generale di ritenzione artefatti. build_app.py a fine build tiene
#     solo le ultime KEEP_N (3) versioni di zip (win) / .dmg (mac) in dist/ (NAS) e
#     rimuove automaticamente quelle piu' vecchie (ordinati per X.Y.Z). Documentata in
#     AGENTS.md/spec.md; sistemati i riferimenti stanchi a build_app.sh/.ps1 (assenti)
#     nella sezione build di spec.md.
# 5.8.4 - 2026-09-02
#   - Build macOS: make_icon.py genera ora anche mandelbrot.icns (solo darwin) da
#     icon_src.png via sips; build_app.py la verifica prima di PyInstaller. Prima
#     l'icns NON veniva prodotto e la build .app falliva (mandelbrot.spec ha
#     icon="mandelbrot.icns" nel BUNDLE). Allineati AGENTS.md/spec.md alla realta'
#     (app su disco di sistema, zip win / dmg mac sul NAS, tolto riferimenti ai
#     build_app.sh/.ps1 assenti).
# 5.8.3 - 2026-09-02
#   - Build: intermedie (build/, workpath) E output app (dist/, distpath)
#     vanno ora sul DISCO DI SISTEMA (temp utente, via tempfile.gettempdir():
#     mandelbrot_build / mandelbrot_dist) invece che nel progetto, che puo'
#     stare su una share di rete dove la build e' lenta. Implementato in
#     build_app.py (--workpath + --distpath). Sul progetto/NAS resta SOLO
#     l'artefatto distributivo: su Windows la zip Mandelbrot-v<ver>-win64.zip,
#     su macOS il .dmg Mandelbrot-v<ver>-macos.dmg (staging app + link
#     /Applications in temp, hdiutil UDZO). Prima sia l'app (dist/Mandelbrot[.app])
#     sia l'archivio finivano nel progetto.
# 5.8.2 - 2026-09-02
#   - UI: durante l'esecuzione del benchmark il cursore della finestra
#     principale passa a 'watch' (sabbia/pallina = occupato) e torna alla
#     freccia a terminazione. Prima, mentre il benchmark girava in un thread
#     dedicato, il cursore restava 'arrow' e non segnava l'occupazione.
#     Impostato in run_benchmark(), ripristinato in _bench_done() (unico punto
#     di uscita, anche su errore).
# 5.8.1 - 2026-09-02
#   - Benchmark: nel grafico a barre il nome hardware NON compare piu'
#     nell'etichetta della run corrente, che riporta ora solo il metodo attivo
#     (es. "CUDA f32 - questa run"); il nome hardware resta visibile sotto,
#     nella griglia "Parametri del test" (riga "Hardware", introdotta in 5.8.0).
#     I 3 riferimenti storici (BENCH_REF) mantengono il proprio nome
#     (9900X CPU, 5070 Ti Vulkan, 5070 Ti CUDA) per distinguerli.
# 5.8.0 - 2026-09-02
#   - Nome dell'hardware attivo (hw_name(), nessuna dipendenza nuova: CPU dal
#     registro Windows / 'sysctl' su macOS; GPU da CuPy/pyobjc/wgpu) ora
#     visibile nella barra di stato dopo ogni render e nel titolo finestra.
#   - Benchmark: l'etichetta della run corrente nel grafico a barre riporta
#     anche il nome hardware accorciato (es. "GeForce RTX 5070 Ti (CUDA f32)
#     - questa run"; prefisso vendor tolto, margine sinistro allargato) e la
#     griglia dei parametri ha una riga "Hardware" col nome completo.
# 5.7.0 - 2026-09-02
#   - Benchmark: il dialog di risultato mostra ora anche un GRAFICO A BARE
#     ORIZZONTALI (tk.Canvas, nessuna dipendenza nuova) che confronta la run
#     corrente coi 3 riferimenti storici in BENCH_REF (6.62 9900X CPU,
#     177 5070 Ti Vulkan, 250 5070 Ti CUDA, rendering/s). La barra della run
#     corrente e' evidenziata in verde e l'etichetta riporta il METODO attivo
#     (es. "CUDA f32 - questa run"); le barre storiche sono grigie.
#   - Scala orizzontale LINEARE AUTOMATICA: l'asse va da 0 al massimo tra i
#     valori (se la run corrente supera i riferimenti l'asse si adatta), step
#     "nice" della serie 1/2/5x10^n con griglia verticale ed etichette; il
#     valore e' scritto a fine barra, cosi' resta leggibile anche la barra
#     minuscola della CPU (6.62).
# 5.6.1 - 2026-09-02
#   - Avviso Numba assente: quando Numba non e' installato (CPU -> fallback
#     numpy single-core) l'etichetta del backend mostra "CPU f32/f64 (numpy)"
#     (titolo + barra di stato), prima era silenziosa. Nuovo flag
#     _NUMBA_AVAILABLE (njit non None) usato da backend().
# 5.6.0 - 2026-09-02
#   - NUOVO BACKEND GPU VULKAN (wgpu / wgpu-native, cross-platform): quarto
#     motore selezionabile accanto a CPU, CUDA (NVIDIA) e Metal (Apple
#     Silicon), per usare GPU AMD/NVIDIA/Intel anche dove CuPy/PyOpenGL non
#     arrivano (es. iGPU AMD su Windows). f32-ONLY (come Metal: la f64 su GPU
#     consumer e' lenta e qui inutile). Rilevato all'avvio: se wgpu e'
#     installato e c'e' un adapter GPU, il backend e' disponibile; altrimenti
#     l'app degrada a CPU/CUDA/Metal come prima.
#   - UI "Motore": i due checkbutton CPU/GPU diventano 4 RADIO (CPU, CUDA,
#     Metal, Vulkan); i backend non disponibili restano visibili ma
#     DISABILITATI (grigi). Il backend attivo e' persistito in config per
#     nome ("cpu"/"cuda"/"metal"/"vulkan"); i vecchi valori ("gpu") migrano al
#     default GPU.
#   - Kernel WGSL (compute shader): stesso criterio di escape, interior
#     analitico e coloring (smooth iteration) di CUDA/MSL/CPU -> stessa
#     immagine a parte 1-qualche ULP sul bordo caotico (limitazione intrinseca
#     di f32). Vincoli WGSL (vs MSL/CUDA): nessun tipo u8 -> LUT e output sono
#     array<u32> con colori packed 0x00RRGGBB (1 px = 1 u32), unpackati in
#     numpy (h,w,3) al readback; parametri in due buffer uniform (vec4<f32>
#     cx,cy,half e vec4<i32> w,h,mi,pal); LUT = tutte le palette concatenate
#     in un buffer storage (selezione per indice pal, come CUDA/MSL);
#     min/max/pow/sqrt/log/log2 sono i built-in WGSL (senza suffisso 'f').
#   - Dispatch: lo slot GPU "generico" unico e' ORA una scelta tra 4 backend
#     (CPU/CUDA/Metal/Vulkan) selezionabile a runtime. _ACTIVE (stringa) e'
#     l'unico stato del motore; il default e' il primo GPU disponibile in
#     ordine CUDA > Metal > Vulkan, altrimenti CPU. f64 resta solo su CPU e
#     su CUDA con kernel f64; su Metal/Vulkan il bottone f64 e' disabilitato
#     (stato dinamico, come prima).
#   - Build: mandelbrot.spec raccoglie wgpu (collect_all) su Windows/macOS;
#     l'app resta self-contained (wgpu-native lib bundle dentro la wheel).
# 5.5.0 - 2026-09-01
#   - Build dell'app multipiattaforma: aggiunta la build Windows (one-dir)
#     dist/Mandelbrot/Mandelbrot.exe con GPU CUDA (CuPy) incluso ma runtime
#     CUDA NON bundled: la GPU funziona solo se l'utente installa driver
#     NVIDIA + runtime CUDA (che CuPy trova via cuda-pathfinder da CUDA_PATH/
#     PATH/Program Files); la CPU (Numba/numpy) e' sempre disponibile.
#     Self-contained senza dipendenze di Python/librerie.
#     La build macOS (Mandelbrot.app, GPU Metal) resta invariata.
#     - build_app.py: script di build unificato (icona -> PyInstaller -> post),
#       ramificato su sys.platform; build_app.sh ora lo redirect; nuovo
#       build_app.ps1 per Windows.
#     - mandelbrot.spec: riscritto multipiattaforma (rami darwin/win32).
#       win32: collect_all(cupy) (solo moduli, NO DLL CUDA), icona
#       mandelbrot.ico, versione EXE (versioninfo), runtime hook hook_dlldir.py
#       (os.add_dll_directory su EXE + _internal, difensivo); darwin: invariato
#       (pyobjc/Metal, libomp da torch, BUNDLE .app, firma ad-hoc).
#     - make_icon.py: oltre a icon_src.png genera mandelbrot.ico (multi-size
#       16..256) via Pillow; palette dell'icona invariata ('termal').
#     - spec.md/AGENTS.md: sezione 'Build dell'app' aggiornata per entrambe
#       le piattaforme.
# 5.4.5 - 2026-08-31
#   - Icona app: ripristinata la palette 'termal' (la 5.4.4 l'aveva portata a
#     'fuoco'; su richiesta torna 'termal'). make_icon.py usa di nuovo
#     apply_palette('termal').
#   - Menu File "Salva zona": ora DISABILITATO finche' non esiste un nome/file
#     di zona (view_file=None, es. all'avvio); si abilita dopo "Salva zona con
#     nome..." o "Carica zona...". Prima faceva fallback a 'chiedi nome'; ora
#     l'utente deve definire esplicitamente il primo nome con "Salva zona con
#     nome...". Aggiunto MandelbrotApp._update_save_zone_state() che gestisce
#     lo stato dell'entry menu, chiamato all'init e dopo salvataggio/caricamento.
# 5.4.4 - 2026-08-31
#   - Icona app: palette cambiata da 'termal' (che parte col ghiaccio, percio'
#     l'icona risultava 'ghiacciata') a 'fuoco'. make_icon.py usa ora
#     apply_palette('fuoco'), coerente con la palette di default dell'app.
#     Nessuna modifica al rendering/program: solo la palette dell'icona.
# 5.4.3 - 2026-08-31
#   - FIX: il pulsante 'Annulla' del dialog di conferma benchmark non aveva
#     alcun command -> clic inerte (funzionavano solo il tasto X ed Esc).
#     Ora ha un comando (win._annullato=True + destroy) e annulla come
#     promesso da docstring/spec.
# 5.4.2 - 2026-08-31
#   - UI: rimossa la macchina del tema Chiaro/Scuro (THEMES/_apply_theme/_recolor,
#     radio "Tema:" in toolbar, chiave "theme" in config): forzarne i colori su
#     tk.Button/Checkbutton/Radiobutton su macOS li degradava a widget flat con
#     cornice nera e testo illeggibile, soprattutto in Dark Mode. I controlli
#     tornano widget nativi: seguono il tema del sistema (scuro->bottone aqua
#     scuro con testo chiaro, nessuna cornice) e restano sempre leggibili. I
#     dialog di benchmark usano il colore testo di sistema (dinamico) e solo gli
#     accenti OK/KO in toni medi leggibili sia su chiaro che su scuro.
# 5.4.1 - 2026-08-31
#   - UI: tema selezionabile Chiaro/Scuro (radio button "Tema:" in toolbar,
#     persistito in config sotto "theme", default "dark"). Causa: su macOS in
#     Dark Mode tk.Button ha fg='Black' di default + i dialog di benchmark usano
#     testi scuri (#444/#555/#0a7d33/#b00020) -> testo scuro su sfondo scuro =
#     toolbar (−1000/+1000, Benchmark, Reset), label e i dialog 'prima/dopo' il
#     benchmark invisibili. Ogni tema e' autosufficiente (bg+fg coerenti) e
#     applicato su tutte le widget via option_add + ri-coloring a runtime
#     (MandelbrotApp._apply_theme/_recolor); i dialog di benchmark usano i colori
#     del tema attivo (verde/rosso/tenue). Sempre leggibile, indipendente dal
#     tema di sistema. Reset NON cambia il tema (e' una preferenza UI).
#     Solo colori UI, nessun effetto sul rendering.
# 5.4.0 - 2026-08-31
#   - NUOVO BACKEND GPU METAL (Apple Silicon): lo slot "GPU" e' ORA generico
#     (Motore CPU/GPU, config "cpu"/"gpu"). Su questa Mac GPU=Metal, su una
#     NVIDIA GPU=CUDA. Rilevamento: preferisco CUDA (ha f64) se presente,
#     altrimenti Metal (pyobjc+Metal). Metal su Apple Silicon NON supporta
#     'double' -> il backend Metal e' f32-ONLY; f64 resta solo sulla CPU (Numba).
#     Con Metal attivo il bottone f64 e' disabilitato (come CUDA senza kernel
#     f64); tornando a CPU f32/f64 sono di nuovo selezionabili (stato dinamico).
#   - Kernel MSL: stesso criterio di escape, interior analitico e coloring
#     (smooth iteration) di CUDA e CPU (v5.2.0) -> stesse immagini a parte 1-
#     qualche ULP di FMA/contrazione sul bordo caotico (limitazione intrinseca
#     di f32, NON un difetto: Metal f32 all'altezza di CUDA f32 e CPU f32).
#     Parametri (cx,cy,half,w,h,mi,pal) in uno struct via buffer [[buffer(1)]];
#     output [[buffer(0)]]. 1 px/thread, threadgroup 16x16 (256). Deterministico
#     (2 run bit-identici). H2D/D2H via buf.contents().as_buffer(N) (memoryview
#     scrivibile, C-level). compute_gpu e' ORA un dispatcher (CUDA/Metal);
#     backend() mostra "Metal f32"/"CUDA f32|f64"/"CPU f32|f64".
#   - Baseline: + riferimenti baseline/<zona>_metal_f32.npy (Metal, deterministico)
#     + misura Metal; i <zona>_gpu_*.npy (CUDA, storici) restano il gold f32 per
#     il cross-check del gate.
#   - Gate: portatile per piattaforma. Sempre CPU f64/f32 bit-id ai riferimenti.
#     GPU=CUDA: come prima (GPU f32/f64 bit-id a <zona>_gpu_*.npy, CPUf64~GPUf64
#     <=2%). GPU=Metal (no CUDA): Metal f32 bit-id a <zona>_metal_f32.npy +
#     determinismo (2 run) + cross-check Metal f32 ~ gold CUDA f32 entro la
#     varianza intrinseca di f32 (<=max(2%, 1.5 x CPUf32~CUDAf32)).
# 5.3.0 - 2026-08-31
#   - CPU f32: il toggle "precisione f32/f64" vale ORA anche per il motore CPU
#     (prima la CPU era SEMPRE f64/complex128). Workspace CPU per dtype
 #     (f32/f64, key (w,h,prec)); il kernel Numba si auto-specializza su
 #     complex64 (stesso codice); il fallback numpy usa la stessa sequenza di
 #     ufunc in parti esplicite (no FMA) anche in f32; self-test Numba-vs-numpy
 #     PER PRECISIONE (f64: bit-identita'; f32: it[] esatto + mag[] entro
 #     tolleranza — in f32 Numba/LLVM contrae in FMA/riassocia in modo
 #     dipendente dal contesto del loop, non riproducibile in modo affidabile
 #     con numpy) + warmup/compilazione di ENTRAMBE le precisioni all'avvio
 #     (flag di ok per precisione: se una delle due non passa resta il fallback
 #     solo per quella). backend() mostra "CPU f32"/"CPU f64"; UI: i bottoni
 #     f32/f64 sono abilitati anche in CPU (la guardia _KERNEL_F64 is None vale
 #     solo per CUDA). Il percorso CPU f64 resta bit-identico (gate); la CPU f32
 #     e' bit-identica al SUO riferimento (entrambi Numba).
#   - Baseline: nuovi riferimenti baseline/<zona>_cpu_f32.npy + misura CPU f32;
#     i <zona>_cpu.npy restano = CPU f64 (nessun rename).
#   - Gate: + check CPU f32 bit-identica ai riferimenti <zona>_cpu_f32.npy;
#     gli altri check sono invariati (GPU f32/f64 bit-id, CPU f64 bit-id,
#     CPU f64 ~ GPU f64 <=2%).
#   - Nota (misurata in Fase 0, 960x540 auto_mi): f32 NON accelera il loop di
#     escape (catena dipendente seriale, latency-bound: scalare f32 ~ scalare
#     f64, in z3 leggermente piu' lento: 0.84-0.96x); il guadagno e' solo sui
#     passi memory-bound (geometria/prefiltro/coloring): ~1.6-1.7x sulle
#     regioni esterne (z1/z2), ~1.07x in deep zoom (z3, regione del benchmark
#     integrato). Mantenuto come opzione dell'utente.
# 5.2.1 - 2026-08-30
#   - Cleanup: rimossi i riferimenti storici baseline/*_cpu_v4151.npy (3 file,
#     non piu' usati dal gate da v5.2.0; recuperabili dalla storia git).
#     Aggiornate le menzioni in spec.md/AGENTS.md/gate.py. Nessun cambio di
#     codice/comportamento (solo rimozione dati + docs).
# 5.2.0 - 2026-08-30
#   - FIX (CPU vs CUDA): la regione LONTANA dall'origine (punti che fuggono
#     alla 1a iterazione, it=0) era NERA in CPU ma ROSSASTRA in CUDA. Causa:
#     la CPU usava il coloring grezzo t=(it/mi)^0.35 e la riga rgb[it==0]=0
#     (servita a annerire i pixel INTERIORI) che colpireva anche i punti
#     fuggiti a it=0; la GPU invece usa lo smooth-iteration
#     nu = it+1-log2(0.5*ln|z|^2) (>0 per i punti lontani) e annerisce solo
#     i non-fuggiti (flag esc). Ora la CPU usa lo STESSO coloring della GPU:
#     il kernel Numba e il fallback numpy esportano anche mag=|z|^2 alla fuga;
#     la colorazione e' nu=it+1-log2(0.5*ln(mag)), t=(nu/mi)^0.35, e solo i
#     pixel mai fuggiti (mag==0) restano neri. CPU e CUDA ora producono lo
#     stesso colore (a parte 1-2 ULP di log2/log libm vs CUDA -> <=1 entry LUT;
#     documentato nel gate).
#   - Gate: il vecchio check CPU "continuita' <=1.5% dei pixel vs v4.15.1"
#     (pensato per l'effetto FMA) e' SOSTITUITO da un check piu' forte e
#     diretto: CPU e GPUf64 devono rendere la STESSA immagine (<=2% dei pixel
#     con max diff per canale > 8; misurato 0.011-0.75%: solo bordo caotico +
#     1-2 ULP di log2/log libm-vs-CUDA). Il confronto-pixel vs v4.15.1 non ha
#     piu' senso: la formula di coloring e' cambiata VOLUTARIAMENTE, quindi
#     tutta l'immagine si ri-colore di poco (baseline/*_cpu_v4151.npy resta
#     come riferimento storico, non piu' usato dal gate).
#   - Riferimenti CPU baseline/*_cpu.npy RIGENERATI (nuova formula); GPU
#     invariati (kernel non toccato).
# 5.1.2 - 2026-08-30
#   - All'avvio il programma parte SEMPRE con la configurazione di default:
#     vista sull'INTERO insieme di Mandelbrot (CX0/CY0/HALF0) + MI auto, come
#     la prima volta. Lo stato della vista (cx, cy, half, mi, mi_auto) NON viene
#     piu' ripristinato da config.json (vecchi valori ignorati); la vista si
#     recupera solo caricando esplicitamente un file zona ('Carica zona...').
#     La vista non e' piu' neppure salvata in config (dato morto); la persiste
#     solo il file zona. Precisione/palette/motore/benchmark restano persistiti.
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
import struct
import numpy as np
from PIL import Image, ImageTk

# --- Costanti di vista e rendering ---
# v5.9.4: default 1280x720 (16:9, come prima): con la toolbar cresciuta
# (dropdown GPU) 960px non bastavano e il canvas si stirava. Il benchmark
# resta a 960x540 (BENCH, comparabilita' storica).
INIT_W, INIT_H = 1280, 720
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

# Riferimenti storici per il grafico a barre del benchmark (rendering/s,
# stessa regione/parametri): evidenziano il salto CPU->GPU su macchine note.
BENCH_REF = (
    ("AMD 9900X (storico)", 6.62),
    ("4070 Super Vulkan (storico)", 124.0),
    ("5070 Ti Vulkan (storico)", 179.0),
    ("4070 Super CUDA (storico)", 250.0),
    ("5070 Ti CUDA (storico)", 350.0),
)

VERSION = "5.10.0"

# Ultime 10 modifiche di versione per Help -> "Novità recenti..."
# (versione, data, descrizione breve). Fonte embedded: i commenti STORICO
# non sopravvivono alla build PyInstaller, quindi il dialog legge da qui.
# REGOLA BUMP: aggiungere la voce nuova in testa e tenere max 10.
HISTORY = (
    ("5.10.0", "2026-09-03", "Foto 2x2 + Foto 4x4 (NxN) e pulsante Ricalcola."),
    ("5.9.8", "2026-09-03", "Zoom-out macOS: tasti globali + click destro x0.5."),
    ("5.9.7", "2026-09-03", "Bugfix Foto: typo nome pulsante, ora funziona."),
    ("5.9.6", "2026-09-03", "Pulsante Foto: vista a 2x con antialiasing (hourglass)."),
    ("5.9.5", "2026-09-03", "Single-core: ora mostra il perche' (status+Info)."),
    ("5.9.4", "2026-09-03", "Finestra default 1280x720 (16:9)."),
    ("5.9.3", "2026-09-03", "Ref 4070 Super Vulkan corretto a 124."),
    ("5.9.2", "2026-09-03", "Dropdown GPU anche per Vulkan (adapter selezionabile)."),
    ("5.9.1", "2026-09-03", "Benchmark: worker in pausa, hint device; refs su GPU locali."),
    ("5.9.0", "2026-09-03", "Dropdown scelta GPU se > 1 CUDA (persistita in config)."),
)

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

_CUDA_OK = False
_KERNEL_F32 = None
_KERNEL_F64 = None
_BUF = None
try:
    import cupy as cp
    if cp.cuda.is_available():
        _src, _name = _build_kernel("f32")
        _KERNEL_F32 = cp.RawKernel(_src, _name, options=("--use_fast_math",))
        _CUDA_OK = True
        try:
            _src, _name = _build_kernel("f64")
            _KERNEL_F64 = cp.RawKernel(_src, _name, options=("--use_fast_math",))
        except Exception:
            _KERNEL_F64 = None
except Exception:
    _CUDA_OK = False

# v5.9.0: GPU CUDA multiple. _CUDA_DEVICES = [(id, nome)] rilevate all'avvio;
# _CUDA_DEV = indice selezionato (default 0). Il dropdown "GPU:" in toolbar
# compare solo se > 1. Il device CuPy e' PER-THREAD: i percorsi di render,
# benchmark e warmup entrano in 'with cp.cuda.Device(_CUDA_DEV)' (il .use()
# qui sotto vale solo per il thread chiamante).
_CUDA_DEVICES = []
_CUDA_DEV = 0
if _CUDA_OK:
    try:
        for _i in range(cp.cuda.runtime.getDeviceCount()):
            _nm = cp.cuda.runtime.getDeviceProperties(_i)["name"]
            if isinstance(_nm, bytes):
                _nm = _nm.decode("utf-8", "replace")
            _CUDA_DEVICES.append((_i, str(_nm).strip()))
    except Exception:
        _CUDA_DEVICES = []

def _cuda_short_name(name):
    # "NVIDIA GeForce RTX 5070 Ti" -> "GeForce RTX 5070 Ti" (dropdown compatto)
    if name.startswith("NVIDIA "):
        return name[len("NVIDIA "):]
    return name

def _cuda_label(i):
    _id, _nm = _CUDA_DEVICES[i]
    return "%d: %s" % (_id, _cuda_short_name(_nm))

def set_cuda_device(i):
    """Seleziona la GPU CUDA (indice in _CUDA_DEVICES); True se ok.
    Invalida _BUF (realloc sul nuovo device) e la cache hw_name."""
    global _CUDA_DEV, _BUF
    if not _CUDA_DEVICES:
        return False
    try:
        i = int(i)
    except (TypeError, ValueError):
        return False
    _CUDA_DEV = max(0, min(i, len(_CUDA_DEVICES) - 1))
    _BUF = None
    _HW_CACHE.pop("cuda", None)
    try:
        cp.cuda.Device(_CUDA_DEVICES[_CUDA_DEV][0]).use()
    except Exception:
        pass
    return True

# ---------------- GPU (Metal, Apple Silicon) ----------------
# Backend Metal per Apple GPU (M1...). Metal su Apple Silicon NON supporta
# 'double' -> il backend Metal e' f32-ONLY (f64 resta solo sulla CPU, Numba).
# Stesso criterio di escape, interior analitico e coloring (smooth iteration)
# del kernel CUDA e del percorso CPU (v5.2.0): le tre vie producono lo stesso
# colore, a parte 1-qualche ULP di FMA/contrazione sul bordo caotico (limitazione
# intrinseca di f32, NON un difetto: Metal f32 e' all'altezza di CUDA f32 e CPU f32).
# Vincoli pyobjc (scoperti a caro prezzo):
#   - gli argomenti scalari vanno in uno struct passato via buffer [[buffer(1)]]
#     (ne' scalari nudi ne' [[buffer]] su singoli float sono accettati);
#   - MSL usa fmin/fmax/pow/sqrt/log2/log (senza suffisso 'f') e 'half' e'
#     riservata (quindi il campo si chiama 'hs');
#   - i puntatori richiedono l'address space esplicito (device/constant);
#   - H2D/D2H: buf.contents().as_buffer(N) restituisce una memoryview scrivibile
#     (C-level, ~0.01 ms) -> e' il ponte rapido (la varlist NON e' bytes-like);
#   - setBuffer:offset:atIndex: ha firma (buffer, offset, INDEX) -> (pbuf, 0, 1).
def _build_metal_msl():
    names = list(PALETTES)
    consts = "\n".join(
        "constant unsigned char LUT_%s[768] = { %s };"
        % (n.upper(), _fmt_lut(make_lut(pal))) for n, pal in PALETTES.items())
    select = "    constant unsigned char* lut = LUT_%s;" % names[0].upper()
    for i, n in enumerate(names[1:], start=1):
        select += "\n    if (p.pal == %d) lut = LUT_%s;" % (i, n.upper())
    return r'''
#include <metal_stdlib>
using namespace metal;

@@CONSTS@@

struct Params {
    float cx;
    float cy;
    float hs;
    int w;
    int h;
    int mi;
    int pal;
};

kernel void mandel(
    device unsigned char* out [[buffer(0)]],
    constant Params& p [[buffer(1)]],
    uint2 pos [[thread_position_in_grid]])
{
    int col = (int)pos.x;
    int row = (int)pos.y;
    if (col >= p.w || row >= p.h) return;

@@SELECT@@

    float cx = p.cx, cy = p.cy, hs = p.hs;
    int w = p.w, h = p.h, mi = p.mi;

    float x0 = cx + hs * ((float)(2 * col - w) / (float)w);
    float y0 = cy + hs * ((float)h / (float)w) * ((float)(2 * row - h) / (float)h);

    device unsigned char* pp = out + (size_t)(row * w + col) * 3;

    // Interior analitico: bulbo periodica-2 + cardioide principale (stesso criterio CUDA/CPU)
    if (x0 >= -2.0f && x0 <= 0.4f && y0 >= -1.3f && y0 <= 1.3f) {
        float d2 = (x0 + 1.0f) * (x0 + 1.0f) + y0 * y0;
        if (d2 <= 0.0625f) { pp[0] = 0; pp[1] = 0; pp[2] = 0; return; }
        float A = 1.0f - 4.0f * x0;
        float B = -4.0f * y0;
        float R = sqrt(A * A + B * B);
        if (R < 2.0f * sqrt(0.5f * (R + A))) { pp[0] = 0; pp[1] = 0; pp[2] = 0; return; }
    }

    float a = cx * cx + (x0 - cx);
    float two_cx = 2.0f * cx;
    float wr = -cx, wi = 0.0f;
    int it = 0;
    bool esc = false;
    float mag2 = 0.0f;
    for (int i = 0; i < mi; ++i) {
        if (esc) break;
        float nr = wr * wr - wi * wi + two_cx * wr + a;
        float ni = two_cx * wi + 2.0f * wr * wi + y0;
        wr = nr; wi = ni;
        float zr = wr + cx;
        mag2 = zr * zr + wi * wi;
        if (mag2 > 4.0f) { esc = true; it = i; }
    }
    if (!esc) { pp[0] = 0; pp[1] = 0; pp[2] = 0; return; }

    float nu = (float)it + 1.0f - log2(0.5f * log(mag2));
    float t = pow(fmin(1.0f, fmax(0.0f, nu / (float)mi)), 0.35f);
    int idx = (int)(fmin(1.0f, fmax(0.0f, t)) * 255.0f);
    pp[0] = lut[idx * 3 + 0];
    pp[1] = lut[idx * 3 + 1];
    pp[2] = lut[idx * 3 + 2];
}
'''.replace("@@CONSTS@@", consts).replace("@@SELECT@@", select)


class MetalBackend:
    """Backend Metal (Apple GPU) f32, deterministico (2 run bit-identici).
    I parametri (cx, cy, half, w, h, mi, pal) vanno in uno struct passato via
    buffer [[buffer(1)]]; l'output e' un buffer [[buffer(0)]]. H2D/D2H via
    buf.contents().as_buffer(N) (memoryview scrivibile, C-level). Un lock
    serializza le chiamate (il compute e' sincrono: commit+waitUntilCompleted+read).
    """
    TH = 16  # threads per threadgroup (16x16 = 256, sotto il max 1024 di M1)

    def __init__(self):
        import Metal
        self._Metal = Metal
        self.dev = Metal.MTLCreateSystemDefaultDevice()
        if self.dev is None:
            raise RuntimeError("nessun dispositivo Metal")
        self.q = self.dev.newCommandQueue()
        src = _build_metal_msl()
        self.lib, err = self.dev.newLibraryWithSource_options_error_(src, None, None)
        if self.lib is None:
            raise RuntimeError("compilazione MSL fallita: %s" % err)
        fn = self.lib.newFunctionWithName_('mandel')
        self.ps, err = self.dev.newComputePipelineStateWithFunction_error_(fn, None)
        if self.ps is None:
            raise RuntimeError("pipeline Metal fallita: %s" % err)
        self._out = None
        self._pbuf = self.dev.newBufferWithLength_options_(
            28, Metal.MTLResourceStorageModeShared)
        self._lock = threading.Lock()

    def _out_buf(self, need):
        if self._out is None or self._out.length() < need:
            self._out = self.dev.newBufferWithLength_options_(
                need, self._Metal.MTLResourceStorageModeShared)
        return self._out

    def compute(self, cx, cy, half, w, h, mi, pal=0):
        M = self._Metal
        need = w * h * 3
        with self._lock:
            out = self._out_buf(need)
            payload = struct.pack('<fffiiii', cx, cy, half, w, h, mi, pal)
            self._pbuf.contents().as_buffer(len(payload))[:] = payload
            gx = (w + self.TH - 1) // self.TH
            gy = (h + self.TH - 1) // self.TH
            cmd = self.q.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self.ps)
            enc.setBuffer_offset_atIndex_(out, 0, 0)         # buffer[0] = out
            enc.setBuffer_offset_atIndex_(self._pbuf, 0, 1)   # buffer[1] = params (offset=0, INDEX=1)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                M.MTLSizeMake(gx, gy, 1), M.MTLSizeMake(self.TH, self.TH, 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            mv = out.contents().as_buffer(need)
            return np.frombuffer(mv, dtype=np.uint8, count=need).reshape(h, w, 3).copy()


_METAL_OK = False
_METAL_BE = None
try:
    import Metal  # pyobjc: disponibile su macOS con i framework di sistema
    if Metal.MTLCreateSystemDefaultDevice() is not None:
        _METAL_BE = MetalBackend()
        _METAL_OK = True
except Exception:
    _METAL_OK = False
    _METAL_BE = None

# ---------------- GPU (Vulkan, wgpu) ----------------
# Backend Vulkan via wgpu (wgpu-native) per GPU discrete/integrate
# (AMD/NVIDIA/Intel), cross-platform (Windows/macOS/Linux). f32-ONLY (come
# Metal: la f64 su GPU consumer e' lenta e qui inutile). Stesso criterio di
# escape, interior analitico e coloring (smooth iteration) dei kernel
# CUDA/MSL/CPU -> stessa immagine a parte 1-qualche ULP sul bordo caotico
# (limitazione intrinseca di f32, NON un difetto).
# Vincoli WGSL (vs MSL/CUDA):
#   - nessun tipo u8: LUT e output sono array<u32> con colori packed
#     0x00RRGGBB (1 px = 1 u32); al readback si unpacka in numpy (h,w,3);
#   - i parametri vanno in DUE buffer uniform (vec4<f32> = cx,cy,half e
#     vec4<i32> = w,h,mi,pal); LUT e output sono buffer storage;
#   - min/max/pow/sqrt/log/log2 sono i built-in WGSL (senza suffisso 'f',
#     a differenza di MSL che vuole fmin/fmax/fpow/...);
#   - H2D via queue.write_buffer (buffer persistenti, uso UNIFORM|COPY_DST),
#     D2H via queue.read_buffer (buffer out uso STORAGE|COPY_SRC);
#   - un lock serializza le chiamate (compute sincrono: submit + read_buffer).
def _build_vulkan_wgsl():
    return r'''
@group(0) @binding(0) var<storage, read> lut: array<u32>;
@group(0) @binding(1) var<uniform> p: vec4<f32>;
@group(0) @binding(2) var<uniform> dim: vec4<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = i32(gid.x);
    let row = i32(gid.y);
    let w = dim.x;
    let h = dim.y;
    if (col >= w || row >= h) { return; }
    let cx = p.x;
    let cy = p.y;
    let hs = p.z;
    let mi = dim.z;
    let pal = dim.w;

    let x0 = cx + hs * (f32(2 * col - w) / f32(w));
    let y0 = cy + hs * (f32(h) / f32(w)) * (f32(2 * row - h) / f32(h));
    let ridx = row * w + col;

    // Interior analitico: bulbo periodica-2 + cardioide principale (stesso criterio)
    if (x0 >= -2.0 && x0 <= 0.4 && y0 >= -1.3 && y0 <= 1.3) {
        let d2 = (x0 + 1.0) * (x0 + 1.0) + y0 * y0;
        if (d2 <= 0.0625) { out[ridx] = 0u; return; }
        let A = 1.0 - 4.0 * x0;
        let B = -4.0 * y0;
        let R = sqrt(A * A + B * B);
        if (R < 2.0 * sqrt(0.5 * (R + A))) { out[ridx] = 0u; return; }
    }

    let a = cx * cx + (x0 - cx);
    let two_cx = 2.0 * cx;
    var wr = -cx;
    var wi = 0.0;
    var esc = false;
    var it = 0;
    var mag2 = 0.0;
    for (var i = 0; i < mi; i = i + 1) {
        if (esc) { break; }
        let nr = wr * wr - wi * wi + two_cx * wr + a;
        let ni = two_cx * wi + 2.0 * wr * wi + y0;
        wr = nr;
        wi = ni;
        let zr = wr + cx;
        mag2 = zr * zr + wi * wi;
        if (mag2 > 4.0) { esc = true; it = i; }
    }
    if (!esc) { out[ridx] = 0u; return; }

    let nu = f32(it) + 1.0 - log2(0.5 * log(mag2));
    let t = pow(min(1.0, max(0.0, nu / f32(mi))), 0.35);
    let idx = i32(min(1.0, max(0.0, t)) * 255.0);
    out[ridx] = lut[pal * 256 + idx];
}
'''


class VulkanBackend:
    """Backend Vulkan (wgpu) f32, deterministico (2 run bit-identici).
    LUT (tutte le palette concatenate, u32 packed 0x00RRGGBB) e output
    (h*w u32 packed) sono buffer storage; i parametri (cx,cy,half in
    vec4<f32>; w,h,mi,pal in vec4<i32>) sono due buffer uniform. H2D via
    queue.write_buffer (buffer persistenti), D2H via queue.read_buffer.
    Un lock serializza le chiamate (compute sincrono: submit + read_buffer).
    """
    TH = 16  # workgroup 16x16 = 256 thread (sotto il max di AMD/NVIDIA/Intel)

    def __init__(self, adapter=None):
        import wgpu
        self._w = wgpu
        self._lock = threading.Lock()
        if adapter is None:
            if not wgpu.gpu.enumerate_adapters_sync():
                raise RuntimeError("nessun adapter GPU (Vulkan)")
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter is None:
                raise RuntimeError("nessun adapter (Vulkan)")
        self._init_on_adapter(adapter)

    def _init_on_adapter(self, adapter):
        # (Ri)crea TUTTE le risorse device-bound su 'adapter'. Chiamato da
        # __init__ (nessuna contesa ancora) e da select_adapter (sotto lock).
        wgpu = self._w
        self.dev = adapter.request_device_sync()
        if self.dev is None:
            raise RuntimeError("device Vulkan fallito")
        try:
            self.name = dict(adapter.info).get("device", "GPU")
        except Exception:
            self.name = "GPU"
        self.lut_buf = self._make_lut_buffer()
        shader = self.dev.create_shader_module(code=_build_vulkan_wgsl())
        self.pipe = self.dev.create_compute_pipeline(
            layout="auto",
            compute=wgpu.ProgrammableStage(module=shader, entry_point="main"))
        self.layout = self.pipe.get_bind_group_layout(0)
        uflags = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        self.p_buf = self.dev.create_buffer(size=16, usage=uflags)
        self.dim_buf = self.dev.create_buffer(size=16, usage=uflags)
        self._out = None
        self._out_size = 0
        self._bg = None
        self._bg_out = None

    def select_adapter(self, adapter):
        # v5.9.2: cambio GPU a runtime (dropdown): ricrea device, pipeline,
        # buffer e bind group sul nuovo adapter (mai in gara con compute()).
        with self._lock:
            self._init_on_adapter(adapter)

    def _make_lut_buffer(self):
        # Tutte le palette concatenate (ordine PALETTES = indice pal), u32
        # packed 0x00RRGGBB: 1 entry = 1 u32. La selezione per indice pal
        # avviene nel shader (lut[pal*256 + idx]), come CUDA/MSL.
        lut = np.empty((len(PALETTES), 256), dtype=np.uint32)
        for i, (_name, pal) in enumerate(PALETTES.items()):
            l = make_lut(pal).astype(np.uint32)  # (256,3)
            lut[i] = l[:, 0] | (l[:, 1] << 8) | (l[:, 2] << 16)
        data = np.ascontiguousarray(lut).ravel().tobytes()
        return self.dev.create_buffer_with_data(
            data=data, usage=self._w.BufferUsage.STORAGE)

    def _out_buf(self, need):
        if self._out is None or self._out_size < need:
            self._out = self.dev.create_buffer(
                size=need,
                usage=self._w.BufferUsage.STORAGE | self._w.BufferUsage.COPY_SRC)
            self._out_size = need
            self._bg = None  # il bind group punta sul buffer vecchio
        return self._out

    def _ensure_bg(self, out):
        if self._bg is None or self._bg_out is not out:
            self._bg = self.dev.create_bind_group(layout=self.layout, entries=[
                self._w.BindGroupEntry(binding=0, resource=self.lut_buf),
                self._w.BindGroupEntry(binding=1, resource=self.p_buf),
                self._w.BindGroupEntry(binding=2, resource=self.dim_buf),
                self._w.BindGroupEntry(binding=3, resource=out),
            ])
            self._bg_out = out
        return self._bg

    def compute(self, cx, cy, half, w, h, mi, pal=0):
        need = w * h * 4  # 1 u32 per pixel
        with self._lock:
            out = self._out_buf(need)
            self.dev.queue.write_buffer(
                self.p_buf, 0, struct.pack('<ffff', cx, cy, half, 0.0))
            self.dev.queue.write_buffer(
                self.dim_buf, 0, struct.pack('<iiii', w, h, mi, pal))
            bg = self._ensure_bg(out)
            enc = self.dev.create_command_encoder()
            p = enc.begin_compute_pass()
            p.set_pipeline(self.pipe)
            p.set_bind_group(0, bg)
            p.dispatch_workgroups(
                (w + self.TH - 1) // self.TH, (h + self.TH - 1) // self.TH, 1)
            p.end()
            self.dev.queue.submit([enc.finish()])
            mv = self.dev.queue.read_buffer(out, 0, w * h * 4)
            u = np.frombuffer(mv, dtype=np.uint32).reshape(h, w)
            rgb = np.empty((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = (u & 0xFF)
            rgb[:, :, 1] = (u >> 8) & 0xFF
            rgb[:, :, 2] = (u >> 16) & 0xFF
            return rgb


_VULKAN_OK = False
_VULKAN_BE = None
# v5.9.2: adapter GPU fisici (backend Vulkan) enumerati all'avvio.
# _VULKAN_ADAPTERS = [(indice_enum, nome)]; _VULKAN_DEV = posizione
# selezionata (default 0). Il dropdown "GPU:" li mostra quando il motore
# attivo e' Vulkan (prima pilotava solo CUDA: Vulkan usava sempre
# l'high-performance -> benchmark quasi uguali su GPU diverse).
_VULKAN_ADAPTERS = []
_VULKAN_DEV = 0
try:
    import wgpu  # wgpu-native: cross-platform (Windows/macOS/Linux)
    _enum = wgpu.gpu.enumerate_adapters_sync()
    _seen = set()
    for _ei, _a in enumerate(_enum):
        try:
            _inf = dict(_a.info)
        except Exception:
            continue
        if _inf.get("backend_type") != "Vulkan":
            continue
        _key = (_inf.get("vendor_id"), _inf.get("device_id"))
        if _key in _seen:
            continue
        _seen.add(_key)
        _VULKAN_ADAPTERS.append((_ei, str(_inf.get("device", "GPU")).strip()))
    if not _VULKAN_ADAPTERS:
        raise RuntimeError("nessun adapter GPU (Vulkan)")
    # default = stesso adapter di prima (high-performance): prima voce che
    # matcha (vendor, device), altrimenti la prima enumerata.
    try:
        _hp = dict(wgpu.gpu.request_adapter_sync(
            power_preference="high-performance").info)
        _hpk = (_hp.get("vendor_id"), _hp.get("device_id"))
        for _k, (_ei, _nm) in enumerate(_VULKAN_ADAPTERS):
            _inf = dict(_enum[_ei].info)
            if (_inf.get("vendor_id"), _inf.get("device_id")) == _hpk:
                _VULKAN_ADAPTERS[0], _VULKAN_ADAPTERS[_k] = \
                    _VULKAN_ADAPTERS[_k], _VULKAN_ADAPTERS[0]
                break
    except Exception:
        pass
    _VULKAN_BE = VulkanBackend(_enum[_VULKAN_ADAPTERS[0][0]])
    _VULKAN_OK = True
except Exception:
    _VULKAN_OK = False
    _VULKAN_BE = None
    _VULKAN_ADAPTERS = []
    _VULKAN_DEV = 0


def _vulkan_short_name(name):
    # "NVIDIA GeForce RTX 5070 Ti" -> "GeForce RTX 5070 Ti" (dropdown compatto)
    if name.startswith("NVIDIA "):
        return name[len("NVIDIA "):]
    return name

def _vulkan_label(i):
    return "%d: %s" % (i, _vulkan_short_name(_VULKAN_ADAPTERS[i][1]))

def set_vulkan_adapter(i):
    """Seleziona l'adapter Vulkan (posizione in _VULKAN_ADAPTERS); True se ok.
    Ricrea le risorse device-bound via backend (sotto lock) e invalida la
    cache hw_name."""
    global _VULKAN_DEV
    if _VULKAN_BE is None or not _VULKAN_ADAPTERS:
        return False
    try:
        i = int(i)
    except (TypeError, ValueError):
        return False
    i = max(0, min(i, len(_VULKAN_ADAPTERS) - 1))
    try:
        import wgpu
        _enum = wgpu.gpu.enumerate_adapters_sync()
        _VULKAN_BE.select_adapter(_enum[_VULKAN_ADAPTERS[i][0]])
    except Exception:
        return False
    _VULKAN_DEV = i
    _HW_CACHE.pop("vulkan", None)
    return True

# ---------------- Selezione backend (v5.6.0) ----------------
# Quattro backend selezionabili a runtime:
#   cpu     - CPU (Numba/numpy), sempre disponibile, f32 e f64
#   cuda    - CUDA (NVIDIA, CuPy), f32 (+ f64 se il kernel f64 e' compilato)
#   metal   - Metal (Apple Silicon, pyobjc), f32-only
#   vulkan  - Vulkan (wgpu), f32-only, AMD/NVIDIA/Intel cross-platform
# Ogni backend e' rilevato all'avvio (_CUDA_OK/_METAL_OK/_VULKAN_OK).
# _BACKENDS_OK e' l'elenco di quelli disponibili (ordine di preferenza);
# _ACTIVE e' il backend corrente (stringa); il default e' il primo GPU
# disponibile in ordine CUDA > Metal > Vulkan, altrimenti CPU.
_BACKENDS_OK = []
if _CUDA_OK:
    _BACKENDS_OK.append("cuda")
if _METAL_OK:
    _BACKENDS_OK.append("metal")
if _VULKAN_OK:
    _BACKENDS_OK.append("vulkan")
_BACKENDS_OK.append("cpu")

def _backend_ok(name):
    return name in _BACKENDS_OK

def _default_backend():
    return _BACKENDS_OK[0]

_ACTIVE = _default_backend()
# _GPU = c'e' almeno un backend GPU disponibile (usato per il warmup all'avvio).
_GPU = _ACTIVE != "cpu"

# v5.8.13: accenti colore UI per motore (toni medi, leggibili su tema chiaro
# e scuro). Usati SOLO su Label/Frame (status bar, barra accento, titoli
# dialog): Button/Checkbutton/Radiobutton restano nativi (lezione 5.4.2).
BACKEND_FG = {"cpu": "#1f6feb", "cuda": "#2ea44f",
              "metal": "#8957e5", "vulkan": "#d97706"}
ERR_FG = "#e5534b"

def _backend_fg():
    return BACKEND_FG.get(_ACTIVE, "#1f6feb")

def _gpu_supports_f64():
    # f64 solo su CUDA con kernel f64 compilato; Metal e Vulkan sono f32-only.
    return _ACTIVE == "cuda" and _KERNEL_F64 is not None

_PREC = "f32"

def set_prec(p):
    global _PREC
    # v5.6.0: la guardia f64 vale per il backend corrente (Metal/Vulkan sono
    # f32-only; CUDA senza kernel f64). La CPU supporta sempre sia f32 sia f64.
    if p == "f64" and _ACTIVE != "cpu" and not _gpu_supports_f64():
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

def _compute_gpu_cuda(cx, cy, half, w, h, mi, buf=None, prec=None):
    global _BUF
    # v5.9.0: tutto sul device selezionato (il current-device CuPy e'
    # per-thread: worker, benchmark e warmup girano su thread diversi).
    _dev = _CUDA_DEVICES[_CUDA_DEV][0] if _CUDA_DEVICES else 0
    with cp.cuda.Device(_dev):
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

def _compute_gpu_metal(cx, cy, half, w, h, mi, prec=None):
    # Metal e' f32-only (Apple Silicon non supporta 'double').
    if prec == "f64":
        raise ValueError("Metal: solo f32 ('double' non supportato su Apple Silicon)")
    rgb = _METAL_BE.compute(cx, cy, half, w, h, mi, pal=list(PALETTES).index(_PALETTE))
    return Image.fromarray(rgb)

def _compute_gpu_vulkan(cx, cy, half, w, h, mi, prec=None):
    # Vulkan (wgpu) e' f32-only (stessa scelta di Metal).
    if prec == "f64":
        raise ValueError("Vulkan: solo f32")
    rgb = _VULKAN_BE.compute(cx, cy, half, w, h, mi, pal=list(PALETTES).index(_PALETTE))
    return Image.fromarray(rgb)

def compute_gpu(cx, cy, half, w, h, mi, buf=None, prec=None):
    # v5.6.0: dispatcher del backend GPU attivo (CUDA / Metal / Vulkan).
    # 'buf' e' usato solo dal percorso CUDA (Metal/Vulkan usano i propri buffer).
    if _ACTIVE == "metal":
        return _compute_gpu_metal(cx, cy, half, w, h, mi, prec=prec)
    if _ACTIVE == "vulkan":
        return _compute_gpu_vulkan(cx, cy, half, w, h, mi, prec=prec)
    return _compute_gpu_cuda(cx, cy, half, w, h, mi, buf=buf, prec=prec)

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
# v5.3.0: flag di ok PER PRECISIONE (f32 e f64 si auto-specializzano in kernel
# Numba distinti; se il self-test di bit-identita' NON passa per una
# precisione, il fallback numpy resta attivo solo per quella).
_NUMBA_OK = {"f64": False, "f32": False}
# v5.9.5: diagnostica del fallback single-core (visibile in Help >
# Informazioni e come avviso in status bar). "import": "ok" o motivo del
# fallimento; per precisione: "warmup in corso..." | "ok (multi-core)" |
# motivo ("self-test..." / "compilazione fallita: ..."); "tempo": secondi
# del warmup (None se non concluso).
_NUMBA_STATUS = {"import": "ok", "f64": "warmup in corso...",
                 "f32": "warmup in corso...", "tempo": None}
_GEN = np.zeros(1, dtype=np.int32)
try:
    from numba import njit, prange

    # NB: NO cache=True: il cache di numba e' legato al nome del modulo e
    # mandel.py e' eseguito/importato con nomi diversi (__main__ vs mandel vs
    # loader dinamico) -> il load della cache falliva. La compilazione (~1-2 s)
    # e' pagata dal thread di warmup all'avvio, prima del primo render CPU.
    @njit(parallel=True)
    def _mandel_escape(c, diverged, it, mag, mi, row0, row1):
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
                        mag[row, col] = z_re * z_re + z_im * z_im
                        break
except Exception as ex:
    njit = None
    prange = None
    # v5.9.5: motivo registrato (prima muto) per la diagnostica in UI.
    _NUMBA_STATUS["import"] = "numba non importabile: " + str(ex)[:160]
    _NUMBA_STATUS["f64"] = _NUMBA_STATUS["f32"] = "numba non disponibile"

# Numba assente (import fallito) -> il motore CPU gira sul fallback numpy
# single-core; lo segnalo nell'etichetta del backend (vedi backend()).
_NUMBA_AVAILABLE = (njit is not None)

def _numba_selftest(cdt):
    """Self-test del kernel Numba contro il loop numpy di riferimento (4x4 con
    casi limite: interiori, fuga rapida/lenta, pre-diverged, overflow), STESSA
    formula. Criterio per precisione:
      f64: it[] e mag[] BIT-IDENTICI (Numba f64 e numpy f64 usano la stessa
           aritmetica a 3 arrotondamenti — verificato).
      f32: it[] (decisioni di fuga) ESATTAMENTE uguali; mag[] entro una
           toller stretta. In f32 Numba/LLVM applica contrazioni FMA e
           riassociazioni che DIPENDONO DAL CONTESTO del loop (verificato:
           z_re^2-z_im^2 resta a 3 arrotondamenti, mentre 2*z_re*z_im+ci e
           z_re^2+z_im^2 vengono contratti in FMA) -> la bit-identita' con
           numpy non e' riproducibile in modo affidabile. La differenza e' di
           1-qualche ULP su una piccolissima frazione di pixel di bordo
           caotico (irrelevante: al piu' 1 entry di LUT nel coloring).
    """
    fdt = np.float32 if cdt is np.complex64 else np.float64
    c = np.array([
        -0.5 + 0.5j, 0.7 + 0.1j, -1.7 + 0j, 0.3 + 0.6j,
        -0.1 - 0.9j, 2.5j, 0.0, 1.0,
        -0.749 - 0.001j, 0.1 - 0.7j, -1.2 - 0.5j, 0.9 + 0.9j,
        -0.28 + 0.01j, -2.0j, 0.5 - 1.9j, -0.75 + 0.12j,
    ], dtype=cdt).reshape(4, 4)
    d = np.zeros((4, 4), dtype=bool)
    d[0, 0] = True  # pre-diverged (come il prefiltro interior)
    it_numba = np.zeros((4, 4), dtype=np.int32)
    mag_numba = np.zeros((4, 4), dtype=fdt)
    _mandel_escape(c, d, it_numba, mag_numba, 64, 0, 4)
    # riferimento numpy (stessa formula del fallback in compute_cpu)
    zr = np.zeros((4, 4), dtype=fdt)
    zi = np.zeros((4, 4), dtype=fdt)
    cr = c.real
    ci = c.imag
    div = d.copy()
    it_ref = np.zeros((4, 4), dtype=np.int32)
    mag_ref = np.zeros((4, 4), dtype=fdt)
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
            mag_ref[m] = zr[m] * zr[m] + zi[m] * zi[m]
    # it[] (decisioni di fuga) DEVE coincidere SEMPRE, per entrambe le precisioni
    if not np.array_equal(it_numba, it_ref):
        return False
    if cdt is np.complex64:
        # f32: mag[] entro tolleranza stretta (differenze FMA/riassociazione
        # di Numba, ~1-qualche ULP). Cattura errori grossolani (formula errata)
        # tollerando il rumore di arrotondamento.
        esc = (it_numba > 0) & (it_ref > 0)
        if not esc.any():
            return True
        a = mag_numba[esc].astype(np.float64)
        b = mag_ref[esc].astype(np.float64)
        rel = np.abs(a - b) / np.maximum(b, 1.0)
        return bool((rel <= 1e-2).all())
    # f64: bit-identita' stretta
    return bool(np.array_equal(mag_numba, mag_ref))

def _numba_diag():
    """Riga diagnostica CPU/Numba per Help > Informazioni (v5.9.5): se non
    si va in multi-core, spiega il perche' (da _NUMBA_STATUS)."""
    if _NUMBA_OK.get("f32") and _NUMBA_OK.get("f64"):
        t = _NUMBA_STATUS.get("tempo")
        return "Numba multi-core attivo (f32+f64%s)." % (
            ", warmup %ss" % t if t is not None else "")
    parts = []
    if _NUMBA_STATUS.get("import") != "ok":
        parts.append(str(_NUMBA_STATUS["import"]))
    for p in ("f64", "f32"):
        if not _NUMBA_OK.get(p):
            parts.append("%s: %s" % (p, _NUMBA_STATUS.get(p, "?")))
    return "Single-core (numpy). Motivo: " + "; ".join(parts)


def _numba_warmup():
    """Compila in background all'avvio i kernel di ENTRAMBE le precisioni
    (f64 e f32; il kernel Numba si auto-specializza sul dtype in ingresso) e
    fa il self-test contro il loop numpy di riferimento, una precisione per
    volta (f64: bit-identita'; f32: it[] esatto + mag[] entro tolleranza —
    vedi _numba_selftest); se il test non passa (o la compilazione fallisce)
    resta il fallback numpy per quella precisione. Esito/motivo/tempo in
    _NUMBA_STATUS (v5.9.5: niente piu' except muti per la diagnostica UI).
    """
    t0 = time.perf_counter()
    if njit is None:
        return
    for prec, cdt in (("f64", np.complex128), ("f32", np.complex64)):
        fdt = np.float32 if cdt is np.complex64 else np.float64
        try:
            if not _numba_selftest(cdt):
                # self-test violato: fallback numpy per questa precisione
                _NUMBA_STATUS[prec] = "self-test di correttezza fallito"
                continue
            # warmup: compila il percorso parallelo a dimensioni realistiche
            c2 = np.zeros((64, 64), dtype=cdt)
            d2 = np.zeros((64, 64), dtype=bool)
            it2 = np.zeros((64, 64), dtype=np.int32)
            mag2 = np.zeros((64, 64), dtype=fdt)
            _mandel_escape(c2, d2, it2, mag2, 32, 0, 64)
            _NUMBA_OK[prec] = True
            _NUMBA_STATUS[prec] = "ok (multi-core)"
        except Exception as ex:
            _NUMBA_STATUS[prec] = "compilazione fallita: " + str(ex)[:160]
    _NUMBA_STATUS["tempo"] = round(time.perf_counter() - t0, 1)

_CPU_WS = {}

def _cpu_ws(w, h, prec):
    # v5.3.0: workspace per PRECISIONE (dtype f32/f64). X/Y restano float64 in
    # entrambi i casi: la griglia e' esatta e il prodotto con half viene
    # calcolato in f64 poi arrotondato al dtype di destinazione (no double
    # rounding).
    fdt = np.float32 if prec == "f32" else np.float64
    cdt = np.complex64 if prec == "f32" else np.complex128
    key = (w, h, prec)
    ws = _CPU_WS.get(key)
    if ws is None:
        ws = {
            "X": np.arange(w) - w / 2,
            "Y": np.arange(h) - h / 2,
            "tx": np.empty(w, dtype=fdt),
            "ty": np.empty(h, dtype=fdt),
            "real": np.empty((h, w), dtype=fdt),
            "imag": np.empty((h, w), dtype=fdt),
            "c": np.empty((h, w), dtype=cdt),
            "w4": np.empty((h, w), dtype=cdt),
            # v5.0.0: z in parti reali/immaginarie (NO np.square: quel ufunc e'
            # compilato da numpy con FMA -> bit diversi dal percorso Numba).
            "zr": np.empty((h, w), dtype=fdt),
            "zi": np.empty((h, w), dtype=fdt),
            "tr": np.empty((h, w), dtype=fdt),
            "ti": np.empty((h, w), dtype=fdt),
            "diverged": np.empty((h, w), dtype=bool),
            "it": np.empty((h, w), dtype=np.int32),
            # v5.2.0: |z|^2 al momento della fuga (coloring smooth, come la GPU);
            # 0 = mai fuggito (interiore).
            "mag": np.empty((h, w), dtype=fdt),
        }
        if len(_CPU_WS) >= 6:
            _CPU_WS.pop(next(iter(_CPU_WS)))
        _CPU_WS[key] = ws
    return ws

def compute_cpu(cx, cy, half, w, h, mi, my_gen=0, prec="f64"):
    # v5.3.0: prec = "f32"/"f64" (default f64). Tutti gli array di lavoro sono
    # del dtype corrispondente; il percorso f64 resta bit-identico al
    # precedente (stesse operazioni, stesso ordine, stesso dtype).
    ws = _cpu_ws(w, h, prec)
    fdt = np.float32 if prec == "f32" else np.float64
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
    mag = ws["mag"]
    mag.fill(0.0)
    # Interior analitico (stesso criterio del kernel GPU): bulbo periodica-2 +
    # cardioide principale (|1 - sqrt(1 - 4c)| < 1). Questi pixel non divergono
    # mai -> restano it=0 (neri) ed esclusi dal loop: ~2.5x sulla CPU.
    w4 = ws["w4"]
    np.multiply(c, -4.0, out=w4)
    w4 += 1.0
    diverged |= (np.abs(w4) < 2.0 * np.sqrt(0.5 * (np.abs(w4) + np.real(w4))))
    diverged |= (np.abs(c + 1.0) <= 0.25)
    if _NUMBA_OK.get(prec):
        # v5.0.0: escape loop parallelo Numba (bit-identico, verificato a
        # runtime dal self-test di _numba_warmup; v5.3.0: il kernel si
        # auto-specializza sul dtype in ingresso, f64 e f32).
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
            _mandel_escape(c, diverged, it, mag, mi, b0, min(b0 + band, h))
    else:
        # Fallback numpy (e riferimento del gate di correttezza).
        # v5.0.0: quadrato complesso in parti esplicite (a*a-b*b, 2*a*b),
        # MAI np.square (quel ufunc e' compilato da numpy con FMA -> bit
        # diversi).
        # v5.3.0: il fallback e' la stessa formula in f32 e f64 (ufunc
        # dtype-agnostic). In f32 NON e' bit-identico al kernel Numba
        # (contrazioni FMA/riassociazioni di Numba/LLVM dipendenti dal
        # contesto del loop) -> differenza di 1-qualche ULP su una piccolissima
        # frazione di pixel di bordo caotico; il self-test lo tollera (it[]
        # esatto, mag[] entro 1e-2 relativo). E' usato solo se Numba e' assente.
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
                mag[m] = zr[m] * zr[m] + zi[m] * zi[m]
    # v5.2.0: coloring IDENTICO al kernel GPU (smooth iteration):
    # nu = it + 1 - log2(0.5*ln|z|^2), t = (nu/mi)^0.35, dove |z|^2 e' il
    # modulo al momento della fuga (mag, > 4). I pixel mai fuggiti
    # (mag == 0) restano neri. (v<=5.1.x: t=(it/mi)^0.35 + rgb[it==0]=0,
    # che confondeva "interiore" con "fuggito alla 1a iterazione" e
    # anneriva la regione lontana, dove la GPU dava il rosso scuro LUT[0..].)
    # NB: log2/log CPU (libm) vs device (CUDA) possono differire di 1-2 ULP
    # -> al massimo 1 entry di LUT di differenza vs GPU f64 (documentato
    # nel gate: check CPU-vs-GPUf64 con bound piccolo).
    esc = mag > 0.0
    nu = np.zeros((h, w), dtype=fdt)
    nu[esc] = it[esc].astype(fdt) + 1.0 - np.log2(0.5 * np.log(mag[esc]))
    t = np.power(np.clip(nu / mi, 0.0, 1.0), 0.35).ravel()
    idx = (t * 255).astype(np.uint8)
    rgb = _LUT[idx].reshape((h, w, 3)).copy()
    rgb[~esc] = 0
    return Image.fromarray(rgb)

# ---------------- Dispatch backend ----------------
# v5.6.0: _ACTIVE (stringa) e' l'unico stato del motore (cpu/cuda/metal/vulkan).

def backend():
    # v5.6.0: 4 backend selezionabili; la precisione (f32/f64) e' comune.
    # Metal/Vulkan sono f32-only, CUDA f32 (+f64 se il kernel e' compilato).
    # v5.6.1: se Numba e' assente la CPU usa il fallback numpy single-core ->
    # lo segnalo nell'etichetta (titolo + barra di stato), prima era silenzioso.
    # v5.8.10: indicazione single/multi-core esplicita e PER PRECISIONE: il
    # fallback numpy scatta per singola precisione (_NUMBA_OK[prec], vedi
    # compute_cpu), quindi l'etichetta segue _NUMBA_OK[_PREC] e non il solo
    # import (_NUMBA_AVAILABLE). Prima del warmup _NUMBA_OK e' False e i primi
    # render usano davvero il fallback -> "single-core" e' veritiero.
    if _ACTIVE != "cpu":
        return _ACTIVE.upper() + " " + _PREC
    if _NUMBA_OK.get(_PREC):
        return "CPU " + _PREC + " multi-core"
    return "CPU " + _PREC + " single-core (numpy)"

_HW_CACHE = {}

def hw_name():
    """Nome dell'hardware attivo (GPU o CPU), in cache per backend (v5.8.0).
    Rilevamento senza dipendenze nuove: CPU dal registro Windows (winreg) o
    'sysctl' su macOS; GPU da CuPy (CUDA), pyobjc (Metal) o wgpu (Vulkan,
    gia' rilevato in VulkanBackend.name). In caso di errore -> generico."""
    if _ACTIVE in _HW_CACHE:
        return _HW_CACHE[_ACTIVE]
    is_cpu = (_ACTIVE == "cpu")
    name = "CPU" if is_cpu else "GPU"
    try:
        import platform
        if is_cpu:
            s = platform.system()
            if s == "Windows":
                import winreg
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                try:
                    name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
                finally:
                    winreg.CloseKey(k)
            elif s == "Darwin":
                import subprocess
                name = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True).strip()
            else:
                name = platform.processor() or name
        elif _ACTIVE == "cuda":
            import cupy as cp
            # v5.9.0: device selezionato (dropdown GPU); con piu' GPU
            # l'ordine CUDA puo' differire da quello nvidia-smi.
            _dev = _CUDA_DEVICES[_CUDA_DEV][0] if _CUDA_DEVICES else 0
            name = cp.cuda.runtime.getDeviceProperties(_dev)["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
        elif _ACTIVE == "metal":
            _n = _METAL_BE.dev.name
            name = str(_n() if callable(_n) else _n)
        elif _ACTIVE == "vulkan":
            name = str(_VULKAN_BE.name)
    except Exception:
        pass
    if not name:
        name = "CPU" if is_cpu else "GPU"
    _HW_CACHE[_ACTIVE] = name
    return name

def compute(cx, cy, half, w, h, mi, buf=None, prec=None, my_gen=0):
    if _ACTIVE != "cpu":
        return compute_gpu(cx, cy, half, w, h, mi, buf=buf, prec=prec)
    # v5.3.0: la precisione selezionata (f32/f64) vale anche per la CPU.
    return compute_cpu(cx, cy, half, w, h, mi, my_gen=my_gen, prec=prec or _PREC)

# v4.16.0 / v5.6.0: warmup GPU in background all'avvio: init del backend GPU
# default (CUDA/Metal/Vulkan), compilazione shader e prime allocazione pagati
# FUORI dal primo render reale (che senza warmup e' 8-17 ms piu' lento).
def _gpu_warmup():
    try:
        compute_gpu(CX0, CY0, HALF0, 64, 64, 64, prec="f32")
        if _gpu_supports_f64():
            compute_gpu(CX0, CY0, HALF0, 64, 64, 64, prec="f64")
    except Exception:
        pass

def _warmup_cuda_device():
    # v5.9.0: scalda il device appena selezionato (compilazione kernel +
    # prime allocazioni fuori dal primo render). Solo se CUDA e' attivo.
    if _ACTIVE != "cuda":
        return
    try:
        compute_gpu(CX0, CY0, HALF0, 64, 64, 64, prec="f32")
    except Exception:
        pass

def _warmup_vulkan_adapter():
    # v5.9.2: come sopra per l'adapter Vulkan appena selezionato.
    if _ACTIVE != "vulkan":
        return
    try:
        compute_gpu(CX0, CY0, HALF0, 64, 64, 64, prec="f32")
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
        # v5.6.0: 4 radio (CPU/CUDA/Metal/Vulkan) nel MESSIMO frame = gruppo
        # radio unico (Tkinter li raggruppa da solo, senza 'variable'). I
        # backend non disponibili restano visibili ma DISABILITATI (grigi).
        self.backend_btns = {}
        for _be in ("cpu", "cuda", "metal", "vulkan"):
            _b = tk.Radiobutton(bk, text=_be.capitalize(), value=_be,
                               command=lambda b=_be: self.set_backend(b))
            _b.pack(side="left", padx=2, pady=3)
            self.backend_btns[_be] = _b
        for _be, _b in self.backend_btns.items():
            if not _backend_ok(_be):
                _b.config(state="disabled")
        self.backend_btns[_ACTIVE].select()
        # v5.9.0/v5.9.2: dropdown scelta GPU: contenuto per motore attivo
        # (device CUDA / adapter Vulkan), visibile solo se > 1.
        self.gpu_frame = tk.Frame(bk)
        tk.Label(self.gpu_frame, text="GPU:").pack(side="left")
        self.gpu_var = None
        self.gpu_menu = None
        self._refresh_gpu_menu()
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
        # v5.4.0: la disponibilita' di f64 dipende dallo slot GPU corrente
        # (Metal e' f32-only; CUDA senza kernel f64). Stato dinamico: si
        # aggiorna a ogni cambio di motore (vedi _sync_precision_buttons).
        self._sync_precision_buttons()
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
        self.recalc_btn = tk.Button(self.btns, text="Ricalcola", command=self.recalc)
        self.recalc_btn.pack(side="right", padx=2, pady=3)
        self.photo2_btn = tk.Button(self.btns, text="Foto 2x2",
                                    command=lambda: self.take_photo(2))
        self.photo2_btn.pack(side="right", padx=2, pady=3)
        self.photo4_btn = tk.Button(self.btns, text="Foto 4x4",
                                    command=lambda: self.take_photo(4))
        self.photo4_btn.pack(side="right", padx=2, pady=3)

    def _build_canvas_status(self):
        # --- barra accento, canvas al centro, status in fondo ---
        self.accent = tk.Frame(self.root, height=3, bg=_backend_fg())
        self.accent.pack(fill="x")
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(self.root, text="render...")
        self.status.pack(fill="x")

    def _build_menu(self):
        self.menu = tk.Menu(self.root)
        self.mfile = tk.Menu(self.menu, tearoff=0)
        self.mfile.add_command(label="Salva immagine... (Ctrl+S)", command=self.save_png)
        self.mfile.add_command(label="Carica zona...", command=self.load_zone_as)
        self.mfile.add_command(label="Salva zona", command=self.save_zone)
        self.save_zone_entry = self.mfile.index(tk.END)
        self.mfile.add_command(label="Salva zona con nome...", command=self.save_zone_as)
        self.mfile.add_separator()
        self.mfile.add_command(label="Esci", command=self.on_exit)
        self.menu.add_cascade(label="File", menu=self.mfile)
        self.mhelp = tk.Menu(self.menu, tearoff=0)
        self.mhelp.add_command(label="Istruzioni...", command=self.show_help)
        self.mhelp.add_command(label="Novità recenti...", command=self.show_recent)
        self.mhelp.add_separator()
        self.mhelp.add_command(label="Informazioni...", command=self.show_about)
        self.menu.add_cascade(label="Help", menu=self.mhelp)
        self.root.config(menu=self.menu)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self._update_save_zone_state()

    def _update_save_zone_state(self):
        # "Salva zona" e' disponibile solo se esiste gia' un nome/file di zona
        # (view_file); altrimenti va prima definito con "Salva zona con nome..."
        # o con "Carica zona...".
        state = "normal" if self.view_file else "disabled"
        self.mfile.entryconfig(self.save_zone_entry, state=state)

    def show_help(self):
        # Guida rapida (menu Help -> Istruzioni...).
        win = tk.Toplevel(self.root)
        win.title("Istruzioni")
        win.resizable(False, False)
        body = tk.Frame(win, padx=26, pady=20)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Insieme di Mandelbrot \u2014 istruzioni",
                 font=("Segoe UI", 14, "bold"),
                 foreground=_backend_fg()).pack(anchor="w")
        text = (
            "Navigazione: rotella per zoomare al cursore (x1.25 / x0.8), "
            "click per zoom x2 al cursore, click destro (due dita sul trackpad) "
            "per zoom x0.5 al cursore, trascinamento per spostare la vista, "
            "tasti + / - per zoom x2 / x0.5 al centro (funzionano sempre, "
            "tranne mentre si scrive nelle caselle), r per il reset.\n\n"
            "Motore e precisione: toolbar Motore (CPU / CUDA / Metal / Vulkan) "
            "e Precisione (f32 / f64); i backend non disponibili restano grigi. "
            "Con piu' GPU, il dropdown GPU sceglie device/adapter del motore. "
            "CUDA richiede driver NVIDIA + runtime; Vulkan funziona out-of-the-box.\n\n"
            "Iterazioni: Auto calcola mi dallo zoom; in manuale si imposta il "
            "valore (Invio) o lo si cambia di \u00b11000.\n\n"
            "Palette, file, foto e benchmark: palette dal registro (default fuoco); "
            "menu File per salvare PNG (Ctrl+S) e zone JSON; Foto 2x2 / Foto 4x4 "
            "ricalcolano la vista a 2x / 4x per lato e la mostrano con "
            "antialiasing (media 2x2 / 4x4, da non toccare durante il calcolo; "
            "4x4 = 16x pixel, molto piu' lento); Ricalcola rifa il rendering "
            "della vista corrente; Benchmark esegue "
            "il test standardizzato di 8 s nella vista corrente."
        )
        tk.Label(body, text=text, wraplength=460, justify="left").pack(anchor="w", pady=(8, 0))
        def chiudi(_e=None):
            win._annullato = False
            win.destroy()
        tk.Button(win, text="Chiudi", command=chiudi).pack(pady=(18, 20))
        win.bind("<Return>", chiudi)
        self._modal(win)

    def show_recent(self):
        # Ultime 10 modifiche (menu Help -> Novità recenti...), da HISTORY.
        win = tk.Toplevel(self.root)
        win.title("Novità recenti")
        win.resizable(False, False)
        body = tk.Frame(win, padx=26, pady=20)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Ultime modifiche",
                 font=("Segoe UI", 14, "bold"),
                 foreground=_backend_fg()).pack(anchor="w")
        rows = tk.Frame(body)
        rows.pack(fill="x", pady=(8, 0))
        for i, (ver, date, desc) in enumerate(HISTORY[:10]):
            tk.Label(rows, text=f"v{ver} ({date})", width=18, anchor="w",
                     font=("Consolas", 11, "bold")).grid(row=i, column=0, sticky="nw", pady=2)
            tk.Label(rows, text=desc, wraplength=340, justify="left",
                     anchor="w").grid(row=i, column=1, sticky="nw", pady=2)
        def chiudi(_e=None):
            win._annullato = False
            win.destroy()
        tk.Button(win, text="Chiudi", command=chiudi).pack(pady=(18, 20))
        win.bind("<Return>", chiudi)
        self._modal(win)

    def show_about(self):
        # Scheda autore/versione (menu Help -> Informazioni...).
        win = tk.Toplevel(self.root)
        win.title("Informazioni")
        win.resizable(False, False)
        body = tk.Frame(win, padx=26, pady=20)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Insieme di Mandelbrot",
                 font=("Segoe UI", 14, "bold"),
                 foreground=_backend_fg()).pack(anchor="w")
        tk.Label(body, text=f"Versione {VERSION}").pack(anchor="w", pady=(6, 0))
        tk.Label(body, text=f"{backend()} ({hw_name()})").pack(anchor="w")
        tk.Label(body, text="Autore: Francesco Ferrara").pack(anchor="w", pady=(10, 0))
        tk.Label(body, text="Email: occhiobello@gmail.com").pack(anchor="w")
        # v5.9.5: diagnostica CPU/Numba (perche' single-core?).
        tk.Label(body, text="CPU: " + _numba_diag(), wraplength=460,
                 justify="left").pack(anchor="w", pady=(10, 0))
        def chiudi(_e=None):
            win._annullato = False
            win.destroy()
        tk.Button(win, text="Chiudi", command=chiudi).pack(pady=(18, 20))
        win.bind("<Return>", chiudi)
        self._modal(win)

    def _bind_events(self):
        self.press_pos = None
        self.dragged = False
        self._size = (0, 0)
        self.root.bind("<Control-s>", lambda e: self.save_png())
        # v5.9.8: tasti zoom/reset legati alla FINESTRA (non al canvas):
        # prima valevano solo col focus sul canvas, dopo un click su
        # bottoni/caselle sembravano morti (tipico su macOS). La guardia
        # in _key_zoom evita di rubare i tasti mentre si scrive in entry.
        self.root.bind("<Key-r>", lambda e: self._key_reset())
        self.root.bind("<Key-plus>", lambda e: self._key_zoom(2.0))
        self.root.bind("<Key-minus>", lambda e: self._key_zoom(0.5))
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        # v5.9.8: click destro (due dita sul trackpad) = zoom x0.5 al
        # cursore: alternativa alla rotella che non dipende dagli eventi
        # MouseWheel. Bind su 2 e 3 (mapping destro su macOS/X11).
        self.canvas.bind("<ButtonPress-2>",
                         lambda e: self.zoom_at(*self.p2c(e.x, e.y), 0.5))
        self.canvas.bind("<ButtonPress-3>",
                         lambda e: self.zoom_at(*self.p2c(e.x, e.y), 0.5))
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Configure>", self.on_configure)

    def _key_zoom(self, f):
        # Zoom da tastiera: ignorato solo mentre il focus e' su una Entry
        # (l'utente sta digitando, es. '-' nelle iterazioni).
        try:
            if isinstance(self.root.focus_get(), tk.Entry):
                return
        except Exception:
            pass
        self.zoom_center(f)

    def _key_reset(self):
        try:
            if isinstance(self.root.focus_get(), tk.Entry):
                return
        except Exception:
            pass
        self.reset()

    # ---------------- Helper UI ----------------
    def _refresh_title(self):
        t = f"Insieme di Mandelbrot v{VERSION} - {backend()} ({hw_name()})"
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
        global _ACTIVE
        # v5.6.0: 4 backend (cpu/cuda/metal/vulkan); selezionabile solo se
        # disponibile all'avvio (gli altri restano grigi/disabilitati).
        if not _backend_ok(be):
            return False
        _ACTIVE = be
        for b in self.backend_btns.values():
            b.deselect()
        self.backend_btns[be].select()
        if getattr(self, "accent", None) is not None:
            self.accent.config(bg=_backend_fg())
        return True

    def _select_precision(self, p):
        if not set_prec(p):
            return False
        self.f32_btn.deselect()
        self.f64_btn.deselect()
        (self.f32_btn if p == "f32" else self.f64_btn).select()
        return True

    def _sync_precision_buttons(self):
        # v5.4.0: la disponibilita' di f64 dipende dallo slot GPU corrente:
        # Metal e' f32-only, CUDA senza kernel f64 non fa f64. Con CPU f32/f64
        # sono sempre selezionabili. Se la precisione corrente non e' piu'
        # disponibile si torna a f32; il bottone selezionato riflette _PREC.
        f64_ok = (_ACTIVE == "cpu") or _gpu_supports_f64()
        if not f64_ok and _PREC == "f64":
            set_prec("f32")
        self.f64_btn.config(state="normal" if f64_ok else "disabled")
        self.f32_btn.deselect()
        self.f64_btn.deselect()
        (self.f32_btn if _PREC == "f32" else self.f64_btn).select()

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
        try:
            self.canvas.focus_set()
        except Exception:
            pass
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
            self.status.config(text="MI invalidi: intero tra 50 e 100000",
                                   foreground=ERR_FG)
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
        self._refresh_gpu_menu()
        self._sync_precision_buttons()
        self._refresh_title()
        self.request_render("motore: " + b.upper())

    def _gpu_labels(self):
        # v5.9.2: voci del dropdown per il motore attivo (vuoto = nascosto).
        if _ACTIVE == "cuda" and len(_CUDA_DEVICES) > 1:
            return [_cuda_label(i) for i in range(len(_CUDA_DEVICES))]
        if _ACTIVE == "vulkan" and len(_VULKAN_ADAPTERS) > 1:
            return [_vulkan_label(i) for i in range(len(_VULKAN_ADAPTERS))]
        return []

    def _gpu_pos(self):
        return _VULKAN_DEV if _ACTIVE == "vulkan" else _CUDA_DEV

    def _refresh_gpu_menu(self):
        # v5.9.2: ricostruisce il dropdown per il motore attivo (set_backend,
        # load config, reset); nascosto se il motore ha <= 1 GPU.
        if self.gpu_menu is not None:
            self.gpu_menu.destroy()
            self.gpu_menu = None
            self.gpu_var = None
        labels = self._gpu_labels()
        if not labels:
            self.gpu_frame.pack_forget()
            return
        self.gpu_frame.pack(side="left", padx=(10, 0))
        self.gpu_var = tk.StringVar(value=labels[self._gpu_pos()])
        self.gpu_menu = tk.OptionMenu(self.gpu_frame, self.gpu_var, *labels,
                                      command=self.choose_gpu_device)
        self.gpu_menu.pack(side="left", padx=2, pady=3)

    def _sync_gpu_menu(self):
        # v5.9.0: allinea il valore senza ricostruire (dopo una selezione).
        labels = self._gpu_labels()
        if self.gpu_var is not None and labels:
            self.gpu_var.set(labels[self._gpu_pos()])

    def _select_cuda_device(self, i):
        if not set_cuda_device(i):
            return False
        self._sync_gpu_menu()
        return True

    def _select_vulkan_adapter(self, i):
        # v5.9.2: come sopra per l'adapter Vulkan.
        if not set_vulkan_adapter(i):
            return False
        self._sync_gpu_menu()
        return True

    def choose_gpu_device(self, label):
        # v5.9.0/v5.9.2: cambio GPU dal dropdown ("<id>: <nome>").
        try:
            _id = int(str(label).split(":", 1)[0])
        except (TypeError, ValueError):
            self._sync_gpu_menu()
            return
        if _ACTIVE == "vulkan":
            if 0 <= _id < len(_VULKAN_ADAPTERS) and _id != _VULKAN_DEV:
                self._select_vulkan_adapter(_id)
                self._refresh_title()
                threading.Thread(target=_warmup_vulkan_adapter,
                                 daemon=True).start()
                self.request_render("gpu: " + _vulkan_short_name(
                    _VULKAN_ADAPTERS[_VULKAN_DEV][1]))
            else:
                self._sync_gpu_menu()
            return
        _pos = next((k for k, (d, _n) in enumerate(_CUDA_DEVICES) if d == _id),
                    None)
        if _pos is None or _pos == _CUDA_DEV:
            self._sync_gpu_menu()
            return
        self._select_cuda_device(_pos)
        self._refresh_title()
        threading.Thread(target=_warmup_cuda_device, daemon=True).start()
        self.request_render("gpu: " + _cuda_short_name(_CUDA_DEVICES[_CUDA_DEV][1]))

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
        self._select_backend(_default_backend())
        self._select_cuda_device(0)
        self._select_vulkan_adapter(0)
        self._refresh_gpu_menu()
        self._select_precision("f32")
        self._sync_precision_buttons()
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
        self._photo_running = False
        self._photo_result = None
        self._photo_finished = False
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
            # v5.9.1: durante il benchmark i render interattivi sono sospesi
            # (niente contesa col thread bench sullo stesso device: su una
            # GPU display, kernel pesante in vista + benchmark insieme
            # rischiava TDR/reset -> cudaErrorDevicesUnavailable).
            if self._bench_running:
                continue
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
        if self._photo_finished:
            self._photo_finished = False
            img, view, n, rt, err = self._photo_result
            self._photo_done(img, view, n, rt, err)
        self.root.after(30, self._poll)

    def _show(self, img, msg, rt=0.0):
        w, h = self.canvas_size()
        if img.size != (w, h):
            img = img.resize((w, h), Image.NEAREST)
        self.pil = img
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.photo)
        # v5.9.5: se la CPU resta in single-core, avviso persistente (rosso)
        # col rimando al motivo (si auto-cancella al passaggio a multi-core).
        _single = (_ACTIVE == "cpu" and not _NUMBA_OK.get(_PREC))
        _suffix = (" \u00b7 single-core: motivo in Help > Informazioni"
                   if _single else "")
        self.status.config(text=f"{msg} | {backend()} \u00b7 {hw_name()} | palette: {_PALETTE} | render: {rt*1000:.0f} ms{_suffix}",
                           foreground=ERR_FG if _single else _backend_fg())
        # v5.8.10: il titolo segue il warmup Numba (single->multi a warmup
        # concluso); backend() e' ricalcolato a ogni frame, il titolo no.
        self._refresh_title()

    # ---------------- File: PNG, zona (JSON), config ----------------
    def save_png(self):
        if getattr(self, "pil", None) is None:
            self.status.config(text="niente immagine da salvare",
                                   foreground=ERR_FG)
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
            self.status.config(text="errore salvataggio: " + str(ex),
                                   foreground=ERR_FG)

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
            self._update_save_zone_state()
            self.status.config(text="zona salvata: " + path)
        except Exception as ex:
            self.status.config(text="errore salvataggio zona: " + str(ex),
                                   foreground=ERR_FG)

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
            self.status.config(text="errore caricamento zona: " + str(ex),
                                   foreground=ERR_FG)
            return
        self.mi_auto_var.set(self.mi_auto)
        st = "disabled" if self.mi_auto else "normal"
        self.mi_minus.config(state=st)
        self.mi_plus.config(state=st)
        self.view_file = path
        self._refresh_title()
        self._update_save_zone_state()
        self.request_render("zona caricata: " + os.path.basename(path))

    def save_config(self):
        # v5.1.2: la VISTA (cx/cy/half/mi) non e' piu' persistita in config:
        # l'app parte sempre con la configurazione di default (intero insieme,
        # MI auto); la vista si salva solo col file zona ('Salva zona').
        # NB: view_file NON e' persistito (v5.1.1): 'Salva zona' chiede sempre
        # il nome finche' non si carica/salva una zona in quella sessione.
        # v5.6.0: salvo il backend ATTIVO per nome (cpu/cuda/metal/vulkan).
        c = dict(precision=_PREC, palette=_PALETTE,
                 backend=_ACTIVE,
                 cuda_device=_CUDA_DEV,
                 vulkan_adapter=_VULKAN_DEV,
                 bench=dict(self.bench))
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
        # v5.1.2: la vista (cx, cy, half, mi, mi_auto) NON viene ripristinata:
        # l'app parte SEMPRE con la configurazione di default (intero insieme,
        # MI auto, come la prima volta). I vecchi valori in config sono
        # ignorati; la vista si recupera solo con 'Carica zona...'.
        # v5.1.1: view_file non viene ripristinato: 'Salva zona' chiede sempre
        # il nome finche' non si carica/salva una zona in questa sessione.
        self._load_bench(c.get("bench"))
        self._select_palette(c.get("palette", "fuoco"))
        # v5.6.0: il backend e' per nome (cpu/cuda/metal/vulkan); i vecchi
        # valori "gpu" (v5.4.x/5.5.0) migrano al default GPU. _select_backend
        # scarta (restituendo False) i backend non disponibili, restando sul
        # default di avvio (che e' sempre disponibile).
        be = c.get("backend", _default_backend())
        if be == "gpu":
            be = _default_backend()
        self._select_backend(be)
        # v5.9.0: device CUDA persistito (set_cuda_device clamp-a il range).
        self._select_cuda_device(c.get("cuda_device", 0))
        # v5.9.2: adapter Vulkan persistito (idem); poi il dropdown segue il
        # motore attivo (potrebbe cambiare contenuto rispetto all'avvio).
        self._select_vulkan_adapter(c.get("vulkan_adapter", 0))
        self._refresh_gpu_menu()
        # la precisione va DOPO il motore: la disponibilita' di f64 dipende
        # dallo slot GPU corrente (set_prec la rifiuta se lo slot non la fa).
        self._select_precision(c.get("precision", "f32"))
        self._sync_precision_buttons()
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
            ("Hardware", hw_name()),
            ("Durata", f"{b['secs']:.0f} s"),
        ]

    def _bench_chart(self, parent, rps):
        """Grafico a barre orizzontali del benchmark: i riferimenti storici
        (BENCH_REF, con nome hardware) + la run corrente (rps) evidenziata, col
        solo metodo attivo nell'etichetta (es. "CUDA f32 - questa run"). Il nome
        hardware della run corrente NON e' nel grafico (v5.8.1: mostrato sotto
        dal dialog chiamante). Scala orizzontale lineare automatica sul massimo
        dei valori; altezza in base al numero di barre. Ritorna il Canvas
        (da packare da chi chiama)."""
        bars = list(BENCH_REF)
        # v5.8.1: la run corrente mostra solo il metodo (backend); il nome
        # hardware NON e' nel grafico -> mostrato sotto dal dialog chiamante.
        bars.append((backend() + " \u2014 questa run",
                      float(rps)))
        raw_max = max(v for _, v in bars)
        # step "nice" per ~5 divisioni (serie 1/2/5 x 10^n); l'asse termina
        # esattamente sull'ultimo tick >= massimo (mai oltre)
        raw = raw_max / 5.0
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m2 * mag for m2 in (1.0, 2.0, 5.0, 10.0) if m2 * mag >= raw)
        nt = max(1, int(math.ceil(raw_max / step - 1e-9)))
        axis_max = nt * step
        W = 590
        L, R, T, B = 230, 60, 8, 26
        pw = W - L - R
        bar_h, gap = 20, 10
        n = len(bars)
        total = n * bar_h + (n - 1) * gap
        # v5.9.1: altezza dinamica (150px fissi non bastavano per 6 barre).
        H = T + total + B
        y0 = T
        cv = tk.Canvas(parent, width=W, height=H, highlightthickness=0)
        def x(v):
            return L + pw * v / axis_max
        for k in range(nt + 1):
            t = k * step
            xt = x(t)
            cv.create_line(xt, T, xt, H - B, fill="#a0a0a0")
            cv.create_text(xt, H - B + 12, text="%g" % t, font=("Consolas", 9))
        cv.create_line(L, H - B, W - R, H - B, fill="#8a8a8a")
        for i, (name, v) in enumerate(bars):
            y = y0 + i * (bar_h + gap)
            cur = (i == n - 1)
            col = "#2ea44f" if cur else "#8a8a8a"
            cv.create_text(L - 8, y + bar_h / 2, text=name, anchor="e",
                           font=("Segoe UI", 9, "bold" if cur else "normal"))
            xe = max(x(v), L + 1)
            cv.create_rectangle(L, y, xe, y + bar_h, fill=col, outline="")
            cv.create_text(xe + 6, y + bar_h / 2, text="%g" % v, anchor="w",
                           font=("Consolas", 10, "bold"))
        return cv

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
        tk.Label(body, text="Parametri del test (regione e durata comparabili tra versioni):").pack(anchor="w", pady=(2, 12))
        rows = tk.Frame(body)
        rows.pack(fill="x")
        for i, (k, v) in enumerate(self._bench_rows()):
            tk.Label(rows, text=k, width=12, anchor="w").grid(row=i, column=0, sticky="w", pady=3)
            tk.Label(rows, text=v, anchor="e",
                     font=("Consolas", 12)).grid(row=i, column=1, sticky="e",
                                                  padx=(18, 0), pady=3)
        btns = tk.Frame(win)
        btns.pack(fill="x", padx=26, pady=(16, 18))

        def go(_e=None):
            win._annullato = False
            win.destroy()
        def cancel(_e=None):
            win._annullato = True
            win.destroy()
        annulla = tk.Button(btns, text="Annulla", command=cancel)
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
            tk.Frame(body, bg="#8a8a8a", height=1).pack(fill="x", pady=(10, 0))
            tk.Label(body, text=f"{count/secs:.2f}",
                     font=("Segoe UI", 42, "bold"),
                     foreground="#2ea44f").pack(pady=(16, 0))
            tk.Label(body, text="rendering / secondo",
                     font=("Segoe UI", 13, "bold"),
                     foreground="#2ea44f").pack(pady=(0, 8))
            tk.Label(body, text=f"{count} rendering in {secs:.1f} s   \u00b7   {secs/count*1000:.0f} ms ciascuno",
            ).pack(pady=(0, 16))
            tk.Label(body, text="Confronto coi riferimenti storici (rendering/s):",
                     ).pack(anchor="w", pady=(0, 4))
            self._bench_chart(body, count / secs).pack(anchor="w", pady=(2, 14))
        else:
            tk.Label(body, text="BENCHMARK FALLITO",
                     font=("Segoe UI", 20, "bold"),
                     foreground="#e5534b").pack(pady=(16, 6))
            tk.Label(body, text=str(err), foreground="#e5534b",
                     wraplength=420, justify="left").pack(anchor="w", pady=(0, 16))
            # v5.9.1: hint operativo se il device e' occupato o in reset (TDR).
            if err is not None and ("DevicesUnavailable" in str(err)
                                    or "busy or unavailable" in str(err)):
                tk.Label(body, text="Device CUDA occupato o in reset: riprova "
                         "tra poco o riavvia l'app prima di rilanciare.",
                         wraplength=420, justify="left").pack(anchor="w",
                                                             pady=(0, 16))
        tk.Label(body, text="Parametri del test:").pack(anchor="w", pady=(0, 4))
        rows = tk.Frame(body)
        rows.pack(fill="x", anchor="w")
        for i, (k, v) in enumerate(self._bench_rows()):
            tk.Label(rows, text=k, width=12, anchor="w").grid(row=i, column=0, sticky="w", pady=2)
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
        self.root.config(cursor="watch")
        self.status.config(text=f"benchmark in corso ({self.bench['secs']:.0f} s)...")
        threading.Thread(target=self._bench_worker, daemon=True).start()

    def _bench_worker(self):
        b = self.bench
        mi = auto_mi(b["half"])
        need = b["w"] * b["h"] * 3
        bench_buf = None
        # v5.6.0: il buffer di lavoro serve solo al percorso CUDA (Metal/Vulkan
        # usano il proprio buffer, CPU la memoria numpy).
        if _ACTIVE == "cuda":
            try:
                import cupy as cp
                # v5.9.0: buffer sul device selezionato (per-thread).
                _dev = _CUDA_DEVICES[_CUDA_DEV][0] if _CUDA_DEVICES else 0
                with cp.cuda.Device(_dev):
                    bench_buf = cp.empty((need,), dtype=cp.uint8)
            except Exception:
                bench_buf = None
        def render():
            # v5.1.0: modalita' CORRENTE (motore+precisione selezionati), non
            # piu' CUDA f32 fisso. compute() dispatcha su _ACTIVE e usa
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
        self.root.config(cursor="")
        self.status.config(text="benchmark completato")
        self._bench_result_dialog(count, secs, err)

    def recalc(self):
        # v5.10.0: rifa il rendering della vista corrente (preview + full).
        self.request_render("ricalcolo manuale")

    def take_photo(self, n=2):
        # v5.9.6: foto antialiasing: ricalcola la vista corrente a NxN per
        # lato e mostra la media NxN dei pixel vicini. Eseguita
        # in background (come il benchmark): cursore hourglass, UI viva.
        # Per vedere la foto bisogna aspettare senza toccare: se vista,
        # palette, motore o precisione cambiano durante il calcolo, la
        # foto stantia viene scartata.
        # v5.9.7: all'avvio invalida i render interattivi in volo/pendenti
        # (stessa vista): bump di _GEN (ferma le bande CPU, fa scartare il
        # frame al worker) + cancella il full-render ritardato di
        # request_render, altrimenti sovrascriverebbero la foto con la
        # versione non antialiased.
        # v5.10.0: fattore N parametrico (2 = Foto 2x2, 4 = Foto 4x4):
        # 4x4 = 16x pixel, molto piu' lento e pesante in memoria.
        if getattr(self, "_photo_running", False):
            self.status.config(text="foto gia' in corso")
            return
        w, h = self.canvas_size()
        view = (self.cx, self.cy, self.half, self.eff_mi(),
                _PALETTE, _ACTIVE, _PREC)
        if self._full_timer is not None:
            self.root.after_cancel(self._full_timer)
            self._full_timer = None
        self._gen += 1
        _GEN[0] = self._gen
        self._photo_running = True
        for b in (self.photo2_btn, self.photo4_btn):
            b.config(state="disabled")
        self.root.config(cursor="watch")
        self.status.config(text=f"foto in corso (antialiasing {n}x{n}, non toccare)...")
        threading.Thread(target=self._photo_worker, args=(view, w, h, n),
                         daemon=True).start()

    def _photo_worker(self, view, w, h, n):
        # Box-filter NxN sullo spazio RGB: unico code-path per tutti i
        # backend (la GPU colora in-kernel, nessun it/mag su host).
        # Nessuna cancellazione durante il calcolo (il risultato stantio
        # e' scartato in _photo_done, non qui).
        t0 = time.perf_counter()
        try:
            big = compute(view[0], view[1], view[2], n * w, n * h, view[3])
            if big.size != (n * w, n * h):
                big = big.resize((n * w, n * h), Image.BILINEAR)
            a = np.asarray(big)
            small = (a.reshape(h, n, w, n, 3).mean(axis=(1, 3)) + 0.5).astype(np.uint8)
            img = Image.fromarray(small, "RGB")
            self._photo_result = (img, view, n, time.perf_counter() - t0, None)
        except Exception as ex:
            self._photo_result = (None, view, n, 0.0, str(ex))
        self._photo_finished = True

    def _photo_done(self, img, view, n, rt, err):
        # Unico punto di uscita (anche in errore): ripristina sempre
        # cursore e pulsanti, come _bench_done.
        self._photo_result = None
        self._photo_running = False
        for b in (self.photo2_btn, self.photo4_btn):
            b.config(state="normal")
        self.root.config(cursor="")
        if err:
            self.status.config(text=f"foto fallita: {err}")
        elif (self.cx, self.cy, self.half, self.eff_mi(),
                _PALETTE, _ACTIVE, _PREC) != view:
            self.status.config(text="foto scartata (vista cambiata)")
        else:
            self._show(img, f"foto antialiasing {n}x{n}", rt)


def main():
    root = tk.Tk()
    MandelbrotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


