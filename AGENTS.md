# Convenzioni del progetto

## Struttura del progetto
- `mandel.py` — il programma (visualizzatore interattivo, Tkinter + 4 backend
  selezionabili: CPU (numpy/Numba), CUDA (CuPy), Metal (pyobjc), Vulkan (wgpu),
  con fallback CPU); versione in **due punti** che devono coincidere (header
  `# VERSIONE:` per build + costante runtime `VERSION` per titolo/JSON) e
  STORICO nel commento iniziale del file (fonte di verità).
- `spec.md` — specifica del programma; DEVE restare in sync col sorgente
   (vedi regola di versionamento sotto).
- `mandelbrot_*.json` — zone salvate dall'app (dato utente, gitignorato).
- `build_app.py` — script di build self-contained (one-dir) multipiattaforma
  (icona → PyInstaller → post). `mandelbrot.spec` = ricetta PyInstaller
  multipiattaforma; `hook_dlldir.py` = runtime hook Windows (percorso DLL per CuPy);
  `make_icon.py` = genera `icon_src.png` + `mandelbrot.ico` (+ `mandelbrot.icns` su macOS).

## Versionamento sorgente (OBBLIGATORIO)
Ogni modifica a `mandel.py` (o ad altri sorgenti Python del
progetto) DEVE:
 1. Incrementare la versione in `mandel.py` in **entrambi** i punti, che
     DEVONO coincidere (se divergono, il titolo/JSON mostrano una versione
     diversa da EXE/.app/zip):
       - l'header `# VERSIONE: X.Y.Z` nel blocco di commento iniziale (è quello
         che `mandelbrot.spec`/`build_app.py` leggono per EXE/.app/zip);
       - la costante runtime `VERSION = "X.Y.Z"` (usata in titolo finestra +
         JSON zona).
     - bugfix = patch (x.y.Z), modifica minima (es. un dialog in piu, un testo,
       un opzione) = patch (x.y.Z), modifica funzionale = minor (x.Y.0),
       riscrittura/architettura = major (X.0.0)
2. Aggiungere una voce allo STORICO nello stesso blocco, formato:
   `VERSIONE - AAAA-MM-GG - descrizione delle modifiche`.
3. Aggiornare `spec.md` in modo che rispecchi il nuovo stato del programma
   (la spec DEVE restare sempre in sincronia col sorgente).
Il blocco è intitolato "Insieme di Mandelbrot - visualizzatore interattivo"
e si trova in cima al file.

## Build dell'app (multipiattaforma, OBBLIGATORIA prima di ogni commit)
Prima di committare DEVE essere rigenerata l'app self-contained (one-dir, con
`python build_app.py` — solo su richiesta, non automatica), così l'artefatto
riflette il sorgente. Intermedie (build/) e app one-dir vanno sul **disco di
sistema** (temp utente); sul **progetto/NAS** resta solo l'artefatto distributivo:
- **Windows** → app `Mandelbrot/Mandelbrot.exe` + `_internal/` su disco di sistema
  (CPU sempre; GPU Vulkan (wgpu) bundled e subito disponibile; GPU CuPy incluso ma
  **runtime CUDA NON bundled**: la GPU CUDA funziona solo se l'utente installa
  driver NVIDIA + runtime CUDA, che CuPy trova via cuda-pathfinder da
  `CUDA_PATH`/PATH/Program Files) + zip `dist/Mandelbrot-v<ver>-win64.zip` sul NAS;
- **macOS** → app `Mandelbrot.app` su disco di sistema (GPU Metal/pyobjc +
  Vulkan/wgpu, firma ad-hoc) + `.dmg` `dist/Mandelbrot-v<ver>-macos.dmg` sul NAS.
- **Regola generale (ritenzione artefatti)**: su `dist/` (progetto/NAS) tenere
  SOLO le ultime **3 versioni** dei distributivi (`KEEP_N` in `build_app.py`);
  a ogni build `build_app.py` rimuove automaticamente gli zip/.dmg delle versioni
  piu' vecchie. Per cambiarne il numero, modificare `KEEP_N`.
- Ordine: modifica sorgente → **`python build_app.py`** → (verifica) → `git commit`.
- File: `build_app.py` (unico script, ramificato su `sys.platform`: icona →
  PyInstaller → post → zip su Windows / .dmg su macOS) + `mandelbrot.spec` (ricetta
  PyInstaller multipiattaforma) + `hook_dlldir.py` (runtime hook Windows:
  `os.add_dll_directory` su cartella EXE + `_internal`, difensivo per le DLL
  bundled) + `make_icon.py` (`icon_src.png` + `mandelbrot.ico` + `mandelbrot.icns` su macOS).
- `dist/`, `build/`, `mandelbrot.icns`, `mandelbrot.ico`, `icon_src.png` sono
  gitignorati: l'app NON va committata, ma va SEMPRE rigenerata prima del commit
  per lasciarla aggiornata.
- Dettaglio tecnico: sezione "Build dell'app (multipiattaforma)" di `spec.md`.

## Note operative
- Gotchas di implementazione (bit-identità, numpy FMA, Numba, vincoli CuPy,
  Metal/pyobjc (v5.4.0), `__constant__`, pinned memory): vedi la sezione "Note
  tecniche" di `spec.md` (casa unica, più dettagliata).
- Ambiente: la GPU è condivisa con il server LLM locale (llama.cpp) che NON va
  fermato (l'agente gira su quello stesso server). I benchmark GPU hanno rumore di
  fondo: usare workload bounded (n launch fissi, non loop a tempo) e confrontare
  versioni a parità di condizioni (o meglio: A/B nello stesso processo alternato).
- Dopo pause lunghe (>~10 s, es. render CPU da 3–18 s) il GPU torna a clock idle: la
  prima misura GPU è 10–15× più lenta del valore a regime. Scalare il clock con un
  breve burst prima di misurare.
