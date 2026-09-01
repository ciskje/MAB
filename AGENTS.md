# Convenzioni del progetto

## Struttura del progetto
- `mandel.py` — il programma (visualizzatore interattivo, Tkinter + GPU
  (CUDA/CuPy o Metal/pyobjc, slot "GPU" generico v5.4.0) + Numba);
  versione e STORICO nel commento iniziale del file (fonte di verità).
- `spec.md` — specifica del programma; DEVE restare in sync col sorgente
   (vedi regola di versionamento sotto).
- `mandelbrot_*.json` — zone salvate dall'app (dato utente, gitignorato).
- `build_app.py` — script di build self-contained (one-dir) multipiattaforma
  (icona → PyInstaller → post); `build_app.sh` lo redirect, `build_app.ps1` wrapper
  Windows. `mandelbrot.spec` = ricetta PyInstaller multipiattaforma; `hook_dlldir.py`
  = runtime hook Windows (percorso DLL per CuPy); `make_icon.py` = genera
  `icon_src.png` + `mandelbrot.ico`.

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

## Build dell'app (multipiattaforma, OBBLIGATORIA prima di ogni commit)
Prima di committare DEVE essere rigenerata l'app self-contained (one-dir), così
l'artefatto riflette esattamente il sorgente committato:
- **Windows** → `dist/Mandelbrot/Mandelbrot.exe` + `_internal/` (CPU sempre;
  GPU CuPy incluso ma **runtime CUDA NON bundled**: la GPU funziona solo se
  l'utente installa driver NVIDIA + runtime CUDA, che CuPy trova via
  cuda-pathfinder da `CUDA_PATH`/PATH/Program Files) + zip
  `dist/Mandelbrot-v<ver>-win64.zip`; **macOS** → `dist/Mandelbrot.app`
  (GPU Metal/pyobjc, firma ad-hoc).
- Ordine: modifica sorgente → **`python build_app.py`** → (verifica) → `git commit`.
  (Su macOS anche `./build_app.sh`, ora redirect a `build_app.py`; su Windows
  `.\build_app.ps1` oppure `python build_app.py`.)
- File: `build_app.py` (unico script, ramificato su `sys.platform`: icona →
  PyInstaller → post → zip su Windows) + `mandelbrot.spec` (ricetta
  PyInstaller multipiattaforma) + `hook_dlldir.py` (runtime hook Windows:
  `os.add_dll_directory` su cartella EXE + `_internal`, difensivo per le DLL
  bundled) + `make_icon.py` (`icon_src.png` + `mandelbrot.ico`).
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
