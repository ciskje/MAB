# Convenzioni del progetto

## Struttura del progetto
- `mandel.py` — il programma (visualizzatore interattivo, Tkinter + GPU
  (CUDA/CuPy o Metal/pyobjc, slot "GPU" generico v5.4.0) + Numba);
  versione e STORICO nel commento iniziale del file (fonte di verità).
- `spec.md` — specifica del programma; DEVE restare in sync col sorgente
  (vedi regola di versionamento sotto).
- `baseline.py` — tool da riga di comando: rigenera i frame di riferimento
  `baseline/*.npy` + misure per stadio (scrive `baseline.txt`).
  Uso: `python baseline.py [path/mandel.py]`.
- `gate.py` — gate di correttezza permanente (criterio in `spec.md`, sezione "Note tecniche").
  Uso: `python gate.py [path/mandel.py]`; exit 0 = PASS.
- `baseline/*.npy` — frame di riferimento (3 zone × GPU f32/f64 + CPU f64
  (`*_cpu.npy`) + CPU f32 (`*_cpu_f32.npy`, v5.3.0) + Metal f32
  (`*_metal_f32.npy`, v5.4.0)); usati dal gate. I `<zona>_gpu_*.npy` (CUDA) restano
  il gold f32/f64 e il righello della varianza f32 per il cross-check Metal.
  (I riferimenti CPU v4.15.1 storici sono stati rimossi in v5.2.1,
  recuperabili dalla storia git.) `baseline.txt` = ultimo report misure.
- `mandelbrot_*.json` — zone salvate dall'app (dato utente, gitignorato).

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
