# Convenzioni del progetto

## Versionamento sorgente (OBBLIGATORIO)
Ogni modifica a `mandel_interattivo.py` (o ad altri sorgenti Python del
progetto) DEVE:
1. Incrementare la versione nel blocco di commento iniziale del sorgente
   (attualmente `VERSIONE: 4.0.0` in `mandel_interattivo.py`).
    - bugfix = patch (x.y.Z), modifica minima (es. un dialog in piu, un testo,
      un opzione) = patch (x.y.Z), modifica funzionale = minor (x.Y.0),
      riscrittura/architettura = major (X.0.0)
2. Aggiungere una voce allo STORICO nello stesso blocco, formato:
   `VERSIONE - AAAA-MM-GG - descrizione delle modifiche`.
Il blocco è intitolato "Insieme di Mandelbrot - visualizzatore interattivo"
e si trova in cima al file.

## Note tecniche (imparate a caro prezzo)
- PowerShell 5.1: NON usare `Start-Process` con `-RedirectStandardOutput`
  verso path UNC (va in timeout). Usare `Start-Process -PassThru`.
- Tkinter: `tk.Checkbutton` (ortografia americana), mai `tk.CheckButton`.
- CuPy: `np.asarray(array_cupy)` NON è permesso (TypeError); usare `.get()`.
- CuPy RawKernel: NON passare la palette LUT come argomento array device
  (l'output del kernel la sovrascriveva a ogni launch). Le palette sono
  incorporate nel kernel come `__constant__` (vedi v4.0.0).
- Il tool `write` NON funziona su path UNC; usare `edit` (che funziona).
- NVRTC: niente flag `-O3`/`--opt-level`; la compilazione è lazy alla
  prima chiamata del kernel.
- Argomenti scalari dei RawKernel: array numpy size-1, non scalari Python.
