# Esploratore interattivo dell'insieme di Mandelbrot

Applicazione grafica in Python per esplorare l'insieme di Mandelbrot con una palette “fuoco” (nero → rosso → arancio → giallo → bianco), supporto GPU CUDA quando disponibile e ottimizzazione per CPU multi-core.

Il programma permette di navigare in modo interattivo sul frattale, effettuare zoom, cambiare precisione di calcolo, regolare il numero di iterazioni e visualizzare informazioni in tempo reale sul punto sotto il cursore e sullo stato della vista.

## Funzionalità principali

- Visualizzazione interattiva dell'insieme di Mandelbrot
- Palette termica a fuoco, con:
  - nero per l'interno dell'insieme
  - sfumature di rosso/arancio/giallo/bianco per le regioni divergenti
- Rendering con colorazione continua “smooth”
- Supporto accelerazione GPU tramite CUDA
- Ottimizzazione automatica per CPU multi-core con PyTorch
- Selezione della precisione del calcolo:
  - single (float32 / complex64)
  - double (float64 / complex128)
- Controlli per impostare il numero di iterazioni
- Modalità di iterazioni automatiche in base allo zoom
- Zoom centrato sul cursore
- Pan (spostamento della vista)
- Selezione area per zoom
- Barra di stato con:
  - coordinate del punto sotto il mouse
  - modulo del numero complesso
  - posizione del pixel
  - centro della vista
  - ampiezza della regione osservata
  - livello di zoom
  - precisione
  - numero di iterazioni
  - tempo di rendering
- Anteprima rapida durante l'interazione e rendering completo dopo un breve ritardo
- Reset della vista con un comando rapido

## Requisiti

Il programma richiede:

- Python 3
- tkinter
- numpy
- torch

Installazione consigliata:

```bash
pip install torch numpy
```

Nota:
- Se la macchina dispone di una GPU NVIDIA con CUDA supportato, PyTorch può usare la GPU automaticamente
- In assenza di CUDA, il programma usa la CPU con calcolo multi-thread

## Avvio

```bash
python test.py
```

Oppure, se il file è eseguibile:

```bash
chmod +x test.py
./test.py
```

## Interfaccia

Nella parte superiore della finestra trovi:

- Spinbox per il numero di iterazioni
- Pulsanti “- 100” e “+ 100” per regolare rapidamente le iterazioni
- Checkbox “Iterazioni automatiche (in base allo zoom)”
- ComboBox per la precisione del calcolo
- Indicatori dello stato del backend:
  - CUDA
  - CPU multi-core
- Pulsante per cambiare backend di calcolo
- Bottone “Reimposta vista (R)”

Sotto la toolbar compare la barra di stato con informazioni sul frattale e sul mouse.

## Come funziona il programma

Il programma calcola l'insieme di Mandelbrot su una griglia di punti complessi. Per ogni pixel:

- si costruisce il numero complesso c
- si applica la trasformazione iterativa:
  z = z² + c
- si verifica se il modulo di z supera 2
- se il punto diverge, viene colorato in base al numero di iterazioni
- se il punto non diverge mai entro il limite impostato, viene considerato parte dell'insieme e dipinto di nero

La colorazione “smooth” usa una scala continua, così i bordi tra i vari settori del frattale risultano più dettagliati e belli.

## Controlli e comandi

### Mouse

- Rotella del mouse:
  - su: zoom in
  - giù: zoom out
  - il centro del zoom è il punto sotto il cursore

- Click sinistro:
  - zoom avanti di 2x sul punto cliccato

- Click destro:
  - zoom indietro di 2x sul punto cliccato

- Trascinamento con click sinistro:
  - spostamento (pan) della vista

- Shift + drag con click sinistro:
  - selezione di un'area da ingrandire

### Tastiera

- `+` o `=`
  - aumenta di 100 le iterazioni

- `-`
  - diminuisce di 100 le iterazioni

- `R` o `r`
  - reimposta la vista iniziale

### Pulsanti e controlli dell'interfaccia

- Spinbox iterazioni:
  - imposta il valore manuale del numero di iterazioni

- `- 100` / `+ 100`:
  - modifica il numero di iterazioni in modo rapido

- “Iterazioni automatiche”:
  - attiva la regolazione automatica del numero di iterazioni in base al livello di zoom

- Precisione:
  - single: calcolo veloce
  - double: calcolo più preciso, ma più lento

- Passa a CPU / Passa a CUDA:
  - alterna il backend di calcolo, se la GPU è disponibile

## Legenda dei comandi

| Comando | Azione |
|---|---|
| Rotella su | Zoom in centrato sul cursore |
| Rotella giù | Zoom out centrato sul cursore |
| Click sinistro | Zoom x2 sul punto cliccato |
| Click destro | Dezoom x2 sul punto cliccato |
| Trascina con sinistro | Pan / spostamento della vista |
| Shift + drag con sinistro | Seleziona area e zoom su di essa |
| `+` / `=` | Aumenta iterazioni |
| `-` | Diminuisce iterazioni |
| `R` | Reset vista |
| Spinbox iterazioni | Imposta valore manuale |
| Pulsanti `+ 100` / `- 100` | Modifica iterazioni |
| Checkbox automatiche | Regola iterazioni in base allo zoom |

## Stato della vista e informazioni

La barra di stato mostra informazioni utili:
- centro della vista in coordinate complesse
- ampiezza della finestra di osservazione
- fattore di zoom
- precisione
- iterazioni attive
- tempo di rendering

Quando il mouse si muove sul frattale, viene mostrato anche:
- il numero complesso c associato al pixel
- il modulo |c|
- le coordinate del pixel

## Performance

Il programma è progettato per sfruttare entrambe le piattaforme:

- GPU CUDA:
  - molto più veloce su calcoli grandi e ad alta risoluzione
  - ideale per render delle immagini con risoluzioni elevate

- CPU multi-core:
  - PyTorch esegue operazioni vettoriali in parallelo
  - ottimizza il carico sui core disponibili

Inoltre:
- viene eseguito un rendering di anteprima a risoluzione ridotta durante l'interazione
- il rendering completo avviene dopo un breve ritardo per mantenere la UI reattiva

## Note tecniche

Il programma usa:
- tkinter per l'interfaccia
- numpy per la gestione dei colori e la conversione in immagini
- torch per il calcolo vettoriale del frattale
- PhotoImage di Tkinter per mostrare il frattale nel canvas

## Considerazioni

Questo progetto è un ottimo esempio di:
- frattali generati in tempo reale
- utilizzo di GPU con PyTorch
- interfaccia grafica desktop in Python
- ottimizzazione di rendering e interazione utente

## Licenza

Nessuna licenza è dichiarata nel codice sorgente; per questo motivo il programma è da considerarsi come codice sorgente libero da riutilizzare e modificare, salvo eventuali restrizioni esterne non specificate.
