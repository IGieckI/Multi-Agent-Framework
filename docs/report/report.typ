// Rapporto sugli esperimenti. Compilare con:  python docs/report/build.py   (oppure: typst compile docs/report/report.typ)
// Le tabelle leggono data/data.json (make_data.py); i grafici sono in figures/ (make_figures.py).
// Il testo e i numeri citati nel testo si modificano qui a mano.

#let d = json("data/data.json")

// ---------- stile
#set document(title: "Comunicazione tra agenti LLM in WordCraft", author: "Giacomo Antonelli")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.3cm), numbering: "1")
#set text(font: ("Libertinus Serif", "Noto Serif"), size: 10.5pt, lang: "it")
#set par(justify: true, leading: 0.62em, spacing: 0.95em)
#show heading: set text(font: ("Noto Sans", "DejaVu Sans"), weight: "bold", fill: rgb("#0b0b0b"))
#show heading.where(level: 1): it => { v(0.9em); text(14pt, it); v(0.3em) }
#show heading.where(level: 2): it => { v(0.5em); text(11pt, it); v(0.15em) }
#set heading(numbering: "1.1")
#show figure.caption: set text(9pt)
#set figure(gap: 0.8em)
#show figure: set block(breakable: false)
#set table(stroke: none, inset: (x: 5pt, y: 3.2pt))
#show table: set text(9pt, font: ("Noto Sans", "DejaVu Sans"), hyphenate: false)
#show table: set par(justify: false)
#show table.cell.where(y: 0): set text(weight: "bold")

// ---------- numeri in formato italiano
#let f(x) = if x == none { "–" } else { str(calc.round(x, digits: 1)).replace(".", ",") }
#let sg(x) = if x == none { "–" } else if x > 0 { "+" + f(x) } else if x < 0 { "−" + f(-x) } else { "0" }
#let ci(c) = if c == none { "–" } else [#sg(c.d) #text(fill: rgb("#52514e"), size: 0.9em)[[#sg(c.lo); #sg(c.hi)]]]
#let p(r) = if r == none { "–" } else { f(r.p) }
#let NAME = ("qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "gemma3-4b": "Gemma 3-4B", "llama3.2-3b": "Llama 3.2-3B",
  "qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "gemma3-1b": "Gemma 3-1B", "llama3.2-1b": "Llama 3.2-1B",
  "smollm2-360m": "SmolLM2-360M", "smollm2-1.7b": "SmolLM2-1.7B", "mock_norepeat": "Casuali", "all": "Tutti")
#let M4 = ("qwen3-4b", "qwen3-8b", "gemma3-4b", "llama3.2-3b")
#let M3 = ("qwen3-4b", "gemma3-4b", "llama3.2-3b")
#let rule = table.hline(stroke: 0.6pt + rgb("#52514e"))
#let thin = table.hline(stroke: 0.4pt + rgb("#c3c2b7"))

// ---------- titolo
#align(left)[
  #text(font: ("Noto Sans", "DejaVu Sans"), 20pt, weight: "bold")[Report Esperimenti Early Novembre]
  #v(-0.4em)
]
#v(0.6em)

= Recap

Il task studia il comportamento di squadre di piccoli modelli open che giocano a WordCraft. A ogni turno un agente combina due oggetti del proprio inventario per ottenerne uno nuovo e ha un numero limitato di turni (per ora 4) per creare un oggetto obiettivo. La difficoltà si regola con la profondità (quanti passaggi servono servono per raggiungere l'obiettivo) e con i distrattori (oggetti inutili nell'inventario).

I protocolli/topologie che ho utilizzato in questi esperimenti sono stati: Completamente connesso (tutti leggono i messaggi di tutti), Stella (c'è un nodo centrale che fa da intermediario tra i vari agenti), ed alcune varianti del completamente connesso dove condividono solo i risultati delle combinazioni, solo i successi o solo i fallimenti.

Come 'baseline' di confronto ho usato queste, dove la configurazione topologica/protocollo resta uguale: gli stessi agenti senza collegamenti (team che non comunica), un agente singolo con lo stesso numero di turni, un agente singolo con il budget di token dell'intera squadra, una squadra muta con temperature diverse per agente. 

Il successo di una squadra significa che almeno un agente ha creato l'obiettivo.

Per le run ho utilizzato sempre il cluster giano, in particolare ho cercato di runnare il più possibile sulle rtx 2080 (meno occupate in generale), ma per molte cose ho dovuto usare le L40, tipo per esperimenti multi-modello, o Gemma che ha bisogno dei bfloat, che altrimenti va in overflow. In ogni caso, ogni run è riproducibile dai config.

= Primo test

*Obiettivo:* capire quali modelli e quali impostazioni del compito fossero utilizzabili (gli llm riescono a capire la task e rispondere correttamente?). 

*Configurazione:* 10 modelli × 3 protocolli (6 agenti isolati, 6 completamente connessi, stella con 3) × 4 impostazioni (profondità 1 o 2 × 0 o 3 distrattori) × 6 repliche × 3 versioni, temperatura 1.0

#figure(
  table(columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, right, right, right),
    rule,
    [Modello], [Agente singolo], [C. conn. muti], [C. conn. comun.], [Stella muti], [Stella comun.], [Risposte rifiutate],
    thin,
    ..for (m, v) in d.pilot { (NAME.at(m), p(v.one), p(v.fc_mute), p(v.fc_talk), p(v.st_mute), p(v.st_talk), f(v.refused)) },
    rule),
  caption: [Percentuale di successo per modello e percentuale di risposte rifiutate dal parser.],
) <tab-pilot>

*Analisi dei risultati*
- La comunicazione non ha aiutato in modo significativo nessun modello, e i casi più netti vanno nella direzione opposta: nel completamente connesso Gemma 3-4B scende dal 71% al 46% e SmolLM2-360M dal 46% al 29%. Solo SmolLM2-1.7B sale (dal 54% al 62%)
- Il guadagno di una squadra segue la diversità dei suoi agenti: gli agenti Qwen3 si comportano quasi come copie (ridondanza 0,76–0,95) e in squadra guadagnano poco (Qwen3-8B: 54% da solo, 58% in squadra). Llama 3.2 e SmolLM2 sono i più vari (ridondanza 0,50–0,66) e guadagnano di più: Llama 3.2-3B passa dal 17% al 46–62%
- Llama 3.2 e SmolLM2 si vedevano rifiutare il 59–76% delle risposte: copiavano l'esempio svolto nel prompt (centinaia di risposte parlano di un 'cyborg', che era nel prompt di esempio ma non nel task) e giocavano al Little Alchemy originale ('stone', 'water') invece che con il proprio inventario
- Due delle quattro impostazioni erano o vicino al 10% o al 90% (profondità 1 senza distrattori 92–100%, profondità 2 con 3 distrattori 8–13%), da allora uso solo le due intermedie
- Il 44% delle run era un duplicato esatto di altre run: il framework ora esegue una sola volta ogni configurazione distinta, ho anche corretto una metrica di fine run che premiava le squadre piccole

= Prompt v2

*Obiettivo:* eliminare problemi con la copia dell'prompt di esempio. Il prompt v2 presenta l'esempio come un gioco diverso e chiede di usare solo gli oggetti del proprio inventario, nient'altro cambia. 6 modelli sugli stessi obiettivi del primo test, confrontati run per run.

#figure(
  table(columns: (auto, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, right),
    rule,
    [Modello], [Rifiutate v1 (%)], [Rifiutate v2 (%)], [Successo v1 (%)], [Successo v2 (%)],
    thin,
    ..for (m, v) in d.v2 { (NAME.at(m), f(v.refused_v1), f(v.refused_v2), f(v.solved_v1), f(v.solved_v2)) },
    rule),
  caption: [Prompt v1 vs Prompt v2],
) <tab-v2>

*Analisi dei risultati*
Le risposte rifiutate calano molto per Llama 3.2-3B (dal 59% al 22%) e SmolLM2-1.7B, poco per SmolLM2-360M. Llama 3.2-3B sale dal 38% al 54% di successo, vicino a Qwen3-4B (62%, invariato). Da qui in poi uso sempre il v2 e SmolLM2 resta inaffidabile quindi viene escluso dagli esperimenti successivi.

= Esperimento principale

*Goal:* misurare con dati sufficienti l'effetto della comunicazione. 

*Configurazione:* Qwen3-4B, Qwen3-8B, Gemma 3-4B, Llama 3.2-3B × completamente connesso (6 agenti) e stella (3) × 2 impostazioni (profondità 1 con 3 distrattori, profondità 2 senza) × 20 obiettivi × 5 versioni.

#figure(image("figures/main.svg", width: 100%),
  caption: [Percentuale di successo della squadra muta vs con comunicazione per ogni modello]) <fig-main>

#figure(
  table(columns: (auto, 1fr, 1.6fr, 1fr, 1.6fr, 0.9fr, 0.9fr), align: (left, right, right, right, right, right, right),
    rule,
    [Modello], [C. conn. muti], [Comunicazione − muti], [Stella muti], [Comunicazione − muti], [Agente singolo], [Seguiti alla cieca],
    thin,
    ..for m in (..M4, "all") { let e = d.main.at(m); (NAME.at(m), p(e.fully_connected.all_independent), ci(e.fully_connected.talk_minus_mute),
      p(e.star.all_independent), ci(e.star.talk_minus_mute), p(e.one), f(100 * e.herding.base.blind)) },
    rule),
  caption: [Esperimento principale: successo (%) e differenze appaiate nei vari tipi + 'Seguiti alla cieca': percentuale di volte che un agente ha ripetuto un azione che un compagno gli aveva indicato come fallimento],
) <tab-main>

*Analisi dei risultati*
- La comunicazione con 'completamente connesso' peggiora la squadra per ogni modello, da −5 (Llama) a −12,5 punti (Qwen3-4B e Gemma), −9,4 in media. Anche i livelli in modalità singola sono molto diversi: Llama è il migliore in squadra (75% da muti) pur essendo il peggiore da solo (38%).
- Qwen3-4B e Qwen3-8B seguono alla cieca il 42–45% dei suggerimenti ricevuti, Gemma il 15%, Llama il 12%. I percorsi distinti esplorati calano per Gemma (da 0,72 a 0,55) e i Qwen (da 0,53 a 0,40–0,45), quasi per niente per Llama (da 0,80 a 0,78).
- Comunicazione con 'Stella' è neutra in media (+1,2), ma con effetti opposti: Qwen3-4B +10 [+2,5 - +20], Gemma +5, Qwen3-8B 0, Llama −10 [−22,5 - 0]. 
Nota: L'hub scrive pochi messaggi (2,5 per run).
- Il single agent con il token budget della squadra batte i sei agenti che comunicano: per Gemma (+12,5), Qwen3-4B e Qwen3-8B (+7,5), non per Llama (0), e rispetto ai sei agenti muti è pari o poco sotto [-5 - 0].
Nota: il fatto che gioca più a lungo può aver influenzato (17 turni in media).
- Il costo nel completamente connesso costa dal 41% (Llama) al 65% (Qwen3-4B) di token in più per run, e quasi il doppio per obiettivo risolto.
- Sperimentando con temperature diverse (0.4–1.3): danneggiano Llama (−12,5 [−22,5 - −2,5]), per gli altri da −2,5 a +2,5.

= Follow-up: struttura, dimensione e canali

*Obiettivo:* dall'esperimento precedente, non era chiaro se i risultati fossero più dovuti alla struttura/topologia (fully-connected/stella) o la dimensione (3/6 agenti) e non diceva da quale canale venisse l'effetto gregge (tipico di Qwen).

*Configurazione:* Qwen3-4B, Gemma 3-4B, Llama 3.2-3B × 6 protocolli × le stesse impostazioni, obiettivi e versioni; temperature centrate su 1,0 (0,7, 1,0, 1,3).

#figure(image("figures/followup.svg", width: 82%),
  caption: [Comunicazione squadra muta per protocollo e modello]) <fig-followup>

- Il completamente connesso a 6 peggiora per tutti (da −10 a −12,5), a 3 agenti dipende dal modello (Llama −12,5, Gemma −7,5, Qwen3-4B −2,5). La stella è neutra per tutti, a 3 e a 6 agenti. La causa è quindi più dovuta alla struttura, non la dimensione.
- I due canali danneggiano modelli diversi: condividere solo i risultati danneggia solo Llama (−12,5 [−22,5 - −2,5]), che li copia, condividere solo i messaggi danneggia di più Gemma (−10) che Qwen e Llama (−5).
- Lo sweep di temperature centrate su 1.0 (invece che 0.8 come prima) non cambia nulla (da −5 a +2,5), e mostrare il budget all'agente singolo come aggiunta al prompt nemmeno.

#figure(image("figures/team_size.svg", width: 82%),
  caption: [Vantaggio di 6 agenti su 3, per modello]) <fig-size>

- Il vantaggio di una squadra più grande è quasi tutto di Llama, i cui agenti sono i più diversi: da muti 6 agenti battono 3 di 15 punti [+5 - +27,5] nel completamente connesso e di 10 nella stella. Per Qwen3-4B 6 e 3 agenti muti hanno lo stesso esito su ogni obiettivo, per Gemma +2,5. Con la comunicazione il vantaggio sparisce per Qwen e Gemma, si riduce per Llama nella stella (+5) e resta nel completamente connesso (+17,5).

= Confronto con agenti casuali

*Obiettivo:* tarare i risultati dei modelli rispetto a un puro random agent. 

*Configurazione:* agenti che scelgono una coppia del proprio inventario a caso (e una variante che non ripete coppie già provate né segnalate come fallite dai compagni) × 5 protocolli × le stesse impostazioni × 200 obiettivi.

#figure(image("figures/random.svg", width: 100%),
  caption: [Squadre LLM di 6 agenti (mute e con comunicazione) e agente singolo, per modello, contro 6 agenti casuali e un agente casuale]) <fig-random>

- A profondità 1, 6 agenti casuali risolvono l'85%: come le squadre LLM mute (75–100% secondo il modello). La conoscenza dei modelli paga per un agente singolo: 60–70% contro il 25% di un agente casuale.
- A profondità 2 i casuali risolvono il 67%. Le squadre LLM mute vanno dal 60% (Gemma) al 50% (Llama) fino al 30% e 25% (Qwen3-8B e Qwen3-4B), e con la comunicazione scendono ancora (dal 15% al 50%).
- I modelli esplorano male: a profondità 2 un agente singolo ripete una coppia già provata nel 46–49% dei turni con Qwen e Llama, nel 21% con Gemma, contro le istruzioni.
- Per gli agenti casuali (che ignorano ripetizioni e fallimenti degli altri) condividere i fallimenti aiuta molto (80,5% contro 64% a profondità 2), per gli LLM no.

*NOTA:* tutti gli agenti vedono lo stesso inventario e lo stesso obiettivo, quindi la comunicazione non ha nulla da trasmettere oltre a chi prova cosa. Ha quindi senso provare a far comunicare degli indizi 'pre-made' che diano sempre un informazione utile aggiunta, per vedere se gli altri agenti riescono a trarne beneficio.

= Scambio di Note Private Custom

*Obiettivo:* dare alla comunicazione qualcosa da trasmettere di utile e nuovo. 

*Configurazione:* ogni agente riceve 2 note private, cioè ricette vere visibili solo a lui. Con le note divise le due ricette del percorso verso l'obiettivo vanno a due agenti diversi e le altre note sono ricette fuori percorso (esche), nelle altre condizioni tutti hanno entrambe le ricette, oppure nessuno ha note. Un agente singolo riceve sempre entrambe. Siccome la task usata fino ad ora sarebbe stata troppo semplice con questo setting, viene resa più difficile impostando depth=2 e distractors=6, e mi assicuro non ci siano "shortcut" ovvero oggetti di stage 2 ottenibili con solo oggetti di stage 0.
Qwen3-4B, Gemma 3-4B, Llama 3.2-3B e agenti casuali × 5 protocolli × profondità 2 con 6 distrattori (9 oggetti, 45 coppie) × 20 obiettivi.

#figure(
  table(columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, left, right, right, right, right, right),
    rule,
    [Modello], [Note], [Agente singolo], [C. conn. muti], [C. conn. comun.], [Stella comun.], [Solo fallimenti],
    thin,
    ..for m in (..M3, "mock_norepeat") { for h in ("none", "split", "all") { let v = d.hints.at(m).at(h)
      (NAME.at(m), ("none": "nessuna", "split": "divise", "all": "tutti").at(h), p(v.one), p(v.fc_mute), p(v.fc_talk), p(v.st_talk), p(v.fail_talk)) } },
    rule),
  caption: [Note private: successo (%) per modello],
) <tab-hints>

*Analisi dei risultati*
- Senza note quasi nessuno risolve (0–19%). Con le note divise le squadre risolvono circa un quarto degli obiettivi con o senza comunicazione, per ogni modello (Qwen3-4B dal 19% al 25%, Gemma dal 25% al 31%, Llama 25% e 25%).
- Un agente singolo con entrambe le ricette risolve il 50% (Qwen3-4B, Llama) o l'81% (Gemma), una squadra muta in cui tutti le hanno l'81–94%. L'informazione è nella squadra, ma non viene messa insieme.
- Dai log: chi ha la nota la menziona di rado (la ricetta dell'intermedio nel 6–11% dei suoi messaggi) e spesso non la usa nemmeno lui. Gli altri la provano raramente, e le esche assorbono il 27–40% dei turni.
- Condividere solo i fallimenti non aiuta gli LLM: il 20% delle loro azioni ripete una coppia che un compagno ha appena segnalato come fallita.

= Comunicare e usare le note

*Obiettivo:* confermare il risultato su 40 nuovi obiettivi senza scorciatoie e distinguere due fallimenti: chi ha la nota non pensa a comunicarla, oppure chi la riceve non sa usarla?

*Configurazione:* note divise con esche, divise senza esche, tutti con entrambe, completamente connesso e stella. Poi due istruzioni aggiunte al prompt per un maggior sfruttamento delle ricette:
- per i detentori: "The other players do not know your notes: use your message to tell them what your notes say.";
- per tutti: "Recipes the other players tell you from their notes are true: when one helps make the target, use it."
Infine le stesse celle con Qwen3-8B, per vedere se un modello più capace riesce dove i 4B falliscono.

#figure(image("figures/hints2.svg", width: 100%),
  caption: [Note divise, completamente connesso: successo (%) per condizione e modello. A destra, in grigio, le baselines con tutta l'informazione in una testa o in tutte.]) <fig-hints2>

#figure(
  table(columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, left, right, right, right, right, right, right),
    rule,
    [Modello], [Note], [Muti], [Comun.], [\+ condividi], [\+ usa], [1 agente, entrambe], [Stella comun.],
    thin,
    ..for m in M4 { for (k, lab) in (("True", "divise + esche"), ("False", "divise")) { let v = d.hints2.at(m).at(k)
      (NAME.at(m), lab, p(v.fc_mute), p(v.fc_talk), p(v.share), p(v.use), p(v.one), p(v.st_talk)) } },
    thin,
    ..for m in M4 { let v = d.hints2.at(m).all
      (NAME.at(m), "tutti", p(v.fc_mute), p(v.fc_talk), "–", "–", p(v.one), p(v.st_talk)) },
    rule),
  caption: [Successo (%) con note private per modello e condizione],
) <tab-hints2>

*Analisi Risultati*

- Il risultato si conferma per i 4B: Con le note divise e le esche, la comunicazione senza istruzioni cambia poco e in direzioni diverse. Un agente con entrambe le ricette risolve il 75%, 78% e 57%, tutti con entrambe il 90–92%.
- L'istruzione a condividere funziona dal lato di chi parla, non di chi ascolta: Il successo sale per tutti (Qwen3-4B +22, Gemma +12, Llama +10 punti), ma la squadra resta sotto quella in cui tutti hanno le note.

#figure(image("figures/tell_receive.svg", width: 92%),
  caption: [Chi comunica e chi usa, quota dei detentori che nominano la ricetta dell'intermedio nel primo messaggio (asse x) e quota dei riceventi che la giocano all'azione successiva (asse y), senza istruzione (vuoto) e con l'istruzione a condividere (pieno).]) <fig-tell>

- Con l'istruzione i detentori nominano la ricetta quasi sempre con Qwen (97%), spesso con Gemma (62%, e 100% senza esche), meno di una volta su due con Llama (30–47%).
- Chi la riceve la gioca all'azione successiva poco e in modo diverso per modello: Qwen3-4B 12–21%, Llama 8–27%, Gemma 4%. Gemma comunica le note ma quasi non usa quelle ricevute.
- L'istruzione a usare non migliora nessun modello (con le esche −5 per tutti, senza: Qwen +8, Gemma −12, Llama −8, nessuno significativo), e le quote di uso restano le stesse. Il problema non è di fiducia.
- Le esche costano a Gemma e Llama (da +10 a +20 punti senza esche), quasi nulla a Qwen. Con l'istruzione i detentori comunicano anche le esche (57% dei messaggi).
- Quando tutti hanno entrambe le ricette, la comunicazione peggiora la squadra per ogni modello (da −5 a −10), più agenti muti invece aiutano, soprattutto chi da solo segue peggio le note (+10 Qwen, +20 Gemma, +35 Llama sull'agente singolo).

Qwen3-8B mette insieme le note comunicando: È il primo risultato positivo della comunicazione. Con le note divise +27,5 punti [+12,5 - +45] nel completamente connesso (dal 20% al 47,5%) e +25 nella stella. Con l'istruzione a condividere risolve il 72,5%, quanto un agente con entrambe le ricette (67,5%). Già senza istruzioni i suoi detentori nominano la ricetta finale nel primo messaggio nel 57–60% dei casi e i suoi riceventi giocano la ricetta dell'intermedio nel 40–51% dei casi. Le esche non lo danneggiano. Da solo o in una squadra muta non è migliore di Qwen3-4B (−7,5 e −2,5 punti): tutto il suo vantaggio viene dalla comunicazione. È però un solo modello su 40 obiettivi.

= Analisi di un Problema Ricorrente con la Reiterazione di un Oggetto Intermedio

*Obiettivo:* capire perché quando ogni agente ha già entrambe le ricette e non c'è nulla da mettere insieme, la comunicazione nel completamente connesso dovrebbe essere innocua, ed invece abbassa il successo per tutti i modelli: Qwen3-8B −25 punti (dall'87,5% al 62,5%), Gemma −10, Qwen3-4B e Llama −7,5.

*Configurazione:* lettura dei log delle run perse da Qwen3-8B, il caso più marcato, poi, per ogni modello, la quota delle azioni che rifanno l'oggetto intermedio quando l'agente lo ha già creato, nel completamente connesso e nella stella, da muti e con comunicazione.

#figure(image("figures/remake.svg", width: 100%),
  caption: [Tutti con entrambe le ricette: quota delle azioni che rifanno l'oggetto intermedio dopo averlo già creato, squadra muta (vuoto) e con comunicazione (pieno)]) <fig-remake>

- Nel caso di Qwen3-8B: Negli 11 obiettivi persi tutte le 264 azioni della squadra (6 agenti × 4 turni) sono la ricetta dell'intermedio: ogni agente lo crea al primo turno e lo rifà nei tre successivi (198 ripetizioni), anche se ce l'ha già nell'inventario. Il secondo passo non arriva mai, mentre i messaggi ripetono "Let's make explosion first!".
- Succede a tutti i modelli, già da muti: Le azioni che rifanno l'intermedio sono il 60% per Qwen3-8B, il 51% per Llama, il 29% per Qwen3-4B e il 17% per Gemma.
- Nel completamente connesso la comunicazione peggiora il problema: Le ripetizioni salgono per Gemma (dal 17% al 48%), Qwen3-8B (dal 60% all'80%) e Qwen3-4B (dal 29% al 43%), per Llama restano al 52%.
- Nella stella dipende dal modello: Per Qwen3-8B l'hub fa avanzare gli spoke: le ripetizioni scendono dal 64% al 34% e il successo sale al 95%, la cella migliore dello studio. Per gli altri no: Gemma sale dal 3% al 29%, Qwen3-4B resta al 23%, Llama scende appena (dal 51% al 44%).

Risultato: rifare l'oggetto intermedio è un modo di fallire comune a tutti i modelli testati: gli agenti perdono il filo dei propri progressi e rieseguono il primo passo del piano scritto nelle note. Nel completamente connesso i messaggi che ripetono il piano ("prima fai X") rafforzano l'errore, e questo spiega in buona parte perché la comunicazione abbassi il successo quando tutti hanno le note. Potrebbe spiegare anche perché un agente singolo con entrambe le ricette fallisca ancora dal 20% (Qwen3-4B) al 45% (Llama) dei casi, ma sull'agente singolo non è stato misurato. Il grid in corso segna nel prompt gli oggetti già creati, per vedere se l'errore sparisce.

= Conclusioni Importanti

+ *Con modelli da 3–4B la squadra vale per i tentativi, non per la comunicazione:* Il guadagno di una squadra dipende da quanto i suoi agenti sono diversi tra loro (Llama molto, Qwen quasi nulla), e agenti casuali forniscono tentativi più vari di qualsiasi squadra LLM.
+ *Il completamente connesso è dannoso per ogni modello testato*, per un effetto gregge che passa sia dai risultati condivisi sia dai messaggi, con pesi diversi per modello. La stella è la struttura più sicura ed economica, ma il suo effetto dipende dal modello.
+ *La comunicazione ha un valore solo se c'è informazione privata da trasmettere, e solo se il modello sa usarla:* I modelli da 4B possono essere spinti a comunicare, ma non a usare quello che ricevono. Qwen3-8B sì: è la prima evidenza che la capacità di ricezione, non la volontà di comunicare, è il collo di bottiglia.
+ Quasi ogni risultato aggregato ha almeno un modello che va nella direzione opposta o non si muove: le conclusioni vanno lette per modello.

= In corso

In coda sul cluster:
- Gemma 3-12B sulle celle con note private, istruzioni comprese: il risultato di Qwen3-8B dipende dalla dimensione o da Qwen?
- Qwen3-8B su altri 40 obiettivi (80 in totale): aumento la popolazione di esperimenti per vedere se i risultati sono reali o frutto di un caso fortunato.
- Oggetti già creati nel prompt (una riga che li elenca): elimina il ciclo dell'intermedio?
- Squadre miste (Qwen3-4B, Gemma 3-4B e Llama 3.2-3B a turno): la diversità tra famiglie di modelli aiuta?

#v(1em)
