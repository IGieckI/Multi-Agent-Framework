// Rapporto sugli esperimenti, v2. Compilare con:  python docs/report/build.py --v2   (oppure: typst compile docs/report/report_v2.typ)
// Le sezioni della v1 leggono data/data.json (make_data.py), le nuove data/data_v2.json (make_data_v2.py);
// i grafici sono in figures/ (make_figures.py, make_figures_v2.py: v2_*.svg).
// Il testo e i numeri citati nel testo si modificano qui a mano.

#let d = json("data/data.json")
#let d2 = json("data/data_v2.json")

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
#let NAME2 = NAME + ("gemma3-12b": "Gemma 3-12B")
#let f2(x) = if x == none { "–" } else { str(calc.round(x, digits: 2)).replace(".", ",") }
// differenza appaiata con intervallo e, in grigio, obiettivi vinti/persi (test del segno esatto)
#let cs(c) = if c == none { "–" } else [#ci(c) #text(fill: rgb("#52514e"), size: 0.85em)[#c.w/#c.l]]
#let small(body) = { show table: set text(8pt); body }   // tabelle larghe
#let VER = ("all_independent": "muti", "base": "comun.", "one_agent": "1 agente")

// ---------- titolo
#align(left)[
  #text(font: ("Noto Sans", "DejaVu Sans"), 20pt, weight: "bold")[Report Esperimenti Novembre (v2)]
  #v(-0.4em)
]
#v(0.6em)

*Novità della v2:* dalla @sec-big in poi riportano gli esperimenti lanciati dopo la prima versione (7–12 novembre): modelli più grandi, il ciclo dell'intermedio, squadre miste, una revisione critica dei risultati e gli esperimenti che ne sono seguiti (canali, nomi inventati), la confidenza dei modelli e i turni in sequenza. Le ultime due sezioni prima delle conclusioni correggono le misure di adozione e di copia (@sec-baseline) e confrontano i risultati con la letteratura recente (@sec-lett). Le sezioni precedenti sono quelle della v1, con rimandi dove i nuovi risultati le correggono.

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
  table(columns: (auto, 0.9fr, 1.6fr, 0.9fr, 1.6fr, 0.9fr, 1.3fr), align: (left, right, right, right, right, right, right),
    rule,
    [Modello], [C. conn. muti], [Comunicazione − muti], [Stella muti], [Comunicazione − muti], [Agente singolo], [Ripete un fallimento (muti → com.)],
    thin,
    ..for m in (..M4, "all") { let e = d.main.at(m); (NAME.at(m), p(e.fully_connected.all_independent), ci(e.fully_connected.talk_minus_mute),
      p(e.star.all_independent), ci(e.star.talk_minus_mute), p(e.one), {let b = d2.adopt.main.fully_connected.at(if m == "all" { "pooled" } else { m }).blind_adoption; f(calc.round(b.sil)) + " → " + f(calc.round(b.talk))}) },
    rule),
  caption: [Esperimento principale: successo (%) e differenze appaiate nei vari tipi + 'Ripete un fallimento' (completamente connesso): percentuale di azioni in cui un agente gioca una coppia che un compagno ha appena provato senza successo; per la squadra muta è contata su ciò che avrebbe ricevuto (le azioni dei compagni nel turno prima, @sec-baseline). Nella v1 la colonna riportava solo il valore con la comunicazione ("Seguiti alla cieca")],
) <tab-main>

*Analisi dei risultati*
- La comunicazione con 'completamente connesso' peggiora la squadra per ogni modello, da −5 (Llama) a −12,5 punti (Qwen3-4B e Gemma), −9,4 in media. Anche i livelli in modalità singola sono molto diversi: Llama è il migliore in squadra (75% da muti) pur essendo il peggiore da solo (38%).
- Nella v1 scrivevo che Qwen3-4B e Qwen3-8B "seguono alla cieca" il 42–45% dei suggerimenti ricevuti (Gemma 15%, Llama 12%). Il confronto con la squadra muta lo ridimensiona (@sec-baseline): gli agenti Qwen muti ripetono il fallimento di un compagno già nel 31–43% delle azioni, perché giocano comunque le stesse coppie; la comunicazione aggiunge +11 punti per Qwen3-4B e +4 per Qwen3-8B, con intervalli che includono lo 0. Resta il calo dei percorsi distinti esplorati, misurato in entrambe le condizioni: Gemma da 0,72 a 0,55, i Qwen da 0,53 a 0,40–0,45, quasi niente per Llama (da 0,80 a 0,78).
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
- I due canali danneggiano modelli diversi: condividere solo i risultati danneggia solo Llama (−12,5 [−22,5 - −2,5]), che rifà di più quello che hanno fatto i compagni (ma anche Gemma lo fa, senza perdere: @sec-baseline), condividere solo i messaggi danneggia di più Gemma (−10) che Qwen e Llama (−5).
- Lo sweep di temperature centrate su 1.0 (invece che 0.8 come prima) non cambia nulla (da −5 a +2,5), e mostrare il budget all'agente singolo come aggiunta al prompt nemmeno.

#figure(image("figures/team_size.svg", width: 82%),
  caption: [Vantaggio di 6 agenti su 3, per modello]) <fig-size>

- Il vantaggio di una squadra più grande è quasi tutto di Llama, i cui agenti sono i più diversi (seguito con 12 agenti e la confidenza dei modelli nella @sec-conf): da muti 6 agenti battono 3 di 15 punti [+5 - +27,5] nel completamente connesso e di 10 nella stella. Per Qwen3-4B 6 e 3 agenti muti hanno lo stesso esito su ogni obiettivo, per Gemma +2,5. Con la comunicazione il vantaggio sparisce per Qwen e Gemma, si riduce per Llama nella stella (+5) e resta nel completamente connesso (+17,5).

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

Qwen3-8B mette insieme le note comunicando: È il primo risultato positivo della comunicazione. Con le note divise +27,5 punti [+12,5 - +45] nel completamente connesso (dal 20% al 47,5%) e +25 nella stella. Con l'istruzione a condividere risolve il 72,5%, quanto un agente con entrambe le ricette (67,5%). Già senza istruzioni i suoi detentori nominano la ricetta finale nel primo messaggio nel 57–60% dei casi e i suoi riceventi giocano la ricetta dell'intermedio nel 40–51% dei casi. Le esche non lo danneggiano. Da solo o in una squadra muta non è migliore di Qwen3-4B (−7,5 e −2,5 punti): tutto il suo vantaggio viene dalla comunicazione. È però un solo modello su 40 obiettivi. *Aggiornamento (v2):* su 80 obiettivi il risultato regge (@sec-big), ma senza istruzione il guadagno passa dai risultati condivisi e non dai messaggi (@sec-canali).

= Analisi di un Problema Ricorrente con la Reiterazione di un Oggetto Intermedio

*Obiettivo:* capire perché quando ogni agente ha già entrambe le ricette e non c'è nulla da mettere insieme, la comunicazione nel completamente connesso dovrebbe essere innocua, ed invece abbassa il successo per tutti i modelli: Qwen3-8B −25 punti (dall'87,5% al 62,5%), Gemma −10, Qwen3-4B e Llama −7,5.

*Configurazione:* lettura dei log delle run perse da Qwen3-8B, il caso più marcato, poi, per ogni modello, la quota delle azioni che rifanno l'oggetto intermedio quando l'agente lo ha già creato, nel completamente connesso e nella stella, da muti e con comunicazione.

#figure(image("figures/remake.svg", width: 100%),
  caption: [Tutti con entrambe le ricette: quota delle azioni che rifanno l'oggetto intermedio dopo averlo già creato, squadra muta (vuoto) e con comunicazione (pieno)]) <fig-remake>

- Nel caso di Qwen3-8B: Negli 11 obiettivi persi tutte le 264 azioni della squadra (6 agenti × 4 turni) sono la ricetta dell'intermedio: ogni agente lo crea al primo turno e lo rifà nei tre successivi (198 ripetizioni), anche se ce l'ha già nell'inventario. Il secondo passo non arriva mai, mentre i messaggi ripetono "Let's make explosion first!".
- Succede a tutti i modelli, già da muti: Le azioni che rifanno l'intermedio sono il 60% per Qwen3-8B, il 51% per Llama, il 29% per Qwen3-4B e il 17% per Gemma.
- Nel completamente connesso la comunicazione peggiora il problema: Le ripetizioni salgono per Gemma (dal 17% al 48%), Qwen3-8B (dal 60% all'80%) e Qwen3-4B (dal 29% al 43%), per Llama restano al 52%.
- Nella stella dipende dal modello: Per Qwen3-8B l'hub fa avanzare gli spoke: le ripetizioni scendono dal 64% al 34% e il successo sale al 95%, la cella migliore dello studio. Per gli altri no: Gemma sale dal 3% al 29%, Qwen3-4B resta al 23%, Llama scende appena (dal 51% al 44%).

Risultato: rifare l'oggetto intermedio è un modo di fallire comune a tutti i modelli testati: gli agenti perdono il filo dei propri progressi e rieseguono il primo passo del piano scritto nelle note. Nel completamente connesso i messaggi che ripetono il piano ("prima fai X") rafforzano l'errore, e questo spiega in buona parte perché la comunicazione abbassi il successo quando tutti hanno le note. Potrebbe spiegare anche perché un agente singolo con entrambe le ricette fallisca ancora dal 20% (Qwen3-4B) al 45% (Llama) dei casi, ma sull'agente singolo non è stato misurato. Il seguito è nella @sec-loop: segnare nel prompt gli oggetti creati rompe il ciclo per Qwen, cambiare l'esempio no.

= Modelli più grandi: Gemma 3-12B e Qwen3-8B su 80 obiettivi <sec-big>

*Obiettivo:* capire se il risultato di Qwen3-8B dipende dalla dimensione o dalla famiglia (un secondo modello grande, Gemma 3-12B) e se regge su più obiettivi (Qwen3-8B anche sugli obiettivi 40–79, 80 in totale).

*Configurazione:* le celle con note private della sezione precedente (note divise con e senza esche, tutti con entrambe, completamente connesso e stella) con e senza l'istruzione a condividere. Gemma 3-12B su 40 obiettivi, Qwen3-8B su 80; in tabella anche i due 4B sugli stessi 40 obiettivi. Da qui in poi, accanto all'intervallo al 95%, riporto in grigio gli obiettivi vinti/persi nel confronto appaiato (test del segno esatto, vedi @sec-review).

#small(figure(
  table(columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, auto, auto), align: (left, left, right, right, right, right, right, right),
    rule,
    [Modello], [Note], [Muti], [Comun.], [\+ condividi], [1 agente, entrambe], [Comun. − muti], [Condividi − comun.],
    thin,
    ..for m in ("qwen3-4b", "gemma3-4b", "qwen3-8b", "gemma3-12b") { for (k, lab) in (("True", "divise + esche"), ("False", "divise")) { let v = d2.big.at(m).at(k)
      (NAME2.at(m), lab, p(v.mute), p(v.talk), p(v.share), p(v.one), cs(v.talk_minus_mute), cs(v.share_minus_talk)) } },
    thin,
    ..for m in ("qwen3-4b", "gemma3-4b", "qwen3-8b", "gemma3-12b") { let v = d2.big.at(m).all
      (NAME2.at(m), "tutti", p(v.mute), p(v.talk), "–", "–", cs(v.talk_minus_mute), "–") },
    rule),
  caption: [Note private, completamente connesso: successo (%) e differenze appaiate (punti). Qwen3-8B su 80 obiettivi, gli altri su 40.],
)) <tab-big>

- Gemma 3-12B non comunica le note di sua iniziativa: nel primo messaggio i detentori non nominano mai una ricetta del percorso, e la comunicazione aggiunge 2,5 punti. Con l'istruzione a condividere sale invece dal 52,5% al 92,5% (+40, 16 obiettivi vinti e nessuno perso), quanto un agente con entrambe le ricette: chi riceve la ricetta finale la gioca all'azione successiva nel 73–92% dei casi (Gemma 3-4B nel 40–45%).
- Il risultato di Qwen3-8B ha quindi due parti: comunicare spontaneamente è una caratteristica di Qwen, usare quello che si riceve migliora con la dimensione.
- Qwen3-8B su 80 obiettivi conferma tutto con intervalli più stretti: con le note divise la comunicazione aggiunge 32,5 punti (27 obiettivi vinti, 1 perso); quando tutti hanno entrambe le ricette perde 38,8 punti nel completamente connesso (1/32) ma ne guadagna 13,8 nella stella.
- Gemma 3-12B, quando tutti hanno entrambe le ricette, perde 22,5 punti nella stella (0/9): l'hub scrive a ogni turno "Let's focus on getting 'explosion' next!" e gli spoke rifanno l'intermedio.

= Il ciclo dell'intermedio: due correzioni del prompt <sec-loop>

*Obiettivo:* gli agenti rifanno l'intermedio che hanno già. Due ipotesi: non si accorgono di cosa hanno creato (l'inventario è una lista in cui nulla lo distingue), oppure imitano l'esempio del prompt, che mostra solo il primo passo di un piano a due passi ("I make 'robot' now and combine it with 'human' in the next round").

*Configurazione:* tutti con entrambe le ricette (con esche), gli stessi 40 obiettivi. (a) Una riga sotto l'inventario: "Items you made (already in your inventory): ...". (b) Prompt v3: il v2 più lo stesso esempio un turno dopo, con l'intermedio già nell'inventario e la risposta che gioca il secondo passo.

#small(figure(
  table(columns: (auto, auto, 1fr, 1fr, auto, 1fr), align: (left, left, right, right, right, right),
    rule,
    [Modello], [Squadra], [Senza riga], [Con riga], [Con − senza], [Ripetizioni (%)],
    thin,
    ..for m in M4 { for v in ("all_independent", "base", "one_agent") { let e = d2.made.at(m).at(v)
      (NAME2.at(m), VER.at(v), p(e.at("without")), p(e.at("with")), cs(e.diff), [#f(e.remake_without) → #f(e.remake_with)]) } },
    rule),
  caption: [(a) La riga con gli oggetti creati, completamente connesso: successo (%), differenza appaiata e quota delle azioni che rifanno l'intermedio già creato.],
)) <tab-made>

#small(figure(
  table(columns: (auto, auto, 1fr, 1fr, auto, 1fr), align: (left, left, right, right, right, right),
    rule,
    [Modello], [Squadra], [Prompt v2], [Prompt v3], [v3 − v2], [Ripetizioni (%)],
    thin,
    ..for m in ("qwen3-8b", "gemma3-12b", "gemma3-4b", "llama3.2-3b") { for v in ("all_independent", "base", "one_agent") { let e = d2.ex2.at(m).at(v)
      (NAME2.at(m), VER.at(v), p(e.v2), p(e.v3), cs(e.diff), [#f(e.remake_v2) → #f(e.remake_v3)]) } },
    rule),
  caption: [(b) L'esempio che gioca il secondo passo (prompt v3), completamente connesso: successo (%), differenza appaiata e ripetizioni dell'intermedio.],
)) <tab-ex2>

- La riga con gli oggetti creati rompe il ciclo per Qwen: Qwen3-8B con la comunicazione passa dal 62,5% al 92,5% (+30, 12/0) e le ripetizioni dall'80% al 41%; l'agente singolo guadagna 15 punti. Qwen3-4B +10 con la comunicazione. Gemma e Llama continuano a ripetere e il successo non cambia.
- Il nuovo esempio aiuta un po' Qwen3-8B con la comunicazione (+17,5, 8/1), ma le ripetizioni quasi non cambiano (dall'80% al 75%), e per Gemma 3-12B l'agente singolo peggiora (−12,5).
- Quindi non è l'esempio: gli agenti Qwen non riconoscono di avere già l'intermedio, e bisogna dirglielo. Con i nomi inventati (@sec-nomi) il ciclo peggiora proprio per Qwen. La riga va tenuta negli esperimenti futuri.

= Squadre miste

*Obiettivo:* la diversità tra famiglie di modelli aiuta come quella tra gli agenti di Llama?

*Configurazione:* squadre con Qwen3-4B, Gemma 3-4B e Llama 3.2-3B a turno (due agenti per famiglia nel completamente connesso, uno per famiglia nella stella), sulle impostazioni principali (20 obiettivi per impostazione) e sulle note private (40 obiettivi).

- Da mute risolvono quanto previsto pescando agenti indipendenti dalle squadre di un solo modello (a profondità 2 il 60% contro il 58% atteso) e non più di Llama da solo (0 punti di differenza): la diversità tra famiglie non aggiunge niente oltre a quella di Llama.
- Con la comunicazione nel completamente connesso perdono meno (−2,5 contro −5/−12,5 delle squadre di un solo modello) e ripetono meno i fallimenti dei compagni (20% contro 42% di Qwen3-4B, ma già da muti: 25% contro 31%, @sec-baseline); nella stella con le note divise guadagnano 17,5 punti (l'hub è sempre Qwen). Intervalli larghi: indicazioni, non risultati.

= Revisione critica dei risultati <sec-review>

Prima dei nuovi esperimenti ho riletto log e numeri delle sezioni precedenti cercando errori e spiegazioni alternative. Sono emersi quattro problemi:
+ *Messaggi persi:* il framework scartava il messaggio di un turno se l'azione veniva rifiutata (per esempio una combinazione con un oggetto che l'agente non ha ancora). Succedeva soprattutto al detentore della ricetta finale, che prova a giocarla prima che l'intermedio esista: per Gemma 3-4B si perdeva fino al 53% dei messaggi che la nominavano. Ho aggiunto un'opzione che pubblica comunque il messaggio.
+ *Due canali:* nel completamente connesso ogni combinazione riuscita viene mostrata a tutti, quindi la ricetta dell'intermedio può arrivare con i risultati anche senza messaggi. Le analisi precedenti contavano solo i messaggi.
+ *Una fuga nelle note divise:* le squadre mute risolvono il 20–50% anche con le note divise, perché chi ha la ricetta dell'intermedio spesso indovina quella finale (i modelli conoscono Little Alchemy). Il confronto "comunicazione − muti" sottostima quindi quello che la comunicazione mette insieme, e confonde la dimensione del modello con la sua memoria del gioco.
+ *L'esempio nel prompt* mostra solo il primo passo di un piano a due passi: una possibile causa del ciclo dell'intermedio (verificata nella @sec-loop).

Sul piano statistico: alcuni intervalli bootstrap che escludono lo 0 si basano su pochissimi obiettivi discordanti (per esempio 4 vinti e 0 persi) e non passano un test del segno esatto (p = 0,13). Da qui in poi riporto anche gli obiettivi vinti/persi, e tratto come esplorativi gli effetti per singolo modello con pochi obiettivi discordanti.

= Messaggi persi: l'informazione arriva, il successo cambia poco

*Configurazione:* le squadre con l'istruzione a condividere (note divise, completamente connesso) rifatte pubblicando anche i messaggi dei turni rifiutati; stessi 40 obiettivi.

#small(figure(
  table(columns: (auto, auto, 1fr, 1fr, auto, 1fr, 1fr), align: (left, left, right, right, right, right, right),
    rule,
    [Modello], [Note], [Condividi], [\+ pubblica], [Differenza], [Riceventi informati (su 5)], [La giocano dopo (%)],
    thin,
    ..for m in ("qwen3-4b", "gemma3-4b", "qwen3-8b") { for (k, lab) in (("True", "divise + esche"), ("False", "divise")) { let e = d2.pub.at(m).at(k)
      (NAME2.at(m), lab, p(e.share), p(e.pub), cs(e.diff), [#f2(e.told_share) → #f2(e.told_pub)], [#f(e.play_share) → #f(e.play_pub)]) } },
    rule),
  caption: [Pubblicare i messaggi dei turni rifiutati: successo (%), differenza appaiata, compagni che ricevono la ricetta finale da chi la detiene, e quota di chi la gioca all'azione successiva quando può.],
)) <tab-pub>

- La ricetta finale raggiunge quasi tutti (Gemma 3-4B da 2,1 a 3,2 riceventi su 5, Qwen3-4B da 4,0 a 4,9), ma il successo cambia da 0 a +7,5 punti, senza differenze significative. Il limite dei 4B resta l'uso di quello che ricevono, non i messaggi persi, e il confronto tra dimensioni della @sec-big non dipendeva da questo errore.

= Quale canale porta l'informazione <sec-canali>

*Obiettivo:* capire se il guadagno di Qwen3-8B viene dai messaggi o dai risultati condivisi.

*Configurazione:* note divise con esche, Qwen3-8B e Gemma 3-12B, completamente connesso con solo i risultati oppure solo i messaggi, e solo i messaggi con l'istruzione a condividere; confronto con le squadre mute e con entrambi i canali sugli stessi 40 obiettivi.

#figure(image("figures/v2_channels.svg", width: 90%),
  caption: [Successo (%) per canale. A destra della linea tratteggiata, con l'istruzione a condividere.]) <fig-channels>

- Per Qwen3-8B il guadagno senza istruzione viene dai risultati: solo risultati +#f(d2.channels.at("qwen3-8b").results_minus_mute.d) punti sulle mute (10/1), solo messaggi +2,5; con entrambi i canali 47,5%, quasi come con i soli risultati (42,5%). Chi risolve è il detentore della ricetta finale (16 run su 17): vede che un compagno ha creato l'intermedio e gioca la propria nota.
- I messaggi da soli non bastano: i detentori nominano la ricetta finale nel 65% delle run, ma chi la riceve la gioca all'azione successiva solo nel 10% dei casi, perché non ha ancora l'intermedio.
- Con l'istruzione a condividere i messaggi da soli portano quasi tutto: +37,5 punti per Qwen3-8B e +35 per Gemma 3-12B sui soli messaggi senza istruzione, a 12,5 e 7,5 punti da entrambi i canali. Ora chi risolve è soprattutto il detentore dell'intermedio, che riceve la ricetta finale.

= Oggetti con nomi inventati <sec-nomi>

*Obiettivo:* eliminare la fuga. Se nessuna ricetta si può indovinare, con le note divise l'unico modo per risolvere è metterle insieme.

*Configurazione:* gli stessi obiettivi e le stesse note, ma ogni oggetto ha un nome inventato ('philosophy' + 'rivulet' = 'small' diventa 'vidob' + 'defeseg' = 'zonut'), e il prompt non parla più di intuizione del mondo reale. Qwen3-4B, Qwen3-8B, Gemma 3-4B e Gemma 3-12B nelle stesse condizioni (muti, comunicazione, istruzione a condividere, agente singolo, tutti con entrambe); i messaggi dei turni rifiutati vengono pubblicati.

#figure(image("figures/v2_renamed.svg", width: 100%),
  caption: [Note divise con esche, completamente connesso: successo (%) con i nomi veri e con i nomi inventati, sugli stessi obiettivi e note.]) <fig-renamed>

#small(figure(
  table(columns: (auto, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, right),
    rule,
    [Modello], [Comun. − muti], [Condividi − comun.], [Inventati − veri: comun.], [Inventati − veri: condividi],
    thin,
    ..for m in ("qwen3-4b", "qwen3-8b", "gemma3-4b", "gemma3-12b") { let e = d2.renamed.at(m)
      (NAME2.at(m), cs(e.talk_minus_mute), cs(e.share_minus_talk), cs(e.renamed_minus_real.talk), cs(e.renamed_minus_real.share)) },
    rule),
  caption: [Nomi inventati, note divise: differenze appaiate (punti), e confronto con gli stessi obiettivi a nomi veri.],
)) <tab-renamed>

- La fuga sparisce: con le note divise le squadre mute risolvono 0 obiettivi su 40 per ogni modello, come gli agenti casuali.
- Ora la comunicazione mette insieme le note: +35 punti per Qwen3-4B (14/0), +45 per Qwen3-8B (18/0), +17,5 per Gemma 3-12B (7/0), +7,5 per Gemma 3-4B. Con i nomi veri il guadagno di Qwen3-4B era nascosto dalle squadre mute che indovinavano: la sua squadra che comunica risolve quanto prima (35% contro 30%), mentre quella muta crolla.
- Con l'istruzione a condividere Qwen3-8B risolve il 92,5%, più di un agente con entrambe le ricette (37,5%) e di una squadra muta in cui tutti le hanno (60%); Gemma 3-12B l'87,5%. Gemma 3-4B resta al 7,5%: anche con l'istruzione nomina la ricetta solo nel 30% delle run, e chi la riceve la gioca nel 17% dei casi.
- Gemma 3-12B con i nomi veri risolveva invece soprattutto indovinando: con la comunicazione scende dal 52,5% al 17,5% (1/15), con l'istruzione resta all'87,5%.
- La dimensione conta per usare le ricette ricevute (con l'istruzione Qwen3-8B − Qwen3-4B +35 punti, 15/1; Gemma 3-12B − Gemma 3-4B +80, 32/0), molto meno per comunicarle spontaneamente (+10 per entrambe le famiglie, non significativo).
- Senza nomi veri il ciclo dell'intermedio peggiora per Qwen: l'agente singolo con entrambe le ricette rifà l'intermedio nell'#f(d2.renamed.at("qwen3-8b").logs.remake_one_renamed)% delle azioni successive (Qwen3-8B; #f(d2.renamed.at("qwen3-8b").logs.remake_one_real)% con i nomi veri) e scende dal 67,5% al 37,5%; la squadra Qwen3-8B in cui tutti hanno entrambe le ricette e comunicano scende al 15%. Con 'small' nell'inventario si capisce che 'alchemist' + 'small' ora si può giocare, con 'zonut' no. Per Gemma le ripetizioni non cambiano.

= Confidenza e dimensione della squadra <sec-conf>

*Obiettivo:* verificare l'ipotesi del tutor: una squadra più grande aiuta un modello solo quando è incerto (giusto per esempio una volta su tre, così uno dei sei agenti trova la risposta), non quando sbaglia o indovina "con sicurezza".

*Configurazione:* (a) dalle squadre mute già fatte: i sei agenti ricevono lo stesso primo prompt, quindi le loro sei prime risposte sono campioni della stessa distribuzione; (b) 12 agenti muti sulle impostazioni principali (i primi sei rigiocano esattamente i sei agenti già fatti, quindi 12 − 6 misura cosa aggiungono i sei nuovi); (c) la probabilità che il modello assegna alla combinazione scelta, registrata durante le nuove run e ricalcolata per le vecchie (circa 5.900 run di quattro modelli). Ricalcolo e registrazione coincidono (correlazione 0,996).

#figure(image("figures/v2_size.svg", width: 100%),
  caption: [A sinistra, successo delle squadre mute per numero di agenti (impostazioni principali, 40 obiettivi). A destra, per ogni obiettivo, probabilità media della prima combinazione e quota dei 12 agenti che giocano la coppia più comune.]) <fig-size2>

#small(figure(
  table(columns: (auto, 1fr, 1fr, 1fr, 1fr, auto, auto, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, right, right, right, right, right, right, right),
    rule,
    [Modello], [1], [3], [6], [12], [12 − 6], [12 − 3], [P giusta], [P errata], [AUROC], [ρ],
    thin,
    ..for m in M3 { let e = d2.size.at(m)
      (NAME2.at(m), p(e.at("1")), p(e.at("3")), p(e.at("6")), p(e.at("12")), cs(e.at("12-6")), cs(e.at("12-3")),
       f2(e.conf.p_right), f2(e.conf.p_wrong), f2(e.conf.auroc), f2(e.conf.rho_agreement)) },
    rule),
  caption: [Successo (%) con 1, 3, 6 e 12 agenti muti; probabilità media della prima combinazione quando è giusta e quando è errata; AUROC (0,5 = non distingue); ρ: correlazione per obiettivo (Spearman) tra probabilità e accordo tra agenti.],
)) <tab-size>

- Gli agenti Qwen e Gemma rispondono quasi uguali: a profondità 1 l'86–91% dei sei agenti gioca la coppia più comune, contro il 56% di Llama. Per questo solo Llama guadagna da squadre più grandi.
- 12 agenti non aggiungono niente a 6 per Qwen3-4B e quasi niente per Gemma (+2,5) e Llama (+5); Llama guadagna ancora da 3 a 12 (+20, 8/0). La previsione fatta dai primi sei agenti si conferma sui sei nuovi (per esempio Llama a profondità 1: 0,99 previsto, 1,00 osservato).
- La probabilità della combinazione non distingue le risposte giuste dalle sbagliate (AUROC 0,44–0,64; Qwen3-4B è più sicuro quando sbaglia), ma misura quanto gli agenti sono d'accordo (correlazione per obiettivo 0,49–0,85). E dove il modello è incerto più agenti aiutano di più (Gemma: +20 punti da 1 a 12 agenti negli obiettivi incerti, +5 in quelli sicuri).
- L'ipotesi regge in questa forma: una squadra guadagna in proporzione a quanto i suoi agenti sono in disaccordo, e la confidenza del modello misura il disaccordo, non la correttezza. Un modello sicuro e sbagliato non si salva aggiungendo agenti.

= Turni in sequenza e catena

*Obiettivo:* altro suggerimento del tutor: se gli agenti giocano uno dopo l'altro e vedono le mosse di chi li precede nello stesso turno, nasce una cascata utile, in cui chi viene dopo evita gli errori di chi viene prima?

*Configurazione:* due protocolli, sulle impostazioni principali e sulle note private: _in sequenza_ (completamente connesso, ma l'agente k vede le risposte degli agenti 0…k−1 dello stesso turno) e _catena_ (ogni agente riceve solo risultati e messaggio del precedente). Qwen3-4B, Qwen3-8B, Gemma 3-4B e Llama 3.2-3B, confrontati con il completamente connesso in parallelo e con la squadra muta sugli stessi obiettivi.

#figure(image("figures/v2_chain.svg", width: 82%),
  caption: [Impostazioni principali: differenze appaiate (punti) della catena e dei turni in sequenza rispetto al completamente connesso in parallelo e alla squadra muta.]) <fig-chain>

- La catena batte il completamente connesso in parallelo di #f(d2.chain.pooled.chain_minus_par.d) punti in media (20 obiettivi vinti, 2 persi), ma è pari alla squadra muta (+#f(d2.chain.pooled.chain_minus_mute.d)): recupera quello che la comunicazione completa fa perdere e spende il 20–30% di token in meno. Nella prima stesura attribuivo il recupero al fatto che gli agenti copiano meno (per Gemma 1,0 eventi di copia per run contro 2,2), ma era un artefatto della misura, che conta solo le copie dall'unico agente ascoltato: contando tutti i compagni la catena copia quanto il parallelo o di più (Gemma #f2(d2.adopt.chain_copies.at("gemma3-4b").chain) contro #f2(d2.adopt.chain_copies.at("gemma3-4b").par), @sec-baseline). Perché recuperi resta aperto; un'ipotesi è che ogni agente legga molto meno testo per turno (@sec-lett).
- I turni in sequenza non creano una cascata utile: pari al parallelo (+2,5) e sotto la squadra muta (−7,5, 4/16). Gli agenti Qwen giocano già la stessa prima coppia anche in parallelo (88–89%), e chi gioca dopo non risolve più di chi gioca prima.
- Con le note: quando tutti hanno entrambe le ricette, la sequenza peggiora Qwen3-8B (−22,5 sul parallelo, −47,5 sulla squadra muta): vedere "prima fai X" nello stesso turno rafforza il ciclo dell'intermedio. Con le note divise Llama guadagna (+15, 6/0) e Qwen3-8B perde (−17,5).

= Una correzione: adozione e copie a confronto con la squadra muta <sec-baseline>

*Problema:* le misure di "adozione" (giocare il risultato di un compagno, ripetere un suo fallimento) e gli "eventi di copia" (trovare un oggetto dopo un compagno) contano solo ciò che un agente ha ricevuto, quindi per la squadra muta valgono zero per costruzione, e finora la comunicazione era confrontata con zero. Ma gli agenti di uno stesso modello giocano spesso le stesse coppie anche senza parlarsi (@sec-conf). La letteratura recente misura l'influenza dei pari proprio contro un controllo senza messaggi (@sec-lett).

*Correzione:* alla squadra muta do la casella che il completamente connesso le avrebbe consegnato (le azioni dei compagni nel turno precedente) e conto le copie rispetto a tutti i compagni. Sulle squadre che comunicano questo riproduce esattamente i valori registrati, quindi "comunicazione − muti" è ciò che la comunicazione aggiunge. Nessuna nuova run.

#let cc(c) = [#str(calc.round(c.d, digits: 2)).replace(".", ",") #text(fill: rgb("#52514e"), size: 0.9em)[[#str(calc.round(c.lo, digits: 2)).replace(".", ","); #str(calc.round(c.hi, digits: 2)).replace(".", ",")]]]
#let arrow(c) = f(c.sil) + " → " + f(c.talk)
#let arrow2(c) = f2(c.sil) + " → " + f2(c.talk)

#small(figure(
  table(columns: (auto, 0.9fr, 1.3fr, 0.9fr, 1.3fr, 0.9fr, 1.3fr), align: (left, right, right, right, right, right, right),
    rule,
    [Squadra], [Gioca un risultato (%)], [comun. − muti], [Ripete un fallimento (%)], [comun. − muti], [Copie per run], [comun. − muti],
    thin,
    ..for (m, e) in (..M4.map(m => (m, d2.adopt.main.fully_connected.at(m))), ("all", d2.adopt.main.fully_connected.pooled), ("mix", d2.adopt.mixed.fully_connected.at("mix-qgl"))) {
      (if m == "mix" { "Mista (Q/G/L)" } else { NAME.at(m) },
       arrow(e.result_adoption), ci(e.result_adoption), arrow(e.blind_adoption), ci(e.blind_adoption), arrow2(e.copy_events), cc(e.copy_events)) },
    thin,
    ..for (k, lab) in (("fully_connected", "Entrambi i canali"), ("fully_connected_results", "Solo risultati"), ("fully_connected_messages", "Solo messaggi")) {
      let e = d2.adopt.followup.at(k).pooled
      (lab, arrow(e.result_adoption), ci(e.result_adoption), arrow(e.blind_adoption), ci(e.blind_adoption), arrow2(e.copy_events), cc(e.copy_events)) },
    rule),
  caption: [Completamente connesso, muti → con comunicazione, e differenza appaiata [intervallo bootstrap per obiettivo]. Righe in alto: esperimento principale (e la squadra mista, 40 obiettivi); in basso: follow-up per canale, media dei tre modelli da 4B. "Ripete un fallimento" era la colonna "Seguiti alla cieca".],
)) <tab-baseline>

- *Ripetere un fallimento non dipende dalla comunicazione:* in media 26% da muti e 27% con la comunicazione (+0,8). Gli agenti Qwen lo fanno già da muti (31–43%) perché sono quasi copie l'uno dell'altro; Llama lo fa meno quando comunica (−10,6). Quello che nella v1 chiamavo "seguire alla cieca" è soprattutto accordo tra agenti, non imitazione.
- *La comunicazione fa giocare i risultati dei compagni* (+16 punti in media, Gemma +25) e aggiunge circa mezza copia per run.
- *I due canali agiscono in modo opposto:* con i soli risultati gli agenti rifanno quello che hanno fatto i compagni (+1,1 copie per run) e ripetono meno i loro fallimenti, perché li vedono (−4,9); con i soli messaggi riprovano i suggerimenti dei compagni anche quando sono falliti (+9,1, Qwen3-4B +16,7) senza copiare di più. Con entrambi i canali i fallimenti mostrati compensano il secondo effetto.
- *Cosa cambia:* la perdita del completamente connesso e il calo dei percorsi distinti restano; cadono "l'effetto gregge è tipico di Qwen" misurato con la ripetizione dei fallimenti (è accordo, non imitazione), "le squadre miste seguono meno alla cieca" (i loro agenti si somigliano meno: 25% già da muti) e "la catena copia meno" (contando tutti i compagni copia quanto il parallelo o di più: Gemma #f2(d2.adopt.chain_copies.at("gemma3-4b").chain) contro #f2(d2.adopt.chain_copies.at("gemma3-4b").par), Llama #f2(d2.adopt.chain_copies.at("llama3.2-3b").chain) contro #f2(d2.adopt.chain_copies.at("llama3.2-3b").par)).

= Confronto con la letteratura <sec-lett>

Una ricerca bibliografica fatta senza conoscere questo progetto, verificata sugli articoli (note in `docs/LITERATURE_2026-11-22.md`), arriva alla stessa conclusione generale: i guadagni attribuiti alla comunicazione tra agenti LLM spesso spariscono a parità di calcolo, e la comunicazione aiuta soprattutto quando l'informazione è davvero distribuita. I lavori più vicini:

- *A parità di costo un agente basta:* il dibattito spesso non batte self-consistency e chain-of-thought (Zhang et al., arXiv 2502.08788; Choi et al., 2508.17536, dove il voto spiega gran parte del guadagno); in ALFWorld una squadra pianificatore–esecutore–critico su un modello da 7B non batte un agente singolo a parità di chiamate (2609.04217). Come qui: l'agente singolo con il budget della squadra pareggia sei agenti muti e batte sei che comunicano, e gli agenti casuali sono una baseline forte.
- *Conformità in squadre omogenee di modelli piccoli:* con 10 agenti da 7–8B dello stesso modello il dibattito fa convergere sulla risposta più comune (fino all'85,5%), già con 2 pari, e costa 2–3 volte i token dell'autocorrezione isolata; un controllo con ragionamenti di problemi non correlati batte il dibattito (2605.00914). Corrisponde al calo del completamente connesso e all'accordo tra agenti Qwen e Gemma.
- *L'influenza va misurata contro un controllo senza pari:* il 37% degli agenti cambia risposta anche senza vedere nessuno (2606.00820); ICR (2610.01042, proprio con Qwen3-4B e Qwen3-8B) separa la correzione di un ricevente sbagliato dalla conservazione di uno giusto e trova che messaggi più ricchi aumentano entrambe le influenze, buona e cattiva. Da qui la correzione della @sec-baseline.
- *Chi viene creduto:* misurando i logit del ricevente con messaggi tolti o riattribuiti, un "track record" dichiarato del mittente sposta le decisioni più dello status, e attribuire un testo al ricevente stesso pesa ancora di più (2610.35930); profili di affidabilità dei pari costruiti dalla storia delle interazioni migliorano Qwen3-4B (2601.21742). Qui invece dire ai riceventi che le note dei compagni sono vere non ha cambiato niente (sezione "Comunicare e usare le note"): la differenza da capire è tra spostare una preferenza e giocare davvero la ricetta, spesso senza avere ancora l'intermedio.
- *Quando la comunicazione aiuta:* un sistema multi-agente può simulare un agente singolo se i messaggi non hanno limiti; il guadagno nasce dal comprimere il contesto, e aiuta soprattutto i modelli deboli quando i messaggi conservano l'informazione necessaria (2607.16133). È coerente con la catena (meno testo da leggere, recupera la perdita), con la riga degli oggetti creati e con i nomi inventati, dove l'informazione necessaria è una sola ricetta.
- *Eterogeneità e indipendenza:* mescolare modelli migliora il dibattito rispetto alla media dei due modelli (2502.08788); confrontate con la famiglia migliore, come qui, le squadre miste mute non guadagnano. Le architetture a voto presumono errori indipendenti (2601.22290); qui gli agenti di un modello sbagliano insieme (@sec-conf).

Cosa aggiunge questo lavoro: un compito agentico in cui l'ambiente verifica ogni azione (il canale dei risultati è informazione verificata, i messaggi no), note private che non si possono indovinare, una baseline di agenti casuali e i risultati per singolo modello.

= Conclusioni

+ *La squadra vale per i tentativi diversi, e la diversità si può misurare:* il guadagno di agenti in più dipende da quanto i campioni del modello sono in disaccordo (Llama molto, Qwen quasi nulla), e la confidenza del modello sulle proprie risposte misura proprio questo disaccordo, non la correttezza. Gli agenti casuali restano una baseline forte.
+ *Il completamente connesso è dannoso quando non c'è niente da trasmettere*, per un effetto gregge: i risultati condivisi fanno rifare quello che hanno fatto i compagni, i messaggi fanno riprovare i loro suggerimenti anche quando falliscono, e i percorsi esplorati calano. Le misure di "adozione" vanno sempre confrontate con la squadra muta, i cui agenti giocano già spesso le stesse coppie (@sec-baseline). Strutture con meno contatti (stella, catena) recuperano la perdita ma non battono gli agenti muti; i turni in sequenza non aiutano.
+ *Quando l'informazione è davvero divisa, la comunicazione la mette insieme.* Con i nomi inventati le squadre mute risolvono lo 0% e quelle che comunicano fino al 45% senza istruzioni, fino al 92,5% con l'istruzione a condividere, più di un agente con tutte le note. Con i nomi veri questo era nascosto dagli agenti che indovinavano.
+ *Comunicare spontaneamente è di Qwen, usare quello che si riceve cresce con la dimensione.* Senza istruzione il guadagno passa soprattutto dai risultati condivisi (si vede che un compagno ha creato l'intermedio), non dai messaggi. Gemma 3-4B non usa quasi nulla di quello che riceve.
+ *Un errore di stato, non di comunicazione, limita le squadre:* gli agenti (soprattutto Qwen) rifanno l'intermedio che hanno già, e la comunicazione che ripete il piano lo rafforza. Dirgli cosa hanno creato lo risolve, cambiare l'esempio no.
+ Gli effetti per singolo modello vanno letti insieme al test del segno: molti si basano su pochi obiettivi discordanti.

= Prossimi passi

- *Nomi inventati con la riga degli oggetti creati e l'agente singolo con il budget della squadra:* separa la capacità di mettere insieme le note dal ciclo dell'intermedio, e verifica a parità di token il risultato principale (Qwen3-8B con l'istruzione 92,5% contro 37,5% di un agente con entrambe le ricette).
- *Misurare l'effetto dei singoli messaggi senza nuove run:* ripunteggiare i turni registrati con il messaggio, senza, o sostituito da quello di un altro obiettivo, misurando quanto cambia la probabilità della mossa giusta (il metodo di 2610.35930). Dice se Gemma 3-4B ignora le ricette ricevute e se nel completamente connesso pesa il contenuto dei messaggi o solo la loro presenza.
- *Il lato del ricevente:* con i nomi inventati e l'istruzione a condividere, provare una riga con l'affidabilità dei compagni (quante ricette riportate erano valide), un elenco di chi sa cosa, un'istruzione generica ("di' agli altri solo quello che sanno e tu no") al posto di quella specifica, e due o tre riformulazioni dell'istruzione.
- *Messaggi placebo nel completamente connesso* (messaggi presi da un altro obiettivo), se il punto precedente indica che conta la presenza dei messaggi.
- Un secondo modello da 8B di un'altra famiglia (per esempio Llama 3.1-8B) sui nomi inventati, e decidere se usare d'ora in avanti i nomi inventati e la riga degli oggetti creati come impostazioni predefinite.

#v(1em)
