import hashlib
import json
import re

import requests

from src.config import (
    API_URL, MODEL_NAME, OUTPUTS_DIR,
    S2_BATCH_SIZE, S2_BATCH_STRIDE, S2_MAX_TOKENS, S2_RETRY_TEMPERATURES, S2_TIMEOUT,
)


"""
Stage 2: Action Grouping (raggruppamento delle transizioni in azioni).

Le transizioni dello Stage 1 descrivono mezzo secondo alla volta.
Un LLM, guardando SOLO IL TESTO (niente immagini), le raggruppa in azioni complete
usando gli stati start / continue / end / none come indizi per i confini.

Il paper passa tutto il video in una volta; con un modello piccolo e un contesto corto
lavoriamo invece a BLOCCHI di 20 transizioni che si sovrappongono di 5, e alla fine
uniamo le azioni dei blocchi.
"""


# Etichetta della versione del prompt (v3 = terza versione).
# Entra nell'impronta dei checkpoint: se modifichi il prompt, cambiala (es. "..._v4")
# cosi' i blocchi gia' elaborati con il prompt vecchio vengono rifatti automaticamente.
PROMPT_VERSION = "stage2_lines_en_v3"


def make_batches(transitions):
    """Divide le transizioni in blocchi da S2_BATCH_SIZE, uno ogni S2_BATCH_STRIDE.
    Con 20 e 15: blocco 1 = transizioni 0-19, blocco 2 = 15-34 --> (5 in comune).
    La sovrapposizione evita che un'azione a cavallo tra due blocchi venga tagliata."""

    batches, start = [], 0

    while start < len(transitions):

        batches.append(transitions[start:start + S2_BATCH_SIZE])

        if start + S2_BATCH_SIZE >= len(transitions):
            break

        start += S2_BATCH_STRIDE #incremento di start attraverso lo stride
    return batches


def compute_batch_hash(transitions):
    """Impronta di un blocco: cambia se cambiano le transizioni o il prompt.
    Un checkpoint vale solo se la sua impronta coincide con quella attuale."""

    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "transitions": [(t["transition_id"], t["description"], t["status"]) for t in transitions],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] 
    # trasforma il testo in byte, calcola l'impronta con SHA-256 e infine restituisce i primi 16 esadecimali


def load_checkpoint(checkpoint_file, batch_hash, force):
    """Risultato salvato di un blocco, oppure None se va (ri)calcolato."""

    if force or not checkpoint_file.exists():
        return None
    
    ckpt = json.loads(checkpoint_file.read_text(encoding="utf-8"))

    if ckpt.get("batch_hash") != batch_hash:
        print("Checkpoint obsoleto: ricalcolo.")
        return None
    
    print("Checkpoint valido trovato: recupero risultato.")
    return ckpt["result"]


def save_json(path, data):
    """Scrittura atomica"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)



def build_prompt(transitions):
    """Costruisce il testo che viene mandato al modello"""

    transitions_text = "\n".join(
        f"{t['transition_id']} | {t['status']} | {t['description']}" for t in transitions
    )
    # in ogni riga mette "<id> | <stato> | <descrizione>""

    return f"""Group the following egocentric video transitions into atomic physical actions.
                Each line is: <id> | <status> | <caption (0.5s)>

                RULES:
                - Follow chronological order.
                - Use <status> to find boundaries:
                * "start" begins a new physical action.
                * "continue" continues the same action.
                * "end" finishes the current action.
                * "none" means no interaction (skip these IDs).
                - Merge consecutive IDs that belong to the same physical manipulation.
                - Split into separate lines when a new physical action starts.
                - Write each action label in English as a short, concrete, single-verb phrase.
                - If no valid actions exist, output: NONE

                Return ONLY lines in this exact format (no JSON, no bullets, no extra text):
                <start_id>-<end_id> | <short English action phrase>

                TRANSITIONS:
                {transitions_text}
                """


def group_actions(transitions, temperature):
    """Chiede al modello di raggruppare un blocco di transizioni. 
    Restituisce il testo grezzo."""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": build_prompt(transitions)}],
        "temperature": temperature,
        "max_tokens": S2_MAX_TOKENS,
    }

    response = requests.post(API_URL, json=payload, timeout=S2_TIMEOUT)

    if not response.ok:
        print("\nERRORE HTTP:", response.status_code)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


LINE_RE = re.compile(r"^\s*(\d+)\s*[-–to]+\s*(\d+)\s*\|\s*(.+?)\s*$", re.IGNORECASE)


def split_consecutive(ids):
    """In presenza di un buco tra la lista di id consecutivi, si effettua lo splitting"""

    if not ids:
        return []
    
    chunks, current = [], [ids[0]]
    for prev, cur in zip(ids, ids[1:]):

        if cur == prev + 1: # se è consecutivo lo aggiunge al chunk
            current.append(cur)
        else: # se c'è un salto il gruppo in corso è finito e va messo tra quelli completati
            chunks.append(current)
            current = [cur] # si apre un nuovo gruppo

    chunks.append(current)
    return chunks


def parse_and_build_actions(raw_text, transitions):
    """Trasforma la risposta del modello in azioni con tempi di inizio e fine.
    Sistema eventuali errori(quelli tollerati) altrimenti solleva un errore"""


    transition_map = {t["transition_id"]: t for t in transitions}
    expected_ids = set(transition_map)
    min_id, max_id = min(expected_ids), max(expected_ids)

    if raw_text.strip().upper() == "NONE":  # il modello dice: nessuna azione
        return {"actions": [], "ignored_transition_ids": sorted(expected_ids)}

    seen_in_groups = set() # id gia' assegnati a un'azione
    groups = []
    for raw_line in raw_text.splitlines():
        # toglie spazi ed eventuali simboli di elenco, saltando righe vuote 
        # e che non rispettano il formato
        
        line = raw_line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue


        # estrae l'etichetta e i due numeri, raddrizzando gli intervalli se necessario
        # scartando le righe senza etichetta
        start_id, end_id = int(match.group(1)), int(match.group(2))
        label = match.group(3).strip().strip("\"'")
        if start_id > end_id:
            start_id, end_id = end_id, start_id
        if not label or "<short" in label.lower(): # il modello ha ricopiato il modello di riga
            continue


        # decide quali transizioni appartengono davvero a quest'azione
        # - tiene solo gli id che esistono davvero nel blocco
        # - scarta quelli assegnati ad un'azione precedente(ogni transizione appartiene al massimo ad un'azione)
        valid_ids = [
            tid for tid in range(max(start_id, min_id), min(end_id, max_id) + 1)
            if tid in expected_ids and tid not in seen_in_groups
        ]

        # se rimane qualche id allora li segna come presi e crea l'azione
        # se ci sono buchi, split_consecutive la divide in più azioni con la stessa etichetta
        if valid_ids:
            seen_in_groups.update(valid_ids)
            for chunk in split_consecutive(valid_ids):     # un buco nell'intervallo -> piu' azioni
                groups.append({"transition_ids": chunk, "action_label": label})



    # se non è stata trovata nessuna azione valida, ma ne blocco c'erano interazioni valide il modello ha sbagliato
    # la risposta viene scartata e si riprova
    if not groups and any(t["status"] != "none" for t in transitions):
        raise ValueError("Nessuna riga valida '<start>-<end> | <action>' trovata nella risposta.")

    groups.sort(key=lambda g: g["transition_ids"][0]) # in ordine di tempo

    # per ogni azione crea il dizionario
    actions = [
        {
            "action_id": action_id,
            "action_label": group["action_label"],
            "start_time": transition_map[group["transition_ids"][0]]["start_time"],   # inizio 1a transizione
            "end_time": transition_map[group["transition_ids"][-1]]["end_time"],      # fine ultima transizione
            "transition_ids": group["transition_ids"],
        }
        for action_id, group in enumerate(groups)
    ]


    return {"actions": actions, "ignored_transition_ids": sorted(expected_ids - seen_in_groups)}



def merge_overlapping_actions(all_actions, transition_map):
    """Unisce le azioni dei blocchi sovrapposti: due azioni che condividono almeno
    una transizione diventano una sola.

    Le azioni vengono ordinate per prima transizione,
    poi ognuna viene confrontata con l'ultima azione già unita."""

    if not all_actions:
        return []

    # ordina per prima transizione, e a parità di inizio dalla più lunga alla più corta
    all_actions.sort(key=lambda a: (a["transition_ids"][0], -len(a["transition_ids"])))
    

    merged = []
    for act in all_actions:

        # la prima azione entra direttamente, perchè non c'è niente con cui confrontarla
        if not merged:
            merged.append(act)
            continue


        prev = merged[-1]
        prev_set, cur_set = set(prev["transition_ids"]), set(act["transition_ids"])

        # se c'è intersezione tra transizioni, le due azioni si sovrappongono
        if prev_set & cur_set: # hanno transizioni in comune
            combined_ids = sorted(prev_set | cur_set)
            label = prev["action_label"] if len(prev_set) >= len(cur_set) else act["action_label"]
            merged.pop() # toglie dalla lista l'azione precedente e aggiunge quella unita, ricalcolando inizio e fine
            for chunk in split_consecutive(combined_ids): # resta continua nel tempo
                merged.append({
                    "action_label": label,
                    "start_time": transition_map[chunk[0]]["start_time"],
                    "end_time": transition_map[chunk[-1]]["end_time"],
                    "transition_ids": chunk,
                })
        else:
            merged.append(act)

    for idx, act in enumerate(merged): # rinumera da 0
        act["action_id"] = idx
    return merged



def process_batch(batch, batch_index, raw_file_handle):
    """Elabora un blocco provando temperature crescenti finche' la risposta e' leggibile.
    """

    for attempt, temp in enumerate(S2_RETRY_TEMPERATURES, start=1):
        try:
            print(f"Elaborazione (tentativo {attempt}/{len(S2_RETRY_TEMPERATURES)}, temp={temp})...")
            raw_result = group_actions(batch, temperature=temp)
            batch_result = parse_and_build_actions(raw_result, batch)
        except ValueError as error: # risposta illeggibile: si riprova con temperatura più alta
            print(f"  [Avviso] Errore parsing: {error}")
            continue

        # risposta letta correttamente 
        raw_file_handle.write(f"\n\n===== BLOCCO {batch_index} (Try {attempt}, T={temp}) =====\n{raw_result}")
        raw_file_handle.flush() # forza scrittura immediata sul disco
        return batch_result
    return None


def run(video_id, force=False):
    """Esegue lo Stage 2 su un video. 
    I blocchi gia' fatti vengono ripresi dai checkpoint."""


    # costruisce i percorsi inuput e output
    input_file = OUTPUTS_DIR / video_id / "stage1" / "transitions_cleaned.json"
    output_dir = OUTPUTS_DIR / video_id / "stage2"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # controlla che lo stage 1 sia stato fatto, carica le transizioni e prepara il dizionario
    if not input_file.exists():
        raise FileNotFoundError(f"File JSON di input non trovato: {input_file}")
    
    transitions = json.loads(input_file.read_text(encoding="utf-8"))
    transition_map = {t["transition_id"]: t for t in transitions}

    
    batches = make_batches(transitions)
    print(f"Transizioni caricate: {len(transitions)} | Blocchi: {len(batches)} "
          f"(size={S2_BATCH_SIZE}, stride={S2_BATCH_STRIDE})")

    
    raw_actions, all_ignored, failed_blocks = [], set(), []

    
    with open(output_dir / "action_grouping_stage2_raw.txt", "a", encoding="utf-8") as raw_file_handle:


        for batch_index, batch in enumerate(batches, start=1):
            # per ogni blocco costruisce il nome del suo checkpoint, calcola l'impronta e prova a caricare il risultato salvato
            print("\n" + "=" * 55)
            print(f"STAGE 2 - BLOCCO {batch_index}/{len(batches)} (Transizioni: {len(batch)})")

            checkpoint_file = checkpoint_dir / f"block_{batch_index:02d}.json"
            batch_hash = compute_batch_hash(batch)
            batch_result = load_checkpoint(checkpoint_file, batch_hash, force)

            # se non c'è un checkpoint valido, elabora il blocco con il modello
            if batch_result is None:
                batch_result = process_batch(batch, batch_index, raw_file_handle)

                # se dopo tutti i tentativi fallisce passa al blocco successivo, senza salvare nessun checkpoint
                if batch_result is None:
                    print(f"  [ERRORE] Blocco {batch_index} fallito dopo {len(S2_RETRY_TEMPERATURES)} tentativi.")
                    failed_blocks.append(batch_index)
                    continue

                # se riesce salva subito il checkpoint 
                save_json(checkpoint_file, {"batch_hash": batch_hash, "prompt_version": PROMPT_VERSION,
                                            "result": batch_result})

            # aggiunge le azioni del blocco ai contenitori
            raw_actions.extend(batch_result["actions"])
            all_ignored.update(batch_result["ignored_transition_ids"])
            print(f"Azioni trovate: {len(batch_result['actions'])} | "
                  f"Ignorate: {len(batch_result['ignored_transition_ids'])}")


    # unisce le azioni dei blocchi sovrapposti e raccoglie in un insieme tutte le transizioni
    # che sono finite in qualche azione finale
    final_actions = merge_overlapping_actions(raw_actions, transition_map)
    used_ids = {tid for a in final_actions for tid in a["transition_ids"]}

    output_file = output_dir / "action_grouping_stage2.json"
    save_json(output_file, {
        "video_id": video_id,
        "prompt_version": PROMPT_VERSION,
        "failed_blocks": failed_blocks,
        "actions": final_actions,
        "ignored_transition_ids": sorted(all_ignored - used_ids),   # ignorate in tutti i blocchi
    })

    print("\n" + "=" * 55)
    print("STAGE 2 COMPLETATO")
    print(f"Azioni finali (dopo merge): {len(final_actions)} | Blocchi falliti: {failed_blocks}")
    print(f"Output salvato in: {output_file}")