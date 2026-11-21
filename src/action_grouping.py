import hashlib
import json
import re
import requests
from pathlib import Path

# ============================================================
# CONFIGURAZIONE
# ============================================================
BATCH_SIZE = 20
BATCH_STRIDE = 15  # Overlap di 5 transizioni (2.5 secondi)
SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"
PROMPT_VERSION = "stage2_lines_en_v3"
RETRY_TEMPERATURES = [0.1, 0.4, 0.7]

LINE_RE = re.compile(r"^\s*(\d+)\s*[-–to]+\s*(\d+)\s*\|\s*(.+?)\s*$", re.IGNORECASE)


def compute_batch_hash(transitions):
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "transitions": [
                (t["transition_id"], t["description"], t["status"])
                for t in transitions
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def split_consecutive(ids):
    if not ids:
        return []
    chunks, current = [], [ids[0]]
    for prev, cur in zip(ids, ids[1:]):
        if cur == prev + 1:
            current.append(cur)
        else:
            chunks.append(current)
            current = [cur]
    chunks.append(current)
    return chunks


def group_actions(transitions, temperature=0.1):
    lines = [
        f"{t['transition_id']} | {t['status']} | {t['description']}"
        for t in transitions
    ]
    transitions_text = "\n".join(lines)

    prompt = f"""Group the following egocentric video transitions into atomic physical actions.
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

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 384,
    }

    response = requests.post(SERVER_URL, json=payload, timeout=600)
    if not response.ok:
        print("\nERRORE HTTP:", response.status_code)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def parse_and_build_actions(raw_text, transitions):
    transition_map = {t["transition_id"]: t for t in transitions}
    expected_ids = set(transition_map.keys())
    min_id, max_id = min(expected_ids), max(expected_ids)

    if raw_text.strip().upper() == "NONE":
        return {"actions": [], "ignored_transition_ids": sorted(expected_ids)}

    seen_in_groups = set()
    groups = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue

        start_id, end_id = int(match.group(1)), int(match.group(2))
        label = match.group(3).strip().strip("\"'")

        if start_id > end_id:
            start_id, end_id = end_id, start_id
        if not label or "<short" in label.lower():
            continue

        # Filtra gli ID effettivamente presenti nel blocco e non già assegnati
        valid_ids = [
            tid for tid in range(max(start_id, min_id), min(end_id, max_id) + 1)
            if tid in expected_ids and tid not in seen_in_groups
        ]

        if valid_ids:
            seen_in_groups.update(valid_ids)
            for chunk in split_consecutive(valid_ids):
                groups.append({"transition_ids": chunk, "action_label": label})

    if not groups and any(t["status"] != "none" for t in transitions):
        raise ValueError("Nessuna riga valida '<start>-<end> | <action>' trovata nella risposta.")

    groups.sort(key=lambda g: g["transition_ids"][0])
    ignored_ids = sorted(expected_ids - seen_in_groups)

    actions = []
    for action_id, group in enumerate(groups):
        ids = group["transition_ids"]
        actions.append({
            "action_id": action_id,
            "action_label": group["action_label"],
            "start_time": transition_map[ids[0]]["start_time"],
            "end_time": transition_map[ids[-1]]["end_time"],
            "transition_ids": ids,
        })

    return {"actions": actions, "ignored_transition_ids": ignored_ids}


def merge_overlapping_actions(all_actions, transition_map):
    if not all_actions:
        return []

    all_actions.sort(key=lambda a: (a["transition_ids"][0], -len(a["transition_ids"])))
    merged = []

    for act in all_actions:
        if not merged:
            merged.append(act)
            continue

        prev = merged[-1]
        prev_set = set(prev["transition_ids"])
        cur_set = set(act["transition_ids"])

        if prev_set & cur_set:
            combined_ids = sorted(prev_set | cur_set)
            label = prev["action_label"] if len(prev_set) >= len(cur_set) else act["action_label"]
            chunks = split_consecutive(combined_ids)
            merged.pop()
            for chunk in chunks:
                merged.append({
                    "action_label": label,
                    "start_time": transition_map[chunk[0]]["start_time"],
                    "end_time": transition_map[chunk[-1]]["end_time"],
                    "transition_ids": chunk,
                })
        else:
            merged.append(act)

    for idx, act in enumerate(merged):
        act["action_id"] = idx
    return merged


def run(video_id: str, force: bool = False):
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "outputs" / video_id / "stage1" / "transitions_cleaned.json"
    output_dir = project_root / "outputs" / video_id / "stage2"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "action_grouping_stage2.json"
    raw_file = output_dir / "action_grouping_stage2_raw.txt"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"File JSON di input non trovato: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        transitions = json.load(f)

    transition_map = {t["transition_id"]: t for t in transitions}

    batches = []
    start = 0
    while start < len(transitions):
        batch = transitions[start : start + BATCH_SIZE]
        batches.append(batch)
        if start + BATCH_SIZE >= len(transitions):
            break
        start += BATCH_STRIDE

    print(f"Transizioni caricate: {len(transitions)} | Blocchi: {len(batches)} (size={BATCH_SIZE}, stride={BATCH_STRIDE})")

    raw_actions = []
    all_ignored = set()
    failed_blocks = []

    with open(raw_file, "a", encoding="utf-8") as raw_file_handle:
        for batch_index, batch in enumerate(batches, start=1):
            checkpoint_file = checkpoint_dir / f"block_{batch_index:02d}.json"
            batch_hash = compute_batch_hash(batch)

            print("\n" + "=" * 55)
            print(f"STAGE 2 - BLOCCO {batch_index}/{len(batches)} (Transizioni: {len(batch)})")

            loaded = False
            if not force and checkpoint_file.exists():
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                if ckpt.get("batch_hash") == batch_hash:
                    print("Checkpoint valido trovato: recupero risultato.")
                    batch_result = ckpt["result"]
                    loaded = True
                else:
                    print("Checkpoint obsoleto: ricalcolo.")

            if not loaded:
                success = False
                for attempt, temp in enumerate(RETRY_TEMPERATURES, start=1):
                    try:
                        print(f"Elaborazione con Qwen (Tentativo {attempt}/{len(RETRY_TEMPERATURES)}, temp={temp})...")
                        raw_result = group_actions(batch, temperature=temp)
                        batch_result = parse_and_build_actions(raw_result, batch)

                        raw_file_handle.write(f"\n\n===== BLOCCO {batch_index} (Try {attempt}, T={temp}) =====\n{raw_result}")
                        raw_file_handle.flush()

                        with open(checkpoint_file, "w", encoding="utf-8") as f:
                            json.dump(
                                {"batch_hash": batch_hash, "prompt_version": PROMPT_VERSION, "result": batch_result},
                                f,
                                indent=4,
                                ensure_ascii=False,
                            )
                        success = True
                        break
                    except ValueError as e:
                        print(f"  [Avviso] Errore parsing: {e}")

                if not success:
                    print(f"  [ERRORE] Blocco {batch_index} fallito dopo {len(RETRY_TEMPERATURES)} tentativi.")
                    failed_blocks.append(batch_index)
                    continue

            raw_actions.extend(batch_result["actions"])
            all_ignored.update(batch_result["ignored_transition_ids"])
            print(f"Azioni trovate: {len(batch_result['actions'])} | Ignorate: {len(batch_result['ignored_transition_ids'])}")

    final_actions = merge_overlapping_actions(raw_actions, transition_map)
    used_ids = {tid for a in final_actions for tid in a["transition_ids"]}
    final_ignored = sorted(all_ignored - used_ids)

    final_result = {
        "video_id": video_id,
        "prompt_version": PROMPT_VERSION,
        "failed_blocks": failed_blocks,
        "actions": final_actions,
        "ignored_transition_ids": final_ignored,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("STAGE 2 COMPLETATO")
    print(f"Azioni finali (dopo merge): {len(final_actions)} | Blocchi falliti: {failed_blocks}")
    print(f"Output salvato in: {output_file}")