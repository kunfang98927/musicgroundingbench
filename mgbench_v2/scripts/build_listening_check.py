"""WP8: select ~150 stratified test items and prepare audio (mp3) + items.json for the listening-check page."""
import argparse, json, os, random, subprocess, wave
from collections import defaultdict

B2 = "/scratch/kunfang/mgbench_v2_rc2"          # --qa2b overrides (rc3)
B3 = "/scratch/kunfang/mgbench_v2_rc2_3n"
NEW = "/lustre09/project/6002780/kunfang/music_grounding/mgbench_v2_out/new_transposed"      # wavs of the new true-transposed clips
QUOTA_NEW = {"Rp-U1": 4, "Rp-G1": 3, "Rp-G2": 2, "Rp-U2": 2, "R-U3": 1}
WAV2 = "/scratch/kunfang/two_bar_dataset"
WAV3 = "/lustre09/project/6002780/kunfang/music_grounding/note-level-temporal-grounding/datasets/piano-melody-3notes"
QUOTA_2B = {"R-G1": 3, "R-G2": 8, "R-G3": 4, "R-G4": 8, "R-G5": 5, "R-G6": 5, "I-G1": 3, "I-G2": 3, "I-G3": 3, "I-G4": 5, "I-G5": 3, "I-G6": 3, "I-G7": 3,
            "B-G1": 3, "B-G2": 2, "H-G1": 3, "H-G2": 4, "Rp-G1": 3, "Rp-G2": 2, "T-G1": 4,
            "R-U3": 7, "R-U4": 6, "R-U6": 3, "I-U1": 3, "I-U2": 3, "I-U3": 2, "B-U1": 2, "B-U2": 2, "B-U3": 3, "B-U4": 1,
            "H-U1": 3, "H-U2": 3, "H-U3": 1, "A-U1": 3, "A-G1": 3, "A-G2": 3, "A-U2": 3, "Rp-U1": 3, "Rp-U2": 3, "T-U1": 3, "T-U2": 2, "T-U3": 2, "T-U4": 2, "T-U5": 1}
QUOTA_3N = {"A-PITCH": 3, "O-NOTE": 3, "R-VEL": 5, "R-PITCH": 1, "P-PDIFF": 1, "P-CONTOUR": 1}


def wav_dur(p):
    with wave.open(p) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/lustre09/project/6002780/kunfang/music_grounding/mgbench_v2_out/listening_check")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--qa2b", default=B2)
    ap.add_argument("--regen", default=None, help="comma-separated 2B families to regenerate from --qa2b; with --merge-into the other families keep their existing items (ids, numbers, ratings)")
    ap.add_argument("--merge-into", default=None, help="existing items.json to keep for the families not in --regen")
    ap.add_argument("--quota-json", default=None, help='json {"family": k}: only these 2B families are picked (k items each) and every existing item is kept ("--merge-into" required)')
    ap.add_argument("--id-suffix", default="b", help="new ids are <set>-<family>-<suffix><NN> so that old ratings never attach to a regenerated item")
    ap.add_argument("--extra", default=None, help="json list of extra items (e.g. the new true-transposed clips) appended as-is")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/audio", exist_ok=True)
    rng = random.Random(a.seed)
    items, used_clips = [], set()
    regen = set(a.regen.split(",")) if a.regen else None
    quota_override = json.load(open(a.quota_json)) if a.quota_json else None
    if quota_override is not None:
        regen = set(quota_override)
    kept = []
    if a.merge_into:
        old = json.load(open(a.merge_into))
        kept = [i for i in old if quota_override is not None or not (i["set"] == "MGBench-2B" and i["family"] in regen)]
        used_clips |= {i["source"] for i in kept}
    def pick(recs, quota, tag, wavroot, only_new=False):
        by = defaultdict(list)
        for r in recs:
            by[r["v2"]["family"]].append(r)
        for fam, k in quota.items():
            def wav_of(r):
                return f"{NEW}/{r['audio_path']}" if "_tr" in r["sample_id"] else f"{wavroot}/{r['audio_path']}"
            pool = [r for r in by[fam] if r["sample_id"] not in used_clips and os.path.exists(wav_of(r)) and (("_tr" in r["sample_id"]) == only_new)]
            rng.shuffle(pool)
            if regen is not None and (tag != "2B" or fam not in regen):
                continue
            if quota_override is not None:
                k = quota_override[fam]
            for r in pool[:k]:
                used_clips.add(r["sample_id"])
                kind = "understanding" if "gold_answer" in r else "grounding"
                iid = f"{tag}-{fam}-{a.id_suffix if regen else ''}{len([i for i in items + kept if i['family'] == fam and i['id'].split('-')[-1].startswith(a.id_suffix)]) + 1:02d}" if regen else f"{tag}-{fam}-{len([i for i in items if i['family'] == fam]) + 1:02d}"
                src = wav_of(r)
                dst = f"audio/{iid}.mp3"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "32000", "-b:a", "64k", "-map_metadata", "-1", f"{a.out}/{dst}"], check=True)
                spans = r["answer_spans_sec"]
                items.append({
                    "id": iid, "set": "MGBench-" + tag, "family": fam, "concept": r.get("concept", "3N"), "kind": kind, "question": r["question"],
                    "answer": r.get("answer_text") if kind == "understanding" else (r.get("answer_text") or r.get("answer")),
                    "spans": spans, "notes": [[n["start"], n["end"], n["pitch"], n["velocity"]] for n in r["notes"]], "audio": dst, "dur": round(wav_dur(src), 3),
                    "source": r["sample_id"], "params": r["v2"].get("params", {}), "frame": r["v2"].get("frame"),
                })
    g = json.load(open(f"{a.qa2b}/grounding_qa/test.json")); u = json.load(open(f"{a.qa2b}/understanding_qa/test.json"))
    pick(g + u, QUOTA_2B, "2B", WAV2)
    if quota_override is None:
        pick(g + u, QUOTA_NEW, "2B", WAV2, only_new=True)
    if regen is None:
        pick(json.load(open(f"{B3}/llm_test.json")), QUOTA_3N, "3N", WAV3)
    if a.extra:
        items += json.load(open(a.extra))
    rng.shuffle(items)
    base = max([i["n"] for i in kept], default=0)
    for i, it in enumerate(items):
        it["n"] = base + i + 1
    items = kept + items
    json.dump(items, open(f"{a.out}/items.json", "w"))
    print(len(items), "items;", sum(1 for i in items if i["kind"] == "grounding"), "grounding;", round(sum(os.path.getsize(f"{a.out}/{i['audio']}") for i in items) / 1e6, 1), "MB audio")


if __name__ == "__main__":
    main()
