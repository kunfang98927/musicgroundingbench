"""Step 3N-2: write metadata.csv (one row per clip: audio/midi/json paths,
first note onset, last note offset, split). Only reads MIDI-derived JSON, so
it works with no rendered audio at all - the audio_path column is just the
path the .wav *would* live at.

Usage:
    python scripts/02_build_metadata.py --data_dir piano-melody-3notes
"""

import argparse
import csv
import json
from pathlib import Path


def build_metadata(data_dir: Path) -> None:
    metadata_path = data_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "midi_path", "json_path", "first_onset", "last_offset", "split"])

        for split in ["train", "val", "test"]:
            midi_dir = data_dir / split / "midi"
            json_dir = data_dir / split / "meta"
            audio_dir = data_dir / split / "wav"

            for midi_path in sorted(midi_dir.glob("*.mid")):
                base_name = midi_path.stem
                json_path = json_dir / f"{base_name}.json"

                with open(json_path, "r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                events = meta.get("events", [])
                first_onset = min((n["onset"] for n in events), default=0.0)
                last_offset = max((n["offset"] for n in events), default=0.0)

                writer.writerow([
                    (audio_dir / f"{base_name}.wav").relative_to(data_dir),
                    midi_path.relative_to(data_dir),
                    json_path.relative_to(data_dir),
                    first_onset, last_offset, split,
                ])

    print(f"[DONE] wrote {metadata_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=str, required=True)
    args = ap.parse_args()
    build_metadata(Path(args.data_dir))


if __name__ == "__main__":
    main()
