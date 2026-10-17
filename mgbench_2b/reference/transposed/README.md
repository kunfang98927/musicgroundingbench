# Transposed MusicGroundingBench-2B clips

200 clips (train 120, val 20, test 60; ids `<split>_tr0001` ...) in which bar 1 is taken from a
generated clip of the same split and bar 2 repeats it one bar later, transposed by 1 to 7
semitones. `manifest.json` lists the source clip and the transposition of each clip;
`<split>/meta/<id>.json` holds the timing frame (`basic_info`) and the same information.

These files are inputs of the question-answer build (`build_qa_2b.py --transposed`). They were
made with `mgbench_qa/scripts/make_transposed_midi.py` under an earlier revision of the clip
checks, so they are kept here instead of being regenerated (see REPRODUCING.md).
