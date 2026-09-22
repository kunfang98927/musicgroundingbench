"""MERT-v1-95M features (mean over the 13 hidden states) for a list of wavs, same recipe as datasets/scripts/extract_mert.py
(mono mean of the channels, no resampling on load, torchaudio Resample 44.1k -> 24k, Wav2Vec2FeatureExtractor, float32 [T, 768]).
The loader is soundfile instead of librosa (librosa's `decorator` dependency is missing in ground-env); the arithmetic is identical.

python -m mgbench_v2.scripts.extract_mert_v2 --root DIR --out FILE.h5      # keys: <split>/wav/<id>.wav for every DIR/<split>/wav/*.wav
python -m mgbench_v2.scripts.extract_mert_v2 --check --n 5                  # re-extract released clips and compare with the stored h5
"""
import argparse, glob, os
import h5py, numpy as np, soundfile as sf, torch, torchaudio.transforms as T
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

MERT = "/scratch/kunfang/MERT-v1-95M"
H5_2B = "/scratch/kunfang/two_bar_dataset/mert_features/two_bar_dataset_mean-13.h5"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MERT, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MERT, trust_remote_code=True)
    if not hasattr(cfg, "conv_pos_batch_norm"):
        cfg.conv_pos_batch_norm = False
    model = AutoModel.from_pretrained(MERT, config=cfg, trust_remote_code=True).to(DEV).eval()
    return proc, model


def extract(path, proc, model):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)                                             # librosa.load(mono=True) == channel mean
    a = torch.from_numpy(x).float()
    if sr != proc.sampling_rate:
        a = T.Resample(sr, proc.sampling_rate)(a.unsqueeze(0)).squeeze(0)
    inp = proc(a, sampling_rate=proc.sampling_rate, return_tensors="pt").to(DEV)
    with torch.no_grad():
        hs = model(**inp, output_hidden_states=True).hidden_states
    return torch.stack(hs).squeeze(1).mean(dim=0).cpu().numpy().astype("float32")     # [T, 768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--check", action="store_true"); ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    proc, model = load_model()
    if a.check:
        ref = h5py.File(H5_2B, "r")
        files = sorted(glob.glob("/scratch/kunfang/two_bar_dataset/test/wav/*.wav"))[:a.n]
        for f in files:
            key = "test/wav/" + os.path.basename(f)
            new, old = extract(f, proc, model), np.array(ref[key])
            m = min(len(new), len(old))
            cos = float(np.mean(np.sum(new[:m] * old[:m], 1) / (np.linalg.norm(new[:m], axis=1) * np.linalg.norm(old[:m], axis=1))))
            print(key, "shape new/old", new.shape, old.shape, "max|diff|", float(np.abs(new[:m] - old[:m]).max()), "mean|diff|", float(np.abs(new[:m] - old[:m]).mean()), "mean cos", round(cos, 6), "old rms", float(np.sqrt((old ** 2).mean())))
        return
    files = sorted(glob.glob(f"{a.root}/*/wav/*.wav"))
    with h5py.File(a.out, "a") as h:
        for f in files:
            key = "/".join(f.split("/")[-3:])
            if key in h:
                continue
            h.create_dataset(key, data=extract(f, proc, model), dtype="float32")
    print("wrote", len(files), "entries ->", a.out)


if __name__ == "__main__":
    main()
