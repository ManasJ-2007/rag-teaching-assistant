"""Speech-to-text for lecture recordings with faster-whisper (Whisper, CTranslate2 backend).

Input: a folder of .wav/.mp3/.mp4/.mkv files named like lecture_05.wav or 05_dqn.mp4 (video audio is
decoded by faster-whisper via PyAV, no separate ffmpeg step needed).
Output: data/transcripts/lecture_XX.json with timed segments, ready for rag.chunking.chunk_transcripts.

Usage:  python -m rag.transcribe data/audio --model base.en
"""
import argparse
import glob
import json
import os
import time

from .chunking import lecture_number

EXTS = (".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".webm")


def load_audio(path):
    """WAV -> 16 kHz mono float32 array (read directly); other formats are decoded by faster-whisper."""
    if not path.lower().endswith(".wav"):
        return path
    import wave
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly
    with wave.open(path, "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width]).astype(np.float32)
    x = (x - 128) / 128 if width == 1 else x / float(2 ** (8 * width - 1))
    x = x.reshape(-1, ch).mean(axis=1)
    g = gcd(16000, sr)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def transcribe_folder(folder, out_dir, model_size="base.en", device="cpu"):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type="int8")
    os.makedirs(out_dir, exist_ok=True)
    stats = []
    for path in sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(EXTS)):
        n = lecture_number(path.replace("lecture_", ""))
        t = time.perf_counter()
        segments, info = model.transcribe(load_audio(path), beam_size=5, vad_filter=True)
        segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
        elapsed = time.perf_counter() - t
        meta_path = os.path.splitext(path)[0] + ".script.json"
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        out = {"lecture": n, "title": meta.get("title", os.path.basename(path)), "source": os.path.basename(path),
               "duration": round(info.duration, 1), "segments": segs, "sections": meta.get("sections", [])}
        json.dump(out, open(os.path.join(out_dir, f"lecture_{n:02d}.json"), "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        stats.append({"lecture": n, "audio_s": info.duration, "seconds": elapsed})
        print(f"lecture {n:02d}: {info.duration / 60:.1f} min audio in {elapsed:.0f}s "
              f"({info.duration / elapsed:.1f}x real time), {len(segs)} segments")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=os.path.join("data", "transcripts"))
    ap.add_argument("--model", default="base.en")
    args = ap.parse_args()
    stats = transcribe_folder(args.folder, args.out, args.model)
    json.dump(stats, open(os.path.join(args.out, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
