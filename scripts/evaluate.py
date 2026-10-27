"""Measure the teaching assistant: retrieval accuracy on 46 labelled questions, latency, and speech-to-text WER.

Retrieval is scored at two levels:
  section  the top-k chunks include the lecture section that answers the question
  lecture  the top-k chunks include the right lecture (video), which is what a student needs to rewatch
for two corpora: the written notes and the Whisper transcripts of the spoken lectures.

Usage:  python scripts/evaluate.py   ->   results/retrieval.json, results/retrieval.md
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.chunking import chunk_notes, chunk_transcripts  # noqa: E402
from rag.retrievers import BM25, Dense, Hybrid, Reranked  # noqa: E402
from rag.text import words_for_wer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
KS = (1, 3, 5)


def wer(ref, hyp):
    """Word error rate = (substitutions + deletions + insertions) / reference words (Levenshtein on words)."""
    r, h = words_for_wer(ref), words_for_wer(hyp)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1] / max(len(r), 1)


def correct(chunk, q, level):
    if chunk.lecture != q["lecture"]:
        return False
    return level == "lecture" or any(chunk.section.lower().startswith(s) for s in q["sections"])


def evaluate(retriever, chunks, questions):
    out = {}
    lat = []
    for level in ("section", "lecture"):
        ranks = []
        for q in questions:
            t = time.perf_counter()
            hits = retriever.search(q["q"], 10)
            lat.append(time.perf_counter() - t)
            rank = next((r for r, (i, _) in enumerate(hits, 1) if correct(chunks[i], q, level)), None)
            ranks.append(rank)
        for k in KS:
            out[f"{level}_hit@{k}"] = float(np.mean([r is not None and r <= k for r in ranks]))
        out[f"{level}_mrr"] = float(np.mean([1 / r if r else 0 for r in ranks]))
    out["latency_ms"] = float(np.median(lat) * 1000)
    return out


def build(chunks):
    texts = [c.text for c in chunks]
    bm25 = BM25(texts)
    mini = Dense(texts, "sentence-transformers/all-MiniLM-L6-v2")
    bge = Dense(texts, "BAAI/bge-small-en-v1.5")
    hybrid = Hybrid(bm25, bge)
    return {"BM25 (from scratch)": bm25, "Dense: MiniLM-L6": mini, "Dense: BGE-small": bge,
            "Hybrid: BM25 + BGE (RRF)": hybrid, "Hybrid + cross-encoder rerank": Reranked(hybrid, texts)}


def main():
    questions = json.load(open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8"))
    corpora = {"notes": chunk_notes(glob.glob(os.path.join(ROOT, "data", "lectures", "*.md")))}
    tr = sorted(glob.glob(os.path.join(ROOT, "data", "transcripts", "lecture_*.json")))
    if tr:
        corpora["transcripts"] = chunk_transcripts(tr)

    results = {"questions": len(questions), "chunks": {k: len(v) for k, v in corpora.items()}, "retrieval": {}}
    for name, chunks in corpora.items():
        results["retrieval"][name] = {}
        for rname, r in build(chunks).items():
            results["retrieval"][name][rname] = res = evaluate(r, chunks, questions)
            print(f"{name:12s} {rname:32s} section hit@1 {res['section_hit@1']:.3f} hit@3 {res['section_hit@3']:.3f} "
                  f"lecture hit@1 {res['lecture_hit@1']:.3f} mrr {res['section_mrr']:.3f} {res['latency_ms']:.1f} ms")

    if tr:
        per, ref_words = [], 0
        for path in tr:
            t = json.load(open(path, encoding="utf-8"))
            script = os.path.join(ROOT, "data", "audio", f"lecture_{t['lecture']:02d}.script.json")
            ref = json.load(open(script, encoding="utf-8"))["text"]
            n = len(words_for_wer(ref))
            per.append((wer(ref, " ".join(s["text"] for s in t["segments"])), n))
            ref_words += n
        timing = json.load(open(os.path.join(ROOT, "data", "transcripts", "_timing.json")))
        audio = sum(x["audio_s"] for x in timing)
        results["speech_to_text"] = {
            "model": "faster-whisper base.en (int8, CPU)", "audio_minutes": round(audio / 60, 1),
            "words": ref_words, "wer": sum(w * n for w, n in per) / ref_words,
            "real_time_factor": audio / sum(x["seconds"] for x in timing)}
        print("speech-to-text", results["speech_to_text"])

    json.dump(results, open(os.path.join(RES, "retrieval.json"), "w"), indent=2)
    md = [f"# Retrieval evaluation ({len(questions)} labelled questions)", ""]
    for name, rows in results["retrieval"].items():
        md += [f"## Corpus: {name} ({results['chunks'][name]} chunks)",
               "| Retriever | Section Hit@1 | Section Hit@3 | Section Hit@5 | Section MRR | Lecture Hit@1 | Lecture Hit@3 | Median latency |",
               "|---|---|---|---|---|---|---|---|"]
        for rname, r in rows.items():
            md.append(f"| {rname} | {r['section_hit@1']:.1%} | {r['section_hit@3']:.1%} | {r['section_hit@5']:.1%} | "
                      f"{r['section_mrr']:.3f} | {r['lecture_hit@1']:.1%} | {r['lecture_hit@3']:.1%} | {r['latency_ms']:.1f} ms |")
        md.append("")
    if "speech_to_text" in results:
        s = results["speech_to_text"]
        md += ["## Speech to text", f"{s['model']}: {s['audio_minutes']} min of lecture audio, {s['words']:,} words, "
               f"**WER {s['wer']:.1%}**, **{s['real_time_factor']:.1f}x faster than real time**."]
    open(os.path.join(RES, "retrieval.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
