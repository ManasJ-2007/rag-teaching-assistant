"""Evaluate the answer-generation step with a local LLM (Ollama).

For each of the 46 labelled questions: retrieve the top-3 chunks (hybrid + cross-encoder re-ranking),
send them to the LLM with the grounded prompt in rag/generate.py, and record
  - whether the answer cites the correct lecture ("Lecture N" with the labelled N)
  - whether the model said the excerpts do not contain the answer
  - latency per answer
Every answer is saved so it can be read and checked by hand.

Usage:  ollama pull llama3.2
        python scripts/evaluate_answers.py [--corpus notes|transcripts]   ->   results/answers_<corpus>.json / .md
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.chunking import chunk_notes, chunk_transcripts  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.retrievers import BM25, Dense, Hybrid, Reranked  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
NOT_FOUND = re.compile(r"(do(es)? not|don't|doesn't) (contain|mention|cover|provide)|not (in|covered in) the (lecture|excerpt)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("notes", "transcripts"), default="notes")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    if args.corpus == "transcripts":
        chunks = chunk_transcripts(glob.glob(os.path.join(ROOT, "data", "transcripts", "lecture_*.json")))
    else:
        chunks = chunk_notes(glob.glob(os.path.join(ROOT, "data", "lectures", "*.md")))
    texts = [c.text for c in chunks]
    retriever = Reranked(Hybrid(BM25(texts), Dense(texts, "BAAI/bge-small-en-v1.5")), texts)
    questions = json.load(open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8"))
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    rows = []
    for n, q in enumerate(questions, 1):
        hits = [chunks[i] for i, _ in retriever.search(q["q"], args.k)]
        t = time.perf_counter()
        a = answer(q["q"], hits, "ollama")
        dt = time.perf_counter() - t
        cited = sorted({int(x) for x in re.findall(r"Lecture\s*(\d+)", a)})
        rows.append({"q": q["q"], "gold_lecture": q["lecture"], "retrieved_lectures": [c.lecture for c in hits],
                     "cited_lectures": cited, "cites_gold": q["lecture"] in cited,
                     "retrieved_gold": q["lecture"] in [c.lecture for c in hits],
                     "said_not_found": bool(NOT_FOUND.search(a)), "seconds": round(dt, 2), "answer": a})
        print(f"{n:2d}/{len(questions)}  cites gold: {rows[-1]['cites_gold']!s:5}  {dt:5.1f}s  {q['q'][:60]}")

    N = len(rows)
    summary = {
        "model": model, "corpus": args.corpus, "questions": N, "k": args.k,
        "answers_citing_correct_lecture": sum(r["cites_gold"] for r in rows) / N,
        "answers_citing_any_lecture": sum(bool(r["cited_lectures"]) for r in rows) / N,
        "said_not_found": sum(r["said_not_found"] for r in rows) / N,
        "gold_lecture_in_retrieved": sum(r["retrieved_gold"] for r in rows) / N,
        "median_seconds": sorted(r["seconds"] for r in rows)[N // 2],
    }
    os.makedirs(RES, exist_ok=True)
    json.dump({"summary": summary, "answers": rows}, open(os.path.join(RES, f"answers_{args.corpus}.json"), "w",
                                                           encoding="utf-8"), indent=1, ensure_ascii=False)
    md = [f"# Generated answers ({model}, {args.corpus}, top-{args.k} chunks)", "",
          f"- Answers citing the correct lecture: **{summary['answers_citing_correct_lecture']:.1%}**",
          f"- Answers citing any lecture: {summary['answers_citing_any_lecture']:.1%}",
          f"- Correct lecture among the retrieved chunks: {summary['gold_lecture_in_retrieved']:.1%}",
          f"- Model said the excerpts do not contain the answer: {summary['said_not_found']:.1%}",
          f"- Median time per answer: {summary['median_seconds']:.1f} s (CPU)", ""]
    for r in rows:
        md += [f"### {r['q']}", f"*Labelled lecture {r['gold_lecture']} · cited {r['cited_lectures'] or 'none'}*", "",
               r["answer"].strip(), ""]
    open(os.path.join(RES, f"answers_{args.corpus}.md"), "w", encoding="utf-8").write("\n".join(md))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
