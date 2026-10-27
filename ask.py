"""Ask the teaching assistant a question.

Usage:
    python ask.py "Why does DQN need a target network?"
    python ask.py --corpus transcripts "What is the TD error?"        # answer from the lecture recordings
    python ask.py --backend ollama "..."                              # generate with a local LLM
    python ask.py                                                     # interactive mode
"""
import argparse
import glob
import os

from rag.chunking import chunk_notes, chunk_transcripts
from rag.generate import answer
from rag.retrievers import BM25, Dense, Hybrid, Reranked

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(corpus):
    if corpus == "transcripts":
        paths = glob.glob(os.path.join(ROOT, "data", "transcripts", "lecture_*.json"))
        if not paths:
            raise SystemExit("No transcripts yet: run `python -m rag.transcribe data/audio` first.")
        return chunk_transcripts(paths)
    return chunk_notes(glob.glob(os.path.join(ROOT, "data", "lectures", "*.md")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--corpus", choices=("notes", "transcripts"), default="notes")
    ap.add_argument("--backend", choices=("extractive", "ollama", "openai"), default="extractive")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    chunks = load(args.corpus)
    texts = [c.text for c in chunks]
    dense = Dense(texts, "BAAI/bge-small-en-v1.5")
    retriever = Reranked(Hybrid(BM25(texts), dense), texts)

    def scorer(q, sentences):
        return retriever.model.predict([(q, s) for s in sentences], show_progress_bar=False)

    def run(q):
        hits = [chunks[i] for i, _ in retriever.search(q, args.k)]
        print("\nSources:")
        for c in hits:
            print("  -", c.location())
        print("\n" + answer(q, hits, args.backend, scorer) + "\n")

    if args.question:
        run(args.question)
    else:
        while (q := input("Ask a question (blank to quit): ").strip()):
            run(q)


if __name__ == "__main__":
    main()
