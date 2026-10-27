# Retrieval evaluation (46 labelled questions)

## Corpus: notes (47 chunks)
| Retriever | Section Hit@1 | Section Hit@3 | Section Hit@5 | Section MRR | Lecture Hit@1 | Lecture Hit@3 | Median latency |
|---|---|---|---|---|---|---|---|
| BM25 (from scratch) | 78.3% | 91.3% | 95.7% | 0.855 | 82.6% | 95.7% | 0.1 ms |
| Dense: MiniLM-L6 | 73.9% | 87.0% | 93.5% | 0.819 | 87.0% | 93.5% | 5.9 ms |
| Dense: BGE-small | 78.3% | 91.3% | 93.5% | 0.850 | 87.0% | 97.8% | 11.7 ms |
| Hybrid: BM25 + BGE (RRF) | 87.0% | 95.7% | 95.7% | 0.909 | 95.7% | 97.8% | 11.9 ms |
| Hybrid + cross-encoder rerank | 91.3% | 95.7% | 95.7% | 0.938 | 95.7% | 97.8% | 491.0 ms |

## Corpus: transcripts (53 chunks)
| Retriever | Section Hit@1 | Section Hit@3 | Section Hit@5 | Section MRR | Lecture Hit@1 | Lecture Hit@3 | Median latency |
|---|---|---|---|---|---|---|---|
| BM25 (from scratch) | 58.7% | 87.0% | 95.7% | 0.733 | 73.9% | 95.7% | 0.1 ms |
| Dense: MiniLM-L6 | 63.0% | 87.0% | 87.0% | 0.737 | 80.4% | 95.7% | 6.3 ms |
| Dense: BGE-small | 63.0% | 82.6% | 84.8% | 0.723 | 78.3% | 89.1% | 12.3 ms |
| Hybrid: BM25 + BGE (RRF) | 60.9% | 91.3% | 93.5% | 0.754 | 80.4% | 95.7% | 12.2 ms |
| Hybrid + cross-encoder rerank | 76.1% | 89.1% | 95.7% | 0.839 | 91.3% | 95.7% | 395.6 ms |

## Speech to text
faster-whisper base.en (int8, CPU): 28.9 min of lecture audio, 3,896 words, **WER 25.6%**, **3.4x faster than real time**.
