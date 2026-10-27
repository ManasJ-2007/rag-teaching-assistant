"""Answer generation: build a grounded prompt from retrieved chunks and send it to an LLM.

Backends
    extractive  (default, offline) returns the most relevant sentences with their sources
    ollama      a local Ollama server, e.g. `ollama pull llama3.2`  (env OLLAMA_MODEL)
    openai      any OpenAI-compatible API (env OPENAI_API_KEY, optional OPENAI_BASE_URL, OPENAI_MODEL)
"""
import json
import os
import re
import urllib.request

import numpy as np

SYSTEM = ("You are a teaching assistant for a reinforcement learning course. Answer the student's question "
          "using ONLY the lecture excerpts provided. Cite the lecture (and timestamp if given) you used. "
          "If the excerpts do not contain the answer, say so and suggest which lecture to revisit.")


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{i + 1}] {c.location()}\n{c.text}" for i, c in enumerate(chunks))
    return f"Lecture excerpts:\n{context}\n\nStudent question: {question}\nAnswer:"


def extractive_answer(question, chunks, scorer=None, n_passages=2):
    """Pick the sentences that best answer the question and return each with the sentence that follows it.

    scorer(question, sentences) -> relevance scores (e.g. a cross-encoder); falls back to word overlap."""
    sents = []
    for c in chunks:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", c.text) if s.strip()]
        sents += [(c, s, parts[j + 1] if j + 1 < len(parts) else "") for j, s in enumerate(parts)
                  if len(s.split()) > 5]
    if not sents:
        return "I could not find this in the lectures."
    if scorer is not None:
        scores = np.asarray(scorer(question, [s for _, s, _ in sents]), dtype=float)
    else:
        q = set(question.lower().split())
        scores = np.array([len(q & set(s.lower().split())) for _, s, _ in sents], dtype=float)
    lines, used = [], set()
    for i in np.argsort(-scores):
        c, s, nxt = sents[i]
        if s in used:
            continue
        used.update({s, nxt})
        lines.append(f"- {s} {nxt}".rstrip() + f"\n  ({c.location()})")
        if len(lines) == n_passages:
            break
    return "From the lectures:\n" + "\n".join(lines)


def _post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
                                                                                 **headers})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def answer(question, chunks, backend="extractive", scorer=None):
    if backend == "extractive":
        return extractive_answer(question, chunks, scorer)
    prompt = build_prompt(question, chunks)
    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = _post(f"{host}/api/chat", {"model": os.environ.get("OLLAMA_MODEL", "llama3.2"), "stream": False,
                                       "messages": [{"role": "system", "content": SYSTEM},
                                                    {"role": "user", "content": prompt}]}, {})
        return r["message"]["content"]
    if backend == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        r = _post(f"{base}/chat/completions",
                  {"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0.2,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
                  {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return r["choices"][0]["message"]["content"]
    raise ValueError(f"unknown backend {backend!r}")
