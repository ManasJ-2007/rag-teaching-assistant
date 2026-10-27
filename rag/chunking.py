"""Split lecture notes (markdown) or lecture transcripts (timed segments) into retrievable chunks.

Every chunk keeps where it came from: lecture number, lecture title, section heading, and for
transcripts the start/end time in the video, so the assistant can say "Lecture 5, 03:12".
"""
import json
import os
import re
from dataclasses import asdict, dataclass

from .text import clean_markdown


@dataclass
class Chunk:
    id: int
    lecture: int
    title: str
    section: str
    text: str
    start: float = None
    end: float = None

    def location(self):
        where = f"Lecture {self.lecture}: {self.title}"
        if self.start is not None:
            where += f" @ {fmt_time(self.start)}-{fmt_time(self.end)}"
        return f"{where} | {self.section}"


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def lecture_number(path):
    m = re.match(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def sections(md):
    """[(heading, body_markdown)] split on #, ## and ### headings."""
    out, heading, buf = [], None, []
    for line in md.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)", line)
        if m:
            if heading is not None or buf:
                out.append((heading or "", "\n".join(buf)))
            heading, buf = clean_markdown(m.group(1)), []
        else:
            buf.append(line)
    out.append((heading or "", "\n".join(buf)))
    return [(h, b) for h, b in out if clean_markdown(b)]


def windows(words, size, overlap):
    step = max(size - overlap, 1)
    for i in range(0, max(len(words) - overlap, 1), step):
        yield words[i:i + size]


def chunk_notes(paths, size=120, overlap=30):
    chunks = []
    for path in sorted(paths):
        md = open(path, encoding="utf-8").read()
        title = clean_markdown(md.splitlines()[0].lstrip("# "))
        for heading, body in sections(md):
            for w in windows(clean_markdown(body).split(), size, overlap):
                text = f"{heading}. " + " ".join(w)
                chunks.append(Chunk(len(chunks), lecture_number(path), title, heading, text))
    return chunks


def chunk_transcripts(paths, window_s=45.0, overlap_s=10.0):
    """Transcript JSON: {"lecture", "title", "segments": [{"start","end","text"}], "sections": [...]}."""
    chunks = []
    for path in sorted(paths):
        t = json.load(open(path, encoding="utf-8"))
        segs, bounds = t["segments"], t.get("sections", [])
        start_t = 0.0
        while segs and start_t < segs[-1]["end"]:
            inside = [s for s in segs if s["start"] >= start_t and s["start"] < start_t + window_s]
            if inside:
                a, b = inside[0]["start"], inside[-1]["end"]
                chunks.append(Chunk(len(chunks), t["lecture"], t["title"], section_at(bounds, a, b),
                                    " ".join(s["text"].strip() for s in inside), a, b))
            start_t += window_s - overlap_s
    return chunks


def section_at(bounds, a, b):
    """Section whose time range overlaps [a, b] the most (known when audio was generated per section)."""
    best, best_overlap = "", 0.0
    for sec in bounds:
        ov = min(b, sec["end"]) - max(a, sec["start"])
        if ov > best_overlap:
            best, best_overlap = sec["heading"], ov
    return best


def save_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=1, ensure_ascii=False)


def load_chunks(path):
    return [Chunk(**c) for c in json.load(open(path, encoding="utf-8"))]
