"""Turn the lecture notes into spoken "lecture recordings" (Windows text-to-speech), one WAV per lecture.

This stands in for real lecture videos so the speech-to-text half of the pipeline can be run and
measured end to end: the spoken script is known exactly, so transcription word error rate (WER) and
the time range of every section are known too. With real videos, skip this and run
`python -m rag.transcribe path/to/videos`.

Writes data/audio/lecture_XX.wav, data/audio/lecture_XX.script.json (spoken text + section times).
Usage:  python scripts/make_lecture_audio.py        (Windows only: uses System.Speech)
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.chunking import lecture_number, sections  # noqa: E402
from rag.text import clean_markdown  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUDIO = os.path.join(ROOT, "data", "audio")

PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 1
foreach ($line in Get-Content -LiteralPath $args[0] -Encoding UTF8) {
    $txt, $wav = $line -split "`t"
    $s.SetOutputToWaveFile($wav)
    $s.Speak((Get-Content -LiteralPath $txt -Raw -Encoding UTF8))
}
$s.Dispose()
"""


def main():
    os.makedirs(AUDIO, exist_ok=True)
    tmp = tempfile.mkdtemp()
    jobs, plan = [], {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "lectures", "*.md"))):
        n = lecture_number(path)
        md = open(path, encoding="utf-8").read()
        title = clean_markdown(md.splitlines()[0].lstrip("# "))
        plan[n] = {"title": title, "parts": []}
        for j, (heading, body) in enumerate(sections(md)):
            text = (f"{heading}. " if heading else "") + clean_markdown(body)
            txt, wav = os.path.join(tmp, f"{n}_{j}.txt"), os.path.join(tmp, f"{n}_{j}.wav")
            open(txt, "w", encoding="utf-8").write(text)
            jobs.append(f"{txt}\t{wav}")
            plan[n]["parts"].append({"heading": heading, "text": text, "wav": wav})
    job_file, ps_file = os.path.join(tmp, "jobs.txt"), os.path.join(tmp, "tts.ps1")
    open(job_file, "w", encoding="utf-8").write("\n".join(jobs))
    open(ps_file, "w", encoding="utf-8").write(PS)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_file, job_file], check=True)

    total = 0.0
    for n, lec in plan.items():
        out = os.path.join(AUDIO, f"lecture_{n:02d}.wav")
        t, bounds = 0.0, []
        with wave.open(out, "wb") as w:
            for k, part in enumerate(lec["parts"]):
                with wave.open(part["wav"], "rb") as r:
                    if k == 0:
                        w.setparams(r.getparams())
                    frames = r.readframes(r.getnframes())
                    dur = r.getnframes() / r.getframerate()
                w.writeframes(frames)
                bounds.append({"heading": part["heading"], "start": round(t, 2), "end": round(t + dur, 2)})
                t += dur
        script = {"lecture": n, "title": lec["title"], "duration": round(t, 1), "sections": bounds,
                  "text": " ".join(p["text"] for p in lec["parts"])}
        json.dump(script, open(out.replace(".wav", ".script.json"), "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        total += t
        print(f"lecture {n:02d}: {t / 60:5.1f} min, {len(bounds)} sections")
    print(f"total audio: {total / 60:.1f} min")


if __name__ == "__main__":
    main()
