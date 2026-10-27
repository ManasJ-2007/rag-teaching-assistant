"""Text normalisation shared by chunking, BM25 and the speech pipeline."""
import re

GREEK = {"γ": "gamma", "ε": "epsilon", "θ": "theta", "π": "pi", "μ": "mu", "τ": "tau", "δ": "delta",
         "λ": "lambda", "β": "beta", "α": "alpha", "σ": "sigma", "ρ": "rho", "φ": "phi"}
STOPWORDS = set("""a an the and or of to in on for with by is are was were be been it its this that these those as at from
into than then so such can do does did how what why which who when where whose if not no but also we you i our your
their there here each every all any more most only same other one two""".split())


def clean_markdown(md):
    """Markdown/LaTeX -> plain readable text (keeps the words, drops the markup)."""
    s = re.sub(r"```.*?```", " ", md, flags=re.S)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\\(mathbb|mathcal|mathrm|text|operatorname)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    for g, word in GREEK.items():
        s = s.replace(g, f" {word} ")
    s = re.sub(r"[`*_$#{}^|\\]", " ", s)
    s = re.sub(r"-{3,}", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def stem(w):
    """Tiny suffix stripper (good enough for BM25 on a small course corpus)."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def words_for_wer(text):
    return re.findall(r"[a-z0-9]+", text.lower())
