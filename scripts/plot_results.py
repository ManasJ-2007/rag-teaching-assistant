"""Bar chart of results/retrieval.json -> results/retrieval.png"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    r = json.load(open(os.path.join(RES, "retrieval.json")))["retrieval"]
    names = list(r["notes"])
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (corpus, title) in zip(axes, [("notes", "Written notes"), ("transcripts", "Whisper transcripts of lectures")]):
        for off, (key, label, color) in zip((-0.2, 0.2), [("section_hit@1", "Hit@1", "#1a73e8"),
                                                          ("section_hit@3", "Hit@3", "#8ab4f8")]):
            vals = [r[corpus][n][key] * 100 for n in names]
            ax.bar(x + off, vals, 0.4, label=label, color=color)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x, [n.replace(": ", ":\n").replace(" + ", " +\n").replace(" (", "\n(") for n in names], fontsize=7)
        ax.set_title(f"{title}: correct section retrieved")
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of 46 questions")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "retrieval.png"), dpi=130)


if __name__ == "__main__":
    main()
