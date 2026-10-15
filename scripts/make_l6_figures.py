"""Figures for the L6 study notes (ethics, data, reproducibility).

Run:  python make_l6_figures.py   (writes l6_*.pdf into ../figures; needs numpy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, PURPLE, RED, GREEN = "#0037a5", "#690087", "#af0000", "#006928"


def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="k", fs=5.8, color="k", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=0.6, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arr(ax, x0, y0, x1, y1, col="k", text=None, dy=0.08):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=5.3, color=col)


# ---- 1. pseudonymised vs anonymised, and how hard anonymisation is ---------------
fig, ax = plt.subplots(figsize=(W, 2.0))
box(ax, 0.0, 2.0, 2.2, 0.9, "name: J. Jansen\nborn: 1961-04-02\nscan.nii", "#fde0e0", RED)
box(ax, 3.0, 2.0, 2.2, 0.9, "ID: 4711\nscan.nii", "#fff2cc", "#b08000")
box(ax, 3.0, 1.15, 2.2, 0.55, "key file: 4711 = J. Jansen\n(kept, locked away)", "#fff9e6", "#b08000", fs=5.2)
box(ax, 6.0, 2.0, 2.2, 0.9, "ID: 4711\nscan.nii", "#e9f5e9", GREEN)
box(ax, 6.0, 1.15, 2.2, 0.55, "key file deleted", "#ffffff", "#999", fs=5.2, color="#777", ls="--")
ax.plot([6.15, 8.05], [1.2, 1.65], color=RED, lw=0.8)
arr(ax, 2.25, 2.45, 2.95, 2.45)
arr(ax, 5.25, 2.45, 5.95, 2.45)
for x, t, c in [(1.1, "original", RED), (4.1, "pseudo-anonymised", "#b08000"), (7.1, "anonymised", GREEN)]:
    ax.text(x, 3.05, t, ha="center", va="bottom", fontsize=6.5, color=c, weight="bold")
# spectrum of difficulty
ax.annotate("", (8.2, 0.45), (0.0, 0.45), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8))
for x, t in [(0.5, "biopsy\nslide"), (2.5, "head MRI\n(deface!)"), (4.9, "whole-body\nCT"), (7.4, "iris\nphoto")]:
    ax.plot(x, 0.45, "o", color=PURPLE, ms=3)
    ax.text(x, 0.33, t, ha="center", va="top", fontsize=5.6)
ax.text(0.0, 0.62, "easy to anonymise", fontsize=5.6, color=GREEN, ha="left")
ax.text(8.2, 0.62, "impossible (the body is the ID)", fontsize=5.6, color=RED, ha="right")
ax.set_xlim(-0.1, 8.35); ax.set_ylim(-0.25, 3.3); ax.axis("off")
fig.savefig(OUT / "l6_anonymisation.pdf"); plt.close(fig)

# ---- 2. federated learning and heterogeneous sites --------------------------------
fig = plt.figure(figsize=(W, 1.75))
ax = fig.add_axes([0.0, 0.0, 0.58, 1.0])
box(ax, 1.4, 2.35, 1.9, 0.6, "server: average\nthe weights", "#e8eefb", BLUE)
for i, (x, name) in enumerate([(0.0, "hospital A"), (1.65, "hospital B"), (3.3, "hospital C")]):
    box(ax, x, 0.15, 1.4, 0.95, f"{name}\ndata stays here\ntrain locally", "#f2f2f2", "k", fs=5.3)
    cx = x + 0.7
    arr(ax, cx - 0.12, 1.15, 2.35 - 0.15 + 0.15 * (i - 1), 2.3, col=GREEN)
    arr(ax, 2.35 + 0.15 + 0.15 * (i - 1), 2.3, cx + 0.12, 1.15, col=BLUE)
ax.text(0.15, 1.75, "weights\nup", fontsize=5.3, color=GREEN, ha="left")
ax.text(4.55, 1.75, "global\nmodel\ndown", fontsize=5.3, color=BLUE, ha="right")
ax.set_xlim(-0.1, 4.8); ax.set_ylim(0, 3.1); ax.axis("off")
ax2 = fig.add_axes([0.66, 0.22, 0.34, 0.6])
x = np.linspace(0, 1, 300)
for mu, sd, c, n in [(0.35, 0.07, "#555", "A"), (0.5, 0.1, PURPLE, "B"), (0.68, 0.06, RED, "C")]:
    ax2.plot(x, np.exp(-(x - mu) ** 2 / (2 * sd ** 2)), color=c, lw=1, label=f"site {n}")
ax2.set_yticks([]); ax2.set_xticks([]); ax2.set_xlabel("image intensity", fontsize=5.8, labelpad=1)
ax2.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, handlelength=1)
for sp in ["top", "right", "left"]: ax2.spines[sp].set_visible(False)
ax2.set_title("different scanners and\npopulations: heterogeneous", fontsize=6, color=RED)
fig.savefig(OUT / "l6_federated.pdf"); plt.close(fig)

# ---- 3. randomness: are you testing a model or a seed? ----------------------------
rng = np.random.default_rng(4)
a = rng.normal(0.850, 0.008, 10)
b = rng.normal(0.852, 0.008, 10)
fig, ax = plt.subplots(figsize=(W * 0.8, 1.35))
for j, (v, col, name) in enumerate([(a, BLUE, "method A"), (b, PURPLE, "method B")]):
    ax.scatter(v, np.full(10, j) + rng.uniform(-0.08, 0.08, 10), s=8, color=col, lw=0)
    ax.plot([v.mean()] * 2, [j - 0.25, j + 0.25], color=col, lw=1)
    ax.text(0.818, j, name, ha="right", va="center", fontsize=6, color=col)
s1 = int(np.argmax(a - b))
s2 = int(np.argmax(b - a))
for s, txt, y in [(s1, f"only seed {s1}: A wins", 1.42), (s2, f"only seed {s2}: B wins", -0.45)]:
    ax.plot([a[s], b[s]], [0, 1], color="#999", lw=0.6, ls="--")
    ax.text(a[s] if y < 0 else b[s], y, txt, fontsize=5.6, ha="center", va="center", color="#555")
ax.set_ylim(-0.65, 1.65); ax.set_yticks([]); ax.set_xlim(0.82, 0.875)
ax.set_xlabel("Dice, 10 training runs with different seeds", fontsize=6); ax.tick_params(labelsize=5)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
fig.savefig(OUT / "l6_seeds.pdf"); plt.close(fig)
print("written to", OUT)
