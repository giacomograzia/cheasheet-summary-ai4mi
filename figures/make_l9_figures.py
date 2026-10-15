"""Figures for the L9 study notes (classification).

Run:  python make_l9_figures.py   (writes l9_*.pdf next to this file; needs numpy, scipy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, PURPLE, RED, GREEN = "#0037a5", "#690087", "#af0000", "#006928"
rng = np.random.default_rng(2)

# ---- 1. a virtual panel: one network per rater, then vote --------------------------
classes = ["NDBE", "LGD", "HGD", "IND"]
cases = [("case 1: clear", [6, 1, 0, 0]), ("case 2: ambiguous", [3, 3, 1, 0])]
fig, axs = plt.subplots(1, 2, figsize=(W, 1.45), gridspec_kw=dict(wspace=0.3))
for ax, (name, votes) in zip(axs, cases):
    frac = np.array(votes) / sum(votes)
    top = int(np.argmax(frac))
    cons = frac[top] > 0.75
    cols = [GREEN if (i == top and cons) else ("#9db7e8" if i == top else "#cccccc") for i in range(4)]
    ax.bar(classes, frac, color=cols, ec="k", lw=0.4)
    ax.axhline(0.75, color=RED, lw=0.7, ls="--")
    ax.text(3.45, 0.77, "75%", color=RED, fontsize=5.5, ha="right", va="bottom")
    ax.set_ylim(0, 1); ax.tick_params(labelsize=5.5)
    verdict = f"consensus: {classes[top]}" if cons else "no consensus: uncertain"
    ax.set_title(f"{name}\n{verdict}", fontsize=6.3, color=GREEN if cons else RED)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axs[0].set_ylabel("fraction of the 7\nrater networks", fontsize=5.8)
fig.savefig(OUT / "l9_panel.pdf"); plt.close(fig)

# ---- 2. multiple-instance learning on a whole-slide image ------------------------
H, Wd, P = 60, 96, 6
tissue = ndi.gaussian_filter(rng.random((H, Wd)), 6) > 0.5
tissue = ndi.binary_opening(tissue, iterations=2)
tex = np.where(tissue, 0.75 + 0.15 * ndi.gaussian_filter(rng.random((H, Wd)), 1), 0.97)
rgb = np.stack([tex, 0.6 + 0.35 * tex - 0.3 * tissue, 0.75 + 0.2 * tex], -1).clip(0, 1)  # pinkish H&E look
gh, gw = H // P, Wd // P
frac_t = tissue.reshape(gh, P, gw, P).mean((1, 3))
keep = frac_t > 0.3
cand = np.argwhere(keep)
hot = tuple(cand[len(cand) // 2])  # the one tumour patch ("smoking gun")
att = np.where(keep, rng.uniform(0.02, 0.12, (gh, gw)), np.nan)
att[hot] = 1.0
fig, axs = plt.subplots(1, 2, figsize=(W, 1.35), gridspec_kw=dict(wspace=0.08))
axs[0].imshow(rgb, interpolation="nearest")
for i in range(gh):
    for j in range(gw):
        axs[0].add_patch(Rectangle((j * P - 0.5, i * P - 0.5), P, P, fill=not keep[i, j],
                                   fc=(1, 1, 1, 0.75), ec="#555" if keep[i, j] else "#ddd", lw=0.3))
axs[0].add_patch(Rectangle((hot[1] * P - 0.5, hot[0] * P - 0.5), P, P, fill=False, ec=RED, lw=1.2))
axs[0].set_title(f"slide = bag of {keep.sum()} tissue patches\n(background patches dropped)", fontsize=6.2)
im = axs[1].imshow(att, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
axs[1].set_title("attention weight per patch\n(one \"smoking gun\" patch)", fontsize=6.2)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT / "l9_mil.pdf"); plt.close(fig)


# ---- 3. shortcut learning: a spurious feature that only exists in the training hospital ----
def make(n, spurious_corr):
    yl = rng.random(n) < 0.5
    signal = rng.normal(0, 1, n) + 0.9 * yl  # the real, weak disease signal
    marker = np.where(rng.random(n) < spurious_corr, yl, rng.random(n) < 0.5).astype(float)
    marker = marker + rng.normal(0, 0.12, n)  # e.g. "implant / contrast present"
    return np.c_[signal, marker], yl


def fit_logreg(X, y, steps=3000, lr=0.5):
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - y) / len(y)
    return w


Xtr, ytr = make(400, 0.95)
Xte, yte = make(400, 0.0)
w = fit_logreg(Xtr, ytr)
acc = lambda X, y: (((np.c_[X, np.ones(len(X))] @ w) > 0) == y).mean()
fig, axs = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(wspace=0.12))
xx = np.linspace(-3, 4, 50)
for ax, (X, y, title, col) in zip(axs, [(Xtr, ytr, f"training hospital\naccuracy {acc(Xtr, ytr):.0%}", GREEN),
                                       (Xte, yte, f"external hospital\naccuracy {acc(Xte, yte):.0%}", RED)]):
    ax.scatter(X[~y, 0], X[~y, 1], s=2, color=BLUE, lw=0, label="healthy")
    ax.scatter(X[y, 0], X[y, 1], s=2, color=RED, lw=0, label="cancer")
    ax.plot(xx, -(w[0] * xx + w[2]) / w[1], color="k", lw=0.8)
    ax.set_xlim(-3, 4); ax.set_ylim(-0.5, 1.5); ax.set_title(title, fontsize=6.3, color=col)
    ax.set_xlabel("real disease signal", fontsize=5.8); ax.tick_params(labelsize=5)
axs[0].set_ylabel("spurious feature\n(implant / contrast)", fontsize=5.8)
axs[1].set_yticks([])
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=5.5, frameon=False, loc="lower center", ncol=2,
           markerscale=3, bbox_to_anchor=(0.5, -0.2))
fig.savefig(OUT / "l9_shortcut.pdf"); plt.close(fig)
print("written to", OUT, " train/test acc:", acc(Xtr, ytr), acc(Xte, yte), "weights", w)
