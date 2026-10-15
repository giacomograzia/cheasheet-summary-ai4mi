"""Figures for the guest-lecture study notes.

Run:  python make_guest_figures.py   (writes guest_*.pdf next to this file; needs numpy, scipy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, PURPLE, RED, GREEN = "#0037a5", "#690087", "#af0000", "#006928"
rng = np.random.default_rng(5)

# ---- 1. equivariance (segmentation) vs invariance (classification) ----------------
N = 64
rr, cc = np.mgrid[:N, :N]
d = np.hypot(rr - 32, cc - 36)
lv = d < 9  # left-ventricle cavity
myo = (d >= 9) & (d < 14)  # myocardium ring
rv = (np.hypot(rr - 30, cc - 18) < 13) & ~(d < 15)  # right-ventricle crescent
lab = np.zeros((N, N), int); lab[rv] = 1; lab[myo] = 2; lab[lv] = 3
img = np.choose(lab, [0.15, 0.75, 0.35, 0.8]) + rng.normal(0, 0.04, (N, N))
img = ndi.gaussian_filter(img, 0.8)
ang = 70
img_r = ndi.rotate(img, ang, reshape=False, order=1, mode="nearest")
lab_r = ndi.rotate(lab, ang, reshape=False, order=0)
seg_cmap = ListedColormap(["#eeeeee", "#7f9fdf", "#e06060", "#e0b050"])
fig, axs = plt.subplots(2, 3, figsize=(W, 2.35), gridspec_kw=dict(wspace=0.25, hspace=0.25))
for row, (im, lb, nm) in enumerate([(img, lab, "$x$"), (img_r, lab_r, r"$g\cdot x$ (rotated)")]):
    axs[row, 0].imshow(im, cmap="gray", vmin=0, vmax=1)
    axs[row, 1].imshow(lb, cmap=seg_cmap, vmin=0, vmax=3, interpolation="nearest")
    axs[row, 0].set_ylabel(nm, fontsize=6.5)
    axs[row, 2].axis("off")
    axs[row, 2].text(0.5, 0.5, "class:\n\"normal heart\"", ha="center", va="center", fontsize=7,
                     bbox=dict(fc="#e9f5e9", ec=GREEN, lw=0.6, boxstyle="round"))
    for ax in axs[row, :2]:
        ax.set_xticks([]); ax.set_yticks([])
axs[0, 0].set_title("input", fontsize=6.5)
axs[0, 1].set_title("segmentation:\nrotates along\n(equivariant)", fontsize=6.2, color=BLUE)
axs[0, 2].set_title("classification:\nstays the same\n(invariant)", fontsize=6.2, color=GREEN)
fig.savefig(OUT / "guest_equivariance.pdf"); plt.close(fig)

# ---- 2. cardiac phase on a circle -------------------------------------------------
T = 30
fig, axs = plt.subplots(1, 2, figsize=(W, 1.45), gridspec_kw=dict(wspace=0.35, width_ratios=[1.4, 1]))
ax = axs[0]
t = np.arange(T)
ax.scatter(t, np.zeros(T), c=t, cmap="twilight", s=10)
ax.annotate("", (T - 1, 0.25), (0, 0.25), arrowprops=dict(arrowstyle="<->", color=RED, lw=0.7))
ax.text(T / 2, 0.32, "frame 0 and frame 29: 29 apart as numbers,\nbut neighbours in the heartbeat", ha="center",
        fontsize=5.5, color=RED)
ax.plot([0, 0], [-0.15, 0.15], color="k", lw=0.6); ax.text(0, -0.3, "ED", ha="center", fontsize=5.5)
ax.set_ylim(-0.5, 0.7); ax.set_xlim(-2, T + 1); ax.axis("off")
ax.set_title("phase as a frame number: a line", fontsize=6.5)
ax = axs[1]
phi = 2 * np.pi * t / T
ax.add_patch(Circle((0, 0), 1, fill=False, ec="#999", lw=0.5))
ax.scatter(np.cos(phi), np.sin(phi), c=t, cmap="twilight", s=10, zorder=3)
ax.text(1.12, 0, "ED", fontsize=5.5, va="center")
ax.text(0, 0, r"$(\cos\varphi,\sin\varphi)$", ha="center", va="center", fontsize=6)
ax.set_xlim(-1.3, 1.45); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("on a circle: periodic,\nand every frame has a target", fontsize=6.5, color=GREEN)
fig.savefig(OUT / "guest_phase_circle.pdf"); plt.close(fig)

# ---- 3. data leakage: random split vs target (group) split -------------------------
k, per = 8, 14
centres = rng.uniform(-4, 4, (k, 2))
pts = np.concatenate([c + rng.normal(0, 0.28, (per, 2)) for c in centres])
grp = np.repeat(np.arange(k), per)
rand_test = rng.random(len(pts)) < 0.2
grp_test = np.isin(grp, [2, 6])
fig, axs = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(wspace=0.08))
for ax, test, title, col in [(axs[0], rand_test, "random split: every test point\nhas a near-twin in train $\\to$ 99.9%", RED),
                             (axs[1], grp_test, "target split: whole groups unseen\n$\\to$ honest performance", GREEN)]:
    ax.scatter(pts[~test, 0], pts[~test, 1], s=5, color="#9db7e8", lw=0, label="train")
    ax.scatter(pts[test, 0], pts[test, 1], s=9, marker="x", color=RED, lw=0.7, label="test")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=6.2, color=col)
    ax.set_xlim(-5.2, 5.2); ax.set_ylim(-5.2, 5.2); ax.set_aspect("equal")
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=5.5, frameon=False, loc="lower center", ncol=2,
           markerscale=1.5, bbox_to_anchor=(0.5, -0.06))
fig.savefig(OUT / "guest_splits.pdf"); plt.close(fig)

# ---- 4. abstaining: answer fewer cases, be right more often -------------------------
n = 2000
score = rng.normal(0, 1, n)  # how sure the model is (higher = surer)
correct = rng.random(n) < 1 / (1 + np.exp(-(1.0 + 1.9 * score)))
order = np.argsort(-score)
cov = np.arange(1, n + 1) / n
prec = np.cumsum(correct[order]) / np.arange(1, n + 1)
i20 = int(0.2 * n) - 1
fig, ax = plt.subplots(figsize=(W * 0.72, 1.45))
ax.plot(cov * 100, prec * 100, color=PURPLE, lw=1)
ax.plot(cov[i20] * 100, prec[i20] * 100, "o", color=GREEN, ms=3.5)
ax.annotate(f"answer 20%: {prec[i20]:.0%} right\n(abstain on 80% → sequencing)", (20, prec[i20] * 100),
            (28, prec[i20] * 100 + 3), fontsize=5.6, color=GREEN, va="bottom",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.5))
ax.plot(100, prec[-1] * 100, "o", color=RED, ms=3.5)
ax.text(98, prec[-1] * 100 - 3, f"answer all: {prec[-1]:.0%}", fontsize=5.6, color=RED, ha="right", va="top")
ax.set_xlabel("cases answered (%)", fontsize=6); ax.set_ylabel("correct (%)", fontsize=6)
ax.set_ylim(prec[-1] * 100 - 12, 112); ax.set_yticks(range(60, 101, 10)); ax.spines["left"].set_bounds(55, 100); ax.tick_params(labelsize=5)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.savefig(OUT / "guest_abstain.pdf"); plt.close(fig)
print("written to", OUT, "precision@20%:", prec[i20], "overall:", prec[-1])
