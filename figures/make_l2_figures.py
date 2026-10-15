"""Figures for the L2 study notes (notation + losses), one toy example throughout.

Toy scan: 6x6 slice, classes 0 = background, 1 = liver, 2 = tumour.
Run:  python make_l2_figures.py   (writes l2_*.pdf next to this file)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLUE, PURPLE, RED, GREEN = "#0037a5", "#690087", "#af0000", "#006928"
CLASS_COLS = ["#e6e6e6", "#7fbf7f", "#e06060"]  # background, liver, tumour
CLASS_NAMES = ["background", "liver", "tumour"]
cls_cmap = ListedColormap(CLASS_COLS)

# ---- toy data ---------------------------------------------------------------
N = 6
Y = np.zeros((N, N), int)
Y[1:5, 1:5] = 1
Y[2:4, 3] = 2
rng = np.random.default_rng(0)
x = np.choose(Y, [0.1, 0.55, 0.9]) + rng.normal(0, 0.04, (N, N))
x = np.clip(x, 0, 1)
y = np.stack([(Y == k).astype(int) for k in range(3)])  # one-hot, (K+1) x 6 x 6

logits = 3.0 * y + rng.normal(0, 0.6, y.shape)
logits[:, 4, 4] = [2.0, 1.0, 0.0]  # a mistake: liver voxel predicted background
s = np.exp(logits) / np.exp(logits).sum(0, keepdims=True)
HI = (0, 5)  # highlighted background voxel with the text's example probabilities
s[:, HI[0], HI[1]] = [0.7, 0.2, 0.1]
S = s.argmax(0)
VOX = (2, 3)  # highlighted tumour voxel


def grid(ax, n=N):
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)


def box(ax, r, c, col=BLUE, lw=1.2):
    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=col, lw=lw, zorder=5))


def arrow(fig, xy0, xy1, text):
    fig.patches.append(FancyArrowPatch(xy0, xy1, transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=8, color="k", lw=0.8))
    fig.text((xy0[0] + xy1[0]) / 2, xy0[1] + 0.05, text, ha="center", va="bottom", fontsize=6.5)


# ---- 1. Omega, x, channels, voxel ---------------------------------------------
fig = plt.figure(figsize=(W, 3.3))
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
rr, cc = np.meshgrid(range(N), range(N), indexing="ij")
ax.scatter(cc, rr, s=6, c="k")
ax.scatter([VOX[1]], [VOX[0]], s=30, facecolors="none", edgecolors=BLUE, lw=1)
ax.annotate(r"voxel $i=(2,3)$", (VOX[1], VOX[0]), (VOX[1] + 0.5, VOX[0] - 1.5), color=BLUE, fontsize=6.5,
            bbox=dict(fc="white", ec="none", pad=0.5), arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
ax.set_xlim(-0.6, N - 0.4); ax.set_ylim(N - 0.4, -0.6)
ax.set_xticks(range(N)); ax.set_yticks(range(N)); ax.tick_params(labelsize=5, length=0)
ax.set_title(r"$\Omega\subset\mathbb{R}^2$: just the positions ($D=2$)" "\n" r"$|\Omega|=6\times6=36$ voxels")

ax = fig.add_subplot(gs[0, 1])
ax.imshow(x, cmap="gray", vmin=0, vmax=1)
for r in range(N):
    for c in range(N):
        ax.text(c, r, f"{x[r, c]:.1f}", ha="center", va="center", fontsize=4.6,
                color="k" if x[r, c] > 0.5 else "w")
box(ax, *VOX); grid(ax)
ax.set_title(r"$x\in\mathbb{R}^{1\times\Omega}$ ($M=1$, e.g. CT)" "\n" r"one number per voxel, $x^{(i)}$")

ax = fig.add_subplot(gs[1, 0])
chans = [("T1", x), ("T2", 1 - 0.8 * x), ("FLAIR", np.where(Y == 2, 0.95, 0.35 * x + 0.1))]
for j, (name, img) in enumerate(chans):
    o = 1.6 * j
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(o, o + N, o + N, o), zorder=3 - j)
    ax.add_patch(Rectangle((o, o), N, N, fill=False, ec="k", lw=0.5, zorder=3 - j))
    ax.text(o + N + 0.2, o + 0.3, name, fontsize=6, va="top")
ax.set_xlim(-0.3, 3.2 + N + 2.2); ax.set_ylim(3.2 + N + 0.3, -0.3); ax.axis("off")
ax.set_title(r"$x\in\mathbb{R}^{M\times\Omega}$, $M=3$ channels" "\n" r"(MRI sequences): shape $3\times6\times6$")

ax = fig.add_subplot(gs[1, 1])
d = np.array([0.55, 0.4])  # depth offset of the drawn brick
front = np.array([[0, 0], [1, 0], [1, 2], [0, 2]])
ax.add_patch(Polygon(front + d, fc="#c9d7f2", ec="k", lw=0.5, ls="--"))
ax.add_patch(Polygon([[0, 2], [1, 2], [1, 2] + d, [0, 2] + d], fc="#b3c6ec", ec="k", lw=0.6))
ax.add_patch(Polygon([[1, 0], [1, 0] + d, [1, 2] + d, [1, 2]], fc="#8faad9", ec="k", lw=0.6))
ax.add_patch(Polygon(front, fc="#a9bfe8", ec="k", lw=0.8))
ax.text(0.5, -0.12, "1 mm", ha="center", va="top", fontsize=6)
ax.text(-0.1, 1, "2 mm", ha="right", va="center", fontsize=6)
ax.text(1 + d[0] / 2 + 0.08, d[1] / 2 - 0.05, "1 mm", ha="left", va="center", fontsize=6)
ax.text(0.5, 1, r"$x^{(i)}$", ha="center", va="center", fontsize=8, color=BLUE)
ax.set_xlim(-0.9, 2.3); ax.set_ylim(-0.45, 2.55); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("one voxel = a brick, one value\n" r"$1\times1\times2$ mm (anisotropic)")
fig.savefig(OUT / "l2_omega_x.pdf"); plt.close(fig)

# ---- 2. label Y -> one-hot y ------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(W, 1.35), gridspec_kw=dict(width_ratios=[1, 1, 1, 1], wspace=0.15))
axs[0].imshow(Y, cmap=cls_cmap, vmin=0, vmax=2)
for r in range(N):
    for c in range(N):
        axs[0].text(c, r, Y[r, c], ha="center", va="center", fontsize=5)
box(axs[0], *VOX); grid(axs[0])
axs[0].set_title(r"$Y\in\mathbb{K}^{\Omega}$" "\n" r"class map, $6\times6$")
for k in range(3):
    ax = axs[k + 1]
    ax.imshow(y[k], cmap=ListedColormap(["white", CLASS_COLS[k] if k else "#a0a0a0"]), vmin=0, vmax=1)
    for r in range(N):
        for c in range(N):
            ax.text(c, r, y[k, r, c], ha="center", va="center", fontsize=4.6, color="#555")
    box(ax, *VOX); grid(ax)
    ax.set_title(rf"$y^{{(\cdot,{k})}}$" "\n" rf"{CLASS_NAMES[k]}")
    for sp in ax.spines.values(): sp.set_color("#999")
fig.savefig(OUT / "l2_onehot.pdf"); plt.close(fig)

# ---- 3. softmax s_theta -> argmax S -----------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(W, 1.35), gridspec_kw=dict(wspace=0.15))
cmaps = ["Greys", "Greens", "Reds"]
for k in range(3):
    ax = axs[k]
    ax.imshow(s[k], cmap=cmaps[k], vmin=0, vmax=1)
    grid(ax); box(ax, *HI, col=PURPLE)
    ax.text(HI[1], HI[0], f"{s[k, HI[0], HI[1]]:.1f}", ha="center", va="center", fontsize=5,
            color="w" if s[k, HI[0], HI[1]] > 0.5 else "k")
    ax.set_title(rf"$s_\theta^{{(\cdot,{k})}}$" "\n" rf"P({CLASS_NAMES[k]})")
axs[3].imshow(S, cmap=cls_cmap, vmin=0, vmax=2)
grid(axs[3]); box(axs[3], 4, 4, col=RED)
axs[3].text(4, 4, "!", ha="center", va="center", color=RED, fontsize=7, weight="bold")
axs[3].set_title(r"$S=\arg\max_k\, s_\theta$" "\n" "prediction")
fig.savefig(OUT / "l2_softmax.pdf"); plt.close(fig)

# ---- 4. simplex + cross-entropy ---------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.1]))
ax = axs[0]
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
bary = lambda p: np.asarray(p) @ V
ctr = bary([1 / 3] * 3)
for k in range(3):  # argmax regions: corner, two edge midpoints, centre
    m1, m2 = (V[k] + V[(k + 1) % 3]) / 2, (V[k] + V[(k + 2) % 3]) / 2
    ax.add_patch(Polygon([V[k], m1, ctr, m2], fc=CLASS_COLS[k], ec="none", alpha=0.6))
ax.add_patch(Polygon(V, fill=False, ec="k", lw=0.7))
labels = [r"$[1,0,0]$" "\nbackground", r"$[0,1,0]$" "\nliver", r"tumour" "\n" r"$[0,0,1]$"]
offs = [(-0.02, -0.04, "right", "top"), (1.02, -0.04, "left", "top"), (0.5, 0.9, "center", "bottom")]
for k in range(3):
    ax.plot(*V[k], "ko", ms=3)
    ax.text(offs[k][0], offs[k][1], labels[k], ha=offs[k][2], va=offs[k][3], fontsize=5.5)
p = bary([0.7, 0.2, 0.1]); ax.plot(*p, "o", color=PURPLE, ms=3.5)
ax.annotate(r"$[0.7,0.2,0.1]$", p, (p[0] + 0.02, -0.2), ha="left", va="top", fontsize=5.5, color=PURPLE,
            arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.5))
ax.plot(*ctr, "kx", ms=4)
ax.annotate(r"$[\frac{1}{3},\frac{1}{3},\frac{1}{3}]$", ctr, (0.9, 0.62), fontsize=5.5,
            arrowprops=dict(arrowstyle="-", lw=0.5))
ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.25, 1.15); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"simplex $\Delta^2$ (3 classes)" "\n" r"corner = one-hot $y$" "\n" r"inside = softmax $s_\theta$")

ax = axs[1]
t = np.linspace(0.004, 1, 400)
ax.plot(t, -np.log(t), color=BLUE, lw=1)
for v, txt, tx, ty in [(0.9, "0.11", 0.66, 0.75), (0.5, "0.69", 0.52, 1.3), (0.01, "4.6", 0.08, 5.0)]:
    ax.plot(v, -np.log(v), "o", color=PURPLE, ms=3)
    ax.annotate(rf"$s={v}\to{txt}$", (v, -np.log(v)), (tx, ty), fontsize=5.5,
                color=PURPLE)
ax.annotate("slope $-1/s\\to-\\infty$\n(confident & wrong)", (0.012, 4.4), (0.25, 3.6), fontsize=5.5,
            color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
ax.set_xlabel(r"$s$ = P(true class)"); ax.set_ylabel(r"$-\log s$")
ax.set_xlim(0, 1.05); ax.set_ylim(0, 5.6); ax.tick_params(labelsize=5.5)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.set_title("cross-entropy per voxel")
fig.savefig(OUT / "l2_simplex_ce.pdf"); plt.close(fig)

# ---- 5. Dice 1D example -----------------------------------------------------
fig, ax = plt.subplots(figsize=(W, 0.95))
lab, pred = np.zeros(10, int), np.zeros(10, int)
lab[2:6], pred[3:7] = 1, 1  # voxels 3-6 and 4-7 (1-indexed)
for row, (name, v) in enumerate([("label $y$", lab), ("prediction $s$", pred)]):
    for i in range(10):
        both = lab[i] and pred[i]
        fc = (CLASS_COLS[2] if both else "#f4b6b6") if v[i] else "white"
        ax.add_patch(Rectangle((i, -row), 0.95, 0.8, fc=fc, ec="#777", lw=0.5))
        ax.text(i + 0.47, -row + 0.4, v[i], ha="center", va="center", fontsize=6)
    ax.text(-0.2, -row + 0.4, name, ha="right", va="center", fontsize=6.5)
for i in range(10):
    ax.text(i + 0.47, 0.95, i + 1, ha="center", va="bottom", fontsize=5, color="#777")
ax.text(10.3, 0.4, r"$I=\sum y\,s=3$ (dark)" "\n" r"$U=\sum(y+s)=4+4=8$", fontsize=6.5, va="center",
        color=BLUE)
ax.text(10.3, -0.6, r"Dice $=\frac{2\cdot3}{8}=0.75$, loss $=0.25$", fontsize=6.5, va="center", color=PURPLE)
ax.set_xlim(-3.2, 17); ax.set_ylim(-1.1, 1.3); ax.axis("off")
fig.savefig(OUT / "l2_dice.pdf"); plt.close(fig)
print("written to", OUT)
