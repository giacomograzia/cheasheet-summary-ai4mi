"""Figures for the L5 (Metrics I) and L7 (Metrics II) study notes.

Run:  python make_metrics_figures.py   (writes l5_*.pdf and l7_*.pdf into ../figures; needs numpy, scipy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from scipy import ndimage as ndi
from scipy.signal import convolve2d

OUT = Path(__file__).resolve().parent.parent / "figures"
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, PURPLE, RED, GREEN = "#0037a5", "#690087", "#af0000", "#006928"
LAB, PRED = "#e06060", "#5b8fd9"
rng = np.random.default_rng(7)


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # 8 neighbours (diagonals included)


def boundary(m, kernel=KERNEL):
    n = convolve2d(m.astype(int), kernel, mode="same")
    return m & (n != kernel.sum())


def distances(S, Y):
    """Distances from each boundary voxel of one mask to the other boundary (both directions)."""
    bS, bY = boundary(S), boundary(Y)
    phiS, phiY = ndi.distance_transform_edt(~bS), ndi.distance_transform_edt(~bY)
    return phiS[bY], phiY[bS], phiS, phiY, bS, bY


def hd(S, Y):
    dS, dY, *_ = distances(S, Y)
    return max(dS.max(), dY.max())


def clean(ax, title=None, **kw):
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, **kw)


def show_pair(ax, Y, S, title=None, **kw):
    rgb = np.ones(Y.shape + (3,))
    rgb[Y & ~S] = [0.95, 0.65, 0.65]
    rgb[S & ~Y] = [0.7, 0.8, 0.97]
    rgb[Y & S] = [0.55, 0.4, 0.65]
    ax.imshow(rgb, interpolation="nearest")
    clean(ax, title, **kw)


# ============================== L5 =================================================
# ---- 1. Dice vs Jaccard -------------------------------------------------------------
n = 40
rr, cc = np.mgrid[:n, :n]
Y = (rr - 20) ** 2 + (cc - 16) ** 2 < 11 ** 2
S = (rr - 20) ** 2 + (cc - 24) ** 2 < 11 ** 2
I, U = (Y & S).sum(), (Y | S).sum()
fig, axs = plt.subplots(1, 2, figsize=(W, 1.55), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.2]))
show_pair(axs[0], Y, S, f"label {Y.sum()} vx, prediction {S.sum()} vx\noverlap $|\\cap|$ = {I}")
axs[0].set_xlabel(f"Dice $=\\frac{{2\\cdot{I}}}{{{Y.sum()}+{S.sum()}}}={dice(Y, S):.2f}$,  "
                  f"IoU $=\\frac{{{I}}}{{{U}}}={I / U:.2f}$", fontsize=6.3)
d = np.linspace(0, 1, 200)
ax = axs[1]
ax.plot(d, d / (2 - d), color=BLUE, lw=1)
ax.plot(d, d, color="#bbb", lw=0.5, ls="--")
for name, v in [("M1", 0.9369), ("M2", 0.9391)]:
    ax.plot(v, v / (2 - v), "o", ms=2.5, color=PURPLE)
ax.annotate("M1, M2: same order\nin Dice and in IoU", (0.938, 0.883), (0.12, 0.72), fontsize=5.8,
            color=PURPLE, arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.5))
ax.set_xlabel("Dice", fontsize=6); ax.set_ylabel("IoU $=$ Dice$/(2-$Dice$)$", fontsize=6)
ax.set_title("IoU is a monotone function of Dice", fontsize=6.5)
ax.tick_params(labelsize=5)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.savefig(OUT / "l5_dice_iou.pdf"); plt.close(fig)

# ---- 2. Dice favours over-segmentation --------------------------------------------
G = 100
N = np.arange(0, 61)
over = 2 * G / (2 * G + N)
under = 2 * (G - N) / (2 * G - N)
n = 20
Ysq = np.zeros((n, n), bool); Ysq[5:15, 5:15] = True  # |G| = 100
Sover = Ysq.copy(); Sover[15:17, 5:15] = True  # +20 voxels outside
Sunder = Ysq.copy(); Sunder[13:15, 5:15] = False  # -20 voxels inside
fig, axs = plt.subplots(1, 3, figsize=(W, 1.35), gridspec_kw=dict(wspace=0.3, width_ratios=[0.8, 0.8, 1.3]))
show_pair(axs[0], Ysq, Sover, f"over: +20 vx\nDice = {dice(Ysq, Sover):.3f}", color=GREEN)
show_pair(axs[1], Ysq, Sunder, f"under: $-$20 vx\nDice = {dice(Ysq, Sunder):.3f}", color=RED)
ax = axs[2]
ax.plot(N, over, color=GREEN, lw=1, label="over-seg.")
ax.plot(N, under, color=RED, lw=1, label="under-seg.")
ax.set_xlabel("$N$ wrong voxels ($|G|=100$)", fontsize=6); ax.set_ylabel("Dice", fontsize=6)
ax.legend(fontsize=5.5, frameon=False); ax.tick_params(labelsize=5)
ax.set_title("same error size,\nover-seg. always scores higher", fontsize=6.3)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.savefig(OUT / "l5_overseg.pdf"); plt.close(fig)

# ---- 3. Dice ignores shape and location ----------------------------------------------
n = 40
rr, cc = np.mgrid[:n, :n]
Yb = np.zeros((n, n), bool); Yb[8:32, 8:32] = True  # 576 voxels
k = 60
SA = Yb.copy(); SA[26:32, 8:18] = False  # a contiguous chunk of 60 voxels missing
SB = Yb.copy()
inner = np.argwhere(Yb)
pick = inner[rng.choice(len(inner), k, replace=False)]
SB[pick[:, 0], pick[:, 1]] = False  # 60 scattered voxels missing
SC = np.zeros((n, n), bool); SC[8:32, 8:32] = True; SC[8:14, 8:18] = False; SC[1:7, 30:40] = True  # piece far away
fig, axs = plt.subplots(1, 3, figsize=(W, 1.35), gridspec_kw=dict(wspace=0.1))
for ax, Sx, name in zip(axs, [SA, SB, SC], ["chunk missing", "scattered holes", "piece far away"]):
    show_pair(ax, Yb, Sx, f"{name}\nDice {dice(Yb, Sx):.2f}\nHD {hd(Sx, Yb):.1f} px", fontsize=6.3)
fig.savefig(OUT / "l5_shape.pdf"); plt.close(fig)


# ---- 4. corner cases table (Dice and Hausdorff) --------------------------------------
def corner_table(fname, title, cells, note):
    fig, ax = plt.subplots(figsize=(W * 0.62, 1.15))
    rows = ["something\nto predict", "nothing\nto predict"]
    cols = ["predicted\nsomething", "predicted\nnothing"]
    for i in range(2):
        ax.text(-0.05, 1 - i - 0.5, rows[i], ha="right", va="center", fontsize=6)
        ax.text(i + 0.5, 1.08, cols[i], ha="center", va="bottom", fontsize=6)
        for j in range(2):
            txt, col = cells[i][j]
            ax.add_patch(Rectangle((j, 1 - i - 1), 1, 1, fc=col, ec="k", lw=0.5))
            ax.text(j + 0.5, 1 - i - 0.5, txt, ha="center", va="center", fontsize=6)
    ax.set_xlim(-1.1, 2.05); ax.set_ylim(-1.05, 1.55); ax.axis("off")
    ax.set_title(title, fontsize=6.8, pad=12)
    ax.text(0.5, -1.15, note, ha="center", va="top", fontsize=5.6, color=BLUE)
    fig.savefig(OUT / fname); plt.close(fig)


corner_table("l5_corner.pdf", "Dice corner cases",
             [[("normal\n(0 to 1)", "#ffffff"), ("0", "#fde0e0")],
              [("0", "#fde0e0"), ("$0/0$ = ?\n+$\\epsilon$ $\\to$ 1", "#fff2cc")]],
             "fix: $(2I+\\epsilon)/(U+\\epsilon)$; biases scores up on empty cases")
corner_table("l7_hd_corner.pdf", "Hausdorff corner cases",
             [[("normal\n(mm)", "#ffffff"), ("undefined", "#fde0e0")],
              [("undefined", "#fde0e0"), ("0", "#e9f5e9")]],
             "default for undefined: scan diagonal (mm)")

# ---- 5. reporting: distributions, inter-rater, what is a sample ----------------------
a = np.clip(rng.normal(0.85, 0.03, 40), 0, 1)
b = np.r_[np.clip(rng.normal(0.915, 0.015, 36), 0, 1), [0.25, 0.3, 0.35, 0.4]]
b = b - (b.mean() - a.mean())
fig, axs = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.1]))
ax = axs[0]
for j, (v, col) in enumerate([(a, BLUE), (b, PURPLE)]):
    ax.scatter(j + rng.uniform(-0.12, 0.12, len(v)), v, s=3, color=col, lw=0)
    ax.plot([j - 0.25, j + 0.25], [v.mean()] * 2, color="k", lw=1)
ax.axhspan(0.88, 0.92, color="#ddd", zorder=0)
ax.text(1.5, 0.9, "inter-rater\n≈0.9", fontsize=5.5, va="center", ha="left", color="#555")
ax.set_xticks([0, 1]); ax.set_xticklabels(["method A", "method B"], fontsize=6)
ax.set_xlim(-0.5, 2.1); ax.set_ylabel("Dice per patient", fontsize=6); ax.tick_params(labelsize=5)
ax.set_title(f"same mean ({a.mean():.2f}),\nvery different behaviour", fontsize=6.5)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax = axs[1]
units = ["fold\n(5)", "scan\n(40)", "slice\n(~5000)", "voxel\n(~$10^7$)"]
counts = [5, 40, 5e3, 1e7]
cols = [GREEN, GREEN, RED, RED]
ax.bar(range(4), counts, color=cols, alpha=0.75)
ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(units, fontsize=5.5)
ax.tick_params(labelsize=5)
ax.set_title("what is one sample?\nslices/voxels = fake $n$ (p-hacking)", fontsize=6.5)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.savefig(OUT / "l5_reporting.pdf"); plt.close(fig)

# ============================== L7 =================================================
# ---- 6. accuracy vs precision ----------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(W, 1.05), gridspec_kw=dict(wspace=0.05))
cases = [("accurate +\nprecise", (0, 0), 0.08, GREEN), ("precise,\nnot accurate", (0.5, 0.35), 0.08, PURPLE),
         ("accurate,\nnot precise", (0, 0), 0.35, PURPLE), ("neither", (0.45, -0.4), 0.3, RED)]
for ax, (name, c, s, col) in zip(axs, cases):
    for r_, fc in [(1, "#f3f3f3"), (0.66, "#e3e3e3"), (0.33, "#d0d0d0")]:
        ax.add_patch(Circle((0, 0), r_, fc=fc, ec="#999", lw=0.4))
    ax.plot(0, 0, "+", color="k", ms=4)
    p = rng.normal(c, s, (10, 2))
    ax.scatter(p[:, 0], p[:, 1], s=4, color=col, zorder=3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(name, fontsize=6.2, color=col)
fig.savefig(OUT / "l7_accuracy_precision.pdf"); plt.close(fig)

# ---- 7. Hausdorff step by step (lecture example: circle vs ellipse) ------------------
n = 64
Xs, Ys = np.mgrid[-32:32, -32:32]
S = (Xs ** 2 + Ys ** 2) ** 0.5 < 20
Y = (Xs ** 2 + 2 * Ys ** 2 + Xs * Ys) ** 0.5 < 25
dS, dY, phiS, phiY, bS, bY = distances(S, Y)
nS = convolve2d(S.astype(int), KERNEL, mode="same")
allD = np.r_[dS, dY]
fig, axs = plt.subplots(1, 4, figsize=(W, 1.35), gridspec_kw=dict(wspace=0.12, width_ratios=[1, 1, 1, 1.25]))
show_pair(axs[0], Y, S, "1. masks\nred $Y$, blue $S$", fontsize=6)
rgb = np.ones(S.shape + (3,)); rgb[S] = [0.8, 0.85, 0.95]; rgb[bS] = [0.2, 0.35, 0.8]; rgb[bY] = [0.85, 0.3, 0.3]
axs[1].imshow(rgb, interpolation="nearest")
clean(axs[1], "2. boundaries\n(< 8 neighbours)", fontsize=6)
axs[2].imshow(phiY, cmap="magma")
axs[2].contour(bY.astype(float), [0.5], colors=LAB, linewidths=0.5)
ys_, xs_ = np.nonzero(bS)
axs[2].scatter(xs_, ys_, s=0.6, c=phiY[bS], cmap="cool", lw=0)
clean(axs[2], "3. EDT of $\\partial Y$,\nread on $\\partial S$", fontsize=6)
ax = axs[3]
ax.hist(allD, bins=20, color="#bbb")
for v, name, col in [(allD.max(), "HD (max)", RED), (np.percentile(allD, 95), "HD95", PURPLE),
                     (allD.mean(), "ASSD (mean)", BLUE)]:
    ax.axvline(v, color=col, lw=0.9, label=f"{name}: {v:.1f}")
ax.legend(fontsize=4.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
ax.set_title("4. reduce\n(all distances)", fontsize=6); ax.set_yticks([]); ax.tick_params(labelsize=5)
ax.set_xlabel("distance (px)", fontsize=5.5, labelpad=1)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
fig.savefig(OUT / "l7_hd_steps.pdf"); plt.close(fig)

# ---- 8. when the boundary is the whole object -------------------------------------------
n = 16
thin = np.zeros((n, n), bool); thin[3:13, 7] = True; thin[3:13, 8] = True  # 2 voxels wide
diag = np.zeros((n, n), bool); np.fill_diagonal(diag, True); diag[0, 0] = diag[15, 15] = False
fat = np.zeros((n, n), bool); fat[3:13, 3:13] = True
fig, axs = plt.subplots(1, 3, figsize=(W, 1.25), gridspec_kw=dict(wspace=0.1))
for ax, m, name in zip(axs, [fat, thin, diag], ["thick object", "2 voxels wide", "diagonal line"]):
    b = boundary(m)
    rgb = np.ones((n, n, 3)); rgb[m] = [0.75, 0.82, 0.95]; rgb[b] = [0.88, 0.38, 0.38]
    ax.imshow(rgb, interpolation="nearest")
    frac = b.sum() / m.sum()
    clean(ax, f"{name}\nboundary = {100 * frac:.0f}%", fontsize=6.2,
          color=GREEN if frac < 1 else RED)
fig.savefig(OUT / "l7_thin.pdf"); plt.close(fig)

# ---- 9. wrong slice order (same in label and prediction) -----------------------------
n = 40
rr, cc = np.mgrid[:n, :n]
Xc, Yy = np.mgrid[-32:32, -32:32]
Sc = (Xc ** 2 + Yy ** 2) ** 0.5 < 20
Yc = (Xc ** 2 + 2 * Yy ** 2 + Xc * Yy) ** 0.5 < 25
perm = np.random.default_rng(0).permutation(64)
Yp, Sp = Yc[perm], Sc[perm]
fig, axs = plt.subplots(1, 2, figsize=(W * 0.7, 1.4), gridspec_kw=dict(wspace=0.1))
def assd(S_, Y_):
    a_, b_, *_ = distances(S_, Y_)
    return np.r_[a_, b_].mean()


show_pair(axs[0], Yc, Sc, f"correct slice order\nDice {dice(Yc, Sc):.2f}\nHD {hd(Sc, Yc):.1f}, ASSD {assd(Sc, Yc):.1f}",
          fontsize=6.3)
show_pair(axs[1], Yp, Sp, f"slices shuffled (both)\nDice {dice(Yp, Sp):.2f}\nHD {hd(Sp, Yp):.1f}, ASSD {assd(Sp, Yp):.1f}",
          fontsize=6.3)
axs[1].title.set_color(RED)
fig.savefig(OUT / "l7_slice_order.pdf"); plt.close(fig)

# ---- 10. calibration: reliability diagram and the number of bins --------------------
m = 4000
crng = np.random.default_rng(1)
conf = crng.uniform(0.3, 1, m)
true_acc = np.clip(conf + 0.18 * np.sin(2 * np.pi * (conf - 0.3) / 0.7 * 3), 0, 1)  # over- and under-confident
correct = crng.random(m) < true_acc


def ece(nb):
    edges = np.linspace(0, 1, nb + 1)
    tot, acc_b, conf_b, cnt = 0, [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.sum() == 0:
            acc_b.append(np.nan); conf_b.append(np.nan); cnt.append(0); continue
        a_, c_ = correct[sel].mean(), conf[sel].mean()
        tot += sel.sum() / m * abs(a_ - c_)
        acc_b.append(a_); conf_b.append(c_); cnt.append(sel.sum())
    return tot, edges, np.array(acc_b)


fig, axs = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(wspace=0.45))
e10, edges, accb = ece(10)
ax = axs[0]
ax.bar(edges[:-1], np.nan_to_num(accb), width=0.1, align="edge", color="#9db7e8", ec=BLUE, lw=0.4,
       label="accuracy")
mids = (edges[:-1] + edges[1:]) / 2
ax.bar(edges[:-1], np.where(np.isnan(accb), 0, mids - np.nan_to_num(accb)), bottom=np.nan_to_num(accb),
       width=0.1, align="edge", color="#f4b6b6", ec=RED, lw=0.4, label="gap")
ax.plot([0, 1], [0, 1], "k--", lw=0.6)
ax.set_xlabel("confidence (bin)", fontsize=6); ax.set_ylabel("accuracy in bin", fontsize=6)
ax.set_title(f"reliability diagram\nECE = {e10:.3f} (10 bins)", fontsize=6.5)
ax.legend(fontsize=5, frameon=False, loc="upper left"); ax.tick_params(labelsize=5)
ax = axs[1]
nbs = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
ax.plot(nbs, [ece(k)[0] for k in nbs], "o-", color=PURPLE, ms=2.5, lw=0.8)
ax.set_xscale("log"); ax.set_ylim(0, None); ax.set_xlabel("number of bins", fontsize=6); ax.set_ylabel("ECE", fontsize=6)
ax.set_title("same model, same data:\nECE depends on the bins", fontsize=6.5)
ax.tick_params(labelsize=5)
for a_ in axs:
    for sp in ["top", "right"]: a_.spines[sp].set_visible(False)
fig.savefig(OUT / "l7_calibration.pdf"); plt.close(fig)
print("written to", OUT)
