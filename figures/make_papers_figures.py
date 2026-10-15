"""Figures for the 'Course papers' study notes.

Numbers marked 'from the paper' are copied from the PDFs in ../../papers; everything else is a toy example.
Run:  python make_papers_figures.py   (writes papers_*.pdf next to this file; needs numpy, scipy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import interpolate
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})
BLUE, PURPLE, RED, GREEN, GREY = "#0037a5", "#690087", "#af0000", "#006928", "#777777"


def textbox(ax, x, y, text, fc, fs=5.6, padx=0.03, pady=0.035, minw=0.0):
    """Box sized to its text; returns (cx, cy, w, h) in data coordinates."""
    txt = ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.15)
    bb = txt.get_window_extent(ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())
    w, h = max(bb.width + 2 * padx, minw), bb.height + 2 * pady
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, zorder=2, lw=0.5, ec="k", fc=fc,
                                boxstyle=f"round,pad=0,rounding_size={0.18 * min(w, h)}"))
    return (x, y, w, h)


def edge(b, toward):
    """Point where the line from the centre of box b towards `toward` leaves the box."""
    cx, cy, w, h = b
    dx, dy = toward[0] - cx, toward[1] - cy
    s = min(w / 2 / abs(dx) if dx else np.inf, h / 2 / abs(dy) if dy else np.inf)
    return (cx + s * dx, cy + s * dy)


def connect(ax, a, b, col="k"):
    """Arrow from the border of box a to the border of box b (or to a point b)."""
    end = edge(b, a[:2]) if len(b) == 4 else b
    ax.annotate("", end, edge(a, b[:2]), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=7))


def clean(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=5.5, length=2)


# ---- 1. nnU-Net: three kinds of parameters + why the GPU matters -------------
fig = plt.figure(figsize=(W, 2.05))
ax = fig.add_axes([0.0, 0.0, 0.66, 0.9])
ax.set_xlim(-0.21, 1.12); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("nnU-Net: how it configures itself", fontsize=6.5)
fp = textbox(ax, 0.52, 0.88, "dataset fingerprint\nspacing, sizes, modality,\nintensities, nr. classes", "#f6d5e6")
rb = textbox(ax, 0.52, 0.6, "rule-based parameters\nresampling, normalisation,\npatch & batch size, topology", "#cdeccd")
gpu = textbox(ax, 1.0, 0.6, "GPU\nmemory", "#eeeeee")
em = textbox(ax, 0.68, 0.3, "empirical (5-fold CV)\n2D / 3D / cascade / ensemble?\nkeep largest component?", "#fde9c8")
fx = textbox(ax, 0.02, 0.3, "fixed\nU-Net, CE+Dice,\nSGD, augment.", "#cfdcf5")
connect(ax, fp, rb); connect(ax, gpu, rb); connect(ax, fx, em)
connect(ax, rb, em)
out = (em[0], 0.05)
connect(ax, em, out)
ax.text(out[0], out[1] - 0.01, "trained nnU-Net", ha="center", va="top", fontsize=5.8)

ax = fig.add_axes([0.68, 0.32, 0.3, 0.55])
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="k", lw=0.6))
ax.text(0.5, 1.05, "median image shape", ha="center", va="bottom", fontsize=5.8)
ax.add_patch(Rectangle((0.05, 0.05), 0.48, 0.48, fc="#a9bfe8", ec=BLUE, lw=0.7, alpha=0.8))
ax.text(0.08, 0.08, "small\nGPU", ha="left", va="bottom", fontsize=5.6)
ax.add_patch(Rectangle((0.3, 0.3), 0.65, 0.65, fill=False, ec=GREEN, lw=1, ls="--"))
ax.text(0.92, 0.92, "big GPU", ha="right", va="top", fontsize=5.6, color=GREEN)
fig.text(0.83, 0.27, "patch shrunk until a\nbatch of 2 fits: other\npatch $\\Rightarrow$ other depth\n$\\Rightarrow$ maybe another\n'best' configuration",
         ha="center", va="top", fontsize=5.4, color=RED, linespacing=1.15)
fig.savefig(OUT / "papers_nnunet.pdf"); plt.close(fig)

# ---- 2. Loss Odyssey: 4 families + how imbalanced the tasks are --------------
fig = plt.figure(figsize=(W, 1.95))
ax = fig.add_axes([0.0, 0.0, 0.5, 0.95])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top = textbox(ax, 0.5, 0.93, "20 segmentation losses", "#eeeeee", 5.8)
fams = [("distribution", "CE, weighted CE,\nTopK, focal", "#cfdcf5"),
        ("region", "Dice, IoU,\nTversky", "#cdeccd"),
        ("boundary", "boundary loss,\nHD loss", "#fde9c8"),
        ("compound", "e.g. Dice + CE,\nDice + TopK", "#f6d5e6")]
for k, (name, ex, fc) in enumerate(fams):
    y = 0.73 - 0.165 * k
    b = textbox(ax, 0.2, y, name, fc, 5.8, padx=0.04, minw=0.36)
    ax.text(0.43, y, ex, va="center", fontsize=5.4, linespacing=1.1)
ax.text(0.5, 0.06, "no single winner;\nDice-based compounds most robust", ha="center", va="center",
        fontsize=5.6, color=RED, linespacing=1.15)

ax = fig.add_axes([0.62, 0.2, 0.37, 0.62])
tasks = ["liver", "pancreas", "liver\ntumour", "multi-\norgan"]
lo = np.array([42, 503, 576, 30]); hi = np.array([42, 503, 576, 1243])  # from the paper (BG:FG)
ax.bar(range(4), lo, color="#9db7e8", width=0.62)
ax.bar(3, hi[3] - lo[3], bottom=lo[3], color="#9db7e8", width=0.62, alpha=0.45, hatch="////", ec="w", lw=0)
for i, (a_, b_) in enumerate(zip(lo, hi)):
    ax.text(i, b_ * 1.25, f"{a_}:1" if a_ == b_ else f"{a_}–\n{b_}:1", ha="center", va="bottom", fontsize=5, linespacing=1)
ax.set_yscale("log"); ax.set_ylim(10, 2e4)
ax.set_xticks(range(4)); ax.set_xticklabels(tasks, fontsize=5.2, linespacing=1.05)
ax.set_ylabel("background : foreground", fontsize=5.6)
ax.set_title("imbalance of the 4 tasks\n(from the paper)", fontsize=6)
clean(ax)
fig.savefig(OUT / "papers_loss.pdf"); plt.close(fig)

# ---- 3. Metrics Reloaded: Dice on small structures + aggregation -------------
fig, axs = plt.subplots(1, 3, figsize=(W, 1.25), gridspec_kw=dict(wspace=0.1))
for ax, side in zip(axs, [12, 4, 2]):
    n = 16
    y = np.zeros((n, n), bool); s = np.zeros((n, n), bool)
    o = (n - side) // 2
    y[o:o + side, o:o + side] = True
    s[o:o + side, o + 1:o + side + 1] = True  # same error everywhere: shifted by 1 pixel
    rgb = np.ones((n, n, 3))
    rgb[y & ~s] = [0.95, 0.7, 0.7]; rgb[s & ~y] = [0.7, 0.75, 0.95]; rgb[y & s] = [0.55, 0.75, 0.55]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    dsc = 2 * (y & s).sum() / (y.sum() + s.sum())
    ax.set_title(f"{side}×{side} object\nDice {dsc:.2f}", color=RED if dsc < 0.6 else "k")
fig.savefig(OUT / "papers_metrics_small.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(W * 0.8, 1.35))
pat = ["P1 (20 images)", "P2 (2)", "P3 (2)", "P4 (2)", "P5 (2)"]
vals = [0.9, 0.5, 0.5, 0.4, 0.8]
nimg = [20, 2, 2, 2, 2]
ax.bar(range(5), vals, color=["#9db7e8"] + ["#c9d7f2"] * 4, width=0.6)
per_img = np.dot(vals, nimg) / sum(nimg); per_pat = np.mean(vals)
ax.axhline(per_img, color=RED, lw=0.8, ls="--"); ax.axhline(per_pat, color=GREEN, lw=0.8)
ax.text(2, per_img + 0.02, f"mean over all 28 images: {per_img:.2f}", ha="center", va="bottom", fontsize=6, color=RED)
ax.text(2, per_pat - 0.02, f"mean over 5 patients: {per_pat:.2f}", ha="center", va="top", fontsize=6, color=GREEN)
ax.set_xlim(-0.5, 4.5)
ax.set_xticks(range(5)); ax.set_xticklabels(pat, fontsize=5.5); ax.set_ylim(0, 1.0)
ax.set_title("aggregation: per image vs per patient")
ax.set_ylabel("mean Dice per patient", fontsize=6)
clean(ax)
fig.savefig(OUT / "papers_metrics_agg.pdf"); plt.close(fig)

# ---- 4. Implementation pitfalls: boundary choice + empty-mask outputs --------
n = 14
rr, cc = np.mgrid[:n, :n]
mask = ((rr - 6.5) / 5) ** 2 + ((cc - 6.5) / 4) ** 2 < 1
b4 = mask & ~ndi.binary_erosion(mask, ndi.generate_binary_structure(2, 1))
b8 = mask & ~ndi.binary_erosion(mask, ndi.generate_binary_structure(2, 2))
fig = plt.figure(figsize=(W, 1.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.55], wspace=0.3)
for k, (b, name) in enumerate([(b4, "4-neighbour erosion"), (b8, "8-neighbour erosion")]):
    ax = fig.add_subplot(gs[0, k])
    rgb = np.ones((n, n, 3)); rgb[mask] = [0.85, 0.88, 0.95]; rgb[b] = [0.0, 0.22, 0.65]
    ax.imshow(rgb, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}:\n{b.sum()} boundary px", fontsize=5.8)
ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
rows = [("Google DeepMind", "∞"), ("MedPy", "error"), ("Metrics Reloaded", "NaN"), ("MISeval", "random > 0"),
        ("MONAI", "NaN"), ("pymia", "∞"), ("seg-metrics", "0"), ("SimpleITK", "error")]
ax.text(0, 1.0, "HD when one mask is empty\n(Table 2 of the paper):", fontsize=5.8, va="top", weight="bold")
for i, (tool, out) in enumerate(rows):
    y = 0.74 - 0.093 * i
    ax.text(0.02, y, tool, fontsize=5.6, va="center")
    ax.text(0.98, y, out, fontsize=5.6, va="center", ha="right", color=RED)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(OUT / "papers_pitfalls.pdf"); plt.close(fig)

# ---- 5. DisTorch: few boundary voxels, brute-force pairs, runtime ------------
n = 40
rr, cc = np.mgrid[:n, :n]
g = ((rr - 20) / 13) ** 2 + ((cc - 18) / 11) ** 2 < 1
s = ((rr - 21) / 12) ** 2 + ((cc - 22) / 11) ** 2 < 1
bd = lambda m: m & ~ndi.binary_erosion(m)
bg_, bs_ = bd(g), bd(s)
fig, axs = plt.subplots(1, 2, figsize=(W, 1.7), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.45))
ax = axs[0]
rgb = np.ones((n, n, 3)); rgb[g] = [0.92, 0.92, 0.92]
rgb[bg_] = [0.0, 0.22, 0.65]; rgb[bs_] = [0.69, 0.0, 0.0]; rgb[bg_ & bs_] = [0.4, 0.4, 0.4]
ax.imshow(rgb, interpolation="nearest")
pg = np.argwhere(bg_ & ~bs_); ps = np.argwhere(bs_ & ~bg_)
d = np.sqrt(((ps[:, None, :] - pg[None, :, :]) ** 2).sum(-1))
i_far = d.min(1).argmax(); j_far = d[i_far].argmin()
for i in range(0, len(ps), 9):  # a few of the pairwise distances
    for j in range(0, len(pg), 14):
        ax.plot([ps[i, 1], pg[j, 1]], [ps[i, 0], pg[j, 0]], color=GREY, lw=0.15, alpha=0.6)
ax.plot([ps[i_far, 1], pg[j_far, 1]], [ps[i_far, 0], pg[j_far, 0]], color=PURPLE, lw=1.2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"area {g.sum()} px, boundary {bg_.sum()} px\nall boundary pairs; purple = HD", fontsize=5.8)
ax = axs[1]
tools = ["MedPy", "MeshMetrics", "MONAI", "MONAI\n+cuCIM", "DisTorch"]
ms = [2.6e4, 8.5e3, 723, 24.9, 29.1]  # SegTHOR, ms per scan, from the paper (Table 1)
cols = ["#c9c9c9", "#c9c9c9", "#9db7e8", "#9db7e8", "#7fbf7f"]
ax.barh(range(5), ms, color=cols, height=0.6)
for i, v in enumerate(ms):
    ax.text(v * 1.2, i, f"{v:g} ms" if v < 1000 else f"{v / 1000:g} s", va="center", fontsize=5.3)
ax.set_xscale("log"); ax.set_xlim(10, 3e5); ax.invert_yaxis()
ax.set_yticks(range(5)); ax.set_yticklabels(tools, fontsize=5.4)
ax.set_title("HD runtime per SegTHOR scan\n(from the paper; log scale)", fontsize=6)
clean(ax)
fig.savefig(OUT / "papers_distorch.pdf"); plt.close(fig)

# ---- 6. A pixel is not a little square --------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(W, 1.3), gridspec_kw=dict(width_ratios=[1.6, 1, 1], wspace=0.2))
ax = axs[0]
t = np.linspace(0, 10, 500)
f = np.sin(t * 0.9) + 0.4 * np.sin(t * 2.1)
xs = np.arange(0, 11, 1.0)
ys = np.sin(xs * 0.9) + 0.4 * np.sin(xs * 2.1)
ax.plot(t, f, color="#bbbbbb", lw=1.6, label="true signal")
ax.step(xs, ys, where="mid", color=RED, lw=0.8, label="box filter = 'little squares'")
ax.plot(t, interpolate.CubicSpline(xs, ys)(t), color=GREEN, lw=0.9, label="smooth reconstruction")
ax.plot(xs, ys, "o", color="k", ms=2.5, label="pixels = point samples", zorder=5)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.03), frameon=False, ncol=2)
ax.set_title("1D signal")
small = np.array([[0.1, 0.4, 0.2], [0.5, 1.0, 0.6], [0.2, 0.6, 0.3]])
axs[1].imshow(small, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axs[1].set_title("nearest zoom:\nsquares")
axs[2].imshow(small, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
axs[2].set_title("bicubic zoom:\nsame 9 samples")
for ax in axs[1:]:
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT / "papers_pixel.pdf"); plt.close(fig)
print("written to", OUT)
