"""Figures for the L3 study notes (registration).

Run:  python make_l3_figures.py   (writes l3_*.pdf into ../figures; needs numpy, scipy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent.parent / "figures"
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, PURPLE, RED, GREEN = "#0037a5", "#690087", "#af0000", "#006928"
rng = np.random.default_rng(3)

# ---- toy slice: body with two organs, as a tissue map ------------------------
N = 64
rr, cc = np.mgrid[:N, :N]
tissue = np.zeros((N, N), int)
tissue[((rr - 32) / 24) ** 2 + ((cc - 32) / 28) ** 2 < 1] = 1  # body
tissue[((rr - 26) / 8) ** 2 + ((cc - 22) / 10) ** 2 < 1] = 2  # organ A (e.g. liver)
tissue[(rr - 40) ** 2 + (cc - 42) ** 2 < 6 ** 2] = 3  # organ B (e.g. kidney)
CT_VALS = np.array([0.0, 0.4, 0.8, 0.6])
MR_VALS = np.array([0.6, 0.2, 0.9, 0.1])  # different modality: unrelated brightness per tissue


def render(vals, t=tissue, noise=0.02):
    return ndi.gaussian_filter(vals[t].astype(float), 0.8) + rng.normal(0, noise, t.shape)


def move(im, angle=12, shift=(5, -6), order=1):
    return ndi.shift(ndi.rotate(im, angle, reshape=False, order=order), shift, order=order)


def clean(ax, title=None, **kw):
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, **kw)


def overlay(a, b):  # fixed in magenta, moving in green: aligned parts turn grey/white
    rgb = np.stack([a, b, a], -1)
    return np.clip(rgb, 0, 1)


# ---- 1. the registration problem -----------------------------------------------
F = render(CT_VALS)
M = move(F)
TM = ndi.rotate(ndi.shift(M, (-5, 6), order=1), -12, reshape=False, order=1)  # undo: T(M)
ssd = lambda a, b: float(((a - b) ** 2).sum())
fig, axs = plt.subplots(1, 4, figsize=(W, 1.25), gridspec_kw=dict(wspace=0.08))
axs[0].imshow(F, cmap="gray", vmin=0, vmax=1); clean(axs[0], "fixed\n" r"$\mathcal{F}$")
axs[1].imshow(M, cmap="gray", vmin=0, vmax=1); clean(axs[1], "moving\n" r"$\mathcal{M}$")
axs[2].imshow(overlay(F, M)); clean(axs[2], f"before\nSSD = {ssd(F, M):.0f}", color=RED)
axs[3].imshow(overlay(F, TM)); clean(axs[3], f"after, $\\mathcal{{T}}(\\mathcal{{M}})$\nSSD = {ssd(F, TM):.0f}",
                                     color=GREEN)
fig.savefig(OUT / "l3_problem.pdf"); plt.close(fig)

# ---- 2. similarity measures: what relation between intensities they assume -----
pairs = [("same modality", CT_VALS, "SSD: identity"),
         ("brightness change", 0.6 * CT_VALS + 0.3, "NCC: linear"),
         ("other modality", MR_VALS, "MI: any dependency")]
pairs[1] = (pairs[1][0], np.where(np.arange(4) == 0, 0.3, pairs[1][1]), pairs[1][2])
fig, axs = plt.subplots(2, 3, figsize=(W, 2.45), gridspec_kw=dict(wspace=0.35, hspace=0.45,
                                                                 height_ratios=[1, 1.1]))
for j, (name, vals, rel) in enumerate(pairs):
    Mj = render(vals)
    axs[0, j].imshow(Mj, cmap="gray", vmin=0, vmax=1); clean(axs[0, j], f"$\\mathcal{{M}}$: {name}")
    ax = axs[1, j]
    idx = rng.choice(N * N, 900, replace=False)
    ax.scatter(F.ravel()[idx], Mj.ravel()[idx], s=1.5, color=BLUE, alpha=0.5, lw=0)
    ax.set_xlim(-0.1, 1.05); ax.set_ylim(-0.1, 1.05); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.tick_params(labelsize=5, length=2)
    ax.set_xlabel(r"$\mathcal{F}^{(i)}$", fontsize=6, labelpad=0)
    if j == 0:
        ax.set_ylabel(r"$\mathcal{M}^{(i)}$", fontsize=6, labelpad=0)
    ax.set_title(rel, fontsize=6.5, color=[GREEN, PURPLE, RED][j])
fig.savefig(OUT / "l3_similarity.pdf"); plt.close(fig)


# ---- 3. SSD vs MI while sliding a multi-modal image ----------------------------
def mutual_info(a, b, bins=16):
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[[-0.1, 1.1], [-0.1, 1.1]])
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    H = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    return H(px) + H(py) - H(p[nz])


MR = render(MR_VALS)
shifts = np.arange(-12, 13)
ssd_c = [ssd(F, ndi.shift(MR, (0, s), order=1)) for s in shifts]
mi_c = [mutual_info(F, ndi.shift(MR, (0, s), order=1)) for s in shifts]
fig, axs = plt.subplots(1, 3, figsize=(W, 1.25), gridspec_kw=dict(wspace=0.45, width_ratios=[0.8, 1, 1]))
axs[0].imshow(overlay(F, MR)); clean(axs[0], "CT vs MRI,\naligned")
axs[1].plot(shifts, ssd_c, color=RED, lw=1)
axs[1].set_title("SSD vs shift: worst\nexactly at 0 (wrong)", fontsize=6, color=RED)
axs[2].plot(shifts, mi_c, color=GREEN, lw=1)
axs[2].set_title("MI vs shift: max\nat 0 (correct)", fontsize=6, color=GREEN)
for ax in axs[1:]:
    ax.axvline(0, color="#999", lw=0.5, ls="--"); ax.tick_params(labelsize=5)
    ax.set_xlabel("shift (px)", fontsize=6, labelpad=1); ax.set_yticks([])
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.savefig(OUT / "l3_ssd_vs_mi.pdf"); plt.close(fig)
print("SSD argmin shift:", shifts[int(np.argmin(ssd_c))], " MI argmax shift:", shifts[int(np.argmax(mi_c))])

# ---- 4. affine building blocks ---------------------------------------------------
shape = np.array([[0, 0], [1, 0], [1, 0.25], [0.3, 0.25], [0.3, 0.5], [0.8, 0.5], [0.8, 0.75],
                  [0.3, 0.75], [0.3, 1.2], [0, 1.2]]) - [0.5, 0.6]  # a letter "F": shows orientation
a = np.deg2rad(25)
mats = [("identity", np.eye(3), "$A=I$"),
        ("translation", np.array([[1, 0, 0.5], [0, 1, 0.3], [0, 0, 1]]), "$t_x,t_y$ in\nlast column"),
        ("scaling", np.diag([1.5, 0.7, 1]), r"diag$(a,b,1)$" "\n" "= voxel size"),
        ("rotation", np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]),
         r"$\cos\alpha,\ \pm\sin\alpha$"),
        ("shear", np.array([[1, 0.6, 0], [0, 1, 0], [0, 0, 1]]), "off-diagonal\n$c_x$")]
fig, axs = plt.subplots(1, 5, figsize=(W, 1.25), gridspec_kw=dict(wspace=0.1))
g = np.linspace(-1, 1, 5)
for ax, (name, A, txt) in zip(axs, mats):
    for v in g:  # transformed grid lines
        for line in (np.stack([np.full(20, v), np.linspace(-1, 1, 20)], 1),
                     np.stack([np.linspace(-1, 1, 20), np.full(20, v)], 1)):
            p = (A @ np.c_[line, np.ones(len(line))].T).T
            ax.plot(p[:, 0], p[:, 1], color="#bbb", lw=0.4)
    ax.add_patch(Polygon(shape, fill=False, ec="#999", lw=0.5, ls=":"))
    p = (A @ np.c_[shape, np.ones(len(shape))].T).T
    ax.add_patch(Polygon(p[:, :2], fc="#a9bfe8", ec=BLUE, lw=0.7))
    ax.set_xlim(-1.6, 1.8); ax.set_ylim(-1.5, 1.7); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(name, fontsize=6.5)
    ax.text(0.1, -1.65, txt, ha="center", va="top", fontsize=5.3, color=BLUE)
fig.savefig(OUT / "l3_affine.pdf"); plt.close(fig)

# ---- 5. deformation field, smoothness and folds ----------------------------------
n = 24
Y, X = np.mgrid[:n, :n].astype(float)


def bump(amp, sigma):
    ux = amp * np.exp(-((X - 12) ** 2 + (Y - 12) ** 2) / (2 * sigma ** 2))
    return ux, 0.4 * ux


def jac_det(ux, uy):
    duxdy, duxdx = np.gradient(ux)
    duydy, duydx = np.gradient(uy)
    return (1 + duxdx) * (1 + duydy) - duxdy * duydx


fig, axs = plt.subplots(1, 3, figsize=(W, 1.5), gridspec_kw=dict(wspace=0.1))
ux, uy = bump(4, 5)
s = slice(None, None, 2)
axs[0].quiver(X[s, s], Y[s, s], ux[s, s], -uy[s, s], color=BLUE, scale=40, width=0.008)
axs[0].set_title("deformation field:\none arrow per voxel", fontsize=6.5)
for ax, (amp, sig, title, col) in zip(axs[1:], [(4, 5, "smooth: $\\det J>0$", GREEN),
                                               (9, 2.2, "fold: $\\det J\\leq0$", RED)]):
    ux, uy = bump(amp, sig)
    J = jac_det(ux, uy)
    ax.imshow(J, cmap="RdBu", vmin=-1, vmax=3, extent=(-0.5, n - 0.5, n - 0.5, -0.5), alpha=0.35)
    for i in range(0, n, 2):  # warped grid lines
        ax.plot(X[i] + ux[i], Y[i] + uy[i], color="k", lw=0.4)
        ax.plot(X[:, i] + ux[:, i], Y[:, i] + uy[:, i], color="k", lw=0.4)
    if J.min() <= 0:
        ax.contour(J, [0], colors=RED, linewidths=0.9)
    ax.set_title(title + f"\n(min $\\det J$ = {J.min():.1f})", fontsize=6.5, color=col)
for ax in axs:
    ax.set_xlim(-1, n); ax.set_ylim(n, -1); ax.set_aspect("equal"); clean(ax)
fig.savefig(OUT / "l3_deformation.pdf"); plt.close(fig)

# ---- 6. pyramid: coarse to fine ---------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(W, 1.25), gridspec_kw=dict(wspace=0.15))
for ax, f, txt in zip(axs, [8, 4, 1], ["1/8 res.: rigid\n(rough position)", "1/4 res.: affine\n(size, shear)",
                                         "full res.: elastic\n(local detail)"]):
    small = M[::f, ::f]
    ax.imshow(small, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    clean(ax, txt, fontsize=6.5)
fig.savefig(OUT / "l3_pyramid.pdf"); plt.close(fig)

# ---- 7. VoxelMorph schematic -------------------------------------------------------
fig, ax = plt.subplots(figsize=(W, 1.6))


def box(x, y, w, h, text, fc, ec="k", fs=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arr(x0, y0, x1, y1, col="k"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7))


box(0, 2.2, 1.3, 0.7, r"fixed $\mathcal{F}$", "#eeeeee")
box(0, 1.1, 1.3, 0.7, r"moving $\mathcal{M}$", "#eeeeee")
box(1.95, 1.4, 1.75, 1.2, "U-Net\n2 in-channels\n3 out-channels\nno softmax", "#e8eefb", BLUE, 5.8)
box(4.2, 1.65, 1.7, 0.7, "displacement\nfield $t_\\theta$", "#e8eefb", BLUE, 5.8)
box(4.2, 0.2, 1.7, 0.75, "spatial\ntransformer", "#fff2cc", "#b08000", 5.8)
box(6.55, 0.2, 1.8, 0.75, r"warped $\mathcal{T}(\mathcal{M})$", "#eeeeee")
box(6.3, 1.5, 2.5, 1.1, "loss =\nsimilarity($\\mathcal{F}$, $\\mathcal{T}(\\mathcal{M})$)\n+ $\\lambda\\,$smoothness",
    "#e9f5e9", GREEN, 5.8)
arr(1.35, 2.55, 2.0, 2.2); arr(1.35, 1.45, 2.0, 1.8)
arr(3.65, 2.0, 4.2, 2.0); arr(5.05, 1.6, 5.05, 1.0)
arr(1.35, 1.3, 4.2, 0.6); arr(5.95, 0.57, 6.55, 0.57)
arr(7.45, 1.0, 7.45, 1.45); arr(1.35, 2.75, 6.3, 2.45)
ax.text(0.0, 0.1, "no labels needed:\nthe loss only\ncompares images", fontsize=5.6, color=GREEN, ha="left")
ax.set_xlim(-0.1, 8.95); ax.set_ylim(-0.05, 3.05); ax.axis("off")
fig.savefig(OUT / "l3_voxelmorph.pdf"); plt.close(fig)
print("written to", OUT)
