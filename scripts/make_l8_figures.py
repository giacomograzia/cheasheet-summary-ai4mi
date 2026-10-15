"""Figures for the L8 study notes (reconstruction; foundation models & domain shift).

Run:  python make_l8_figures.py   (writes l8_*.pdf into ../figures; needs numpy, matplotlib)
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
rng = np.random.default_rng(0)

# ---- toy "brain" phantom: a few ellipses (Shepp-Logan style) ----------------------
N = 128
y, x = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]


def ell(cx, cy, a, b, ang=0):
    t = np.deg2rad(ang)
    xr, yr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t), -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1


img = 0.9 * ell(0, 0, 0.69, 0.92) - 0.6 * ell(0, -0.02, 0.62, 0.87)
img += 0.35 * ell(0.22, 0, 0.11, 0.31, -18) + 0.35 * ell(-0.22, 0, 0.16, 0.41, 18)
img += 0.25 * ell(0, 0.35, 0.21, 0.25) + 0.3 * ell(-0.08, -0.6, 0.05, 0.03) + 0.3 * ell(0.06, -0.6, 0.03, 0.05)
img = np.clip(img, 0, 1)

kspace = np.fft.fftshift(np.fft.fft2(img))  # mu = F^{-1}(x): image -> k-space (cheatsheet notation)


def recon(mask):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace * mask)))


rows = np.arange(N)
m_every2 = np.zeros((N, N)); m_every2[rows % 2 == 0] = 1  # keep every 2nd line: factor 2
m_centre = np.zeros((N, N)); m_centre[N // 2 - 8:N // 2 + 8] = 1  # only the 16 central lines
m_random = np.zeros((N, N)); m_random[rng.random(N) < 0.22] = 1; m_random[N // 2 - 6:N // 2 + 6] = 1

fig, axs = plt.subplots(2, 4, figsize=(W, 2.05), gridspec_kw=dict(wspace=0.06, hspace=0.62))
axs[0, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
axs[0, 0].set_title("image $x$\n(fully sampled)", fontsize=6)
axs[1, 0].imshow(np.log1p(np.abs(kspace)), cmap="magma")
axs[1, 0].set_title(r"k-space $|\mu|$" "\n(centre = coarse)", fontsize=6)
for j, (m, name, col) in enumerate([(m_every2, "every 2nd line\n(2$\\times$ faster)", RED),
                                    (m_centre, "centre only\n(12% of lines)", PURPLE),
                                    (m_random, "centre + random\n(28% of lines)", BLUE)], start=1):
    axs[1, j].imshow(np.log1p(np.abs(kspace)) * m, cmap="magma")
    axs[1, j].set_title(name, fontsize=6, color=col)
    r = recon(m)
    axs[0, j].imshow(r / r.max(), cmap="gray", vmin=0, vmax=1)
for j, t in enumerate(["", "ghost copy\n(aliasing)", "blurry,\nno detail", "noise-like\nartefacts"]):
    if t:
        axs[0, j].set_title(t, fontsize=6)
for ax in axs.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.07, 0.5, "", fontsize=5)
fig.savefig(OUT / "l8_kspace.pdf"); plt.close(fig)

# ---- learned reconstruction training pipeline (+ VORTEX consistency) ----------------
fig, ax = plt.subplots(figsize=(W, 1.45))


def box(x0, y0, w, h, text, fc="#f2f2f2", ec="k", fs=5.0):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=0.6))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs)


def arr(x0, y0, x1, y1, text=None, col="k", dy=0.07):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7))
    if text:
        ax.text((x0 + x1) / 2, max(y0, y1) + dy, text, ha="center", va="bottom", fontsize=5.2, color=col)


xs, w, h = [0, 1.95, 3.9, 5.85, 7.65], 1.5, 0.62
box(xs[0], 1.1, w, h, "fully sampled\nimage $x$")
box(xs[1], 1.1, w, h, "k-space\n" r"$\mu=\mathsf{F}^{-1}(x)$")
box(xs[2], 1.1, w, h, "drop lines:\n" r"$\tilde\mu$", "#fff2cc", "#b08000")
box(xs[3], 1.1, w, h, "network\n" r"$\mathcal{N}_\theta$", "#e8eefb", BLUE)
box(xs[4], 1.1, w, h, "output\n" r"$\tilde x_\theta$")
for x0 in xs[:-1]:
    arr(x0 + w + 0.03, 1.41, x0 + 1.95 - 0.03, 1.41)
ax.annotate("", (xs[4] + w / 2, 1.06), (xs[0] + w / 2, 1.06), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.7,
                                                         connectionstyle="arc3,rad=0.25"))
ax.text(4.6, 0.05, r"$\mathcal{L}_{rec}=\sum_i(\tilde x_\theta^{(i)}-x^{(i)})^2$: compare with the truth we started from",
        ha="center", fontsize=5.6, color=GREEN)
ax.text(4.6, 2.05, "VORTEX: also feed a differently augmented $\\tilde\\mu$ and ask for the same output",
        ha="center", fontsize=5.4, color=PURPLE)
ax.set_xlim(-0.1, 9.25); ax.set_ylim(-0.05, 2.25); ax.axis("off")
fig.savefig(OUT / "l8_pipeline.pdf"); plt.close(fig)

# ---- domain shift and intensity normalisation --------------------------------------
src = np.clip(img + rng.normal(0, 0.04, img.shape), 0, 1)
tgt = np.clip(0.55 * img + 0.3 + rng.normal(0, 0.08, img.shape), 0, 1)  # other scanner: other contrast + noise
body = img > 0.05
z = lambda a: (a - a[body].mean()) / a[body].std()
fig, axs = plt.subplots(1, 4, figsize=(W, 1.25), gridspec_kw=dict(wspace=0.25, width_ratios=[0.8, 0.8, 1.2, 1.2]))
axs[0].imshow(src, cmap="gray", vmin=0, vmax=1); axs[0].set_title("source\n(scanner 1)", fontsize=6)
axs[1].imshow(tgt, cmap="gray", vmin=0, vmax=1); axs[1].set_title("target\n(scanner 2)", fontsize=6, color=RED)
for ax in axs[:2]:
    ax.set_xticks([]); ax.set_yticks([])
for ax, (a, b, title) in zip(axs[2:], [(src[body], tgt[body], "raw intensities:\nshifted"),
                                        (z(src)[body], z(tgt)[body], "after z-score\nper image: aligned")]):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ax.hist(a, bins=40, range=(lo, hi), color="#555", alpha=0.6, label="source")
    ax.hist(b, bins=40, range=(lo, hi), color=RED, alpha=0.5, label="target")
    ax.set_yticks([]); ax.tick_params(labelsize=5); ax.set_title(title, fontsize=6)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
axs[2].legend(fontsize=4.8, frameon=False, loc="upper right")
fig.savefig(OUT / "l8_domain_shift.pdf"); plt.close(fig)
print("written to", OUT)
