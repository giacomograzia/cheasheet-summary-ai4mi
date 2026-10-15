"""Figures for the L4 study notes (segmentation in practice).

Run:  python make_l4_figures.py   (writes l4_*.pdf into ../figures; needs numpy, scipy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent.parent / "figures"
W = 3.7  # column width in inches

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 7,
    "axes.titlesize": 7, "axes.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, PURPLE, RED, GREEN = "#0037a5", "#690087", "#af0000", "#006928"
rng = np.random.default_rng(1)

# ---- toy organ: an ellipse with a bump, so flips and rotations are visible ---
N = 48
rr, cc = np.mgrid[:N, :N]
organ = ((rr - 26) / 13) ** 2 + ((cc - 22) / 16) ** 2 < 1
organ |= (rr - 13) ** 2 + (cc - 33) ** 2 < 5 ** 2
img = 0.25 + 0.4 * organ + 0.08 * np.sin(cc / 7) + rng.normal(0, 0.05, (N, N))
img = ndi.gaussian_filter(img, 0.7)


def show(ax, im, mask=None, title=None, col=RED, cmap="gray"):
    ax.imshow(im, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    if mask is not None:
        ax.contour(mask.astype(float), [0.5], colors=col, linewidths=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


def farrow(fig, x0, x1, y, text=""):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=8, color="k", lw=0.8))
    if text:
        fig.text((x0 + x1) / 2, y + 0.03, text, ha="center", va="bottom", fontsize=6)


# ---- 1. U-Net schematic + size divisibility ----------------------------------
fig, ax = plt.subplots(figsize=(W, 1.9))
sizes = [104, 52, 26, 13]
ys = [3, 2, 1, 0]
for lvl, (s, yv) in enumerate(zip(sizes, ys)):
    h = 0.55 * s / 104 + 0.25
    for xc, col in [(lvl * 0.9, "#a9bfe8"), (6.3 - lvl * 0.9, "#b9e0b9")]:
        if lvl == 3 and col == "#b9e0b9":
            continue
        ax.add_patch(Rectangle((xc, yv - h / 2), 0.5, h, fc=col, ec="k", lw=0.5))
        ax.text(xc + 0.25, yv - h / 2 - 0.08, str(s), ha="center", va="top", fontsize=5.5)
    if lvl < 3:  # pooling down, upsampling up, skip across
        x_enc, x_dec = lvl * 0.9, 6.3 - lvl * 0.9
        ax.annotate("", (x_enc + 0.9 + 0.1, ys[lvl + 1] + 0.15), (x_enc + 0.45, yv - 0.2),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.7))
        ax.annotate("", (x_dec + 0.05, yv - 0.2), (x_dec - 0.9 + 0.45, ys[lvl + 1] + 0.15),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.7))
        ax.annotate("", (x_dec - 0.05, yv + 0.12), (x_enc + 0.55, yv + 0.12),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.6, ls="--"))
ax.text(0.25, 3.62, "input\nimage", ha="center", fontsize=6)
ax.text(6.55, 3.62, "output\nmask", ha="center", fontsize=6)
ax.text(3.4, 3.0, "skip connection: copy fine detail", ha="center", va="top", fontsize=5.5, color=BLUE)
ax.text(1.0, 0.05, "encoder\npool ÷2", ha="center", fontsize=5.5, color=RED)
ax.text(5.5, 0.05, "decoder\nupsample ×2", ha="center", fontsize=5.5, color=GREEN)
ax.text(7.4, 2.4, "3 poolings ⇒ size must be\ndivisible by $2^3=8$", fontsize=6, va="center")
ax.text(7.4, 1.4, r"$100\to50\to25\to12.5$  (not whole!)", fontsize=6, color=RED, va="center")
ax.text(7.4, 0.8, r"pad to 104: $104\to52\to26\to13$  (fine)", fontsize=6, color=GREEN, va="center")
ax.set_xlim(-0.2, 11.2); ax.set_ylim(-0.55, 4.1); ax.axis("off")
fig.savefig(OUT / "l4_unet.pdf"); plt.close(fig)

# ---- 2a. isotropic vs anisotropic voxels, and the fix (resampling) ---------
F = 20.0  # field of view in mm (x horizontal, z vertical: a side view through the scan)


def sample(dx, dz):
    xs, zs = np.arange(0, F, dx) + dx / 2, np.arange(0, F, dz) + dz / 2
    Zg, Xg = np.meshgrid(zs, xs, indexing="ij")
    return ((Zg - 10) ** 2 + (Xg - 10) ** 2 < 7 ** 2).astype(float)  # round organ, radius 7 mm


def mm_panel(ax, m, dx, dz, title):
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1.3, extent=(0, F, F, 0), interpolation="nearest")
    for v in np.arange(0, F + 1e-6, dx): ax.axvline(v, color="#999", lw=0.25)
    for v in np.arange(0, F + 1e-6, dz): ax.axhline(v, color="#999", lw=0.25)
    ax.add_patch(Circle((10, 10), 7, fill=False, ec=RED, lw=0.6, ls="--"))
    ax.set_xlim(0, F); ax.set_ylim(F, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=6)


iso_m, ani_m = sample(1, 1), sample(1, 2.5)  # 20 x 20 and 8 x 20 voxels
fixed = ndi.zoom(ani_m, (2.5, 1), order=0)  # nearest neighbour for a label map
fig, axs = plt.subplots(1, 4, figsize=(W, 1.45), gridspec_kw=dict(wspace=0.2))
mm_panel(axs[0], iso_m, 1, 1, "isotropic\n" r"$1\times1\times1$ mm" "\n" r"$20\times20$ voxels")
mm_panel(axs[1], ani_m, 1, 2.5, "anisotropic\n" r"$1\times1\times2.5$ mm" "\n" r"$20\times8$ voxels")
axs[2].imshow(ani_m, cmap="Blues", vmin=0, vmax=1.3, interpolation="nearest")
axs[2].set_xticks([]); axs[2].set_yticks([])
axs[2].set_title("network sees\nthe $20\\times8$ array:\nsquashed", fontsize=6, color=RED)
mm_panel(axs[3], fixed, 1, 1, "fix: resample\nto $1$ mm" "\n" r"$20\times20$ again", )
axs[3].title.set_color(GREEN)
axs[0].set_ylabel("$z$ (slices)", fontsize=6); axs[0].set_xlabel("$x$ (in-plane)", fontsize=6)
fig.savefig(OUT / "l4_iso_aniso.pdf"); plt.close(fig)

# ---- 2a'. what resampling does to the stack of slices ------------------------
z_old = np.arange(0, F, 2.5) + 1.25  # 8 measured slice centres
z_new = np.arange(0, F, 1.0) + 0.5   # 20 resampled slice centres
prof = lambda z: 0.15 + 0.7 * (1 / (1 + np.exp(-(z - 6.5) / 0.7)) - 1 / (1 + np.exp(-(z - 14) / 0.7)))
fig, axs = plt.subplots(1, 3, figsize=(W, 1.5), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1, 1.6]))
val_old = prof(z_old)
for ax, dz, title, val in [(axs[0], 2.5, "measured\n8 slices $\\times$ 2.5 mm", val_old),
                           (axs[1], 1.0, "resampled\n20 slices $\\times$ 1 mm", np.interp(z_new, z_old, val_old))]:
    for z0, v in zip(np.arange(0, F, dz), val):
        ax.add_patch(Rectangle((0, z0), 6, dz, fc=plt.cm.Greys(0.15 + 0.6 * v), ec="k", lw=0.3))
    for zc in z_old:  # measured centres, also marked on the new stack
        ax.plot([6.3, 7.3], [zc, zc], color=PURPLE, lw=0.8)
    ax.set_xlim(0, 7.6); ax.set_ylim(F, 0); ax.axis("off"); ax.set_title(title, fontsize=6)
axs[0].text(-0.5, F / 2, "$z$ (20 mm of body)", rotation=90, ha="right", va="center", fontsize=5.5)
axs[1].text(3.8, F + 0.6, "same 20 mm,\nthinner slices", ha="center", va="top", fontsize=5)
axs[0].text(3.8, F + 0.6, "purple ticks =\nmeasured centres", ha="center", va="top", fontsize=5, color=PURPLE)
zz = np.linspace(0, F, 400)
ax = axs[2]
ax.plot(zz, prof(zz), color="#bbb", lw=0.8, label="true tissue (never stored)")
ax.plot(z_old, prof(z_old), "o", color=PURPLE, ms=2.5, label="measured slices")
ax.plot(z_new, np.interp(z_new, z_old, prof(z_old)), "-", color=BLUE, lw=0.7, marker=".", ms=2.5,
        label="image: linear interp.")
lab_old = (prof(z_old) > 0.5).astype(float)
nn = lab_old[np.abs(z_new[:, None] - z_old[None]).argmin(1)]
ax.step(z_new, 0.02 + 0.9 * nn, where="mid", color=GREEN, lw=0.7, label="label: nearest (0/1)")
ax.set_xlabel("$z$ (mm)", fontsize=6); ax.set_ylabel("intensity", fontsize=6)
ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yticks([]); ax.tick_params(labelsize=5, length=2)
ax.set_title("how new slices get values", fontsize=6)
ax.legend(fontsize=4.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
fig.savefig(OUT / "l4_resample_stack.pdf"); plt.close(fig)

# ---- 2b. 2.5D input -----------------------------------------------------------
fig = plt.figure(figsize=(W, 1.3))
ax = fig.add_axes([0.0, 0.0, 0.45, 0.85])
for j, name in enumerate([r"$z{-}1$", r"$z$", r"$z{+}1$"]):
    o = 7 * j
    sl = ndi.shift(img, (0, (j - 1) * 2), order=1, mode="nearest")
    ax.imshow(sl, cmap="gray", vmin=0, vmax=1, extent=(o, o + N, o + N, o), zorder=3 - j)
    ax.add_patch(Rectangle((o, o), N, N, fill=False, ec=BLUE if j == 1 else "k",
                           lw=1.0 if j == 1 else 0.5, zorder=3 - j))
    ax.text(o + N + 1, o + 3, name, fontsize=6, va="top")
ax.set_xlim(-1, 14 + N + 9); ax.set_ylim(14 + N + 1, -1); ax.axis("off")
ax.set_title("3 neighbouring slices", fontsize=6.5)
farrow(fig, 0.47, 0.56, 0.45, "3 channels")
fig.text(0.6, 0.45, "2D U-Net", fontsize=7, va="center", bbox=dict(fc="#e8eefb", ec=BLUE, lw=0.6))
farrow(fig, 0.75, 0.84, 0.45)
ax2 = fig.add_axes([0.85, 0.2, 0.17, 0.5])
show(ax2, img, organ)
ax2.set_title("mask of\nslice $z$ only", fontsize=6)
fig.savefig(OUT / "l4_25d.pdf"); plt.close(fig)

# ---- 3. augmentation --------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(W, 1.2), gridspec_kw=dict(wspace=0.08))
show(axs[0], img, organ, "original $x$, $y$")
show(axs[1], img[:, ::-1], organ[:, ::-1], "flip: $x$ and $y$")
rot_i = ndi.rotate(img, 30, reshape=False, order=1, mode="nearest")
rot_m = ndi.rotate(organ.astype(float), 30, reshape=False, order=0) > 0.5
show(axs[2], rot_i, rot_m, "rotate: $x$ and $y$")
show(axs[3], np.clip(img + rng.normal(0, 0.12, img.shape), 0, 1), organ, "noise: only $x$")
for ax in axs[1:3]:
    for sp in ax.spines.values(): sp.set_color(BLUE); sp.set_linewidth(1)
for sp in axs[3].spines.values(): sp.set_color(PURPLE); sp.set_linewidth(1)
fig.savefig(OUT / "l4_augment.pdf"); plt.close(fig)

# ---- 4. sub-patching: most patches miss a small class ------------------------
fig, axs = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw=dict(wspace=0.25, width_ratios=[1, 1.15]))
tum = (rr - 30) ** 2 + (cc - 17) ** 2 < 3 ** 2
show(axs[0], img, None, "4×4 patches of $12\\times12$")
axs[0].contour(tum.astype(float), [0.5], colors=RED, linewidths=0.8)
P = 12
for i in range(4):
    for j in range(4):
        has = tum[i * P:(i + 1) * P, j * P:(j + 1) * P].any()
        axs[0].add_patch(Rectangle((j * P - 0.5, i * P - 0.5), P, P, fill=not has,
                                   fc=(1, 1, 1, 0.55), ec=RED if has else "w", lw=1.2 if has else 0.5))
n_empty = sum(not tum[i * P:(i + 1) * P, j * P:(j + 1) * P].any() for i in range(4) for j in range(4))
axs[0].text(N / 2, N + 4, f"white = no tumour ({n_empty} of 16)", ha="center", va="top", fontsize=6, color=RED)
ax = axs[1]
ax.axis("off")
ax.text(0, 0.95, "in a patch without tumour:", fontsize=6.5, va="top", weight="bold")
ax.text(0, 0.78, r"label has 0 tumour voxels" "\n" r"$\Rightarrow$ $I=U=0$ $\Rightarrow$ Dice $=0/0$ (NaN)",
        fontsize=6.5, va="top", color=RED)
ax.text(0, 0.47, r"fix: $\frac{2I+\epsilon}{U+\epsilon}$, or skip/handle" "\n" "empty classes in the loss",
        fontsize=6.5, va="top", color=GREEN)
ax.text(0, 0.17, "the network also never sees\nthe whole scan at once", fontsize=6.5, va="top")
fig.savefig(OUT / "l4_patches.pdf"); plt.close(fig)

# ---- 5. post-processing: largest connected component -------------------------
blob = lambda r0, c0, rad: (rr - r0) ** 2 + (cc - c0) ** 2 < rad ** 2
pred_a = organ | blob(6, 6, 2.2) | blob(42, 40, 2) | blob(40, 8, 1.5)
truth_b = blob(14, 14, 6) | blob(32, 34, 7) | blob(36, 12, 5)
pred_b = truth_b.copy()


def lcc(m):
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1), lab, n


fig, axs = plt.subplots(2, 3, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.08, hspace=0.35))
from matplotlib.colors import ListedColormap
comp_cmap = ListedColormap(["white", "#e06060", "#7fbf7f", "#7f9fdf", "#e0b050", "#b07fd0"])
for row, (pred, ok) in enumerate([(pred_a, True), (pred_b, False)]):
    keep, lab, n = lcc(pred)
    axs[row, 0].imshow(pred, cmap="Greys", vmin=0, vmax=1.4)
    axs[row, 1].imshow(lab, cmap=comp_cmap, vmin=0, vmax=5, interpolation="nearest")
    axs[row, 2].imshow(keep, cmap="Greys", vmin=0, vmax=1.4)
    for ax in axs[row]:
        ax.set_xticks([]); ax.set_yticks([])
    axs[row, 1].set_title(f"components: {n}")
    axs[row, 2].set_title("keep largest: " + ("good" if ok else "wrong!"), color=GREEN if ok else RED)
axs[0, 0].set_title("organ + speckles")
axs[1, 0].set_title("3 real lesions")
fig.text(0.5, 0.02, "fine when exactly one object is expected; deletes real lesions otherwise",
         ha="center", fontsize=6, color=RED)
fig.savefig(OUT / "l4_lcc.pdf"); plt.close(fig)

# ---- 6. DenseCRF idea: snap a blurry prediction to image edges ---------------
edge_img = 0.25 + 0.45 * organ + rng.normal(0, 0.04, (N, N))
soft = ndi.gaussian_filter(ndi.shift(organ.astype(float), (2, 3), order=0), 3.5)
fig, axs = plt.subplots(1, 3, figsize=(W, 1.3), gridspec_kw=dict(wspace=0.08))
show(axs[0], edge_img, None, "image:\nclear edges")
axs[1].imshow(edge_img, cmap="gray", vmin=0, vmax=1)
axs[1].imshow(np.ma.masked_less(soft, 0.05), cmap="Reds", alpha=0.6, vmin=0, vmax=1)
axs[1].contour(soft, [0.5], colors=RED, linewidths=0.8)
axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].set_title("network output:\nblurry, off the edge")
show(axs[2], edge_img, organ, "after CRF:\nfollows the edges", col=GREEN)
fig.savefig(OUT / "l4_crf.pdf"); plt.close(fig)

# ---- 7. weak annotations ----------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(W, 1.0), gridspec_kw=dict(wspace=0.06))
show(axs[0], img, organ, "full mask")
show(axs[1], img, None, "image tag")
axs[1].text(N / 2, N / 2, "“organ\npresent”", ha="center", va="center", fontsize=6, color="w",
            bbox=dict(fc=BLUE, ec="none", pad=1.5))
show(axs[2], img, None, "dots")
axs[2].plot([22, 30], [28, 22], "o", color=GREEN, ms=3)
axs[2].plot([5, 42], [42, 6], "o", color=BLUE, ms=3)
show(axs[3], img, None, "scribbles")
t = np.linspace(0, 1, 30)
axs[3].plot(12 + 18 * t, 26 + 3 * np.sin(6 * t), color=GREEN, lw=1.2)
axs[3].plot(3 + 40 * t, 44 - 2 * np.sin(5 * t), color=BLUE, lw=1.2)
show(axs[4], img, None, "bounding box")
ys_, xs_ = np.nonzero(organ)
axs[4].add_patch(Rectangle((xs_.min() - 0.5, ys_.min() - 0.5), np.ptp(xs_) + 1, np.ptp(ys_) + 1,
                           fill=False, ec=GREEN, lw=1))
fig.savefig(OUT / "l4_weak.pdf"); plt.close(fig)

# ---- 8. boundary loss: signed distance map -----------------------------------
phi = ndi.distance_transform_edt(~organ) - ndi.distance_transform_edt(organ)
pred_bl = ndi.shift(organ.astype(float), (0, 6), order=0) > 0.5
fig, axs = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(wspace=0.3))
lim = np.abs(phi).max()
im = axs[0].imshow(phi, cmap="RdBu_r", vmin=-lim, vmax=lim)
axs[0].contour(organ.astype(float), [0.5], colors="k", linewidths=0.8)
axs[0].set_xticks([]); axs[0].set_yticks([])
axs[0].set_title(r"$\phi$ = signed distance to label edge" "\n" "blue $<0$ inside, red $>0$ outside")
cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5)
axs[1].imshow(phi, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.35)
axs[1].contour(organ.astype(float), [0.5], colors="k", linewidths=0.8)
axs[1].contour(pred_bl.astype(float), [0.5], colors=PURPLE, linewidths=0.9, linestyles="--")
axs[1].set_xticks([]); axs[1].set_yticks([])
axs[1].text(46, 30, r"$s\,\phi>0$: penalised," "\nmore the further out", fontsize=5.8, color=RED,
            ha="right", va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
axs[1].set_title(r"prediction $s$ (dashed) leaks right" "\n" r"loss $=\sum_i s^{(i)}\phi^{(i)}$")
fig.savefig(OUT / "l4_boundary.pdf"); plt.close(fig)

# ---- 9. hard structure: the esophagus in a thoracic CT slice ----------------
H = 64
r2, c2 = np.mgrid[:H, :H]
ct = np.full((H, H), 0.05)
body = ((r2 - 32) / 24) ** 2 + ((c2 - 32) / 30) ** 2 < 1
ct[body] = 0.45
lungs = (((r2 - 30) / 14) ** 2 + ((c2 - 18) / 9) ** 2 < 1) | (((r2 - 30) / 14) ** 2 + ((c2 - 46) / 9) ** 2 < 1)
ct[lungs] = 0.12
heart = ((r2 - 26) / 9) ** 2 + ((c2 - 34) / 10) ** 2 < 1
aorta = (r2 - 40) ** 2 + (c2 - 36) ** 2 < 4 ** 2
trachea = (r2 - 36) ** 2 + (c2 - 28) ** 2 < 2.6 ** 2
esoph = (r2 - 41) ** 2 + (c2 - 30) ** 2 < 2.0 ** 2
ct[heart] = 0.62; ct[aorta] = 0.7; ct[trachea] = 0.08; ct[esoph] = 0.49
ct = ndi.gaussian_filter(ct + rng.normal(0, 0.03, ct.shape), 0.6)
fig, axs = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(wspace=0.1))
show(axs[0], ct, None, "toy thoracic CT slice")
axs[0].annotate("esophagus?", (30, 41), (6, 60), color=RED, fontsize=6,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
lab = np.zeros((H, H)); lab[heart] = 1; lab[aorta] = 2; lab[trachea] = 3; lab[esoph] = 4
seg_cmap = ListedColormap(["#eeeeee", "#e06060", "#e0b050", "#7f9fdf", "#7fbf7f"])
axs[1].imshow(lab, cmap=seg_cmap, vmin=0, vmax=4, interpolation="nearest")
axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].set_title("SegTHOR labels")
counts = [heart.sum(), aorta.sum(), trachea.sum(), esoph.sum()]
for k, (name, n) in enumerate(zip(["heart", "aorta", "trachea", "esophagus"], counts)):
    axs[1].text(H + 2, 10 + 12 * k, f"{name}: {n} px", fontsize=6, va="center",
                color=seg_cmap(k + 1), weight="bold")
fig.savefig(OUT / "l4_esophagus.pdf"); plt.close(fig)
# ---- 10. notation for a CT volume and a 2.5D input ---------------------------
from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize=(W, 1.75))
d = np.array([0.45, 0.3])  # depth direction (z, slices) drawn as an oblique offset
n_draw = 14  # slices drawn (stand-in for 200)
for j in range(n_draw):
    o = d * j / n_draw * 4
    hl = j in (6, 7, 8)
    ax.add_patch(Polygon([(0, 0) + o, (2, 0) + o, (2, 2) + o, (0, 2) + o],
                         fc="#a9bfe8" if hl else "#dcdcdc", ec=BLUE if hl else "#888",
                         lw=0.8 if hl else 0.4, zorder=n_draw - j))
ax.text(1, -0.12, "512 columns", ha="center", va="top", fontsize=6)
ax.text(-0.08, 1, "512\nrows", ha="right", va="center", fontsize=6)
ax.annotate("", (2 + 4 * d[0] + 0.05, 4 * d[1]), (2.05, 0),
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.text(2 + 2 * d[0] + 0.12, 2 * d[1] - 0.12, "200 slices ($z$)", fontsize=6, va="top")
ax.text(3.95, 2.75, r"slices $z{-}1, z, z{+}1$", fontsize=6, color=BLUE, ha="left")
ax.text(1.9, 3.35, "full CT volume\n" r"$\Omega\subset\mathbb{R}^3$, $|\Omega|=512\cdot512\cdot200$" "\n"
        r"$x$: $M=1$ (HU), shape $1\times512\times512\times200$", ha="center", va="bottom", fontsize=6)

ax.annotate("", (5.25, 1.2), (4.3, 1.2), arrowprops=dict(arrowstyle="-|>", lw=0.8))
for j, name in enumerate([r"$z{-}1$", r"$z$", r"$z{+}1$"]):
    o = np.array([0.18, 0.18]) * j
    ax.add_patch(Rectangle((5.4, 0.3) + o, 1.6, 1.6, fc="#a9bfe8", ec=BLUE, lw=0.6, zorder=3 - j))
ax.text(6.35, -0.1, "2.5D input $x$\n" r"$M=3$ channels" "\n" r"$3\times512\times512$" "\n" r"$\Omega\subset\mathbb{R}^2$, $|\Omega|=512^2$",
        ha="center", va="top", fontsize=6)

ax.annotate("", (9.0, 1.2), (7.75, 1.2), arrowprops=dict(arrowstyle="-|>", lw=0.8))
cols = ["#e6e6e6", "#e06060", "#e0b050", "#7f9fdf", "#7fbf7f"]
for j, c in enumerate(cols):
    o = np.array([0.12, 0.12]) * j
    ax.add_patch(Rectangle((9.15, 0.3) + o, 1.6, 1.6, fc=c, ec="k", lw=0.4, zorder=6 - j))
ax.text(10.2, -0.1, r"output $s_\theta$ (slice $z$)" "\n" r"$K{+}1=5$ maps" "\n" r"$5\times512\times512$" "\n" "(bg + 4 organs)",
        ha="center", va="top", fontsize=6)
ax.set_xlim(-0.6, 11.6); ax.set_ylim(-1.55, 4.3); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(OUT / "l4_ct_notation.pdf"); plt.close(fig)
print("written to", OUT)
