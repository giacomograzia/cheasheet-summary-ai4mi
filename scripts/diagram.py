"""Small helpers for box-and-arrow diagrams: boxes sized to their text, arrows from border to border.

Set the axes limits (and figure size) before calling textbox, since the text is measured in data units.
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch


def textbox(ax, x, y, text, fc="#f2f2f2", ec="k", fs=5.8, color="k", ls="-", pad=3.5, minw=0.0, minh=0.0):
    """Draw `text` centred at (x, y) inside a rounded box with `pad` points of inner padding.

    Returns the box as (cx, cy, w, h) in data coordinates.
    """
    t = ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=3, linespacing=1.15)
    r = ax.figure.canvas.get_renderer()
    bb = t.get_window_extent(r)
    px = pad * ax.figure.dpi / 72
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0 - px, bb.y0 - px), (bb.x1 + px, bb.y1 + px)])
    w, h = max(x1 - x0, minw), max(y1 - y0, minh)
    rs = 0.15 * min(w, h)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                                fc=fc, ec=ec, lw=0.6, ls=ls, zorder=2))
    return (x, y, w, h)


def side(b, which, frac=0.5):
    """Point on a box side: which in {'top','bottom','left','right'}, frac along that side (0..1)."""
    cx, cy, w, h = b
    if which in ("top", "bottom"):
        return (cx - w / 2 + frac * w, cy + (h / 2 if which == "top" else -h / 2))
    return (cx + (w / 2 if which == "right" else -w / 2), cy - h / 2 + frac * h)


def edge(b, toward):
    """Where the straight line from the centre of box b towards a point leaves the box."""
    cx, cy, w, h = b
    dx, dy = toward[0] - cx, toward[1] - cy
    s = min(w / 2 / abs(dx) if dx else np.inf, h / 2 / abs(dy) if dy else np.inf)
    return (cx + s * dx, cy + s * dy)


def arrow(ax, p0, p1, col="k", lw=0.7):
    """Arrow from point p0 to point p1; the head's tip lands exactly on p1."""
    ax.annotate("", p1, p0, zorder=4, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, shrinkA=0,
                                                       shrinkB=0, mutation_scale=7))


def connect(ax, a, b, col="k"):
    """Straight arrow between two boxes, from border to border along the line joining their centres."""
    arrow(ax, edge(a, b[:2]), edge(b, a[:2]), col)


def path(ax, points, col="k", lw=0.7):
    """Polyline through `points`, with an arrowhead on the last segment."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=col, lw=lw, zorder=4, solid_capstyle="butt")
    arrow(ax, points[-2], points[-1], col, lw)
