"""Motion graph: the frame-difference signal, the detection threshold and the
detected page segments (coloured by status). Matplotlib is optional and only
needed for PNG output (the UI draws the same chart with Altair)."""

from __future__ import annotations

from typing import Optional

import numpy as np

# Reference palette (light surface): series / categorical slots and chart ink.
COLORS = {
    "signal": "#2a78d6",     # slot 1 blue
    "keep": "#1baf7a",       # slot 3 aqua
    "duplicate": "#eb6834",  # slot 2 orange
    "blank": "#898781",      # muted
    "deleted": "#898781",
    "threshold": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
STATUS_LABELS = {"keep": "Page", "duplicate": "Duplicate (merged)", "blank": "Blank", "deleted": "Deleted"}


def motion_png(analysis, smoothed: np.ndarray, threshold: float, project, out_path: str,
               width_in: float = 14.0) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return None
    t = analysis.times
    fig, ax = plt.subplots(figsize=(width_in, 3.6), dpi=110)
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])
    seen = set()
    for p in project.pages:
        if p.source != "auto":
            continue
        c = COLORS.get(p.status, COLORS["muted"])
        ax.axvspan(p.t_start, p.t_end, color=c, alpha=0.18, lw=0)
        seen.add(p.status)
    ax.plot(t, np.maximum(smoothed, 1e-3), color=COLORS["signal"], lw=1.2)
    ax.axhline(threshold, color=COLORS["threshold"], lw=1, ls=(0, (4, 3)))
    ax.text(t[-1], threshold, f"  threshold {threshold:.2f}", va="center", ha="left",
            fontsize=8, color=COLORS["ink2"], clip_on=False)
    # Page numbers above kept segments (thinned out when dense).
    kept = [p for p in project.pages if p.included and p.source == "auto"]
    stride = max(1, int(np.ceil(len(kept) / 60)))
    ymax = float(max(np.max(smoothed), threshold) * 3)
    for n, p in enumerate(project.pages):
        if p.included and p.source == "auto":
            num = project.page_number(p.id)
            if num and (num - 1) % stride == 0:
                ax.text((p.t_start + p.t_end) / 2, ymax * 0.55, str(num), ha="center", va="top",
                        fontsize=7, color=COLORS["ink2"])
    ax.set_yscale("log")
    ax.set_ylim(max(1e-3, float(np.percentile(smoothed, 1)) * 0.5), ymax)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)", color=COLORS["ink2"], fontsize=9)
    ax.set_ylabel("Motion (mean abs diff)", color=COLORS["ink2"], fontsize=9)
    ax.set_title("Motion over time - shaded spans are detected pages", loc="left", fontsize=10,
                 color=COLORS["threshold"])
    ax.grid(True, which="major", axis="y", color=COLORS["grid"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    handles = [Patch(color=COLORS[s], alpha=0.35, label=STATUS_LABELS[s]) for s in ("keep", "duplicate", "blank")
               if s in seen]
    if handles:
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8,
                  frameon=False, ncol=len(handles), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
