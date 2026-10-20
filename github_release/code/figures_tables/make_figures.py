"""make_figures.py - regenerate all data figures (Figures 3-6) from the result files.

Usage: python make_figures.py --sim results_revision --bc results_blockchain --out figures
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyArrowPatch

ap = argparse.ArgumentParser(description="Regenerate all data figures of the manuscript from the result files.")
ap.add_argument("--sim", required=True, help="folder with the simulation results (results_revision)")
ap.add_argument("--bc", required=True, help="folder with the blockchain results (results_blockchain)")
ap.add_argument("--out", required=True, help="output folder for the figure files")
args = ap.parse_args()
SIM, BC, OUT = Path(args.sim), Path(args.bc), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = json.load(open(SIM / "parameters_used.json"))

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "lines.linewidth": 1.9,
    "axes.linewidth": 0.9, "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.6,
    "legend.framealpha": 0.95, "legend.edgecolor": "#b0b0b0", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# ---------------------------------------------------------------- palette
BLUE, MAGENTA, RED, ORANGE, LIME = "#0039e6", "#d000c8", "#e60000", "#ff8c00", "#2fc22f"
VIOLET, SLATE, BROWN, GOLD = "#7b2fe0", "#7d8fa8", "#9c4a12", "#ffc400"
SIENNA, ORCHID, DARKGRAY = "#b5651d", "#a83dd6", "#4a4a4a"
ATTACK = {"byzantine": BLUE, "fdi": MAGENTA, "disruption": RED, "adaptive": ORANGE, "none": LIME}
ATK_LAB = {"byzantine": "Byzantine (equivocating)", "fdi": "FDI", "disruption": "Disruption (equivocating)",
           "adaptive": "Adaptive", "none": "No attack",
           "byzantine_single": "Byzantine (single value)", "disruption_single": "Disruption (single value)"}
DEF_COL = {"no_defense": RED, "ledger_only": BLUE, "signed_echo_only": DARKGRAY, "screening_only": ORANGE,
           "ledger+screening": LIME, "signed_echo+screening": SIENNA, "wmsr_baseline": ORCHID}
DEF_LAB = {"no_defense": "No defense", "ledger_only": "Blockchain only", "signed_echo_only": "Signed exchange only",
           "screening_only": "Screening only", "ledger+screening": "Proposed (ledger + screening)",
           "signed_echo+screening": "Signed exchange + screening", "wmsr_baseline": "W-MSR baseline"}
REF_PRIOR = dict(color=VIOLET, ls=(0, (1.2, 1.6)), lw=1.6, marker="x", ms=5, mew=1.3, label="GNSS prior only")
REF_ORACLE = dict(color=DARKGRAY, ls=(0, (6, 2, 1.5, 2)), lw=1.6, marker="+", ms=7, mew=1.4,
                  label="Ideal WLS (compromised links removed)")
ERR = dict(ecolor=SLATE, elinewidth=1.0, capsize=2.5)


MK = dict(mfc="white", mew=1.3, ms=4.5)        # open markers, drawn sparsely along each line


def marked(ax, x, y, step, offset=0, **kw):
    """Plot a line with markers every `step` points, starting at `offset` (avoids marker overlap)."""
    idx = [i for i in range(len(x)) if i >= offset and (i - offset) % step == 0]
    return ax.plot(np.asarray(x), np.asarray(y), markevery=idx, **{**MK, **kw})


def ref_line(ax, y, x0, x1, style, offset=0, n=200, step=25):
    """Horizontal reference line with sparse markers."""
    xs = np.linspace(x0, x1, n)
    ax.plot(xs, np.full(n, y), markevery=list(range(offset, n, step)), **style)


def panel_label(ax, text, pad=30, size=None):
    """Panel label centred below the axes (below tick labels and x-axis label)."""
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -pad), textcoords="offset points",
                ha="center", va="top", fontsize=size or plt.rcParams["axes.titlesize"])


def save(fig, name):
    # fixed metadata so that repeated runs give byte-identical PDF files
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    print("wrote", name)


# ================================================================ Figure 3: six-UAV convergence
s6 = pd.read_csv(SIM / "six_uav_summary.csv"); ref6 = s6.iloc[0]
tr = pd.read_csv(SIM / "six_uav_traces.csv")
MARK = {"byzantine": "o", "disruption": "s", "fdi": "^", "adaptive": "D", "none": "v"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, dname, title, xmax in [(axes[0], "no_defense", "(a) No defense", 400),
                               (axes[1], "ledger+screening", "(b) Proposed defense", 140)]:
    step = xmax // 10
    for k, atk in enumerate(["byzantine", "disruption", "fdi", "adaptive", "none"]):
        t = tr[(tr.attack_name == atk) & (tr.defense_name == dname)]
        marked(ax, t.iteration, t.rmse_honest_m, step, offset=(k * step) // 5, color=ATTACK[atk], label=ATK_LAB[atk],
               marker=MARK[atk], ls="--" if atk == "none" else "-", lw=2.0 if atk == "none" else 1.8)
    ref_line(ax, ref6.ref_prior_only_rmse, 0, xmax, REF_PRIOR, offset=6)
    ref_line(ax, ref6.ref_oracle_wls_rmse, 0, xmax, REF_ORACLE, offset=18)
    ax.set_ylim(bottom=0); ax.set_xlabel("ADMM iteration"); panel_label(ax, title)
    ax.set_xlim(0, xmax)
for ax in axes:
    ax.set_ylabel("Honest-UAV RMSE (m)")
h, l = axes[1].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
save(fig, "fig_six_convergence")

# --------------------------------------------------------------- automatic label placement (Figure 4)
DIRS = [(0, 1, "center", "bottom"), (0.72, 0.72, "left", "bottom"), (1, 0, "left", "center"),
        (-0.72, 0.72, "right", "bottom"), (0.72, -0.72, "left", "top"), (-1, 0, "right", "center"),
        (0, -1, "center", "top"), (-0.72, -0.72, "right", "top")]


def _rect_point_dist(bb, p):
    dx = max(bb.x0 - p[0], 0, p[0] - bb.x1); dy = max(bb.y0 - p[1], 0, p[1] - bb.y1)
    return np.hypot(dx, dy)


def place_labels(fig, ax):
    """Place each U_i label as close as possible to its marker without overlapping any marker, box,
    arrow, link, error line, or previously placed label (distances in points, independent of axis scale)."""
    fig.canvas.draw(); R = fig.canvas.get_renderer(); px = fig.dpi / 72.0; T = ax.transData
    obs = ax._uav_obstacles; axbb = ax.get_window_extent(R)
    circles = [(T.transform(p), r * px, w) for p, r, w in obs["circles"]]
    squares = [(T.transform(p), h * px, w) for p, h, w in obs["squares"]]
    segs = []
    for a, b, th, w in obs["segments"]:
        pa, pb = T.transform(a), T.transform(b)
        segs.append((np.linspace(pa, pb, 60), th * px / 2, w))
    placed = []

    def cost_and_clearance(bb):
        cost, clear = 0.0, np.inf
        for c, r, w in circles:
            d = _rect_point_dist(bb, c) - r; clear = min(clear, d); cost += w if d < 0 else 0
        for c, h, w in squares:
            d = max(bb.x0 - (c[0] + h), (c[0] - h) - bb.x1, bb.y0 - (c[1] + h), (c[1] - h) - bb.y1)
            clear = min(clear, d); cost += w if d < 0 else 0
        for pts, half, w in segs:
            d = min(_rect_point_dist(bb, q) for q in pts) - half; clear = min(clear, d); cost += w if d < 0 else 0
        for pb in placed:
            if bb.overlaps(pb): cost += 5
        if not (axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1):
            cost += 100
        return cost, clear

    for i in range(P.n):
        p = P.p_true[i]; r_pts = (np.sqrt(290) / 2 + 1.2) if i in MAL else (np.sqrt(170) / 2 + 0.5)
        kw = dict(fontsize=10, color=RED if i in MAL else "#1f2a44", zorder=8)
        best = None
        for level, scale in enumerate((1.0, 1.35, 1.75, 2.3)):
            d = r_pts + 1.5 * scale + (scale - 1.0) * r_pts
            for pref, (dx, dy, ha, va) in enumerate(DIRS):
                ann = ax.annotate(f"$U_{i}$", xy=p, xytext=(dx * d, dy * d), textcoords="offset points",
                                  ha=ha, va=va, **kw)
                bb = ann.get_window_extent(R).padded(0.6 * px); ann.remove()
                cost, clear = cost_and_clearance(bb)
                key = (cost, level, pref)          # nearest collision-free position, fixed preference order
                if best is None or key < best[0]:
                    best = (key, (dx * d, dy * d, ha, va))
            if best[0][0] == 0:
                break
        dxy, ha, va = best[1][:2], best[1][2], best[1][3]
        ann = ax.annotate(f"$U_{i}$", xy=p, xytext=dxy, textcoords="offset points", ha=ha, va=va, **kw)
        placed.append(ann.get_window_extent(R).padded(0.6 * px))


# ================================================================ Figure 4: maps with one legend
pos = pd.read_csv(SIM / "six_uav_final_positions.csv")


class Net:
    """Six-UAV geometry read from the result file; links rebuilt with the communication radius
    recorded in parameters_used.json (the rule used by the experiment)."""
    def __init__(self, d):
        d = d.sort_values("uav")
        self.n = len(d)
        self.p_true = d[["true_x", "true_y"]].to_numpy()
        self.g = d[["prior_x", "prior_y"]].to_numpy()
        r = PARAMS["six_uav"]["comm_radius_m"]
        self.edges = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)
                      if np.linalg.norm(self.p_true[i] - self.p_true[j]) <= r]


first = pos[(pos.attack_name == "none") & (pos.defense_name == "no_defense")]
P = Net(first)
MAL = set(int(u) for u in pos[(pos.attack_name == "fdi") & pos.malicious].uav.unique())
assert MAL == set(PARAMS["six_uav"]["malicious"]), "compromised set differs from parameters_used.json"
# targets: all neighbours of each compromised UAV (parameters_used.json: six_uav.targets)
MAP_BOX_RATIO = 0.85                     # common height/width ratio of all map panels
TARGETS = {m: sorted({j for e in P.edges for j in e if m in e and j != m}) for m in sorted(MAL)}
print("six-UAV links:", P.edges, "| compromised:", sorted(MAL), "| targets:", TARGETS)


class HandlerArrow(HandlerBase):
    def create_artists(self, legend, orig_handle, xd, yd, w, h, fs, trans):
        a = FancyArrowPatch((-xd, h / 2 - yd), (w - xd, h / 2 - yd), arrowstyle="-|>", mutation_scale=11,
                            color=RED, lw=1.7, ls="--", transform=trans)
        return [a]


class ArrowKey:
    pass


def draw_map(ax, atk, dfn, est_color, title):
    d = pos[(pos.attack_name == atk) & (pos.defense_name == dfn)].sort_values("uav")
    est = d[["est_x", "est_y"]].to_numpy(); hon = [i for i in range(P.n) if i not in MAL]
    for i, j in P.edges:
        ax.plot(P.p_true[[i, j], 0], P.p_true[[i, j], 1], color=SLATE, lw=1.2, ls=(0, (4, 2.5)), zorder=1)
    for m, tg in TARGETS.items():
        for t in tg:
            a, b = P.p_true[m], P.p_true[t]
            ax.annotate("", xy=a + 0.80 * (b - a), xytext=a + 0.18 * (b - a), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8, ls="--", mutation_scale=13,
                                        shrinkA=0, shrinkB=0))
    for i in hon:
        ax.plot([P.p_true[i, 0], est[i, 0]], [P.p_true[i, 1], est[i, 1]], color=est_color, lw=1.5, zorder=2)
    ax.scatter(P.g[:, 0], P.g[:, 1], marker="x", s=55, lw=1.8, color="black", zorder=4)
    ax.scatter(P.p_true[:, 0], P.p_true[:, 1], marker="*", s=170, color=GOLD, edgecolors="#e07b00", lw=0.9, zorder=5)
    ax.scatter(est[hon, 0], est[hon, 1], marker="o", s=95, facecolors="white", edgecolors=est_color, lw=2.2, zorder=6)
    mal = sorted(MAL)
    ax.scatter(P.p_true[mal, 0], P.p_true[mal, 1], marker="s", s=290, facecolors="none", edgecolors=RED, lw=2.4, zorder=7)
    pts = np.vstack([P.p_true, P.g, est[hon]])
    # identical frame for every panel: the data range is widened around the panel's content to a common
    # height/width ratio, so all panels have the same size while 1 m in x equals 1 m in y
    lo, hi = pts.min(0) - 3.0, pts.max(0) + 3.0
    c, (w, h) = (lo + hi) / 2, hi - lo
    if h / w < MAP_BOX_RATIO:
        h = w * MAP_BOX_RATIO
    else:
        w = h / MAP_BOX_RATIO
    ax.set_xlim(c[0] - w / 2, c[0] + w / 2); ax.set_ylim(c[1] - h / 2, c[1] + h / 2)
    ax.set_box_aspect(MAP_BOX_RATIO)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    rm = np.sqrt(np.mean(np.sum((est[hon] - P.p_true[hon]) ** 2, 1)))
    panel_label(ax, title, pad=35, size=11.5)
    obs = dict(circles=[], squares=[], segments=[])
    for i in range(P.n):
        obs["circles"].append((P.p_true[i], np.sqrt(170) / 2 + 0.5, 3))          # true position (star)
        obs["circles"].append((P.g[i], np.sqrt(55) / 2 + 0.9, 3))                # GNSS prior (cross)
    for i in hon:
        obs["circles"].append((est[i], np.sqrt(95) / 2 + 1.1, 3))                # estimate (circle)
        obs["segments"].append((P.p_true[i], est[i], 1.5, 1.5))                  # estimation-error line
    for m in MAL:
        obs["squares"].append((P.p_true[m], np.sqrt(290) / 2 + 1.2, 3))          # compromised-UAV box
    for i, j in P.edges:
        obs["segments"].append((P.p_true[i], P.p_true[j], 1.2, 1))               # communication link
    for m, tg in TARGETS.items():
        for t in tg:
            a, b = P.p_true[m], P.p_true[t]
            obs["segments"].append((a + 0.18 * (b - a), a + 0.80 * (b - a), 2.0, 2))   # attack arrow
            obs["circles"].append((a + 0.80 * (b - a), 4.0, 2))                        # arrow head
    ax._uav_obstacles = obs
    return rm


with plt.rc_context({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10}):
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.9))
    panels = [(axes[0, 0], "byzantine", "no_defense", BLUE, "(a) Byzantine attack, no defense"),
              (axes[0, 1], "fdi", "no_defense", MAGENTA, "(b) FDI attack, no defense"),
              (axes[1, 0], "byzantine", "ledger+screening", LIME, "(c) Proposed defense, Byzantine attack"),
              (axes[1, 1], "fdi", "ledger+screening", LIME, "(d) Proposed defense, FDI attack"),
              (axes[1, 2], "disruption", "ledger+screening", LIME, "(e) Proposed defense, disruption attack")]
    for ax, atk, dfn, col, title in panels:
        print(title, f"{draw_map(ax, atk, dfn, col, title):.2f}")
    lax = axes[0, 2]; lax.axis("off")
    handles = [Line2D([], [], ls="none", marker="x", ms=9, mew=2.0, color="black", label="GNSS prior"),
               Line2D([], [], ls="none", marker="*", ms=15, mfc=GOLD, mec="#e07b00", label="True position"),
               Line2D([], [], ls="none", marker="o", ms=10, mfc="white", mec=BLUE, mew=2.4, label="Estimate, Byzantine attack"),
               Line2D([], [], ls="none", marker="o", ms=10, mfc="white", mec=MAGENTA, mew=2.4, label="Estimate, FDI attack"),
               Line2D([], [], ls="none", marker="o", ms=10, mfc="white", mec=LIME, mew=2.4, label="Estimate, proposed defense"),
               Line2D([], [], color=SLATE, lw=1.6, ls=(0, (4, 2.5)), label="Communication link"),
               ArrowKey(),
               Line2D([], [], ls="none", marker="s", ms=15, mfc="none", mec=RED, mew=2.6, label="Compromised UAV")]
    labels = [h.get_label() if not isinstance(h, ArrowKey) else "Attack path (targeted link)" for h in handles]
    lax.legend(handles, labels, handler_map={ArrowKey: HandlerArrow()}, loc="center", fontsize=11,
               frameon=True, borderpad=1.0, labelspacing=1.0, handlelength=2.8)
    fig.tight_layout(h_pad=2.2, w_pad=0.8)
    for ax, *_ in panels:
        place_labels(fig, ax)
    save(fig, "fig_six_maps")

# ================================================================ Figure 5: residuals and decisions
t = tr[(tr.attack_name == "fdi") & (tr.defense_name == "ledger+screening")]
n_it = int(t.iteration.max())
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for k, (col, lab, ls, c, m) in enumerate([("primal_own", r"Own-copy primal $r_{\mathrm{own}}$", "-", BLUE, "o"),
                                          ("primal_screened", r"Screened primal $r_{\mathrm{scr}}$", "--", ORANGE, "s"),
                                          ("primal_original", r"Original primal $r_{\mathrm{orig}}$", "-.", MAGENTA, "^"),
                                          ("dual", r"Dual $s$", ":", DARKGRAY, "D")]):
    marked(axes[0], t.iteration, np.maximum(t[col], 1e-9), 12, offset=3 * k, ls=ls, color=c, marker=m, label=lab, lw=1.9)
axes[0].set_yscale("log")
ref_line(axes[0], PARAMS["solver"]["eps_primal"], 0, n_it, dict(color=RED, lw=1.2, ls=(0, (1, 1.5)), marker="x", ms=5,
         mew=1.2, label=r"Tolerance $\epsilon_p=\epsilon_d$"), offset=10, step=20)
axes[0].set_xlabel("ADMM iteration"); axes[0].set_ylabel("RMS residual (m)"); axes[0].legend(fontsize=7.2)
panel_label(axes[0], "(a) Primal and dual residuals")
for k, (col, lab, c, ls, m) in enumerate([("tp", "Flagged compromised messages", LIME, "-", "o"),
                                          ("fp", "Flagged honest messages", RED, "-", "s"),
                                          ("fn", "Accepted compromised messages", ORANGE, "--", "^")]):
    marked(axes[1], t.iteration, t[col], 12, offset=4 * k, color=c, ls=ls, marker=m, label=lab, lw=1.9)
W = PARAMS["proposed_screening"]["screen_start_iter"]
axes[1].axvline(W, color=VIOLET, lw=1.4, ls=(0, (1.2, 1.6)))
axes[1].text(W + 2.5, 1.2, "end of warm-up", fontsize=7.5, color=VIOLET, bbox=dict(fc="white", ec="none", pad=0.5))
axes[1].set_xlabel("ADMM iteration"); axes[1].set_ylabel("Messages per iteration"); axes[1].legend(fontsize=7.2)
panel_label(axes[1], "(b) Screening decisions")
fig.tight_layout(); save(fig, "fig_six_residuals")

# ================================================================ Figure 6: adaptive attacker
A = pd.read_csv(SIM / "adaptive_attack_table.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
for ax, M in zip(axes, (100, 200, 300)):
    gg = A[(A.M == M) & (A.attack == "adaptive") & (A.defense_name == "ledger+screening")]
    for kp, c, m, lab, lw in [(2.5, LIME, "s", r"Proposed ($\kappa$ = 2.5, selected)", 2.4),
                              (2.0, BLUE, "o", r"Proposed with $\kappa$ = 2", 1.6),
                              (3.0, ORANGE, "^", r"Proposed with $\kappa$ = 3", 1.6)]:
        q = gg[np.isclose(gg.kappa, kp)].sort_values("attack_adaptive_margin")
        ax.errorbar(q.attack_adaptive_margin, q.final_rmse_honest_m_mean, yerr=q.final_rmse_honest_m_ci95,
                    marker=m, ms=4.5, mfc="white", mew=1.3, color=c, lw=lw, label=lab, **ERR)
    nd = A[(A.M == M) & (A.attack == "adaptive") & (A.defense_name == "no_defense")].sort_values("attack_adaptive_margin")
    ax.plot(nd.attack_adaptive_margin, nd.final_rmse_honest_m_mean, color=RED, ls="--", marker="D", ms=4,
            mfc="white", mew=1.3, label="No defense")
    x0, x1 = 0.2, 1.55
    ref_line(ax, gg.ref_prior_only_rmse_mean.mean(), x0, x1, REF_PRIOR, offset=8)
    ref_line(ax, gg.ref_oracle_wls_rmse_mean.mean(), x0, x1, REF_ORACLE, offset=20)
    ax.set_xlim(x0, x1)
    ax.axvline(1.0, color=SLATE, lw=1.0, ls=(0, (3, 2)))
    ax.set_ylim(bottom=0); ax.set_xlabel(r"Attack offset $\gamma_a = \|\boldsymbol{\delta}\|/\tau$")
    panel_label(ax, f"({'abc'[(M // 100) - 1]}) {M} compromised UAVs", pad=32)
axes[0].set_ylabel("Honest-UAV RMSE (m)")
h, l = axes[0].get_legend_handles_labels()
rank = lambda lab: (0 if lab.startswith("Proposed (") else 1 if lab.startswith("Proposed with") else 2 if lab == "No defense" else 3)
order = sorted(range(len(l)), key=lambda i: (rank(l[i]), i))       # selected setting first, then variants, baseline, references
h, l = [h[i] for i in order], [l[i] for i in order]
fig.tight_layout(rect=(0, 0, 1, 0.82))
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
save(fig, "fig_adaptive")
print("figures done")
