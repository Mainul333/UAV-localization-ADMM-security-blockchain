"""make_tables.py - regenerate all data tables of the manuscript (LaTeX table bodies) from the result files.

Usage: python make_tables.py --sim results_revision --bc results_blockchain --out tables
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True); ap.add_argument("--bc", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--load", default=None, help="folder with the load-test results (default: --bc)")
ap.add_argument("--fast", default=None, help="folder with the optimized load-test results (optional)")
args = ap.parse_args()
SIM, BC, TAB = Path(args.sim), Path(args.bc), Path(args.out)
LOAD = Path(args.load) if args.load else BC
FAST = Path(args.fast) if args.fast else None
TAB.mkdir(parents=True, exist_ok=True)
ATK = {"none": "No attack", "byzantine": "Byzantine (equivocating)", "byzantine_single": "Byzantine (single value)",
       "disruption": "Disruption (equivocating)", "disruption_single": "Disruption (single value)",
       "fdi": "FDI", "adaptive": "Adaptive"}
L = pd.read_csv(SIM / "large_scale_table.csv")
refL = L.groupby("M")[["ref_prior_only_rmse_mean", "ref_wls_attack_free_rmse_mean", "ref_oracle_wls_rmse_mean"]].mean()
T = pd.read_csv(SIM / "threshold_sweep_table.csv")
k = T[(T.sweep == "kappa") & (T.problem == "large")]
oc = pd.read_csv(BC / "onchain_summary_all_runs.csv")

# ------------------------------------------------------------------ TABLES
def write(name, body):
    (TAB / f"{name}.tex").write_text(body)

# Table: 1000-UAV main results
# defense columns: baselines first, the proposed defense (ledger + screening) last
cols = ["no_defense", "ledger_only", "screening_only", "signed_echo_only", "signed_echo+screening",
        "wmsr_baseline", "ledger+screening"]
atks = ["none", "byzantine", "byzantine_single", "disruption", "disruption_single", "fdi", "adaptive"]
rows = []
for M in (100, 200, 300):
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{M} compromised UAVs (prior only {refL.loc[M,'ref_prior_only_rmse_mean']:.2f} m; "
                f"attack-free WLS {refL.loc[M,'ref_wls_attack_free_rmse_mean']:.2f} m; ideal WLS {refL.loc[M,'ref_oracle_wls_rmse_mean']:.2f} m)}}}} \\\\")
    for a in atks:
        cells = []
        for dn in cols:
            r = L[(L.M == M) & (L.attack == a) & (L.defense_name == dn)]
            if r.empty:
                cells.append("--"); continue
            r = r.iloc[0]
            mark = "$^{\\dagger}$" if r.converged_frac < 1 else ""
            best = dn == "ledger+screening"
            v = f"{r.reported_rmse_honest_m_mean:.2f}$\\pm${r.reported_rmse_honest_m_ci95:.2f}{mark}"
            cells.append(f"\\textbf{{{v}}}" if best else v)
        rows.append(f"{ATK[a]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
write("tab_large_main", "\n".join(rows))

# Table: detection statistics of the proposed defense (rows grouped by the number of compromised UAVs)
rows = []
for M in (100, 200, 300):
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{M} compromised UAVs}}}} \\\\")
    for a in ["none", "byzantine", "byzantine_single", "disruption", "disruption_single", "fdi", "adaptive"]:
        r = L[(L.M == M) & (L.attack == a) & (L.defense_name == "ledger+screening")].iloc[0]
        mdr = "--" if not np.isfinite(r.missed_detection_rate_mean) else f"{100*r.missed_detection_rate_mean:.1f}"
        rej = r.ledger_rejected_mean / max(1, r.iterations_mean)
        rows.append(f"{ATK[a]} & {100*r.false_alarm_rate_mean:.1f} & {mdr} & {r.iterations_mean:.0f} & {rej:.0f} \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
write("tab_detection", "\n".join(rows))

# Table: six-UAV Monte Carlo
mc = pd.read_csv(SIM / "six_uav_mc_table.csv")
rows = []
for pl in ("fixed", "random"):
    g = mc[mc.placement == pl]
    rr = g.iloc[0]
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{'Compromised $U_1$, $U_4$' if pl=='fixed' else 'Two compromised UAVs placed at random'}: "
                f"prior only {rr.ref_prior_only_rmse_mean:.2f} m, attack-free WLS {rr.ref_wls_attack_free_rmse_mean:.2f} m, "
                f"ideal WLS {rr.ref_oracle_wls_rmse_mean:.2f} m}}}} \\\\")
    for a in ["none", "byzantine", "byzantine_single", "disruption", "fdi", "adaptive"]:
        cells = []
        for dn in ["no_defense", "ledger_only", "screening_only", "wmsr_baseline", "ledger+screening"]:
            r = g[(g.attack == a) & (g.defense_name == dn)].iloc[0]
            mark = "$^{\\dagger}$" if r.converged_frac < 1 else ""
            v = f"{r.reported_rmse_honest_m_mean:.2f}$\\pm${r.reported_rmse_honest_m_ci95:.2f}{mark}"
            cells.append(f"\\textbf{{{v}}}" if dn == "ledger+screening" else v)
        rows.append(f"{ATK[a]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
write("tab_six_mc", "\n".join(rows))

# Table: on-chain runs
# "none" = states read from the ledger without screening; it is used only with FDI, where no conflicting
# records exist, so it is identical to "ledger only" (the label used in the manuscript).
defl = {"none": "Blockchain only", "ledger": "Blockchain only", "ledger+screening": "Proposed"}
order = [("none", "equivocate", "ledger+screening"), ("fdi", "equivocate", "none"), ("fdi", "equivocate", "ledger+screening"),
         ("byzantine", "equivocate", "ledger"), ("byzantine", "equivocate", "ledger+screening"),
         ("byzantine", "single", "ledger+screening"), ("disruption", "equivocate", "ledger+screening"),
         ("adaptive", "equivocate", "ledger+screening")]
rows = []
for a, cp, dop in order:
    r = oc[(oc.attack == a) & (oc.commit_policy == cp) & (oc.defense_option == dop)].iloc[0]
    name = {"none": "No attack", "fdi": "FDI", "adaptive": "Adaptive",
            "byzantine": "Byzantine (equivocating)" if cp == "equivocate" else "Byzantine (single value)",
            "disruption": "Disruption (equivocating)"}[a]
    rows.append(f"{name} & {defl[dop]} & {int(r.iterations)} & {r.final_rmse_honest_m:.3f} & {r.offchain_simulation_rmse_honest_m:.3f} & "
                f"{1e6*r.max_abs_diff_onchain_vs_simulation_m:.1f} & {int(r.total_equivocation_events)} & "
                f"{r.mean_iteration_wall_s:.2f} & {r.end_to_end_s:.0f} \\\\")
write("tab_onchain", "\n".join(rows) + "\n\\bottomrule")


# Table: sensitivity to noise levels and initial errors (two labeled groups; 1000 UAVs, 200 compromised, proposed defense)
n = T[T.sweep == "noise"]; ini = T[T.sweep == "init"]
def sens_row(sg, sy, s0, g):
    r0 = g[g.attack == "none"].iloc[0]; r1 = g[g.attack == "fdi"].iloc[0]
    return (f"{sg:g} & {sy:g} & {s0:g} & {2.5*sy:.3g} & {r0.final_rmse_honest_m_mean:.2f} & {100*r0.false_alarm_rate_mean:.1f} & "
            f"{r1.final_rmse_honest_m_mean:.2f}{'$^{\\dagger}$' if r1.converged_frac < 1 else ''} & "
            f"{100*r1.missed_detection_rate_mean:.1f} & {r0.ref_prior_only_rmse_mean:.2f} \\\\")
rows = ["\\multicolumn{9}{l}{\\textit{Noise levels (initial error $s_0=0$)}} \\\\"]
for (sg, sy), g in n.groupby(["problem_sigma_g", "problem_sigma_y"]):
    rows.append(sens_row(sg, sy, 0, g))
rows += ["\\midrule", "\\multicolumn{9}{l}{\\textit{Initial errors ($\\sigma_g=1.5$\\,m, $\\sigma_y=0.5$\\,m)}} \\\\"]
for s0, g in ini.groupby("solver_init_noise_std"):
    rows.append(sens_row(1.5, 0.5, s0, g))          # s0 = 0 is the reference for the larger initial errors
write("tab_sensitivity", "\n".join(rows) + "\n\\bottomrule")

# Table: screening rules (1000 UAVs, 200 compromised): fixed kappa with warm-up, clipping, chi-square rule
def thr_row(name, g, proposed=False):
    cells = [name, f"{100*g.loc['none'].false_alarm_rate_mean:.1f}", f"{g.loc['none'].final_rmse_honest_m_mean:.2f}",
             f"{g.loc['fdi'].final_rmse_honest_m_mean:.2f}", f"{100*g.loc['fdi'].missed_detection_rate_mean:.1f}",
             f"{g.loc['byzantine_single'].final_rmse_honest_m_mean:.2f}", f"{g.loc['adaptive'].final_rmse_honest_m_mean:.2f}"]
    if proposed:   # the screening rule used by the proposed defense
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return " & ".join(cells) + " \\\\"
rows = []
kk = k[(k.screening_mode == "replace")]
for W in (0, 10, 30):
    for kp in (2.0, 2.5, 3.0):
        g = kk[(kk.screen_start_iter == W) & (np.isclose(kk.kappa, kp))].set_index("attack")
        if not g.empty:
            prop = (W == 10 and np.isclose(kp, 2.5))
            rows.append(thr_row(f"Replace, $\\kappa={kp:g}$, $W={W}$" + (" (proposed)" if prop else ""), g, prop))
kc = k[(k.screening_mode == "clip") & (k.screen_start_iter == 10) & (np.isclose(k.kappa, 2.5))].set_index("attack")
rows.append(thr_row("Clip, $\\kappa=2.5$, $W=10$", kc))
runs = pd.read_csv(SIM / "threshold_sweep_runs.csv")
c2 = runs[(runs.sweep == "chi2") & (runs.problem == "large")]
for (al, est), g in c2.groupby(["alpha_fa", c2.est_std.fillna(-1)]):
    if al not in (0.05, 0.01) or est >= 0:
        continue  # chi-square rule with the prior standard deviation (the rule discussed in the text)
    g = g.groupby("attack")[["false_alarm_rate", "final_rmse_honest_m", "missed_detection_rate", "tau_mean_m"]].mean()
    g = g.rename(columns={"false_alarm_rate": "false_alarm_rate_mean", "final_rmse_honest_m": "final_rmse_honest_m_mean",
                          "missed_detection_rate": "missed_detection_rate_mean"})
    rows.append(thr_row(f"$\\chi^2$, $\\alpha={al:g}$ ($\\tau\\approx{g.loc['none'].tau_mean_m:.1f}$\\,m)", g))
write("tab_threshold", "\n".join(rows) + "\n\\bottomrule")

# Table: blockchain load test, 6 to 1000 submitting UAVs, default and optimized configuration (measured; five rounds)
ls_ = pd.read_csv(LOAD / "ledger_load_test_summary.csv").set_index("N")
fs_ = pd.read_csv(FAST / "fast_load_test_summary.csv").set_index("N") if FAST else None
rows = []
for N, r in ls_.iterrows():
    cells = [f"{int(N)}", f"{r.latency_mean_s:.2f}$\\pm${r.latency_std_s:.2f}", f"{r.blocks_spanned_mean:.1f}",
             f"{r.onchain_tps_mean:.0f}", f"{r.block_size_max_bytes / 1000:.1f}"]
    if fs_ is not None and N in fs_.index:
        f = fs_.loc[N]
        cells += [f"{f.latency_mean_s:.2f}$\\pm${f.latency_std_s:.2f}", f"{f.blocks_spanned_mean:.1f}",
                  f"{f.onchain_tps_mean:.0f}", f"{f.block_size_max_bytes / 1000:.1f}"]
    rows.append(" & ".join(cells) + " \\\\")
write("tab_load", "\n".join(rows) + "\n\\bottomrule")
print("tables written to", TAB)
