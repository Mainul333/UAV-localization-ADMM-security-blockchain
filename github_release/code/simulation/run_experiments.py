"""
run_experiments.py
==================

Regenerates the simulation evidence requested in the reviews. Every output is
written to --out (default: results_revision/). Nothing here needs the blockchain;
on-chain measurements are in blockchain_experiments.py.

Experiments (run any subset, or "all"):

  sanity     Attack-free ADMM vs. centralized WLS; reproduction of the blockchain
             prototype run (Table 3); Reviewer 3's two-UAV relabelling example;
             random relabelling test; prior-only / WLS / oracle references.
             -> R3 C1, C3c
  six        Six-UAV ablation (all attacks x all defenses, including the secured
             disruption case), traces, residuals, final positions for maps.
             -> R1 C7, R2 C4, R3 C3, C5
  six_mc     Six-UAV Monte Carlo over noise/geometry and over attacker placement.
             -> R1 C6, R2 C6, R3 C6c
  threshold  kappa / warm-up / replace-vs-clip sweep, chi-square rule, noise-ratio
             and initial-error sensitivity, false-alarm and missed-detection rates.
             -> R1 C3, R2 C3, R3 C6b-c
  adaptive   Threshold-aware attacker (offset just below tau), margin sweep.
             -> R1 C3, R2 C4, R3 C4b
  large      1000-UAV table (replacement for Table 4) with ablation, W-MSR baseline,
             signed-echo comparison, Monte Carlo CIs, converged flags, tail metrics.
             -> R1 C5-7, R2 C4-6, R3 C5, C7c
  runtime    Solver time and message counts versus network size (simulation side
             of the scalability discussion). -> R2 C5, R3 C7

Examples
  python run_experiments.py all --seeds 10 --workers 8
  python run_experiments.py sanity six --quick
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from secure_localization import (
    AttackConfig, DefenseConfig, SolverConfig, Problem, PROPOSED_SCREENING, SUBMITTED_SCREENING,
    ablation_defenses, centralized_wls, choose_malicious, make_attack, random_knn_problem,
    run_case, six_uav_problem, two_uav_relabelling_example, relabelling_test, to_legacy_map_dict,
    DEFAULT_ATTACKS,
)

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

# -----------------------------------------------------------------------------
# Global experiment settings (report these in the parameter table)
# -----------------------------------------------------------------------------
SOLVER = SolverConfig(rho=1.0, max_iter=800, eps_primal=1e-4, eps_dual=1e-4, min_iter=5, tail_window=100)
SIX_MALICIOUS = (1, 4)
LARGE_N = 1000
LARGE_MALICIOUS = (100, 200, 300)
BASE_SEED = 7            # seed of the submitted experiments; Monte Carlo seeds are 7, 8, 9, ...

# Attack scenarios. "equivocate" = receiver-dependent messages as defined in the
# manuscript; "single" = rational attacker that commits one consistent false value.
ATTACKS = {
    "none": ("none", {}),
    "byzantine": ("byzantine", dict(commit_policy="equivocate")),
    "byzantine_single": ("byzantine", dict(commit_policy="single")),
    "disruption": ("disruption", dict(commit_policy="equivocate")),
    "disruption_single": ("disruption", dict(commit_policy="single")),
    "fdi": ("fdi", {}),
    "adaptive": ("adaptive", {}),
}
MAIN_DEFENSES = ["no_defense", "ledger_only", "screening_only", "ledger+screening",
                 "signed_echo_only", "signed_echo+screening", "wmsr_baseline", "submitted_config"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ci95(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    t = stats.t.ppf(0.975, x.size - 1) if stats is not None else 1.96
    return float(x.mean()), float(t * x.std(ddof=1) / np.sqrt(x.size))


def rmse_honest(P: Problem, X: np.ndarray, malicious) -> float:
    m = np.ones(P.n, bool); m[list(malicious)] = False
    return float(np.sqrt(np.mean(np.sum((X[m] - P.p_true[m]) ** 2, 1))))


def make_problem(kind: str, seed: int, **kw) -> Problem:
    if kind == "six":
        return six_uav_problem(seed=seed, **kw)
    return random_knn_problem(n=kw.pop("n", LARGE_N), seed=seed, **kw)


def malicious_for(kind: str, P: Problem, seed: int, M: int | None, placement: str = "fixed"):
    if kind == "six":
        if placement == "random":
            rng = np.random.default_rng(seed + 555)
            return tuple(sorted(rng.choice(P.n, size=2, replace=False).tolist()))
        return SIX_MALICIOUS
    return choose_malicious(P.n, M, seed)


def references(P: Problem, malicious) -> dict:
    Xw = centralized_wls(P)
    Xo = centralized_wls(P, exclude_nodes=malicious)
    return dict(ref_prior_only_rmse=rmse_honest(P, P.g, malicious),
                ref_wls_attack_free_rmse=rmse_honest(P, Xw, malicious),
                ref_oracle_wls_rmse=rmse_honest(P, Xo, malicious))


def job(spec: dict) -> dict:
    """One simulation run (picklable for multiprocessing)."""
    P = make_problem(spec["problem"], spec["seed"], **spec.get("problem_kw", {}))
    mal = malicious_for(spec["problem"], P, spec["seed"], spec.get("M"), spec.get("placement", "fixed"))
    kind, akw = ATTACKS[spec["attack"]]
    akw = dict(akw); akw.update(spec.get("attack_kw", {}))
    A = make_attack(kind, mal, **akw)
    D = DefenseConfig(**spec["defense"])
    S = SolverConfig(**{**asdict(SOLVER), **spec.get("solver_kw", {}), "store_traces": False})
    s = run_case(P, A, D, S)["summary"]
    s.update({k: v for k, v in spec.items() if k not in ("defense", "problem_kw", "attack_kw", "solver_kw")})
    s.update({f"attack_{k}": v for k, v in spec.get("attack_kw", {}).items()})
    s.update({f"problem_{k}": v for k, v in spec.get("problem_kw", {}).items()})
    s.update({f"solver_{k}": v for k, v in spec.get("solver_kw", {}).items()})
    if spec.get("with_refs"):
        s.update(references(P, mal))
    return s


def run_jobs(specs, workers: int, label: str):
    t0 = time.time()
    out = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(job, specs, chunksize=1)):
                out.append(r)
                if (k + 1) % 25 == 0:
                    print(f"  [{label}] {k + 1}/{len(specs)} runs, {time.time() - t0:.0f}s")
    else:
        for k, sp in enumerate(specs):
            out.append(job(sp))
            if (k + 1) % 25 == 0:
                print(f"  [{label}] {k + 1}/{len(specs)} runs, {time.time() - t0:.0f}s")
    print(f"  [{label}] done: {len(specs)} runs in {time.time() - t0:.0f}s")
    return pd.DataFrame(out)


def defense_dict(name: str, **override) -> dict:
    return {**asdict(ablation_defenses()[name]), **override}


def add_reductions(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Reduction relative to 'no_defense' for the same attack/seed/M, using the
    reported RMSE: final value if converged, tail average over the last
    tail_window iterations if the run hit max_iter (non-convergent disruption)."""
    base = df[df["defense_name"] == "no_defense"].set_index(keys)["reported_rmse_honest_m"]
    idx = df.set_index(keys).index
    ref = base.reindex(idx).to_numpy()
    df["reduction_vs_no_defense_pct"] = 100.0 * (ref - df["reported_rmse_honest_m"].to_numpy()) / ref
    if "ref_prior_only_rmse" in df:
        df["better_than_prior_only"] = df["reported_rmse_honest_m"] < df["ref_prior_only_rmse"]
    return df


def aggregate(df: pd.DataFrame, group_cols, metrics) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_cols, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["n_runs"] = len(grp)
        for m in metrics:
            if m not in grp:
                continue
            if grp[m].dtype == bool:
                row[f"{m}_frac"] = float(grp[m].mean())
            else:
                mu, hw = ci95(grp[m].to_numpy(float))
                row[f"{m}_mean"] = mu
                row[f"{m}_ci95"] = hw
        rows.append(row)
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, out: Path, name: str):
    p = out / name
    df.to_csv(p, index=False)
    print(f"  wrote {p}")


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def exp_sanity(out: Path, args):
    print("\n[sanity]")
    rows = []
    # 1) Reproduce the blockchain prototype run (rho = 1, 40 iterations, Table 3).
    P = six_uav_problem(7)
    r = run_case(P, make_attack("none", ()), DefenseConfig(),
                 SolverConfig(rho=1.0, max_iter=40, eps_primal=0, eps_dual=0))
    rows.append(dict(check="reproduce blockchain prototype (40 iterations, rho=1)",
                     value=r["summary"]["final_rmse_all_m"], expected=1.683646,
                     passed=abs(r["summary"]["final_rmse_all_m"] - 1.683646) < 1e-5))
    # 2) Attack-free ADMM vs centralized WLS (six and large)
    for name, P in [("six", six_uav_problem(7)), ("large", random_knn_problem(LARGE_N, 7))]:
        Xw = centralized_wls(P)
        r = run_case(P, make_attack("none", ()), DefenseConfig(),
                     SolverConfig(rho=1.0, max_iter=5000, eps_primal=1e-7, eps_dual=1e-7))
        diff = float(np.max(np.abs(r["x_hat"] - Xw)))
        rows.append(dict(check=f"{name}: max |x_ADMM - x_WLS| (attack-free, tight tolerance)",
                         value=diff, expected=0.0, passed=diff < 1e-5,
                         iterations=r["summary"]["iterations"]))
        r2 = run_case(P, make_attack("none", ()), DefenseConfig(), SOLVER)
        rows.append(dict(check=f"{name}: iterations / RMSE with experiment tolerance",
                         value=r2["summary"]["final_rmse_all_m"],
                         expected=float(np.sqrt(np.mean(np.sum((Xw - P.p_true) ** 2, 1)))),
                         passed=True, iterations=r2["summary"]["iterations"]))
    # 3) Two-UAV example of Reviewer 3, Comment 1
    ex = two_uav_relabelling_example()
    (out / "two_uav_relabelling_example.json").write_text(json.dumps(ex, indent=2))
    rows.append(dict(check="Eq. (47) label-invariant in two-UAV example", value=float(ex["eq47_label_invariant"]),
                     expected=0.0, passed=not ex["eq47_label_invariant"]))
    # 4) Random relabelling of the six- and 1000-UAV networks under attack + defense
    D = DefenseConfig(exchange="ledger", exclude_equivocators=True, **PROPOSED_SCREENING)
    for name, P, mal in [("six", six_uav_problem(7), SIX_MALICIOUS),
                         ("large", random_knn_problem(LARGE_N, 7), choose_malicious(LARGE_N, 200, 7))]:
        for atk in ["fdi", "adaptive"]:
            d = relabelling_test(P, make_attack(atk, mal), D, SOLVER, seed=1)
            rows.append(dict(check=f"{name}: relabelling invariance under {atk} + ledger+screening (max diff, m)",
                             value=d, expected=0.0, passed=d < 1e-8))
    # 5) References
    for name, P, mals in [("six", six_uav_problem(7), [SIX_MALICIOUS]),
                          ("large", random_knn_problem(LARGE_N, 7),
                           [choose_malicious(LARGE_N, M, 7) for M in LARGE_MALICIOUS])]:
        for mal in mals:
            ref = references(P, mal)
            for k, v in ref.items():
                rows.append(dict(check=f"{name}, {len(mal)} malicious: {k}", value=v, expected=np.nan, passed=True))
    df = pd.DataFrame(rows)
    write(df, out, "sanity_checks.csv")
    print(df.to_string())


def exp_six(out: Path, args):
    print("\n[six] six-UAV ablation with traces (seed 7)")
    P = six_uav_problem(BASE_SEED)
    defenses = ablation_defenses()
    summaries, traces, positions, legacy = [], [], [], {}
    for atk_name, (kind, akw) in ATTACKS.items():
        A = make_attack(kind, SIX_MALICIOUS, **akw)
        for dname in MAIN_DEFENSES:
            r = run_case(P, A, defenses[dname], SOLVER)
            s = dict(r["summary"], attack_name=atk_name, defense_name=dname)
            summaries.append(s)
            tr = pd.DataFrame(r["traces"]); tr["attack_name"] = atk_name; tr["defense_name"] = dname
            traces.append(tr)
            for i in range(P.n):
                positions.append(dict(attack_name=atk_name, defense_name=dname, uav=i,
                                      true_x=P.p_true[i, 0], true_y=P.p_true[i, 1],
                                      prior_x=P.g[i, 0], prior_y=P.g[i, 1],
                                      est_x=r["x_hat"][i, 0], est_y=r["x_hat"][i, 1],
                                      malicious=i in r["malicious"]))
            if dname in ("no_defense", "ledger+screening"):
                legacy[f"{atk_name}|{dname}"] = to_legacy_map_dict(P, r, f"{atk_name} / {dname}")
    df = pd.DataFrame(summaries)
    for k, v in references(P, SIX_MALICIOUS).items():
        df[k] = v
    df = add_reductions(df, ["attack_name"])
    write(df, out, "six_uav_summary.csv")
    write(pd.concat(traces), out, "six_uav_traces.csv")
    write(pd.DataFrame(positions), out, "six_uav_final_positions.csv")
    with open(out / "six_uav_legacy_map_inputs.pkl", "wb") as f:
        pickle.dump(legacy, f)
    if any(r["warnings"] for r in summaries):
        print("  NOTE:", summaries[1]["warnings"])
    plot_six(pd.concat(traces), df, out)
    cols = ["attack_name", "defense_name", "iterations", "converged", "reported_rmse_honest_m",
            "tail_mean_rmse_honest_m", "tail_max_rmse_honest_m", "false_alarm_rate", "missed_detection_rate",
            "ledger_rejected", "reduction_vs_no_defense_pct"]
    print(df[cols].round(3).to_string())


def plot_six(tr: pd.DataFrame, summ: pd.DataFrame, out: Path):
    attacks = ["byzantine", "byzantine_single", "disruption", "fdi", "adaptive", "none"]
    show = {"no_defense": ("tab:red", "-"), "ledger_only": ("tab:blue", "--"),
            "screening_only": ("tab:orange", "-."), "ledger+screening": ("tab:green", "-"),
            "wmsr_baseline": ("tab:purple", ":"), "submitted_config": ("gray", ":")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, atk in zip(axes.ravel(), attacks):
        for dname, (c, ls) in show.items():
            t = tr[(tr.attack_name == atk) & (tr.defense_name == dname)]
            ax.plot(t.iteration, t.rmse_honest_m, color=c, ls=ls, lw=1.6, label=dname)
        ref = summ.iloc[0]
        ax.axhline(ref["ref_wls_attack_free_rmse"], color="k", lw=0.8, ls="--", label="attack-free WLS")
        ax.axhline(ref["ref_prior_only_rmse"], color="k", lw=0.8, ls=":", label="prior only")
        ax.set_title(atk); ax.set_xlabel("Iteration"); ax.set_ylabel("Honest-UAV RMSE (m)")
        ax.set_yscale("log"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_six_uav_convergence_ablation.png", dpi=300); fig.savefig(out / "fig_six_uav_convergence_ablation.pdf")
    plt.close(fig)
    # Residuals of the secured FDI case (R3 C3c)
    t = tr[(tr.attack_name == "fdi") & (tr.defense_name == "ledger+screening")]
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, lab in [("primal_own", "own-copy primal"), ("primal_screened", "screened primal"),
                     ("primal_original", "original (finalized-state) primal"), ("dual", "dual")]:
        ax.semilogy(t.iteration, np.maximum(t[col], 1e-12), label=lab)
    ax.set_xlabel("Iteration"); ax.set_ylabel("RMS residual (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Six UAVs, FDI, ledger+screening")
    fig.tight_layout(); fig.savefig(out / "fig_six_uav_residuals_fdi.png", dpi=300); plt.close(fig)


def exp_six_mc(out: Path, args):
    print("\n[six_mc] six-UAV Monte Carlo")
    seeds = [BASE_SEED + s for s in range(args.seeds)]
    specs = []
    for placement in ("fixed", "random"):
        for seed in seeds:
            for atk in ATTACKS:
                for dname in MAIN_DEFENSES:
                    specs.append(dict(problem="six", seed=seed, placement=placement, attack=atk,
                                      defense_name=dname, defense=defense_dict(dname), with_refs=True))
    df = run_jobs(specs, args.workers, "six_mc")
    df = add_reductions(df, ["placement", "seed", "attack"])
    write(df, out, "six_uav_mc_runs.csv")
    agg = aggregate(df, ["placement", "attack", "defense_name"],
                    ["reported_rmse_honest_m", "final_rmse_honest_m", "tail_mean_rmse_honest_m", "tail_max_rmse_honest_m",
                     "iterations", "converged", "false_alarm_rate", "missed_detection_rate", "reduction_vs_no_defense_pct",
                     "better_than_prior_only", "ref_prior_only_rmse", "ref_wls_attack_free_rmse", "ref_oracle_wls_rmse"])
    write(agg, out, "six_uav_mc_table.csv")


def exp_threshold(out: Path, args):
    print("\n[threshold] screening calibration")
    seeds = [BASE_SEED + s for s in range(max(1, args.seeds // 2 if args.quick else args.seeds))]
    kappas = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0] if not args.quick else [2.0, 3.0, 4.0]
    warm = [0, 10, 30] if not args.quick else [0, 10]
    attacks = ["none", "fdi", "byzantine_single", "adaptive"]
    specs = []
    for prob, M in [("large", 200), ("six", None)]:
        for seed in seeds:
            for mode in ("replace", "clip"):
                for kp in kappas:
                    for W in warm:
                        d = dict(exchange="ledger", filter="threshold", screening_mode=mode, threshold_mode="fixed",
                                 kappa=kp, screen_start_iter=W, exclude_equivocators=True)
                        for atk in attacks:
                            specs.append(dict(problem=prob, M=M, seed=seed, attack=atk, defense_name="sweep",
                                              defense=asdict(DefenseConfig(**d)), sweep="kappa", with_refs=True))
            for alpha in (0.1, 0.05, 0.01):
                for est in (None, 0.5):
                    d = dict(exchange="ledger", filter="threshold", screening_mode="replace", threshold_mode="chi2",
                             alpha_fa=alpha, est_std=est, screen_start_iter=0, exclude_equivocators=True)
                    for atk in attacks:
                        specs.append(dict(problem=prob, M=M, seed=seed, attack=atk, defense_name="sweep",
                                          defense=asdict(DefenseConfig(**d)), sweep="chi2", with_refs=True))
    # Noise-ratio and initial-error sensitivity of the proposed configuration (large network)
    prop = dict(exchange="ledger", exclude_equivocators=True, **PROPOSED_SCREENING)
    for seed in seeds:
        for sg in (0.5, 1.5, 3.0):
            for sy in (0.25, 0.5, 1.0):
                for atk in ("none", "fdi"):
                    specs.append(dict(problem="large", M=200, seed=seed, attack=atk, defense_name="proposed",
                                      defense=asdict(DefenseConfig(**prop)), sweep="noise",
                                      problem_kw=dict(sigma_g=sg, sigma_y=sy), with_refs=True))
        for init in (0.0, 2.0, 5.0):
            for atk in ("none", "fdi"):
                specs.append(dict(problem="large", M=200, seed=seed, attack=atk, defense_name="proposed",
                                  defense=asdict(DefenseConfig(**prop)), sweep="init",
                                  solver_kw=dict(init_noise_std=init), with_refs=True))
    df = run_jobs(specs, args.workers, "threshold")
    write(df, out, "threshold_sweep_runs.csv")
    agg = aggregate(df, ["sweep", "problem", "attack", "screening_mode", "threshold_mode", "kappa", "alpha_fa",
                         "screen_start_iter", "problem_sigma_g", "problem_sigma_y", "solver_init_noise_std"]
                    if "problem_sigma_g" in df and "solver_init_noise_std" in df else
                    ["sweep", "problem", "attack", "screening_mode", "threshold_mode", "kappa", "alpha_fa", "screen_start_iter"],
                    ["false_alarm_rate", "false_alarm_rate_2nd_half", "missed_detection_rate", "final_rmse_honest_m",
                     "tail_mean_rmse_honest_m", "iterations", "converged", "ref_prior_only_rmse"])
    write(agg, out, "threshold_sweep_table.csv")
    plot_threshold(agg, out)


def plot_threshold(agg: pd.DataFrame, out: Path):
    a = agg[(agg.sweep == "kappa")]
    for prob in a.problem.unique():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for (mode, W), g in a[a.problem == prob].groupby(["screening_mode", "screen_start_iter"]):
            lab = f"{mode}, warm-up {int(W)}"
            g0 = g[g.attack == "none"].sort_values("kappa")
            axes[0].plot(g0.kappa, g0.false_alarm_rate_mean, marker="o", label=lab)
            g1 = g[g.attack == "fdi"].sort_values("kappa")
            axes[1].plot(g1.kappa, g1.missed_detection_rate_mean, marker="o", label=lab)
            axes[2].plot(g1.kappa, g1.final_rmse_honest_m_mean, marker="o", label=lab)
        prior = a[a.problem == prob].ref_prior_only_rmse_mean.mean()
        axes[2].axhline(prior, color="k", ls=":", label="prior only")
        axes[0].set_ylabel("False-alarm rate (no attack)"); axes[1].set_ylabel("Missed-detection rate (FDI)")
        axes[2].set_ylabel("Honest RMSE under FDI (m)"); axes[2].set_yscale("log")
        for ax in axes:
            ax.set_xlabel(r"$\kappa$"); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7); fig.suptitle(f"Screening calibration ({prob})")
        fig.tight_layout(); fig.savefig(out / f"fig_threshold_sweep_{prob}.png", dpi=300); plt.close(fig)


def exp_adaptive(out: Path, args):
    print("\n[adaptive] threshold-aware attacker")
    seeds = [BASE_SEED + s for s in range(max(1, args.seeds // 2 if args.quick else args.seeds))]
    margins = [0.25, 0.5, 0.75, 0.9, 0.99, 1.1, 1.5] if not args.quick else [0.5, 0.9, 1.1]
    kappas = [2.0, 2.5, 3.0]
    specs = []
    for seed in seeds:
        for M in LARGE_MALICIOUS:
            for kp in kappas:
                d = asdict(DefenseConfig(exchange="ledger", exclude_equivocators=True,
                                         **{**PROPOSED_SCREENING, "kappa": kp}))
                for mg in margins:
                    specs.append(dict(problem="large", M=M, seed=seed, attack="adaptive", defense_name="ledger+screening",
                                      defense=d, attack_kw=dict(adaptive_margin=mg), with_refs=True))
                specs.append(dict(problem="large", M=M, seed=seed, attack="fdi", defense_name="ledger+screening",
                                  defense=d, with_refs=True))
            for mg in margins:
                specs.append(dict(problem="large", M=M, seed=seed, attack="adaptive", defense_name="no_defense",
                                  defense=defense_dict("no_defense"), attack_kw=dict(adaptive_margin=mg),
                                  with_refs=True))
    df = run_jobs(specs, args.workers, "adaptive")
    write(df, out, "adaptive_attack_runs.csv")
    agg = aggregate(df, ["M", "attack", "defense_name", "kappa", "attack_adaptive_margin"],
                    ["final_rmse_honest_m", "tail_mean_rmse_honest_m", "missed_detection_rate", "false_alarm_rate",
                     "converged", "ref_prior_only_rmse", "ref_oracle_wls_rmse"])
    write(agg, out, "adaptive_attack_table.csv")
    fig, axes = plt.subplots(1, len(LARGE_MALICIOUS), figsize=(14, 4), sharey=True)
    for ax, M in zip(axes, LARGE_MALICIOUS):
        g = agg[(agg.M == M) & (agg.attack == "adaptive") & (agg.defense_name == "ledger+screening")]
        for kp, gg in g.groupby("kappa"):
            gg = gg.sort_values("attack_adaptive_margin")
            ax.errorbar(gg.attack_adaptive_margin, gg.final_rmse_honest_m_mean, yerr=gg.final_rmse_honest_m_ci95,
                        marker="o", capsize=3, label=fr"$\kappa$={kp:g}")
        ax.axhline(g.ref_prior_only_rmse_mean.mean(), color="k", ls=":", label="prior only")
        ax.axhline(g.ref_oracle_wls_rmse_mean.mean(), color="k", ls="--", label="oracle WLS")
        ax.axvline(1.0, color="gray", lw=0.8)
        ax.set_yscale("log"); ax.set_title(f"{M} malicious UAVs"); ax.set_xlabel(r"attack offset / $\tau$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Honest RMSE (m)"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_adaptive_attacker.png", dpi=300); plt.close(fig)


def exp_large(out: Path, args):
    print("\n[large] 1000-UAV table (Table 4 replacement)")
    seeds = [BASE_SEED + s for s in range(args.seeds)]
    specs = []
    for seed in seeds:
        for M in LARGE_MALICIOUS:
            for atk in ATTACKS:
                for dname in MAIN_DEFENSES:
                    specs.append(dict(problem="large", M=M, seed=seed, attack=atk, defense_name=dname,
                                      defense=defense_dict(dname), with_refs=True))
    df = run_jobs(specs, args.workers, "large")
    df = add_reductions(df, ["seed", "M", "attack"])
    write(df, out, "large_scale_runs.csv")
    agg = aggregate(df, ["M", "attack", "defense_name"],
                    ["reported_rmse_honest_m", "final_rmse_honest_m", "final_mean_err_honest_m", "final_max_err_honest_m",
                     "tail_mean_rmse_honest_m", "tail_max_rmse_honest_m", "iterations", "converged",
                     "false_alarm_rate", "missed_detection_rate", "ledger_submitted", "ledger_rejected",
                     "total_flagged", "reduction_vs_no_defense_pct", "better_than_prior_only",
                     "ref_prior_only_rmse", "ref_wls_attack_free_rmse", "ref_oracle_wls_rmse", "runtime_s"])
    write(agg, out, "large_scale_table.csv")
    # Compact paper-style table (seed 7 values + mean +/- CI)
    comp = agg[agg.defense_name.isin(["no_defense", "ledger_only", "screening_only", "ledger+screening",
                                     "wmsr_baseline"])]
    keep = ["M", "attack", "defense_name", "n_runs", "converged_frac", "iterations_mean",
            "reported_rmse_honest_m_mean", "reported_rmse_honest_m_ci95",
            "final_rmse_honest_m_mean", "final_rmse_honest_m_ci95", "tail_mean_rmse_honest_m_mean",
            "tail_max_rmse_honest_m_mean", "false_alarm_rate_mean", "missed_detection_rate_mean",
            "reduction_vs_no_defense_pct_mean", "better_than_prior_only_frac"]
    write(comp[[c for c in keep if c in comp]], out, "large_scale_table_compact.csv")


def exp_runtime(out: Path, args, gas_per_tx=279_000, block_gas_limit=30_000_000, block_period_s=1.0):
    """Simulation-side timing and message counts. The ledger projection uses the
    measured gas per transaction of the prototype; replace block_gas_limit and
    block_period_s with the values reported by blockchain_experiments.py."""
    print("\n[runtime] solver time and message counts vs N")
    rows = []
    for N in ([6, 100, 250, 500, 1000, 2000] if not args.quick else [6, 250, 1000]):
        P = six_uav_problem(7) if N == 6 else random_knn_problem(N, 7)
        mal = SIX_MALICIOUS if N == 6 else choose_malicious(N, N // 10, 7)
        for dname in ("no_defense", "ledger+screening", "signed_echo+screening"):
            D = ablation_defenses()[dname]
            r = run_case(P, make_attack("fdi" if dname != "no_defense" else "none", mal), D,
                         SolverConfig(**{**asdict(SOLVER), "store_traces": False}))
            s = r["summary"]
            tx_per_block = max(1, block_gas_limit // gas_per_tx)
            rows.append(dict(N=N, defense=dname, iterations=s["iterations"],
                             solver_time_per_iter_ms=1e3 * s["runtime_s"] / max(1, s["iterations"]),
                             direct_msgs_per_iter=2 * P.n_edges,
                             ledger_tx_per_iter=P.n if D.exchange == "ledger" else 0,
                             echo_msgs_per_iter=s["echo_msgs_per_iter"],
                             projected_blocks_per_iter=int(np.ceil(P.n / tx_per_block)) if D.exchange == "ledger" else 0,
                             projected_ledger_latency_per_iter_s=(np.ceil(P.n / tx_per_block) * block_period_s
                                                                  if D.exchange == "ledger" else 0.0),
                             assumed_gas_per_tx=gas_per_tx, assumed_block_gas_limit=block_gas_limit,
                             assumed_block_period_s=block_period_s))
    df = pd.DataFrame(rows)
    write(df, out, "runtime_scaling_simulation.csv")
    print(df.round(3).to_string())


def write_parameters(out: Path):
    params = dict(
        solver=asdict(SOLVER), six_uav=dict(seed=BASE_SEED, malicious=SIX_MALICIOUS, sigma_g=1.5, sigma_y=0.5,
                                            comm_radius_m=14.0, position_jitter_std=0.8,
                                            initialization="x_i(0) = g_i, edge copies = priors, duals = 0",
                                            targets="all neighbours of each malicious UAV"),
        large=dict(n=LARGE_N, malicious_counts=LARGE_MALICIOUS, area_m=250.0, k_neighbors=8, sigma_g=1.5,
                   sigma_y=0.5, placement="uniform without replacement, rng(seed + 1000 + M)",
                   graph="symmetrized 8-nearest-neighbour graph of true positions"),
        attacks=DEFAULT_ATTACKS, attack_scenarios={k: [v[0], v[1]] for k, v in ATTACKS.items()},
        receiver_dependent_direction="rotation of the base direction by ((131*j + 17*i) mod 360) degrees",
        proposed_screening=PROPOSED_SCREENING, submitted_screening=SUBMITTED_SCREENING,
        monte_carlo_seeds="7, 8, 9, ... (seed 7 = submitted network)",
        metrics=dict(rmse="honest UAVs only", tail_window=SOLVER.tail_window,
                     convergence="RMS own-copy and screened primal residuals < eps_primal and RMS dual < eps_dual"),
    )
    (out / "parameters_used.json").write_text(json.dumps(params, indent=2, default=str))


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+",
                    choices=["all", "sanity", "six", "six_mc", "threshold", "adaptive", "large", "runtime"])
    ap.add_argument("--out", default="results_revision")
    ap.add_argument("--seeds", type=int, default=10, help="Monte Carlo runs (seeds 7, 8, ...)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--quick", action="store_true", help="smaller grids for a fast check")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_parameters(out)
    exps = ["sanity", "six", "six_mc", "threshold", "adaptive", "large", "runtime"] if "all" in args.experiments \
        else args.experiments
    funcs = dict(sanity=exp_sanity, six=exp_six, six_mc=exp_six_mc, threshold=exp_threshold,
                 adaptive=exp_adaptive, large=exp_large, runtime=exp_runtime)
    for e in exps:
        funcs[e](out, args)
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
