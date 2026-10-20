"""Fixed-horizon disruption experiment (response to the switching-period concern).

Runs the disruption attacks over a fixed horizon of 400 iterations (two complete
switching cycles, L = 100) without early stopping, for the proposed defense and the
blockchain-only configuration, with 100/200/300 compromised UAVs and seeds 7-16.
Reports the RMSE just before the first switch, the mean and maximum RMSE after the
first switch (iterations 101-400), and the first iteration that meets the stopping
tolerance used in the paper (1e-4).
"""
import sys, time, numpy as np, pandas as pd
from dataclasses import asdict
import run_experiments as rx
from secure_localization import make_attack, DefenseConfig, SolverConfig, run_case

H = 400
rows = []
t0 = time.time()
SEEDS = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(7, 17)
for seed in SEEDS:
    for M in (100, 200, 300):
        P = rx.make_problem("large", seed); mal = rx.malicious_for("large", P, seed, M)
        for atk in ("disruption", "disruption_single"):
            kind, akw = rx.ATTACKS[atk]
            for dname in ("ledger+screening", "ledger_only"):
                A = make_attack(kind, mal, **akw)
                D = DefenseConfig(**rx.defense_dict(dname))
                S = SolverConfig(**{**asdict(rx.SOLVER), "max_iter": H, "eps_primal": 0.0, "eps_dual": 0.0, "store_traces": True})
                tr = run_case(P, A, D, S)["traces"]
                r = np.asarray(tr["rmse_honest_m"])
                ok = (np.asarray(tr["primal_own"]) < 1e-4) & (np.asarray(tr["primal_screened"]) < 1e-4) & (np.asarray(tr["dual"]) < 1e-4)
                first = int(np.argmax(ok)) + 1 if ok.any() else -1
                rows.append(dict(seed=seed, M=M, attack=atk, defense=dname, horizon=len(r),
                                 rmse_it100=r[99], mean_rmse_101_400=r[100:].mean(), max_rmse_101_400=r[100:].max(),
                                 mean_rmse_1_400=r.mean(), final_rmse=r[-1], first_tolerance_iteration=first))
    print(f"seed {seed} done, {time.time()-t0:.0f}s", flush=True)
d = pd.DataFrame(rows)
import os
if len(sys.argv) > 2 and os.path.exists("results_disruption_horizon_runs.csv") and int(sys.argv[1]) != 7:
    d = pd.concat([pd.read_csv("results_disruption_horizon_runs.csv"), d], ignore_index=True)
d.to_csv("results_disruption_horizon_runs.csv", index=False)
g = d.groupby(["attack", "defense", "M"]).agg(rmse_it100=("rmse_it100", "mean"), mean_after=("mean_rmse_101_400", "mean"),
                                               max_after=("max_rmse_101_400", "mean"), max_after_worst=("max_rmse_101_400", "max"),
                                               final=("final_rmse", "mean"), first_tol=("first_tolerance_iteration", "mean"),
                                               runs=("seed", "count")).reset_index()
g.to_csv("results_disruption_horizon_summary.csv", index=False)
print(g.round(3).to_string(index=False))
