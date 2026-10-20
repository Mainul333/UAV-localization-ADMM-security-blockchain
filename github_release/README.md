# Blockchain-Assisted Secure Distributed UAV Localization: code and results of the revision

This repository contains everything needed to verify and reproduce the results of the revised manuscript.
It replaces the archive of the original submission (`uav_large_scale_csv_results.zip`, seeds 41-60),
which does not correspond to the revised results.

## Contents
| Folder | Content |
|---|---|
| `code/simulation/` | `secure_localization.py` (secure ADMM, attacks, defenses), `run_experiments.py` (all simulations of Tables 4, 6-9 and Figures 3-6), `exp_disruption_horizon.py` (fixed-horizon disruption check, Section 7.3) |
| `code/blockchain/` | Geth Clique prototype: `RUN_ME.sh` (on-chain runs, Table 5), `RUN_LOADTEST.sh` (default load test), `RESET_CHAIN.sh` + `RUN_FASTTEST.sh` (optimized load test, Table 10), smart-contract artifact `UAVConsistencyLedger.json` |
| `code/figures_tables/` | `reproduce.sh`, `make_figures.py`, `make_tables.py`: regenerate Figures 3-6 and Tables 4-10 from the result archives; `MANIFEST.sha256` lists checksums of all inputs and outputs |
| `results/` | `results_revision.zip` (all simulations, per-run CSV files, seeds 7-16), `blockchain_results_20261019_115609.zip` (on-chain runs with per-iteration logs), `loadtest_results_20261019_225159.zip` (default load test, per-round data and node logs), `fasttest_results_20261020_014341.zip` (optimized load test), `disruption_fixed_horizon/` (400-iteration disruption runs) |

## Reproducing the figures and tables
```bash
pip install -r requirements.txt
cd results && for z in *.zip; do unzip -q "$z"; done && cd ..
bash code/figures_tables/reproduce.sh results/results_revision results/blockchain_run/results_blockchain \
     results/loadtest_run/results_blockchain results/fasttest_run/results_blockchain reproduced
```
The outputs are byte-identical to the files listed in `MANIFEST.sha256`.

## Re-running the experiments
* Simulations: `cd code/simulation && python run_experiments.py all --seeds 10 --out results_rerun`
  (seeds 7-16; parameters in Table 3 of the paper; single experiments: `sanity`, `six`, `six_mc`, `threshold`, `adaptive`, `large`, `runtime`).
* Fixed-horizon disruption check: `cd code/simulation && python exp_disruption_horizon.py` (about 5 minutes; writes `results_disruption_horizon_*.csv`).
* Blockchain: Geth 1.13.14 with the six-node Clique network (chain id 12345, period 1 s); see `code/blockchain/README.txt`.
  The optimized load test re-initializes the chain with a 350-million gas limit (`RESET_CHAIN.sh`).

Software versions used for the figure and table scripts are listed in `requirements.txt`; the blockchain runs used
Python 3.10 on Windows 10 with Geth 1.13.14 (see `provenance.json` inside each blockchain result archive).
No private keys are included; the scripts create or decrypt keys locally and delete them after each run.
