#!/usr/bin/env bash
# =============================================================================
# RUN_ME.sh - the ONLY file you run. It does everything for the blockchain part:
#
#   1. finds your network folder ("6 Nodes") and backs up node1..node6
#   2. starts the boot node and the six Geth nodes in the background
#   3. waits until blocks are being produced
#   4. builds keys.json from the node folders and checks/registers the contract
#   5. runs all blockchain experiments
#   6. writes blockchain_results_<date>.zip  (upload this file)
#   7. stops the nodes it started
#
# Where to put it: unzip this folder INSIDE "6 Nodes", next to geth.exe:
#     ...\Geth Using Installed Software\6 Nodes\uav_blockchain_run\RUN_ME.sh
# How to run (Git Bash):
#     cd ".../6 Nodes/uav_blockchain_run"
#     bash RUN_ME.sh
# If it stops half way, run it again with:   RESUME=1 bash RUN_ME.sh
# =============================================================================

set -u
cd "$(dirname "$0")" || exit 1
PKG="$(pwd)"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# ------------------------------- settings ------------------------------------
RESUME="${RESUME:-0}"
NET_DIR="${NET_DIR:-}"                          # set only if the network folder is elsewhere
RPC="${RPC:-http://127.0.0.1:8545}"
EXPECTED_NET="${EXPECTED_NET:-}"                # read from command-3.bat when empty
ONCHAIN_MAX_ITER="${ONCHAIN_MAX_ITER:-300}"
KAPPA="${KAPPA:-2.5}"
WARMUP="${WARMUP:-10}"
LOAD_SIZES="${LOAD_SIZES:-6,25,50,100,200}"
LOAD_ROUNDS="${LOAD_ROUNDS:-5}"
FUND_ETH="${FUND_ETH:-0.2}"
ID_OFFSET="${ID_OFFSET:-10000}"
KEEP_NODES="${KEEP_NODES:-0}"                   # 1 = leave the nodes running at the end
BACKUP_CHAIN="${BACKUP_CHAIN:-1}"               # 1 = copy node1..node6 once before the first run
START_TIMEOUT="${START_TIMEOUT:-900}"

RUN_DIR="blockchain_run"
OUT="$RUN_DIR/results_blockchain"
LOG="$RUN_DIR/logs"
NODE_LOG="$RUN_DIR/node_logs"
PRIV="private"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT" "$LOG" "$NODE_LOG" "$PRIV"
rm -f "$LOG/steps.txt"

STARTED=0
PIDS=()

say()  { echo; echo "### $*"; }
fail() { echo; echo "STOPPED: $*"; exit 1; }

show_node_logs() {
  for f in "$NODE_LOG"/bootnode.log "$NODE_LOG"/node*.log; do
    [ -f "$f" ] || continue
    echo "----- last lines of $f"; tail -n 15 "$f"
  done
}

cleanup() {
  if [ "$STARTED" = "1" ] && [ "$KEEP_NODES" != "1" ]; then
    echo; echo "Stopping the nodes started by this script ..."
    for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -INT "$p" 2>/dev/null; done      # ask Geth to shut down cleanly
    for k in $(seq 1 20); do
      alive=0; for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$p" 2>/dev/null && alive=1; done
      [ "$alive" = "0" ] && break; sleep 1
    done
    for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$p" 2>/dev/null && kill "$p" 2>/dev/null; done
    sleep 2
    for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done
  fi
  rm -f "$PRIV/keys.json"
}
trap cleanup EXIT
trap 'echo; echo "Interrupted."; exit 130' INT TERM

# ------------------------------- 1. Python -----------------------------------
say "1/7 Checking Python"
PY=""
for c in python "py -3" python3; do
  if $c -c "import sys; assert sys.version_info >= (3, 8)" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || fail "Python 3.8+ not found. Install Python, or set PYTHON_CMD and rerun."
echo "Python: $($PY --version 2>&1)"
if ! $PY -c "import numpy, scipy, pandas, web3, eth_account" >/dev/null 2>&1; then
  echo "Installing missing packages (numpy, scipy, pandas, web3) ..."
  $PY -m pip install --quiet numpy scipy pandas "web3>=6" || fail "pip install failed"
fi
echo "web3.py $($PY -c 'import web3; print(web3.__version__)')"

# ------------------------------- 2. network folder ---------------------------
say "2/7 Finding the network folder"
is_net() { [ -n "$1" ] && [ -f "$1/boot.key" ] && [ -d "$1/node1" ] && { [ -f "$1/geth.exe" ] || [ -f "$1/geth" ]; }; }
NET=""
for c in "$NET_DIR" ".." "../.." "." "../6 Nodes" "../../6 Nodes"; do
  if is_net "$c"; then NET="$(cd "$c" && pwd)"; break; fi
done
[ -n "$NET" ] || fail "Network folder not found. Unzip uav_blockchain_run INSIDE the '6 Nodes' folder (next to geth.exe), or run with NET_DIR=\"/path/to/6 Nodes\" bash RUN_ME.sh"
echo "Network folder: $NET"
GETH="./geth.exe";     [ -f "$NET/geth.exe" ]     || GETH="./geth"
BOOT="./bootnode.exe"; [ -f "$NET/bootnode.exe" ] || BOOT="./bootnode"
for i in 1 2 3 4 5 6; do
  [ -f "$NET/node$i/password.txt" ] && ls "$NET/node$i/keystore/"* >/dev/null 2>&1 \
    || fail "node$i is missing its keystore or password.txt"
done
GVER="$(cd "$NET" && $GETH version 2>/dev/null | grep -i '^version' | head -1)"
echo "Geth in network folder: ${GVER:-unknown}"
MINOR="$(echo "$GVER" | sed -n 's/.*1\.\([0-9][0-9]*\)\..*/\1/p')"
if [ -n "$MINOR" ] && [ "$MINOR" -ge 14 ]; then
  fail "This Geth ($GVER) is too new for a Clique network. Use the geth.exe that came with your network folder."
fi
CMD3="$(ls "$NET"/command-3.bat 2>/dev/null | head -1)"
ENODE="$(grep -o 'enode://[^ "]*' "$CMD3" 2>/dev/null | head -1)"
[ -n "$EXPECTED_NET" ] || EXPECTED_NET="$(grep -o -- '--networkid [0-9]*' "$CMD3" 2>/dev/null | head -1 | awk '{print $2}')"
[ -n "$EXPECTED_NET" ] || EXPECTED_NET="123454321"
[ -n "$ENODE" ] || fail "Could not read the boot-node address (enode) from command-3.bat"
echo "Network id: $EXPECTED_NET"

# ------------------------------- 3. start the network ------------------------
say "3/7 Starting the blockchain network"
$PY chain_status.py probe "$RPC" > "$LOG/probe.txt" 2>/dev/null
PROBE=$?
if [ "$PROBE" = "0" ]; then
  RUNNING_NET="$(cat "$LOG/probe.txt")"
  if [ "$RUNNING_NET" = "$EXPECTED_NET" ]; then
    echo "Your network is already running on $RPC; using it (it will be left running)."
  else
    fail "Port 8545 is used by another Ethereum node (network id $RUNNING_NET), for example the mainnet Geth. Close it (Ctrl+C in its window) and run again."
  fi
elif [ "$PROBE" = "3" ]; then
  fail "Port 8545 is used by another program (for example stunnel from command-1.bat). Close it and run again."
else
  if [ "$BACKUP_CHAIN" = "1" ] && [ ! -d "$NET/backup_before_revision" ]; then
    echo "Backing up node1..node6 to $NET/backup_before_revision (one time) ..."
    mkdir -p "$NET/backup_before_revision" && \
      cp -r "$NET"/node1 "$NET"/node2 "$NET"/node3 "$NET"/node4 "$NET"/node5 "$NET"/node6 "$NET/backup_before_revision/" \
      || fail "Backup failed (disk space?)"
  fi
  NLOG="$PKG/$NODE_LOG"
  cd "$NET" || fail "cannot enter $NET"
  echo "Starting boot node ..."
  $BOOT -nodekey boot.key -addr :30305 > "$NLOG/bootnode.log" 2>&1 &
  PIDS+=($!); STARTED=1
  sleep 3
  for i in 1 2 3 4 5 6; do
    KS="$(ls node$i/keystore/* | head -1)"
    ADDR="0x$(sed -n 's/.*"address" *: *"\([0-9a-fA-F]*\)".*/\1/p' "$KS")"
    API="eth,net,web3,personal,admin"; [ "$i" = "1" ] && API="eth,net,web3,personal,clique,admin"
    echo "Starting node$i (signer $ADDR) ..."
    $GETH --datadir "node$i" --port $((30305 + i)) --ipcdisable --bootnodes "$ENODE" --networkid "$EXPECTED_NET" \
      --nat extip:127.0.0.1 --netrestrict 127.0.0.0/8 \
      --http --http.addr 127.0.0.1 --http.port $((8544 + i)) --http.api "$API" \
      --allow-insecure-unlock --unlock "$ADDR" --password "node$i/password.txt" \
      --authrpc.port $((8559 + i)) --mine --miner.etherbase "$ADDR" > "$NLOG/node$i.log" 2>&1 &
    PIDS+=($!)
    sleep 1
  done
  cd "$PKG" || exit 1
  echo "Waiting for the nodes to answer (up to $START_TIMEOUT s) ..."
  if ! $PY chain_status.py wait_http "$RPC" "$EXPECTED_NET" "$START_TIMEOUT"; then
    show_node_logs
    fail "The nodes did not start. Copy the lines above into the chat."
  fi
fi
echo "Connecting the nodes and waiting for steady block production (up to $START_TIMEOUT s)."
echo "If the nodes were stopped abruptly last time, they first re-process their stored blocks; this is normal and can take several minutes."
if ! $PY chain_status.py mesh 8545 6 "$START_TIMEOUT"; then
  show_node_logs
  fail "The network is not producing blocks steadily. Copy the lines above into the chat."
fi

# ------------------------------- 4. keys and contract ------------------------
say "4/7 Preparing keys and contract"
$PY prepare_chain.py --rpc "$RPC" --net-dir "$NET" --artifact UAVConsistencyLedger.json \
  --keys-out "$PRIV/keys.json" --report "$OUT/chain_preparation.json" || fail "Chain preparation failed (see above)."
$PY check_geth.py --rpc "$RPC" --keys "$PRIV/keys.json" --artifact UAVConsistencyLedger.json \
  --wait 6 --max-iter "$ONCHAIN_MAX_ITER" || { show_node_logs; fail "Chain check failed (see the FAIL lines above)."; }

# ------------------------------- 5. experiments ------------------------------
say "5/7 Running the blockchain experiments (about 30-60 min)"
record() { echo "$1|$2|$3" >> "$LOG/steps.txt"; }
run_step() {
  local name="$1"; shift
  local done_flag="$LOG/.done_$name"
  if [ "$RESUME" = "1" ] && [ -f "$done_flag" ]; then
    echo "[$name] already finished, skipped"; record "$name" "OK (earlier run)" 0; return 0
  fi
  echo; echo "=== [$name] $(date '+%H:%M:%S')"
  local t0; t0=$(date +%s)
  "$@" 2>&1 | tee "$LOG/$name.log"
  local rc=${PIPESTATUS[0]} dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ]; then touch "$done_flag"; record "$name" "OK" "$dt"
  else record "$name" "FAILED (exit $rc; see logs/$name.log)" "$dt"; echo "!! $name failed; continuing."; fi
}
COMMON="--provider $RPC --artifact UAVConsistencyLedger.json --out $OUT --keys $PRIV/keys.json"
ADMM="--max-iter $ONCHAIN_MAX_ITER --kappa $KAPPA --warmup $WARMUP"
BE="blockchain_experiments.py"
run_step bc_01_metadata            $PY $BE metadata     $COMMON
run_step bc_02_equivocation        $PY $BE equivocation $COMMON
run_step bc_03_admm_attack_free    $PY $BE secure_admm  $COMMON $ADMM --attack none       --defense ledger+screening
run_step bc_04_admm_fdi_undefended $PY $BE secure_admm  $COMMON $ADMM --attack fdi        --defense none
run_step bc_05_admm_fdi            $PY $BE secure_admm  $COMMON $ADMM --attack fdi        --defense ledger+screening
run_step bc_06_admm_byz_ledger     $PY $BE secure_admm  $COMMON $ADMM --attack byzantine  --commit-policy equivocate --defense ledger
run_step bc_07_admm_byz            $PY $BE secure_admm  $COMMON $ADMM --attack byzantine  --commit-policy equivocate --defense ledger+screening
run_step bc_08_admm_byz_single     $PY $BE secure_admm  $COMMON $ADMM --attack byzantine  --commit-policy single     --defense ledger+screening
run_step bc_09_admm_disruption     $PY $BE secure_admm  $COMMON $ADMM --attack disruption --commit-policy equivocate --defense ledger+screening
run_step bc_10_admm_adaptive       $PY $BE secure_admm  $COMMON $ADMM --attack adaptive   --defense ledger+screening
run_step bc_11_load_test           $PY $BE load_test    $COMMON --sizes "$LOAD_SIZES" --rounds "$LOAD_ROUNDS" \
                                   --fund-eth "$FUND_ETH" --id-offset "$ID_OFFSET" --accounts-file "$PRIV/loadtest_accounts.json"

# ------------------------------- 6. overview and zip -------------------------
say "6/7 Collecting results"
mkdir -p "$RUN_DIR/node_log_tails"
for f in "$NODE_LOG"/*.log; do [ -f "$f" ] && tail -n 300 "$f" > "$RUN_DIR/node_log_tails/$(basename "$f")"; done
$PY - "$OUT" "$RUN_DIR" "$NET" "$STARTED" "$GVER" <<'EOF'
import glob, json, os, platform, sys, datetime, importlib
import pandas as pd
out, run_dir, net, started, gver = sys.argv[1:6]
rows = []
for f in sorted(glob.glob(os.path.join(out, "onchain_secure_admm_*_summary.json"))):
    s = json.load(open(f))
    rows.append({k: s.get(k) for k in ("attack", "commit_policy", "defense_option", "iterations", "converged",
                                       "final_rmse_honest_m", "offchain_simulation_rmse_honest_m",
                                       "max_abs_diff_onchain_vs_simulation_m", "total_equivocation_events",
                                       "mean_iteration_wall_s", "end_to_end_s")})
if rows:
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "onchain_summary_all_runs.csv"), index=False)
    pd.set_option("display.width", 250); print(df.round(6).to_string(index=False))
pk = {m: getattr(importlib.import_module(m), "__version__", None) for m in ("numpy", "scipy", "pandas", "web3")}
json.dump(dict(finished=datetime.datetime.now().isoformat(timespec="seconds"), geth=gver,
               network_folder=net, nodes_started_by_script=started == "1", python=sys.version,
               platform=platform.platform(), packages=pk), open(os.path.join(run_dir, "provenance.json"), "w"), indent=2)
EOF
ZIP="blockchain_results_${STAMP}.zip"
$PY - "$RUN_DIR" "$ZIP" <<'EOF'
import os, sys, zipfile
from pathlib import Path
run_dir, zip_name = Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(run_dir.rglob("*")):
        if p.is_dir() or p.name.startswith(".done_") or p.parent.name == "node_logs":
            continue
        if p.suffix.lower() in (".json", ".csv", ".txt", ".log"):
            if "private_key" in p.read_text(encoding="utf-8", errors="ignore"):
                continue
        z.write(p, p.relative_to(run_dir.parent))
    for f in ("RUN_ME.sh", "blockchain_experiments.py", "secure_localization.py", "prepare_chain.py",
              "check_geth.py", "chain_status.py", "UAVConsistencyLedger.json"):
        if os.path.exists(f):
            z.write(f, f"{run_dir.name}/code_used/{f}")
print(f"Created {zip_name} ({os.path.getsize(zip_name) / 1e6:.2f} MB)")
EOF

# ------------------------------- 7. summary ----------------------------------
say "7/7 Summary"
printf "%-28s %-50s %8s\n" "step" "status" "seconds"
while IFS='|' read -r n s t; do printf "%-28s %-50s %8s\n" "$n" "$s" "$t"; done < "$LOG/steps.txt"
if grep -q FAILED "$LOG/steps.txt"; then
  echo; echo "Some steps failed. Run again with:  RESUME=1 bash RUN_ME.sh   (finished steps are skipped)"
fi
echo
echo "Upload this file: $PKG/$ZIP"
