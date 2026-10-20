#!/usr/bin/env bash
# =============================================================================
# RUN_FASTTEST.sh - optimized blockchain load test (latency reduction).
# Starts all six nodes with a raised block gas limit target (--miner.gaslimit),
# waits until the chain's gas limit has risen (about 40 min the first time only),
# then measures one iteration with 6 to 1000 UAVs when every UAV signs its own
# record and each record is broadcast to all validators.
# Put it in the uav_blockchain_run folder (next to RUN_ME.sh) and run:
#     bash RUN_FASTTEST.sh
# Upload fasttest_results_<date>.zip at the end.
# Run RESET_CHAIN.sh once before, if the network does not produce blocks steadily.
# The nodes run in archive mode, so an abrupt stop no longer rewinds the chain.
#
# (Start-up steps shared with RUN_ME.sh:)
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
LOAD_SIZES="${LOAD_SIZES:-6,25,50,100,200,500,1000}"
GAS_TARGET="${GAS_TARGET:-350000000}"      # target block gas limit for all nodes
GAS_WAIT_MAX="${GAS_WAIT_MAX:-5400}"      # maximum wait for the gas limit to rise (s)
LOAD_ROUNDS="${LOAD_ROUNDS:-5}"
FUND_ETH="${FUND_ETH:-0.2}"
ID_OFFSET="${ID_OFFSET:-10000}"
KEEP_NODES="${KEEP_NODES:-0}"                   # 1 = leave the nodes running at the end
BACKUP_CHAIN="${BACKUP_CHAIN:-1}"               # 1 = copy node1..node6 once before the first run
START_TIMEOUT="${START_TIMEOUT:-900}"

RUN_DIR="fasttest_run"
OUT="$RUN_DIR/results_blockchain"
LOG="$RUN_DIR/logs"
NODE_LOG="$RUN_DIR/node_logs"
PRIV="private"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT" "$LOG" "$NODE_LOG" "$PRIV"

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
      --authrpc.port $((8559 + i)) --miner.gaslimit "$GAS_TARGET" --state.scheme hash --gcmode archive --mine --miner.etherbase "$ADDR" > "$NLOG/node$i.log" 2>&1 &
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

# ------------------------------- 5. wait for the raised gas limit -------------
NEED_GAS=$($PY -c "print(max(int(s) for s in '$LOAD_SIZES'.split(','))*280000+350000)")
say "5/7 Waiting until the block gas limit reaches $NEED_GAS (target $GAS_TARGET)"
echo "Geth raises the limit by about 0.1% per block. From the default 30 M this takes about 40 minutes"
echo "the first time; afterwards the raised limit stays in the chain and later runs start immediately."
$PY - "$RPC" "$NEED_GAS" "$GAS_WAIT_MAX" <<'EOF' || fail "The gas limit did not reach the required value (see above)."
import json, sys, time, urllib.request
rpc, need, tmax = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
def gas_limit():
    req = urllib.request.Request(rpc, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber",
                                 "params": ["latest", False]}).encode(), headers={"Content-Type": "application/json"})
    b = json.loads(urllib.request.urlopen(req, timeout=10).read())["result"]
    return int(b["gasLimit"], 16), int(b["number"], 16)
t0 = time.time(); g0, n0 = gas_limit(); last = 0
while True:
    g, n = gas_limit()
    if g >= need:
        print(f"Block gas limit {g:,} at block {n}: ready."); sys.exit(0)
    el = time.time() - t0
    if el > 120 and g <= g0:
        print(f"The gas limit is not rising ({g:,}). Were the nodes started by this script with --miner.gaslimit?"); sys.exit(1)
    if el > tmax:
        print(f"Timeout after {tmax:.0f} s with gas limit {g:,}."); sys.exit(1)
    if time.time() - last > 30:
        rate = (g / g0) ** (1 / max(1, n - n0)) if n > n0 and g > g0 else 1.000976
        import math
        eta = math.log(need / g) / math.log(rate) if rate > 1 else float("nan")
        print(f"  gas limit {g:>12,} at block {n} ({100 * g / need:5.1f}% of the requirement); about {eta / 60:4.0f} min left", flush=True)
        last = time.time()
    time.sleep(5)
EOF

# ------------------------------- 6. optimized load test -----------------------
say "6/7 Optimized load test: $LOAD_SIZES UAVs, $LOAD_ROUNDS rounds each"
RPCS=$($PY -c "print(','.join(f'http://127.0.0.1:{8545+i}' for i in range(6)))")
$PY fast_load_test.py --rpcs "$RPCS" --artifact UAVConsistencyLedger.json --keys "$PRIV/keys.json" \
    --accounts-file "$PRIV/loadtest_accounts.json" --sizes "$LOAD_SIZES" --rounds "$LOAD_ROUNDS" \
    --id-offset "$ID_OFFSET" --fund-eth "$FUND_ETH" --out "$OUT" 2>&1 | tee "$LOG/fast_load_test.log"
FT_RC=${PIPESTATUS[0]}
[ "$FT_RC" = "0" ] || echo "!! The optimized load test stopped with exit code $FT_RC; partial results (if any) are still packaged."

# ------------------------------- 7. package ----------------------------------
say "7/7 Collecting results"
mkdir -p "$RUN_DIR/node_log_tails"
for f in "$NODE_LOG"/*.log; do [ -f "$f" ] && tail -n 400 "$f" > "$RUN_DIR/node_log_tails/$(basename "$f")"; done
$PY - "$RUN_DIR" "$NET" "$GVER" "$GAS_TARGET" <<'EOF'
import json, os, sys, datetime, platform
run_dir, net, gver, target = sys.argv[1:5]
json.dump(dict(finished=datetime.datetime.now().isoformat(timespec="seconds"), geth=gver, network_folder=net,
               miner_gaslimit_target=int(target), python=sys.version, platform=platform.platform()),
          open(os.path.join(run_dir, "provenance.json"), "w"), indent=2)
EOF
ZIP="fasttest_results_${STAMP}.zip"
$PY - "$RUN_DIR" "$ZIP" <<'EOF'
import os, sys, zipfile
from pathlib import Path
run_dir, zip_name = Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(run_dir.rglob("*")):
        if p.is_dir() or p.parent.name == "node_logs":
            continue
        if p.suffix.lower() in (".json", ".csv", ".txt", ".log") and "private_key" in p.read_text(encoding="utf-8", errors="ignore"):
            continue
        z.write(p, p.relative_to(run_dir.parent))
    for f in ("RUN_FASTTEST.sh", "fast_load_test.py", "blockchain_experiments.py"):
        if os.path.exists(f):
            z.write(f, f"{run_dir.name}/code_used/{f}")
print(f"Created {zip_name} ({os.path.getsize(zip_name) / 1e6:.2f} MB)")
EOF
echo
echo "Upload this file: $PKG/$ZIP"
