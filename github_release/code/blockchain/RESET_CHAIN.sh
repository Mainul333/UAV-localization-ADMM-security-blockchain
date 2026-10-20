#!/usr/bin/env bash
# =============================================================================
# RESET_CHAIN.sh - starts a fresh chain on the same six-node network.
# Backs up the chain data of every node (keys, passwords and node identities are
# kept), writes genesis_fast.json = genesis.json with a block gas limit of 350 M,
# and initializes all nodes from it with the hash state scheme. Run it once, with
# all Geth windows closed, then run RUN_FASTTEST.sh.
#
# (Original description of the shared start-up code:)
# RUN_FASTTEST.sh - optimized blockchain load test (latency reduction).
# Starts all six nodes with a raised block gas limit target (--miner.gaslimit),
# waits until the chain's gas limit has risen (about 40 min the first time only),
# then measures one iteration with 6 to 1000 UAVs when every UAV signs its own
# record and each record is broadcast to all validators.
# Put it in the uav_blockchain_run folder (next to RUN_ME.sh) and run:
#     bash RUN_FASTTEST.sh
# Upload fasttest_results_<date>.zip at the end.
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
GAS_GENESIS="${GAS_GENESIS:-350000000}"    # gas limit written into genesis_fast.json
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
say "1/6 Checking Python"
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
say "2/6 Finding the network folder"
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

# ------------------------------- 3. nodes must be stopped ---------------------
say "3/6 Checking that no node is running"
$PY - <<'EOF' || fail "Some nodes are still running. Close all Geth windows (or wait until the previous script has stopped them) and run again."
import json, sys, urllib.request
alive = []
for p in range(8545, 8551):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{p}", data=b'{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}',
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2); alive.append(p)
    except Exception:
        pass
if alive:
    print("Nodes answering on ports", alive); sys.exit(1)
print("No node is running.")
EOF

# ------------------------------- 4. back up chain data ------------------------
say "4/6 Backing up the current chain data (keys and passwords stay in place)"
cp UAVConsistencyLedger.json "UAVConsistencyLedger_before_reset_${STAMP}.json"
for i in 1 2 3 4 5 6; do
  if [ -d "$NET/node$i/geth" ]; then
    mv "$NET/node$i/geth" "$NET/node$i/geth_backup_${STAMP}" || fail "Could not move node$i/geth (is a node still running?)"
    echo "node$i: chain data moved to node$i/geth_backup_${STAMP}"
  fi
done

# ------------------------------- 5. new genesis -------------------------------
say "5/6 Writing genesis_fast.json (your genesis.json with gas limit $GAS_GENESIS)"
$PY - "$NET/genesis.json" "$NET/genesis_fast.json" "$GAS_GENESIS" <<'EOF' || fail "Could not write genesis_fast.json"
import json, sys
src, dst, gas = sys.argv[1], sys.argv[2], int(sys.argv[3])
g = json.load(open(src))
old = g.get("gasLimit")
g["gasLimit"] = hex(gas)
json.dump(g, open(dst, "w"), indent=2)
print(f"gasLimit {old} -> {hex(gas)} ({gas:,}); chain id {g['config'].get('chainId')}, Clique period "
      f"{g['config'].get('clique', {}).get('period')} s, signers and accounts unchanged")
EOF

# ------------------------------- 6. initialize the nodes ----------------------
say "6/6 Initializing the six nodes from genesis_fast.json (hash state scheme)"
for i in 1 2 3 4 5 6; do
  OUTI="$( (cd "$NET" && $GETH --datadir "node$i" --state.scheme hash init genesis_fast.json) 2>&1 )"
  echo "$OUTI" | grep -q "Successfully wrote genesis state" || { echo "$OUTI" | tail -5; fail "geth init failed for node$i"; }
  # keep the node's network identity
  if [ -f "$NET/node$i/geth_backup_${STAMP}/nodekey" ]; then mkdir -p "$NET/node$i/geth" && cp "$NET/node$i/geth_backup_${STAMP}/nodekey" "$NET/node$i/geth/nodekey"; fi
  echo "node$i: initialized"
done
cat > "$NET/CHAIN_RESET_${STAMP}.txt" <<EOF2
Chain re-initialized on ${STAMP} by RESET_CHAIN.sh
genesis: genesis_fast.json (genesis.json with gasLimit ${GAS_GENESIS})
previous chain data: nodeX/geth_backup_${STAMP}
EOF2
echo
echo "Done. Now run:   bash RUN_FASTTEST.sh"
echo "(The contract is deployed again and the test accounts are funded and registered automatically.)"
