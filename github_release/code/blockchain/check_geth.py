"""
check_geth.py - read-only check that your Geth node is ready for run_blockchain.sh

    python check_geth.py                       # defaults below
    python check_geth.py --rpc http://127.0.0.1:8545 --keys keys.json --artifact UAVConsistencyLedger.json

Each line is PASS, WARN or FAIL with a hint. Exit code 0 means no FAIL.
Nothing is sent to the chain.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from web3 import Web3

FAILS = []


def say(status, msg, hint=None):
    print(f"[{status}] {msg}")
    if hint:
        print(f"       -> {hint}")
    if status == "FAIL":
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", default="http://127.0.0.1:8545")
    ap.add_argument("--keys", default="keys.json")
    ap.add_argument("--artifact", default="UAVConsistencyLedger.json")
    ap.add_argument("--wait", type=int, default=15, help="seconds to watch for new blocks")
    ap.add_argument("--max-iter", type=int, default=300)
    args = ap.parse_args()

    # 1. Connection ----------------------------------------------------------
    w3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={"timeout": 5}))
    try:
        resp = w3.provider.make_request("web3_clientVersion", [])
        client = resp.get("result")
        if client is None:
            raise RuntimeError(resp.get("error"))
        say("PASS", f"Connected to {args.rpc}: {client}")
    except Exception as exc:
        text = str(exc)
        if "refused" in text.lower() or "10061" in text or "Max retries" in text:
            hint = ("Nothing is listening there. Start Geth with HTTP enabled, e.g. add "
                    "--http --http.addr 127.0.0.1 --http.port 8545 --http.api eth,net,web3,clique "
                    "to the command you used for the Table 3 run. If Geth runs on Windows and this "
                    "terminal is WSL, use Git Bash instead (127.0.0.1 in WSL is not the Windows host).")
        elif "timed out" in text.lower():
            hint = "The port answers too slowly; check that the address/port belongs to Geth."
        else:
            hint = "Check the RPC URL and that Geth was started with --http."
        say("FAIL", f"Cannot reach {args.rpc} ({text[:120]})", hint)
        return finish()

    try:
        from web3.middleware import ExtraDataToPOAMiddleware as poa
    except ImportError:
        from web3.middleware import geth_poa_middleware as poa
    w3.middleware_onion.inject(poa, layer=0)

    # 2. Network -------------------------------------------------------------
    art = json.loads(Path(args.artifact).read_text(encoding="utf-8")) if Path(args.artifact).exists() else None
    if art is None:
        say("FAIL", f"{args.artifact} not found", "Copy the Truffle artifact into this folder.")
        return finish()
    net = str(w3.net.version)
    nets = list(art.get("networks", {}).keys())
    if net in nets:
        say("PASS", f"Network id {net} matches the artifact (chain id {w3.eth.chain_id})")
    else:
        say("FAIL", f"Connected to network id {net}, but the artifact has a contract only for {nets}",
            "Start the same private network (same genesis/datadir), or redeploy with 'truffle migrate' "
            "and copy the new UAVConsistencyLedger.json here.")
        return finish()

    # 3. Block production ----------------------------------------------------
    b0 = w3.eth.block_number
    blk = w3.eth.get_block("latest")
    age = time.time() - blk.timestamp
    time.sleep(args.wait)
    b1 = w3.eth.block_number
    if b1 > b0:
        say("PASS", f"Blocks are being produced ({b1 - b0} new in {args.wait} s; latest #{b1})")
    else:
        say("FAIL", f"No new blocks in {args.wait} s (latest #{b0}, sealed {age:.0f} s ago)",
            "No signer is sealing, or the nodes are not connected to each other (with 6 signers at least "
            "4 must be connected). Check the peer counts; start the signers with --mine and the signer "
            "account unlocked.")
    try:
        signers = w3.provider.make_request("clique_getSigners", ["latest"]).get("result")
        say("PASS" if signers else "WARN", f"Clique signers: {signers}",
            None if signers else "Add 'clique' to --http.api so the signer set can be reported (Reviewer 3, C4c).")
    except Exception:
        say("WARN", "clique_getSigners unavailable", "Add 'clique' to --http.api to record the signer set.")
    say("PASS", f"Block gas limit {blk.gasLimit:,}; about {blk.gasLimit // 279_000} state records fit in one block")

    # 4. Contract ------------------------------------------------------------
    address = Web3.to_checksum_address(art["networks"][net]["address"])
    if len(w3.eth.get_code(address)) == 0:
        say("FAIL", f"No contract code at {address}",
            "The chain was probably reset. Redeploy with 'truffle migrate --reset' and copy the new "
            "UAVConsistencyLedger.json here.")
        return finish()
    c = w3.eth.contract(address=address, abi=art["abi"])
    say("PASS", f"Contract found at {address}")

    # 5. Keys, owner, registration, balances --------------------------------
    if not Path(args.keys).exists():
        say("FAIL", f"{args.keys} not found",
            "Run: python make_keys_json.py --notebook Distributed_UAV_Localization.ipynb")
        return finish()
    keys = json.loads(Path(args.keys).read_text())
    if len(keys) < 6:
        say("FAIL", f"{args.keys} has {len(keys)} accounts; six are needed")
        return finish()
    owner = c.functions.owner().call()
    u0 = Web3.to_checksum_address(keys[0]["address"])
    if owner == u0:
        say("PASS", f"U0 ({u0}) is the contract owner")
    else:
        say("FAIL", f"Contract owner is {owner}, but U0 in {args.keys} is {u0}",
            "Put the deployer account first in keys.json (it registers the load-test accounts).")
    gp = w3.eth.gas_price
    need = args.max_iter * 2 * 300_000 * gp
    for i, k in enumerate(keys[:6]):
        a = Web3.to_checksum_address(k["address"])
        reg, uid = c.functions.uavsByAddress(a).call()[:2]
        bal = w3.eth.get_balance(a)
        if reg and int(uid) == i:
            say("PASS", f"U{i} registered as UAV {uid}; balance {Web3.from_wei(bal, 'ether'):.3f} ETH")
        elif not reg:
            holder = c.functions.addressOfUav(i).call()
            if int(holder, 16) == 0:
                say("WARN", f"U{i} not registered yet", "The script will register it (owner key needed).")
            else:
                say("FAIL", f"UAV id {i} belongs to {holder}, not U{i} ({a})", "Check the key order in keys.json.")
        else:
            say("FAIL", f"U{i} is registered as UAV {uid}, expected {i}", "Check the key order in keys.json.")
        if bal < need:
            say("WARN", f"U{i} may run out of gas funds (has {Web3.from_wei(bal, 'ether'):.3f} ETH, "
                        f"about {Web3.from_wei(need, 'ether'):.3f} ETH needed at the current gas price)",
                "Send ETH to this account from a funded account.")
    return finish()


def finish():
    print()
    if FAILS:
        print(f"{len(FAILS)} problem(s) found. Fix the FAIL items above and run again.")
        sys.exit(1)
    print("Geth is ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
