"""
fast_load_test.py - optimized blockchain load test (latency reduction)

Measures the time that one ADMM iteration spends on the blockchain when the
protocol is deployed as intended rather than driven from one sequential client:

  1. Raised block gas limit. All nodes are started with --miner.gaslimit (see
     RUN_FASTTEST.sh), so that the state records of one iteration fit into one
     block instead of 10.
  2. Parallel signing. In a swarm, every UAV signs its own record on its own
     processor. This is emulated by signing all records before the timed window;
     the signing time per record is measured and reported separately.
  3. Broadcast submission. Every signed record is sent to all validators in
     JSON-RPC batches, so that whichever signer seals the next block already
     holds the record (the previous test sent everything to one node, and the
     blocks sealed by other signers were often empty).

The smart contract, the record format, and the gas per record are unchanged.

Timed quantities per round (N UAVs, one record each):
  broadcast_s   time to hand all signed records to the validators
  inclusion_s   time until the block containing the last record is observed
  read_s        time to read all receipts (status, gas) back
  latency_s     inclusion_s + read_s = time of one blockchain iteration
"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from web3 import Web3

import blockchain_experiments as bx


def rpc_batch(url, calls, chunk=200, timeout=120):
    """Send JSON-RPC calls as batches; returns the results (or error dicts) in order."""
    out = []
    for s in range(0, len(calls), chunk):
        part = calls[s:s + chunk]
        payload = [{"jsonrpc": "2.0", "id": s + i, "method": m, "params": p} for i, (m, p) in enumerate(part)]
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        resp = sorted(r.json(), key=lambda x: x["id"])
        out += [x.get("result", {"error": x.get("error")}) for x in resp]
    return out


def as_int(v):
    return int(v, 16) if isinstance(v, str) else int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpcs", default=",".join(f"http://127.0.0.1:{8545 + i}" for i in range(6)),
                    help="comma-separated HTTP endpoints of all validators (first one is used for reading)")
    ap.add_argument("--artifact", default="UAVConsistencyLedger.json")
    ap.add_argument("--address", default=None)
    ap.add_argument("--keys", required=True, help="keys of the six node accounts (index 0 = contract owner)")
    ap.add_argument("--accounts-file", required=True)
    ap.add_argument("--sizes", default="6,25,50,100,200,500,1000")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--id-offset", type=int, default=10000)
    ap.add_argument("--fund-eth", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=200, help="records per JSON-RPC batch request")
    ap.add_argument("--out", default="fasttest_results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rpcs = [u.strip() for u in args.rpcs.split(",") if u.strip()]
    sizes = sorted(int(s) for s in args.sizes.split(","))

    w3, tester = bx.connect(rpcs[0])
    keys = json.loads(Path(args.keys).read_text())
    owner = bx.Signer(w3, keys[0]["address"], keys[0]["private_key"])
    contract, _ = bx.load_contract(w3, argparse.Namespace(artifact=args.artifact, address=args.address), owner, tester)
    chain_id = int(w3.eth.chain_id)

    reachable = []
    for u in rpcs:
        try:
            rpc_batch(u, [("eth_blockNumber", [])]); reachable.append(u)
        except Exception as e:
            print(f"  note: {u} not reachable ({e.__class__.__name__}); it is skipped for broadcasting")
    print(f"Broadcasting to {len(reachable)} validator endpoint(s): {', '.join(reachable)}")

    # accounts: reuse the funded and registered load-test accounts, add more if needed
    accts = bx.load_or_create_accounts(Path(args.accounts_file), max(sizes), w3)
    fund_wei = Web3.to_wei(args.fund_eth, "ether")
    balances = rpc_batch(rpcs[0], [("eth_getBalance", [a["address"], "latest"]) for a in accts])
    hs = [owner.send(to=Web3.to_checksum_address(a["address"]), value=fund_wei)
          for a, b in zip(accts, balances) if as_int(b) < fund_wei // 2]
    if hs:
        print(f"Funding {len(hs)} accounts ..."); bx.wait(w3, hs)
    bx.ensure_registered(w3, contract, owner, [a["address"] for a in accts],
                         [args.id_offset + i for i in range(len(accts))], sequential=tester)
    nonces = [as_int(n) for n in rpc_batch(rpcs[0], [("eth_getTransactionCount", [a["address"], "pending"]) for a in accts])]

    gas_limit = int(w3.eth.get_block("latest").gasLimit)
    need = int(max(sizes) * 280_000 + bx.GAS_RECORD)
    print(f"Block gas limit: {gas_limit:,} (one block holds about {(gas_limit - bx.GAS_RECORD) // 279_000 + 1} records; "
          f"{max(sizes)} records need {need:,})")
    if gas_limit < need:
        print("  note: the gas limit is below the requirement; the records of the largest sizes will span several blocks")

    rows = []
    for N in sizes:
        for rnd in range(args.rounds):
            name, sid = bx.new_session(w3, f"fasttest-{N}")
            gp = int(w3.eth.gas_price)
            states = np.random.default_rng(rnd).normal(scale=100.0, size=(N, 2))
            # ---- parallel signing, emulated before the timed window
            t_sign0 = time.perf_counter()
            raws, hashes = [], []
            for i in range(N):
                rec = bx.make_record(w3, sid, name, 1, args.id_offset + i, states[i])
                tx = bx.submit_fn(contract, sid, 1, rec).build_transaction(
                    {"from": Web3.to_checksum_address(accts[i]["address"]), "nonce": nonces[i],
                     "gas": bx.GAS_RECORD, "gasPrice": gp, "chainId": chain_id})
                signed = w3.eth.account.sign_transaction(tx, private_key=accts[i]["private_key"])
                raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
                raws.append(Web3.to_hex(raw)); hashes.append(Web3.to_hex(Web3.keccak(raw)))
                nonces[i] += 1
            sign_s = time.perf_counter() - t_sign0
            pending = set(hashes)
            prev_block = w3.eth.block_number
            # ---- timed window: broadcast to all validators in parallel batches
            t0 = time.perf_counter()
            calls = [("eth_sendRawTransaction", [r]) for r in raws]
            with ThreadPoolExecutor(max_workers=len(reachable)) as pool:
                results = list(pool.map(lambda u: rpc_batch(u, calls, chunk=args.batch), reachable))
            broadcast_s = time.perf_counter() - t0
            rejected = [r for res in results[:1] for r in res if isinstance(r, dict) and r.get("error")
                        and not any(w in str(r["error"]).lower() for w in ("known", "imported", "already"))]
            if rejected:
                print(f"  warning: {len(rejected)} records rejected by the first node, e.g. {rejected[0]['error']}")
            # ---- wait for inclusion: inspect every new block
            blocks, cur, t_incl = [], prev_block, None
            while pending and time.perf_counter() - t0 < 600:
                head = w3.eth.block_number
                while cur < head:
                    cur += 1
                    b = w3.eth.get_block(cur)
                    ours = [Web3.to_hex(h) for h in b.transactions if Web3.to_hex(h) in pending]
                    if ours:
                        pending -= set(ours)
                        blocks.append(dict(number=cur, ts=int(b.timestamp), n=len(ours), size=int(b["size"]),
                                           gas_used=int(b.gasUsed), gas_limit=int(b.gasLimit)))
                if pending:
                    time.sleep(0.05)
            t_incl = time.perf_counter() - t0
            if pending:
                print(f"  N={N} round {rnd}: {len(pending)} records not included after 600 s"); 
            # ---- read the records back (receipts in batches)
            t_r = time.perf_counter()
            receipts = rpc_batch(rpcs[0], [("eth_getTransactionReceipt", [h]) for h in hashes], chunk=args.batch)
            read_s = time.perf_counter() - t_r
            ok = [r for r in receipts if isinstance(r, dict) and r.get("status") is not None]
            failed = sum(1 for r in ok if as_int(r["status"]) != 1) + (len(hashes) - len(ok))
            gas_used = [as_int(r["gasUsed"]) for r in ok]
            first_ts = w3.eth.get_block(blocks[0]["number"] - 1).timestamp if blocks else 0
            span = max(1, (blocks[-1]["ts"] - first_ts)) if blocks else float("nan")
            latency = t_incl + read_s
            rows.append(dict(N=N, round=rnd, sign_total_s=sign_s, sign_per_record_ms=1000 * sign_s / N,
                             broadcast_s=broadcast_s, inclusion_s=t_incl, read_s=read_s, latency_s=latency,
                             client_tps=N / latency, onchain_tps=N / span, chain_inclusion_time_s=span,
                             blocks_spanned=len(blocks), max_tx_per_block=max((b["n"] for b in blocks), default=0),
                             blocks_until_first_inclusion=(blocks[0]["number"] - prev_block) if blocks else -1,
                             block_size_max_bytes=max((b["size"] for b in blocks), default=0),
                             block_gas_limit=blocks[-1]["gas_limit"] if blocks else gas_limit,
                             mean_gas_used=float(np.mean(gas_used)) if gas_used else float("nan"),
                             failed_tx=int(failed), validators_broadcast=len(reachable)))
            r = rows[-1]
            print(f"N={N:5d} round {rnd}: latency {latency:6.2f} s (broadcast {broadcast_s:5.2f}, inclusion {t_incl:5.2f}, "
                  f"read {read_s:4.2f}) | {r['blocks_spanned']} block(s), max {r['max_tx_per_block']} records/block | "
                  f"on-chain TPS {r['onchain_tps']:6.1f} | signing {r['sign_per_record_ms']:.2f} ms/record | failed {failed}",
                  flush=True)
    with open(out / "fast_load_test_rounds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import pandas as pd
    d = pd.DataFrame(rows)
    summ = d.groupby("N").agg(rounds=("round", "count"), latency_mean_s=("latency_s", "mean"), latency_std_s=("latency_s", "std"),
                              broadcast_mean_s=("broadcast_s", "mean"), inclusion_mean_s=("inclusion_s", "mean"),
                              read_mean_s=("read_s", "mean"), client_tps_mean=("client_tps", "mean"),
                              onchain_tps_mean=("onchain_tps", "mean"), blocks_spanned_mean=("blocks_spanned", "mean"),
                              max_tx_per_block=("max_tx_per_block", "max"), block_size_max_bytes=("block_size_max_bytes", "max"),
                              block_gas_limit=("block_gas_limit", "max"), sign_per_record_ms=("sign_per_record_ms", "mean"),
                              mean_gas_used=("mean_gas_used", "mean"), failed_tx=("failed_tx", "sum")).reset_index()
    summ.to_csv(out / "fast_load_test_summary.csv", index=False)
    json.dump(dict(rpcs=reachable, sizes=sizes, rounds=args.rounds, gas_per_record_limit=bx.GAS_RECORD,
                   block_gas_limit_at_start=gas_limit), open(out / "fast_load_test_config.json", "w"), indent=2)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
