"""
blockchain_experiments.py
=========================

On-chain measurements for the revision. Run against your Geth PoA network, or
use --provider tester for a dry run on an in-memory chain (the contract is then
deployed from the Truffle artifact automatically).

Subcommands
  metadata      Facts the reviewers asked to report: client/Geth version, chain id,
                block period, block gas limit, Clique signers, gas per state record.
                -> Reviewer 3, Comment 4c
  equivocation  Demonstrates the one-record-per-UAV-per-iteration rule on chain:
                first record accepted, a conflicting second record flagged
                (EquivocationDetected, isConsistent == false), an identical
                resubmission harmless. -> Reviewer 1, Comments 1-2; Reviewer 3, Comment 4
  secure_admm   Six-UAV blockchain-integrated run of the revised secure ADMM WITH an
                attack (the malicious UAVs commit false states on chain). Compares the
                on-chain result with the off-chain simulation of the same case and
                reports per-iteration timing versus plain ADMM computation.
                -> Reviewer 2, Comment 5; Reviewer 3, Comments 4-5
  load_test     Latency/throughput of one ADMM iteration (one record per UAV) versus
                the number of UAV accounts, plus a projection to 1000 UAVs.
                -> Reviewer 2, Comment 5; Reviewer 3, Comment 7

Keys
  --keys keys.json : [{"address": "0x...", "private_key": "0x..."}, ...] in UAV order
                     (index 0 must be the contract owner / deployer).
  Without --keys the script uses unlocked node accounts (w3.eth.accounts).
  Keep keys.json and loadtest_accounts.json private; never commit them, and clear
  notebook outputs that print keys.

Examples
  python blockchain_experiments.py metadata     --provider http://127.0.0.1:8545 --keys keys.json
  python blockchain_experiments.py equivocation --provider http://127.0.0.1:8545 --keys keys.json
  python blockchain_experiments.py secure_admm  --provider http://127.0.0.1:8545 --keys keys.json --attack fdi
  python blockchain_experiments.py load_test    --provider http://127.0.0.1:8545 --keys keys.json --sizes 6,25,50,100,200
  python blockchain_experiments.py all          --provider tester
"""

from __future__ import annotations

import argparse
import json
import secrets
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from web3 import Web3

from secure_localization import (
    DefenseConfig, SolverConfig, SecureADMM, PROPOSED_SCREENING, make_attack, run_case, six_uav_problem,
)

SCALE = 10 ** 6            # fixed-point scaling used on chain (1 micrometre)
GAS_RECORD = 350_000       # gas limit per state record (measured use was ~279k)
GAS_REGISTER = 300_000
GAS_DEPLOY = 4_000_000


# =============================================================================
# Connection, signing, contract
# =============================================================================
def connect(provider: str):
    if provider == "tester":
        from web3 import EthereumTesterProvider
        return Web3(EthereumTesterProvider()), True
    w3 = Web3(Web3.HTTPProvider(provider, request_kwargs={"timeout": 300}))
    poa = None
    try:
        from web3.middleware import ExtraDataToPOAMiddleware as poa       # web3 >= 7
    except ImportError:
        try:
            from web3.middleware import geth_poa_middleware as poa         # web3 6
        except ImportError:
            pass
    if poa is not None:
        w3.middleware_onion.inject(poa, layer=0)
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to {provider}")
    return w3, False


class Signer:
    """Sends transactions from one account; local nonce tracking allows several
    transactions from the same account without waiting (thread-safe)."""

    def __init__(self, w3, address, key=None):
        self.w3 = w3
        self.address = Web3.to_checksum_address(address)
        self.key = key
        self._nonce = None
        self._lock = threading.Lock()

    def _next_nonce(self):
        with self._lock:
            if self._nonce is None:
                self._nonce = self.w3.eth.get_transaction_count(self.address, "pending")
            n = self._nonce
            self._nonce += 1
            return n

    def send(self, fn=None, to=None, value=0, gas=GAS_RECORD, gas_price=None):
        w3 = self.w3
        if self.key is None:
            if fn is not None:
                return fn.transact({"from": self.address, "gas": gas})
            return w3.eth.send_transaction({"from": self.address, "to": to, "value": value})
        params = {"from": self.address, "nonce": self._next_nonce(), "gas": gas,
                  "gasPrice": gas_price if gas_price is not None else w3.eth.gas_price,
                  "chainId": int(w3.eth.chain_id)}
        if fn is not None:
            tx = fn.build_transaction(params)
        else:
            tx = dict(params, to=to, value=value, gas=21_000)
        signed = w3.eth.account.sign_transaction(tx, private_key=self.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        self.last_raw_size = len(raw)          # exact size of the signed transaction (bytes)
        return w3.eth.send_raw_transaction(raw)


def get_signers(w3, args):
    if args.keys:
        entries = json.loads(Path(args.keys).read_text())
        return [Signer(w3, e["address"], e["private_key"]) for e in entries]
    return [Signer(w3, a) for a in w3.eth.accounts]


def wait(w3, tx_hashes, timeout=600):
    return [w3.eth.wait_for_transaction_receipt(h, timeout=timeout, poll_latency=0.05) for h in tx_hashes]


def load_contract(w3, args, owner: Signer, tester: bool):
    art = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    address = args.address or art.get("networks", {}).get(str(w3.net.version), {}).get("address")
    if address is None and tester:
        C = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        rc = wait(w3, [owner.send(C.constructor(), gas=GAS_DEPLOY)])[0]
        address = rc.contractAddress
        print(f"Deployed UAVConsistencyLedger on the tester chain at {address} (gas {rc.gasUsed})")
    if address is None:
        raise RuntimeError("Contract address not found; pass --address")
    return w3.eth.contract(address=Web3.to_checksum_address(address), abi=art["abi"]), art


def ensure_registered(w3, contract, owner: Signer, addresses, ids, sequential=False):
    pending = []
    for uav_id, addr in zip(ids, addresses):
        addr = Web3.to_checksum_address(addr)
        if contract.functions.uavsByAddress(addr).call()[0]:
            continue
        holder = contract.functions.addressOfUav(int(uav_id)).call()
        if int(holder, 16) != 0:
            raise RuntimeError(f"UAV id {uav_id} already belongs to {holder}; choose another --id-offset")
        h = owner.send(contract.functions.registerUAV(int(uav_id), addr, f"UAV-{uav_id}"), gas=GAS_REGISTER)
        if sequential:
            wait(w3, [h])
        else:
            pending.append(h)
    if pending:
        wait(w3, pending)
    return True


def events(contract, name, receipt):
    ev = getattr(contract.events, name)()
    try:
        from web3.logs import DISCARD
        return ev.process_receipt(receipt, errors=DISCARD)
    except Exception:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ev.process_receipt(receipt)


def make_record(w3, session_id, session_name, k, uav_id, state):
    xs, ys = int(round(float(state[0]) * SCALE)), int(round(float(state[1]) * SCALE))
    payload = json.dumps(dict(session=session_name, iteration=int(k), uav_id=int(uav_id),
                              x_scaled=xs, y_scaled=ys), sort_keys=True, separators=(",", ":"))
    pdig = w3.keccak(text=payload)
    sdig = Web3.solidity_keccak(["int256", "int256"], [xs, ys])
    nonce = secrets.randbits(128)                       # one-time value, Eq. (32)
    cm = Web3.solidity_keccak(["bytes32", "uint256", "uint256", "bytes32", "uint256"],
                              [session_id, int(k), int(uav_id), pdig, nonce])
    return dict(commitment=cm, payload_digest=pdig, state_digest=sdig, x_scaled=xs, y_scaled=ys)


def submit_fn(contract, session_id, k, rec):
    return contract.functions.submitStateCommitment(session_id, int(k), rec["commitment"], rec["payload_digest"],
                                                    rec["state_digest"], rec["x_scaled"], rec["y_scaled"])


def new_session(w3, prefix):
    name = f"{prefix}-{time.time_ns()}"
    return name, w3.keccak(text=name)


def setup(args):
    w3, tester = connect(args.provider)
    signers = get_signers(w3, args)
    if len(signers) < 6:
        raise RuntimeError("Need at least six UAV accounts (index 0 = contract owner)")
    contract, art = load_contract(w3, args, signers[0], tester)
    ensure_registered(w3, contract, signers[0], [s.address for s in signers[:6]], range(6), sequential=tester)
    return w3, tester, signers, contract, art


# =============================================================================
# metadata
# =============================================================================
def cmd_metadata(args, ctx=None):
    w3, tester, signers, contract, art = ctx or setup(args)
    latest = w3.eth.get_block("latest")
    n0 = max(0, latest.number - 50)
    ts = [w3.eth.get_block(b).timestamp for b in range(n0, latest.number + 1)]
    period = float(np.median(np.diff(ts))) if len(ts) > 2 else None
    try:
        signers_clique = w3.provider.make_request("clique_getSigners", ["latest"]).get("result")
    except Exception:
        signers_clique = None
    session_name, sid = new_session(w3, "gas-estimate")
    rec = make_record(w3, sid, session_name, 1, 1, (0.0, 0.0))
    try:
        gas_est = submit_fn(contract, sid, 1, rec).estimate_gas({"from": signers[1].address})
    except Exception as exc:
        gas_est = f"estimate failed: {exc}"
    meta = dict(
        client_version=w3.client_version, chain_id=int(w3.eth.chain_id), net_version=str(w3.net.version),
        latest_block=int(latest.number), block_gas_limit=int(latest.gasLimit),
        median_block_period_s=period, clique_signers=signers_clique,
        n_clique_signers=len(signers_clique) if signers_clique else None,
        contract_address=contract.address, solc_version=art.get("compiler", {}).get("version"),
        gas_per_state_record_estimate=gas_est,
        notes=("Clique: one in-turn signer seals each block; there is no q-of-M approval certificate. "
               "Report finality as 'included in a sealed block and read after k confirmations'."),
    )
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "chain_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))
    return meta


# =============================================================================
# equivocation demo
# =============================================================================
def cmd_equivocation(args, ctx=None):
    w3, tester, signers, contract, art = ctx or setup(args)
    s = signers[args.uav]
    uid = int(contract.functions.uavsByAddress(s.address).call()[1])
    name, sid = new_session(w3, "equivocation-demo")
    k = 1
    rec_a = make_record(w3, sid, name, k, uid, (1.0, 2.0))
    rec_b = make_record(w3, sid, name, k, uid, (9.0, -4.0))          # conflicting state
    log = []
    for label, rec in [("first record (state A)", rec_a), ("conflicting record (state B)", rec_b),
                       ("identical resubmission of A", rec_a)]:
        rc = wait(w3, [s.send(submit_fn(contract, sid, k, rec))])[0]
        ev = {n: len(events(contract, n, rc)) for n in
              ("StateCommitted", "EquivocationDetected", "DuplicateConsistentCommitment")}
        log.append(dict(step=label, block=int(rc.blockNumber), status=int(rc.status), gas_used=int(rc.gasUsed), **ev))
    rec = contract.functions.getStateRecord(sid, k, uid).call()
    result = dict(
        uav_id=uid, session=name, steps=log,
        finalized_state=[rec[5] / SCALE, rec[6] / SCALE],
        finalized_equals_first_record=(rec[5], rec[6]) == (rec_a["x_scaled"], rec_a["y_scaled"]),
        isConsistent=bool(contract.functions.isConsistent(sid, k, uid).call()),
        conflict_count=int(contract.functions.getConflictCount(sid, k, uid).call()),
    )
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "equivocation_demo.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# =============================================================================
# secure ADMM on chain (six UAVs)
# =============================================================================
def planned_records(sim: SecureADMM, x_new: np.ndarray, k: int):
    """States each UAV submits in iteration k (same attacker logic as the simulation)."""
    A = sim.A
    recs = {i: [x_new[i]] for i in range(sim.P.n)}
    if A.kind == "none":
        return recs
    if A.kind == "adaptive":
        tau = sim._attacker_tau()
        pred = x_new[sim.own] - sim.yd
        for m, tg in sim.targets.items():
            if tg:
                es = [sim.edge_index[(i, m)] for i in tg]
                recs[m] = [pred[es].mean(axis=0) + A.adaptive_margin * tau[es].min() * sim.d_unit]
        return recs
    sign = -1.0 if (A.kind == "disruption" and (k // max(1, A.switch_period)) % 2 == 1) else 1.0
    off = sign * sim.base_off
    for m, tg in sim.targets.items():
        if not tg or A.commit_policy == "honest":
            continue
        vals = [x_new[m] + off[sim.edge_index[(i, m)]] for i in tg]      # receiver order; first is accepted
        if A.commit_policy == "equivocate" and A.is_receiver_dependent():
            uniq = []
            for v in vals:
                if not any(np.allclose(v, u, atol=1e-12) for u in uniq):
                    uniq.append(v)
            recs[m] = uniq
        else:
            recs[m] = [vals[0]]
    return recs


def cmd_secure_admm(args, ctx=None):
    w3, tester, signers, contract, art = ctx or setup(args)
    P = six_uav_problem(7)
    akw = {} if args.attack == "none" else {"commit_policy": args.commit_policy}
    A = make_attack(args.attack, (1, 4), **akw)
    if args.defense == "none":
        D = DefenseConfig(exchange="ledger", filter="none", exclude_equivocators=False)
    elif args.defense == "ledger":
        D = DefenseConfig(exchange="ledger", filter="none", exclude_equivocators=True)
    else:
        D = DefenseConfig(exchange="ledger", exclude_equivocators=True,
                          **{**PROPOSED_SCREENING, "kappa": args.kappa, "screen_start_iter": args.warmup})
    S = SolverConfig(rho=1.0, max_iter=args.max_iter, eps_primal=1e-4, eps_dual=1e-4)
    sim = SecureADMM(P, A, D, S)
    reference = run_case(P, A, D, S)                     # identical case, off-chain ledger model
    plain = run_case(P, make_attack("none", ()), DefenseConfig(), S)

    uav = signers[:6]
    id_of = [int(contract.functions.uavsByAddress(s.address).call()[1]) for s in uav]
    if sorted(id_of) != list(range(6)):
        raise RuntimeError(f"Accounts 0..5 must be registered as UAV ids 0..5, got {id_of}")
    signer_of_uav = {uid: s for uid, s in zip(id_of, uav)}
    name, sid = new_session(w3, f"secure-admm-{args.attack}-{args.defense}")

    x = P.g.copy(); a = x[sim.own].copy(); b = x[sim.nb].copy()
    u = np.zeros_like(a); v = np.zeros_like(b)
    fac = sim.theta / (2 * sim.theta + S.rho)
    hr, hn = sim.honest_recv, sim.honest_nodes
    rows, converged = [], False
    pool = ThreadPoolExecutor(max_workers=6)
    t_loop = time.perf_counter()
    for k in range(S.max_iter):
        it = k + 1
        t0 = time.perf_counter()
        num = sim.w[:, None] * P.g
        np.add.at(num, sim.own, S.rho * (a + u))
        x_new = num / (sim.w + S.rho * sim.deg)[:, None]
        recs = planned_records(sim, x_new, k)
        t_comp1 = time.perf_counter() - t0

        # submit: every UAV in parallel, multiple records of one UAV in nonce order
        def send_all(uid):
            return [signer_of_uav[uid].send(submit_fn(contract, sid, it, make_record(w3, sid, name, it, uid, st)))
                    for st in recs[uid]]
        t1 = time.perf_counter()
        if tester:
            hashes = [h for uid in range(6) for h in send_all(uid)]
        else:
            hashes = [h for hs in pool.map(send_all, range(6)) for h in hs]
        t_submit = time.perf_counter() - t1
        receipts = wait(w3, hashes)
        t_final = time.perf_counter() - t1
        t2 = time.perf_counter()
        finalized, n_eqv = {}, 0
        for rc in receipts:
            for ev in events(contract, "StateCommitted", rc):
                finalized[int(ev["args"]["uavId"])] = np.array([ev["args"]["xScaled"], ev["args"]["yScaled"]], float) / SCALE
            n_eqv += len(events(contract, "EquivocationDetected", rc))
        consistent = {j: bool(contract.functions.isConsistent(sid, it, j).call()) for j in range(6)}
        t_read = time.perf_counter() - t2

        t3 = time.perf_counter()
        view = np.array([finalized[int(j)] for j in sim.nb])
        eqv = np.array([D.exclude_equivocators and not consistent[int(j)] for j in sim.nb])
        pred = x_new[sim.own] - sim.yd
        res = view - pred
        r = np.linalg.norm(res, axis=1)
        flag = np.zeros(sim.E2, bool)
        if D.filter == "threshold" and k >= D.screen_start_iter:
            flag = (r > sim.tau) & hr
        used = view.copy()
        used[flag] = pred[flag]
        used[eqv & hr] = pred[eqv & hr]
        c1 = x_new[sim.own] - u; c2 = used - v; dif = sim.yd - c1 + c2
        a_new = c1 + fac[:, None] * dif; b_new = c2 - fac[:, None] * dif
        u = u + a_new - x_new[sim.own]; v = v + b_new - used
        rms = lambda z: float(np.sqrt(np.sum(z[hr] ** 2) / max(1, hr.sum())))
        p_own, p_scr, dual = rms(a_new - x_new[sim.own]), rms(b_new - used), S.rho * rms(a_new - a)
        t_comp2 = time.perf_counter() - t3
        err = np.linalg.norm(x_new - P.p_true, axis=1)
        rows.append(dict(iteration=it, rmse_honest_m=float(np.sqrt(np.mean(err[hn] ** 2))),
                         records_submitted=len(hashes), records_accepted=len(finalized),
                         equivocation_events=n_eqv, flagged=int((flag | (eqv & hr)).sum()),
                         primal_own=p_own, primal_screened=p_scr, dual=dual,
                         blocks=len({rc.blockNumber for rc in receipts}),
                         compute_s=t_comp1 + t_comp2, submit_s=t_submit, finality_s=t_final, read_s=t_read,
                         iteration_wall_s=time.perf_counter() - t0))
        x, a, b = x_new, a_new, b_new
        if it >= S.min_iter and p_own < S.eps_primal and p_scr < S.eps_primal and dual < S.eps_dual:
            converged = True
            break
    pool.shutdown()
    df = pd.DataFrame(rows)
    hon = np.where(hn)[0]
    summary = dict(
        attack=args.attack, commit_policy=A.commit_policy, defense_option=args.defense, defense=D.label(),
        exclude_equivocators=D.exclude_equivocators, iterations=len(df),
        converged=converged, final_rmse_honest_m=float(df.rmse_honest_m.iloc[-1]),
        offchain_simulation_rmse_honest_m=reference["summary"]["final_rmse_honest_m"],
        offchain_simulation_iterations=reference["summary"]["iterations"],
        max_abs_diff_onchain_vs_simulation_m=float(np.max(np.abs(x[hon] - reference["x_hat"][hon]))),
        total_records=int(df.records_submitted.sum()), total_equivocation_events=int(df.equivocation_events.sum()),
        end_to_end_s=float(time.perf_counter() - t_loop), mean_iteration_wall_s=float(df.iteration_wall_s.mean()),
        mean_compute_s=float(df.compute_s.mean()), mean_finality_s=float(df.finality_s.mean()),
        plain_admm_compute_per_iteration_s=plain["summary"]["runtime_s"] / max(1, plain["summary"]["iterations"]),
        plain_admm_iterations=plain["summary"]["iterations"],
        note=("Differences between on-chain and simulated estimates come only from the 1e-6 m fixed-point "
              "encoding of states on chain."),
    )
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.attack}_{args.commit_policy}_{args.defense}"
    df.to_csv(out / f"onchain_secure_admm_{tag}_iterations.csv", index=False)
    (out / f"onchain_secure_admm_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# =============================================================================
# load test
# =============================================================================
def load_or_create_accounts(path: Path, n: int, w3):
    accts = json.loads(path.read_text()) if path.exists() else []
    while len(accts) < n:
        acc = w3.eth.account.create()
        key = acc.key.hex() if hasattr(acc.key, "hex") else str(acc.key)
        accts.append(dict(address=acc.address, private_key=key if key.startswith("0x") else "0x" + key))
    path.write_text(json.dumps(accts, indent=1))
    return accts


def cmd_load_test(args, ctx=None):
    w3, tester, signers, contract, art = ctx or setup(args)
    sizes = sorted(int(s) for s in args.sizes.split(","))
    owner = signers[0]
    acc_path = Path(args.accounts_file)
    accts = load_or_create_accounts(acc_path, max(sizes), w3)
    print(f"Load-test accounts stored in {acc_path} (keep private).")

    # fund
    fund_wei = Web3.to_wei(args.fund_eth, "ether")
    hs = [owner.send(to=Web3.to_checksum_address(a["address"]), value=fund_wei)
          for a in accts if w3.eth.get_balance(Web3.to_checksum_address(a["address"])) < fund_wei // 2]
    if hs:
        wait(w3, hs)
    # register with ids id_offset, id_offset+1, ...
    ensure_registered(w3, contract, owner, [a["address"] for a in accts],
                      [args.id_offset + i for i in range(len(accts))], sequential=tester)

    rows = []
    pool = ThreadPoolExecutor(max_workers=args.threads)
    for N in sizes:
        for rnd in range(args.rounds):
            name, sid = new_session(w3, f"loadtest-{N}")
            subs = [Signer(w3, a["address"], a["private_key"]) for a in accts[:N]]
            gp = w3.eth.gas_price
            states = np.random.default_rng(rnd).normal(scale=100.0, size=(N, 2))   # drawn before threading

            def send(idx):
                return subs[idx].send(submit_fn(contract, sid, 1,
                                                make_record(w3, sid, name, 1, args.id_offset + idx, states[idx])),
                                      gas_price=gp)
            start_block = w3.eth.block_number
            t0 = time.perf_counter()
            hashes = [send(i) for i in range(N)] if tester else list(pool.map(send, range(N)))
            t_sub = time.perf_counter() - t0
            if tester:
                receipts = wait(w3, hashes)
            else:
                receipts = list(pool.map(lambda h: w3.eth.wait_for_transaction_receipt(h, timeout=900, poll_latency=0.1),
                                         hashes))
            t_all = time.perf_counter() - t0
            per_block = Counter(int(rc.blockNumber) for rc in receipts)
            b_first, b_last = min(per_block), max(per_block)
            blocks = [w3.eth.get_block(bn) for bn in range(b_first, b_last + 1)]
            t_prev = w3.eth.get_block(b_first - 1).timestamp           # block before the first inclusion
            span = max(1, blocks[-1].timestamp - t_prev)                # chain time needed to include all records (s)
            sizes = [int(b["size"]) for b in blocks if int(b.number) in per_block]
            tx_sizes = [getattr(x, "last_raw_size", np.nan) for x in subs]
            rows.append(dict(N=N, round=rnd, submit_s=t_sub, iteration_latency_s=t_all,
                             throughput_tx_per_s=N / t_all,                 # client-observed TPS (kept for compatibility)
                             client_tps=N / t_all,                          # submission start -> last receipt
                             onchain_tps=N / span,                          # from block timestamps
                             chain_inclusion_time_s=span,
                             blocks_spanned=len(per_block),
                             max_tx_per_block=max(per_block.values()),
                             blocks_until_first_inclusion=b_first - start_block,
                             block_size_mean_bytes=float(np.mean(sizes)), block_size_max_bytes=int(max(sizes)),
                             tx_size_mean_bytes=float(np.nanmean(tx_sizes)),
                             mean_gas_used=float(np.mean([rc.gasUsed for rc in receipts])),
                             failed_tx=int(sum(rc.status != 1 for rc in receipts))))
            r = rows[-1]
            print(f"N={N:5d} round {rnd}: latency {t_all:6.2f} s | client TPS {r['client_tps']:6.1f} | on-chain TPS "
                  f"{r['onchain_tps']:6.1f} | {len(per_block)} block(s), max {r['max_tx_per_block']} tx/block | "
                  f"block {r['block_size_mean_bytes'] / 1000:6.1f} kB | tx {r['tx_size_mean_bytes']:.0f} B | failed {r['failed_tx']}")
    pool.shutdown()
    df = pd.DataFrame(rows)
    summ = df.groupby("N").agg(rounds=("round", "count"), latency_mean_s=("iteration_latency_s", "mean"),
                               latency_std_s=("iteration_latency_s", "std"),
                               throughput_mean_tps=("throughput_tx_per_s", "mean"),
                               client_tps_mean=("client_tps", "mean"), client_tps_std=("client_tps", "std"),
                               onchain_tps_mean=("onchain_tps", "mean"), onchain_tps_std=("onchain_tps", "std"),
                               chain_inclusion_time_mean_s=("chain_inclusion_time_s", "mean"),
                               blocks_spanned_mean=("blocks_spanned", "mean"),
                               max_tx_per_block=("max_tx_per_block", "max"),
                               block_size_mean_bytes=("block_size_mean_bytes", "mean"),
                               block_size_max_bytes=("block_size_max_bytes", "max"),
                               tx_size_mean_bytes=("tx_size_mean_bytes", "mean"),
                               mean_gas_used=("mean_gas_used", "mean"), failed_tx=("failed_tx", "sum")).reset_index()

    # Projection (clearly a model, not a measurement)
    latest = w3.eth.get_block("latest")
    ts = [w3.eth.get_block(bn).timestamp for bn in range(max(0, latest.number - 50), latest.number + 1)]
    period = float(np.median(np.diff(ts))) if len(ts) > 2 and not tester else float("nan")
    gas_tx = float(df.mean_gas_used.mean())
    cap = int(latest.gasLimit // gas_tx)
    proj = []
    for n_target in (100, 500, 1000):
        blocks = int(np.ceil(n_target / cap))
        for iters in [int(i) for i in args.project_iters.split(",")]:
            lat = (blocks + args.confirmations) * period
            proj.append(dict(N=n_target, iterations=iters, tx_capacity_per_block=cap, blocks_per_iteration=blocks,
                             block_period_s=period, confirmations=args.confirmations,
                             projected_latency_per_iteration_s=lat, projected_total_s=lat * iters,
                             assumptions="all records submitted at once; gas-limited blocks; no network delay"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ledger_load_test_rounds.csv", index=False)
    summ.to_csv(out / "ledger_load_test_summary.csv", index=False)
    pd.DataFrame(proj).to_csv(out / "ledger_scaling_projection.csv", index=False)
    print(summ.round(3).to_string())
    print(pd.DataFrame(proj).round(2).to_string())
    return summ


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["metadata", "equivocation", "secure_admm", "load_test", "all"])
    ap.add_argument("--provider", default="http://127.0.0.1:8545", help="RPC URL or 'tester'")
    ap.add_argument("--artifact", default="UAVConsistencyLedger.json")
    ap.add_argument("--address", default=None)
    ap.add_argument("--keys", default=None)
    ap.add_argument("--out", default="results_blockchain")
    ap.add_argument("--uav", type=int, default=1, help="account index used in the equivocation demo")
    ap.add_argument("--attack", default="fdi", choices=["none", "byzantine", "disruption", "fdi", "adaptive"])
    ap.add_argument("--commit-policy", default="equivocate", choices=["equivocate", "single", "honest"])
    ap.add_argument("--defense", default="ledger+screening", choices=["none", "ledger", "ledger+screening"])
    ap.add_argument("--kappa", type=float, default=PROPOSED_SCREENING["kappa"])
    ap.add_argument("--warmup", type=int, default=PROPOSED_SCREENING["screen_start_iter"])
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--sizes", default="6,25,50,100,200,500,1000")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--fund-eth", type=float, default=1.0)
    ap.add_argument("--id-offset", type=int, default=10_000)
    ap.add_argument("--accounts-file", default="loadtest_accounts.json")
    ap.add_argument("--confirmations", type=int, default=1)
    ap.add_argument("--project-iters", default="74,200",
                    help="iteration counts for the projection (attack-free / defended ADMM at 1000 UAVs)")
    args = ap.parse_args()
    ctx = setup(args)
    if args.command in ("metadata", "all"):
        cmd_metadata(args, ctx)
    if args.command in ("equivocation", "all"):
        cmd_equivocation(args, ctx)
    if args.command in ("secure_admm", "all"):
        if args.command == "all":
            for atk, pol in [("none", "equivocate"), ("fdi", "equivocate"), ("byzantine", "equivocate"),
                             ("byzantine", "single"), ("adaptive", "equivocate")]:
                args.attack, args.commit_policy = atk, pol
                cmd_secure_admm(args, ctx)
        else:
            cmd_secure_admm(args, ctx)
    if args.command in ("load_test", "all"):
        cmd_load_test(args, ctx)


if __name__ == "__main__":
    main()
