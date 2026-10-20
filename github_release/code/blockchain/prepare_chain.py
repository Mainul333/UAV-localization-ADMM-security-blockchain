"""
prepare_chain.py - makes the running private chain ready for the experiments

1. Builds keys.json from the node folders (node1..node6 keystore + password.txt),
   in UAV order U0..U5. The file stays in the private/ folder and is never zipped.
2. Checks that the contract from the Truffle artifact exists on the chain.
   If the chain no longer has it (for example after a reset), the contract is
   deployed again from the same artifact by U0 and the artifact copy is updated.
3. Registers U0..U5 as UAV ids 0..5 if they are not registered yet.
4. Writes chain_preparation.json (what was found and what was changed).
"""

import argparse
import glob
import json
import os
import stat
import sys
from pathlib import Path

from eth_account import Account
from web3 import Web3


def connect(rpc):
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 60}))
    try:
        from web3.middleware import ExtraDataToPOAMiddleware as poa
    except ImportError:
        from web3.middleware import geth_poa_middleware as poa
    w3.middleware_onion.inject(poa, layer=0)
    if not w3.is_connected():
        sys.exit(f"Cannot reach {rpc}")
    return w3


def send(w3, acct, fn, gas):
    tx = fn.build_transaction({"from": acct.address, "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
                               "gas": gas, "gasPrice": w3.eth.gas_price, "chainId": int(w3.eth.chain_id)})
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
    return w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(raw), timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", default="http://127.0.0.1:8545")
    ap.add_argument("--net-dir", required=True)
    ap.add_argument("--artifact", default="UAVConsistencyLedger.json")
    ap.add_argument("--keys-out", default="private/keys.json")
    ap.add_argument("--report", default="chain_preparation.json")
    args = ap.parse_args()

    # 1. keys from the node folders --------------------------------------------
    accts = []
    for i in range(1, 7):
        ks = sorted(glob.glob(os.path.join(args.net_dir, f"node{i}", "keystore", "*")))
        pw_file = os.path.join(args.net_dir, f"node{i}", "password.txt")
        if not ks or not os.path.exists(pw_file):
            sys.exit(f"node{i}: keystore or password.txt missing")
        key = Account.decrypt(json.load(open(ks[0])), open(pw_file).read().strip())
        accts.append(Account.from_key(key))
        print(f"U{i - 1} = node{i}: {accts[-1].address}")
    Path(args.keys_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.keys_out).write_text(json.dumps(
        [{"address": a.address, "private_key": "0x" + a.key.hex().removeprefix("0x")} for a in accts], indent=1))
    try:
        os.chmod(args.keys_out, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    # 2. contract ------------------------------------------------------------------
    w3 = connect(args.rpc)
    net = str(w3.net.version)
    art_path = Path(args.artifact)
    art = json.loads(art_path.read_text(encoding="utf-8"))
    report = dict(net_version=net, chain_id=int(w3.eth.chain_id), client=w3.client_version,
                  latest_block=int(w3.eth.block_number), redeployed=False, registered_now=[])
    address = art.get("networks", {}).get(net, {}).get("address")
    has_code = bool(address) and len(w3.eth.get_code(Web3.to_checksum_address(address))) > 0
    if has_code:
        print(f"Contract found at {address} (network {net})")
    else:
        print(f"No contract at {address} on network {net}: deploying it again from the artifact (owner U0) ...")
        backup = art_path.with_name(art_path.stem + ".original.json")
        if not backup.exists():
            backup.write_text(art_path.read_text(encoding="utf-8"), encoding="utf-8")
        C = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        rc = send(w3, accts[0], C.constructor(), 4_000_000)
        address = rc.contractAddress
        art.setdefault("networks", {})[net] = {"address": address, "transactionHash": rc.transactionHash.hex()}
        art_path.write_text(json.dumps(art, indent=2), encoding="utf-8")
        report.update(redeployed=True, deploy_gas=int(rc.gasUsed), deploy_block=int(rc.blockNumber))
        print(f"Deployed at {address} (gas {rc.gasUsed})")
    report["contract_address"] = address
    c = w3.eth.contract(address=Web3.to_checksum_address(address), abi=art["abi"])

    # 3. owner and registrations -------------------------------------------------------
    owner = c.functions.owner().call()
    report["owner"] = owner
    if owner != accts[0].address:
        sys.exit(f"Contract owner is {owner}, not U0 ({accts[0].address}); cannot register UAVs.")
    for i, a in enumerate(accts):
        reg, uid = c.functions.uavsByAddress(a.address).call()[:2]
        if reg and int(uid) == i:
            continue
        if reg:
            sys.exit(f"U{i} is registered as UAV {uid}, expected {i}")
        rc = send(w3, accts[0], c.functions.registerUAV(i, a.address, f"UAV-{i}"), 300_000)
        report["registered_now"].append(i)
        print(f"Registered U{i} (block {rc.blockNumber})")
    print("U0..U5 registered as UAV ids 0..5")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
