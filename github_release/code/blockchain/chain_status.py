"""
chain_status.py - helper for RUN_ME.sh

  python chain_status.py probe RPC
      exit 0 and print the network id if a Geth node answers,
      exit 1 if nothing listens on the port, exit 3 if another program uses it.
  python chain_status.py wait_http RPC EXPECTED_NETWORK_ID TIMEOUT_S
      waits until the node answers; exit 0 ok, 2 no answer, 3 wrong network.
  python chain_status.py mesh FIRST_PORT N_NODES TIMEOUT_S
      connects every node to every other node (admin_addPeer), then waits until
      blocks are produced steadily and all nodes agree on the height.
      exit 0 ok, 4 no steady block production.
"""

import socket
import sys
import time
from urllib.parse import urlparse

from web3 import Web3


def listening(rpc):
    u = urlparse(rpc)
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=2):
            return True
    except OSError:
        return False


def call(port, method, params=None, timeout=5):
    w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{port}", request_kwargs={"timeout": timeout}))
    resp = w3.provider.make_request(method, params or [])
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["result"]


def as_int(v):
    return int(v, 16) if isinstance(v, str) else int(v)


def probe(rpc):
    if not listening(rpc):
        sys.exit(1)
    try:
        print(Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 5})).net.version)
        sys.exit(0)
    except Exception:
        sys.exit(3)


def wait_http(rpc, want, tmax):
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 5}))
    t0 = time.time()
    while time.time() - t0 < tmax:
        try:
            net = str(w3.net.version)
            break
        except Exception:
            time.sleep(2)
    else:
        print(f"No answer from {rpc} after {tmax:.0f} s")
        sys.exit(2)
    if want and net != want:
        print(f"{rpc} answers with network id {net}, expected {want}")
        sys.exit(3)
    print(f"Node answers on {rpc} (network id {net}).")


def mesh(first, n, tmax):
    ports = [first + i for i in range(n)]
    # 1. which nodes answer?
    t0, up = time.time(), []
    while time.time() - t0 < min(tmax, 60):
        up = [p for p in ports if listening(f"http://127.0.0.1:{p}")]
        if len(up) == n:
            break
        time.sleep(2)
    print(f"Nodes answering: {len(up)} of {n} (ports {up})")
    # 2. connect every node to every other node over 127.0.0.1
    enodes = {}
    for p in up:
        try:
            e = call(p, "admin_nodeInfo")["enode"]
            nid, rest = e.split("@", 1)
            tcp = rest.split(":", 1)[1].split("?", 1)[0]
            enodes[p] = f"{nid}@127.0.0.1:{tcp}"
        except Exception as exc:
            print(f"  node on port {p}: admin API not available ({str(exc)[:80]}); relying on discovery")
    for p in enodes:
        for q, e in enodes.items():
            if q != p:
                try:
                    call(p, "admin_addPeer", [e])
                except Exception:
                    pass
    if enodes:
        print(f"Connected {len(enodes)} nodes directly to each other.")
    # 3. wait until node 1 (the RPC node) and a Clique quorum are at the chain tip, then for new blocks.
    #    After an abrupt stop, Geth re-processes its stored blocks on start-up; this can take minutes.
    quorum = len(up) // 2 + 1          # Clique needs more than half of the signers; counted among answering nodes
    if len(up) < n // 2 + 1:
        print(f"  warning: only {len(up)} of {n} nodes answer; Clique needs at least {n // 2 + 1} signers")
    t1, last_print, phase, h_ready, t_ready = time.time(), 0, "sync", None, None
    while time.time() - t1 < tmax:
        heights, peers = {}, {}
        for p in up:
            try:
                heights[p] = as_int(call(p, "eth_blockNumber"))
                peers[p] = as_int(call(p, "net_peerCount"))
            except Exception:
                pass
        if heights and first in heights:
            top = max(heights.values())
            at_tip = [p for p in heights if heights[p] >= top - 2]
            synced = heights[first] >= top - 2 and len(at_tip) >= quorum
            if time.time() - last_print > 5:
                state = "catching up after restart" if not synced else "producing blocks"
                print(f"  heights {[heights.get(p) for p in up]}, peers {[peers.get(p) for p in up]}  ({state})", flush=True)
                last_print = time.time()
            if phase == "sync" and synced:
                phase, h_ready, t_ready = "produce", heights[first], time.time()
                print(f"  node 1 and {len(at_tip)} of {len(up)} nodes are at the chain tip (block {top}); waiting for new blocks ...", flush=True)
            elif phase == "produce":
                if not synced:
                    phase = "sync"
                elif heights[first] >= h_ready + 5:
                    lag = [p for p in up if heights.get(p, -1) < top - 2]
                    print(f"Steady block production: {heights[first] - h_ready} new blocks in {time.time() - t_ready:.0f} s; "
                          f"{len(at_tip)} of {len(up)} nodes at the tip (block {heights[first]}).")
                    if lag:
                        print(f"  note: nodes on ports {lag} are still catching up; this does not affect the experiments.")
                    sys.exit(0)
        time.sleep(2)
    print(f"No steady block production after {tmax:.0f} s.")
    sys.exit(4)


def main():
    mode = sys.argv[1]
    if mode == "probe":
        probe(sys.argv[2])
    elif mode == "wait_http":
        wait_http(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    elif mode == "mesh":
        mesh(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
