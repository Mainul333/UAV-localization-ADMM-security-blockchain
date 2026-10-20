"""
secure_localization.py
======================

Simulation library for the revised experiments of
"Blockchain-Assisted Secure Distributed UAV Localization Under Byzantine and
False Data Injection Attacks".

What changed relative to the code used for the submitted manuscript
-------------------------------------------------------------------
1. Solver. The submitted attack results were produced with decentralized
   gradient descent (DGD), while the manuscript describes ADMM. This library
   implements the ADMM of Section 4.4 (closed-form node update, Eq. (30)).

2. Symmetric, label-invariant edge update (Reviewer 3, Comments 1-2).
   Every UAV keeps its OWN replica of each incident edge problem:
       own copy a_ij (= x_ij), own dual u_ij,
       replica of the neighbour copy b_ij (= x_ji as seen by i), replica dual v_ij.
   UAV i solves the edge problem of Eq. (47) with its own state and its OWN
   screened view of the neighbour, and keeps only its own side. A UAV never
   depends on a neighbour's screening, so compromised lower-index endpoints
   are harmless and relabelling UAVs does not change the result.
   Without attacks and without screening triggers, the replicas held by the
   two endpoints are identical, so the method is exactly standard ADMM
   (verified numerically in run_experiments.py, experiment "sanity").

3. Explicit exchange model (Reviewer 1, Comments 1-2; Reviewer 3, Comment 4).
   exchange = "direct" : unsecured peer-to-peer messages (receiver-specific).
   exchange = "ledger" : every receiver reads the single committed record of
                         the sender for this iteration. A compromised UAV
                         COMMITS A FALSE STATE; it is not assumed honest.
   exchange = "echo"   : signed peer-to-peer messages plus digest echo among
                         common neighbours (no blockchain); equivocation is
                         detected only when two receivers share a neighbour.

4. Attacks on every secured case (Reviewer 3, Comment 5). The submitted
   "secured Byzantine/disruption" rows reused the attack-free run. Here every
   secured configuration actually runs the attack.

5. Calibrated screening threshold (Reviewer 3, Comment 6; Reviewer 2, Comment 3)
   threshold_mode = "fixed": tau = kappa * sigma_y           (manuscript Eq. 43)
   threshold_mode = "chi2" : tau = sqrt(-2 ln alpha) * sqrt(sigma_y^2 + s_i^2 + s_j^2)
   In 2-D, ||e||^2 / var is chi-square with 2 dof, whose tail is exp(-c/2),
   so the chi2 rule gives a per-message false-alarm rate of about alpha when
   s_i, s_j bound the estimate errors (default: the prior std sigma_g).

6. Threshold-aware adaptive attacker (all three reviewers).
7. W-MSR-type trimming baseline (Reviewer 1, Comment 5; Reviewer 2, Comment 4).
8. Detection statistics, original/screened residuals, ledger counts, tail
   metrics, centralized WLS reference (Reviewer 3, Comments 3, 5, 6).

Conventions follow the manuscript: y_ij = p_i - p_j + noise (Eq. 7) and
y_ji = -y_ij (Eq. 9).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False


# =============================================================================
# Problem definition and generators
# =============================================================================

@dataclass
class Problem:
    """Cooperative localization instance (manuscript Section 4.1)."""
    p_true: np.ndarray          # (N, 2) true positions
    g: np.ndarray               # (N, 2) noisy absolute priors, Eq. (5)
    edges: np.ndarray           # (E, 2) undirected edges (i, j), i < j
    y: np.ndarray               # (E, 2) relative measurements y_ij = p_i - p_j + noise
    sigma_g: np.ndarray         # (N,) absolute-prior std
    sigma_y: np.ndarray         # (E,) relative-measurement std
    name: str = "problem"
    seed: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.p_true.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def neighbors(self) -> List[List[int]]:
        nbr = [[] for _ in range(self.n)]
        for i, j in self.edges:
            nbr[int(i)].append(int(j))
            nbr[int(j)].append(int(i))
        return [sorted(x) for x in nbr]

    def is_connected(self) -> bool:
        nbr = self.neighbors()
        seen = {0}
        stack = [0]
        while stack:
            k = stack.pop()
            for j in nbr[k]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return len(seen) == self.n

    def relabel(self, perm: Sequence[int]) -> "Problem":
        """Return the same physical problem with UAV indices relabelled.
        New index perm[old] is assigned to old UAV `old`."""
        perm = np.asarray(perm, dtype=int)
        inv = np.argsort(perm)
        p_true = self.p_true[inv]
        g = self.g[inv]
        sigma_g = self.sigma_g[inv]
        new_edges, new_y = [], []
        for (i, j), y in zip(self.edges, self.y):
            a, b = perm[i], perm[j]
            if a < b:
                new_edges.append((a, b)); new_y.append(y)
            else:
                new_edges.append((b, a)); new_y.append(-y)
        order = np.lexsort((np.array(new_edges)[:, 1], np.array(new_edges)[:, 0]))
        return Problem(p_true, g, np.array(new_edges)[order], np.array(new_y)[order],
                       sigma_g, self.sigma_y[order], name=self.name + "_relabelled",
                       seed=self.seed, meta=dict(self.meta, perm=perm.tolist()))


def six_uav_problem(seed: int = 7, sigma_g: float = 1.5, sigma_y: float = 0.5,
                    comm_radius: float = 14.0) -> Problem:
    """Six-UAV network. With seed=7 and default noise this reproduces the
    network, priors and measurement noise of the submitted prototype
    (same random-number sequence as the original scripts)."""
    rng = np.random.default_rng(seed)
    base = np.array([[-10, -8], [-9, 9], [-1, 0], [3, -10], [9, -2], [10, 10]], dtype=float)
    p_true = base + rng.normal(scale=0.8, size=base.shape)
    n = p_true.shape[0]
    g = p_true + rng.normal(scale=sigma_g, size=p_true.shape)
    edges, ys = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p_true[i] - p_true[j]) <= comm_radius:
                ys.append((p_true[i] - p_true[j]) + rng.normal(scale=sigma_y, size=2))
                edges.append((i, j))
    if len(edges) < n - 1:
        edges, ys = [], []
        for i in range(n - 1):
            ys.append((p_true[i] - p_true[i + 1]) + rng.normal(scale=sigma_y, size=2))
            edges.append((i, i + 1))
    prob = Problem(p_true, g, np.array(edges, dtype=int), np.array(ys, dtype=float),
                   np.full(n, sigma_g), np.full(len(edges), sigma_y),
                   name="six_uav", seed=seed,
                   meta=dict(comm_radius=comm_radius, generator="six_uav"))
    return prob


def _knn_pairs(p: np.ndarray, k: int) -> List[tuple]:
    n = p.shape[0]
    pairs = set()
    if SCIPY_AVAILABLE:
        _, idx = cKDTree(p).query(p, k=k + 1)
        for i in range(n):
            for j in idx[i, 1:]:
                j = int(j)
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
    else:
        for i in range(n):
            d2 = np.sum((p - p[i]) ** 2, axis=1)
            for j in np.argpartition(d2, k + 1)[:k + 1]:
                j = int(j)
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def random_knn_problem(n: int = 1000, seed: int = 7, area_size_m: float = 250.0,
                       k_neighbors: int = 8, sigma_g: float = 1.5,
                       sigma_y: float = 0.5) -> Problem:
    """Large-scale network. With n=1000, seed=7 this reproduces the positions,
    priors, kNN graph and noise draws of the submitted 1000-UAV experiment."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(-area_size_m / 2.0, area_size_m / 2.0, size=(n, 2))
    g = p_true + rng.normal(scale=sigma_g, size=p_true.shape)
    pairs = _knn_pairs(p_true, k_neighbors)
    ys = [(p_true[i] - p_true[j]) + rng.normal(scale=sigma_y, size=2) for i, j in pairs]
    return Problem(p_true, g, np.array(pairs, dtype=int), np.array(ys, dtype=float),
                   np.full(n, sigma_g), np.full(len(pairs), sigma_y),
                   name=f"knn_{n}", seed=seed,
                   meta=dict(area_size_m=area_size_m, k_neighbors=k_neighbors,
                             generator="random_knn"))


def choose_malicious(n: int, count: int, seed: int) -> tuple:
    """Same placement rule as the submitted 1000-UAV experiment."""
    rng = np.random.default_rng(seed + 1000 + count)
    return tuple(sorted(rng.choice(n, size=count, replace=False).astype(int).tolist()))


# =============================================================================
# Centralized WLS reference (Reviewer 3, Comment 3c)
# =============================================================================

def centralized_wls(problem: Problem, exclude_nodes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Exact minimizer of Eq. (12). If exclude_nodes is given, all relative
    measurements incident to those nodes are dropped (oracle reference that
    knows the compromised set; used only as a lower bound, never by the
    distributed algorithm)."""
    n = problem.n
    w = 1.0 / problem.sigma_g ** 2
    theta = 1.0 / problem.sigma_y ** 2
    excl = set(int(v) for v in (exclude_nodes or ()))
    keep = np.array([(int(i) not in excl and int(j) not in excl) for i, j in problem.edges], bool)
    E = problem.edges[keep]
    th = theta[keep]
    Y = problem.y[keep]
    rows = np.concatenate([np.arange(n), E[:, 0], E[:, 1], E[:, 0], E[:, 1]])
    cols = np.concatenate([np.arange(n), E[:, 0], E[:, 1], E[:, 1], E[:, 0]])
    vals = np.concatenate([w, th, th, -th, -th])
    rhs = w[:, None] * problem.g
    np.add.at(rhs, E[:, 0], th[:, None] * Y)
    np.add.at(rhs, E[:, 1], -th[:, None] * Y)
    if SCIPY_AVAILABLE:
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        X = np.column_stack([spla.spsolve(A.tocsc(), rhs[:, c]) for c in range(2)])
    else:
        A = np.zeros((n, n))
        np.add.at(A, (rows, cols), vals)
        X = np.linalg.solve(A, rhs)
    return X


def wls_cost(problem: Problem, X: np.ndarray) -> float:
    w = 1.0 / problem.sigma_g ** 2
    th = 1.0 / problem.sigma_y ** 2
    r = (X[problem.edges[:, 0]] - X[problem.edges[:, 1]]) - problem.y
    return float(0.5 * np.sum(w * np.sum((X - problem.g) ** 2, 1)) + 0.5 * np.sum(th * np.sum(r ** 2, 1)))


# =============================================================================
# Configurations
# =============================================================================

@dataclass
class AttackConfig:
    """Attack model (manuscript Section 4.2-4.3) plus the adaptive attacker.

    kind:
      "none"       : no attack
      "byzantine"  : receiver-dependent constant offsets, Eqs. (15)-(18)
      "disruption" : receiver-dependent offsets with sign switching, Eqs. (22)-(24)
      "fdi"        : one receiver-independent constant offset, Eqs. (19)-(21)
      "adaptive"   : threshold-aware attacker. Sends ONE consistent value
                     x_j = mean_i(x_i - y_ij) + margin * tau * d, i.e. the
                     construction of Reviewer 3, Comment 4b, with ||delta|| < tau.
    commit_policy (only used when exchange == "ledger"):
      "equivocate" : attacker submits all receiver-specific values; the ledger
                     accepts the first (lowest receiver index) and rejects the rest.
      "single"     : rational attacker that sends/commits ONE consistent false
                     value (the value of its first target) on every exchange, so
                     there is nothing to reject on chain or to catch by echo.
      "honest"     : naive attacker that commits its true state and only
                     corrupts off-ledger packets. This reproduces the implicit
                     assumption behind the submitted "secured = baseline" rows
                     and is included only to explain them.
    """
    kind: str = "none"
    malicious: tuple = ()
    targets: Optional[Dict[int, Sequence[int]]] = None   # None -> all neighbours
    direction: tuple = (1.0, -0.6)
    magnitude: float = 0.0
    switch_period: int = 100                              # L in Eq. (24)
    receiver_dependent: Optional[bool] = None             # default True for byzantine/disruption
    commit_policy: str = "equivocate"
    adaptive_margin: float = 0.9                          # fraction of tau used by the adaptive attacker
    adaptive_fallback_tau: float = 1.4                    # tau assumed when the defender uses no threshold

    def is_receiver_dependent(self) -> bool:
        if self.receiver_dependent is not None:
            return bool(self.receiver_dependent)
        return self.kind in ("byzantine", "disruption")


# Defaults used in the submitted experiments (now reported explicitly).
DEFAULT_ATTACKS = {
    "byzantine": dict(direction=(1.0, -0.6), magnitude=1.9),
    "disruption": dict(direction=(1.0, -0.8), magnitude=14.0, switch_period=100),
    "fdi": dict(direction=(1.0, -0.6), magnitude=2.8),
    "adaptive": dict(direction=(1.0, -0.6), magnitude=0.0, adaptive_margin=0.9),
}


def make_attack(kind: str, malicious: Sequence[int], **overrides) -> AttackConfig:
    if kind == "none":
        return AttackConfig(kind="none", malicious=())
    params = dict(DEFAULT_ATTACKS[kind])
    params.update(overrides)
    return AttackConfig(kind=kind, malicious=tuple(int(m) for m in malicious), **params)


@dataclass
class DefenseConfig:
    """Defense configuration.

    exchange : "direct" | "ledger" | "echo"
    filter   : "none" | "threshold" | "msr"
    screening_mode : "replace" | "clip"           (threshold filter only)
    threshold_mode : "fixed" | "chi2"
    kappa    : sensitivity factor for threshold_mode="fixed" (manuscript Eq. 43)
    alpha_fa : target per-message false-alarm rate for threshold_mode="chi2"
    est_std  : s_i used by the chi2 rule; None -> sigma_g (conservative)
    msr_F    : W-MSR-type trimming parameter (assumed max malicious neighbours)
    screen_start_iter : first iteration at which screening is applied
    exclude_equivocators : with a ledger, drop a sender whose record for this
                           iteration was flagged as conflicting (contract
                           isConsistent(...) == false)
    """
    exchange: str = "direct"
    filter: str = "none"
    screening_mode: str = "replace"
    threshold_mode: str = "fixed"
    kappa: float = 2.8
    alpha_fa: float = 0.01
    est_std: Optional[float] = None
    msr_F: int = 1
    screen_start_iter: int = 0
    exclude_equivocators: bool = False

    def label(self) -> str:
        ex = {"direct": "none", "ledger": "ledger", "echo": "signed-echo"}[self.exchange]
        if self.filter == "none":
            return ex if ex != "none" else "no defense"
        if self.filter == "msr":
            f = f"W-MSR(F={self.msr_F})"
        else:
            thr = f"kappa={self.kappa:g}" if self.threshold_mode == "fixed" else f"chi2(alpha={self.alpha_fa:g})"
            f = f"screen-{self.screening_mode}[{thr}]"
        return f if ex == "none" else f"{ex}+{f}"


# Named configurations used for the ablation (Reviewer 2 C4, Reviewer 3 C5c, C7c)
# Proposed screening defaults (justify with the threshold sweep):
#   kappa = 2.5   -> about 4% steady-state false alarms if the residual std is sigma_y
#   warm-up = 10  -> screening starts after the prior-dominated transient
PROPOSED_SCREENING = dict(filter="threshold", screening_mode="replace", threshold_mode="fixed",
                          kappa=2.5, screen_start_iter=10)
SUBMITTED_SCREENING = dict(filter="threshold", screening_mode="replace", threshold_mode="fixed",
                           kappa=2.8, screen_start_iter=0)


def ablation_defenses(threshold_kw: Optional[dict] = None, msr_F: int = 1,
                      include_submitted: bool = True) -> Dict[str, DefenseConfig]:
    thr = dict(PROPOSED_SCREENING)
    thr.update(threshold_kw or {})
    out = {
        "no_defense": DefenseConfig(exchange="direct", filter="none"),
        "ledger_only": DefenseConfig(exchange="ledger", filter="none", exclude_equivocators=True),
        "signed_echo_only": DefenseConfig(exchange="echo", filter="none"),
        "screening_only": DefenseConfig(exchange="direct", **thr),
        "ledger+screening": DefenseConfig(exchange="ledger", exclude_equivocators=True, **thr),
        "signed_echo+screening": DefenseConfig(exchange="echo", **thr),
        "wmsr_baseline": DefenseConfig(exchange="direct", filter="msr", msr_F=msr_F),
    }
    if include_submitted:
        out["submitted_config"] = DefenseConfig(exchange="ledger", exclude_equivocators=False,
                                                **SUBMITTED_SCREENING)
    return out


@dataclass
class SolverConfig:
    rho: float = 1.0
    max_iter: int = 800
    eps_primal: float = 1e-3     # RMS per edge copy (m)
    eps_dual: float = 1e-3       # RMS per edge copy (m), scaled by rho
    min_iter: int = 5
    tail_window: int = 100       # window for tail-averaged / worst-case metrics
    init_noise_std: float = 0.0  # extra perturbation of the initial states (sensitivity study)
    init_seed: int = 12345
    store_traces: bool = True


# =============================================================================
# Secure ADMM simulation
# =============================================================================

def _rot(vec: np.ndarray, ang: np.ndarray) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], axis=-1)


class SecureADMM:
    """Vectorized simulator of the symmetric secure ADMM over directed edges.

    Directed edge e = (owner i, neighbour j) stores UAV i's replica of the
    edge problem for undirected edge {i, j}.
    """

    def __init__(self, problem: Problem, attack: AttackConfig, defense: DefenseConfig,
                 solver: SolverConfig = SolverConfig()):
        self.P, self.A, self.D, self.S = problem, attack, defense, solver
        n = problem.n
        E = problem.edges
        self.own = np.concatenate([E[:, 0], E[:, 1]]).astype(int)
        self.nb = np.concatenate([E[:, 1], E[:, 0]]).astype(int)
        self.yd = np.concatenate([problem.y, -problem.y])            # y_{own,nb}
        self.sig_y = np.concatenate([problem.sigma_y, problem.sigma_y])
        self.theta = 1.0 / self.sig_y ** 2
        self.w = 1.0 / problem.sigma_g ** 2
        self.deg = np.bincount(self.own, minlength=n).astype(float)
        self.E2 = self.own.size
        self.edge_index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(self.own, self.nb))}

        self.mal_nodes = np.zeros(n, bool)
        if attack.kind != "none":
            self.mal_nodes[list(attack.malicious)] = True
        self.honest_nodes = ~self.mal_nodes
        self.honest_recv = self.honest_nodes[self.own]                # edges owned by honest UAVs
        self.from_mal = self.mal_nodes[self.nb]

        # Targets: requested targets intersected with actual neighbours.
        nbrs = problem.neighbors()
        self.targets = {}
        self.warnings = []
        for m in attack.malicious if attack.kind != "none" else ():
            req = nbrs[m] if attack.targets is None or m not in attack.targets else list(attack.targets[m])
            eff = sorted(set(int(r) for r in req) & set(nbrs[m]))
            dropped = sorted(set(int(r) for r in req) - set(nbrs[m]))
            if dropped:
                self.warnings.append(f"UAV {m}: requested targets {dropped} are not neighbours and were ignored")
            self.targets[m] = eff
        self.tgt = np.array([(self.from_mal[e] and int(self.own[e]) in self.targets.get(int(self.nb[e]), []))
                             for e in range(self.E2)], bool)
        # Precomputed index sets (speed): outgoing edges and target edges of each attacker
        self.out_edges = {m: np.where(self.nb == m)[0] for m in self.targets}
        self.tgt_edges = {m: np.array([self.edge_index[(i, m)] for i in tg], int) for m, tg in self.targets.items()}

        # Attack geometry
        d = np.asarray(attack.direction, float)
        self.d_unit = d / max(np.linalg.norm(d), 1e-12)
        if attack.is_receiver_dependent():
            ang = (((self.nb * 131 + self.own * 17) % 360) * np.pi / 180.0).astype(float)
            self.base_off = attack.magnitude * _rot(self.d_unit, ang)
        else:
            self.base_off = np.tile(attack.magnitude * self.d_unit, (self.E2, 1))
        self.base_off[~self.tgt] = 0.0

        # Ledger: which target edge's value does each attacker commit first?
        self.first_edge = {}
        for m, tg in self.targets.items():
            if tg:
                self.first_edge[m] = self.edge_index[(tg[0], m)]
        # A "single" attacker sends one consistent value to all of its targets
        # (on any exchange), i.e. the rational response to a non-equivocation layer.
        if attack.commit_policy == "single":
            for m, e0 in self.first_edge.items():
                out = (self.nb == m) & self.tgt
                self.base_off[out] = self.base_off[e0]

        # Screening thresholds per directed edge
        self.tau = self._thresholds()

        # MSR padded neighbour matrix (edges owned by node i)
        if defense.filter == "msr":
            order = np.argsort(self.own, kind="stable")
            counts = np.bincount(self.own, minlength=n)
            dmax = int(counts.max())
            self.pad = -np.ones((n, dmax), int)
            pos = np.zeros(n, int)
            for e in order:
                i = self.own[e]
                self.pad[i, pos[i]] = e
                pos[i] += 1

        # Echo pairs: (edge i<-j, edge l<-j) for honest i, l that are common neighbours of j
        self.echo_msgs_per_iter = 0
        if defense.exchange == "echo":
            nset = [set(x) for x in nbrs]
            pa, pb = [], []
            for j in range(n):
                for i in nbrs[j]:
                    common = nset[i] & nset[j]
                    self.echo_msgs_per_iter += len(common)   # every receiver echoes to common neighbours
                    if not self.mal_nodes[j] or self.mal_nodes[i]:
                        continue
                    for l in common:
                        if not self.mal_nodes[l]:
                            pa.append(self.edge_index[(i, j)]); pb.append(self.edge_index[(l, j)])
            self.echo_a = np.array(pa, int); self.echo_b = np.array(pb, int)

    # ------------------------------------------------------------------
    def _thresholds(self) -> np.ndarray:
        D = self.D
        if D.threshold_mode == "fixed":
            return D.kappa * self.sig_y
        if D.threshold_mode == "chi2":
            s_i = self.P.sigma_g[self.own] if D.est_std is None else np.full(self.E2, D.est_std)
            s_j = self.P.sigma_g[self.nb] if D.est_std is None else np.full(self.E2, D.est_std)
            return np.sqrt(-2.0 * np.log(D.alpha_fa)) * np.sqrt(self.sig_y ** 2 + s_i ** 2 + s_j ** 2)
        raise ValueError("threshold_mode must be 'fixed' or 'chi2'")

    def _attacker_tau(self) -> np.ndarray:
        if self.D.filter == "threshold":
            return self.tau
        return np.full(self.E2, self.A.adaptive_fallback_tau)

    # ------------------------------------------------------------------
    def _views(self, x: np.ndarray, k: int):
        """Return (view, malicious_msg, ledger_stats, equivocation_flag) for every directed edge."""
        A, D = self.A, self.D
        true_vals = x[self.nb]
        view = true_vals.copy()
        mal_msg = np.zeros(self.E2, bool)
        eqv_flag = np.zeros(self.E2, bool)
        ledger = dict(submitted=self.P.n, accepted=self.P.n, rejected=0)
        if A.kind == "none" or not self.targets:
            return view, mal_msg, ledger, eqv_flag

        if A.kind == "adaptive":
            # One consistent, ledger-valid value per attacker, placed inside the
            # screening ball of its targets (Reviewer 3, Comment 4b).
            tau_att = self._attacker_tau()
            pred = x[self.own] - self.yd
            for m, tg in self.targets.items():
                if not tg:
                    continue
                es = self.tgt_edges[m]
                val = pred[es].mean(axis=0) + A.adaptive_margin * tau_att[es].min() * self.d_unit
                out = self.out_edges[m]
                view[out] = val
                mal_msg[out] = True
            return view, mal_msg, ledger, eqv_flag

        sign = 1.0
        if A.kind == "disruption":
            sign = 1.0 if (k // max(1, A.switch_period)) % 2 == 0 else -1.0
        off = sign * self.base_off

        if D.exchange in ("direct", "echo"):
            view = true_vals + off
            mal_msg = self.tgt.copy()
            if D.exchange == "echo" and self.echo_a.size:
                diff = np.linalg.norm(view[self.echo_a] - view[self.echo_b], axis=1) > 1e-9
                np.logical_or.at(eqv_flag, self.echo_a, diff)
            return view, mal_msg, ledger, eqv_flag

        # Ledger: one accepted record per sender and iteration.
        for m, tg in self.targets.items():
            if not tg:
                continue
            out = self.out_edges[m]
            if A.commit_policy == "honest":
                committed = x[m]                       # attack neutralized: packets are ignored
                n_distinct = 1
            else:
                committed = x[m] + off[self.first_edge[m]]
                n_distinct = 1
                if A.commit_policy == "equivocate" and A.is_receiver_dependent():
                    n_distinct = len(set(map(tuple, np.round(off[self.tgt_edges[m]], 12))))
            view[out] = committed
            mal_msg[out] = A.commit_policy != "honest"
            ledger["submitted"] += n_distinct - 1
            ledger["rejected"] += n_distinct - 1
            if D.exclude_equivocators and n_distinct > 1:
                eqv_flag[out] = True
        return view, mal_msg, ledger, eqv_flag

    # ------------------------------------------------------------------
    def _msr_flags(self, res: np.ndarray) -> np.ndarray:
        """W-MSR-type rule: per coordinate, drop up to F neighbour-implied
        positions above and F below the receiver's own state. res = (x_j + y_ij) - x_i."""
        F = int(self.D.msr_F)
        flags = np.zeros(self.E2, bool)
        if F <= 0:
            return flags
        valid = self.pad >= 0
        for c in range(2):
            R = np.where(valid, res[np.maximum(self.pad, 0), c], np.nan)
            for sgn in (1.0, -1.0):
                Rs = np.where(np.nan_to_num(sgn * R, nan=-np.inf) > 0, sgn * R, -np.inf)
                order = np.argsort(-Rs, axis=1)[:, :F]
                top = np.take_along_axis(Rs, order, axis=1)
                sel = np.take_along_axis(self.pad, order, axis=1)
                ok = np.isfinite(top) & (sel >= 0)
                flags[sel[ok]] = True
        return flags

    # ------------------------------------------------------------------
    def run(self) -> dict:
        P, D, S = self.P, self.D, self.S
        n, rho = P.n, S.rho
        x = P.g.copy()
        if S.init_noise_std > 0:
            x = x + np.random.default_rng(S.init_seed).normal(scale=S.init_noise_std, size=x.shape)
        a = x[self.own].copy(); b = x[self.nb].copy()
        u = np.zeros_like(a); v = np.zeros_like(b)
        fac = self.theta / (2.0 * self.theta + rho)
        hr = self.honest_recv
        n_h_edges = max(1, int(hr.sum()))
        hn = self.honest_nodes

        rec = {k: [] for k in ["iteration", "rmse_honest_m", "rmse_all_m", "mean_err_honest_m",
                               "max_err_honest_m", "primal_own", "primal_screened", "primal_original",
                               "dual", "node_step", "flagged", "tp", "fp", "fn", "tn",
                               "ledger_submitted", "ledger_accepted", "ledger_rejected", "wls_cost"]}
        converged = False
        t0 = time.perf_counter()
        for k in range(S.max_iter):
            # (1) node update, Eq. (30)
            num = self.w[:, None] * P.g
            np.add.at(num, self.own, rho * (a + u))
            x_new = num / (self.w + rho * self.deg)[:, None]

            # (2) exchange
            view, mal_msg, ledger, eqv = self._views(x_new, k)

            # (3) screening at honest receivers, Eqs. (39)-(46)
            pred = x_new[self.own] - self.yd
            res = view - pred
            r = np.linalg.norm(res, axis=1)
            flag = np.zeros(self.E2, bool)
            used = view.copy()
            if k >= D.screen_start_iter:
                if D.filter == "threshold":
                    flag = (r > self.tau) & hr
                elif D.filter == "msr":
                    flag = self._msr_flags(res) & hr
            eq_drop = eqv & hr
            if D.filter == "threshold" and D.screening_mode == "clip":
                rs = np.maximum(r, 1e-12)[:, None]
                used[flag] = pred[flag] + self.tau[flag, None] * res[flag] / rs[flag]
            else:
                used[flag] = pred[flag]
            used[eq_drop] = pred[eq_drop]
            flag_all = flag | eq_drop

            # (4) per-node edge replica update (symmetric form of Eq. (47)) and duals, Eq. (48)
            c1 = x_new[self.own] - u
            c2 = used - v
            dif = self.yd - c1 + c2
            a_new = c1 + fac[:, None] * dif
            b_new = c2 - fac[:, None] * dif
            u = u + a_new - x_new[self.own]
            v = v + b_new - used

            # (5) metrics
            err = np.linalg.norm(x_new - P.p_true, axis=1)
            rms = lambda z: float(np.sqrt(np.sum(z[hr] ** 2) / n_h_edges))
            p_own = rms(a_new - x_new[self.own])
            p_scr = rms(b_new - used)
            p_org = rms(b_new - view)
            dual = rho * rms(a_new - a)
            step = float(np.sqrt(np.mean(np.sum((x_new - x) ** 2, 1))))
            if True:  # always record; store_traces only controls what is returned
                rec["iteration"].append(k + 1)
                rec["rmse_honest_m"].append(float(np.sqrt(np.mean(err[hn] ** 2))))
                rec["rmse_all_m"].append(float(np.sqrt(np.mean(err ** 2))))
                rec["mean_err_honest_m"].append(float(err[hn].mean()))
                rec["max_err_honest_m"].append(float(err[hn].max()))
                rec["primal_own"].append(p_own); rec["primal_screened"].append(p_scr)
                rec["primal_original"].append(p_org); rec["dual"].append(dual)
                rec["node_step"].append(step)
                mm = mal_msg & hr
                rec["flagged"].append(int(flag_all.sum()))
                rec["tp"].append(int((flag_all & mm).sum())); rec["fp"].append(int((flag_all & hr & ~mm).sum()))
                rec["fn"].append(int((~flag_all & mm).sum())); rec["tn"].append(int((~flag_all & hr & ~mm).sum()))
                rec["ledger_submitted"].append(ledger["submitted"] if D.exchange == "ledger" else 0)
                rec["ledger_accepted"].append(ledger["accepted"] if D.exchange == "ledger" else 0)
                rec["ledger_rejected"].append(ledger["rejected"] if D.exchange == "ledger" else 0)
                rec["wls_cost"].append(wls_cost(P, x_new))
            x, a, b = x_new, a_new, b_new
            if k + 1 >= S.min_iter and p_own < S.eps_primal and p_scr < S.eps_primal and dual < S.eps_dual:
                converged = True
                break
        runtime = time.perf_counter() - t0
        return self._summarize(x, rec, converged, runtime)

    # ------------------------------------------------------------------
    def _summarize(self, x, rec, converged, runtime) -> dict:
        P, S = self.P, self.S
        it = rec["iteration"][-1] if rec["iteration"] else 0
        rm = np.array(rec["rmse_honest_m"])
        W = min(S.tail_window, len(rm)) if len(rm) else 0
        tail = rm[-W:] if W else rm
        tp, fp, fn, tn = (np.array(rec[k]) for k in ("tp", "fp", "fn", "tn"))
        half = len(tp) // 2

        def rate(num, den):
            den = float(np.sum(den))
            return float(np.sum(num) / den) if den > 0 else np.nan

        err = np.linalg.norm(x - P.p_true, axis=1)
        out = dict(
            iterations=int(it), converged=bool(converged), hit_max_iter=bool(not converged),
            final_rmse_honest_m=float(np.sqrt(np.mean(err[self.honest_nodes] ** 2))),
            final_rmse_all_m=float(np.sqrt(np.mean(err ** 2))),
            final_mean_err_honest_m=float(err[self.honest_nodes].mean()),
            final_max_err_honest_m=float(err[self.honest_nodes].max()),
            tail_mean_rmse_honest_m=float(tail.mean()) if len(tail) else np.nan,
            # Metric used for tables/reductions: final value for converged runs,
            # tail average for runs that hit max_iter (e.g. disruption), R1 C7.
            reported_rmse_honest_m=(float(np.sqrt(np.mean(err[self.honest_nodes] ** 2))) if converged
                                    else (float(tail.mean()) if len(tail) else np.nan)),
            tail_max_rmse_honest_m=float(tail.max()) if len(tail) else np.nan,
            false_alarm_rate=rate(fp, fp + tn),
            missed_detection_rate=rate(fn, fn + tp),
            false_alarm_rate_2nd_half=rate(fp[half:], fp[half:] + tn[half:]),
            missed_detection_rate_2nd_half=rate(fn[half:], fn[half:] + tp[half:]),
            total_flagged=int(np.sum(rec["flagged"])),
            total_malicious_msgs=int(np.sum(tp) + np.sum(fn)),
            ledger_submitted=int(np.sum(rec["ledger_submitted"])),
            ledger_accepted=int(np.sum(rec["ledger_accepted"])),
            ledger_rejected=int(np.sum(rec["ledger_rejected"])),
            echo_msgs_per_iter=int(self.echo_msgs_per_iter),
            runtime_s=float(runtime),
            n_uavs=P.n, n_edges=P.n_edges, n_malicious=int(self.mal_nodes.sum()),
            attack=self.A.kind, defense=self.D.label(), exchange=self.D.exchange,
            filter=self.D.filter, screening_mode=self.D.screening_mode,
            threshold_mode=self.D.threshold_mode, kappa=self.D.kappa, alpha_fa=self.D.alpha_fa,
            screen_start_iter=self.D.screen_start_iter, est_std=self.D.est_std, msr_F=self.D.msr_F,
            exclude_equivocators=self.D.exclude_equivocators, commit_policy=self.A.commit_policy,
            attack_magnitude=self.A.magnitude, adaptive_margin=self.A.adaptive_margin,
            tau_mean_m=float(self.tau.mean()), rho=S.rho, seed=P.seed,
            warnings="; ".join(self.warnings),
        )
        return dict(summary=out, x_hat=x, traces=rec if S.store_traces else None,
                    malicious=tuple(int(m) for m in np.where(self.mal_nodes)[0]),
                    targets=self.targets)


def run_case(problem: Problem, attack: AttackConfig, defense: DefenseConfig,
             solver: SolverConfig = SolverConfig()) -> dict:
    return SecureADMM(problem, attack, defense, solver).run()


# =============================================================================
# Helpers: legacy-plot adapter, relabelling test, the manuscript's Eq. (47)
# =============================================================================

def to_legacy_map_dict(problem: Problem, result: dict, scenario: str) -> dict:
    """Dictionary accepted by plot_localization_map(...) in the original
    plotting code (Updated_Code.ipynb, cell 0). The original code used
    y = p_j - p_i, so the sign is flipped here."""
    edges = [(int(i), int(j), -y) for (i, j), y in zip(problem.edges, problem.y)]
    return dict(scenario=scenario, p_true=problem.p_true, P_hat=result["x_hat"], g=problem.g,
                edges=edges, malicious_nodes=list(result["malicious"]),
                attack_targets={int(k): list(v) for k, v in result["targets"].items()},
                display_attack_targets={int(k): list(v) for k, v in result["targets"].items()},
                display_malicious_nodes=list(result["malicious"]),
                final_rmse_m=result["summary"]["final_rmse_honest_m"])


def manuscript_eq47_edge_update(x_i, x_j_screened, u_ij, u_ji, y_ij, theta, rho):
    """Literal Eq. (47): edge (i, j), i < j, solved with UAV i's state and UAV
    i's screened view of j. Used only to reproduce Reviewer 3's two-UAV example."""
    c1 = x_i - u_ij; c2 = x_j_screened - u_ji
    fac = theta / (2 * theta + rho)
    dif = y_ij - c1 + c2
    return c1 + fac * dif, c2 - fac * dif


def two_uav_relabelling_example(tau: float = 1.0, rho: float = 1.0, theta: float = 4.0) -> dict:
    """Reviewer 3, Comment 1b. Physical UAVs A at (0,0) and B at (10,0); y_AB = (0,0)."""
    A_pos, B_pos, y = np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 0.0])
    z = np.zeros(2)

    def screen(x_recv, x_send, y_recv_send):
        pred = x_recv - y_recv_send
        return pred if np.linalg.norm(x_send - pred) > tau else x_send

    # Manuscript Eq. (47): only the lower-index endpoint screens.
    # Labelling 1: A=0, B=1
    a1, b1 = manuscript_eq47_edge_update(A_pos, screen(A_pos, B_pos, y), z, z, y, theta, rho)
    # Labelling 2: B=0, A=1 (y_BA = -y_AB)
    b2, a2 = manuscript_eq47_edge_update(B_pos, screen(B_pos, A_pos, -y), z, z, -y, theta, rho)
    # Symmetric update: each endpoint uses its own screening and keeps its own copy.
    a_sym, _ = manuscript_eq47_edge_update(A_pos, screen(A_pos, B_pos, y), z, z, y, theta, rho)
    b_sym, _ = manuscript_eq47_edge_update(B_pos, screen(B_pos, A_pos, -y), z, z, -y, theta, rho)
    return dict(
        eq47_label1=dict(copy_A=a1.tolist(), copy_B=b1.tolist()),
        eq47_label2=dict(copy_A=a2.tolist(), copy_B=b2.tolist()),
        symmetric=dict(copy_A=a_sym.tolist(), copy_B=b_sym.tolist()),
        eq47_label_invariant=bool(np.allclose(a1, a2) and np.allclose(b1, b2)),
    )


def relabelling_test(problem: Problem, attack: AttackConfig, defense: DefenseConfig,
                     solver: SolverConfig, seed: int = 0) -> float:
    """Run the same physical problem with random relabelling; return max |difference|
    of the final estimates (mapped back to the original labels)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(problem.n)
    r1 = run_case(problem, attack, defense, solver)
    prob2 = problem.relabel(perm)
    att2 = AttackConfig(**{**asdict(attack),
                           "malicious": tuple(int(perm[m]) for m in attack.malicious),
                           "targets": None if attack.targets is None else
                           {int(perm[k]): [int(perm[t]) for t in v] for k, v in attack.targets.items()}})
    if att2.is_receiver_dependent():
        # Receiver-dependent directions are keyed by labels; keep them label-free for this test.
        att2 = AttackConfig(**{**asdict(att2), "receiver_dependent": False})
        attack = AttackConfig(**{**asdict(attack), "receiver_dependent": False})
        r1 = run_case(problem, attack, defense, solver)
    r2 = run_case(prob2, att2, defense, solver)
    x2_back = r2["x_hat"][perm]
    return float(np.max(np.abs(r1["x_hat"] - x2_back)))
