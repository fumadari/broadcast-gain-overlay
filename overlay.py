
from __future__ import annotations
import numpy as np

class BroadcastGain:
    """
    2-byte phasic-tonic overlay with coordinate-wise median fuse.
    - Every cycle_len steps, each agent computes p_t, tau_t from residual (surprisal by default),
      quantizes both to int8 with global symmetric mid-tread scale Δ = κ/127, and broadcasts 2 bytes.
    - Each receiver fuses with coordinate-wise median over packets from neighbors (incl. self)
      that arrived within TTL cycles (packet loss allowed). Dropout is applied per packet iid.
    - Gain: g = clip(1 + α*(p_bar - tau_bar), g_min, g_max), held constant until next broadcast.
    """
    def __init__(self, n_agents, rng, kappa=0.5, beta=0.2, alpha=0.5,
                 g_min=0.5, g_max=1.5, cycle_len=4, ttl_cycles=2,
                 fuse="median", dropout=0.0, radius=5, one_byte=False,
                 drop_model="bernoulli",
                 ge_p_g2b=0.05, ge_p_b2g=0.20,  # state transition probs
                 ge_pdrop_g=0.05, ge_pdrop_b=0.9):  # drop probs by state
        self.n = n_agents; self.rng = rng
        self.kappa = kappa; self.beta = beta; self.alpha = alpha
        self.g_min = g_min; self.g_max = g_max
        self.cycle_len = cycle_len; self.ttl = ttl_cycles
        self.fuse = fuse
        self.dropout = dropout
        self.radius = radius
        self.one_byte = one_byte
        self.drop_model = drop_model
        self.ge_p_g2b = ge_p_g2b
        self.ge_p_b2g = ge_p_b2g
        self.ge_pdrop_g = ge_pdrop_g
        self.ge_pdrop_b = ge_pdrop_b

        self.DELTA = kappa / 127.0
        self.t = 0
        self.p = np.zeros(self.n)     # last phasic
        self.tau = np.zeros(self.n)   # tonic EWMA
        self.g = np.ones(self.n)
        # last received per-agent (store (p, tau, time_cycle))
        self.rx_p = [ [] for _ in range(self.n) ]
        if self.drop_model == "burst":
            # 0 = good, 1 = bad, per (receiver i, sender j)
            self._ge_state = np.zeros((self.n, self.n), dtype=np.int8)

    def _link_drops(self, i, j):
        """Return True if packet j->i drops this cycle."""
        if self.drop_model == "bernoulli":
            return self.rng.rand() < self.dropout
        # Gilbert-Elliott
        s = int(self._ge_state[i, j])
        # evolve state
        if s == 0:  # good
            if self.rng.rand() < self.ge_p_g2b:
                s = 1
        else:       # bad
            if self.rng.rand() < self.ge_p_b2g:
                s = 0
        self._ge_state[i, j] = s
        p = self.ge_pdrop_b if s == 1 else self.ge_pdrop_g
        # modulate by global dropout knob for quick sweeps
        p = min(1.0, p * max(1e-9, 1.0 * self.dropout / max(1e-9, 0.3)))
        return self.rng.rand() < p

    def neighbors(self, positions):
        # positions: array of shape (n, 2) with (r,c)
        # returns list of neighbor indices (incl self) per agent within Manhattan radius
        nbrs = []
        for i in range(self.n):
            ri, ci = positions[i]
            d = np.abs(positions[:,0] - ri) + np.abs(positions[:,1] - ci)
            idx = np.where(d <= self.radius)[0]
            if i not in idx:
                idx = np.append(idx, i)
            nbrs.append(idx)
        return nbrs

    def step_before_act(self):
        # called each environment step; determines whether we're in a broadcast cycle
        self.t += 1
        return ((self.t-1) % self.cycle_len) == 0

    def update_residuals(self, residuals):
        # residuals: vector length n (e.g., surprisal or clipped shared reward proxy)
        self.p = np.clip(self.kappa * residuals, -self.kappa, self.kappa)
        # tonic EWMA
        self.tau = (1 - self.beta) * self.tau + self.beta * self.p

    def quantize(self, x):
        q = np.round(x / self.DELTA).astype(int)
        q = np.clip(q, -127, 127)
        return q

    def dequantize(self, q):
        return q.astype(float) * self.DELTA

    def broadcast_and_fuse(self, positions):
        """
        positions: (n,2) array with (r,c).
        Returns updated g for each agent (held for next cycle_len-1 steps).
        """
        ncycle = (self.t-1) // self.cycle_len  # cycle index
        # Form packets
        if self.one_byte:
            q_diff = self.quantize(self.p - self.tau)
            packets = np.stack([q_diff], axis=1)  # shape (n,1)
        else:
            packets = np.stack([self.quantize(self.p), self.quantize(self.tau)], axis=1)

        # Dropout & store arrival at receivers (including self)
        nbrs = self.neighbors(positions)
        for i in range(self.n):
            for j in nbrs[i]:
                if self._link_drops(i, j):
                    continue  # dropped packet
                # deliver packet j -> i
                if self.one_byte:
                    self.rx_p[i].append( (j, np.array([packets[j,0]]), ncycle) )
                else:
                    self.rx_p[i].append( (j, packets[j].copy(), ncycle) )

        # Fuse (coordinate-wise median or mean) over freshest packets within TTL
        new_g = np.empty(self.n, dtype=float)
        for i in range(self.n):
            # Keep only packets within TTL, then take the FRESHEST per sender j
            keep = [p for p in self.rx_p[i] if (ncycle - p[2]) < self.ttl]  # (j, payload, cycle)
            if not keep:
                new_g[i] = self.g[i]  # hold previous
                continue
            
            # Deduplicate: keep only the freshest packet per source
            latest = {}
            for (j, payload, cyc) in keep:
                if (j not in latest) or (cyc > latest[j][1]):
                    latest[j] = (payload, cyc)
            
            arr = np.stack([v[0] for v in latest.values()], axis=0)  # shape (m, 1|2)
            if self.one_byte:
                diff = self.dequantize(arr[:,0])
                if self.fuse == "median":
                    fused = np.median(diff)
                else:
                    fused = float(diff.mean())
                # reconstruct a p,tau pair under one-byte mode by treating tau=0
                p_bar_minus_tau = fused
                g_i = 1.0 + self.alpha * p_bar_minus_tau
            else:
                pvals = self.dequantize(arr[:,0])
                tvals = self.dequantize(arr[:,1])
                if self.fuse == "median":
                    p_bar = float(np.median(pvals))
                    tau_bar = float(np.median(tvals))
                else:
                    p_bar = float(pvals.mean()); tau_bar = float(tvals.mean())
                g_i = 1.0 + self.alpha * (p_bar - tau_bar)
            new_g[i] = np.clip(g_i, self.g_min, self.g_max)
        
        # Prune mailboxes to prevent unbounded growth
        for i in range(self.n):
            self.rx_p[i] = [p for p in self.rx_p[i] if (ncycle - p[2]) < self.ttl]
        
        self.g = new_g
        return self.g.copy()
