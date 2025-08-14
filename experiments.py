
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import argparse, time, os
from pathlib import Path
from env import PlusCorridorEnv
from policy import MoveWaitPolicy
from overlay import BroadcastGain

# --------- Helpers ---------

def run_episode(seed, method, dropout, params):
    """
    Runs a single episode and returns episode-level metrics and time series for optional plotting.
    method: one of {'no_comm','gain_mean','reward_broadcast','broadcast_gain','fixed_g_0.8',...}
    """
    rng = np.random.RandomState(seed)
    env = PlusCorridorEnv(H=24, W=24, n_robots=params["n_robots"], seed=seed)
    pol = MoveWaitPolicy(base_scale=1.0)
    steps = params["steps"]
    cycle_len = 4

    # overlay setup
    fuse = "median" if ("broadcast_gain" in method) else ("mean" if method=="gain_mean" else "median")
    one_byte = params.get("one_byte", False)
    overlay = BroadcastGain(
        n_agents=env.n_robots, rng=rng, kappa=params["kappa"], beta=params["beta"],
        alpha=params["alpha"], g_min=params["g_min"], g_max=params["g_max"], cycle_len=cycle_len,
        ttl_cycles=2, fuse=fuse, dropout=dropout, radius=params["radius"], one_byte=one_byte
    )

    # constant-g control
    fixed_g = None
    if method.startswith("fixed_g_"):
        fixed_g = float(method.split("_")[-1])

    realized_moves_total = 0
    desired_moves_total = 0
    rewards = []
    td_est = 0.0  # simple EWMA baseline for a TD-like error
    td_lambda = 0.1
    td_errs = []

    gain_disp_cycle = []
    pct_g_min = 0.0; pct_g_max = 0.0; g_at_min = 0; g_at_max = 0
    g_history = []

    intents_history = []
    realized_history = []

    # init gains to 1
    g = np.ones(env.n_robots)

    for t in range(steps):
        obs = env.observe()  # shape (N,2)
        base_logits = np.stack([pol.base_logits(o) for o in obs], axis=0)  # (N,2)

        # residual choice: surprisal by default
        # We compute surprisal relative to *pre-gate* base logits to avoid feedback
        base_p = np.exp(base_logits - base_logits.max(axis=1, keepdims=True))
        base_p = base_p / base_p.sum(axis=1, keepdims=True)

        # Step overlay timing
        start_cycle = overlay.step_before_act()
        if start_cycle:
            # Compute residuals available so far (use previous actions' surprisal if available)
            # We lack previous action choices at the very first step; approximate with base entropy
            if len(intents_history) == 0:
                surprisal = -np.log(base_p[:,0] + 1e-9)  # move surprisal
            else:
                last_acts = intents_history[-1]  # list of "move"/"wait"
                a_idx = np.array([0 if a=="move" else 1 for a in last_acts], dtype=int)
                last_probs = base_p[np.arange(env.n_robots), a_idx]
                surprisal = -np.log(last_probs + 1e-9)
            overlay.update_residuals(surprisal)
            # "broadcast" this cycle
            positions = np.array([[rb.r, rb.c] for rb in env.robots], dtype=int)
            g = overlay.broadcast_and_fuse(positions)
            g_history.append(g.copy())
            # Track bound stats (for episode-level report)
            g_at_min += (g <= overlay.g_min + 1e-9).mean()
            g_at_max += (g >= overlay.g_max - 1e-9).mean()

        # Determine per-agent gains for this step
        if method == "no_comm":
            g_step = np.ones(env.n_robots)
        elif method == "gain_mean":
            g_step = g.copy()  # overlay set up with mean fuse
        elif method == "reward_broadcast":
            # Use reward residuals AFTER step; here apply last g
            g_step = g.copy()
        elif method.startswith("fixed_g_"):
            g_step = np.ones(env.n_robots) * fixed_g
        else:  # broadcast_gain (median)
            g_step = g.copy()

        # Sample intents given move/wait gating
        intents = []
        for i in range(env.n_robots):
            # Apply gain only to 'move' logit
            act, surpr = MoveWaitPolicy().sample(rng, base_logits[i], g=g_step[i])
            intents.append(act)
        intents_history.append(intents)

        realized, shared_reward, info = env.step(intents)
        realized_history.append(realized.tolist())

        # Update TD-like error (EWMA baseline over shared reward)
        prev_td = td_est
        td_est = (1-td_lambda)*td_est + td_lambda*shared_reward
        delta = shared_reward + 0.99*td_est - prev_td
        td_errs.append(float(delta))

        realized_moves_total += realized.sum()
        desired_moves_total += (np.array(intents)=="move").sum()
        rewards.append(shared_reward)

        # For reward_broadcast, update residuals with clipped shared reward each cycle start
        if method == "reward_broadcast" and start_cycle:
            overlay.update_residuals(np.clip(np.array([shared_reward]*env.n_robots), -1, 1))
            positions = np.array([[rb.r, rb.c] for rb in env.robots], dtype=int)
            g = overlay.broadcast_and_fuse(positions)

    # CDE analog: fraction of desired "move" intents that were blocked by arbitration
    desired = 0; realized_mv = 0
    for inf in range(len(realized_history)):
        desired += (np.array(intents_history[inf])=="move").sum()
        realized_mv += np.array(realized_history[inf]).sum()
    cde = (desired - realized_mv) / max(1, desired)

    ret = np.sum(rewards)  # episodic return (sum of shared rewards)
    td_var = float(np.var(td_errs))
    gain_disp = float(np.std(np.array(g_history), axis=1).mean()) if len(g_history)>0 else 0.0
    pct_g_min = float(g_at_min / max(1, len(g_history)))
    pct_g_max = float(g_at_max / max(1, len(g_history)))

    row_occ, col_occ = env.occupancy_mats()
    out = {
        "return": ret,
        "cde": cde,
        "td_var": td_var,
        "gain_dispersion": gain_disp,
        "pct_g_min": pct_g_min,
        "pct_g_max": pct_g_max,
        "row_occ": row_occ,
        "col_occ": col_occ,
        "intents": intents_history,
        "realized": realized_history,
    }
    return out

def run_grid(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [1,2,3] if args.mode=="quick" else [1,2,3,4,5]
    drop_model = "bernoulli"
    dropouts = [0.0, 0.1, 0.3]
    methods = ["no_comm", "gain_mean", "reward_broadcast", "broadcast_gain"]
    params = dict(n_robots=20, steps=500, alpha=0.5, beta=0.2, kappa=0.5, g_min=0.5, g_max=1.5, radius=5)

    rows = []
    # Run episodes
    for method in methods:
        for d in dropouts:
            for seed in seeds:
                start = time.time()
                res = run_episode(seed=seed, method=method, dropout=d, params=params)
                dt = time.time()-start
                rows.append({
                    "seed": seed, "method": method, "drop_model": drop_model, "dropout": d,
                    "return": res["return"], "cde": res["cde"], "td_var": res["td_var"],
                    "gain_dispersion": res["gain_dispersion"], "pct_g_min": res["pct_g_min"],
                    "pct_g_max": res["pct_g_max"], "map": "congested_v1",
                    "alpha": params["alpha"], "beta": params["beta"], "kappa": params["kappa"],
                    "radius": params["radius"], "payload_bytes": 2, "runtime_s": dt
                })
                # Save one set of occupancy heatmaps for a canonical setting
                if method=="no_comm" and d==0.0 and seed==seeds[0]:
                    plot_space_time(res["row_occ"], res["col_occ"], out_dir)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"main_results.csv", index=False)

    # Make plots
    make_main_plots(df, out_dir)

    # theory + robustness micro-plots
    plot_theory_varratio(out_dir)
    plot_median_robustness(out_dir)

    # write simple report
    write_report(df, out_dir)

def agg(df, metric, by=["method","dropout"]):
    g = df.groupby(by)[metric].agg(["mean","count","std"]).reset_index()
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g

def make_main_plots(df, out_dir: Path):
    # Return vs Dropout
    g = agg(df, "return")
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    for m in ["no_comm","gain_mean","reward_broadcast","broadcast_gain"]:
        if m not in g["method"].unique(): continue
        d = g[g["method"]==m]
        ax.errorbar(d["dropout"], d["mean"], yerr=d["ci95"], marker="o", label=m)
    ax.set_xlabel("Dropout Rate"); ax.set_ylabel("Episodic Return")
    ax.set_title("Return vs Dropout (Bernoulli)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(out_dir/"return_vs_dropout_bernoulli.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir/"paper_figs_return_vs_dropout_bernoulli.pdf", bbox_inches="tight")
    plt.close(fig)

    # CDE vs Dropout
    g = agg(df, "cde")
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    for m in ["no_comm","gain_mean","reward_broadcast","broadcast_gain"]:
        if m not in g["method"].unique(): continue
        d = g[g["method"]==m]
        ax.errorbar(d["dropout"], d["mean"], yerr=d["ci95"], marker="o", label=m)
    ax.set_xlabel("Dropout Rate"); ax.set_ylabel("CDE (fraction blocked intents)")
    ax.set_title("CDE vs Dropout (Bernoulli)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(out_dir/"cde_vs_dropout_bernoulli.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir/"paper_figs_cde_vs_dropout_bernoulli.pdf", bbox_inches="tight")
    plt.close(fig)

    # TD variance vs Dropout
    g = agg(df, "td_var")
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    for m in ["no_comm","gain_mean","broadcast_gain"]:
        if m not in g["method"].unique(): continue
        d = g[g["method"]==m]
        ax.errorbar(d["dropout"], d["mean"], yerr=d["ci95"], marker="o", label=m)
    ax.set_xlabel("Dropout Rate"); ax.set_ylabel("TD-like Error Variance")
    ax.set_title("TD-Error Variance vs Dropout (Bernoulli)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(out_dir/"tdvar_vs_dropout_bernoulli.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir/"paper_figs_tdvar_vs_dropout_bernoulli.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_space_time(row_occ, col_occ, out_dir: Path):
    row = np.array(row_occ, dtype=int); col = np.array(col_occ, dtype=int)
    fig, ax = plt.subplots(figsize=(7.0,3.2))
    ax.imshow(row.T, origin="lower", aspect="auto")
    ax.set_xlabel("Time step"); ax.set_ylabel("Column index (central row)")
    ax.set_title("Space–Time Occupancy (central row)")
    fig.savefig(out_dir/"space_time_congestion.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir/"paper_figs_space_time_congestion.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_theory_varratio(out_dir: Path):
    # High-pass variance ratio vs beta
    T = 10000
    betas = np.linspace(0.05, 0.95, 10)
    ratios = []
    rng = np.random.RandomState(0)
    p = rng.randn(T)  # white noise
    for beta in betas:
        tau = np.zeros(T)
        for t in range(1,T):
            tau[t] = (1-beta)*tau[t-1] + beta*p[t]
        diff = p - tau
        ratios.append(np.var(diff) / np.var(p))
    fig, ax = plt.subplots(figsize=(5.6,4.0))
    ax.plot(betas, ratios, marker="o", label="Empirical (ρ=0)")
    # analytic for white noise: 2(1-β)^2/(2-β)
    analytic = [2*(1-b)**2/(2-b) for b in betas]
    ax.plot(betas, analytic, linestyle="--", label="Analytic (white)")
    ax.set_xlabel("β"); ax.set_ylabel("Var(p−τ)/Var(p)")
    ax.set_title("High‑pass Variance Ratio")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(out_dir/"theory_varratio.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir/"paper_figs_theory_varratio.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_median_robustness(out_dir: Path):
    # Mini robustness: mean vs median under outliers
    rng = np.random.RandomState(0)
    N = 21
    max_k = 10
    xs = list(range(0, max_k+1))
    mean_err = []
    med_err = []
    for k in xs:
        base = rng.randn(N)
        # replace k points with big outliers
        idx = rng.choice(N, size=k, replace=False)
        base[idx] = base[idx] + rng.choice([-50,50], size=k)
        mean_err.append(np.abs(base.mean()))
        med_err.append(np.abs(np.median(base)))
    fig, ax = plt.subplots(figsize=(5.6,4.0))
    ax.plot(xs, med_err, marker="o", label="Median")
    ax.plot(xs, mean_err, marker="s", label="Mean")
    ax.set_xlabel("Number of outliers"); ax.set_ylabel("Mean absolute error from 0")
    ax.set_title("Median vs Mean Robustness (N=21)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(out_dir/"median_vs_mean_robustness.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir/"paper_figs_median_vs_mean_robustness.pdf", bbox_inches="tight")
    plt.close(fig)

def write_report(df, out_dir: Path):
    g = df.groupby(["method","dropout"])[["return","cde","td_var"]].mean().reset_index()
    # simple summary
    lines = []
    lines.append("# REPORT (quick)\n")
    def gval(m, d, col):
        sel = g[(g["method"]==m) & (g["dropout"]==d)]
        return float(sel[col].iloc[0]) if len(sel)>0 else float("nan")
    lines.append(f"- At dropout=0.0 (Bernoulli): Return BG={gval('broadcast_gain',0.0,'return'):.3f} vs no_comm={gval('no_comm',0.0,'return'):.3f}; CDE BG={gval('broadcast_gain',0.0,'cde'):.3f} vs no_comm={gval('no_comm',0.0,'cde'):.3f}")
    lines.append(f"- Robustness (dropout=0.3): Return median‑fused BG={gval('broadcast_gain',0.3,'return'):.3f} vs mean‑fused={gval('gain_mean',0.3,'return'):.3f}")
    lines.append(f"- TD‑var vs no_comm at 0.0: BG={gval('broadcast_gain',0.0,'td_var'):.4f} vs {gval('no_comm',0.0,'td_var'):.4f}")
    (out_dir/"REPORT.md").write_text("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick", choices=["quick","sanity"])
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default="results/")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        df = pd.read_csv(out/"main_results.csv")
        make_main_plots(df, out)
    else:
        run_grid(args)

if __name__ == "__main__":
    main()
