# scripts/fundamental.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from experiments import run_episode

def main(out="results_fundamental", seeds=(1,2,3), steps=400, drop=0.3, drop_model="burst"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Ns = [10, 20, 30, 40, 60, 80, 100, 120]
    rows = []
    base_params = dict(alpha=0.5, beta=0.2, kappa=0.5, g_min=0.5, g_max=1.5, radius=5, steps=steps)

    for N in Ns:
        for method in ["no_comm", "broadcast_gain"]:
            for s in seeds:
                params = dict(base_params)
                params["n_robots"] = N
                res = run_episode(seed=s, method=method, dropout=drop, params=params, drop_model=drop_model)
                flow = sum(np.array(h).sum() for h in res["realized"]) / steps  # moves per step
                rows.append(dict(N=N, method=method, flow=flow, cde=res["cde"]))

    df = pd.DataFrame(rows)
    df.to_csv(out/"fundamental.csv", index=False)
    
    # plot
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    for m in ["no_comm", "broadcast_gain"]:
        d = df[df.method==m].groupby("N")["flow"].mean()
        ax.plot(d.index, d.values, marker="o", label=m)
    ax.set_xlabel("Robots (density proxy)")
    ax.set_ylabel("Throughput (moves/step)")
    ax.set_title(f"Fundamental Diagram (drop={drop}, {drop_model})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out/f"fundamental_flow_{drop_model}.png", bbox_inches="tight", dpi=200)
    fig.savefig(out/f"paper_figs_fundamental_flow_{drop_model}.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_fundamental")
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--drop_model", default="burst", choices=["bernoulli", "burst"])
    args = ap.parse_args()
    main(out=args.out, drop=args.drop, drop_model=args.drop_model)