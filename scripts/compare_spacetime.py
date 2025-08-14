# scripts/compare_spacetime.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments import run_episode

def main(seed=2, steps=500, drop=0.3, out="results_visuals", drop_model="burst"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    params = dict(n_robots=20, steps=steps, alpha=0.5, beta=0.2, kappa=0.5, g_min=0.5, g_max=1.5, radius=5)

    res_nc = run_episode(seed, "no_comm", drop, params, drop_model)
    res_bg = run_episode(seed, "broadcast_gain", drop, params, drop_model)

    def imshow(ax, mat, title):
        ax.imshow(np.array(mat).T, origin="lower", aspect="auto", cmap='hot')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Central row index")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    imshow(axs[0], res_nc["row_occ"], "No-Comm")
    imshow(axs[1], res_bg["row_occ"], "Broadcast-Gain")
    fig.suptitle(f"Space-Time Occupancy (drop={drop}, seed={seed}, {drop_model})")
    fig.savefig(out/f"spacetime_compare_{drop_model}.png", bbox_inches="tight", dpi=200)
    fig.savefig(out/f"paper_figs_spacetime_compare_{drop_model}.pdf", bbox_inches="tight")
    plt.close(fig)
    
    print(f"No-Comm CDE: {res_nc['cde']:.3f}, Return: {res_nc['return']:.1f}")
    print(f"BG CDE: {res_bg['cde']:.3f}, Return: {res_bg['return']:.1f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--out", default="results_visuals")
    ap.add_argument("--drop_model", default="burst", choices=["bernoulli", "burst"])
    args = ap.parse_args()
    main(seed=args.seed, steps=args.steps, drop=args.drop, out=args.out, drop_model=args.drop_model)