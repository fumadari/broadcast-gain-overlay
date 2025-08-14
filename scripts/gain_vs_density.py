# scripts/gain_vs_density.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments import run_episode

def manhattan_density(P, R):
    """Calculate local density for each agent within Manhattan radius R"""
    # Vectorized computation for efficiency
    d = np.abs(P[:, None, :] - P[None, :, :]).sum(-1)
    return (d <= R).sum(1) - 1  # exclude self

def main(out="results_density", R=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    
    params = dict(n_robots=20, steps=500, alpha=0.5, beta=0.2, kappa=0.5, g_min=0.5, g_max=1.5, radius=5)
    res = run_episode(1, "broadcast_gain", 0.3, params, "burst")
    
    # Take only broadcast instants (cycle_len=4)
    G = np.stack(res["g_history"], axis=0)  # [cycles, n]
    P = np.array(res["positions"])[::4]      # positions each broadcast cycle
    
    # Ensure we have matching lengths
    min_len = min(len(G), len(P))
    G = G[:min_len]
    P = P[:min_len]
    
    D = np.stack([manhattan_density(p, R) for p in P], axis=0)  # [cycles, n]
    
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(D.flatten(), G.flatten(), s=6, alpha=0.5)
    ax.set_xlabel(f"Local density (R={R})")
    ax.set_ylabel("g")
    ax.set_title("Broadcast-Gain vs Local Density (drop=0.3)")
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(D.flatten(), G.flatten(), 1)
    p = np.poly1d(z)
    x_trend = np.linspace(D.min(), D.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.7, label=f"Trend: g = {z[0]:.3f}*density + {z[1]:.3f}")
    ax.legend()
    
    fig.savefig(out/"gain_vs_density.png", bbox_inches="tight", dpi=200)
    fig.savefig(out/"paper_figs_gain_vs_density.pdf", bbox_inches="tight")
    plt.close(fig)
    
    print(f"Correlation coefficient: {np.corrcoef(D.flatten(), G.flatten())[0,1]:.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_density")
    ap.add_argument("--R", type=int, default=4, help="Manhattan radius for density calculation")
    args = ap.parse_args()
    main(out=args.out, R=args.R)