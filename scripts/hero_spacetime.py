#!/usr/bin/env python
"""
Generate the HERO figure: side-by-side space-time congestion heatmaps
Shows jam formation, propagation, and dissipation with/without Broadcast-Gain
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from experiments import run_episode

def compute_congestion_intensity(row_occ, col_occ):
    """Convert occupancy to congestion intensity (normalized density)"""
    # Use row occupancy as primary (horizontal corridor)
    row = np.array(row_occ, dtype=float)
    
    # Compute local density with smoothing
    from scipy.ndimage import gaussian_filter1d
    
    # Smooth across space for visual clarity
    for t in range(len(row)):
        row[t] = gaussian_filter1d(row[t], sigma=1.0)
    
    # Normalize to [0, 1] for color mapping
    if row.max() > 0:
        row = row / row.max()
    
    return row.T  # Transpose for space (y) vs time (x)

def main(seed=2, steps=800, drop=0.3, out="results_hero", drop_model="burst"):
    """Generate the hero figure with enhanced visualization"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Generating HERO space-time figure...")
    
    # Run parameters
    params = dict(n_robots=40, steps=steps, alpha=0.5, beta=0.2, 
                  kappa=0.5, g_min=0.5, g_max=1.5, radius=5)
    
    # Run both scenarios with same seed
    print("  Running no-comm scenario...")
    res_nc = run_episode(seed, "no_comm", drop, params, drop_model)
    
    print("  Running broadcast-gain scenario...")
    res_bg = run_episode(seed, "broadcast_gain", drop, params, drop_model)
    
    # Process occupancy data
    try:
        from scipy.ndimage import gaussian_filter1d
        smooth_available = True
    except ImportError:
        smooth_available = False
        print("  Note: scipy not available, using raw data")
    
    if smooth_available:
        congestion_nc = compute_congestion_intensity(res_nc["row_occ"], res_nc["col_occ"])
        congestion_bg = compute_congestion_intensity(res_bg["row_occ"], res_bg["col_occ"])
    else:
        # Fallback without smoothing
        congestion_nc = np.array(res_nc["row_occ"], dtype=float).T
        congestion_bg = np.array(res_bg["row_occ"], dtype=float).T
        if congestion_nc.max() > 0:
            congestion_nc = congestion_nc / congestion_nc.max()
        if congestion_bg.max() > 0:
            congestion_bg = congestion_bg / congestion_bg.max()
    
    # Create the hero figure
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          wspace=0.15, hspace=0.25)
    
    # Common colormap settings
    cmap = 'hot'  # Hot colormap: black -> red -> yellow -> white
    vmin, vmax = 0, 1
    
    # Top panel: No-comm
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(congestion_nc, aspect='auto', cmap=cmap, 
                     vmin=vmin, vmax=vmax, origin='lower', interpolation='bilinear')
    ax1.set_title(f'WITHOUT Broadcast-Gain (CDE = {res_nc["cde"]:.3f})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Corridor Position', fontsize=11)
    ax1.set_xlabel('')
    
    # Add jam markers
    jam_start = np.where(congestion_nc.mean(axis=0) > 0.3)[0]
    if len(jam_start) > 0:
        ax1.axvline(jam_start[0], color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(jam_start[0], congestion_nc.shape[0]*0.9, 'Jam onset', 
                color='white', fontsize=9, alpha=0.8)
    
    # Bottom panel: Broadcast-Gain
    ax2 = fig.add_subplot(gs[1, :])
    im2 = ax2.imshow(congestion_bg, aspect='auto', cmap=cmap,
                     vmin=vmin, vmax=vmax, origin='lower', interpolation='bilinear')
    ax2.set_title(f'WITH Broadcast-Gain (CDE = {res_bg["cde"]:.3f})', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Corridor Position', fontsize=11)
    ax2.set_xlabel('Time Step', fontsize=11)
    
    # Add clear time marker
    jam_clear = np.where(congestion_bg.mean(axis=0) < 0.2)[0]
    if len(jam_clear) > 100:  # Find first clear after initial period
        clear_idx = jam_clear[jam_clear > 100][0] if any(jam_clear > 100) else jam_clear[-1]
        ax2.axvline(clear_idx, color='lightgreen', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(clear_idx, congestion_bg.shape[0]*0.9, 'Jam clears', 
                color='lightgreen', fontsize=9, alpha=0.8)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Congestion Intensity', rotation=270, labelpad=20, fontsize=11)
    
    # Overall title
    fig.suptitle('Space-Time Congestion: Impact of 2-byte Broadcast-Gain Overlay',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add parameters subtitle
    subtitle = f'({params["n_robots"]} robots, {drop:.0%} {drop_model} dropout, seed={seed})'
    fig.text(0.5, 0.94, subtitle, ha='center', fontsize=10, alpha=0.7)
    
    # Save
    fig.savefig(out/"hero_spacetime_congestion.png", bbox_inches='tight', dpi=300)
    fig.savefig(out/"paper_hero_spacetime_congestion.pdf", bbox_inches='tight')
    plt.close(fig)
    
    # Print metrics
    print(f"\nHERO figure generated!")
    print(f"  No-Comm:        CDE={res_nc['cde']:.3f}, Return={res_nc['return']:.1f}")
    print(f"  Broadcast-Gain: CDE={res_bg['cde']:.3f}, Return={res_bg['return']:.1f}")
    print(f"  CDE reduction: {(res_nc['cde']-res_bg['cde'])/res_nc['cde']*100:.1f}%")
    print(f"\nSaved to: {out}/hero_spacetime_congestion.png")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--out", default="results_hero")
    ap.add_argument("--drop_model", default="burst", choices=["bernoulli", "burst"])
    ap.add_argument("--n_robots", type=int, default=40)
    args = ap.parse_args()
    main(seed=args.seed, steps=args.steps, drop=args.drop, out=args.out, drop_model=args.drop_model)