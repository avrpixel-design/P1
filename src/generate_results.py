#!/usr/bin/env python3
"""
================================================================================
GENERATE RESULTS FROM CACHE - V1(P1) FINAL
================================================================================
Generates publication outputs from cached MI estimation results.

Primary outputs (Results/):
- Figure 2: MI heatmap across combinations (`P1_Extra1`)
- Convergence dynamics (`P1_fig3_convergence`)
- Table 1: Feature extraction summary
- Table 2: Cross-dimension MI analysis (`P1_table2_mi_summary`, `P1_table2_mi_mean_sd`)
- Table 3: Source attribution statistics
- Combined LaTeX bundle (`P1_all_tables.tex`)

Improved visuals (Results1/):
- Improved Figure 2 (`P1_fig2_mi_heatmap`)
- Improved Figure 4 (`P1_fig4_attribution`)

Notes:
- Some legacy figures/tables are intentionally disabled in the current script version.
- Use `output/cache` for direct reproduction, or `output/cache_new` after fresh cache generation.
================================================================================
"""

import argparse
import importlib.util
import math
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import librosa.display
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy import stats
from typing import Dict, Tuple, List
import logging
import json
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# STYLE SETTINGS
# ============================================================================

# Color palette
COLOR_SCHEME = {
    # Primary dimension colors (used for bars, scatter, lines)
    'Emotional': '#C96A4A',     # warm burnt orange
    'Linguistic': '#2F3E63',    # deep slate navy
    'Pathological': '#6F8F7A',  # muted clinical green

    # Secondary colors for source-filter
    'Source': '#6B5FA7',        # deep purple
    'Filter': '#B8A6D9',        # light purple

    # Clustering metrics colors
    'Silhouette': '#3C7A89',        # muted teal-blue (distinct from slate navy)
    'DaviesBouldin': "#DA5EBB",    # desaturated pink (not emotional orange)
    'CalinskiHarabasz': "#4F7D5A", # cool moss green (distinct from pathological green)
    'Stability': '#808080',         # neutral grey

    # [PAPER 1 - v5.0] MI Convergence colors
    'MINE': '#C0392B',         # dark crimson (lower bound, grounded)
    'CLUB': '#1F6FB2',         # deep technical blue (upper bound)
    'KSG': '#2A9D8F',          # teal (non-parametric baseline)
    'True_MI': '#1E8449',      # forest green (converged, stable)
    'Uncertainty': '#D4AC0D',  # muted amber (band/region, non-dominant)
    'Final': '#1E8449',        # alias for converged/final estimate
    'Combined': '#1E8449',     # alias for combined estimate

    # Threshold/reference lines
    'HighThreshold': '#1B5E20',  # deep forest green (print-safe)
    'LowThreshold': '#EF6C00',   # restrained amber-orange (not emotional)
    'Neutral': '#616161',        # darker neutral tone (visible in print)
    'Threshold': '#D4AC0D',      # muted amber (matches uncertainty band)
    'Stable': '#1E8449',         # stable region/labels
    'Unstable': '#C0392B',       # unstable region/labels

    # Heatmap colormap
    'Heatmap': 'cividis',

    # Neutral UI helpers
    'Text': '#333333',
    'Grid': '#E0E0E0',
}

# Backward-compatible alias for existing references.
COLORS = COLOR_SCHEME

# Matplotlib settings
plt.rcParams.update({
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    
    # Figure settings
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    # Axes settings
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.axisbelow': True,
    
    # Grid
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    
    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    
    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'none',
    
    # Ticks
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# Figure sizes for single/double column (INTERSPEECH format)
FIG_WIDTH_SINGLE = 3.5  # inches (single column)
FIG_WIDTH_DOUBLE = 7.0  # inches (double column)


# ============================================================================
# KSG ESTIMATOR
# ============================================================================

class KSGEstimator:
    def __init__(self, k: int = 5):
        self.k = k
    
    def estimate(self, x: np.ndarray, y: np.ndarray) -> float:
        n = x.shape[0]
        if n < self.k + 1:
            return 0.0
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-10)
        y = (y - np.mean(y, axis=0)) / (np.std(y, axis=0) + 1e-10)
        xy = np.hstack([x, y])
        tree_xy, tree_x, tree_y = cKDTree(xy), cKDTree(x), cKDTree(y)
        distances, _ = tree_xy.query(xy, k=self.k + 1, p=np.inf)
        eps = distances[:, -1]
        n_x = np.array([len(tree_x.query_ball_point(x[i], eps[i] - 1e-10, p=np.inf)) - 1 for i in range(n)])
        n_y = np.array([len(tree_y.query_ball_point(y[i], eps[i] - 1e-10, p=np.inf)) - 1 for i in range(n)])
        n_x, n_y = np.maximum(n_x, 1), np.maximum(n_y, 1)
        mi = digamma(self.k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
        return max(0.0, mi)
    
    def estimate_with_std(self, x: np.ndarray, y: np.ndarray, n_bootstrap: int = 10) -> Tuple[float, float]:
        n = x.shape[0]
        estimates = []
        for b in range(n_bootstrap):
            np.random.seed(SEED + b)
            idx = np.random.choice(n, n, replace=True)
            estimates.append(self.estimate(x[idx], y[idx]))
        return np.mean(estimates), np.std(estimates)


# ============================================================================
# NEURAL MI ESTIMATORS
# ============================================================================

class CriticNetwork(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


class ConditionalNetwork(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.mu_head = nn.Linear(hidden_dim, y_dim)
        self.logvar_head = nn.Linear(hidden_dim, y_dim)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -1.0)
    
    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), torch.clamp(self.logvar_head(h), -6, 2)
    
    def log_likelihood(self, x, y):
        mu, logvar = self.forward(x)
        return -0.5 * (logvar + (y - mu)**2 / (torch.exp(logvar) + 1e-8)).sum(dim=1)


class BoundedMIEstimator:
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256, lr: float = 1e-4):
        self.mine_net = CriticNetwork(x_dim, y_dim, hidden_dim)
        self.club_net = ConditionalNetwork(x_dim, y_dim, hidden_dim)
        self.mine_opt = optim.Adam(self.mine_net.parameters(), lr=lr, weight_decay=1e-5)
        self.club_opt = optim.Adam(self.club_net.parameters(), lr=lr, weight_decay=1e-5)
        self.mine_scheduler = ReduceLROnPlateau(self.mine_opt, mode='max', factor=0.5, patience=10)
        self.club_scheduler = ReduceLROnPlateau(self.club_opt, mode='min', factor=0.5, patience=10)
        self.ema_alpha, self.running_mean, self.ema_steps = 0.01, 1.0, 0
        self.mine_history, self.club_history = [], []
    
    def compute_mine(self, x, y, update_ema=True):
        t_joint = self.mine_net(x, y)
        t_marginal = torch.clamp(self.mine_net(x, y[torch.randperm(y.size(0))]), -10, 10)
        exp_t = torch.exp(t_marginal)
        if update_ema and self.mine_net.training:
            self.ema_steps += 1
            self.running_mean = (1 - self.ema_alpha) * self.running_mean + self.ema_alpha * exp_t.mean().detach().item()
            corrected = self.running_mean / (1 - (1 - self.ema_alpha) ** self.ema_steps)
            return torch.mean(t_joint) - torch.log(torch.tensor(corrected + 1e-8))
        return torch.mean(t_joint) - torch.log(exp_t.mean() + 1e-8)
    
    def compute_club(self, x, y):
        pos_ll = self.club_net.log_likelihood(x, y)
        neg_ll = self.club_net.log_likelihood(x, y[torch.randperm(y.size(0))])
        return torch.mean(pos_ll) - torch.mean(neg_ll)
    
    def train(self, x_data, y_data, epochs=100, batch_size=256, patience=15):
        dataset = TensorDataset(torch.FloatTensor(x_data), torch.FloatTensor(y_data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.mine_history, self.club_history = [], []
        best_gap, patience_counter, converged_epoch = float('inf'), 0, None
        self.mine_net.train(); self.club_net.train()
        
        for epoch in range(epochs):
            epoch_mine, epoch_club = [], []
            for x_batch, y_batch in loader:
                self.mine_opt.zero_grad()
                mine_mi = self.compute_mine(x_batch, y_batch)
                (-mine_mi).backward()
                torch.nn.utils.clip_grad_norm_(self.mine_net.parameters(), 1.0)
                self.mine_opt.step()
                
                self.club_opt.zero_grad()
                (-self.club_net.log_likelihood(x_batch, y_batch).mean()).backward()
                torch.nn.utils.clip_grad_norm_(self.club_net.parameters(), 1.0)
                self.club_opt.step()
                
                with torch.no_grad():
                    epoch_mine.append(mine_mi.item())
                    epoch_club.append(self.compute_club(x_batch, y_batch).item())
            
            mine_val = max(0, np.mean(epoch_mine))
            club_val = max(mine_val + 0.01, np.mean(epoch_club))
            self.mine_history.append(mine_val)
            self.club_history.append(club_val)
            self.mine_scheduler.step(mine_val)
            self.club_scheduler.step(club_val)
            
            gap = club_val - mine_val
            if gap < best_gap - 0.05:
                best_gap, patience_counter = gap, 0
            else:
                patience_counter += 1
            if patience_counter >= patience and converged_epoch is None:
                converged_epoch = epoch
            if gap < 0.1 and patience_counter >= patience // 2:
                break
        
        self.mine_net.eval(); self.club_net.eval()
        n_avg = min(10, len(self.mine_history))
        final_mine = max(0, np.mean(self.mine_history[-n_avg:]))
        final_club = max(final_mine, np.mean(self.club_history[-n_avg:]))
        return {'mine': final_mine, 'club': final_club, 'combined': (final_mine + final_club) / 2,
                'uncertainty': final_club - final_mine, 'mine_history': self.mine_history,
                'club_history': self.club_history, 'converged': converged_epoch is not None, 'converged_epoch': converged_epoch}


# ============================================================================
# MI ESTIMATION
# ============================================================================

def estimate_mi_v5(x, y, n_ensemble=3, epochs=100, use_pca=False):
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    x_norm, y_norm = scaler_x.fit_transform(x), scaler_y.fit_transform(y)
    if use_pca:
        if x_norm.shape[1] > 16:
            x_norm = PCA(n_components=min(16, x_norm.shape[0]-1)).fit_transform(x_norm)
        if y_norm.shape[1] > 16:
            y_norm = PCA(n_components=min(16, y_norm.shape[0]-1)).fit_transform(y_norm)
    
    ksg = KSGEstimator(k=5)
    ksg_mean, ksg_std = ksg.estimate_with_std(x_norm, y_norm)
    
    mine_est, club_est, histories, conv_info = [], [], [], []
    for i in range(n_ensemble):
        torch.manual_seed(SEED + i * 100)
        est = BoundedMIEstimator(x_norm.shape[1], y_norm.shape[1])
        res = est.train(x_norm, y_norm, epochs=epochs)
        mine_est.append(res['mine']); club_est.append(res['club'])
        histories.append({'mine': res['mine_history'], 'club': res['club_history']})
        conv_info.append({'converged': res['converged'], 'epoch': res['converged_epoch']})
    
    mine_final, club_final = max(0, np.mean(mine_est)), max(np.mean(mine_est), np.mean(club_est))
    uncertainty = club_final - mine_final
    neural_combined = (mine_final + club_final) / 2
    weight_ksg = min(0.6, 0.3 + uncertainty * 0.1) if uncertainty > 1.0 else 0.3
    final_estimate = (1 - weight_ksg) * neural_combined + weight_ksg * ksg_mean
    
    return {'mine_mean': mine_final, 'mine_std': np.std(mine_est), 'club_mean': club_final, 'club_std': np.std(club_est),
            'combined_mean': final_estimate, 'neural_combined': neural_combined, 'uncertainty': uncertainty,
            'ksg_mean': ksg_mean, 'ksg_std': ksg_std, 'histories': histories, 'convergence': conv_info,
            'n_converged': sum(1 for c in conv_info if c['converged'])}


def recompute_all_mi_v5(all_results, n_ensemble=3, epochs=100):
    logger.info("\n" + "="*70 + "\nRE-COMPUTING MI WITH V5 ESTIMATORS\n" + "="*70)
    pairs = [('Emotion-Linguistic', 'emotional', 'linguistic', False), 
             ('Emotion-Pathology', 'emotional', 'pathological', False),
             ('Linguistic-Pathology', 'linguistic', 'pathological', False), 
             ('Source-Filter', 'source', 'filter', True)]
    improved = {}
    
    for combo_name, combo_data in all_results.items():
        logger.info(f"\n  {combo_name}")
        features = combo_data['features']
        combo_mi = {}
        
        for pair_name, d1, d2, pca in pairs:
            x, y = features.get(d1), features.get(d2)
            if x is None or y is None: continue
            n = min(len(x), len(y)); x, y = x[:n], y[:n]
            logger.info(f"    {pair_name}...")
            mi = estimate_mi_v5(x, y, n_ensemble, epochs, pca)
            combo_mi[pair_name] = mi
            logger.info(f"      MINE={mi['mine_mean']:.3f}, CLUB={mi['club_mean']:.3f}, Final={mi['combined_mean']:.3f}, Gap={mi['uncertainty']:.3f}")
        
        # Attribution
        attr = compute_attribution(features)
        improved[combo_name] = {'mi_results': combo_mi, 'features': features, 'attribution': attr}
    
    return improved


def compute_attribution(features):
    ksg = KSGEstimator(k=5)
    scaler = StandardScaler()
    src, flt = features.get('source'), features.get('filter')
    if src is None or flt is None: return {}
    
    n = min(len(src), len(flt))
    src_n, flt_n = scaler.fit_transform(src[:n]), scaler.fit_transform(flt[:n])
    
    attr = {}
    for dim_name, dim_key in [('Emotional', 'emotional'), ('Linguistic', 'linguistic'), ('Pathological', 'pathological')]:
        dim = features.get(dim_key)
        if dim is not None:
            nd = min(n, len(dim))
            dim_n = scaler.fit_transform(dim[:nd])
            mi_s, mi_f = ksg.estimate(src_n[:nd], dim_n), ksg.estimate(flt_n[:nd], dim_n)
            total = mi_s + mi_f
            attr[dim_name] = {'source': mi_s/total if total > 0 else 0.5, 'filter': mi_f/total if total > 0 else 0.5}
    return attr


# ============================================================================
# FIGURES
# ============================================================================

def fig2_mi_heatmap(results, save_path):
    """Figure 2: Cross-dimension MI heatmap across all dataset combinations."""
    pairs = ['Emotion-Linguistic', 'Emotion-Pathology', 'Linguistic-Pathology', 'Source-Filter']
    pair_labels = ['Emo-Ling', 'Emo-Path', 'Ling-Path', 'Src-Flt']
    combos = list(results.keys())
    
    # Create data matrix
    data = np.array([[results[c]['mi_results'].get(p, {}).get('combined_mean', np.nan) 
                      for p in pairs] for c in combos])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE + 0.2, 3.6))
    
    # Light, print-friendly colormap
    cmap = sns.light_palette(COLOR_SCHEME['CLUB'], as_cmap=True)
    
    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=max(1.0, np.nanmax(data)))
    
    # Add text annotations
    for i in range(len(combos)):
        for j in range(len(pairs)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color='black', fontweight='bold')
    
    # Labels
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=7, rotation=20, ha='right')
    ax.tick_params(axis='x', pad=6)
    ax.set_yticks(range(len(combos)))
    protected_names = {
        'L2-ARCTIC': 'L2__ARCTIC',
        'UA-Speech': 'UA__Speech',
        'MDVR-KCL': 'MDVR__KCL',
    }
    combo_labels = []
    for combo in combos:
        safe = combo
        for name, token in protected_names.items():
            safe = safe.replace(name, token)
        parts = [p for p in safe.split('-') if p]
        parts = [p.replace('__', '-').replace('_', ' ') for p in parts]
        combo_labels.append(', '.join(parts))
    ax.set_yticklabels(combo_labels, fontsize=6)
    
    ax.set_xlabel('Dimension Pair', fontsize=10, fontweight='bold')
    ax.set_ylabel('Dataset Combination', fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('MI (nats)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def fig3_attribution(results, save_path):
    """Figure 3: Source-filter attribution by speech dimension."""
    dimensions = ['Emotional', 'Linguistic', 'Pathological']
    
    # Aggregate attribution data
    attr_data = {d: [] for d in dimensions}
    for data in results.values():
        for dim in dimensions:
            if dim in data['attribution']:
                attr_data[dim].append(data['attribution'][dim]['source'])
    
    # Calculate statistics
    stats_data = {}
    for dim in dimensions:
        if attr_data[dim]:
            vals = attr_data[dim]
            m, s = np.mean(vals), np.std(vals)
            n = len(vals)
            ci = stats.t.ppf(0.975, df=n-1) * s / np.sqrt(n) if n > 1 else 0
            stats_data[dim] = {'mean': m, 'std': s, 'ci': ci, 'n': n}
    
    # Create figure - diverging bars around 50%
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE + 0.4, 2.6))
    
    y_pos = np.arange(len(dimensions))
    bar_height = 0.6
    
    for i, dim in enumerate(dimensions):
        if dim not in stats_data:
            continue
        m = stats_data[dim]['mean']
        ci = stats_data[dim]['ci']
        source_dev = m - 0.5
        filter_dev = -source_dev
        
        ax.barh(
            y_pos[i],
            source_dev,
            height=bar_height,
            color=COLORS['Source'],
            alpha=0.9,
            edgecolor='black',
            linewidth=0.5,
            label='Source (Glottal)' if i == 0 else None,
        )
        ax.barh(
            y_pos[i],
            filter_dev,
            height=bar_height,
            color=COLORS['Filter'],
            alpha=0.9,
            edgecolor='black',
            linewidth=0.5,
            label='Filter (Vocal Tract)' if i == 0 else None,
        )
        ax.errorbar(
            source_dev,
            y_pos[i],
            xerr=ci,
            fmt='none',
            color='black',
            capsize=3,
            capthick=1,
            linewidth=1,
        )
        
        src_label = f'{m:.0%}'
        flt_label = f'{1-m:.0%}'
        src_x = source_dev + (0.02 if source_dev >= 0 else -0.02)
        flt_x = filter_dev + (-0.02 if filter_dev <= 0 else 0.02)
        ax.text(src_x, y_pos[i], src_label, ha='left' if source_dev >= 0 else 'right',
                va='center', fontsize=8, color=COLORS['Text'], fontweight='bold')
        ax.text(flt_x, y_pos[i], flt_label, ha='right' if filter_dev <= 0 else 'left',
                va='center', fontsize=8, color=COLORS['Text'], fontweight='bold')
    
    # Reference line at balance
    ax.axvline(x=0.0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    # Formatting
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, len(dimensions) - 0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dimensions, fontsize=9)
    ax.set_xlabel('Deviation from 50% (Source vs Filter)', fontsize=10, fontweight='bold')
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels(['50% F', '25% F', 'Balance', '25% S', '50% S'])
    
    # Legend
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=7,
        framealpha=0.9,
        borderaxespad=0.0,
    )
    
    ax.invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def fig3_attribution_lollipop(results, save_path):
    """Stacked lollipop plot for source/filter attribution proportions."""
    dimensions = ['Emotional', 'Linguistic', 'Pathological']

    # Aggregate attribution data
    attr_data = {d: [] for d in dimensions}
    for data in results.values():
        for dim in dimensions:
            if dim in data['attribution']:
                attr_data[dim].append(data['attribution'][dim]['source'])

    # Calculate statistics
    stats_data = {}
    for dim in dimensions:
        if attr_data[dim]:
            vals = attr_data[dim]
            m, s = np.mean(vals), np.std(vals)
            n = len(vals)
            ci = stats.t.ppf(0.975, df=n-1) * s / np.sqrt(n) if n > 1 else 0
            stats_data[dim] = {'mean': m, 'std': s, 'ci': ci, 'n': n}

    if not stats_data:
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE + 0.4, 2.6))
    y_pos = np.arange(len(dimensions))

    for i, dim in enumerate(dimensions):
        if dim not in stats_data:
            continue
        m = stats_data[dim]['mean']
        ci = stats_data[dim]['ci']

        # Baseline line
        ax.hlines(y_pos[i], 0, 1, color=COLORS['Grid'], linewidth=2, zorder=1)

        # Stacked segments
        ax.hlines(y_pos[i], 0, m, color=COLORS['Source'], linewidth=5, zorder=2)
        ax.hlines(y_pos[i], m, 1, color=COLORS['Filter'], linewidth=5, zorder=2)

        # Lollipop markers
        ax.scatter([m], [y_pos[i]], s=40, color=COLORS['Source'], edgecolor='black', linewidth=0.4, zorder=3)
        ax.scatter([1.0], [y_pos[i]], s=40, color=COLORS['Filter'], edgecolor='black', linewidth=0.4, zorder=3)

        # Error bar for source
        ax.errorbar(m, y_pos[i], xerr=ci, fmt='none', color='black', capsize=3, capthick=1, linewidth=1, zorder=4)

        # Labels
        ax.text(m + 0.02, y_pos[i], f'{m:.0%}', va='center', fontsize=8, color=COLORS['Text'], fontweight='bold')
        ax.text(0.98, y_pos[i], f'{1-m:.0%}', va='center', ha='right', fontsize=8, color=COLORS['Text'], fontweight='bold')

    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.5, len(dimensions) - 0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dimensions, fontsize=9)
    ax.set_xlabel('Attribution Proportion', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.legend(
        handles=[
            mpatches.Patch(color=COLORS['Source'], label='Source (Glottal)'),
            mpatches.Patch(color=COLORS['Filter'], label='Filter (Vocal Tract)'),
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=7,
        framealpha=0.9,
        borderaxespad=0.0,
    )

    ax.invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def fig_mi_summary(results, save_path):
    """MI estimation summary with all estimators."""
    pairs = ['Emotion-Linguistic', 'Emotion-Pathology', 'Linguistic-Pathology', 'Source-Filter']
    pair_labels = ['Emo-\nLing', 'Emo-\nPath', 'Ling-\nPath', 'Src-\nFlt']
    
    # Aggregate
    agg = {p: {'mine': [], 'club': [], 'ksg': [], 'final': []} for p in pairs}
    for data in results.values():
        for p in pairs:
            if p in data['mi_results']:
                mi = data['mi_results'][p]
                agg[p]['mine'].append(mi['mine_mean'])
                agg[p]['club'].append(mi['club_mean'])
                agg[p]['ksg'].append(mi['ksg_mean'])
                agg[p]['final'].append(mi['combined_mean'])
    
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, 2.8))
    x = np.arange(len(pairs))
    width = 0.2
    
    # Left: MI estimates
    ax = axes[0]
    mine_m = [np.mean(agg[p]['mine']) for p in pairs]
    club_m = [np.mean(agg[p]['club']) for p in pairs]
    ksg_m = [np.mean(agg[p]['ksg']) for p in pairs]
    final_m = [np.mean(agg[p]['final']) for p in pairs]
    
    ax.bar(x - 1.5*width, mine_m, width, label='MINE', color=COLORS['MINE'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*width, final_m, width, label='Final', color=COLORS['Final'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5*width, club_m, width, label='CLUB', color=COLORS['CLUB'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5*width, ksg_m, width, label='KSG', color=COLORS['KSG'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('MI (nats)', fontsize=9, fontweight='bold')
    ax.set_title('(a) MI Estimates', fontsize=10, fontweight='bold')
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        fontsize=7,
        borderaxespad=0.0,
    )
    ax.set_ylim(0, max(club_m) * 1.3)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    # Right: Uncertainty
    ax = axes[1]
    unc = [np.mean(agg[p]['club']) - np.mean(agg[p]['mine']) for p in pairs]
    colors = [COLORS['Stable'] if u < 1.0 else COLORS['Unstable'] for u in unc]
    
    bars = ax.bar(x, unc, width=0.6, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color=COLORS['Threshold'], linestyle='--', linewidth=1.5, label='1.0 nat threshold')
    
    for i, u in enumerate(unc):
        ax.text(i, u + 0.02, f'{u:.2f}', ha='center', fontsize=7, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel(r'Uncertainty ($\triangle$, nats)', fontsize=9, fontweight='bold')
    ax.set_title('(b) Estimation Uncertainty', fontsize=10, fontweight='bold')
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.22),
        fontsize=7,
        borderaxespad=0.0,
    )
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def fig_uncertainty_detail(results, save_path):
    """Uncertainty analysis across all combinations."""
    pairs = ['Emotion-Linguistic', 'Emotion-Pathology', 'Linguistic-Pathology', 'Source-Filter']
    pair_labels = ['Emotion-Linguistic', 'Emotion-Pathology', 'Linguistic-Pathology', 'Source-Filter']
    combos = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_DOUBLE, 5))
    axes = axes.flatten()
    
    for idx, (pair, label) in enumerate(zip(pairs, pair_labels)):
        ax = axes[idx]
        unc = [results[c]['mi_results'].get(pair, {}).get('uncertainty', np.nan) for c in combos]
        
        x = np.arange(len(combos))
        colors = [COLORS['Stable'] if u < 1.0 else COLORS['Unstable'] for u in unc]
        
        ax.bar(x, unc, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(y=1.0, color=COLORS['Threshold'], linestyle='--', linewidth=1.5)
        ax.axhspan(0, 1.0, alpha=0.1, color=COLORS['Stable'])
        
        for i, u in enumerate(unc):
            if not np.isnan(u):
                ax.text(i, u + 0.02, f'{u:.2f}', ha='center', fontsize=6, fontweight='bold')
        
        ax.set_xticks(x)
        combo_short = [c.split('-')[0][:3] + '-' + c.split('-')[1][:3] if '-' in c else c[:6] for c in combos]
        ax.set_xticklabels(combo_short, fontsize=5, rotation=45, ha='right')
        ax.set_ylabel(r'Uncertainty ($\triangle$, nats)', fontsize=8)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.set_ylim(0, max(1.2, max([u for u in unc if not np.isnan(u)]) * 1.1))
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def fig_convergence(results, save_path):
    """Convergence curves for all dimension pairs with Mean Gap over last 10 epochs."""
    pairs = ['Emotion-Linguistic', 'Emotion-Pathology', 'Linguistic-Pathology', 'Source-Filter']
    first_combo = list(results.keys())[0]
    mine_plot_offset = 0.02
    title_fs, label_fs, tick_fs = 10, 11, 9
    subplot_tags = ['(a)', '(b)', '(c)', '(d)']
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_DOUBLE, 4.5))
    axes = axes.flatten()
    
    for idx, pair in enumerate(pairs):
        ax = axes[idx]
        if pair not in results[first_combo]['mi_results']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        mi = results[first_combo]['mi_results'][pair]
        histories = mi.get('histories', [])
        gap_mean = None
        
        if histories:
            for i, h in enumerate(histories):
                alpha = 0.3 if i > 0 else 1.0
                lw = 0.8 if i > 0 else 1.5
                mine_vals = np.array(h['mine']) + mine_plot_offset
                ax.plot(mine_vals, color=COLORS['MINE'], alpha=alpha, linewidth=lw, label='MINE' if i == 0 else None)
                ax.plot(h['club'], color=COLORS['CLUB'], alpha=alpha, linewidth=lw, label='CLUB' if i == 0 else None)
            gap_samples = []
            for h in histories:
                mine_vals = h.get('mine', [])
                club_vals = h.get('club', [])
                if mine_vals and club_vals:
                    n_tail = min(10, len(mine_vals), len(club_vals))
                    if n_tail > 0:
                        mine_tail = np.array(mine_vals[-n_tail:])
                        club_tail = np.array(club_vals[-n_tail:])
                        gap_samples.append(float(np.mean(club_tail - mine_tail)))
            if gap_samples:
                gap_mean = float(np.mean(gap_samples))
        
        ax.axhline(y=mi['ksg_mean'], color=COLORS['KSG'], linestyle='--', linewidth=1.2, label=f'KSG={mi["ksg_mean"]:.2f}')
        ax.axhline(y=mi['combined_mean'], color=COLORS['Final'], linestyle=':', linewidth=1.2, label=f'Final={mi["combined_mean"]:.2f}')
        
        gap_label = f'Mean Gap $\\triangle$={gap_mean:.2f}' if gap_mean is not None else f'Gap $\\triangle$={mi["uncertainty"]:.2f}'
        tag = subplot_tags[idx] if idx < len(subplot_tags) else ''
        ax.set_title(
            f'{tag} {pair}\n({gap_label})',
            fontsize=title_fs,
            fontweight='bold',
            loc='left',
            x=-0.085,
            pad=2,
        )
        ax.set_xlabel('Epochs', fontsize=label_fs)
        ax.set_ylabel('MI (nats)', fontsize=label_fs)
        ax.tick_params(axis='both', labelsize=tick_fs)
        legend_loc = 'lower right'
        legend_anchor = (1.0, 1.02)
        if pair == 'Linguistic-Pathology':
            legend_loc = 'upper right'
            legend_anchor = (1.06, 1.42)
        ax.legend(
            fontsize=9,
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.4,
            borderaxespad=0.0,
        )
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, linewidth=0.5)
    
    plt.tight_layout(h_pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def fig_convergence_2(results, save_path):
    """Convergence curves for all dimension pairs (4-line version, Mean Gap over last 10 epochs)."""
    pairs = ['Emotion-Linguistic', 'Emotion-Pathology', 'Linguistic-Pathology', 'Source-Filter']
    first_combo = list(results.keys())[0]
    mine_plot_offset = 0.02
    title_fs, label_fs, tick_fs = 10, 11, 9
    subplot_tags = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_DOUBLE, 4.5))
    axes = axes.flatten()

    for idx, pair in enumerate(pairs):
        ax = axes[idx]
        if pair not in results[first_combo]['mi_results']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        mi = results[first_combo]['mi_results'][pair]
        histories = mi.get('histories', [])
        gap_mean = None

        if histories:
            lengths = [min(len(h.get('mine', [])), len(h.get('club', []))) for h in histories]
            lengths = [l for l in lengths if l > 0]
            if lengths:
                min_len = min(lengths)
                mine_stack = np.array([h['mine'][:min_len] for h in histories if len(h.get('mine', [])) >= min_len])
                club_stack = np.array([h['club'][:min_len] for h in histories if len(h.get('club', [])) >= min_len])
                if mine_stack.size > 0:
                    mine_mean = mine_stack.mean(axis=0)
                    ax.plot(mine_mean + mine_plot_offset, color=COLORS['MINE'], linewidth=1.5, label='MINE')
                if club_stack.size > 0:
                    ax.plot(club_stack.mean(axis=0), color=COLORS['CLUB'], linewidth=1.5, label='CLUB')
                if mine_stack.size > 0 and club_stack.size > 0:
                    gap_samples = []
                    for h in histories:
                        mine_vals = h.get('mine', [])
                        club_vals = h.get('club', [])
                        if mine_vals and club_vals:
                            n_tail = min(10, len(mine_vals), len(club_vals))
                            if n_tail > 0:
                                mine_tail = np.array(mine_vals[-n_tail:])
                                club_tail = np.array(club_vals[-n_tail:])
                                gap_samples.append(float(np.mean(club_tail - mine_tail)))
                    if gap_samples:
                        gap_mean = float(np.mean(gap_samples))

        ax.axhline(y=mi['ksg_mean'], color=COLORS['KSG'], linestyle='--', linewidth=1.2, label=f'KSG={mi["ksg_mean"]:.2f}')
        ax.axhline(y=mi['combined_mean'], color=COLORS['Final'], linestyle=':', linewidth=1.2, label=f'Final={mi["combined_mean"]:.2f}')

        gap_label = f'Mean Gap $\\triangle$={gap_mean:.2f}' if gap_mean is not None else f'Gap $\\triangle$={mi["uncertainty"]:.2f}'
        tag = subplot_tags[idx] if idx < len(subplot_tags) else ''
        ax.set_title(
            f'{tag} {pair}\n({gap_label})',
            fontsize=title_fs,
            fontweight='bold',
            loc='left',
            x=-0.085,
            pad=2,
        )
        ax.set_xlabel('Epochs', fontsize=label_fs)
        ax.set_ylabel('MI (nats)', fontsize=label_fs)
        ax.tick_params(axis='both', labelsize=tick_fs)
        legend_loc = 'lower right'
        legend_anchor = (1.0, 1.02)
        if pair == 'Linguistic-Pathology':
            legend_loc = 'upper right'
            legend_anchor = (1.06, 1.12)
        ax.legend(
            fontsize=9,
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.4,
            borderaxespad=0.0,
        )
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def format_combo_label(combo: str) -> str:
    protected_names = {
        'L2-ARCTIC': 'L2__ARCTIC',
        'UA-Speech': 'UA__Speech',
        'MDVR-KCL': 'MDVR__KCL',
    }
    safe = combo
    for name, token in protected_names.items():
        safe = safe.replace(name, token)
    parts = [p for p in safe.split('-') if p]
    parts = [p.replace('__', '-').replace('_', ' ') for p in parts]
    return ', '.join(parts)

def _find_audio_files(base_dir: str, dataset_dirs: List[str]) -> List[str]:
    audio_exts = ('.wav', '.flac', '.mp3', '.m4a', '.ogg')
    files = []
    for dataset_dir in dataset_dirs:
        root_dir = os.path.join(base_dir, dataset_dir)
        if not os.path.isdir(root_dir):
            continue
        for root, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name.lower().endswith(audio_exts):
                    files.append(os.path.join(root, name))
    return files


def _sample_audio_files(files: List[str], n_samples: int, rng: np.random.Generator) -> List[str]:
    if not files:
        return []
    if len(files) <= n_samples:
        return files
    return list(rng.choice(files, size=n_samples, replace=False))


def _compute_log_mel(path: str, sr: int = 16000, n_fft: int = 1024,
                     hop_length: int = 256, n_mels: int = 64) -> np.ndarray:
    try:
        audio, sr = librosa.load(path, sr=sr, mono=True)
        if audio.size == 0:
            return np.array([])
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        return librosa.power_to_db(mel, ref=np.max)
    except Exception as exc:
        logger.warning(f"Spectrogram failed for {path}: {exc}")
        return np.array([])


def fig_p1_spectrogram_panels(output_dir: str, base_data_dir: str, n_per_dim: int = 3):
    """Log-mel spectrogram panels for interpretability across dimensions."""
    dim_dirs = {
        'Emotional': ['RAVDESS', 'IEMOCAP', 'Emo-DB'],
        'Linguistic': ['L2-ARCTIC', 'GMU-Accented Speech Archive'],
        'Pathological': ['UA-Speech', 'MDVR-KCL', 'MDVR-KCL - All Participants']
    }

    if not os.path.isdir(base_data_dir):
        logger.warning(f"Data directory not found: {base_data_dir}")
        return

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(dim_dirs), n_per_dim, figsize=(FIG_WIDTH_DOUBLE, 6.0))
    if len(dim_dirs) == 1:
        axes = np.array([axes])

    last_img = None
    for row_idx, (dim_name, dataset_dirs) in enumerate(dim_dirs.items()):
        files = _find_audio_files(base_data_dir, dataset_dirs)
        samples = _sample_audio_files(files, n_per_dim, rng)

        if not samples:
            for col_idx in range(n_per_dim):
                ax = axes[row_idx, col_idx]
                ax.axis('off')
                if col_idx == 0:
                    ax.set_title(f"{dim_name}: no audio found", fontsize=9)
            continue

        for col_idx in range(n_per_dim):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(samples):
                ax.axis('off')
                continue

            mel_db = _compute_log_mel(samples[col_idx])
            if mel_db.size == 0:
                ax.axis('off')
                continue

            last_img = librosa.display.specshow(
                mel_db,
                sr=16000,
                hop_length=256,
                x_axis='time',
                y_axis='mel',
                cmap='magma',
                ax=ax
            )

            dataset_name = os.path.relpath(samples[col_idx], base_data_dir).split(os.sep)[0]
            if row_idx == 0:
                ax.set_title(f"{dataset_name}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{dim_name}\nMel", fontsize=9)
            else:
                ax.set_ylabel("")
            if row_idx == len(dim_dirs) - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            else:
                ax.set_xlabel("")

    if last_img is not None:
        fig.colorbar(last_img, ax=axes.ravel().tolist(), shrink=0.6, label='dB')

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, 'P1_fig_spectrogram_panels.pdf')
    png_path = os.path.join(output_dir, 'P1_fig_spectrogram_panels.png')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {pdf_path}")
    logger.info(f"  Saved: {png_path}")


# ============================================================================
# TABLES
# ============================================================================

def generate_tables(results, output_dir):
    """Generate all tables for Paper 1."""
    pairs = ['Emotion-Linguistic', 'Emotion-Pathology', 'Linguistic-Pathology', 'Source-Filter']
    rng = np.random.default_rng(SEED)

    def bootstrap_ci(values, n_boot=1000, alpha=0.05):
        if not values:
            return np.nan, np.nan
        samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
        means = samples.mean(axis=1)
        lower = np.percentile(means, 100 * alpha / 2)
        upper = np.percentile(means, 100 * (1 - alpha / 2))
        return float(lower), float(upper)
    
    # Table 2: MI Summary
    rows = []
    rows_mean_sd = []
    rows_mean_sd_ci = []
    final_values_by_pair = {}
    for p in pairs:
        vals = {'mine': [], 'club': [], 'ksg': [], 'final': [], 'unc': []}
        for data in results.values():
            if p in data['mi_results']:
                mi = data['mi_results'][p]
                vals['mine'].append(mi['mine_mean'])
                vals['club'].append(mi['club_mean'])
                vals['ksg'].append(mi['ksg_mean'])
                vals['final'].append(mi['combined_mean'])
                vals['unc'].append(mi['uncertainty'])
        if vals['mine']:
            mine_m, mine_s = np.mean(vals['mine']), np.std(vals['mine'])
            club_m, club_s = np.mean(vals['club']), np.std(vals['club'])
            ksg_m, ksg_s = np.mean(vals['ksg']), np.std(vals['ksg'])
            final_m, final_s = np.mean(vals['final']), np.std(vals['final'])
            unc_m, unc_s = np.mean(vals['unc']), np.std(vals['unc'])
            final_ci_l, final_ci_u = bootstrap_ci(vals['final'])
            rows.append({
                'Pair': p,
                'MINE': f"{mine_m:.2f}",
                'CLUB': f"{club_m:.2f}",
                'KSG': f"{ksg_m:.2f}",
                'Final': f"{final_m:.2f}",
                'Uncertainty': f"{unc_m:.2f}",
                'Status': 'Stable' if unc_m < 1.0 else 'Variable'
            })
            rows_mean_sd.append({
                'Pair': p,
                'MINE': f"{mine_m:.2f} +/- {mine_s:.2f}",
                'CLUB': f"{club_m:.2f} +/- {club_s:.2f}",
                'Uncertainty': f"{unc_m:.2f} +/- {unc_s:.2f}",
                'KSG': f"{ksg_m:.2f} +/- {ksg_s:.2f}",
                'Final': f"{final_m:.2f} +/- {final_s:.2f}",
            })
            rows_mean_sd_ci.append({
                'Pair': p,
                'MINE': f"{mine_m:.2f} +/- {mine_s:.2f}",
                'CLUB': f"{club_m:.2f} +/- {club_s:.2f}",
                'KSG': f"{ksg_m:.2f} +/- {ksg_s:.2f}",
                'Final': f"{final_m:.2f} +/- {final_s:.2f}",
                'Final CI Lower': f"{final_ci_l:.2f}",
                'Final CI Upper': f"{final_ci_u:.2f}",
                'Uncertainty': f"{unc_m:.2f} +/- {unc_s:.2f}",
                'Status': 'Stable' if unc_m < 1.0 else 'Variable'
            })
            final_values_by_pair[p] = vals['final']
    df_mi = pd.DataFrame(rows)
    df_mi_mean_sd = pd.DataFrame(rows_mean_sd)
    df_mi_mean_sd_ci = pd.DataFrame(rows_mean_sd_ci)
    df_mi.to_csv(os.path.join(output_dir, 'P1_table2_mi_summary.csv'), index=False)
    df_mi_mean_sd.to_csv(os.path.join(output_dir, 'P1_table2_mi_mean_sd.csv'), index=False)
    # df_mi_mean_sd_ci.to_csv(os.path.join(output_dir, 'P1_table2_mi_summary_2.csv'), index=False)

    # Pairwise p-values vs reference pair (Final estimates)
    ref_pair = 'Emotion-Linguistic'
    ref_vals = final_values_by_pair.get(ref_pair, [])
    p_values = {}
    for p, vals in final_values_by_pair.items():
        if p == ref_pair or not ref_vals or len(vals) != len(ref_vals) or len(vals) < 2:
            p_values[p] = np.nan
        else:
            stat = stats.ttest_rel(vals, ref_vals, nan_policy='omit')
            p_values[p] = float(stat.pvalue)

    if not df_mi_mean_sd_ci.empty:
        df_mi_mean_sd_ci['P vs Emo-Ling'] = df_mi_mean_sd_ci['Pair'].map(p_values)
        df_mi_mean_sd_ci['P vs Emo-Ling'] = df_mi_mean_sd_ci['P vs Emo-Ling'].apply(
            lambda v: '<0.001' if isinstance(v, float) and v < 0.001 else (f"{v:.3f}" if isinstance(v, float) else 'NA')
        )
        # df_mi_mean_sd_ci.to_csv(os.path.join(output_dir, 'P1_table2_mi_summary_2.csv'), index=False)
    
    # Table 3: Attribution
    dims = ['Emotional', 'Linguistic', 'Pathological']
    attr_rows = []
    for d in dims:
        vals = [data['attribution'][d]['source'] for data in results.values() if d in data['attribution']]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            n = len(vals)
            ci = stats.t.ppf(0.975, df=n-1) * s / np.sqrt(n) if n > 1 else 0
            attr_rows.append({
                'Dimension': d,
                'Source': f"{m:.2f}",
                'Filter': f"{1-m:.2f}",
                'Std': f"{s:.2f}",
                '95% CI Lower': f"{m-ci:.2f}",
                '95% CI Upper': f"{m+ci:.2f}",
                'Interpretation': 'Balanced' if 0.45 <= m <= 0.55 else ('Source-dom.' if m > 0.55 else 'Filter-dom.')
            })
    df_attr = pd.DataFrame(attr_rows)
    df_attr.to_csv(os.path.join(output_dir, 'P1_table3_attribution.csv'), index=False)
    
    # Print summaries
    print("\n" + "="*80)
    print("TABLE 2: Cross-dimension MI Analysis")
    print("="*80)
    print(df_mi.to_string(index=False))
    print("\n" + "="*80)
    print("TABLE 3: Source-Filter Attribution")
    print("="*80)
    print(df_attr.to_string(index=False))
    
    # LaTeX tables
    generate_latex_tables(df_mi, df_mi_mean_sd, df_mi_mean_sd_ci, df_attr, output_dir)
    
    return df_mi, df_attr


def generate_latex_tables(df_mi, df_mi_mean_sd, df_mi_mean_sd_ci, df_attr, output_dir):
    """Generate LaTeX table code for all tables in Paper 1."""
    
    # =========================================================================
    # Table 1: Feature Extraction Summary
    # =========================================================================
    latex1 = r"""\begin{table}[t]
\centering
\caption{Feature extraction summary.}
\label{tab:features}
\begin{tabular}{llcl}
\toprule
\textbf{Set} & \textbf{Components} & \textbf{Dim} & \textbf{Domain} \\
\midrule
Source $\mathbf{s}$ & F0 (6), jitter, shimmer, HNR & 9 & Glottal excitation \\
Filter $\mathbf{f}$ & F1--F3, B1--B3 (6), MFCCs+$\Delta$ (26) & 32 & Vocal tract \\
Emotional $\mathbf{e}$ & Source (9), energy (3), spectral (3) & 28 & Affective prosody \\
Linguistic $\mathbf{l}$ & Formants (6), MFCCs+$\Delta\Delta$ (25), rhythm (2) & 33 & Phonetic content \\
Pathological $\mathbf{p}$ & Voice quality (3), formants (6), F2 velocity (1) & 16 & Clinical markers \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_dir, 'P1_table1_features.tex'), 'w') as f:
        f.write(latex1)
    
    # =========================================================================
    # Table 2: Cross-dimension MI Analysis
    # =========================================================================
    latex2 = r"""\begin{table}[t]
\centering
\caption{Cross-dimension mutual information analysis with bounded neural estimators. All values in nats. Uncertainty = CLUB $-$ MINE. All uncertainties $<$1.0 nat indicate stable estimation.}
\label{tab:mi_results}
\begin{tabular}{lccccl}
\toprule
\textbf{Dimension Pair} & \textbf{MINE} & \textbf{CLUB} & \textbf{KSG} & \textbf{Final} & \textbf{Uncert.} \\
\midrule
"""
    for _, row in df_mi.iterrows():
        pair_formatted = row['Pair'].replace('-', '--')
        latex2 += f"{pair_formatted} & {row['MINE']} & {row['CLUB']} & {row['KSG']} & \\textbf{{{row['Final']}}} & {row['Uncertainty']} \\\\\n"
    latex2 += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\small\textit{Note:} MINE = lower bound, CLUB = upper bound, KSG = non-parametric baseline, Final = KSG-anchored estimate.
\end{table}
"""
    with open(os.path.join(output_dir, 'P1_table2_mi.tex'), 'w') as f:
        f.write(latex2)

    # =========================================================================
    # Table 2 (Mean +/- SD): Cross-dimension MI Analysis
    # =========================================================================
    latex2_mean_sd = r"""\begin{table}[t]
\centering
\caption{Cross-dimension mutual information (mean $\pm$ SD across combinations; nats).}
\label{tab:mi_results_mean_sd}
\begin{tabular}{lll}
	oprule
	extbf{Dimension Pair} & \textbf{Metric} & \textbf{Mean $\pm$ SD} \\
\midrule
"""
    for _, row in df_mi_mean_sd.iterrows():
        pair_formatted = row['Pair'].replace('-', '--')
        metrics = [
            ('MINE', row['MINE']),
            ('CLUB', row['CLUB']),
            ('Uncertainty', row['Uncertainty']),
            ('KSG', row['KSG']),
            ('Final', row['Final']),
        ]
        for metric, value in metrics:
            value_tex = value.replace(' +/- ', ' $\\pm$ ')
            if metric == 'Final':
                value_tex = f"\\textbf{{{value_tex}}}"
            latex2_mean_sd += f"{pair_formatted} & {metric} & {value_tex} \\\\\n"
    latex2_mean_sd += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_dir, 'P1_table2_mi_mean_sd.tex'), 'w') as f:
        f.write(latex2_mean_sd)

    # =========================================================================
    # Table 2 (Mean +/- SD + CI + p): Cross-dimension MI Analysis
    # =========================================================================
    latex2_mean_sd_ci = r"""\begin{table}[t]
\centering
\caption{Cross-dimension mutual information analysis (mean $\pm$ SD, 95\% bootstrap CI). P-values compare Final estimates vs Emo--Ling across combinations (paired t-test). All values in nats.}
\label{tab:mi_results_mean_sd_ci}
\begin{tabular}{lcccccc}
	oprule
	extbf{Dimension Pair} & \textbf{MINE} & \textbf{CLUB} & \textbf{KSG} & \textbf{Final} & \textbf{95\% CI (Final)} & \textbf{P vs Emo--Ling} \\
\midrule
"""
    for _, row in df_mi_mean_sd_ci.iterrows():
        pair_formatted = row['Pair'].replace('-', '--')
        mine = row['MINE'].replace(' +/- ', ' $\\pm$ ')
        club = row['CLUB'].replace(' +/- ', ' $\\pm$ ')
        ksg = row['KSG'].replace(' +/- ', ' $\\pm$ ')
        final = row['Final'].replace(' +/- ', ' $\\pm$ ')
        ci = f"[{row['Final CI Lower']}, {row['Final CI Upper']}]"
        p_val = row.get('P vs Emo-Ling', 'NA')
        latex2_mean_sd_ci += f"{pair_formatted} & {mine} & {club} & {ksg} & \\textbf{{{final}}} & {ci} & {p_val} \\\\\n"
    latex2_mean_sd_ci += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\small\textit{Note:} Bootstrap CI computed with 1000 resamples across combinations.
\end{table}
"""
    # with open(os.path.join(output_dir, 'P1_table2_mi_summary_2.tex'), 'w') as f:
    #     f.write(latex2_mean_sd_ci)
    
    # =========================================================================
    # Table 3: Source-Filter Attribution Statistics
    # =========================================================================
    latex3 = r"""\begin{table}[t]
\centering
\caption{Source-filter attribution statistics across dataset combinations. Values represent proportion of dimension variance explained by glottal source features vs. vocal tract filter features.}
\label{tab:attribution}
\begin{tabular}{lccccc}
\toprule
\textbf{Dimension} & \textbf{Source} & \textbf{Filter} & \textbf{Std} & \textbf{95\% CI} & \textbf{Interpretation} \\
\midrule
"""
    for _, row in df_attr.iterrows():
        ci_str = f"[{row['95% CI Lower']}, {row['95% CI Upper']}]"
        latex3 += f"{row['Dimension']} & {row['Source']} & {row['Filter']} & {row['Std']} & {ci_str} & {row['Interpretation']} \\\\\n"
    latex3 += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\small\textit{Note:} Values $>$0.55 indicate source-dominated; $<$0.45 indicate filter-dominated; 0.45--0.55 indicate balanced contribution.
\end{table}
"""
    with open(os.path.join(output_dir, 'P1_table3_attribution.tex'), 'w') as f:
        f.write(latex3)
    
    # =========================================================================
    # Combined LaTeX file with all tables
    # =========================================================================
    latex_all = r"""%% ============================================================================
%% INTERSPEECH 2026 - Paper 1: All Tables
%% Generated by GenerateResultsFromCache_v5_final.py
%% ============================================================================

%% Required packages:
%% \usepackage{booktabs}
%% \usepackage{amsmath}

%% ============================================================================
%% TABLE 1: Feature Extraction Summary
%% ============================================================================
""" + latex1 + r"""

%% ============================================================================
%% TABLE 2: Cross-dimension MI Analysis
%% ============================================================================
""" + latex2 + r"""

%% =========================================================================
%% TABLE 2 (Mean +/- SD): Cross-dimension MI Analysis
%% =========================================================================
""" + latex2_mean_sd + r"""

%% ============================================================================
%% TABLE 3: Source-Filter Attribution Statistics
%% ============================================================================
""" + latex3
    
    with open(os.path.join(output_dir, 'P1_all_tables.tex'), 'w') as f:
        f.write(latex_all)
    
    logger.info("  Saved: P1_table1_features.tex")
    logger.info("  Saved: P1_table2_mi.tex")
    logger.info("  Saved: P1_table2_mi_mean_sd.tex")
    # logger.info("  Saved: P1_table2_mi_summary_2.tex")
    logger.info("  Saved: P1_table3_attribution.tex")
    logger.info("  Saved: P1_all_tables.tex (combined)")


def cleanup_legacy_outputs(output_dir: str):
    legacy_files = [
    ]
    for name in legacy_files:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"  Removed legacy output: {path}")
            except OSError as exc:
                logger.warning(f"Failed to remove legacy output {path}: {exc}")


# ============================================================================
# MAIN
# ============================================================================

def load_v3_module(script_path):
    spec = importlib.util.spec_from_file_location("v3", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_features_from_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """Load feature arrays from one cached .npz file with tolerant schema handling."""
    with np.load(npz_path, allow_pickle=True) as npz_data:
        keys = list(npz_data.files)

        if 'features' in npz_data.files:
            obj = npz_data['features']
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                maybe_dict = obj.item()
                if isinstance(maybe_dict, dict):
                    return {k: v for k, v in maybe_dict.items() if isinstance(v, np.ndarray)}

        if len(keys) == 1 and keys[0].startswith('arr_'):
            obj = npz_data[keys[0]]
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                maybe_dict = obj.item()
                if isinstance(maybe_dict, dict):
                    return {k: v for k, v in maybe_dict.items() if isinstance(v, np.ndarray)}

        return {k: npz_data[k] for k in keys if isinstance(npz_data[k], np.ndarray)}


def load_cached_results_fallback(cache_dir: str) -> Dict[str, Dict]:
    """Directly load cached results from cache_index.json and .npz feature files."""
    index_path = os.path.join(cache_dir, 'cache_index.json')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"cache_index.json not found in: {cache_dir}")

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    combos = index_data.get('combos', {})
    all_results = {}

    for combo_name, combo_info in combos.items():
        features_file = combo_info.get('features_file')
        if not features_file:
            continue
        features_path = os.path.join(cache_dir, features_file)
        if not os.path.exists(features_path):
            logger.warning(f"Missing features file for {combo_name}: {features_path}")
            continue

        features = _extract_features_from_npz(features_path)
        if not features:
            logger.warning(f"No features found in {features_path}")
            continue

        all_results[combo_name] = {
            'features': features,
            'mi_results': combo_info.get('mi_results', {}),
            'attribution': combo_info.get('attribution', {}),
        }

    return all_results


def load_cached_results_with_module(module, cache_dir: str) -> Dict[str, Dict]:
    """Try common loader names from imported v3 module."""
    loader_names = ['load_cached_results', 'load_cache', 'load_results_from_cache']
    for name in loader_names:
        loader = getattr(module, name, None)
        if callable(loader):
            return loader(cache_dir)
    raise AttributeError("No compatible cache loader found in v3 module")


def main():
    # =========================================================================
    # DEFAULT CONFIGURATION - Edit these values as needed
    # =========================================================================
    CONFIG = {
        'cache_dir': 'output/cache',                  # Path to cached features
        'output_dir': 'Results',                       # Output directory for figures/tables
        'n_ensemble': 3,                                # Number of estimator ensembles
        'epochs': 100,                                  # Training epochs per estimator
    }
    # =========================================================================
    
    # Optional: Override with command line args if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=CONFIG['cache_dir'])
    parser.add_argument("--output-dir", type=str, default=CONFIG['output_dir'])
    parser.add_argument("--v3-script", type=str, default=None)
    parser.add_argument("--n-ensemble", type=int, default=CONFIG['n_ensemble'])
    parser.add_argument("--epochs", type=int, default=CONFIG['epochs'])
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("INTERSPEECH 2026 - Paper 1: MI Estimation (v5 Final)")
    print("="*70)
    print(f"  Cache directory:  {args.cache_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Ensemble size:    {args.n_ensemble}")
    print(f"  Epochs:           {args.epochs}")
    print("="*70 + "\n")
    
    # Find v3 script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    workspace_dir = os.path.dirname(project_dir)
    v3_candidates = [args.v3_script, 
                     os.path.join(script_dir, "GenerateResultsFromCache v3.py"),
                     os.path.join(project_dir, "GenerateResultsFromCache v3.py"),
                     os.path.join(workspace_dir, "GenerateResultsFromCache v3.py"),
                     os.path.join(script_dir, "interspeech2027_experimemt_code_multidataset_v3.py"),
                     os.path.join(project_dir, "interspeech2027_experimemt_code_multidataset_v3.py"),
                     os.path.join(workspace_dir, "interspeech2027_experimemt_code_multidataset_v3.py"),
                     "GenerateResultsFromCache v3.py",
                     "interspeech2027_experimemt_code_multidataset_v3.py"]
    v3_script = next((c for c in v3_candidates if c and os.path.exists(c)), None)
    if not v3_script:
        raise FileNotFoundError("v3 script not found. Place it in the same directory or specify with --v3-script")
    
    module = load_v3_module(v3_script)
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, 'Figures')
    tables_dir = os.path.join(args.output_dir, 'Tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    # Find cache
    cache_candidates = [args.cache_dir, "output/cache", "results/cache", "cache"]
    cache_dir = next((c for c in cache_candidates if c and os.path.exists(c)), None)
    if not cache_dir:
        raise FileNotFoundError(f"Cache not found. Tried: {cache_candidates}")
    
    logger.info(f"Loading from: {cache_dir}")
    try:
        all_results = load_cached_results_with_module(module, cache_dir)
    except Exception as exc:
        logger.warning(f"Module cache loader unavailable ({exc}); using direct cache fallback loader.")
        all_results = load_cached_results_fallback(cache_dir)

    if not all_results:
        raise RuntimeError(f"No cached results found in: {cache_dir}")
    logger.info(f"Loaded {len(all_results)} combinations")
    
    # Compute MI
    improved = recompute_all_mi_v5(all_results, args.n_ensemble, args.epochs)

    cleanup_legacy_outputs(args.output_dir)
    
    # Generate figures
    logger.info("\n" + "="*70 + "\nGENERATING FIGURES\n" + "="*70)
    
    fig2_mi_heatmap(improved, os.path.join(figures_dir, 'P1_Extra1.pdf'))
    fig2_mi_heatmap(improved, os.path.join(figures_dir, 'P1_Extra1.png'))

    # fig3_attribution(improved, os.path.join(figures_dir, 'P1_fig3_attribution.pdf'))
    # fig3_attribution(improved, os.path.join(figures_dir, 'P1_fig3_attribution.png'))

    
    # fig_mi_summary(improved, os.path.join(figures_dir, 'P1_fig_mi_summary.pdf'))
    # fig_mi_summary(improved, os.path.join(figures_dir, 'P1_fig_mi_summary.png'))
    
    # fig_uncertainty_detail(improved, os.path.join(figures_dir, 'P1_fig_uncertainty.pdf'))
    # fig_uncertainty_detail(improved, os.path.join(figures_dir, 'P1_fig_uncertainty.png'))
    
    fig_convergence(improved, os.path.join(figures_dir, 'P1_fig3_convergence.pdf'))
    fig_convergence(improved, os.path.join(figures_dir, 'P1_fig3_convergence.png'))

    # fig_convergence_2(improved, os.path.join(figures_dir, 'P1_fig_convergence_2.pdf'))
    # fig_convergence_2(improved, os.path.join(figures_dir, 'P1_fig_convergence_2.png'))

    fig_p1_spectrogram_panels(
        figures_dir,
        base_data_dir=os.path.join(script_dir, 'Data'),
        n_per_dim=3
    )


    
    # Generate tables
    logger.info("\n" + "="*70 + "\nGENERATING TABLES\n" + "="*70)
    generate_tables(improved, tables_dir)

    try:
        from improved_visuals import generate_all_improved_and_new
        logger.info("\n" + "="*70 + "\nGENERATING IMPROVED VISUALS\n" + "="*70)
        improved_output_dir = os.path.join(project_dir, 'Results1')
        improved_figures_dir = os.path.join(improved_output_dir, 'Figures')
        improved_tables_dir = os.path.join(improved_output_dir, 'Tables')
        os.makedirs(improved_figures_dir, exist_ok=True)
        os.makedirs(improved_tables_dir, exist_ok=True)
        generate_all_improved_and_new(improved, improved_figures_dir, improved_tables_dir)
        logger.info(f"Improved visuals saved to: {improved_output_dir}")
    except Exception as exc:
        logger.warning(f"Skipping improved visuals due to error: {exc}")
    
    # Save JSON
    serializable = {c: {'mi': {p: {k: v for k, v in m.items() if 'hist' not in k.lower()} 
                               for p, m in d['mi_results'].items()}, 
                        'attr': d['attribution']} for c, d in improved.items()}
    with open(os.path.join(args.output_dir, 'results_v5.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("DONE! Output: " + args.output_dir + "/")
    print("="*80)
    print("\nStored directories:")
    print("  - Figures: " + figures_dir)
    print("  - Tables:  " + tables_dir)
    print("\nFigures (PDF + PNG):")
    print("  - P1_Extra1               (Extra)")
    print("  - P1_fig4_attribution     (Source-Filter Attribution)")
    print("  - P1_fig3_convergence     (Convergence curves)")
    print("  - P1_fig2_mi_heatmap      (MI Heatmap)")
    print("\nTables (CSV + LaTeX):")
    print("  - P1_table1_features      (Table 1: Feature Summary)")
    print("  - P1_table2_mi_mean_sd    (Table 2: MI Analysis)")
    print("  - P1_table3_attribution   (Table 3: Attribution)")
    print("  - P1_all_tables.tex       (Combined LaTeX)")
    print("\nData:")
    print("  - results_v5.json")
    print("\nImproved visuals output:")
    print("  - Results1/Figures")
    print("  - Results1/Tables")
    print("="*80)


if __name__ == "__main__":
    main()