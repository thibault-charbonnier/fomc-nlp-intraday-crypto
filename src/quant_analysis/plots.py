import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy import stats
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns


def qq_plot(
        distrib_A: pd.Series,
        distrib_B: pd.Series,
        q_levels: List[float] = np.linspace(0, 100, 100)
    ):
    """
    Generate a Q-Q plot to compare two distributions.

    Params:
        distrib_A: pd.Series
            First distribution of returns.
        distrib_B: pd.Series
            Second distribution of returns.
    """
    quantiles_A = np.percentile(distrib_A, q_levels)
    quantiles_B = np.percentile(distrib_B, q_levels)

    plt.figure(figsize=(8, 8))
    plt.scatter(quantiles_A, quantiles_B, color="#0F766E")
    max_val = max(max(quantiles_A), max(quantiles_B))
    min_val = min(min(quantiles_A), min(quantiles_B))
    plt.plot([min_val, max_val], [min_val, max_val], color="#0B5AA2", linestyle='--')
    plt.xlabel('Quantiles of Distribution A')
    plt.ylabel('Quantiles of Distribution B')
    plt.title('Q-Q Plot')
    plt.grid(True)
    plt.show()

def q_shift_plot(
        q_levels: List[float],
        q_diff: pd.Series,
        q_diff_ci: Dict[float, tuple]
    ):
    """
    Plot the quantile shift between two distributions with confidence intervals.

    Params:
        q_levels: List[float]
            List of quantile levels.
        q_diff: pd.Series
            Differences in quantiles between the two distributions.
        q_diff_ci: Dict[float, tuple]
            Confidence intervals for the quantile differences.
    """
    ci_lower = q_diff_ci["lower"]
    ci_upper = q_diff_ci["upper"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        q_levels, q_diff,
        color="#0F766E", linewidth=2,
        marker="o", markersize=6,
        markerfacecolor="#0F766E", markeredgecolor="white", markeredgewidth=0.8,
        label=r"$\Delta Q_\tau$"
    )
    plt.fill_between(q_levels, ci_lower, ci_upper, color="#9ecae1", alpha=0.5, label='Confidence Interval')
    plt.axhline(0, color="black", linestyle="--", linewidth=1)    
    plt.xlabel(r"Quantile $\tau$")
    plt.ylabel(r"$\Delta Q_\tau$")
    plt.title('Quantile Shift Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_buckets_distrib(
        df: pd.DataFrame,
        returns_col: str = "BTCUSDT_20m",
        buckets_col: str = "tone_bucket",
        bins: int = 20
    ):
    """
    Plot distributions of returns for different tone buckets and KDEs.

    Params:
        df: pd.DataFrame
            DataFrame containing tone buckets and return columns.
        returns_col: str
            Column name for returns.
        buckets_col: str
            Column name for tone buckets.
        bins: int
            Number of bins for the histogram.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    df = df[[returns_col, buckets_col]].copy()
    df = df.dropna()
    
    all_vals = df[returns_col].to_numpy()
    xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    x_grid = np.linspace(xmin - pad, xmax + pad, 256)

    _, ax = plt.subplots(figsize=(6.5, 4.2))
    ax_kde = ax.twinx()

    for i, b in enumerate(df[buckets_col].unique()):
        vals = df.loc[df[buckets_col] == b, returns_col].to_numpy()
        if vals.size == 0:
            continue
        color = colors[i % len(colors)]

        ax.hist(
            vals, bins=bins, density=True, alpha=0.45,
            label=f"{b} (n={vals.size})", color=color, edgecolor="black", linewidth=0.3
        )
        ax.axvline(np.mean(vals), ls="--", lw=1, color=color, alpha=0.9)

        kde = stats.gaussian_kde(vals, bw_method="silverman")
        ax_kde.plot(x_grid, kde(x_grid), lw=1.3, color=color, alpha=0.9)

    asset, horizon = returns_col.split("_")
    ax.set_title(f"Bucket distributions for {asset} @ {horizon[:-1]}min horizon")
    ax.set_xlabel("Returns (bps)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.legend(frameon=False)
    ax_kde.set_ylabel("KDE")

    plt.tight_layout()
    plt.show()

def plot_fedfunds_aligned(df: pd.DataFrame, fed_csv: Path, outpath: Path,
                          show_delta: bool = True,
                          plot_mode: str = 'both',
                          ma_window: int = 3):
    """Plot FEDFUNDS restricted to the period where score data exists, and align scales.

    This maps the original `score_stmt` and `score_qa` linearly into the FEDFUNDS
    value range so both can be compared on the same axis. A secondary y-axis shows
    the original score scale (inverse transform).
    """
    plt.close('all')
    sns.set(style='whitegrid')

    fed = pd.read_csv(fed_csv)
    fed['observation_date'] = pd.to_datetime(fed['observation_date'], errors='coerce')
    fed = fed.dropna(subset=['observation_date'])
    fed = fed.sort_values('observation_date')

    df_local = df.copy()
    if 'meeting_date_parsed' not in df_local.columns:
        df_local['meeting_date_parsed'] = pd.to_datetime(df_local['meeting_date'], dayfirst=True, errors='coerce')

    # determine the date window from sentiment meeting dates
    meeting_dates = df_local['meeting_date_parsed'].dropna()
    if meeting_dates.empty:
        raise ValueError('No parsable meeting_date to determine period for aligned plot')

    start = meeting_dates.min() - pd.Timedelta(days=15)
    end = meeting_dates.max() + pd.Timedelta(days=15)

    fed_window = fed[(fed['observation_date'] >= start) & (fed['observation_date'] <= end)].copy()
    if fed_window.empty:
        raise ValueError('No FEDFUNDS data in the meeting date window')

    # compute scaling: map score range to fed range
    fed_min, fed_max = fed_window['FEDFUNDS'].min(), fed_window['FEDFUNDS'].max()
    scores = pd.concat([df_local['score_stmt'], df_local['score_qa']]).dropna()
    score_min, score_max = scores.min(), scores.max()
    if score_max == score_min:
        raise ValueError('Score values are constant; cannot rescale')

    def score_to_fed(s):
        return (s - score_min) / (score_max - score_min) * (fed_max - fed_min) + fed_min

    def fed_to_score(f):
        return (f - fed_min) / (fed_max - fed_min) * (score_max - score_min) + score_min

    # prepare figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fed_window['observation_date'], fed_window['FEDFUNDS'], color='#1f77b4', linewidth=2, label='FEDFUNDS (monthly)')

    # plot scaled scores on same axis (conditional points)
    mask = df_local['meeting_date_parsed'].notna()
    if plot_mode in ('points', 'both'):
        ax.scatter(df_local.loc[mask, 'meeting_date_parsed'], score_to_fed(df_local.loc[mask, 'score_stmt']), color='#FF6B6B', label='score_stmt (scaled)', zorder=6)
        ax.scatter(df_local.loc[mask, 'meeting_date_parsed'], score_to_fed(df_local.loc[mask, 'score_qa']), color='#FFD43B', label='score_qa (scaled)', zorder=6, marker='x')
        if show_delta and 'delta_score' in df_local.columns:
            ax.scatter(df_local.loc[mask, 'meeting_date_parsed'], score_to_fed(df_local.loc[mask, 'delta_score']), color='#6A4C93', label='delta_score (scaled)', zorder=6, marker='^')

    # rolling averages (by meeting sequence) and plot them scaled to the FED axis
    if plot_mode in ('ma', 'both'):
        try:
            df_roll = df_local.sort_values('meeting_date_parsed').set_index('meeting_date_parsed')
            cols = [c for c in ['score_stmt', 'score_qa', 'delta_score'] if c in df_roll.columns]
            roll = df_roll[cols].rolling(window=ma_window, min_periods=1).mean()
            roll_idx = roll.index
            if 'score_stmt' in roll.columns:
                ax.plot(roll_idx, score_to_fed(roll['score_stmt']), color='#FF6B6B', linestyle='-', linewidth=2, alpha=0.9, label=f'score_stmt (roll={ma_window}, scaled)')
            if 'score_qa' in roll.columns:
                ax.plot(roll_idx, score_to_fed(roll['score_qa']), color='#FFD43B', linestyle='-', linewidth=2, alpha=0.9, label=f'score_qa (roll={ma_window}, scaled)')
            if show_delta and 'delta_score' in roll.columns:
                ax.plot(roll_idx, score_to_fed(roll['delta_score']), color='#6A4C93', linestyle='--', linewidth=2, alpha=0.9, label=f'delta_score (roll={ma_window}, scaled)')
        except Exception:
            pass

    ax.set_xlabel('Date')
    ax.set_ylabel('FED funds rate (%)')
    ax.set_title('FEDFUNDS and scores (scores rescaled to FED range)')

    # secondary axis shows original score scale via inverse transform
    secax = ax.secondary_yaxis('right', functions=(fed_to_score, score_to_fed))
    secax.set_ylabel('Original scores')

    # legend
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, loc='upper left')

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    fig.tight_layout()
    out = outpath
    fig.savefig(out, dpi=200)