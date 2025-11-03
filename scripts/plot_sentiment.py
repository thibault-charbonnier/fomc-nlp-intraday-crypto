#!/usr/bin/env python3
"""
Create three plots from the sentiment cache CSV:

- Overlapped histograms of `score_stmt` and `score_qa` (different colors, alpha)
- Histogram of `delta_score` with quantile lines (5%,15%,25% and symmetric 95%,85%,75%)
- Plot of `score_stmt` by `meeting_date` (meeting_date on the y axis)

Saves outputs to `outputs/`.
"""
import argparse
from pathlib import Path
import textwrap
from scipy import stats
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def ensure_delta(df: pd.DataFrame) -> pd.DataFrame:
    if 'delta_score' not in df.columns:
        # Assume delta_score = score_stmt - score_qa
        if 'score_stmt' in df.columns and 'score_qa' in df.columns:
            df['delta_score'] = df['score_stmt'] - df['score_qa']
        else:
            raise KeyError('delta_score missing and cannot be computed (score_stmt or score_qa absent)')
    return df


def plot_overlapped_hist(df: pd.DataFrame, outpath: Path):
    plt.close('all')
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = ['score_stmt', 'score_qa']
    colors = ['#2E86AB', '#F6C85F']
    maxv = df[vals].max().max()
    minv = df[vals].min().min()
    bins = np.linspace(minv, maxv, 40)

    ax.hist(df['score_stmt'].dropna(), bins=bins, color=colors[0], alpha=0.6, label='score_stmt', edgecolor='white')
    ax.hist(df['score_qa'].dropna(), bins=bins, color=colors[1], alpha=0.5, label='score_qa', edgecolor='white')

    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title('Overlapped histograms: score_stmt vs score_qa')
    ax.legend()
    sns.despine()
    out = outpath / 'hist_overlap_score_stmt_score_qa.png'
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


def plot_delta_hist_with_quantiles(df: pd.DataFrame, outpath: Path):
    plt.close('all')
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df['delta_score'].dropna()
    ax.hist(data, bins=60, color='#6A4C93', alpha=0.8, edgecolor='white')
    qs = [0.05, 0.15, 0.25]
    colors = ['#FF6B6B', '#FFA94D', '#FFD43B']

    for q, c in zip(qs, colors):
        v_low = np.quantile(data, q)
        v_high = np.quantile(data, 1 - q)
        ax.axvline(v_low, color=c, linestyle='--', linewidth=1.5)
        ax.axvline(v_high, color=c, linestyle='--', linewidth=1.5)
        ax.text(v_low, ax.get_ylim()[1] * 0.9, f'{int(q*100)}%: {v_low:.3f}', color=c, rotation=90, va='top')
        ax.text(v_high, ax.get_ylim()[1] * 0.9, f'{int((1-q)*100)}%: {v_high:.3f}', color=c, rotation=90, va='top')

    ax.set_xlabel('Delta score (score_stmt - score_qa)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of delta_score with quantiles')
    sns.despine()
    fig.tight_layout()
    out = outpath / 'hist_delta_score_quantiles.png'
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


def plot_score_by_meeting_date(df: pd.DataFrame, outpath: Path):
    plt.close('all')
    sns.set(style='white')
    if 'meeting_date' not in df.columns:
        raise KeyError('meeting_date column not found in data')

    # Try to parse meeting_date to datetime
    df = df.copy()
    # dates in the sentiment CSV use day-month-year like '29-01-2020'
    df['meeting_date_parsed'] = pd.to_datetime(df['meeting_date'], dayfirst=True, errors='coerce')
    if df['meeting_date_parsed'].isna().all():
        raise ValueError('meeting_date could not be parsed to datetimes')

    # Sort by meeting_date
    df = df.sort_values('meeting_date_parsed')

    fig, ax = plt.subplots(figsize=(10, 6))

    # x: score, y: meeting date (ordered vertically)
    x = df['score_stmt']
    y = df['meeting_date_parsed']

    # scatter with some transparency
    ax.scatter(x, y, alpha=0.8, color='#2E86AB', s=40)

    ax.set_xlabel('score_stmt')
    ax.set_ylabel('meeting_date')
    ax.set_title('score_stmt by meeting_date')

    # Format y axis as dates
    ax.yaxis_date()
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    fig.tight_layout()
    out = outpath / 'score_stmt_by_meeting_date.png'
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


def plot_fedfunds_with_scores(df: pd.DataFrame, fed_csv: Path, outpath: Path,
                              show_delta: bool = True,
                              plot_mode: str = 'both',
                              ma_window: int = 3):
    """Plot FEDFUNDS time series (line) and overlay score_stmt and score_qa as points at meeting dates.

    Uses a secondary y-axis for scores so both series are visible. Matches monthly FEDFUNDS
    to meeting dates by using the meeting month for reference (FED data is monthly).
    """
    plt.close('all')
    sns.set(style='whitegrid')

    fed = pd.read_csv(fed_csv)
    fed['observation_date'] = pd.to_datetime(fed['observation_date'], errors='coerce')
    fed = fed.dropna(subset=['observation_date'])
    fed = fed.sort_values('observation_date')

    df_local = df.copy()
    # ensure meeting_date_parsed exists and parsed with dayfirst
    if 'meeting_date_parsed' not in df_local.columns:
        df_local['meeting_date_parsed'] = pd.to_datetime(df_local['meeting_date'], dayfirst=True, errors='coerce')

    # aggregate fed values by month (they already are monthly) and use timestamp at month start
    fed['month'] = fed['observation_date'].dt.to_period('M').dt.to_timestamp()

    # meeting month for matching
    df_local['meeting_month'] = df_local['meeting_date_parsed'].dt.to_period('M').dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 6))

    # plot fed funds line
    ax.plot(fed['observation_date'], fed['FEDFUNDS'], color='#1f77b4', linewidth=2, label='FEDFUNDS (monthly)')
    ax.set_xlabel('Date')
    ax.set_ylabel('FED funds rate (%)', color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')

    # secondary axis for scores
    ax2 = ax.twinx()

    # plot score points at their meeting dates (conditional)
    mask = df_local['meeting_date_parsed'].notna()
    if plot_mode in ('points', 'both'):
        ax2.scatter(df_local.loc[mask, 'meeting_date_parsed'], df_local.loc[mask, 'score_stmt'], color='#FF6B6B', label='score_stmt', zorder=5)
        ax2.scatter(df_local.loc[mask, 'meeting_date_parsed'], df_local.loc[mask, 'score_qa'], color='#FFD43B', label='score_qa', zorder=5, marker='x')
        if show_delta and 'delta_score' in df_local.columns:
            ax2.scatter(df_local.loc[mask, 'meeting_date_parsed'], df_local.loc[mask, 'delta_score'], color='#6A4C93', label='delta_score', zorder=5, marker='^')

    # rolling averages (by meeting sequence). window set by ma_window
    if plot_mode in ('ma', 'both'):
        try:
            df_roll = df_local.sort_values('meeting_date_parsed').set_index('meeting_date_parsed')
            cols = [c for c in ['score_stmt', 'score_qa', 'delta_score'] if c in df_roll.columns]
            roll = df_roll[cols].rolling(window=ma_window, min_periods=1).mean()
            roll_idx = roll.index
            if 'score_stmt' in roll.columns:
                ax2.plot(roll_idx, roll['score_stmt'], color='#FF6B6B', linestyle='-', linewidth=2, alpha=0.9, label=f'score_stmt (roll={ma_window})')
            if 'score_qa' in roll.columns:
                ax2.plot(roll_idx, roll['score_qa'], color='#FFD43B', linestyle='-', linewidth=2, alpha=0.9, label=f'score_qa (roll={ma_window})')
            if show_delta and 'delta_score' in roll.columns:
                ax2.plot(roll_idx, roll['delta_score'], color='#6A4C93', linestyle='--', linewidth=2, alpha=0.9, label=f'delta_score (roll={ma_window})')
        except Exception:
            pass
    ax2.set_ylabel('Scores', color='#333333')
    ax2.tick_params(axis='y', labelcolor='#333333')

    # combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left')

    # format x axis dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    fig.tight_layout()
    out = outpath / 'fedfunds_with_scores.png'
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


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


def _match_fed_to_meetings(df: pd.DataFrame, fed_csv: Path) -> pd.DataFrame:
    """Return a DataFrame with meeting_date_parsed and matched FEDFUNDS value for that meeting's month."""
    fed = pd.read_csv(fed_csv)
    fed['observation_date'] = pd.to_datetime(fed['observation_date'], errors='coerce')
    fed = fed.dropna(subset=['observation_date'])
    fed['month'] = fed['observation_date'].dt.to_period('M').dt.to_timestamp()

    df_local = df.copy()
    if 'meeting_date_parsed' not in df_local.columns:
        df_local['meeting_date_parsed'] = pd.to_datetime(df_local['meeting_date'], dayfirst=True, errors='coerce')
    df_local['meeting_month'] = df_local['meeting_date_parsed'].dt.to_period('M').dt.to_timestamp()

    merged = pd.merge(df_local, fed[['month', 'FEDFUNDS']], left_on='meeting_month', right_on='month', how='left')
    # merged contains columns: meeting_date_parsed, score_stmt, score_qa, delta_score, FEDFUNDS
    return merged


def compute_correlations(df: pd.DataFrame, fed_csv: Path, ma_window: int = 3):
    """Compute Pearson correlations between FEDFUNDS and score_stmt / score_qa.

    Returns a dict with raw and smoothed correlations. The smoothing uses rolling window
    based on meeting sequence (window=ma_window).
    """
    merged = _match_fed_to_meetings(df, fed_csv)
    out = {}

    def pearson(a, b):
        a = np.array(a)
        b = np.array(b)
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 2:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    out['raw'] = {
        'fed_stmt': pearson(merged['FEDFUNDS'], merged['score_stmt']),
        'fed_qa': pearson(merged['FEDFUNDS'], merged['score_qa'])
    }

    # smoothed (rolling by meeting sequence)
    df_roll = merged.sort_values('meeting_date_parsed').set_index('meeting_date_parsed')
    cols = [c for c in ['score_stmt', 'score_qa', 'delta_score'] if c in df_roll.columns]
    roll = df_roll[cols].rolling(window=ma_window, min_periods=1).mean()
    roll_merged = pd.concat([roll, df_roll['FEDFUNDS']], axis=1)
    out['smoothed'] = {
        'fed_stmt': pearson(roll_merged['FEDFUNDS'], roll_merged['score_stmt']) if 'score_stmt' in roll_merged else np.nan,
        'fed_qa': pearson(roll_merged['FEDFUNDS'], roll_merged['score_qa']) if 'score_qa' in roll_merged else np.nan
    }

    return out


def regress_fed_on_score(df: pd.DataFrame, fed_csv: Path, score_col: str = 'score_stmt', ma_window: int = None):
    """Perform a linear regression FEDFUNDS ~ score (or smoothed score if ma_window provided).

    Returns a dict with slope, intercept, stderr, and approximate 95% CI for slope, plus n.
    Uses np.polyfit with cov=True to estimate slope std error; uses normal approx for CI
    (fallback if scipy not available)."""
    merged = _match_fed_to_meetings(df, fed_csv)
    # select series
    data = merged[['FEDFUNDS', score_col]].dropna()
    if data.shape[0] < 2:
        return {'n': int(data.shape[0]), 'slope': np.nan, 'intercept': np.nan, 'slope_se': np.nan, 'slope_ci': (np.nan, np.nan)}

    x = data[score_col].values
    y = data['FEDFUNDS'].values

    if ma_window is not None and ma_window >= 1:
        # smooth x by rolling on meeting sequence
        tmp = merged.sort_values('meeting_date_parsed').set_index('meeting_date_parsed')
        if score_col not in tmp.columns:
            return {'n': int(len(x)), 'slope': np.nan, 'intercept': np.nan, 'slope_se': np.nan, 'slope_ci': (np.nan, np.nan)}
        x = tmp[score_col].rolling(window=ma_window, min_periods=1).mean().dropna().values
        # align y to same index
        y = tmp['FEDFUNDS'].loc[tmp[score_col].rolling(window=ma_window, min_periods=1).mean().dropna().index].values

    # linear fit with covariance
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = coeffs[0], coeffs[1]
    slope_se = float(np.sqrt(cov[0, 0]))
    n = len(x)

    # approximate 95% CI using normal quantile (1.96) as fallback
    z = 1.96
    try:
        tcrit = float(stats.t.ppf(0.975, df=n - 2)) if n > 2 else z
    except Exception:
        tcrit = z

    ci_low = slope - tcrit * slope_se
    ci_high = slope + tcrit * slope_se

    return {'n': int(n), 'slope': float(slope), 'intercept': float(intercept), 'slope_se': float(slope_se), 'slope_ci': (float(ci_low), float(ci_high))}


def plot_fed_vs_score(df: pd.DataFrame, fed_csv: Path, outpath: Path,
                      score_col: str = 'score_stmt',
                      show_regression: bool = True):
    """Scatter plot of FED rate changes (month-over-month) vs a chosen score (score_stmt or score_qa).

    We compute the monthly FEDFUNDS change (delta = this_month - previous_month) and match each
    meeting to its month. The plot shows fed_delta on the y-axis and score on the x-axis.
    Optionally fits and plots a regression line and annotates slope + 95% CI.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    merged = _match_fed_to_meetings(df, fed_csv)
    data = merged[['meeting_date_parsed', 'FEDFUNDS', score_col]].dropna()
    if data.empty:
        raise ValueError('No paired FEDFUNDS and score data available for plotting')

    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter: x = score, y = fed
    ax.scatter(data[score_col], data['FEDFUNDS'], color='#2E86AB', alpha=0.8, label='meetings')
    ax.set_xlabel(score_col)
    ax.set_ylabel('FED funds rate (%)')
    ax.set_title(f'FEDFUNDS vs {score_col}')

    # compute FED month-over-month delta and merge to meetings
    fed = pd.read_csv(fed_csv)
    fed['observation_date'] = pd.to_datetime(fed['observation_date'], errors='coerce')
    fed = fed.dropna(subset=['observation_date']).sort_values('observation_date')
    fed['month'] = fed['observation_date'].dt.to_period('M').dt.to_timestamp()
    fed_month = fed[['month', 'FEDFUNDS']].drop_duplicates().sort_values('month')
    fed_month['fed_delta'] = fed_month['FEDFUNDS'].diff()
    # merge month delta into merged (which already has meeting_month)
    merged2 = merged.merge(fed_month[['month', 'fed_delta']], left_on='meeting_month', right_on='month', how='left')
    data = merged2[['meeting_date_parsed', 'fed_delta', score_col]].dropna()
    if data.empty:
        raise ValueError('No paired FEDFUNDS delta and score data available for plotting')

    # fit regression fed_delta ~ score and plot
    if show_regression:
        x = data[score_col].values
        y = data['fed_delta'].values
        if len(x) >= 2:
            coeffs, cov = np.polyfit(x, y, 1, cov=True)
            slope, intercept = coeffs[0], coeffs[1]
            xs = np.linspace(x.min(), x.max(), 100)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color='#333333', linestyle='--', linewidth=1.8, label=f'reg slope={slope:.3f}')
            # slope CI
            slope_se = float(np.sqrt(cov[0, 0]))
            n = len(x)
            try:
                from scipy import stats
                tcrit = float(stats.t.ppf(0.975, df=n - 2)) if n > 2 else 1.96
            except Exception:
                tcrit = 1.96
            ci_low = slope - tcrit * slope_se
            ci_high = slope + tcrit * slope_se
            ax.annotate(f'slope={slope:.3f}\n95% CI=({ci_low:.3f},{ci_high:.3f})', xy=(0.02, 0.98), xycoords='axes fraction', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend()
    fig.tight_layout()
    out = outpath / f'fed_delta_vs_{score_col}.png'
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(__doc__))
    parser.add_argument('--csv', '-c', type=Path, default=Path('data/sentiment/sentiment_cache_20251101_115727.csv'), help='Path to sentiment CSV')
    parser.add_argument('--outdir', '-o', type=Path, default=Path('outputs'), help='Output directory for plots')
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f'CSV not found: {args.csv}')

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    df = ensure_delta(df)

    plot_overlapped_hist(df, outdir)
    plot_delta_hist_with_quantiles(df, outdir)

    # Only attempt meeting_date plot if column exists
    if 'meeting_date' in df.columns:
        try:
            plot_score_by_meeting_date(df, outdir)
        except Exception as e:
            print(f'Could not create meeting_date plot: {e}')
    else:
        print('meeting_date column not present; skipped meeting_date plot')

    # FEDFUNDS plot: if rates file exists, overlay fed funds line with score points
    fed_csv = Path('data/rates/FEDFUNDS.csv')
    if fed_csv.exists():
        try:
            plot_fedfunds_with_scores(df, fed_csv, outdir)
            # also create an aligned-scale plot restricted to the score period
            plot_fedfunds_aligned(df, fed_csv, outdir, ma_window=2, show_delta=True, plot_mode='both')
            # scatter plots FED vs score for stmt and qa
            try:
                plot_fed_vs_score(df, fed_csv, outdir, score_col='score_stmt', show_regression=True)
            except Exception as _e:
                print(f'Could not create fed vs score (stmt): {_e}')
            try:
                plot_fed_vs_score(df, fed_csv, outdir, score_col='score_qa', show_regression=True)
            except Exception as _e:
                print(f'Could not create fed vs score (qa): {_e}')
        except Exception as e:
            print(f'Could not create FEDFUNDS with scores plot: {e}')
    else:
        print('FEDFUNDS.csv not found; skipped FEDFUNDS plot')


if __name__ == '__main__':
    main()
