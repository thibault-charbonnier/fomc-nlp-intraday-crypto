import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Sequence, Optional
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_bucket_distributions_by_window_wide(
    df: pd.DataFrame,
    horizon_min: int,
    asset_prefix: str = "BTCUSDT",
    buckets: Sequence[str] = ("dovish", "neutral", "hawkish"),
    bins: int = 20,
    density: bool = True,
    plot_kde: bool = False,
    x_label: str = "Return (bps)",
    title_prefix: str = "Return distribution"
) -> None:
    """
    Version 'wide' : on lit dynamiquement la colonne de rendement {asset_prefix}_{horizon_min}
    et la colonne de bucket 'ton_bucket'.

    Paramètres
    ----------
    df : DataFrame
        Doit contenir :
          - 'ton_bucket' (str) : bucket tonal (ex: 'dovish' / 'neutral' / 'hawkish')
          - f'{asset_prefix}_{horizon_min}' : rendements pour l'horizon choisi (ex: 'BTCUSDT_10')
    horizon_min : int
        Fenêtre à tracer (ex: 1, 5, 10, 20 ...)
    asset_prefix : str
        Préfixe de la colonne de rendement (par défaut 'BTCUSDT').
    buckets : Sequence[str]
        Ordre d'affichage des buckets.
    bins : int
        Nombre de bins de l'histogramme.
    density : bool
        Normaliser les histogrammes en densité.
    plot_kde : bool
        Tracer les KDE (seconde échelle) pour lisser la vue.
    x_label : str
        Label de l'axe X.
    title_prefix : str
        Préfixe du titre (le suffixe ajoute l'horizon automatiquement).

    Retour
    ------
    None (trace la figure)
    """
    ret_col = f"{asset_prefix}_{horizon_min}m"
    if "tone_bucket" not in df.columns:
        raise ValueError("Colonne 'tone_bucket' absente du DataFrame.")
    if ret_col not in df.columns:
        raise ValueError(f"Colonne '{ret_col}' absente du DataFrame.")

    sub = df[["tone_bucket", ret_col]].copy()
    sub = sub.dropna(subset=[ret_col])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax_kde = ax.twinx()  # pour KDE éventuelle

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Étendue globale pour des axes cohérents
    all_vals = sub[ret_col].dropna()
    if len(all_vals) > 0:
        global_xmin, global_xmax = float(all_vals.min()), float(all_vals.max())
    else:
        global_xmin, global_xmax = 0.0, 1.0

    x_grid = np.linspace(global_xmin, global_xmax, 200)
    hist_ymax, kde_ymax = 0.0, 0.0

    # Pré-calcul des ymax pour une échelle commune
    for b in buckets:
        vals = sub.loc[sub["tone_bucket"] == b, ret_col].dropna().to_numpy()
        if vals.size == 0:
            continue
        try:
            h_tmp, _ = np.histogram(vals, bins=bins, density=density)
            if np.isfinite(h_tmp).any():
                hist_ymax = max(hist_ymax, float(np.nanmax(h_tmp)))
        except Exception:
            pass
        if plot_kde and vals.size > 1:
            try:
                kde_tmp = stats.gaussian_kde(vals, bw_method="silverman")
                y_tmp = kde_tmp(x_grid)
                if np.isfinite(y_tmp).any():
                    kde_ymax = max(kde_ymax, float(np.nanmax(y_tmp)))
            except Exception:
                pass

    hist_handles, kde_handles = [], []

    # Tracés par bucket
    for i, b in enumerate(buckets):
        vals = sub.loc[sub["tone_bucket"] == b, ret_col].dropna().to_numpy()
        if vals.size == 0:
            continue

        color = colors[i % len(colors)]

        # Histogramme
        n_vals, bins_out, patches = ax.hist(
            vals,
            bins=bins,
            alpha=0.5,
            density=density,
            label=f"{b} (n={len(vals)})",
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
        hist_handles.append(Patch(facecolor=color, edgecolor="black", label=f"{b} (n={len(vals)})"))

        # KDE
        if plot_kde and vals.size > 1:
            try:
                kde = stats.gaussian_kde(vals, bw_method="silverman")
                y = kde(x_grid)
                line, = ax_kde.plot(x_grid, y, linewidth=1.5, color=color)
                kde_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=f"{b} KDE"))
            except Exception:
                pass

        # Marqueur de moyenne
        try:
            ax.axvline(vals.mean(), linestyle="--", linewidth=1, color=color)
        except Exception:
            pass

    # Marges et axes
    xrng = global_xmax - global_xmin
    pad = xrng * 0.05 if xrng > 0 else 1.0
    ax.set_xlim(global_xmin - pad, global_xmax + pad)

    ax.set_title(f"{title_prefix} @ {horizon_min}m ({asset_prefix})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density" if density else "Frequency")

    handles = hist_handles + kde_handles
    if handles:
        ax.legend(handles=handles, frameon=False)

    ax.grid(alpha=0.2)
    if density and hist_ymax > 0:
        ax.set_ylim(0, hist_ymax * 1.08)
    if plot_kde and kde_ymax > 0:
        ax_kde.set_ylim(0, kde_ymax * 1.08)

    plt.tight_layout()
    plt.show()


def plot_window_distributions_by_bucket(
    df_long: pd.DataFrame,
    tone_bucket: str,
    windows: List[int],
    bins: int = 20,
    density: bool = True,
    plot_kde: bool = False
) -> None:
    """
    Pour UN bucket donné (ex: 'hawkish'),
    trace plusieurs sous-graphiques,
    un par horizon (1m,5m,10m...),
    pour visualiser comment l'effet se développe dans le temps.

    Params:
        df_long : pd.DataFrame
            Doit venir de build_returns_long_format().
        tone_bucket : str
            Bucket à tracer ("dovish", "neutral", "hawkish").
        windows : List[int]
            Horizons à tracer (1,5,10,...).
        bins : int
            Nombre de bins pour les histogrammes.
        density : bool
            Normaliser en densité.
    """
    # Sous-échantillonner le bucket demandé
    sub_all = df_long[df_long["tone_bucket"] == tone_bucket].copy()

    n_plots = len(windows)
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(4 * n_plots, 4),
        sharey=True,
    )

    # Si len(windows)==1, axes n'est pas iterable
    if n_plots == 1:
        axes = [axes]

    # couleurs (un bucket -> un jeu de couleurs)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = colors[0]

    # calculer bornes globales pour comparer graphiquement
    sub_selected = sub_all[sub_all["horizon_min"].isin(windows)]["return_bps"].dropna()
    if len(sub_selected) > 0:
        global_xmin, global_xmax = sub_selected.min(), sub_selected.max()
    else:
        global_xmin, global_xmax = 0.0, 1.0
    x_grid = np.linspace(global_xmin, global_xmax, 200)
    hist_ymax = 0.0
    kde_ymax = 0.0

    # pré-calcul y_max
    for h in windows:
        vals_tmp = sub_all.loc[sub_all["horizon_min"] == h, "return_bps"].dropna()
        if len(vals_tmp) == 0:
            continue
        try:
            h_tmp, _ = np.histogram(vals_tmp, bins=bins, density=True)
            hist_ymax = max(hist_ymax, float(np.max(h_tmp)))
        except Exception:
            pass
        if plot_kde and len(vals_tmp) > 1:
            try:
                y_tmp = stats.gaussian_kde(vals_tmp, bw_method="silverman")(x_grid)
                kde_ymax = max(kde_ymax, float(np.max(y_tmp)))
            except Exception:
                pass

    for ax, h in zip(axes, windows):
        ax_kde = ax.twinx()
        vals = sub_all.loc[sub_all["horizon_min"] == h, "return_bps"].dropna()

        ax.hist(
            vals,
            bins=bins,
            alpha=0.6,
            density=density,
            label=f"{tone_bucket} (n={len(vals)})",
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )

        if plot_kde and len(vals) > 1:
            try:
                y = stats.gaussian_kde(vals, bw_method="silverman")(x_grid)
                ax_kde.plot(x_grid, y, linewidth=1.2, color=color)
            except Exception:
                pass

        if len(vals) > 0:
            ax.axvline(vals.mean(), linestyle="--", linewidth=1, color=color)

        ax.set_title(f"{h}m")
        ax.set_xlabel("Return (bps)")
        ax.set_ylabel("Density" if density else "Frequency")
        ax.grid(alpha=0.2)

        # légende par subplot: histogramme + KDE
        handles = [Patch(facecolor=color, edgecolor="black", label=f"{tone_bucket} (n={len(vals)})")]
        if plot_kde and len(vals) > 1:
            handles.append(Line2D([0], [0], color=color, linewidth=1.2, label=f"{tone_bucket} KDE"))
        ax.legend(handles=handles, frameon=False)

        # fixer mêmes limites pour comparaison
        xrng = global_xmax - global_xmin
        pad = xrng * 0.05 if xrng > 0 else 1.0
        ax.set_xlim(global_xmin - pad, global_xmax + pad)
        if density and hist_ymax > 0:
            ax.set_ylim(0, hist_ymax * 1.08)
        if plot_kde and kde_ymax > 0:
            ax_kde.set_ylim(0, kde_ymax * 1.08)

    fig.suptitle(
        f"Return distributions for bucket '{tone_bucket}'\n"
        "across horizons",
        y=1.05,
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()
