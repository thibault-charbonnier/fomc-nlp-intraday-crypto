import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Sequence, Optional
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_bucket_distributions_by_window(
    df_long: pd.DataFrame,
    horizon_min: int,
    buckets: Sequence[str] = ("dovish", "neutral", "hawkish"),
    bins: int = 20,
    density: bool = True,
    plot_kde: bool = False,
) -> None:
    """
    Pour UNE window donnée (ex: 1 minute),
    compare la distribution des rendements entre buckets de ton.

    On overlay les histogrammes pour:
        - dovish
        - neutral
        - hawkish
    sur la même figure.

    Params:
        df_long : pd.DataFrame
            Doit venir de build_returns_long_format(), contient:
            ["horizon_min", "return_bps", "tone_bucket", "symbol", ...]
        horizon_min : int
            Fenêtre qu'on veut visualiser (1, 5, 10 ...)
        buckets : Sequence[str]
            Ordre des buckets à tracer.
        bins : int
            Nb de bins pour les histos.
        density : bool
            Normalise les histogrammes en densité.
    """
    sub = df_long[df_long["horizon_min"] == horizon_min].copy()

    fig, ax = plt.subplots(figsize=(6, 4))

    # axe secondaire pour KDE (échelle séparée)
    ax_kde = ax.twinx()

    # couleur cohérente entre histogramme et KDE
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # calculer étendue globale pour x et y (pour comparer graphiquement)
    all_vals = sub["return_bps"].dropna()
    if len(all_vals) > 0:
        global_xmin, global_xmax = all_vals.min(), all_vals.max()
    else:
        global_xmin, global_xmax = 0.0, 1.0

    x_grid = np.linspace(global_xmin, global_xmax, 200)
    hist_ymax = 0.0
    kde_ymax = 0.0

    # pré-calcul des ymax pour fixer une échelle commune
    for b in buckets:
        vals = sub.loc[sub["tone_bucket"] == b, "return_bps"].dropna()
        if len(vals) == 0:
            continue
        try:
            h_tmp, _ = np.histogram(vals, bins=bins, density=True)
            hist_ymax = max(hist_ymax, float(np.max(h_tmp)))
        except Exception:
            pass
        if plot_kde and len(vals) > 1:
            try:
                kde_tmp = stats.gaussian_kde(vals, bw_method="silverman")
                y_tmp = kde_tmp(x_grid)
                kde_ymax = max(kde_ymax, float(np.max(y_tmp)))
            except Exception:
                pass

    hist_handles = []
    kde_handles = []

    for i, b in enumerate(buckets):
        vals = sub.loc[sub["tone_bucket"] == b, "return_bps"].dropna()
        if len(vals) == 0:
            continue

        color = colors[i % len(colors)]

        # histogramme (avec même couleur que la KDE)
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

        if plot_kde and len(vals) > 1:
            try:
                kde = stats.gaussian_kde(vals, bw_method="silverman")
                # tracer la KDE sur la grille globale pour garder la même échelle
                y = kde(x_grid)
                line, = ax_kde.plot(x_grid, y, linewidth=1.5, color=color)
                kde_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=f"{b} KDE"))
            except Exception:
                pass

        # Marqueur de moyenne (même couleur)
        try:
            ax.axvline(vals.mean(), linestyle="--", linewidth=1, color=color)
        except Exception:
            pass

    # note: les lignes de moyenne sont déjà tracées par bucket dans la boucle

    # étendre légèrement l'axe x pour éviter de couper les queues
    xrng = global_xmax - global_xmin
    pad = xrng * 0.05 if xrng > 0 else 1.0
    ax.set_xlim(global_xmin - pad, global_xmax + pad)

    ax.set_title(f"Return distribution @ {horizon_min}m")
    ax.set_xlabel("Return (bps)")
    ax.set_ylabel("Density" if density else "Frequency")
    # construire une légende claire (histogrammes + KDE si présents)
    handles = hist_handles + kde_handles
    if handles:
        ax.legend(handles=handles, frameon=False)

    ax.grid(alpha=0.2)
    # mettre une échelle y commune si density=True
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
