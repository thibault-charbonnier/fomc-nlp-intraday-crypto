import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List, Tuple, Optional, Dict
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import brier_score_loss
import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import List, Dict, Tuple
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


def fit_logit_direction(
    df: pd.DataFrame,
    features: List[str],
    target_return_col: str = "r_std",
    cluster_col: str = "meeting_id",
) -> Dict[str, object]:
    """
    Régression logistique du signe du retour sur un ensemble de features (ex: score NLP).
    - Modèle : GLM Binomial (logit) avec covariance robuste clusterisée par meeting.
    - Cible : y = 1 si retour > 0, 0 sinon.
    """
    data = df.dropna(subset=features + [target_return_col, cluster_col]).copy()

    y = (data[target_return_col].values > 0).astype(int)
    X = sm.add_constant(data[features], has_constant="add")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})

    ci = res.conf_int()
    coef_table = pd.DataFrame({
        "coef": res.params,
        "std_err": res.bse,
        "z_or_t": res.tvalues,
        "p_value": res.pvalues,
        "ci_low": ci[0],
        "ci_high": ci[1],
    })
    out = {"result": res, "coef_table": coef_table}
    return out 

# ---------- 1) Tableau de bins (proportions + tests) ----------

def bin_table_with_tests(
    df: pd.DataFrame,
    ret_col: str = "r_std",
    bucket_col: str = "bucket",
    buckets: List[str] = ["dovish", "hawkish", "neutral"],
    edges: List[float] = [-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf],
) -> Dict[str, object]:
    d = df.dropna(subset=[ret_col, bucket_col]).copy()
    d["bin"] = pd.cut(d[ret_col], bins=edges, right=True, include_lowest=True)

    # Proportions + IC Wilson par bucket et par bin
    rows = []
    for b in buckets:
        sub = d[d[bucket_col] == b]
        n_tot = len(sub)
        for binv, grp in sub.groupby("bin"):
            k = len(grp)
            p = k / n_tot if n_tot > 0 else np.nan
            ci = proportion_confint(k, n_tot, alpha=0.05, method="wilson") if n_tot>0 else (np.nan, np.nan)
            rows.append({"bucket": b, "bin": binv, "n": n_tot, "k_in_bin": k, "prop": p,
                         "ci_low": ci[0], "ci_high": ci[1]})
    prop_table = pd.DataFrame(rows)

    # Tests de différence de proportions par bin (dovish vs hawkish)
    tests = []
    dv = prop_table[prop_table["bucket"] == "dovish"].set_index("bin")
    hk = prop_table[prop_table["bucket"] == "hawkish"].set_index("bin")
    common_bins = dv.index.intersection(hk.index)
    for binv in common_bins:
        k1, n1 = dv.loc[binv, "k_in_bin"], dv.loc[binv, "n"]
        k2, n2 = hk.loc[binv, "k_in_bin"], hk.loc[binv, "n"]
        if n1>0 and n2>0:
            z, p = proportions_ztest([k1, k2], [n1, n2])
        else:
            z, p = np.nan, np.nan
        tests.append({"bin": binv, "z": z, "p_value": p})
    test_table = pd.DataFrame(tests)

    return {"prop_table": prop_table, "tests_dov_vs_hawk": test_table}

# ---------- 2) Skew quantile (Bowley) + bootstrap cluster Δ ----------

def bowley_skew(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size < 3: return np.nan
    q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
    denom = (q75 - q25)
    if denom == 0: return np.nan
    return (q75 + q25 - 2*q50) / denom

def cluster_bootstrap_stat(
    df: pd.DataFrame,
    cluster_col: str,
    stat_fn,
    B: int = 1000,
    random_state: int = 0
) -> Tuple[float, Tuple[float,float]]:
    rng = np.random.default_rng(random_state)
    clusters = df[cluster_col].dropna().unique()
    stats = []
    for _ in range(B):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([df[df[cluster_col]==c] for c in pick], ignore_index=True)
        stats.append(stat_fn(sample))
    stats = np.array(stats, dtype=float)
    return np.nanmedian(stats), (np.nanpercentile(stats, 2.5), np.nanpercentile(stats, 97.5))

def skew_by_bucket_with_diff(
    df: pd.DataFrame,
    ret_col: str = "r_std",
    bucket_col: str = "bucket",
    cluster_col: str = "meeting_id",
) -> Dict[str, object]:
    d = df.dropna(subset=[ret_col, bucket_col, cluster_col]).copy()
    sk = d.groupby(bucket_col)[ret_col].apply(bowley_skew).to_dict()

    # Δ-skew dovish − hawkish
    def delta_sk(sample: pd.DataFrame):
        g = sample.groupby(bucket_col)[ret_col].apply(bowley_skew)
        if not {"dovish","hawkish"}.issubset(set(g.index)): return np.nan
        return g["dovish"] - g["hawkish"]

    _, ci = cluster_bootstrap_stat(d, cluster_col, delta_sk, B=1000)
    return {"skew_by_bucket": sk, "delta_skew_dov_minus_hawk": {"estimate": sk.get("dovish",np.nan)-sk.get("hawkish",np.nan),
                                                                "ci95": ci}}

# ---------- 3) Shift function (Δ-quantiles) + IC cluster ----------

def quantile_shift_curve(
    df: pd.DataFrame,
    taus: List[float] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    ret_col: str = "r_std",
    bucket_col: str = "bucket",
    cluster_col: str = "meeting_id",
    A: str = "dovish",
    B: str = "hawkish",
    B_boot: int = 1000
) -> pd.DataFrame:
    d = df.dropna(subset=[ret_col, bucket_col, cluster_col]).copy()
    out = []
    for t in taus:
        # estimate
        QA = np.quantile(d.loc[d[bucket_col]==A, ret_col], t) if (d[bucket_col]==A).any() else np.nan
        QB = np.quantile(d.loc[d[bucket_col]==B, ret_col], t) if (d[bucket_col]==B).any() else np.nan
        est = QA - QB

        # bootstrap clustered CI
        def qdiff(sample: pd.DataFrame, tau=t):
            a = sample.loc[sample[bucket_col]==A, ret_col].values
            b = sample.loc[sample[bucket_col]==B, ret_col].values
            if len(a)==0 or len(b)==0: return np.nan
            return np.quantile(a, tau) - np.quantile(b, tau)

        _, (lo, hi) = cluster_bootstrap_stat(d, cluster_col, qdiff, B=B_boot)
        out.append({"tau": t, "delta_Q_tau": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

# ---------- Skew de Bowley (quantile-based) ----------
def bowley_skew(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
    denom = (q75 - q25)
    if denom == 0:
        return np.nan
    return (q75 + q25 - 2*q50) / denom

# ---------- Bootstrap clusterisé du Δ-skew ----------
import numpy as np
import pandas as pd
from scipy.stats import skew as _skew
from typing import Tuple, Dict

def _moment_skew(x: np.ndarray) -> float:
    """Skew (3e moment standardisé) avec correction de biais. NaN-safe."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.std(x, ddof=1) == 0:
        return np.nan
    return float(_skew(x, bias=False))  # Fisher-Pearson, bias-corrected

def skew_bootstrap(
    df: pd.DataFrame,
    ret_col: str,
    bucket_col: str,
    cluster_col: str,
    bucket_A: str = "dovish",
    bucket_B: str = "hawkish",
    B: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> Dict[str, object]:
    """
    Compare l'asymétrie (skew par moments) entre deux buckets.
    Renvoie skew_A, skew_B, delta, IC (percentile bootstrap cluster) et les échantillons bootstrap.
    """
    d = df.dropna(subset=[ret_col, bucket_col, cluster_col]).copy()

    # Estimation sur l'échantillon observé
    A_vals = d.loc[d[bucket_col] == bucket_A, ret_col].values
    B_vals = d.loc[d[bucket_col] == bucket_B, ret_col].values
    skew_A = _moment_skew(A_vals)
    skew_B = _moment_skew(B_vals)
    delta_est = skew_A - skew_B

    # Bootstrap clusterisé par meeting
    rng = np.random.default_rng(random_state)
    clusters = d[cluster_col].unique()
    boot = np.empty(B)
    for b in range(B):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([d[d[cluster_col] == c] for c in pick], ignore_index=True)
        sA = _moment_skew(sample.loc[sample[bucket_col] == bucket_A, ret_col].values)
        sB = _moment_skew(sample.loc[sample[bucket_col] == bucket_B, ret_col].values)
        boot[b] = sA - sB

    lo, hi = np.nanpercentile(boot, [100*alpha/2, 100*(1-alpha/2)])

    print(f"Skew {bucket_A}: {skew_A:.4f},\nSkew {bucket_B}: {skew_B:.4f}, \nΔ-skew: {delta_est:.4f}, \nIC95%: [{lo:.4f}, {hi:.4f}]")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Iterable, Optional, Dict, Any

def _extract_ci(ci, idx: int) -> tuple[float, float]:
    """
    Supporte ci = ndarray shape (k,2) ou DataFrame (k x 2).
    Retourne (low, high) pour le paramètre d'indice idx.
    """
    if hasattr(ci, "iloc"):
        return float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])
    # ndarray
    return float(ci[idx, 0]), float(ci[idx, 1])

def fit_return_on_score_nocluster(
    df: pd.DataFrame,
    target_col: str,                  # ex: "BTCUSDT_20m"
    score_col: str = "score_stmt",    # ex: 0..10
    center_value: float = 5.0,        # recentre 0..10 -> -5..+5
    scale: Optional[float] = None,    # ex: 5.0 pour ramener ~[-1,1]
) -> Dict[str, Any]:
    """
    y = alpha + beta * x + eps, avec x = (score - center_value) / scale (optionnel).
    Estime OLS avec SE robustes HC1 (pas de clustering).
    Retourne alpha/beta, p-values, IC95%, R², n, et l'objet statsmodels.
    """
    used = [target_col, score_col]
    d = df.dropna(subset=used).copy()
    if d.empty:
        raise ValueError(f"Aucune donnée exploitable pour {target_col}.")

    x = d[score_col].astype(float) - center_value
    if scale is not None and scale != 0:
        x = x / scale
    y = d[target_col].astype(float).values
    X = sm.add_constant(x.values, has_constant="add")

    res = sm.OLS(y, X).fit(cov_type="HC1")  # SE robustes, pas de cluster
    ci = res.conf_int()

    a_lo, a_hi = _extract_ci(ci, 0)
    b_lo, b_hi = _extract_ci(ci, 1)

    return {
        "alpha": float(res.params[0]),
        "beta":  float(res.params[1]),
        "p_alpha": float(res.pvalues[0]),
        "p_beta":  float(res.pvalues[1]),
        "alpha_ci95": (a_lo, a_hi),
        "beta_ci95":  (b_lo, b_hi),
        "r2": float(res.rsquared),
        "n": int(res.nobs),
        "se_type": "HC1",
        "result": res,
    }

def run_regressions(
    df: pd.DataFrame,
    horizons_min: Iterable[int],
    asset_prefix: str = "BTCUSDT",
    score_col: str = "score_stmt",
    center_value: float = 5.0,
    scale: Optional[float] = None,
    col_pattern: str = "{asset}_{h}m",
    print_: bool = True,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Boucle sur plusieurs horizons (ex: [10,20,30,60]) et ajuste:
        r_{asset,h} ~ alpha + beta * score_centré
    Cherche d'abord '{asset}_{h}m', sinon fallback '{asset}_{h}'.
    Renvoie un DataFrame récapitulatif (peut être vide si aucune régression n'a tourné).
    """
    rows = []
    for h in horizons_min:
        target_col = col_pattern.format(asset=asset_prefix, h=h)
        if target_col not in df.columns:
            alt = f"{asset_prefix}_{h}"
            if alt in df.columns:
                target_col = alt
            else:
                if print_:
                    print(f"[SKIP] Colonne manquante: {col_pattern.format(asset=asset_prefix, h=h)} ou {alt}")
                continue

        try:
            res = fit_return_on_score_nocluster(
                df=df,
                target_col=target_col,
                score_col=score_col,
                center_value=center_value,
                scale=scale,
            )
        except Exception as e:
            if print_:
                print(f"[ERROR] {target_col}: {e}")
            continue

        if print_:
            d = decimals
            print(f"Régression du score_stmt sur les rendements {asset_prefix} à horizon {h} min :")
            print(f"  alpha = {res['alpha']:.{d}f} "
                  f"(p={res['p_alpha']:.{max(1,d-2)}g}, IC95% [{res['alpha_ci95'][0]:.{d}f}, {res['alpha_ci95'][1]:.{d}f}])")
            print(f"  beta  = {res['beta']:.{d}f} "
                  f"(p={res['p_beta']:.{max(1,d-2)}g}, IC95% [{res['beta_ci95'][0]:.{d}f}, {res['beta_ci95'][1]:.{d}f}])")
            print(f"  R² = {res['r2']:.{d}f}, n = {res['n']}, SE = {res['se_type']}")
            print("-" * 72)

        rows.append({
            "horizon_min": h,
            "alpha": res["alpha"], "alpha_p": res["p_alpha"],
            "alpha_ci_low": res["alpha_ci95"][0], "alpha_ci_high": res["alpha_ci95"][1],
            "beta": res["beta"], "beta_p": res["p_beta"],
            "beta_ci_low": res["beta_ci95"][0], "beta_ci_high": res["beta_ci95"][1],
            "r2": res["r2"], "n": res["n"], "se_type": res["se_type"],
            "target_col": target_col,
        })

    if not rows:
        # retourne un DF vide plutôt que d'échouer au sort
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Iterable, Optional, Dict, Any

def _extract_ci(ci, idx: int) -> tuple[float, float]:
    """Gère conf_int() DataFrame ou ndarray."""
    if hasattr(ci, "iloc"):
        return float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])
    return float(ci[idx, 0]), float(ci[idx, 1])

def fit_logit_sign_nocluster(
    df: pd.DataFrame,
    target_col: str,                  # ex: "BTCUSDT_20m"
    score_col: str = "score_stmt",    # 0..10
    center_value: float = 5.0,        # 0..10 -> -5..+5
    scale: Optional[float] = None,    # ex: 5.0 pour ~[-1,1], sinon None
    decision_threshold: float = 0.5,  # seuil pour le hit-rate in-sample
) -> Dict[str, Any]:
    """
    Modèle: 1[r>0] ~ Logit(alpha + beta * x), x = (score - center_value)/scale.
    SE robustes HC1. Renvoie alpha/beta, p-values, IC95, odds ratio et métriques simples.
    """
    d = df[[target_col, score_col]].dropna().copy()
    if d.empty:
        raise ValueError(f"Aucune donnée exploitable pour {target_col}.")

    x = d[score_col].astype(float) - center_value
    if scale is not None and scale != 0:
        x = x / scale

    y = (d[target_col].astype(float) > 0).astype(int)
    if y.nunique() < 2:
        raise ValueError("La variable cible est constante (tous les signes identiques).")

    X = sm.add_constant(x.values, has_constant="add")
    model = sm.GLM(y.values, X, family=sm.families.Binomial())
    res = model.fit(cov_type="HC1")

    ci = res.conf_int()
    a_lo, a_hi = _extract_ci(ci, 0)
    b_lo, b_hi = _extract_ci(ci, 1)

    p = res.predict(X)
    yhat = (p >= decision_threshold).astype(int)
    hit_rate = float((yhat == y.values).mean())
    brier = float(np.mean((p - y.values) ** 2))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y.values, p))
    except Exception:
        auc = np.nan

    return {
        "alpha": float(res.params[0]),
        "beta":  float(res.params[1]),
        "p_alpha": float(res.pvalues[0]),
        "p_beta":  float(res.pvalues[1]),
        "alpha_ci95": (a_lo, a_hi),
        "beta_ci95":  (b_lo, b_hi),
        "odds_beta": float(np.exp(res.params[1])),
        "odds_beta_ci95": (float(np.exp(b_lo)), float(np.exp(b_hi))),
        "n": int(res.nobs),
        "se_type": "HC1",
        "hit_rate": hit_rate,
        "brier": brier,
        "auc": auc,
        "result": res,
    }

def run_logit_over_horizons(
    df: pd.DataFrame,
    horizons_min: Iterable[int],
    asset_prefix: str = "BTCUSDT",
    score_col: str = "score_stmt",
    center_value: float = 5.0,
    scale: Optional[float] = None,
    col_pattern: str = "{asset}_{h}m",
    decision_threshold: float = 0.5,
    print_: bool = True,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Boucle sur h∈horizons_min et ajuste: 1[r_{asset,h}>0] ~ Logit(alpha + beta * score_centré).
    Cherche d'abord '{asset}_{h}m', sinon fallback '{asset}_{h}'.
    Retourne un tableau récap (alpha/beta, p-values, IC, odds, hit-rate, Brier, AUC).
    """
    rows = []
    for h in horizons_min:
        target_col = col_pattern.format(asset=asset_prefix, h=h)
        if target_col not in df.columns:
            alt = f"{asset_prefix}_{h}"
            if alt in df.columns:
                target_col = alt
            else:
                if print_:
                    print(f"[SKIP] Colonne manquante: {col_pattern.format(asset=asset_prefix, h=h)} ou {alt}")
                continue

        try:
            r = fit_logit_sign_nocluster(
                df=df,
                target_col=target_col,
                score_col=score_col,
                center_value=center_value,
                scale=scale,
                decision_threshold=decision_threshold,
            )
        except Exception as e:
            if print_:
                print(f"[ERROR] {target_col}: {e}")
            continue

        if print_:
            d = decimals
            print(f"Régression logistique (signe) du score_stmt -> {asset_prefix} à horizon {h} min :")
            print(f"  alpha = {r['alpha']:.{d}f} (p={r['p_alpha']:.{max(1,d-2)}g}, "
                  f"IC95% [{r['alpha_ci95'][0]:.{d}f}, {r['alpha_ci95'][1]:.{d}f}])")
            print(f"  beta  = {r['beta']:.{d}f} (p={r['p_beta']:.{max(1,d-2)}g}, "
                  f"IC95% [{r['beta_ci95'][0]:.{d}f}, {r['beta_ci95'][1]:.{d}f}])")
            print(f"  odds(beta) = {r['odds_beta']:.{d}f} "
                  f"(IC95% [{r['odds_beta_ci95'][0]:.{d}f}, {r['odds_beta_ci95'][1]:.{d}f}])")
            print(f"  Hit-rate={r['hit_rate']:.{d}f}, Brier={r['brier']:.{d}f}, AUC={r['auc']:.{d}f}, n={r['n']}, SE={r['se_type']}")
            print("-" * 72)

        rows.append({
            "horizon_min": h,
            "alpha": r["alpha"], "alpha_p": r["p_alpha"],
            "alpha_ci_low": r["alpha_ci95"][0], "alpha_ci_high": r["alpha_ci95"][1],
            "beta": r["beta"], "beta_p": r["p_beta"],
            "beta_ci_low": r["beta_ci95"][0], "beta_ci_high": r["beta_ci95"][1],
            "odds_beta": r["odds_beta"], "odds_beta_ci_low": r["odds_beta_ci95"][0],
            "odds_beta_ci_high": r["odds_beta_ci95"][1],
            "hit_rate": r["hit_rate"], "brier": r["brier"], "auc": r["auc"],
            "n": r["n"], "se_type": r["se_type"], "target_col": target_col,
        })

    return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

def tail_event_logit(df, target_col, score_col="score_stmt", x_bps=25.0):
    d = df[[target_col, score_col]].dropna().copy()
    y = (d[target_col].abs() > x_bps).astype(int).values
    x = (d[score_col].astype(float) - 5.0) / 5.0
    X = sm.add_constant(x.values, has_constant="add")
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
    from sklearn.metrics import roc_auc_score
    p = res.predict(X); auc = roc_auc_score(y, p)
    return {"n": int(len(d)), "beta": float(res.params[1]), "p_beta": float(res.pvalues[1]),
            "odds": float(np.exp(res.params[1])), "auc": float(auc), "res": res}

import statsmodels.api as sm

def logit_sign_extremes(df, target_col, score_col="score_stmt", lower=0.2, upper=0.8):
    d = df[[target_col, score_col]].dropna().copy()
    q1, q2 = d[score_col].quantile([lower, upper])
    d = d[(d[score_col] <= q1) | (d[score_col] >= q2)].copy()
    if d.empty: raise ValueError("Pas assez d'observations extrêmes.")
    x = (d[score_col].astype(float) - 5.0) / 5.0  # ~[-1,1]
    y = (d[target_col].astype(float) > 0).astype(int).values
    X = sm.add_constant(x.values, has_constant="add")
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
    from sklearn.metrics import roc_auc_score
    p = res.predict(X); auc = roc_auc_score(y, p)
    return {"n": int(len(d)), "beta": float(res.params[1]), "p_beta": float(res.pvalues[1]),
            "odds": float(np.exp(res.params[1])), "auc": float(auc), "res": res}
