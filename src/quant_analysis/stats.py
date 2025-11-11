import pandas as pd
import numpy as np
from typing import Dict, Any, Sequence
import statsmodels.api as sm
from scipy.stats import skew
from sklearn.metrics import roc_auc_score


def run_regression(
        features: pd.DataFrame, 
        target: pd.Series
        ) -> Dict[str, Any]:
    """
    Run OLS regression of target on features.

    Params:
        features: pd.DataFrame
            DataFrame containing the feature variables.
        target: pd.Series
            Series containing the target variable.

    Returns:
        Dict[str, Any]
            OLS estimates, p-values, confidence intervals, R-squared, and number of observations.
    """

    df = pd.concat([features, target.rename("_y")], axis=1).dropna()
    X = sm.add_constant(df[features.columns].astype(float))
    y = df["_y"].astype(float)
    res = sm.OLS(y, X).fit(cov_type="HC1")
    ci = res.conf_int()
    return {
        "r2": float(res.rsquared),
        "n": int(res.nobs),
        "params": res.params.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "ci95": {k: (float(ci.loc[k,0]), float(ci.loc[k,1])) for k in res.params.index},
    }

def run_logistic_regression(
        features: pd.DataFrame, 
        target: pd.Series
        ) -> Dict[str, Any]:
    """
    Run logistic regression of target on features.

    Params:
        features: pd.DataFrame
            DataFrame containing the feature variables.
        target: pd.Series
            Series containing the binary target variable.

    Returns:
    """

    df = pd.concat([features, target.rename("_y")], axis=1).dropna()
    X = sm.add_constant(df[features.columns].astype(float))
    y = df["_y"].astype(int)
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="HC1", cov_kwds=None, maxiter=100, disp=0
    )
    ci = res.conf_int()
    auc = roc_auc_score(y, res.predict(X))
    out = {
        "n": int(res.nobs),
        "auc": float(auc),
        "hit_rate": float((res.predict()>.5).mean()),
        "params": res.params.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "ci95": {k: (float(ci.loc[k,0]), float(ci.loc[k,1])) for k in res.params.index},
    }
    if len(features.columns)==1:
        f = features.columns[0]
        out.update({
            "alpha": float(res.params["const"]),
            "beta": float(res.params[f]),
            "p_alpha": float(res.pvalues["const"]),
            "p_beta": float(res.pvalues[f]),
            "alpha_ci95": out["ci95"]["const"],
            "beta_ci95": out["ci95"][f],
        })
    return out

def _clean_array(x: pd.Series) -> np.ndarray:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    return a[np.isfinite(a)]

def _moment_skew(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.std(x, ddof=1) == 0:
        return np.nan
    return float(skew(x, bias=False))

def run_skew_bootstrap(
        distrib_A: pd.Series,
        distrib_B: pd.Series,
        B: int = 5000,
        conf_level: float = 0.95,
        ) -> Dict[str, Any]:
    """
    Compute the difference in skewness between two distributions and its confidence interval via bootstrap.

    Params:
        distrib_A: pd.Series
            First distribution of returns.
        distrib_B: pd.Series
            Second distribution of returns.
        B: int
            Number of bootstrap samples.
        conf_level: float
            Confidence level for the interval.

    Returns:
        Dict[str, Any]
            Distributions skewness, difference in skewness and its confidence interval.
    """
    skew_A = _moment_skew(_clean_array(distrib_A))
    skew_B = _moment_skew(_clean_array(distrib_B))
    skew_diff = skew_A - skew_B

    rng = np.random.default_rng()
    boot_diffs = np.empty(B)
    for b in range(B):
        sample_A = rng.choice(distrib_A, size=len(distrib_A), replace=True)
        sample_B = rng.choice(distrib_B, size=len(distrib_B), replace=True)
        boot_diffs[b] = pd.Series(sample_A).skew() - pd.Series(sample_B).skew()

    alpha = 1 - conf_level
    ci_lower = np.percentile(boot_diffs, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    res = {
        "skew_A": round(skew_A, 3),
        "skew_B": round(skew_B, 3),
        "skew_diff": round(skew_diff, 3),
        "ci_lower": round(float(ci_lower), 3),
        "ci_upper": round(float(ci_upper), 3),
    }
    return pd.Series(res)

def run_quantile_shift(
        distrib_A: pd.Series,
        distrib_B: pd.Series,
        B: int = 5000,
        conf_level: float = 0.95,
        q_levels: Sequence[float] = (0.1, 0.2, 0.5, 0.8, 0.9),
        ) -> Dict[str, Any]:
    """
    Perform quantile shift analysis between two distributions.
    The idea is to compute the difference in quantiles at specified levels.

    Params:
        distrib_A: pd.Series
            First distribution of returns.
        distrib_B: pd.Series
            Second distribution of returns.
        B: int
            Number of bootstrap samples.
        conf_level: float
            Confidence level for the interval.
        q_levels: List[float]
            List of quantile levels to analyze.
    """
    distrib_A = pd.Series(_clean_array(distrib_A))
    distrib_B = pd.Series(_clean_array(distrib_B))
    q_levels = np.asarray(q_levels, float)

    q_A = np.quantile(distrib_A, q_levels)
    q_B = np.quantile(distrib_B, q_levels)
    q_diff = q_A - q_B

    rng = np.random.default_rng()
    boot_diffs = {q: np.empty(B) for q in q_levels}
    for b in range(B):
        sample_A = rng.choice(distrib_A, size=len(distrib_A), replace=True)
        sample_B = rng.choice(distrib_B, size=len(distrib_B), replace=True)
        sample_A_series = pd.Series(sample_A)
        sample_B_series = pd.Series(sample_B)
        for q in q_levels:
            boot_diffs[q][b] = sample_A_series.quantile(q) - sample_B_series.quantile(q)
    
    alpha = 1 - conf_level
    ci_dict = {}
    for q in q_levels:
        ci_lower = np.percentile(boot_diffs[q], 100 * (alpha / 2))
        ci_upper = np.percentile(boot_diffs[q], 100 * (1 - alpha / 2))
        ci_dict[q] = (ci_lower, ci_upper)

    return pd.DataFrame({
        "q_level": q_levels,
        "q_A": q_A,
        "q_B": q_B,
        "q_diff": q_diff,
        "ci_lower": [ci_dict[q][0] for q in q_levels],
        "ci_upper": [ci_dict[q][1] for q in q_levels],
    })

