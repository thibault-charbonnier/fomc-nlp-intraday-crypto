# A Statistical Study of High-Frequency Crypto Reactions to FOMC NLP Tone

**Author:** Thibault Charbonnier  
**Date:** November 2025  
**Contact:** thibault.charbonnier@ensae.fr

> 🔎 The full empirical study, methods, figures and discussion are in the Jupyter notebook:
> **`demo.ipynb`** (see repository root).

---

## Abstract

This empirical study examines various cryptocurrencies' intraday reactions to U.S. monetary-policy communication and reaches a conclusion consistent with recent literature: crypto returns appear **orthogonal to monetary policy at high frequency**. Using an NLP-based tone score on FOMC statements and press-conference transcripts to proxy policy stance, we find **no statistically significant predictive power** for the sign nor magnitude of returns (10–60m horizons). However, distributions differ in the **tails**: **most dovish** vs **most hawkish** events display **significant asymmetry** (heavier right vs left tails). Results should be read with caution given **structural data limits** (few FOMC events; limited minute-level crypto history).

---

## Project Architecture

![Pipeline Schema](image/Pipeline%20schema.png)

---

## Data & Reproducibility

FOMC documents are fetched from the Federal Reserve websites; structure varies by period (fallbacks implemented).

Crypto minute data are sourced from exchange archives; coverage varies around listing dates.

See demo.ipynb for the full pipeline and analysis steps (bucketing, skew/Δ-quantiles, OLS/logit, plots).

---

## Citation

If you use this project, please site :
Charbonnier, T. (2025). A Statistical Study of High-Frequency Crypto Reactions to FOMC NLP Tone. GitHub repository.

---

### Contact & Issues

Questions / bugs: open an issue on GitHub or email thibault.charbonnier@ensae.fr or linkedin https://www.linkedin.com/in/thibault-charbonnier