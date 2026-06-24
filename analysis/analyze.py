#!/usr/bin/env python3
"""
PulseBoard statistical analysis.

Dataset
-------
Source URL : https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv
Downloaded : 2026-06-24 (UTC)
SHA-256    : 714f8a23e9d7429052a1e9b45d20b9adeb64f46c067ed46320e68182e4607b6d
Rows       : 1704 (country x 5-year-interval observations, 1952-2007)
Fields     : country, year, pop, continent, lifeExp, gdpPercap
PII        : none -- country/continent level aggregates only.

This script:
  1. Verifies the raw CSV checksum (fails loudly if the source has drifted).
  2. Cleans/types the data and writes data/gapminder.json for the front-end.
  3. Fits OLS of lifeExp ~ log10(gdpPercap) (headline finding) and reports
     slope, 95% CI, p-value, R^2, n.
  4. Runs a one-way ANOVA of lifeExp across continents (secondary finding)
     plus per-continent mean lifeExp with 95% CIs (t-distribution).
  5. Writes every computed number to data/findings.json. The dashboard reads
     ONLY this file for its headline stats -- nothing is hand-typed into the
     front-end.

Reproduce with:
  python3.11 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  python analysis/analyze.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_CSV = os.path.join(ROOT, "data", "gapminder_raw.csv")
OUT_DATA = os.path.join(ROOT, "data", "gapminder.json")
OUT_FINDINGS = os.path.join(ROOT, "data", "findings.json")

SOURCE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
EXPECTED_SHA256 = "714f8a23e9d7429052a1e9b45d20b9adeb64f46c067ed46320e68182e4607b6d"
DOWNLOAD_DATE = "2026-06-24"


def verify_checksum(path: str, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        sys.exit(
            f"CHECKSUM MISMATCH for {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"Refusing to analyze data that does not match the recorded source."
        )
    print(f"[ok] checksum verified for {os.path.basename(path)}: {actual}")


def ols_with_ci(x: np.ndarray, y: np.ndarray):
    """OLS of y on x (single predictor + intercept) via statsmodels.
    Returns dict with slope, intercept, 95% CI for slope, p-value, R^2, n."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = float(model.params[1])
    intercept = float(model.params[0])
    ci_lo, ci_hi = model.conf_int(alpha=0.05)[1]
    p_value = float(model.pvalues[1])
    r_squared = float(model.rsquared)
    n = int(model.nobs)
    se = float(model.bse[1])
    return {
        "slope": slope,
        "intercept": intercept,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "se": se,
        "p_value": p_value,
        "r_squared": r_squared,
        "n": n,
    }


def mean_ci95(values: np.ndarray):
    n = len(values)
    mean = float(np.mean(values))
    sem = stats.sem(values)
    if n > 1 and sem > 0:
        h = sem * stats.t.ppf(0.975, n - 1)
    else:
        h = 0.0
    return {
        "mean": mean,
        "ci95_low": mean - float(h),
        "ci95_high": mean + float(h),
        "n": int(n),
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
    }


def main():
    print("=== PulseBoard analysis ===")
    verify_checksum(RAW_CSV, EXPECTED_SHA256)

    df = pd.read_csv(RAW_CSV)

    # --- clean / type ---
    df["country"] = df["country"].astype(str).str.strip()
    df["continent"] = df["continent"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df["pop"] = df["pop"].astype(float)
    df["lifeExp"] = df["lifeExp"].astype(float)
    df["gdpPercap"] = df["gdpPercap"].astype(float)
    df = df.dropna(subset=["country", "continent", "year", "pop", "lifeExp", "gdpPercap"])
    df = df[(df["gdpPercap"] > 0) & (df["lifeExp"] > 0) & (df["pop"] > 0)]
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    print(f"[ok] cleaned dataset: {len(df)} rows, "
          f"{df['country'].nunique()} countries, "
          f"{df['continent'].nunique()} continents, "
          f"years {df['year'].min()}-{df['year'].max()}")

    # --- write compact dataset for the front-end ---
    records = df[["country", "continent", "year", "pop", "lifeExp", "gdpPercap"]].to_dict(
        orient="records"
    )
    with open(OUT_DATA, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    print(f"[ok] wrote {OUT_DATA} ({os.path.getsize(OUT_DATA)} bytes, {len(records)} records)")

    # --- headline finding: OLS lifeExp ~ log10(gdpPercap), full dataset ---
    log_gdp = np.log10(df["gdpPercap"].values)
    life = df["lifeExp"].values
    headline_full = ols_with_ci(log_gdp, life)
    print(
        f"[ok] headline OLS (full data): slope={headline_full['slope']:.4f} "
        f"95% CI [{headline_full['ci95_low']:.4f}, {headline_full['ci95_high']:.4f}] "
        f"p={headline_full['p_value']:.3e} R2={headline_full['r_squared']:.4f} "
        f"n={headline_full['n']}"
    )

    # also fit controlling for year (centered) as a robustness check, reported
    # separately -- not the headline, but adds rigor.
    year_c = (df["year"] - df["year"].mean()).values
    X_multi = sm.add_constant(np.column_stack([log_gdp, year_c]))
    model_multi = sm.OLS(life, X_multi).fit()
    headline_controlled = {
        "slope_loggdp": float(model_multi.params[1]),
        "ci95_low_loggdp": float(model_multi.conf_int(alpha=0.05)[1][0]),
        "ci95_high_loggdp": float(model_multi.conf_int(alpha=0.05)[1][1]),
        "p_value_loggdp": float(model_multi.pvalues[1]),
        "slope_year": float(model_multi.params[2]),
        "p_value_year": float(model_multi.pvalues[2]),
        "r_squared": float(model_multi.rsquared),
        "n": int(model_multi.nobs),
    }
    print(
        f"[ok] year-controlled OLS: slope(log10 gdp)={headline_controlled['slope_loggdp']:.4f} "
        f"p={headline_controlled['p_value_loggdp']:.3e} R2={headline_controlled['r_squared']:.4f}"
    )

    # --- secondary finding: one-way ANOVA of lifeExp across continents ---
    continents = sorted(df["continent"].unique().tolist())
    groups = [df.loc[df["continent"] == c, "lifeExp"].values for c in continents]
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"[ok] one-way ANOVA across continents: F={f_stat:.4f} p={anova_p:.3e}")

    per_continent = {}
    for c, g in zip(continents, groups):
        per_continent[c] = mean_ci95(g)

    # pairwise Welch t-tests (continent vs continent) for richness
    pairwise = []
    for i in range(len(continents)):
        for j in range(i + 1, len(continents)):
            a, b = continents[i], continents[j]
            ga, gb = groups[i], groups[j]
            t_stat, p_val = stats.ttest_ind(ga, gb, equal_var=False)
            pairwise.append(
                {
                    "a": a,
                    "b": b,
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "mean_diff": float(np.mean(ga) - np.mean(gb)),
                }
            )

    # --- per-continent counts & year range for UI ---
    meta = {
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()),
        "continents": continents,
        "years": sorted(df["year"].unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "gdp_min": float(df["gdpPercap"].min()),
        "gdp_max": float(df["gdpPercap"].max()),
        "life_min": float(df["lifeExp"].min()),
        "life_max": float(df["lifeExp"].max()),
    }

    findings = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "source_url": SOURCE_URL,
            "sha256": EXPECTED_SHA256,
            "download_date": DOWNLOAD_DATE,
            "license_note": "Gapminder Foundation data, redistributed via plotly/datasets (public, no auth, no PII).",
        },
        "meta": meta,
        "headline": {
            "model": "lifeExp ~ const + slope * log10(gdpPercap)",
            "description": "OLS regression of life expectancy on log10(GDP per capita), full dataset, all years and continents pooled.",
            **headline_full,
        },
        "headline_year_controlled": {
            "model": "lifeExp ~ const + slope_loggdp * log10(gdpPercap) + slope_year * (year - mean(year))",
            "description": "Robustness check: same regression controlling for year.",
            **headline_controlled,
        },
        "anova": {
            "model": "one-way ANOVA: lifeExp ~ continent",
            "f_stat": float(f_stat),
            "p_value": float(anova_p),
            "groups": continents,
        },
        "per_continent_lifeExp": per_continent,
        "pairwise_ttests": pairwise,
    }

    with open(OUT_FINDINGS, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"[ok] wrote {OUT_FINDINGS}")
    print("=== analysis complete ===")


if __name__ == "__main__":
    main()
