"""Distribution, normality, and entropy/complexity: the value-distribution and predictability diagnostics.

Two facets of "what kind of values does this series take, and how predictable is its pattern?": (1) the
DISTRIBUTION of the values (shape, tails, normality) and (2) the COMPLEXITY of the ordering (entropy measures
that quantify regularity vs randomness, and a nonlinearity test). Together they answer whether Gaussian
assumptions hold and how much exploitable structure the sequence carries.

Methods / references (verified 2026-07-04):
  * Summary moments - mean, variance, skewness (asymmetry), kurtosis (tail-heaviness). Standard.
  * KDE - Gaussian kernel density estimate. Parzen 1962, Ann. Math. Stat. 33(3):1065-1076,
    DOI 10.1214/aoms/1177704472. ``scipy.stats.gaussian_kde``.
  * Q-Q plot - sample vs theoretical (normal) quantiles. Wilk & Gnanadesikan 1968, Biometrika 55(1):1-17,
    DOI 10.1093/biomet/55.1.1. ``scipy.stats.probplot``.
  * Jarque-Bera - normality from skew+kurtosis; large-sample. Jarque & Bera 1980, Economics Letters
    6(3):255-259, DOI 10.1016/0165-1765(80)90024-5. ``scipy.stats.jarque_bera``.
  * Shapiro-Wilk - most powerful normality test for small/moderate n. Shapiro & Wilk 1965, Biometrika
    52(3-4):591-611, DOI 10.1093/biomet/52.3-4.591. ``scipy.stats.shapiro``.
  * Sample entropy - bias-corrected regularity (lower = more predictable). Richman & Moorman 2000,
    Am. J. Physiol. 278(6):H2039-H2049, DOI 10.1152/ajpheart.2000.278.6.H2039. ``antropy.sample_entropy``.
  * Permutation entropy - ordinal-pattern entropy, noise-robust. Bandt & Pompe 2002, PRL 88:174102,
    DOI 10.1103/PhysRevLett.88.174102. ``antropy.perm_entropy``.
  * Spectral entropy - Shannon entropy of the normalized PSD (flat spectrum = high). ``antropy.spectral_entropy``.
  * BDS test - i.i.d. vs hidden nonlinear structure via the correlation integral. Broock, Scheinkman,
    Dechert & LeBaron 1996, Econometric Reviews 15(3):197-235, DOI 10.1080/07474939608800353.
    ``statsmodels.tsa.stattools.bds``.
  * catch22 - 22 canonical features (Lubba et al. 2019, DOI 10.1007/s10618-019-00647-x). OPTIONAL: pycatch22
    needs a C toolchain to build; when unavailable the report records that honestly rather than faking it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def _finite(value) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


@dataclass(frozen=True)
class DistributionSummary:
    """Moments + normality verdicts + a KDE/Q-Q payload for the distribution panel."""

    n: int
    mean: float
    std: float
    skewness: float | None
    kurtosis_excess: float | None  # undefined without variation
    jarque_bera_stat: float | None
    jarque_bera_pvalue: float | None
    shapiro_stat: float | None
    shapiro_pvalue: float | None
    normal: bool | None           # None when the normalized tests are unavailable
    reference: str


def summary(x, alpha: float = 0.05) -> DistributionSummary:
    """Moments + Jarque-Bera + Shapiro-Wilk (n <= 5000) with a combined normality verdict."""
    from scipy import stats

    a = _clean(x)
    n = len(a)
    if n == 0:
        raise ValueError("Distribution summary requires finite observations")
    constant = np.ptp(a) == 0
    skew = kurt = jb_stat = jb_p = None
    sh_stat, sh_p = (None, None)
    shapiro_applicable = 3 <= n <= 5000
    if not constant:
        skew = _finite(stats.skew(a))
        kurt = _finite(stats.kurtosis(a, fisher=True))
        jb = stats.jarque_bera(a)
        jb_stat, jb_p = _finite(jb[0]), _finite(jb[1])
        if shapiro_applicable:
            sh = stats.shapiro(a)
            sh_stat, sh_p = _finite(sh[0]), _finite(sh[1])
    normal = None
    if jb_p is not None and (not shapiro_applicable or sh_p is not None):
        normal = bool(jb_p >= alpha and (sh_p is None or sh_p >= alpha))
    return DistributionSummary(
        n=n, mean=float(a[0]) if constant else float(np.mean(a)),
        std=0.0 if constant else float(np.std(a)), skewness=skew, kurtosis_excess=kurt,
        jarque_bera_stat=jb_stat, jarque_bera_pvalue=jb_p,
        shapiro_stat=sh_stat, shapiro_pvalue=sh_p, normal=normal,
        reference="Jarque & Bera 1980, DOI 10.1016/0165-1765(80)90024-5; Shapiro & Wilk 1965, DOI 10.1093/biomet/52.3-4.591",
    )


def kde_curve(x, num: int = 100) -> dict:
    """Gaussian KDE evaluated on a grid, for the histogram+density panel (Parzen 1962)."""
    from scipy import stats

    a = _clean(x)
    if len(a) < 2 or np.ptp(a) == 0:
        raise ValueError("KDE requires varying finite observations")
    kde = stats.gaussian_kde(a)
    grid = np.linspace(float(a.min()), float(a.max()), num)
    return {"grid": grid.tolist(), "density": kde(grid).tolist(),
            "reference": "Parzen 1962, Ann. Math. Stat. 33(3):1065-1076, DOI 10.1214/aoms/1177704472"}


def qq_points(x) -> dict:
    """Normal Q-Q points (theoretical vs ordered sample quantiles) + the fitted reference line."""
    from scipy import stats

    a = _clean(x)
    if not len(a):
        raise ValueError("Q-Q points require finite observations")
    if np.ptp(a) == 0:
        osm, osr = stats.probplot(a, dist="norm", fit=False)
        slope, intercept, r = 0.0, float(a[0]), None
    else:
        (osm, osr), (slope, intercept, r) = stats.probplot(a, dist="norm")
    return {"theoretical": osm.tolist(), "sample": osr.tolist(),
            "slope": float(slope), "intercept": float(intercept), "r": _finite(r) if r is not None else None,
            "reference": "Wilk & Gnanadesikan 1968, Biometrika 55(1):1-17, DOI 10.1093/biomet/55.1.1"}


@dataclass(frozen=True)
class ComplexitySummary:
    """Entropy/complexity measures + the BDS nonlinearity verdict (predictability of the ordering)."""

    sample_entropy: float | None
    permutation_entropy: float | None   # None when no variation supports the panel
    spectral_entropy: float | None
    bds_stat: float | None
    bds_pvalue: float | None
    iid: bool | None             # BDS verdict: True if not rejected (behaves i.i.d.)
    reference: str


def complexity(x, order: int = 3, bds_dim: int = 2, alpha: float = 0.05) -> ComplexitySummary:
    """Sample/permutation/spectral entropy + the BDS i.i.d.-vs-nonlinear test."""
    import antropy as ant
    from statsmodels.tsa.stattools import bds as _bds

    a = _clean(x)
    varying = len(a) > 1 and np.ptp(a) > 0
    samp = _finite(ant.sample_entropy(a)) if varying else None
    perm = _finite(ant.perm_entropy(a, order=order, normalize=True)) if varying else None
    spec = _finite(ant.spectral_entropy(a, sf=1.0, method="welch", normalize=True)) if varying else None
    bds_stat = bds_p = None
    iid = None
    if varying and len(a) >= 20:
        try:
            stat, pval = _bds(a, max_dim=bds_dim)
            bds_stat = _finite(np.ravel(stat)[-1])
            bds_p = _finite(np.ravel(pval)[-1])
            if bds_stat is None or bds_p is None:
                bds_stat = bds_p = None
                raise ValueError("BDS returned nonfinite statistics")
            iid = bool(bds_p >= alpha)
        except Exception:  # noqa: BLE001 - BDS can fail on degenerate input; record as unavailable
            warnings.warn("BDS diagnostic unavailable", RuntimeWarning, stacklevel=2)
    return ComplexitySummary(
        sample_entropy=samp, permutation_entropy=perm, spectral_entropy=spec,
        bds_stat=bds_stat, bds_pvalue=bds_p, iid=iid,
        reference="Richman & Moorman 2000, DOI 10.1152/ajpheart.2000.278.6.H2039; Bandt & Pompe 2002, DOI 10.1103/PhysRevLett.88.174102; BDS: Broock et al. 1996, DOI 10.1080/07474939608800353",
    )


def catch22_features(x) -> dict:
    """The canonical catch22 feature set, in its catch24 form (+ mean and std). Lubba et al. 2019.

    This is the PRINCIPLED answer to "extract many features": 22 features distilled from the ~7700 of hctsa
    by selecting for classification performance and LOW REDUNDANCY - so it carries the information of a large
    feature bank without the shotgun. We bake it per case (it is pinned in requirements-precompute.txt;
    pycatch22 ships prebuilt wheels, no C toolchain needed). Kept import-guarded so the pipeline still runs
    on a slim install, and if it is ever absent the report SAYS so rather than fabricating values.
    """
    try:
        import pycatch22
    except ImportError:
        return {"available": False,
                "reason": "pycatch22 not installed (pin it: requirements-precompute.txt)",
                "reference": "Lubba et al. 2019, Data Min. Knowl. Disc. 33(6):1821-1852, DOI 10.1007/s10618-019-00647-x"}
    a = _clean(x)
    if len(a) < 2 or np.ptp(a) == 0:
        return {"available": False, "reason": "constant sequence has no normalized catch22 features",
                "reference": "Lubba et al. 2019, DOI 10.1007/s10618-019-00647-x"}
    res = pycatch22.catch22_all(a.tolist(), catch24=True)
    return {"available": True,
            "features": dict(zip(res["names"], [float(v) for v in res["values"]])),
            "reference": "Lubba et al. 2019, DOI 10.1007/s10618-019-00647-x"}


def distribution_report(x, order: int = 3) -> dict:
    """Run the distribution + complexity panel and return a JSON-ready report."""
    a = _clean(x)
    s = summary(a)
    c = complexity(a, order=order)
    try:
        kde = kde_curve(a)
    except (ValueError, np.linalg.LinAlgError) as exc:
        kde = {"available": False, "reason": str(exc)}
    return {
        "n": len(a),
        "summary": {"mean": s.mean, "std": s.std, "skewness": s.skewness,
                    "kurtosis_excess": s.kurtosis_excess, "jarque_bera_pvalue": s.jarque_bera_pvalue,
                    "shapiro_pvalue": s.shapiro_pvalue, "normal": s.normal, "reference": s.reference},
        "kde": kde,
        "qq": qq_points(a),
        "complexity": {"sample_entropy": c.sample_entropy, "permutation_entropy": c.permutation_entropy,
                       "spectral_entropy": c.spectral_entropy, "bds_pvalue": c.bds_pvalue, "iid": c.iid,
                       "reference": c.reference},
        "catch22": catch22_features(a),
    }
