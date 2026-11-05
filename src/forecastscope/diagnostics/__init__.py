"""Optional time-series diagnostics delegated to named scientific implementations.

Results describe the supplied sequence. Retrospective use is distinct from
prefix-only forecasting features; statistical assumptions remain method-specific.
"""
from __future__ import annotations

from . import autocorrelation
from . import causality
from . import changepoints
from . import distribution
from . import filters
from . import fractal
from . import nonlinear
from . import seasonality
from . import stationarity
from . import volatility
from .causality import (
    CointegrationResult,
    CrossCorrelation,
    GrangerResult,
    causality_report,
    cointegration,
    cross_correlation,
    granger_causality,
)
from .nonlinear import (
    ChaosMeasures,
    RQA,
    chaos_measures,
    iaaft_surrogate,
    nonlinear_report,
    recurrence_quantification,
    time_delay_embed,
    zero_one_test,
)
from .fractal import (
    FractalDimension,
    HurstResult,
    MultifractalSpectrum,
    dcca_coefficient,
    fractal_dimension,
    fractal_report,
    hurst,
    mfdfa,
)
from .distribution import (
    ComplexitySummary,
    DistributionSummary,
    catch22_features,
    complexity,
    distribution_report,
    kde_curve,
    qq_points,
    summary,
)
from .changepoints import (
    ChangePoints,
    CusumResult,
    RegimeFit,
    binseg,
    changepoints_report,
    cusum_stability,
    markov_regimes,
    pelt,
)
from .filters import (
    EMDResult,
    FilterResult,
    Scalogram,
    baxter_king,
    ceemdan,
    christiano_fitzgerald,
    cwt_scalogram,
    emd,
    filters_report,
    hodrick_prescott,
)
from .autocorrelation import (
    Correlogram,
    PortmanteauTest,
    acf,
    autocorrelation_report,
    bartlett_band,
    box_pierce,
    durbin_watson,
    lag_plot_pairs,
    ljung_box,
    pacf,
)
from .seasonality import (
    Decomp,
    SeasonalSubseries,
    Spectrum,
    mstl_decompose,
    periodogram,
    seasonal_strength,
    seasonal_subseries,
    seasonality_report,
    stl_decompose,
    welch,
)
from .volatility import (
    ArchTest,
    BoxCoxResult,
    GarchFit,
    arch_lm,
    box_cox,
    garch,
    rolling_volatility,
    volatility_report,
)
from .stationarity import (
    TestResult,
    adf,
    combined_verdict,
    dfgls,
    kpss,
    ndiffs,
    nsdiffs,
    phillips_perron,
    stationarity_report,
    zivot_andrews,
)

__all__ = [
    "stationarity",
    "autocorrelation",
    "seasonality",
    "filters",
    "changepoints",
    "volatility",
    "distribution",
    "fractal",
    "nonlinear",
    "causality",
    # stationarity
    "TestResult",
    "adf",
    "kpss",
    "phillips_perron",
    "dfgls",
    "zivot_andrews",
    "ndiffs",
    "nsdiffs",
    "combined_verdict",
    "stationarity_report",
    # autocorrelation
    "Correlogram",
    "PortmanteauTest",
    "acf",
    "pacf",
    "bartlett_band",
    "ljung_box",
    "box_pierce",
    "durbin_watson",
    "lag_plot_pairs",
    "autocorrelation_report",
    # seasonality (seasonal_strength lives here as the canonical home)
    "Spectrum",
    "Decomp",
    "SeasonalSubseries",
    "periodogram",
    "welch",
    "seasonal_strength",
    "stl_decompose",
    "mstl_decompose",
    "seasonal_subseries",
    "seasonality_report",
    # filters
    "FilterResult",
    "EMDResult",
    "Scalogram",
    "hodrick_prescott",
    "baxter_king",
    "christiano_fitzgerald",
    "emd",
    "ceemdan",
    "cwt_scalogram",
    "filters_report",
    # changepoints
    "ChangePoints",
    "CusumResult",
    "RegimeFit",
    "pelt",
    "binseg",
    "cusum_stability",
    "markov_regimes",
    "changepoints_report",
    # volatility
    "ArchTest",
    "GarchFit",
    "BoxCoxResult",
    "arch_lm",
    "garch",
    "box_cox",
    "rolling_volatility",
    "volatility_report",
    # distribution
    "DistributionSummary",
    "ComplexitySummary",
    "summary",
    "kde_curve",
    "qq_points",
    "complexity",
    "catch22_features",
    "distribution_report",
    # fractal
    "HurstResult",
    "MultifractalSpectrum",
    "FractalDimension",
    "hurst",
    "mfdfa",
    "fractal_dimension",
    "dcca_coefficient",
    "fractal_report",
    # nonlinear
    "ChaosMeasures",
    "RQA",
    "time_delay_embed",
    "chaos_measures",
    "recurrence_quantification",
    "iaaft_surrogate",
    "zero_one_test",
    "nonlinear_report",
    # causality
    "CrossCorrelation",
    "GrangerResult",
    "CointegrationResult",
    "cross_correlation",
    "granger_causality",
    "cointegration",
    "causality_report",
]
