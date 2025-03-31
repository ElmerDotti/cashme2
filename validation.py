import pandas as pd
import numpy as np


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calcula o PSI entre duas distribuições."""
    def _distribution(values):
        quantiles = np.linspace(0, 1, buckets + 1)
        bins = np.quantile(values, quantiles)
        return pd.cut(values, bins=bins, include_lowest=True).value_counts(normalize=True)

    expected_dist = _distribution(expected)
    actual_dist = _distribution(actual)
    psi = np.sum((expected_dist - actual_dist) * np.log(expected_dist / actual_dist + 1e-5))
    return psi


def validate_feature_stability(df_current: pd.DataFrame, df_external: pd.DataFrame) -> pd.DataFrame:
    """Compara estabilidade das features selecionadas via PSI."""
    results = []
    for col in df_current.columns.difference(["Target"]):
        if col in df_external.columns:
            psi = population_stability_index(df_current[col], df_external[col])
            results.append((col, psi))
    return pd.DataFrame(results, columns=["Feature", "PSI"]).sort_values("PSI", ascending=False)
