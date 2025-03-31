import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


def calculate_entropy(series: pd.Series) -> float:
    """Calcula a entropia de uma série com base na frequência dos valores."""
    counts = series.value_counts(normalize=True)
    return entropy(counts)


def generate_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas: lag, variação normalizada e entropia."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["Target"])

    entropies = {}
    scaler = MinMaxScaler()

    for col in numeric_cols:
        # Lag de 1
        lag_col = f"{col}_lag1"
        df[lag_col] = df[col].shift(1).fillna(method="bfill")

        # Razão de variação normalizada
        var_col = f"{col}_var"
        diff = df[col] - df[lag_col]
        norm_diff = (diff - diff.min()) / (diff.max() - diff.min())
        df[var_col] = norm_diff

        # Entropia (valor único por coluna, repetido para o dataset)
        ent_col = f"{col}_ent"
        ent_val = calculate_entropy(df[col])
        df[ent_col] = ent_val

        # Score normalizado final entre 0 e 1
        score_col = f"{col}_score"
        df[score_col] = scaler.fit_transform(df[[col]])  # usa apenas a original

    return df
