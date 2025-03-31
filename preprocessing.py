import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza inicial do dataset:
    - Conversão de tipos
    - Interpolação de nulos para variáveis numéricas
    - Preenchimento básico para booleanos e categóricos
    """
    df = df.copy()

    # Identifica colunas por tipo
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.difference(["Target"]).tolist()

    # Interpolação de nulos para colunas numéricas
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", axis=0, limit_direction="both")

    # Preenchimento simples para booleanos e objetos
    df[bool_cols] = df[bool_cols].fillna(False)
    df[object_cols] = df[object_cols].fillna("missing")

    return df
