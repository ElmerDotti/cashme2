import pandas as pd
from pathlib import Path


def load_data(x_path: str, y_path: str) -> pd.DataFrame:
    """Carrega os datasets X e y, unificando-os pelo índice."""
    x_df = pd.read_csv(x_path, index_col=0)
    y_df = pd.read_csv(y_path, index_col=0)

    # Garantir que os índices estão alinhados
    df = x_df.join(y_df, how="inner")
    return df
