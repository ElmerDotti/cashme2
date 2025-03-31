import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def stratified_sampling(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    """Executa amostragem estratificada com base no target."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, _ in splitter.split(df, df["Target"]):
        sampled_df = df.iloc[train_idx]
        return sampled_df


def reduce_dimensionality(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Aplica PCA para reduzir dimensionalidade."""
    features = df.drop(columns=["Target"])
    numeric_cols = features.select_dtypes(include=["float64", "int64"]).columns
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features[numeric_cols])
    reduced_df = pd.DataFrame(reduced, columns=[f"PCA_{i+1}" for i in range(n_components)], index=df.index)
    reduced_df["Target"] = df["Target"].values
    return reduced_df


def cluster_data(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Segmenta dados usando KMeans com base em componentes principais."""
    features = df.drop(columns=["Target"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(features)
    return df
