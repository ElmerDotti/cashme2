from app.data_loader import load_data
from app.preprocessing import preprocess_data
from app.feature_engineering import generate_lag_features
from app.visualization import (
    plot_distributions,
    plot_boxplots,
    plot_correlation_matrix
)
from app.modeling import (
    stratified_sampling,
    reduce_dimensionality,
    cluster_data
)
from app.feature_selection import select_features_with_lgbm
from app.utils import describe_features

import pandas as pd

# Caminhos
X_PATH = "data/X.csv"
Y_PATH = "data/y.csv"
OUTPUT_DIR = "outputs"

def main():
    print("🔄 Carregando dados...")
    df = load_data(X_PATH, Y_PATH)

    print("🧼 Pré-processando dados...")
    df_clean = preprocess_data(df)

    print("🧪 Engenharia de features...")
    df_features = generate_lag_features(df_clean)

    print("📊 Visualizações iniciais...")
    plot_distributions(df_features, OUTPUT_DIR)
    plot_boxplots(df_features, OUTPUT_DIR)
    plot_correlation_matrix(df_features, OUTPUT_DIR)

    print("📉 Amostragem e redução de dimensionalidade...")
    df_sampled = stratified_sampling(df_features)
    df_reduced = reduce_dimensionality(df_sampled)
    df_segmented = cluster_data(df_reduced)

    print("🎯 Seleção de variáveis com otimização...")
    df_final, selected_features = select_features_with_lgbm(df_segmented, OUTPUT_DIR)

    print("🧾 Descrição das variáveis selecionadas:")
    descriptions = describe_features(df_final[selected_features])
    for name, desc in descriptions.items():
        print(f"- {name}: {desc}")

    print("\n✅ Processo finalizado com sucesso!")
    print(f"📥 {len(selected_features)} features salvas em {OUTPUT_DIR}/selected_features.csv")
    print(f"📈 SHAP plot salvo em {OUTPUT_DIR}/shap_values.png")

if __name__ == "__main__":
    main()
