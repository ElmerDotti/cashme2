# 🧠 CashMe - Desafio de Feature Selection

Este projeto resolve um desafio técnico envolvendo seleção de variáveis em um dataset de alta dimensionalidade, com aplicação de técnicas avançadas de pré-processamento, engenharia de variáveis, amostragem, redução de dimensionalidade, modelagem com otimização e análise de interpretabilidade via SHAP.

---

## 🗂️ Estrutura do Projeto

cashme_feature_selection/
 ├── app/ │ 
  ├── config.py │ 
  ├── data_loader.py │ 
  ├── preprocessing.py │ 
  ├── feature_engineering.py │ 
  ├── modeling.py │ 
  ├── visualization.py │ 
  ├── feature_selection.py │ 
  ├── utils.py │ 
  └── main.py 
 ├── data/ │
  ├── X.csv │ 
  └── y.csv 
 ├── outputs/ │
  ├── shap_values.png │ 
  ├── selected_features.csv │ 
 └── ... 
 ├── requirements.txt 
 └── README.md


---

## 🚀 Como executar

1. Clone este repositório e navegue até a pasta do projeto:
   ```bash
   git clone <repo-url>
   cd cashme_feature_selection

