#Propósito: Esta aula aprofunda a comparação entre diferentes modelos de classificação: `SVC`, `DummyClassifier` (modelo base para comparação) e `DecisionTreeClassifier`. 
#Realiza engenharia de features adicionais (conversão de milhas para km, cálculo de idade do carro) e demonstra a visualização da árvore de decisão gerada, o que ajuda a entender como o modelo toma suas decisões.

import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import graphviz
import matplotlib.pyplot as plt

# Carregamento dos dados de preços de carros
dados = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/dd7ba8142321c2c8aaa0ddd6c8862fcc/raw/e694a9b43bae4d52b6c990a5654a193c3f870750/precos.csv")

print("Primeiras 5 linhas do dataset original:")
print(dados.head())

# Engenharia de features:
# 1. Converter milhas_por_ano para km_por_ano
dados['km_por_ano'] = dados['milhas_por_ano'] * 1.60934

# 2. Calcular a idade do carro
dados['idade'] = datetime.today().year - dados['ano_do_modelo']

# 3. Remover features originais que não serão mais usadas
dados.drop(['milhas_por_ano', 'ano_do_modelo'], axis=1, inplace=True)

print("\nDataset após engenharia de features:")
print(dados.head())

# Separação das features (x) e do target (y)
x = dados[['preco', 'idade', 'km_por_ano']]
y = dados['vendido']

# Divisão do dataset em treino e teste
SEED = 0
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, stratify=y)

print(f"\nTamanho do conjunto de treino: {len(raw_train_x)} amostras")
print(f"Tamanho do conjunto de teste: {len(raw_test_x)} amostras")

# --- Modelo SVC (Support Vector Classifier) ---
print("\n--- Avaliação do Modelo SVC ---")
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x_scaled = scaler.transform(raw_train_x)
test_x_scaled = scaler.transform(raw_test_x)

modelo_svc = SVC(gamma='auto')
modelo_svc.fit(train_x_scaled, train_y)
previsoes_svc = modelo_svc.predict(test_x_scaled)

accuracy_svc = accuracy_score(test_y, previsoes_svc)
print(f"Acurácia do SVC: {accuracy_svc:.4f}")

# --- Modelo Dummy Classifier (Baseline) ---
print("\n--- Avaliação do Modelo DummyClassifier (Baseline) ---")
# DummyClassifier serve como uma linha de base para comparar com modelos reais.
# 'stratified' gera previsões aleatórias respeitando as proporções de classe do conjunto de treino.
classificador_dummy = DummyClassifier(strategy='stratified')
classificador_dummy.fit(raw_train_x, train_y)
previsoes_dummy = classificador_dummy.predict(raw_test_x) # Usa raw_test_x pois Dummy não precisa de escalonamento

accuracy_dummy = accuracy_score(test_y, previsoes_dummy)
print(f"Acurácia do DummyClassifier: {accuracy_dummy:.4f}")

# --- Modelo Decision Tree Classifier ---
print("\n--- Avaliação do Modelo DecisionTreeClassifier ---")
# A Árvore de Decisão não precisa de escalonamento de features
modelo_arvore = DecisionTreeClassifier(max_depth=3, random_state=SEED) # Define profundidade máxima para evitar overfitting
modelo_arvore.fit(raw_train_x, train_y)
previsoes_arvore = modelo_arvore.predict(raw_test_x)

accuracy_arvore = accuracy_score(test_y, previsoes_arvore)
print(f"Acurácia da DecisionTreeClassifier: {accuracy_arvore:.4f}")

# Visualização da Árvore de Decisão
print("\nVisualizando a Árvore de Decisão (será gerado um arquivo .dot e exibido um gráfico)...")
structure = export_graphviz(modelo_arvore, 
                            out_file=None, # Não salva em arquivo, retorna como string
                            filled=True, 
                            rounded=True, 
                            feature_names=x.columns, 
                            class_names=['nao', 'sim'])
grafico = graphviz.Source(structure)

# O objeto `grafico` pode ser renderizado diretamente em ambientes como Jupyter/Colab.
# Para salvar em arquivo, você pode usar: grafico.render("decision_tree", view=True)
print(grafico) # Exibe o gráfico no output se o ambiente suportar.
