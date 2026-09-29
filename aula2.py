#Propósito: Esta aula aplica o `LinearSVC` a um conjunto de dados real (`tracking.csv`) para prever se um usuário comprou um produto. 
#Explora a manipulação de dados com `pandas`, a divisão manual do conjunto de dados em treino e teste, e introduz a função `train_test_split` do `scikit-learn` para uma divisão mais robusta,
#incluindo o uso de `stratify` para manter a proporção das classes.

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Carregamento dos dados de tracking
uri_tracking = 'https://gist.githubusercontent.com/guilhermesilveira/b9dd8e4b62b9e22ebcb9c8e89c271de4/raw/c69ec4b708fba03c445397b6a361db4345c83d7a/tracking.csv'
dados = pd.read_csv(uri_tracking)

print("Primeiras 5 linhas do dataset de tracking:")
print(dados.head(5))

# Separação das features (x) e do target (y)
y = dados['comprou']
x = dados[['inicial', 'palestras', 'contato', 'patrocinio']]

print("\nFeatures (x) utilizadas:")
print(x.head())

print(f"\nShape do dataset: {dados.shape}")

# Divisão manual do dataset em treino e teste
tamanho_treino = int(0.75 * dados.shape[0]) # 75% para treino

treino_x_manual = x[:tamanho_treino]
treino_y_manual = y[:tamanho_treino]
teste_x_manual = x[tamanho_treino:]
teste_y_manual = y[tamanho_treino:]

# Treino e avaliação com divisão manual
modelo_manual = LinearSVC()
modelo_manual.fit(treino_x_manual, treino_y_manual)

previsoes_manual = modelo_manual.predict(teste_x_manual)
taxa_acerto_manual = accuracy_score(previsoes_manual, teste_y_manual)
print(f"\nAcurácia com divisão manual: {taxa_acerto_manual:.4f}")

# Divisão do dataset em treino e teste usando train_test_split
# SEED para reprodutibilidade e stratify para manter proporção das classes
SEED = 84364

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, stratify=y)

print(f"\nProporção das classes no treino (y_train):")
print(train_y.value_counts())
print(f"\nProporção das classes no teste (y_test):")
print(test_y.value_counts())

# Treino e avaliação com train_test_split
modelo_split = LinearSVC()
modelo_split.fit(train_x, train_y)

previsoes_split = modelo_split.predict(test_x)
taxa_acerto_split = accuracy_score(previsoes_split, test_y)
print(f"\nAcurácia com train_test_split (com stratify): {taxa_acerto_split:.4f}")
