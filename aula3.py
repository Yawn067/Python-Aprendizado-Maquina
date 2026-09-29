#Propósito: Esta aula combina os conceitos de modelos lineares (`LinearSVC`) e não-lineares (`SVC`) para a predição de finalização de projetos. 
#Ela demonstra a aplicação de ambos os algoritmos, a necessidade de padronização dos dados para modelos sensíveis à escala (`StandardScaler`), 
#e visualiza as fronteiras de decisão para comparar o desempenho e a capacidade de cada modelo de lidar com dados complexos. O objetivo é entender quando cada tipo de modelo é mais adequado.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregamento dos dados de projetos
uri_projetos = 'https://gist.githubusercontent.com/guilhermesilveira/12291c548acaf544596795709020e3db/raw/325bdef098bd9cbc2189215b7e32e22f437f29f3/projetos.csv'
dados = pd.read_csv(uri_projetos)

print("Primeiras 5 linhas do dataset de projetos:")
print(dados.head())

# Engenharia de feature: transformar 'nao_finalizado' em 'finalizado'
dados['finalizado'] = dados['nao_finalizado'].map({0: 1, 1: 0})
print("\nDataset com a nova feature 'finalizado':")
print(dados.head())

# Visualização dos dados para entender a relação entre horas, preço e finalização
plt.figure(figsize=(10, 6))
sns.scatterplot(x='horas_esperadas', y='preco', data=dados, hue='finalizado')
plt.title('Horas Esperadas vs Preço por Status de Finalização')
plt.xlabel('Horas Esperadas')
plt.ylabel('Preço')
plt.show()

plt.figure(figsize=(12, 6))
sns.relplot(x='horas_esperadas', y='preco', data=dados, hue='finalizado', col='finalizado')
plt.suptitle('Relação entre Horas, Preço e Finalização (Separado por Status)', y=1.02)
plt.show()

# Filtrar dados para remover projetos com 'horas_esperadas' igual a 0
dados = dados.query('horas_esperadas > 0')
print("\nDataset após remover projetos com horas_esperadas = 0:")
print(dados.head())

# Separação das features (x) e do target (y)
x = dados[['horas_esperadas', 'preco']]
y = dados["finalizado"]

# Divisão do dataset em treino e teste
SEED = 20 # Para reprodutibilidade
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, stratify=y)

print(f"\nTamanho do conjunto de treino: {len(raw_train_x)} amostras")
print(f"Tamanho do conjunto de teste: {len(raw_test_x)} amostras")

# --- Avaliação do Modelo LinearSVC (Linear Support Vector Classifier) ---
print("\n--- Avaliação do Modelo LinearSVC ---")
modelo_linear = LinearSVC(random_state=SEED)
modelo_linear.fit(raw_train_x, train_y)

previsoes_linear = modelo_linear.predict(raw_test_x)
accuracy_linear = accuracy_score(previsoes_linear, test_y)
print(f"Acurácia do LinearSVC: {accuracy_linear:.4f}")
print("Observação: O LinearSVC pode não ser o melhor modelo aqui, pois o gráfico sugere uma fronteira de decisão não-linear.")

# Geração da superfície de decisão para visualização do LinearSVC
x_min_raw, x_max_raw = raw_test_x.horas_esperadas.min(), raw_test_x.horas_esperadas.max()
y_min_raw, y_max_raw = raw_test_x.preco.min(), raw_test_x.preco.max()

pixels = 100
eixo_x_raw = np.linspace(x_min_raw, x_max_raw, pixels)
eixo_y_raw = np.linspace(y_min_raw, y_max_raw, pixels)
xx_raw, yy_raw = np.meshgrid(eixo_x_raw, eixo_y_raw)
pontos_raw = np.c_[xx_raw.ravel(), yy_raw.ravel()]

z_linear = modelo_linear.predict(pontos_raw)
z_linear = z_linear.reshape(xx_raw.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx_raw, yy_raw, z_linear, alpha=0.3)
sns.scatterplot(x='horas_esperadas', y='preco', data=raw_test_x, hue=test_y, s=80, edgecolor='k')
plt.title('Superfície de Decisão do LinearSVC (Dados Originais)')
plt.xlabel('Horas Esperadas')
plt.ylabel('Preço')
plt.show()

# --- Avaliação do Modelo SVC (Support Vector Classifier) --- com escalonamento
print("\n--- Avaliação do Modelo SVC (Não-Linear) com Escalonamento ---")
# Padronização dos dados com StandardScaler
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x_scaled = scaler.transform(raw_train_x)
test_x_scaled = scaler.transform(raw_test_x)

modelo_svc = SVC(gamma='auto', random_state=SEED)
modelo_svc.fit(train_x_scaled, train_y)

previsoes_svc = modelo_svc.predict(test_x_scaled)
accuracy_svc = accuracy_score(previsoes_svc, test_y)
print(f"Acurácia do SVC (não-linear) após escalonamento: {accuracy_svc:.4f}")

# Geração da superfície de decisão para visualização do SVC
# Usamos os dados escalonados para o grid também
x_min_scaled, x_max_scaled = test_x_scaled[:, 0].min(), test_x_scaled[:, 0].max()
y_min_scaled, y_max_scaled = test_x_scaled[:, 1].min(), test_x_scaled[:, 1].max()

eixo_x_scaled = np.linspace(x_min_scaled, x_max_scaled, pixels)
eixo_y_scaled = np.linspace(y_min_scaled, y_max_scaled, pixels)
xx_scaled, yy_scaled = np.meshgrid(eixo_x_scaled, eixo_y_scaled)
pontos_scaled = np.c_[xx_scaled.ravel(), yy_scaled.ravel()]

z_svc = modelo_svc.predict(pontos_scaled)
z_svc = z_svc.reshape(xx_scaled.shape)

# Para o scatterplot, precisamos de um DataFrame dos dados de teste escalonados
data_test_scaled_df = pd.DataFrame(test_x_scaled, columns=['horas_esperadas_scaled', 'preco_scaled'])

plt.figure(figsize=(10, 6))
plt.contourf(xx_scaled, yy_scaled, z_svc, alpha=0.3)
sns.scatterplot(x='horas_esperadas_scaled', y='preco_scaled', data=data_test_scaled_df, hue=test_y, s=80, edgecolor='k')
plt.title('Superfície de Decisão do SVC (Não-Linear) com Dados Escalonados')
plt.xlabel('Horas Esperadas (Escalonado)')
plt.ylabel('Preço (Escalonado)')
plt.show()

print("\n--- Comparação Final de Acurácias ---")
print(f"Acurácia LinearSVC (sem escalonamento): {accuracy_linear:.4f}")
print(f"Acurácia SVC (com escalonamento): {accuracy_svc:.4f}")

if accuracy_svc > accuracy_linear:
    print("O modelo SVC (não-linear) obteve uma acurácia maior, sugerindo que a fronteira de decisão é mais complexa.")
else:
    print("O modelo LinearSVC obteve uma acurácia similar ou maior, sugerindo que a fronteira de decisão pode ser linear ou que os dados não se beneficiam tanto do SVC.")
