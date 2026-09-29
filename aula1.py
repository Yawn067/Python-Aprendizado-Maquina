#Propósito: Esta aula introduz os conceitos básicos de classificação utilizando o algoritmo Linear Support Vector Classifier (`LinearSVC`) da biblioteca `scikit-learn`. 
#Demonstra a criação de um conjunto de dados de treino, a construção do modelo e a avaliação inicial da acurácia.

**Instruções:** Copie o código abaixo e salve-o como `aula1.py`.
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Definição das características dos animais
# feature: [pelo longo?, perna curta?, faz auau?]
# 1 = sim, 0 = nao

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# Conjunto de dados de treino
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
# Rótulos (labels): 1 = porco, 0 = cachorro
treino_y = [1, 1, 1, 0, 0, 0]

# Criação e treino do modelo LinearSVC
modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

# Exemplo de predição com um animal misterioso
animal_misterioso = [0, 0, 0] # Sem pelo longo, perna curta, faz auau
print(f"Predição para animal misterioso [0,0,0]: {modelo.predict([animal_misterioso])}")

# Conjunto de dados de teste
misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

teste_x = [misterio1, misterio2, misterio3]
# Rótulos verdadeiros para o conjunto de teste
teste_y = [0, 1, 1]

previsoes = modelo.predict(teste_x)

# Calculo da acurácia manualmente
corretos = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_de_acerto = corretos/total * 100
print(f"Acurácia manual: {taxa_de_acerto:.2f}%")

# Calculo da acurácia usando sklearn.metrics.accuracy_score
taxa_de_acerto_sklearn = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia via sklearn.metrics: {taxa_de_acerto_sklearn:.2f}%")
