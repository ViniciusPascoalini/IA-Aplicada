from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np

#Carrega o iris dataset em iris 
iris = load_iris()
X = iris.data
y = iris.target

#divide o dataset em porção de treino e teste
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42)

#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_train)

#Prevendo valores da porção de teste
y_pred = kmeans.predict(X_test)

"""
Calculando as métricas desejadas, a partir da Matriz de confusão
"""

confusionMatrix = confusion_matrix(y_test, y_pred)

truePositives = np.amax(confusionMatrix, axis=1)
print(confusionMatrix)


acc = sum(truePositives)/sum(sum(confusionMatrix))
print('Acurácia: ', acc)

"""
np.sum axis=0 soma os elementos das linhas da matriz
"""
line = np.argmax(confusionMatrix, axis=1)
somaLinhas = np.sum(confusionMatrix, axis=0)
"""
Ordenando linhas
"""
somaLinhaEmOrdem = []

for i in range(len(somaLinhas)):
    somaLinhaEmOrdem.append(somaLinhas[line[i]])

precision = truePositives/somaLinhaEmOrdem
print('\nPrecisão: ', precision)
precisionMean = np.mean(precision) 
print('\nMédia das precisões: ', precisionMean )

"""
np.sum axis=1 soma os elementos das colunas da matriz
"""
recall = truePositives/np.sum(confusionMatrix, axis=1)
print('\nRecall: ', recall)
recallMean = np.mean(recall)
print("\nMédia dos recall's: ", recallMean)

F1Score = 2*precision*recall/(precision + recall)
print('\nF1Score: ', F1Score)
F1ScoreMean = np.mean(F1Score)
print('\nMédia de F1Score: ', F1ScoreMean)