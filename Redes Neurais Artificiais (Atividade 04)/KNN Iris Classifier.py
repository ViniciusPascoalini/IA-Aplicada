from sklearn.neighbors import KNeighborsClassifier
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

#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=5,weights="uniform")
neigh.fit(X_train, y_train)

#Prevendo valores da porção de teste
y_pred = neigh.predict(X_test)


"""
Calculando as métricas desejadas, a partir da Matriz de confusão
"""

confusionMatrix = confusion_matrix(y_test, y_pred)

diagonalPrinc = confusionMatrix.diagonal()
print(confusionMatrix)


acc = sum(diagonalPrinc)/sum(sum(confusionMatrix))
print('Acurácia: ', acc)

"""
np.sum axis=0 soma os elementos das linhas da matriz
"""
precision = diagonalPrinc/np.sum(confusionMatrix, axis=0)
print('\nPrecisão: ', precision)
precisionMean = np.mean(precision) 
print('\nMédia das precisões: ', precisionMean )

"""
np.sum axis=1 soma os elementos das colunas da matriz
"""
recall = diagonalPrinc/np.sum(confusionMatrix, axis=1)
print('\nRecall: ', recall)
recallMean = np.mean(recall)
print("\nMédia dos recall's: ", recallMean)

F1Score = 2*precision*recall/(precision + recall)
print('\nF1Score: ', F1Score)
F1ScoreMean = np.mean(F1Score)
print('\nMédia de F1Score: ', F1ScoreMean)