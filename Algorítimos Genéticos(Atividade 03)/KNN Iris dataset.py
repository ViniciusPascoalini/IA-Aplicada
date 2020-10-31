from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix

import numpy as np
 
#Carrega o iris dataset em iris 
iris = load_iris()
X = iris.data
y = iris.target

#divide o dataset em porção de treino e teste
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.7, random_state=42)

#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=5,weights="uniform")
neigh.fit(X_train, y_train)

#Prevendo valores da porção de teste
y_pred = neigh.predict(X_test)

"""
Gera a Matriz de Contingência, que mostra os acertos e erros do agrupamento,
alem de especificar para qual cluster esses dados foram associados
"""
contMatrix = contingency_matrix(y_pred, y_test)

"""
Aqui estou percorrendo a Matriz de Contingência, calculando a porcentagem de 
acerto para cada cluster e salvando o resultado no vetor clusterScores
"""
nClusters = len(contMatrix)
clusterScores = []
totalHits = 0

for i in range(nClusters):
    
    centr = np.argmax(contMatrix[i,:])
    centrValue = contMatrix[i, centr]
    soma = 0
    
    for j in range(nClusters):
        soma = soma + contMatrix[i,j]
        
    hitPercentage = centrValue/soma
    clusterScores.append(hitPercentage)
    totalHits = totalHits + centrValue
    

globalScore = totalHits/len(y_pred) 
 
"""
Mede a porcentagem total de acertos desconsiderando o nome dado aos clusters
(grau de similaridade)
"""
print(globalScore)

print(contMatrix)

print(clusterScores)