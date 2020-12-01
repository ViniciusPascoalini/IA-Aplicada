# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np

#Carrega o iris dataset em iris 
iris = load_iris()
alldata = iris.data
label = iris.target
ncenters = 3

#divide o dataset em porção de treino e teste
X_train, X_test, y_train, y_test = train_test_split( alldata, label, test_size=0.5, random_state=42)

#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_train.T, ncenters, 2, error=0.005, maxiter=1000, init=None)

cluster_membership = np.argmax(u, axis=0)


# Generate uniformly sampled data spread across the range [0, 10] in x and y
newdata = X_test

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)


y_pred = np.argmax(u, axis=0)

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