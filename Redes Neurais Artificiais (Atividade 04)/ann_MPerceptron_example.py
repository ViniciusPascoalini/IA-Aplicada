# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:23:04 2020
https://www.python-course.eu/neural_networks_with_scikit.php
https://medium.com/as-m%C3%A1quinas-que-pensam/m%C3%A9tricas-comuns-em-machine-learning-como-analisar-a-qualidade-de-chat-bots-inteligentes-m%C3%A9tricas-1ba580d7cc96
https://medium.com/@vitorborbarodrigues/m%C3%A9tricas-de-avalia%C3%A7%C3%A3o-acur%C3%A1cia-precis%C3%A3o-recall-quais-as-diferen%C3%A7as-c8f05e0a513c
@author: 
"""

import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()
# splitting into train and test datasets

from sklearn.model_selection import train_test_split
datasets = train_test_split(iris.data, iris.target,
                            test_size=0.2)

train_data, test_data, train_labels, test_labels = datasets
# scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# we fit the train data
scaler.fit(train_data)

# scaling the train data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

#print(train_data[:3])

# Creating the Model
from sklearn.neural_network import MLPClassifier
# creating an classifier from the model:
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# Training the Model
# let's fit the training data to our model
mlp.fit(train_data, train_labels)


# Metrics the Model
from sklearn.metrics import accuracy_score

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

from sklearn.metrics import confusion_matrix

"""
Calculando as métricas desejadas, a partir da Matriz de confusão
"""

confMatrixWithTrainData =  confusion_matrix(predictions_train, train_labels)
confusionMatrix = confusion_matrix(predictions_test, test_labels)

"""
Com os dados de Treino
"""
print('\n\n ------- Com os dados de Treino: -------\n')

diagonalPrinc = confMatrixWithTrainData.diagonal()
print(confMatrixWithTrainData)


acc = sum(diagonalPrinc)/sum(sum(confMatrixWithTrainData))
print('Acurácia: ', acc)

"""
np.sum axis=0 soma os elementos das linhas da matriz
"""
precision = diagonalPrinc/np.sum(confMatrixWithTrainData, axis=0)
print('\nPrecisão: ', precision)
precisionMean = np.mean(precision) 
print('\nMédia das precisões: ', precisionMean )

"""
np.sum axis=1 soma os elementos das colunas da matriz
"""
recall = diagonalPrinc/np.sum(confMatrixWithTrainData, axis=1)
print('\nRecall: ', recall)
recallMean = np.mean(recall)
print("\nMédia dos recall's: ", recallMean)

F1Score = 2*precision*recall/(precision + recall)
print('\nF1Score: ', F1Score)
F1ScoreMean = np.mean(F1Score)
print('\nMédia de F1Score: ', F1ScoreMean)

"""
Com os dados de Teste
"""

print('\n\n ------- Com os dados de Teste: -------\n')

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