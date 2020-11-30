"""
Created on Sat Oct 24 22:43:02 2020
@author: Arthur Pires,
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

from statistics import mean
# Carregar dataset iris e dividr em 2
data = load_iris() 
data0 = np.array(data.data)

# Parte 1 dataset iris
data1=data0[0:25]
data2=data0[50:75]
data3=data0[100:125]
#Parte 2 dataset iris 
data1c=data0[25:50]
data2c=data0[75:100]
data3c=data0[125:]
# data1c=data1
# data2c=data2
# data3c=data3
# data1c=data0[0:50]
# data2c=data0[50:100]
# data3c=data0[100:]

#Target para a segunda parte do dataset
target = data.target
target1c=target[25:50]
target2c=target[75:100]
target3c=target[125:]

y_test = np.concatenate((target1c, np.concatenate((target2c, target3c), axis=None)), axis=None) 

# Maximos e minimos das entradas (comprimento e largura das pétalas)

maxc=max(data0[:,2])
maxl=max(data0[:,3])
minc=min(data0[:,2])
minl=min(data0[:,3])

mediaCe1=mean(data1[:,2])
mediaCe2=mean(data2[:,2])
mediaCe3=mean(data3[:,2])

mediaLe1=mean(data1[:,3])
mediaLe2=mean(data2[:,3])
mediaLe3=mean(data3[:,3])


# Delta para aumento do espaço de entrada
Dc=.00001
Dl=.00001
# Definindo espaço de entrada
comprimento = ctrl.Antecedent(np.arange(minc-Dc, maxc+Dc, .00001),'comprimento') 
largura = ctrl.Antecedent(np.arange(minl-Dl, maxl+Dl, .00001),'largura') 
# Denifinindo espaço de saída
# especie = ctrl.Consequent(np.arange(0, 2.01, 0.01),'especie')
especie = ctrl.Consequent(np.arange(0, 3.01, 0.01),'especie')

# Definindo conjuntos

     # Comprimento Baixo(CB), Médio(CM) e Alto(CA)
comprimento['CB'] = fuzz.trapmf(comprimento.universe, [minc,minc, mediaCe1,  mediaCe2])
comprimento['CM'] = fuzz.trimf(comprimento.universe, [mediaCe1, mediaCe2,  mediaCe3])
comprimento['CA'] = fuzz.trapmf(comprimento.universe, [mediaCe2, mediaCe3,  maxc,maxc])

    # Largura Baixa(LB), Média(LM) e Alta(LA)
largura['LB'] = fuzz.trapmf(largura.universe, [minl,minl, mediaLe1,  mediaLe2])
largura['LM'] = fuzz.trimf(largura.universe, [mediaLe1, mediaLe2,  mediaLe3])
largura['LA'] = fuzz.trapmf(largura.universe, [mediaLe2, mediaLe3,  maxl,maxl])

# Definindo saída como Espécie 1 (E1), 2 (E2) e 3 (E3) 

especie['E1'] = fuzz.trimf(especie.universe, [0, 0,  1 ])
especie['E2'] = fuzz.trimf(especie.universe, [1, 1.5, 2])
especie['E3'] = fuzz.trimf(especie.universe, [2, 3, 3])

# Visualização dos conjuntos
# comprimento.view()
# largura.view() 
# especie.view() 

# Definindo regras

    #Comprimento Baixo e Largura Baixa => Espécie 1 
regra1=  ctrl.Rule(comprimento['CB'] & (largura['LB']),especie['E1'])
    #Comprimento Médio e Largura Média => Espécie 2
regra2=  ctrl.Rule(comprimento['CM'] & (largura['LM']),especie['E2'])
    #Comprimento Alto e Largura Alta => Espécie 3
regra3=  ctrl.Rule(comprimento['CA'] & (largura['LA']),especie['E3'])
                  
regras=[regra1,regra2,regra3]


# Inserindo as regras
fuzzyIris_ctrl = ctrl.ControlSystem(regras) 
fuzzyIris = ctrl.ControlSystemSimulation(fuzzyIris_ctrl) 


# Rotina para a verificar cada amostra
datatc=np.vstack((data1c,data2c,data3c))
resultados=[]


for i in range(len(datatc[:,1])):

    cteste=datatc[i,2]    
    lteste=datatc[i,3]      
    fuzzyIris.input['comprimento'] = cteste
    fuzzyIris.input['largura'] = lteste

    fuzzyIris.compute()
    resultados.insert(i,fuzzyIris.output['especie'])

# Exibindo os resultados

laux=len(data1c[:,1])

plt.figure()
plt.plot(resultados,label='Resultados')
# plt.ylabel('% de acerto')
plt.xlabel('Segunda parte do dataset')
plt.grid(True)
plt.xticks([12.5,12.5+laux,12.5+(2*laux)],['Espécie 1','Espécie 2','Espécie 3'])

plt.suptitle('Resultados Fuzzy Iris', fontsize=14, fontweight='bold')

y_pred = np.array([])

# Verificação da eficiencia a partir dos limites 

l1=1 # 3*1/3
l2=2 # 3*2/3

plt.axhline(y=l1,color='r',label='Limite Inferior')
plt.axhline(y=l2,color='m',label='Limite Superior')
plt.legend() 


for i in range(3*laux):
 
    if resultados[i]<l1 :
        y_pred = np.append(y_pred, 0)
    elif ((resultados[i]>l1) & (resultados[i]<l2)):
        y_pred = np.append(y_pred, 1)
    else :
        y_pred = np.append(y_pred, 2)
    
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
