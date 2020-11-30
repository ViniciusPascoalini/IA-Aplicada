"""
Created on Sat Oct 24 22:43:02 2020

@author: Arthur Pires,
"""




import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from sklearn.datasets import load_iris

import matplotlib.pyplot as plt


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


# Maximos e minimos das entradas (comprimento e largura das pétalas)
maxc=max(data.data[:,2])
maxl=max(data.data[:,3])
minc=min(data.data[:,2])
minl=min(data.data[:,3])

# Delta para aumento do espaço de entrada
Dc=0.2
Dl=0.3
# Definindo espaço de entrada
comprimento = ctrl.Antecedent(np.arange(minc-Dc, maxc+Dc, .00001),'comprimento') 
largura = ctrl.Antecedent(np.arange(minl-Dl, maxl+Dl, .00001),'largura') 
# Denifinindo espaço de saída
especie = ctrl.Consequent(np.arange(0, 2.5, 0.5),'especie')

# Definindo conjuntos de saida automaticamente
    # Comprimento Baixo(CB), Médio(CM) e Alto(CA)
comprimento.automf(names=['CB','CM','CA'])
    # Largura Baixa(LB), Média(LM) e Alta(LA)
largura.automf(names=['LB','LM','LA'])

# Definindo saída como Espécie 1 (E1), 2 (E2) e 3 (E3) 
especie['E1'] = fuzz.trimf(especie.universe, [0, 0,  1 ])
especie['E2'] = fuzz.trimf(especie.universe, [0.5, 1, 1.5])
especie['E3'] = fuzz.trimf(especie.universe, [1, 2, 2])

# Visualização dos conjuntos
comprimento.view()
largura.view() 
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
plt.figure()
plt.plot(resultados)

acertos1=0
acertos2=0
acertos3=0


# Verificação da eficiencia a partinsertir dos limites obtidos visualmente
# l1=0.7
# l2=1.2
l1=2/3
l2=l1*2

laux=len(data1c[:,1])
for i in range(laux):
 
    if resultados[i]<l1 :
        acertos1+=1
    if ((resultados[i+laux]>l1) & (resultados[i+25]<l2)):
        acertos2+=1
    if resultados[i+(laux*2)]>l2 :
        acertos3+=1
    
print("\n % de acertos em cada grupo: ",[100*acertos1/laux, 100*acertos2/laux,100*acertos3/laux])
print("\n % de acerto Total: ",round(np.mean(100*(acertos1+acertos2+acertos3)/(laux*3)),2))

