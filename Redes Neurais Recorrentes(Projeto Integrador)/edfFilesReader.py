
import mne


import matplotlib.pyplot as plt
import numpy as np

Tamanho=30000

qtensaios=109

Y=np.zeros((qtensaios*14))


datam=np.zeros((14*qtensaios,9,Tamanho))

k=0;

for i in range (1,qtensaios+1):
# for i in range (1,110):
    if i==43 or i==88 or i==89 or i==92 or i==100 or i==104:
        print(i)
    else:
        
        for j in range (1,15):
            ensaio='%03d' %(i) # 1 a 109
            task='%02d'   %(j) #de 1 a 14
    
    
            arq="/S"+ensaio+"/S"+ensaio+"R"+task+".edf"
    
            file = r"/home/vinicius/Downloads/files"+arq
            data = mne.io.read_raw_edf(file)
            raw_data = data.get_data()
        
            # you can get the metadata included in the file and a list of all channels:
            info = data.info
            channels = data.ch_names
            
            # if i==1 and j==1:
            #     aux="T0"
                
            # aux=data.annotations.description.T[j-1]
            
            # if aux=="T0":
            #     T=0
            # elif aux=="T1":
            #     T=1
            # elif aux=="T2":
            #     T=2
            T=0
            auxz=Tamanho-len(raw_data[2])
            auxzeros=np.zeros(auxz)
            
            datam[k,0,:]=np.concatenate((raw_data[9],auxzeros))
            datam[k,1,:]=np.concatenate((raw_data[11],auxzeros))
            datam[k,2,:]=np.concatenate((raw_data[7],auxzeros))
            datam[k,3,:]=np.concatenate((raw_data[13],auxzeros))
            datam[k,4,:]=np.concatenate((raw_data[15],auxzeros))
            datam[k,5,:]=np.concatenate((raw_data[19],auxzeros))
            datam[k,6,:]=np.concatenate((raw_data[1],auxzeros))
            datam[k,7,:]=np.concatenate((raw_data[5],auxzeros))
            datam[k,8,:]=np.concatenate((raw_data[10],auxzeros))
            
            
            if j==3 or j==7 or j==11:
                Y[k]=6+T
            elif j==4 or j==8 or j==12:
                Y[k]=9+T
            elif j==5 or j==9 or j==13:
                Y[k]=12+T
            elif j==6 or j==10 or j==14:
                Y[k]=15+T
            else:
                Y[k]=T 
            k=k+1

x_train=datam[0:700,:,:]
x_test=datam[700:1400,:,:]
y_train=Y[0:700]
y_test=Y[700:1400]
        

# c5 c1 c2 c6
# c1= 9
# c2=11
# c5= 7
# c6=13

# 109 gravações

#14 tarefas
# 1=linha de base olhos abertos
# 2=linha de base olhos fechados

# 3 7 11 = open and close left or right fist

# 4 8 12 imagine opening and closing left or right fist

# 5 9 13 open and close both fists or both feet

# 6 10 14 imagine opening and closing both fists or both feet

#T=T0 rest
#T=T1 left fist------both fists
#T=T2 right fist-----both feet




plt.figure(1)
plt.plot(raw_data[9])



