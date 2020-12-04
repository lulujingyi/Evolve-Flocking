# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:58:33 2020

@author: lu
"""

import flock
import matplotlib.pyplot as plt

#==================Parameters===================================================
width, height =1847,1165        # frame size
N = 100                         # number of entities
minDist = 100.0                 # min dist of approach
maxRuleVel = 0.1                # max magnitude of velocities
maxVel = 2.0                    # max magnitude of final velocity


#==================Crossover genes==============================================
mutation = 0.3
def breed(gene1, gene2, gene3, indexA, indexS, indexC):
    #random.shuffle(index)
    sortgene11 = gene1[indexA]
    #random.shuffle(index)
    sortgene21 = gene2[indexS]
    #random.shuffle(index)
    sortgene31 = gene3[indexC]
    #random.shuffle(index)
    sortgene12 = gene1[indexA]
    #random.shuffle(index)
    sortgene22 = gene2[indexS]
    #random.shuffle(index)
    sortgene32 = gene3[indexC]    
    newgene1 = np.concatenate([sortgene11, sortgene12])*((1-mutation)+np.random.rand(N)*2*mutation)
    newgene2 = np.concatenate([sortgene21, sortgene22])*((1-mutation)+np.random.rand(N)*2*mutation)
    newgene3 = np.concatenate([sortgene31, sortgene32])*((1-mutation)+np.random.rand(N)*2*mutation)

    return newgene1, newgene2, newgene3

#==================Simulation===================================================
chromAlign = np.random.rand(N)*0.01  
chromSep = np.random.rand(N)*0.01  
chromCoh = np.random.rand(N)*0.01   

fitness = []
m = 100                         # number of generations per cycle

for i in list(range(m)):
    print('iter =',i)
    
    #print('A =', sum(chromAlign)/N,'S =', sum(chromSep)/N,'C =', sum(chromCoh)/N) 
    scoreA,scoreS,scoreC = simulation()
    indsA = np.argsort(scoreA.flatten())[int(N/2)::]
    indsS = np.argsort(scoreS.flatten())[int(N/2)::]
    indsC = np.argsort(scoreC.flatten())[int(N/2)::]
    
    #print('Anew =', sum(chromAlign[indsA])/N*2,'Snew =', sum(chromSep[indsS])/N*2,'Cnew =', sum(chromCoh[indsC])/N*2)
    A,S,C = breed(chromAlign, chromSep, chromCoh, indsA, indsS, indsC)
    chromAlign = A
    chromSep = S
    chromCoh = C

    fitness.append(sum(scoreA)/N+sum(scoreS)/N+sum(scoreC)/N)
    #print(sum(scoreA)/N,sum(scoreS)/N,sum(scoreC)/N)

generation = list(range(m))    
plt.plot(generation,fit[0:m])

plt.show()

