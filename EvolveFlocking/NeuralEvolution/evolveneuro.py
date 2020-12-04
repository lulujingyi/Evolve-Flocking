# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 13:59:07 2020

@author: lu
"""

import neuralcontroller
import matplotlib.pyplot as plt

#==================Parameters===================================================
width, height =1847,1165        # frame size
N = 20                          # number of entities
minDist = 100.0                 # min dist of approach
maxRuleVel = 0.1                # max magnitude of velocities
maxVel = 2.0                    # max magnitude of final velocity


#==================Crossover genes==============================================
mutation = 0.3

def breed(gene1, gene2, index):
    #random.shuffle(index)
    sortgene11 = gene1[:,:,index]
    #random.shuffle(index)
    sortgene21 = gene2[:,:,index]
    #random.shuffle(index)
    sortgene12 = gene1[:,:,index]
    #random.shuffle(index)
    sortgene22 = gene2[:,:,index] 
    newgene1 = np.concatenate([sortgene11, sortgene12],axis=2)*((1-mutation)+np.random.rand(5,4,N)*2*mutation)
    newgene2 = np.concatenate([sortgene21, sortgene22],axis=2)*((1-mutation)+np.random.rand(4,2,N)*2*mutation)
    return newgene1, newgene2


#==================Simulation===================================================
fitness = []
gene = []       
m = 100                         # number of generations per cycle
c = 5                           # number of cycles
x = 0
initial = 1

while x in range(c):
    wi = np.ones((5,4,N))*(np.random.rand(5,4,N)-0.5)*initial
    wo = np.ones((4,2,N))*(np.random.rand(4,2,N)-0.5)*initial
    for i in list(range(m)):
        print('iter =',i)
        scoreA,scoreS,scoreC = simulation()
        scoretot = scoreA.flatten() + scoreS.flatten() + scoreC.flatten()   
        inds = np.argsort(scoretot)[int(N/2)::]
        winew,wonew = breed(wi, wo, inds)
        wi = winew
        wo = wonew
        fitness.append(sum(scoretot)/N)
        #print(sum(scoretot)/N)
    x += 1

#==================Plot=========================================================
generation = list(range(m))    
plt.plot(generation,fitness[0:m])
plt.plot(generation,fitness[m:2*m])
plt.plot(generation,fitness[2*m:3*m])
plt.plot(generation,fitness[3*m:4*m])
plt.plot(generation,fitness[4*m:5*m])
plt.xlim([0,m])
plt.xlabel('Generation')
plt.ylabel('fitness')
plt.show()