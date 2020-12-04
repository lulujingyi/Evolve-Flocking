# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:35:34 2020

@author: lu
"""

import Coh
import matplotlib.pyplot as plt

#==================Parameters===================================================
width, height =1847,1165        # frame size
N = 100                         # number of entities
minDist = 100.0                 # min dist of approach
maxRuleVel = 0.1                # max magnitude of velocities
maxVel = 2.0                    # max magnitude of final velocity


#==================Crossover genes==============================================
mutation = 0.3
def breed(gene1, index):
    #random.shuffle(index)
    sortgene11 = gene1[index]
    #random.shuffle(index)
    sortgene12 = gene1[index]
    newgene1 = np.concatenate([sortgene11, sortgene12])*((1-mutation)+np.random.rand(N)*2*mutation)
    return newgene1

def rank(score):
    temp = np.argsort(score)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(score))
    return ranks

#==================Simulation===================================================
fitness = []
gene = []       
m = 100                         # number of generations per cycle
c = 5                           # number of cycles
x = 0
initial = 0.001

while x in range(c):
    chromCoh = np.random.rand(N)*initial
    for i in list(range(m)):
        print('iter =',i)
        #print('C =', sum(chromCoh)/N) 
        scoreC = simulation()
        #select the better half entities
        inds = np.argsort(scoreC.flatten())[int(N/2)::]
        #print('Cnew =', sum(chromCoh[inds])/N*2)
        C = breed(chromCoh, inds)
        chromCoh = C
        
        fitness.append(sum(scoreC.flatten())/N)
        gene.append(sum(chromCoh[inds])/N*2)
        #print(sum(scoreC.flatten())/N)
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