# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:46:24 2020

@author: lu
"""

import Align
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

while x in range(c):
    chromAlign = np.random.rand(N)*0.001
    for i in list(range(m)):
        print('iter =',i)
        #print('A =', sum(chromAlign)/N) 
        scoreA = simulation()
        #select the better half entities
        inds = np.argsort(scoreA.flatten())[int(N/2)::]
       
        #print('Anew =', sum(chromAlign[inds])/N*2)
        A = breed(chromAlign, inds)
        chromAlign = A
        
        fitness.append(sum(scoreA.flatten())/N)
        gene.append(sum(chromAlign[inds])/N*2)
        #print(sum(scoreA.flatten())/N)
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