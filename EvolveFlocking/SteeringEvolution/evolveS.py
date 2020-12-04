# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:33:47 2020

@author: lu
"""

import Sep
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
fitness = []
gene = []
initial = 0.01

while x in range(c):
    chromSep = np.random.rand(N)*initial
    for i in list(range(m)):
        print('iter =',i)
        #print('S =', sum(chromSep)/N)  
        scoreS = simulation()
        #select the better half entities
        inds = np.argsort(scoreS.flatten())[int(N/2)::]
        #print('Snew =', sum(chromSep[inds])/N*2)
        S = breed(chromSep, inds)
        chromSep = S

        fitness.append(sum(scoreS.flatten())/N)
        gene.append(sum(chromSep[inds])/N*2)
        #print(sum(scoreS.flatten())/N)
    x += 1
    
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





