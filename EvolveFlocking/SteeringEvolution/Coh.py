# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:37:45 2020

@author: lu
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
 

class Birds:
    """
    Simulates flock behaviour of birds, using the realistic-looking Boids model (1986)
    """
    def __init__(self):
        self.N = N
        self.minDist = minDist
        self.maxRuleVel = maxRuleVel
        self.maxVel = maxVel*np.ones(N)
        self.maxSp = 2.0

        
        # Computing initial position and velocity
        self.pos = [width / 2.0, height / 2.0] + 300 * (np.random.rand(2 * N)-0.5).reshape(N, 2)
        self.vel = np.random.rand(2 * N).reshape(N, 2)-0.5
        self.time = 0
        self.fitC = 0

    def tick(self, frameNum, pts, beak):
        """
        Update the simulation by one time step
        """
        # get pairwise distances
        self.distMatrix = squareform(pdist(self.pos))
        # apply rules:
        self.vel += self.apply_rules() 
        self.limit(self.vel, self.maxVel)
        self.pos += self.vel
        self.time += 1 
        self.fitC += self.fitnessC()
        self.apply_bc()
        # update data
        pts.set_data(self.pos.reshape(2 * self.N)[::2],
                     self.pos.reshape(2 * self.N)[1::2])
        vec = self.pos + 10 * self.vel / self.maxSp  
        beak.set_data(vec.reshape(2 * self.N)[::2],
                      vec.reshape(2 * self.N)[1::2])

 
    def limit(self, x, max_val):
        """ Limit magnitide of 2D vectors in array X to maxValue """
        for i in range(len(x)):
            vec = x[i]
            mag = norm(vec)
            maxv = max_val[i] 
            if mag > maxv:
                vec[0], vec[1] = vec[0] * maxv / mag, vec[1] * maxv / mag            

                
    def apply_bc(self):
        """ Apply boundary conditions """
        deltaR = 2.0
        for coord,speed in zip(self.pos,self.vel):
            if coord[0] > width - deltaR:
                speed[0] = - speed[0]
            if coord[0] < deltaR:
                speed[0] = - speed[0]
            if coord[1] > height - deltaR:
                speed[1] = - speed[1]
            if coord[1] < deltaR:
                speed[1] = - speed[1]
              
    def apply_rules(self):

        # apply rule #1 - Cohesion
        D = self.distMatrix < 100.0
        vel3 = D.dot(self.pos)/D.sum(axis=1).reshape(self.N, 1)-self.pos
        self.limit(vel3, chromCoh)
 
        return vel3



    def fitnessC(self):
        # Cohesion
        fitC = (1-self.distMatrix/100).sum(axis=1).reshape(N,1)/N
        return fitC            
 
 
def tick(frameNum, pts, beak, birds):
    """ Update function for animation """
    birds.tick(frameNum, pts, beak)
    return pts, beak
 
 
def simulation():
    print('Starting flock simulation...')
 
    # Create birds
    birds = Birds()
 
    # Setup plot
    fig = plt.figure()
    
    ax = plt.axes(xlim=(0, width), ylim=(0, height))

    pts, = ax.plot([], [], markersize=10, c='k', marker='o', ls='None')
    beak, = ax.plot([], [], markersize=4, c='c', marker='o', ls='None')
    anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, birds), interval=10, repeat=False) #frames = 250,

    plt.pause(40)
    plt.close()

    scoreC = birds.fitC

    return scoreC