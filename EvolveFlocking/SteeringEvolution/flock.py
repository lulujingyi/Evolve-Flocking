# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:07:21 2020

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
        self.fitA = 0
        self.fitS = 0
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
        self.fitA += self.fitnessA()
        self.fitS += self.fitnessS()
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
        # apply rule #1 - Separation
        D = self.distMatrix < 20.0
        vel = self.pos * D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, chromSep)
 
        # different distance threshold
        D = self.distMatrix < 40.0
 
        # apply rule #2 - Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, chromAlign)
        vel += vel2
 
        D = self.distMatrix < 50.0
        # apply rule #1 - Cohesion
        #vel3 = D.dot(self.pos) - self.pos
        vel3 = D.dot(self.pos)/D.sum(axis=1).reshape(self.N, 1)-self.pos
        self.limit(vel3, chromCoh)
        vel += vel3
 
        return vel

    def fitnessA(self):     
        # Alignment
        D = self.distMatrix < 100.0
        direction = self.vel/np.linalg.norm(self.vel,axis=1,keepdims=True)
        tot_direction = D.dot(direction) #sum directions of the flock
        phi = np.linalg.norm(tot_direction,axis=1,keepdims=True)/D.sum(axis=1).reshape(N, 1) #norm/number of flock member
        fitA = phi
        return fitA

    def fitnessS(self):
        # Seperation
        D = self.distMatrix < 14.0
        fitS = -(D.sum(axis=1).reshape(N, 1)-1)/1.0 
        return fitS

    def fitnessC(self):
        # Cohesion
        fitC = (1-self.distMatrix/200).sum(axis=1).reshape(N,1)/N
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
    
    scoreA = birds.fitA
    scoreS = birds.fitS
    scoreC = birds.fitC

    return scoreA, scoreS, scoreC


