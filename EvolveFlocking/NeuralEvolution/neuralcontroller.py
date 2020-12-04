# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 10:35:30 2020

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
        self.vel += self.actuator() 
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
                
    def sensor(self):
        #Get sensor inputs
        D = self.distMatrix < 100.0
        angle = (np.arctan2(self.vel[:,0],self.vel[:,1])* 180 / np.pi).reshape(N,1)
        tot_angle = D.dot(angle).reshape(N,1)
        average_angle = tot_angle/D.sum(axis=1).reshape(N, 1)
        sensorA = (average_angle-angle+180.0)/360.0
        
        fourdist = np.zeros((N,4))
        for i in range(N):
                diff = self.pos-self.pos[i]
                dist1 = []
                dist2 = []
                dist3 = []
                dist4 = []
                for x in range(N):
                        if diff[x,0] > 0 and diff[x,1] > 0:
                                dist1.append(math.hypot(diff[x,0],diff[x,1]))
                        if diff[x,0] < 0 and diff[x,1] > 0:
                                dist2.append(math.hypot(diff[x,0],diff[x,1]))                        
                        if diff[x,0] < 0 and diff[x,1] < 0:
                                dist3.append(math.hypot(diff[x,0],diff[x,1]))
                        if diff[x,0] > 0 and diff[x,1] < 0:
                                dist4.append(math.hypot(diff[x,0],diff[x,1]))
                fourdist[i] = [min(dist1,default=0),min(dist2,default=0),min(dist3,default=0),min(dist4,default=0)]
        sensorD = 1 - fourdist/100.0
        sensor_input = np.hstack((sensorA,sensorD))
        return sensor_input

    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s


    def feed_propagation(self,inputs,wi,wo):
        #Create neural network
        ah = np.zeros((4,1))
        ao = np.zeros((2,1))
        for j in range(4):
            sumi = 0.0
            for i in range(5):
                sumi += wi[i][j] * inputs[i]
            ah[j] = sumi
        for j in range(2):
            sumo = 0.0
            for i in range(4):
                sumo += wo[i][j] * ah[i]
            ao[j] = sumo
        return ao.reshape((1,2))


    def actuator(self):
        steer = np.zeros((N,2))
        sensor_input = self.sensor()
        for i in range(N):
                steer[i] = self.feed_propagation(sensor_input[i],wi[:,:,i],wo[:,:,i])
        return steer

    

    def fitnessA(self):     
        # Alignment Fitness
        D = self.distMatrix < 100.0
        direction = self.vel/np.linalg.norm(self.vel,axis=1,keepdims=True)
        tot_direction = D.dot(direction) #sum directions of the flock
        phi = np.linalg.norm(tot_direction,axis=1,keepdims=True)/D.sum(axis=1).reshape(N, 1) #norm/number of flock member
        fitA = phi
        return fitA

    def fitnessS(self):
        # Seperation Fitness
        D = self.distMatrix < 14.0
        fitS = -(D.sum(axis=1).reshape(N, 1)-1)/1.0 
        return fitS

    def fitnessC(self):
        # Cohesion Fitness
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

    plt.pause(20)
    plt.close()
    
    scoreA = birds.fitA
    scoreS = birds.fitS
    scoreC = birds.fitC

    return scoreA, scoreS, scoreC


     
        





