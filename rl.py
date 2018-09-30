# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts includes a series of functions to use in the
development of the project. More precisely, these functions are related with
Reinforcement Learning implementations.

"""

import numpy as np
import pickle as pck

class  experience_buffer ():
    """ Create a buffer to store information to train the model"""
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        index = np.random.choice(np.arange(len(self.buffer)),size)
        return np.array([self.buffer[i] for i in index])
    
class recurrent_experience_buffer():
    """ Create a buffer to store information to train recurrent models"""
    def __init__(self, buffer_size = 100):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample(self,batch_size,trace_length,n_var):
        index = np.random.choice(np.arange(len(self.buffer)),batch_size)
        sampled_episodes = [self.buffer[i] for i in index]
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,episode.shape[0]+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length,:])
        return np.reshape(np.array(sampledTraces),[-1,n_var])

def getNewEpsilon(epsilon):
    """ Decay of epsilon over time"""
    if epsilon < 0.5:
        return epsilon*0.99999     
    return epsilon*0.999999

def getSimpleReward(delta_f):
    """ Reward when cost is not considered"""
    return 2**(10-np.abs(delta_f))

def getFullReward(delta_f,nodes):
    """ Reward when cost is considered"""
    r = 0
    
    for n in nodes:
        r += n.alpha*(n.getZ()**2)
    r *= 100

    return 2**(10-np.abs(delta_f)) - r

def getRatioReward(delta_f,z1,z2):
    """ Reward when cost is considered depending on the power ratios"""
    
    r = 0
    
    if (np.abs(delta_f) < 0.05):
        r = 100
        
    if (np.abs(z1-(z2/2)) < 0.2):
        r += 100
    
    return r
    #return 2**(10-2*np.abs(delta_f)) + 2**(10-5*np.abs(z1-(z2/2)))

def setDiscretePower(action,node):
    """ Perform agent action"""
    if action == 0:
        node.modifyPower(0.003*node.getPower())
    else:
        node.modifyPower(-0.003*node.getPower())
        
    return node

def setContinuousPower(action,node):
    """ Perform agent action"""
    node.modifyPower(action)       
    return node

def getSumZ(nodes):
    Z = 0
    for node in nodes:
        Z += node.getZ()
    return Z

def endEpisode(delta_f):
    """ End the episode if frequency is 10 Hertz far away from setpoint"""
    if np.abs(delta_f) > 50:
        return True
    return False

def saveData(name,data):
    """ Save information about the trained model"""
    with open("data/"+name, "wb") as handle:
        pck.dump(data, handle, protocol=pck.HIGHEST_PROTOCOL)
    
def readData(name):
    """ Load the requested model"""
    with open("../data/"+name, "rb") as handle:
        return pck.load(handle)
    
def updateTargetGraph(tfVars,tau):
    """ Update target graph towards main graph"""
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    """ Commit the update"""
    for op in op_holder:
        sess.run(op)
        
def discountReward(r,gamma):
    """ Discount the set of rewards and normalize them"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) 
    return discounted_r
