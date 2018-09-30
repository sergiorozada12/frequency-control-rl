# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts includes a series of functions to use in the
development of the project. More precisely, these functions are related with
the dynamics and the structure of the system.

"""

class Node:
    """ Implements each generation/load node"""
    def __init__(self,powerSetPoint,alpha=1):
        self.p = powerSetPoint
        self.alpha = alpha
        
    def modifyPower(self,powerVariation):
        self.p += powerVariation
        
    def getPower(self):
        return self.p
    
class Area:
    """ Implements each area frequency conditions"""
    def __init__(self,frequencySetPoint,M,D):
        self.f = frequencySetPoint
        self.delta_f = 0
        self.M = M
        self.D = D
        
    def getDeltaDelta(self,nodes):
        deltaDelta = - self.D*self.delta_f
        
        for node in nodes:
            deltaDelta += node.getPower()
            
        return deltaDelta/self.M
    
    def calculateDeltaF(self,nodes):
        self.delta_f += self.getDeltaDelta(nodes)
        
    def getDeltaF(self):
        return self.delta_f
    
    def getFrequency(self):
        return self.f + self.delta_f
    
class Node_Secondary:
    """ Implements each generation/load node"""
    def __init__(self,z,alpha=1):
        self.z = z
        self.alpha=alpha
        
    def modifyZ(self,deltaZ):
        self.z += deltaZ
        
        if (self.z < 0.5):
            self.z = 0.5
            
        if (self.z > 5):
            self.z = 5
        
    def getZ(self):
        return self.z
    
class Area_Secondary:
    """ Implements each area frequency conditions in Secondary Control"""
    def __init__(self,frequencySetPoint,M,D,Tg,Rd):
        self.f = frequencySetPoint
        self.delta_f = 0
        self.M = M
        self.D = D
        self.Tg = Tg
        self.Rd = Rd
        
    def setLoad(self,Pl):
        self.Pl = Pl
        
    def setGeneration(self,Pg):
        self.Pg = Pg
    
    def calculateDeltaF(self):
        self.delta_f += (self.Pg - self.Pl - self.D*self.delta_f)/self.M
        
    def calculatePg(self,Z):
        self.Pg += (-self.Pg + Z -(1/self.Rd)*self.delta_f)/self.Tg
        
    def getDeltaF(self):
        return self.delta_f
    
    def getFrequency(self):
        return self.f + self.delta_f
    
    def getLoad(self):
        return self.Pl
    
    def getGeneration(self):
        return self.Pg
        
    