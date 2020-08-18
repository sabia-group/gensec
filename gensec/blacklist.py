import os
import numpy as np 
from gensec.modules import *

class Blacklist:

    def __init__(self, structure):

    # def create_blacklist(molecules, list_of_torsions):

        if len(structure.molecules) > 1:
            torsions = np.array([0 for i in structure.list_of_torsions])
            quaternion = produce_quaternion(0, np.array([0, 0, 1]))
            value_com = np.array([0, 0, 0])
            blacklist_one = np.hstack((torsions, quaternion, value_com))
            blacklist = np.hstack((torsions, quaternion, value_com)) 
            for i in range(len(structure.molecules) -1):
                blacklist = np.concatenate((blacklist, blacklist_one), axis=0)
        else:
            torsions = np.array([0 for i in structure.list_of_torsions])
            quaternion = produce_quaternion(0, np.array([0, 0, 1]))
            value_com = np.array([0, 0, 0])
            blacklist = np.hstack((torsions, quaternion, value_com))        
        self.blacklist = blacklist 



    def add_to_blacklist(self, vector):
        self.blacklist = np.vstack((self.blacklist, vector))

    def not_in_blacklist(self, vector):
        check = False
        dist = [np.linalg.norm(point - vector) for point in self.blacklist]
        if len(dist) > 1:
            if all(i > 100 for i in dist):
                check= True
        else:
            if dist[0] > 100:
                check= True
        return check

    def get_blacklist(self):
        for vec in self.blacklist:
            print(vec)

    def get_len(self):
        return len(self.blacklist)
