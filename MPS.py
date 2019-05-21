#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from Tensor_Train import *
from progressbar import ProgressBar
from svd_operations import svd_node_pair,svd_norm_node
class mps:
    def uniform(length,A,V=None,W=None):
        if V is None and W is None: return periodic_MPS(length,A)
        else: return open_MPS(length,A,V,W)

    def random(length,on_site_dim,bond_dim,boundary=None):
        A = np.random.uniform(-2,2,np.array((on_site_dim,bond_dim,bond_dim)))
        V = np.random.uniform(-2,2,np.array((on_site_dim,bond_dim)))
        W = np.random.uniform(-2,2,np.array((on_site_dim,bond_dim)))
        if boundary == "periodic": return periodic_MPS(length,A)
        elif boundary == "open": return open_MPS(length,A,V,W)
        else: #choose open/periodic randomly
            bc = np.random.choice(['periodic','open'])
            if bc == "periodic":
                return periodic_MPS(length,A)
            elif bc == "open":
                return open_MPS(length,A,V,W)

    def set_entry(self,site,tensor):
        self.node[site] = tensor
        if site == 0:
            if np.size(np.shape(tensor))>2:
                self.node[site].legs = "both"
            else:
                self.node[site].legs = "right"
        elif site == self.length-1:
            if np.size(np.shape(tensor))>2:
                self.node[site].legs = "both"
            else:
                self.node[site].legs = "left"
        else:
            self.node[site].legs = "left"

    def overlap(self,MPS2):
        network = rail_network(self,MPS2)
        network.contract()
        return network.contraction
    
    def left_normalize(self):
        pbar=ProgressBar()
        print("Left normalizing MPS")
        for n in pbar(range(0,self.length-1)):
            self.node[n].tensor,self.node[n+1].tensor = svd_node_pair.left(self.node[n],self.node[n+1])
        self.node[self.length-1].tensor = svd_norm_node.left(self.node[self.length-1])

    def right_normalize(self):
        pbar=ProgressBar()
        print("Right normalizing MPS")
        for n in pbar(range(self.length-1,0,-1)):
            self.node[n-1].tensor,self.node[n].tensor = svd_node_pair.right(self.node[n-1],self.node[n])
        self.node[0].tensor = svd_norm_node.right(self.node[0])

    def mixed_normalize(self,site):
        print("Mixed normalizing MPS")
        print("Left norm:")
        pbar=ProgressBar()
        for n in pbar(range(0,self.length-1)):
            self.node[n].tensor,self.node[n+1].tensor = svd_node_pair.left(self.node[n],self.node[n+1])
        # self.node[self.length-1].tensor = svd_norm_node.left(self.node[self.length-1])
        print("Right norm:")
        pbar=ProgressBar()
        for n in pbar(range(self.length-1,site,-1)):
            self.node[n-1].tensor,self.node[n].tensor = svd_node_pair.right(self.node[n-1],self.node[n])
        # self.node[site].tensor = svd_norm_node(self.node[site])

class mpo:
    def uniform(length,Q,V=None,W=None):
        if V is None and W is None: return periodic_MPO(length,Q)
        else: return open_MPO(length,Q,V,W)

    def random(length,on_site_dim,bond_dim,boundary=None):
        O = np.random.uniform(-1,1,np.array((on_site_dim,on_site_dim,bond_dim,bond_dim)))
        V = np.random.uniform(-1,1,np.array((on_site_dim,on_site_dim,bond_dim)))
        W = np.random.uniform(-1,1,np.array((on_site_dim,on_site_dim,bond_dim)))
        if boundary == "periodic": return periodic_MPO(length,O)
        elif boundary == "open": return open_MPO(length,O,V,W)
        else: #choose open/periodic randomly
            bc = np.random.choice(['periodic','open'])
            if bc == "periodic":
                return periodic_MPO(length,O)
            elif bc == "open":
                return open_MPO(length,O,V,W)

    def set_entry(self,site,tensor):
        self.node[site] = tensor
        if site == 0:
            if np.size(np.shape(tensor))>3:
                self.node[site].legs = "both"
            else:
                self.node[site].legs = "right"
        elif site == self.length-1:
            if np.size(np.shape(tensor))>3:
                self.node[site].legs = "both"
            else:
                self.node[site].legs = "left"
        else:
            self.node[site].legs = "left"

    def exp(self,psi1,psi2=None):
        if psi2 is None:
            network = rail_network(psi1,psi1,self)
        else:
            network = rail_network(psi1,psi2,self)
        network.contract()
        return network.contraction

    def act_on(self,psi):
        network = rail_network(psi,Q=self)
        print("MPO|MPS>...")
        pbar=ProgressBar()
        for n in pbar(range(0,psi.length)):
            psi.node[n] = collapsed_MPO_layer.factory(layer(network,n))
        return psi

class periodic_MPS(mps):
    def __init__(self,length,A=None):
        self.length = length
        self.node = dict()
        if A is not None:
            for n in range(0,self.length):
                self.node[n] = rail_node(A,legs="both")
    #form direct sum of each tensor
    def __add__(self,MPS2):
        new_MPS = periodic_MPS(self.length)
        for n in range(0,new_MPS.length):
            shape1 = np.shape(self.node[n].tensor)
            shape2 = np.shape(MPS2.node[n].tensor)
            M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1],shape1[2]+shape2[2],)))
            for physical_index in range(0,shape1[0]):
                M[physical_index][0:shape1[1],0:shape1[2]] = self.node[n].tensor[physical_index]
                M[physical_index][shape1[1]:,shape1[2]:] = MPS2.node[n].tensor[physical_index]
            new_MPS.node[n] = rail_node(M,"both")
        return new_MPS
    
class open_MPS(mps):
    def __init__(self,length,A=None,V=None,W=None):
        self.length = length
        self.node = dict()
        if A is not None:
            self.node[0] = rail_node(V,legs="right")
            for n in range(1,self.length-1):
                self.node[n] = rail_node(A,legs="both")
            self.node[self.length-1] = rail_node(W,legs="left")

    #form direct sum of each tensor
    def __add__(self,MPS2):
        new_MPS = open_MPS(self.length)
        #left edge
        shape1 = np.shape(self.node[0].tensor)
        shape2 = np.shape(MPS2.node[0].tensor)
        M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1])))
        for physical_index in range(0,shape1[0]):
            M[physical_index][0:shape1[1]] = self.node[0].tensor[physical_index]
            M[physical_index][shape1[1]:] = MPS2.node[0].tensor[physical_index]
        new_MPS.node[0] = rail_node(M,"right")

        #middle
        for n in range(1,new_MPS.length-1):
            shape1 = np.shape(self.node[n].tensor)
            shape2 = np.shape(MPS2.node[n].tensor)
            M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1],shape1[2]+shape2[2],)))
            for physical_index in range(0,shape1[0]):
                M[physical_index][0:shape1[1],0:shape1[2]] = self.node[n].tensor[physical_index]
                M[physical_index][shape1[1]:,shape1[2]:] = MPS2.node[n].tensor[physical_index]
            new_MPS.node[n] = rail_node(M,"both")

        #right edge
        shape1 = np.shape(self.node[self.length-1].tensor)
        shape2 = np.shape(MPS2.node[MPS2.length-1].tensor)
        M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1])))
        for physical_index in range(0,shape1[0]):
            M[physical_index][0:shape1[1]] = self.node[self.length-1].tensor[physical_index]
            M[physical_index][shape1[1]:] = MPS2.node[MPS2.length-1].tensor[physical_index]
        new_MPS.node[self.length-1] = rail_node(M,"left")
        return new_MPS

class periodic_MPO(mpo):
    def __init__(self,length,Q=None):
        self.length = length
        self.node = dict()
        if Q is not None:
            for n in range(0,self.length):
                self.node[n] = rail_node(Q,legs="both")

class open_MPO(mpo):
    def __init__(self,length,Q=None,V=None,W=None):
        self.length = length
        self.node = dict()
        if Q is not None:
            #reshape so legs corrspond to: up down left right
            Q=np.einsum('ijkl->klij',Q)
            V=np.einsum('ijk->jki',V)
            W=np.einsum('ijk->jki',W)

            self.node[0] = rail_node(V,legs="right")
            for n in range(1,self.length-1):
                self.node[n] = rail_node(Q,legs="both")
            self.node[self.length-1] = rail_node(W,legs="left")

