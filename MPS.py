#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from Tensor_Train import *
from progressbar import ProgressBar
from svd_operations import svd_node_pair,svd_norm_node
import copy
class mps:
    def uniform(length,A,V=None,W=None):
        if V is None and W is None: return periodic_MPS(length,A)
        else: return open_MPS(length,A,V,W)

    def random(length,on_site_dim,bond_dim,boundary=None):
        A = np.random.uniform(-1,1,np.array((on_site_dim,bond_dim,bond_dim)))
        V = np.random.uniform(-1,1,np.array((on_site_dim,bond_dim)))
        W = np.random.uniform(-1,1,np.array((on_site_dim,bond_dim)))
        if boundary == "periodic": return periodic_MPS(length,A)
        elif boundary == "open": return open_MPS(length,A,V,W)
        else: #choose open/periodic randomly
            bc = np.random.choice(['periodic','open'])
            if bc == "periodic":
                return periodic_MPS(length,A)
            elif bc == "open":
                return open_MPS(length,A,V,W)

    def set_entry(self,site,tensor,legs):
        self.node[site].tensor = tensor
        self.node[site].legs = legs

    def dot(self,MPS2):
        network = rail_network(self.conj(),MPS2)
        network.contract()
        return network.contraction
    
    def left_normalize(self,norm=False,verbose=False):
        if verbose is True:
            print("Left normalizing MPS")
            pbar=ProgressBar()
            for n in pbar(range(0,self.length-1)):
                self.node[n].tensor,self.node[n+1].tensor = svd_node_pair.left(self.node[n],self.node[n+1])
        else:
            for n in range(0,self.length-1):
                self.node[n].tensor,self.node[n+1].tensor = svd_node_pair.left(self.node[n],self.node[n+1])
        if norm is True:
            self.node[self.length-1].tensor = svd_norm_node.left(self.node[self.length-1])

    def right_normalize(self,norm=False,rescale=False,verbose=False):
        pbar=ProgressBar()
        if verbose is True:
            print("Right normalizing MPS")
            for n in pbar(range(self.length-1,0,-1)):
                self.node[n-1].tensor,self.node[n].tensor = svd_node_pair.right(self.node[n-1],self.node[n])
        else:
            for n in range(self.length-1,0,-1):
                self.node[n-1].tensor,self.node[n].tensor = svd_node_pair.right(self.node[n-1],self.node[n])
        if norm is True:
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

    def conj(self):
        psi_conj = copy.deepcopy(self)
        for n in range(0,self.length-1):
            if psi_conj.node is not None:
                psi_conj.node[n].tensor = np.conj(psi_conj.node[n].tensor)
        return psi_conj

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

    def set_entry(self,site,tensor,legs):
        self.node[site].tensor = tensor
        self.node[site].legs = legs

    def exp(self,psi1,psi2=None):
        if psi2 is None:
            network = rail_network(psi1.conj(),psi1,self)
        else:
            network = rail_network(psi1.conj(),psi2,self)
        network.contract()
        return network.contraction

    def dot(self,psi):
        if type(psi) is periodic_MPS or type(psi) is open_MPS:
            network = rail_network(psi,Q=self)
            pbar=ProgressBar()
            for n in range(0,psi.length):
                psi.node[n] = collapsed_MPO_layer.factory(layer(network,n))
            return psi
        elif type(psi) is periodic_MPO or type(psi) is open_MPO:
            new_mpo = copy.deepcopy(psi)
            for n in range(0,self.length):
                new_mpo.node[n] = combine_mpo_nodes_vertical.factory(self.node[n],psi.node[n])
            return new_mpo

class periodic_MPS(mps):
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
        for n in range(0,self.length):
            self.node[n] = rail_node()
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
        for n in range(0,self.length):
            self.node[n] = rail_node()
        if Q is not None:
            for n in range(0,self.length):
                self.node[n] = rail_node(Q,legs="both")

class open_MPO(mpo):
    def __init__(self,length,Q=None,V=None,W=None):
        self.length = length
        self.node = dict()
        for n in range(0,self.length):
            self.node[n] = rail_node()
        if Q is not None:
            #reshape so legs corrspond to: up down left right
            Q=np.einsum('ijkl->klij',Q)
            V=np.einsum('ijk->jki',V)
            W=np.einsum('ijk->jki',W)

            self.node[0] = rail_node(V,legs="right")
            for n in range(1,self.length-1):
                self.node[n] = rail_node(Q,legs="both")
            self.node[self.length-1] = rail_node(W,legs="left")
