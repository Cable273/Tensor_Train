#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from Tensor_Train import *
from progressbar import ProgressBar,FormatLabel,BouncingBar,ReverseBar,Bar
from svd_operations import svd_node_pair,svd_norm_node
from MPS import *
import copy
class dmrg:
    def __init__(self,H,D=None,psi=None):
        print(D)
        self.psi = copy.deepcopy(psi)
        self.H = H
        if psi is None:
            if type(H) is periodic_MPO:
                self.psi = mps.random(self.H.length,np.size(self.H.node[0].tensor,axis=0),D,boundary="periodic")
            elif type(H) is open_MPO:
                self.psi = mps.random(self.H.length,np.size(self.H.node[0].tensor,axis=0),D,boundary="open")
        self.length = self.psi.length

        self.energy = None
        self.energy_vals = []
        self.var = []
        print("Right Normalize trial state")
        self.psi.right_normalize()

        self.network = rail_network(self.psi,self.psi,self.H)
        #combine left/right sides of <psi|psi>, update iteratively to update info
        self.R=dict()
        self.L=dict()

        print("Build initial right blocks")
        temp = layer(self.network,self.network.length-1)
        self.R[self.network.length-1] = collapsed_layer.factory(layer(self.network,self.network.length-1))
        pbar=ProgressBar()
        for n in pbar(range(self.length-2,0,-1)):
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.R[n+1])

    def right_sweep(self):
        #first site
        #form eigenvalue equation
        site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[0],self.R[1])

        #solve via...
        e,u = np.linalg.eigh(site_H_mpo)
        M = u[:,0]
        M = M.reshape(np.shape(self.network.top_row.node[0].tensor))

        #update site and svd to shift norm
        self.psi.node[0].tensor = M
        self.psi.node[0].tensor, self.psi.node[1].tensor = svd_node_pair.left(self.psi.node[0],self.psi.node[1])
        self.L[0] = collapsed_layer.factory(layer(self.network,0))

        pbar=ProgressBar(widgets=['E0='+str(self.energy)+':',Bar()])
        for n in pbar(range(1,self.length-1)):
            site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[n],self.L[n-1],self.R[n+1])

            # solve via...
            e,u = np.linalg.eigh(site_H_mpo)
            M = u[:,0]
            shape_A=np.shape(self.network.top_row.node[n].tensor)
            new_phys_dim = int(np.size(M)/(shape_A[1]*shape_A[2]))
            M = M.reshape(np.array((new_phys_dim,shape_A[1],shape_A[2])))

            #update site and svd to shift norm
            self.psi.node[n].tensor = M
            self.psi.node[n].tensor, self.psi.node[n+1].tensor = svd_node_pair.left(self.psi.node[n],self.psi.node[n+1])
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.L[n] = combine_collapsed_layers.new_collapsed_layer(self.L[n-1],clayer)
        self.psi.node[self.length-1].tensor = svd_norm_node.left(self.psi.node[self.length-1])
        self.network.contract()
        self.energy = self.network.contraction
        self.energy_vals = np.append(self.energy_vals,self.energy)

    def left_sweep(self):
        #first site
        #form eigenvalue equation
        site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[self.length-1],self.L[self.length-2])

        #solve via...
        e,u = np.linalg.eigh(site_H_mpo)
        M = u[:,0]
        M = M.reshape(np.shape(self.network.top_row.node[self.length-1].tensor))

        #update site and svd to shift norm
        self.psi.node[self.length-1].tensor = M
        self.psi.node[self.length-2].tensor, self.psi.node[self.length-1].tensor = svd_node_pair.right(self.psi.node[self.length-2],self.psi.node[self.length-1])
        self.R[self.length-1] = collapsed_layer.factory(layer(self.network,self.length-1))

        pbar=ProgressBar(widgets=['E0='+str(self.energy)+':',ReverseBar()])
        for n in pbar(range(self.length-2,0,-1)):
            site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[n],self.L[n-1],self.R[n+1])

            # solve via...
            e,u = np.linalg.eigh(site_H_mpo)
            M = u[:,0]
            shape_A=np.shape(self.network.top_row.node[n].tensor)
            new_phys_dim = int(np.size(M)/(shape_A[1]*shape_A[2]))
            M = M.reshape(np.array((new_phys_dim,shape_A[1],shape_A[2])))

            #update site and svd to shift norm
            self.psi.node[n].tensor = M
            self.psi.node[n-1].tensor, self.psi.node[n].tensor = svd_node_pair.right(self.psi.node[n-1],self.psi.node[n])
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.R[n+1])
        self.psi.node[0].tensor = svd_norm_node.right(self.psi.node[0])
        self.network.contract()
        self.energy = self.network.contraction
        self.energy_vals = np.append(self.energy_vals,self.energy)

    def run(self,tol,max_sweeps=20):
        self.right_sweep()
        converged = 0
        sweeps=1
        while converged==0:
            self.left_sweep()
            self.right_sweep()
            sweeps = sweeps + 2
            diff = np.abs(self.energy_vals[int(np.size(self.energy_vals)-2)]-self.energy_vals[int(np.size(self.energy_vals)-1)])
            if sweeps > max_sweeps:
                print("Max sweeps reached, "+str(sweeps)+" sweeps, energy="+str(self.energy))
                break
            if diff < tol and sweeps > 10:
                print("Converged, energy="+str(self.energy))
                break
        return self.psi