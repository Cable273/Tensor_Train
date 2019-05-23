#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from Tensor_Train import *
from progressbar import ProgressBar,FormatLabel,BouncingBar,ReverseBar,Bar,AnimatedMarker,RotatingMarker
from svd_operations import svd_node_pair,svd_norm_node
from compression import svd_compress,var_compress
from MPS import *
import scipy.sparse.linalg as linalg
import copy
class dmrg:
    def __init__(self,H,D,psi=None):
        self.psi = copy.deepcopy(psi)
        self.H = H
        self.D = D
        if psi is None:
            if type(H) is periodic_MPO:
                self.psi = mps.random(self.H.length,np.size(self.H.node[0].tensor,axis=0),D,boundary="periodic")
            elif type(H) is open_MPO:
                self.psi = mps.random(self.H.length,np.size(self.H.node[0].tensor,axis=0),D,boundary="open")
        self.length = self.psi.length

        self.energy = None
        self.energy_vals = []
        self.e_diff = None
        self.variance = None
        self.variance_vals = []
        self.H2 = self.H.dot(self.H)
        self.var_network = rail_network(self.psi,self.psi,self.H2)

        print("Right Normalize trial state")
        self.psi.right_normalize(norm=True)

        self.network = rail_network(self.psi,self.psi,self.H)
        #combine left/right sides of <psi|psi>, update iteratively to update info
        self.R=dict()
        self.L=dict()
        self.var_R=dict()
        self.var_L=dict()

        print("Build initial right blocks")
        self.R[self.network.length-1] = collapsed_layer.factory(layer(self.network,self.network.length-1))
        pbar=ProgressBar(widgets=['Energy Network',ReverseBar()])
        for n in pbar(range(self.length-2,0,-1)):
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.R[n+1])

        self.var_R[self.network.length-1] = collapsed_layer.factory(layer(self.var_network,self.var_network.length-1))
        pbar=ProgressBar(widgets=['Variance Network',ReverseBar()])
        for n in pbar(range(self.length-2,0,-1)):
            clayer = collapsed_layer.factory(layer(self.var_network,n))
            self.var_R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.var_R[n+1])

    def right_sweep(self):
        #first site
        #form eigenvalue equation
        site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[0],self.R[1])

        #solve via...
        # shape = np.shape(self.network.top_row.node[0].tensor)
        # M0 = self.network.top_row.node[0].tensor.reshape(np.array((shape[0]*shape[1])))
        # e,u = linalg.eigsh(site_H_mpo,1,v0=M0,which='LM')
        e,u = sp.linalg.eigh(site_H_mpo)
        M = u[:,0]
        M = M.reshape(np.shape(self.network.top_row.node[0].tensor))

        #update site and svd to shift norm
        self.psi.node[0].tensor = M
        self.psi.node[0].tensor, self.psi.node[1].tensor = svd_node_pair.left(self.psi.node[0],self.psi.node[1])
        self.L[0] = collapsed_layer.factory(layer(self.network,0))
        self.var_L[0] = collapsed_layer.factory(layer(self.var_network,0))

        pbar=ProgressBar(widgets=['E0='+str(self.energy)+',Var='+str(self.variance)+':',Bar(marker=RotatingMarker())])
        for n in pbar(range(1,self.length-1)):
            site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[n],self.L[n-1],self.R[n+1])

            # solve via...
            # shape = np.shape(self.network.top_row.node[n].tensor)
            # M0 = self.network.top_row.node[n].tensor.reshape(np.array((shape[0]*shape[1]*shape[2])))
            # e,u = linalg.eigsh(site_H_mpo,1,v0=M0,which='LM')
            e,u = sp.linalg.eigh(site_H_mpo)
            M = u[:,0]
            shape_A=np.shape(self.network.top_row.node[n].tensor)
            new_phys_dim = int(np.size(M)/(shape_A[1]*shape_A[2]))
            M = M.reshape(np.array((new_phys_dim,shape_A[1],shape_A[2])))

            #update site and svd to shift norm
            self.psi.node[n].tensor = M
            self.psi.node[n].tensor, self.psi.node[n+1].tensor = svd_node_pair.left(self.psi.node[n],self.psi.node[n+1])

            clayer = collapsed_layer.factory(layer(self.network,n))
            self.L[n] = combine_collapsed_layers.new_collapsed_layer(self.L[n-1],clayer)
            clayer = collapsed_layer.factory(layer(self.var_network,n))
            self.var_L[n] = combine_collapsed_layers.new_collapsed_layer(self.var_L[n-1],clayer)

        self.psi.node[self.length-1].tensor = svd_norm_node.left(self.psi.node[self.length-1])
        # self.network.contract()
        # self.energy = self.network.contraction

        clayer = collapsed_layer.factory(layer(self.network,self.length-1))
        self.energy = combine_collapsed_layers.scalar(self.L[self.length-2],clayer)
        self.energy_vals = np.append(self.energy_vals,self.energy)

        clayer = collapsed_layer.factory(layer(self.var_network,self.length-1))
        self.variance = combine_collapsed_layers.scalar(self.var_L[self.length-2],clayer)-self.energy**2
        self.variance_vals = np.append(self.variance_vals,self.variance)

    def left_sweep(self):
        #first site
        #form eigenvalue equation
        site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[self.length-1],self.L[self.length-2])

        #solve via...
        # shape = np.shape(self.network.top_row.node[self.length-1].tensor)
        # M0 = self.network.top_row.node[self.length-1].tensor.reshape(np.array((shape[0]*shape[1])))
        # e,u = linalg.eigsh(site_H_mpo,1,v0=M0,which='LM')
        e,u = sp.linalg.eigh(site_H_mpo)
        M = u[:,0]
        M = M.reshape(np.shape(self.network.top_row.node[self.length-1].tensor))

        #update site and svd to shift norm
        self.psi.node[self.length-1].tensor = M
        self.psi.node[self.length-2].tensor, self.psi.node[self.length-1].tensor = svd_node_pair.right(self.psi.node[self.length-2],self.psi.node[self.length-1])
        self.R[self.length-1] = collapsed_layer.factory(layer(self.network,self.length-1))
        self.var_R[self.length-1] = collapsed_layer.factory(layer(self.var_network,self.length-1))

        pbar=ProgressBar(widgets=['E0='+str(self.energy)+',Var='+str(self.variance)+':',ReverseBar(marker=RotatingMarker())])
        for n in pbar(range(self.length-2,0,-1)):
            site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[n],self.L[n-1],self.R[n+1])

            # solve via...
            # shape = np.shape(self.network.top_row.node[n].tensor)
            # M0 = self.network.top_row.node[n].tensor.reshape(np.array((shape[0]*shape[1]*shape[2])))
            # e,u = linalg.eigsh(site_H_mpo,1,v0=M0,which='LM')
            e,u = sp.linalg.eigh(site_H_mpo)
            M = u[:,0]
            shape_A=np.shape(self.network.top_row.node[n].tensor)
            new_phys_dim = int(np.size(M)/(shape_A[1]*shape_A[2]))
            M = M.reshape(np.array((new_phys_dim,shape_A[1],shape_A[2])))

            #update site and svd to shift norm
            self.psi.node[n].tensor = M
            self.psi.node[n-1].tensor, self.psi.node[n].tensor = svd_node_pair.right(self.psi.node[n-1],self.psi.node[n])

            clayer = collapsed_layer.factory(layer(self.network,n))
            self.R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.R[n+1])

            clayer = collapsed_layer.factory(layer(self.var_network,n))
            self.var_R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.var_R[n+1])

        self.psi.node[0].tensor = svd_norm_node.right(self.psi.node[0])

        clayer = collapsed_layer.factory(layer(self.network,0))
        self.energy = combine_collapsed_layers.scalar(clayer,self.R[1])
        self.energy_vals = np.append(self.energy_vals,self.energy)

        clayer = collapsed_layer.factory(layer(self.var_network,0))
        self.variance = combine_collapsed_layers.scalar(clayer,self.var_R[1])-self.energy**2
        self.variance_vals = np.append(self.variance_vals,self.variance)

    def run(self,max_sweeps=100,var_tol=1e-5,var_diff_tol=1e-12,convergence=1e-5):
        self.right_sweep()
        converged = 0
        sweeps=1
        while converged==0:
            self.left_sweep()
            self.right_sweep()
            sweeps = sweeps + 2
            energy_diff = np.abs(self.energy_vals[int(np.size(self.energy_vals)-2)]-self.energy_vals[int(np.size(self.energy_vals)-1)])
            var_diff = np.abs(self.variance_vals[int(np.size(self.variance_vals)-2)]-self.variance_vals[int(np.size(self.variance_vals)-1)])
            if sweeps > max_sweeps:
                print("Max sweeps reached, "+str(sweeps)+" sweeps, energy="+str(self.energy)+", var="+str(self.variance))
                break
            if np.abs(self.variance) < var_tol: 
                print("Min variance reached, "+str(self.energy)+", var="+str(self.variance))
                break
            if var_diff < var_diff_tol:
                print("Variance converged, "+str(self.energy)+", var="+str(self.variance))
                break
            if energy_diff < convergence:
                print("Energy converged, "+str(self.energy)+", var="+str(self.variance))
                break
        self.e_diff=np.zeros(np.size(self.energy_vals))
        for n in range(0,np.size(self.energy_vals,axis=0)-1):
            self.e_diff[n] = np.abs(self.energy_vals[n+1]-self.energy_vals[n])
        return self.psi

    def plot_var(self):
        from matplotlib import rc
        rc('font',**{'size':26})
        ## for Palatino and other serif fonts use:
        #rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)
        # matplotlib.rcParams['figure.dpi'] = 400
        print(self.variance_vals)
        plt.plot(self.variance_vals,marker="s")
        plt.xlabel(r"$\textrm{Sweeps}$")
        plt.ylabel(r"$\sigma(H)$")
        plt.title(r"$\textrm{DMRG Sweep variance}(H)$")
        plt.show()

    def plot_convergence(self):
        from matplotlib import rc
        rc('font',**{'size':26})
        ## for Palatino and other serif fonts use:
        #rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)
        # matplotlib.rcParams['figure.dpi'] = 400
        plt.plot(np.log(self.e_diff),marker="s")
        plt.xlabel(r"$\textrm{Sweeps}$")
        plt.ylabel(r"$\ln(\Delta E)$")
        plt.title(r"$\textrm{DMRG Sweep variance}(H)$")
        plt.show()

