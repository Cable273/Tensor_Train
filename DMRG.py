#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from rail_objects import *
from progressbar import ProgressBar,FormatLabel,BouncingBar,ReverseBar,Bar,AnimatedMarker,RotatingMarker
from svd_operations import svd_node_pair,svd_norm_node
from compression import svd_compress,var_compress
from MPS import *
from combine_rail_objects import *
from collapsed_layers import *
from Tensor_Train import *
import scipy.sparse.linalg as linalg
import copy
class dmrg:
    def __init__(self,H,D,psi=None):
        self.psi = copy.deepcopy(psi)
        self.H = H
        self.D = D
        #no trial given use random MPS
        if psi is None:
            if type(H) is periodic_MPO:
                self.psi = mps.random(self.H.length,np.size(self.H.node[0].tensor,axis=0),D,boundary="periodic")
            elif type(H) is open_MPO:
                self.psi = mps.random(self.H.length,np.size(self.H.node[0].tensor,axis=0),D,boundary="open")
        self.length = self.psi.length

        #values to track
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
        temp = layer(self.network,self.network.length-1)
        pbar=ProgressBar(widgets=['Energy Network',ReverseBar()])
        for n in pbar(range(self.length-2,0,-1)):
            self.R[n] = combine_clayer_layer.new_collapsed_layer(layer(self.network,n),self.R[n+1])

        self.var_R[self.network.length-1] = collapsed_layer.factory(layer(self.var_network,self.var_network.length-1))
        pbar=ProgressBar(widgets=['Variance Network',ReverseBar()])
        for n in pbar(range(self.length-2,0,-1)):
            self.var_R[n] = combine_clayer_layer.new_collapsed_layer(layer(self.var_network,n),self.var_R[n+1])

    def right_sweep(self):
        #first site
        #form eigenvalue equation
        site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[0],self.R[1])
        e,u = sp.linalg.eigh(site_H_mpo)
        M = u[:,0]
        M = M.reshape(np.shape(self.network.top_row.node[0].tensor))

        #update site and svd to shift norm
        self.psi.node[0].tensor = M
        self.psi.node[0].tensor, self.psi.node[1].tensor = svd_node_pair.left(self.psi.node[0],self.psi.node[1],D_cap = self.D)
        self.L[0] = collapsed_layer.factory(layer(self.network,0))
        self.var_L[0] = collapsed_layer.factory(layer(self.var_network,0))

        pbar=ProgressBar(widgets=['E0='+str(self.energy)+',Var='+str(self.variance)+':',Bar(marker=RotatingMarker())])
        for n in pbar(range(1,self.length-1)):
            site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[n],self.L[n-1],self.R[n+1])
            # e,u = sp.sparse.linalg.eigsh(site_H_mpo,k=1)
            e,u = np.linalg.eigh(site_H_mpo)
            M = u[:,0]
            shape_A=np.shape(self.network.top_row.node[n].tensor)
            new_phys_dim = int(np.size(M)/(shape_A[1]*shape_A[2]))
            M = M.reshape(np.array((new_phys_dim,shape_A[1],shape_A[2])))

            #update site 
            self.psi.node[n].tensor = M

            #update errors
            clayer = combine_clayer_layer.new_collapsed_layer(self.L[n-1],layer(self.network,n))
            self.energy = combine_collapsed_layers.scalar(clayer,self.R[n+1])
            self.energy_vals = np.append(self.energy_vals,self.energy)
            clayer = combine_clayer_layer.new_collapsed_layer(self.var_L[n-1],layer(self.var_network,n))
            self.variance = combine_collapsed_layers.scalar(clayer,self.var_R[n+1]) - self.energy**2
            self.variance_vals = np.append(self.variance_vals,self.variance)

            #shift norm
            self.psi.node[n].tensor, self.psi.node[n+1].tensor = svd_node_pair.left(self.psi.node[n],self.psi.node[n+1],D_cap = self.D)

            #form new L blocks
            self.L[n] = combine_clayer_layer.new_collapsed_layer(self.L[n-1],layer(self.network,n))
            self.var_L[n] = combine_clayer_layer.new_collapsed_layer(self.var_L[n-1],layer(self.var_network,n))


        self.psi.node[self.length-1].tensor = svd_norm_node.left(self.psi.node[self.length-1])
        self.energy = combine_clayer_layer.scalar(self.L[self.length-2],layer(self.network,self.length-1))
        self.energy_vals = np.append(self.energy_vals,self.energy)
        self.variance = combine_clayer_layer.scalar(self.var_L[self.length-2],layer(self.var_network,self.length-1))-self.energy**2
        self.variance_vals = np.append(self.variance_vals,self.variance)

    def left_sweep(self):
        #first site
        site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[self.length-1],self.L[self.length-2])
        e,u = sp.linalg.eigh(site_H_mpo)
        M = u[:,0]
        M = M.reshape(np.shape(self.network.top_row.node[self.length-1].tensor))

        #update site and svd to shift norm
        self.psi.node[self.length-1].tensor = M
        self.psi.node[self.length-2].tensor, self.psi.node[self.length-1].tensor = svd_node_pair.right(self.psi.node[self.length-2],self.psi.node[self.length-1],D_cap = self.D)
        self.R[self.length-1] = collapsed_layer.factory(layer(self.network,self.length-1))
        self.var_R[self.length-1] = collapsed_layer.factory(layer(self.var_network,self.length-1))

        pbar=ProgressBar(widgets=['E0='+str(self.energy)+',Var='+str(self.variance)+':',ReverseBar(marker=RotatingMarker())])
        for n in pbar(range(self.length-2,0,-1)):
            site_H_mpo = combine_mpoNode_clayers.factory(self.network.mid_row.node[n],self.L[n-1],self.R[n+1])
            # e,u = sp.sparse.linalg.eigsh(site_H_mpo,k=1)
            e,u = np.linalg.eigh(site_H_mpo)
            M = u[:,0]
            shape_A=np.shape(self.network.top_row.node[n].tensor)
            new_phys_dim = int(np.size(M)/(shape_A[1]*shape_A[2]))
            M = M.reshape(np.array((new_phys_dim,shape_A[1],shape_A[2])))

            #update site
            self.psi.node[n].tensor = M

            #update errors
            clayer = combine_clayer_layer.new_collapsed_layer(self.L[n-1],layer(self.network,n))
            self.energy = combine_collapsed_layers.scalar(clayer,self.R[n+1])
            self.energy_vals = np.append(self.energy_vals,self.energy)
            clayer = combine_clayer_layer.new_collapsed_layer(self.var_L[n-1],layer(self.var_network,n))
            self.variance = combine_collapsed_layers.scalar(clayer,self.var_R[n+1]) - self.energy**2
            self.variance_vals = np.append(self.variance_vals,self.variance)

            #shift norm
            self.psi.node[n-1].tensor, self.psi.node[n].tensor = svd_node_pair.right(self.psi.node[n-1],self.psi.node[n],D_cap = self.D)

            #update R blocks
            self.R[n] = combine_clayer_layer.new_collapsed_layer(layer(self.network,n),self.R[n+1])
            self.var_R[n] = combine_clayer_layer.new_collapsed_layer(layer(self.var_network,n),self.var_R[n+1])


        self.psi.node[0].tensor = svd_norm_node.right(self.psi.node[0])
        self.energy = combine_clayer_layer.scalar(layer(self.network,0),self.R[1])
        self.energy_vals = np.append(self.energy_vals,self.energy)
        self.variance = combine_clayer_layer.scalar(layer(self.var_network,0),self.var_R[1])-self.energy**2
        self.variance_vals = np.append(self.variance_vals,self.variance)

    def run(self,max_sweeps=100,var_tol=1e-15,var_diff_tol=1e-18,convergence=1e-15):
        max_sweeps = 100
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

class idmrg:
    def __init__(self,H,phys_dim,D):
        self.D = D
        self.phys_dim = phys_dim
        self.H = H

        #assuming uniform MPS of the form LAAAAAR
        self.H_left_boundary_tensor = self.H.node[0].tensor
        self.H_right_boundary_tensor = self.H.node[self.H.length-1].tensor
        self.H_internal_tensor = self.H.node[1].tensor

        #setup MPO to grow with chain. Contract 2 site MPO to form H and get GS by ED
        if type(H) is open_MPO:
            self.H_grown = open_MPO(2)
            self.psi_grown = open_MPS(2)
            self.H_grown.set_entry(0,self.H_left_boundary_tensor,"right")
            self.H_grown.set_entry(1,self.H_right_boundary_tensor,"left")
            H = np.einsum('aci,bdi->abcd',self.H_grown.node[0].tensor,self.H_grown.node[1].tensor)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        elif type(H) is periodic_MPO:
            self.H_grown = periodic_MPO(2)
            self.psi_grown = periodic_MPS(2)
            self.H_grown.set_entry(0,self.H_left_boundary_tensor,"both")
            self.H_grown.set_entry(1,self.H_right_boundary_tensor,"both")
            H = np.einsum('acef,bdfe->abcd',self.H_grown.node[0].tensor,self.H_grown.node[1].tensor)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))

        e,u = np.linalg.eigh(H)
        gs = u[:,0]
        gs = gs.reshape((phys_dim,phys_dim))
        U,S,Vh = np.linalg.svd(gs)

        A = U
        B = np.transpose(Vh)

        # if type(H) is open_MPO:
        self.psi_grown.set_entry(0,A,"right")
        self.psi_grown.set_entry(1,B,"left")
        self.length = self.psi_grown.length

        #form L=A, R=B blocks for efficient H construction
        self.network = rail_network(self.psi_grown,self.psi_grown,self.H_grown)
        self.L = collapsed_layer.factory(layer(self.network,0))
        self.R = collapsed_layer.factory(layer(self.network,1))

        #inital guess for lanczos
        self.v0 = None
        self.A_shape = np.shape(A)

    #add two sites to centre of chain, optimize and convert to mps nodes
    def grow(self,keep_singular=0):
        #add two sites
        psi_grown = open_MPS(self.length+2)
        H_grown = open_MPO(psi_grown.length)
        if type(self.H) is open_MPO:
            psi_grown.set_entry(0,self.psi_grown.node[0].tensor,"right")
            H_grown.set_entry(0,self.H_grown.node[0].tensor,"right")
        elif type(self.H) is periodic_MPO:
            psi_grown.set_entry(0,self.psi_grown.node[0].tensor,"both")
            H_grown.set_entry(0,self.H_grown.node[0].tensor,"both")

        c=1
        for n in range(1,int(self.length/2)):
            psi_grown.set_entry(c,self.psi_grown.node[n].tensor,"both")
            H_grown.set_entry(c,self.H_grown.node[n].tensor,"both")
            c=c+1
        c=int(psi_grown.length/2+1)
        for n in range(int(self.length/2),self.length-1):
            psi_grown.set_entry(c,self.psi_grown.node[n].tensor,"both")
            H_grown.set_entry(c,self.H_grown.node[n].tensor,"both")
            c=c+1

        if type(self.H) is open_MPO:
            psi_grown.set_entry(psi_grown.length-1,self.psi_grown.node[self.psi_grown.length-1].tensor,"left")
            H_grown.set_entry(H_grown.length-1,self.H_grown.node[self.H_grown.length-1].tensor,"left")
        elif type(self.H) is periodic_MPO:
            psi_grown.set_entry(psi_grown.length-1,self.psi_grown.node[self.psi_grown.length-1].tensor,"both")
            H_grown.set_entry(H_grown.length-1,self.H_grown.node[self.H_grown.length-1].tensor,"both")

        if type(self.H) is open_MPO:
            temp_A =  np.einsum('abc,gjbi->gjaic',self.L.tensor,self.H_internal_tensor)
            temp_B = np.einsum('hkie,def->hkdif',self.H_internal_tensor,self.R.tensor)
            H_opt = np.einsum('gjaic,hkdif->ghadjkcf',temp_A,temp_B)
        elif type(self.H) is periodic_MPO:
            temp_A =  np.einsum('ijklmn,acmo->acijklno',self.L.tensor,self.H_internal_tensor)
            temp_B = np.einsum('pqrijk,bdoq->bdprijko',self.H_internal_tensor,self.R.tensor)
            H_opt = np.einsum('acijklno,bdprijko->labpncdr',temp_A,temp_B)

        shape = np.shape(H_opt)
        H_opt = H_opt.reshape(shape[0]*shape[1]*shape[2]*shape[3],shape[4]*shape[5]*shape[6]*shape[7])
        # e,u = sp.sparse.linalg.eigsh(H_opt)
        if self.v0 is None:
            # e,u = sp.sparse.linalg.eigsh(H_opt,k=1)
            e,u = np.linalg.eigh(H_opt)
        else:
            # e,u = sp.sparse.linalg.eigsh(H_opt,v0=self.v0,k=1)
            e,u= np.linalg.eigh(H_opt)
        M = u[:,0]
        if np.max(self.A_shape) == self.D:
            self.v0 = M
        L_shape = np.shape(self.L.tensor)
        R_shape = np.shape(self.R.tensor)
        W_shape = np.shape(self.H_internal_tensor)

        M = M.reshape(W_shape[0],W_shape[0],L_shape[0],R_shape[0])

        #reshape node pair to two mps nodes
        M = np.einsum('ijku->ikju',M)
        shape = np.shape(M)
        M = M.reshape(np.array((shape[0]*shape[1],shape[2]*shape[3])))
        A,S,B = np.linalg.svd(M,full_matrices=False)
        A=A[:,0:self.D]
        S=S[0:self.D]
        B=B[0:self.D,:]
        if keep_singular == 1:
            A = np.dot(A,np.power(np.diag(S),0.5))
            B = np.dot(np.power(np.diag(S),0.5),B)

        A = A.reshape(np.array((shape[0],shape[1],np.size(S))))

        B = B.reshape(np.array((np.size(S),shape[2],shape[3])))
        B = np.einsum('ijk->jik',B)

        self.A_shape = np.shape(A)

        index = int(psi_grown.length/2-1)
        psi_grown.set_entry(index,A,"both")
        psi_grown.set_entry(index+1,B,"both")
        H_grown.set_entry(index,self.H_internal_tensor,"both")
        H_grown.set_entry(index+1,self.H_internal_tensor,"both")
        self.H_grown = H_grown

        self.psi_grown = psi_grown
        self.length = self.psi_grown.length

        self.network = rail_network(self.psi_grown,self.psi_grown,H_grown)
        self.L = combine_clayer_layer.new_collapsed_layer(self.L,layer(self.network,index))
        self.R = combine_clayer_layer.new_collapsed_layer(layer(self.network,index+1),self.R)

    def run(self,N):
        print("Infinite DMRG to L="+str(N))
        pbar=ProgressBar()
        for n in pbar(range(0,int((N-2)/2)-1)):
            self.grow()
        self.grow(keep_singular = 1)
        H2 = self.H.dot(self.H)
        self.E = self.H.exp(self.psi_grown)
        self.var = H2.exp(self.psi_grown)-self.E**2
        print("E= "+str(self.E)+", var= "+str(self.var))
        return self.psi_grown
