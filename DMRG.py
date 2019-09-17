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
        temp = layer(self.network,self.network.length-1)
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
        self.psi.node[0].tensor, self.psi.node[1].tensor = svd_node_pair.left(self.psi.node[0],self.psi.node[1],D_cap = self.D)
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

            #update site 
            self.psi.node[n].tensor = M

            #update errors
            clayer = collapsed_layer.factory(layer(self.network,n))
            clayer = combine_collapsed_layers.new_collapsed_layer(self.L[n-1],clayer)
            self.energy = combine_collapsed_layers.scalar(clayer,self.R[n+1])
            self.energy_vals = np.append(self.energy_vals,self.energy)
            clayer = collapsed_layer.factory(layer(self.var_network,n))
            clayer = combine_collapsed_layers.new_collapsed_layer(self.var_L[n-1],clayer)
            self.variance = combine_collapsed_layers.scalar(clayer,self.var_R[n+1]) - self.energy**2
            self.variance_vals = np.append(self.variance_vals,self.variance)

            #shift norm
            self.psi.node[n].tensor, self.psi.node[n+1].tensor = svd_node_pair.left(self.psi.node[n],self.psi.node[n+1],D_cap = self.D)

            #form new L blocks
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.L[n] = combine_collapsed_layers.new_collapsed_layer(self.L[n-1],clayer)
            clayer = collapsed_layer.factory(layer(self.var_network,n))
            self.var_L[n] = combine_collapsed_layers.new_collapsed_layer(self.var_L[n-1],clayer)


        self.psi.node[self.length-1].tensor = svd_norm_node.left(self.psi.node[self.length-1])

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
        self.psi.node[self.length-2].tensor, self.psi.node[self.length-1].tensor = svd_node_pair.right(self.psi.node[self.length-2],self.psi.node[self.length-1],D_cap = self.D)
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

            #update site
            self.psi.node[n].tensor = M

            #update errors
            clayer = collapsed_layer.factory(layer(self.network,n))
            clayer = combine_collapsed_layers.new_collapsed_layer(self.L[n-1],clayer)
            self.energy = combine_collapsed_layers.scalar(clayer,self.R[n+1])
            self.energy_vals = np.append(self.energy_vals,self.energy)
            clayer = collapsed_layer.factory(layer(self.var_network,n))
            clayer = combine_collapsed_layers.new_collapsed_layer(self.var_L[n-1],clayer)
            self.variance = combine_collapsed_layers.scalar(clayer,self.var_R[n+1]) - self.energy**2
            self.variance_vals = np.append(self.variance_vals,self.variance)

            #shift norm
            self.psi.node[n-1].tensor, self.psi.node[n].tensor = svd_node_pair.right(self.psi.node[n-1],self.psi.node[n],D_cap = self.D)

            #update R blocks
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
    #init with 4 site MPO + g 
    def __init__(self,H_4site,phys_dim,D):
        self.D = D
        method = dmrg(H_4site,self.D)
        self.psi =  method.run()

        self.psi_grown = copy.deepcopy(self.psi)
        self.H_4site = H_4site
        self.H_grown = copy.deepcopy(self.H_4site)
        self.phys_dim = phys_dim
        self.length = self.psi.length

        #fix norm AASBB
        self.psi.node[0].tensor, self.psi.node[1].tensor = svd_node_pair.left(self.psi.node[0],self.psi.node[1],D_cap=self.D)
        self.psi.node[2].tensor,self.psi.node[3].tensor = svd_node_pair.right(self.psi.node[2],self.psi.node[3],D_cap=self.D)

        T = np.einsum('ijk,ukv->ijuv',self.psi.node[1].tensor,self.psi.node[2].tensor)
        shape0 = np.shape(T)
        T = T.reshape((np.array((shape0[0]*shape0[1],shape0[2]*shape0[3]))))
        
        U,S,Vh = np.linalg.svd(T,full_matrices=False)
        U=U[:,0:self.D]
        S=S[0:self.D]
        Vh=Vh[0:self.D,:]
        U = np.dot(U,np.power(np.diag(S),0.5))
        Vh = np.dot(np.power(np.diag(S),0.5),Vh)

        A = U.reshape(np.array((shape0[0],shape0[1],np.size(S))))

        Vh = Vh.reshape(np.array((np.size(S),shape0[2],shape0[3])))
        B = np.einsum('ijk->jik',Vh)
        shape = np.shape(B)
        B = B.reshape(np.array((shape[0],shape[1],shape[2])))

        self.psi.node[1].tensor = A
        self.psi.node[2].tensor = B


        #form L=AA, R=BB blocks for efficient H construction
        self.network = rail_network(self.psi_grown,self.psi_grown,self.H_4site)
        clayer0 = collapsed_layer.factory(layer(self.network,0))
        clayer1 = collapsed_layer.factory(layer(self.network,1))
        self.L = combine_collapsed_layers.new_collapsed_layer(clayer0,clayer1)

        clayer2 = collapsed_layer.factory(layer(self.network,2))
        clayer3 = collapsed_layer.factory(layer(self.network,3))
        self.R = combine_collapsed_layers.new_collapsed_layer(clayer2,clayer3)

    #add two sites to centre of chain, optimize and convert to mps nodes
    def grow(self):
        #add two sites
        psi_grown = open_MPS(self.length+2)
        H_grown = open_MPO(psi_grown.length)
        psi_grown.set_entry(0,self.psi_grown.node[0].tensor,"right")
        H_grown.set_entry(0,self.H_grown.node[0].tensor,"right")
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
        psi_grown.set_entry(psi_grown.length-1,self.psi_grown.node[self.psi_grown.length-1].tensor,"left")
        H_grown.set_entry(H_grown.length-1,self.H_grown.node[self.H_grown.length-1].tensor,"both")

        #construct eigenvalue prob to optimize 2 sites
        H_opt = np.einsum('abc,gjbi,hkie,def->ghadjkcf',self.L.tensor,self.H_4site.node[1].tensor,self.H_4site.node[2].tensor,self.R.tensor)
        shape = np.shape(H_opt)
        H_opt = H_opt.reshape(shape[0]*shape[1]*shape[2]*shape[3],shape[4]*shape[5]*shape[6]*shape[7])
        e,u = np.linalg.eigh(H_opt)
        M = u[:,0]
        L_shape = np.shape(self.L.tensor)
        R_shape = np.shape(self.R.tensor)
        W_shape = np.shape(self.H_4site.node[1].tensor)

        M = M.reshape(W_shape[0],W_shape[0],L_shape[0],R_shape[0])

        #reshape node pair to two mps nodes
        M = np.einsum('ijku->ikju',M)
        shape = np.shape(M)
        M = M.reshape(np.array((shape[0]*shape[1],shape[2]*shape[3])))
        A,S,B = np.linalg.svd(M,full_matrices=False)
        A=A[:,0:self.D]
        S=S[0:self.D]
        B=B[0:self.D,:]
        A = np.dot(A,np.power(np.diag(S),0.5))
        B = np.dot(np.power(np.diag(S),0.5),B)

        A = A.reshape(np.array((shape[0],shape[1],np.size(S))))

        B = B.reshape(np.array((np.size(S),shape[2],shape[3])))
        B = np.einsum('ijk->jik',B)

        index = int(psi_grown.length/2-1)
        psi_grown.set_entry(index,A,"both")
        psi_grown.set_entry(index+1,B,"both")
        H_grown.set_entry(index,self.H_4site.node[1].tensor,"both")
        H_grown.set_entry(index+1,self.H_4site.node[2].tensor,"both")
        self.H_grown = H_grown

        self.psi_grown = psi_grown
        self.length = self.psi_grown.length

        self.network = rail_network(self.psi_grown,self.psi_grown,H_grown)
        clayer = collapsed_layer.factory(layer(self.network,index))
        self.L = combine_collapsed_layers.new_collapsed_layer(self.L,clayer)

        clayer = collapsed_layer.factory(layer(self.network,index+1))
        self.R = combine_collapsed_layers.new_collapsed_layer(clayer,self.R)

    def run(self,N):
        print("Infinite DMRG to L="+str(N))
        pbar=ProgressBar()
        for n in pbar(range(0,int((N-4)/2))):
            self.grow()
        return self.psi_grown
