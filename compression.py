#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy as np
from Tensor_Train import *
from progressbar import ProgressBar,FormatLabel,BouncingBar,ReverseBar,Bar
from svd_operations import svd_node_pair
from MPS import *
import copy
def diff_norm_mixed_mps(psi,site,norm):
    if type(psi) is open_MPS:
        if site==0 or site==psi.length-1:
            M = np.einsum('ij,ik->jk',np.conj(psi.node[site].tensor),psi.node[site].tensor)
            return 1-np.trace(M)/norm
        else:
            M_dagger = np.conj(np.einsum('ijk->ikj',psi.node[site].tensor))
            M = np.einsum('ijk,ikc->jc',M_dagger,psi.node[site].tensor)
            return 1-np.trace(M)/norm
    else:
        M_dagger = np.conj(np.einsum('ijk->ikj',psi.node[site].tensor))
        M = np.einsum('ijk,ikc->jc',M_dagger,psi.node[site].tensor)
        return 1-np.trace(M)/norm

class svd_compress:
    def __init__(self,psi,D):
        self.psi = copy.deepcopy(psi)
        self.psi_orig = copy.deepcopy(psi)
        self.length = psi.length
        self.D = D

    def compress(self,norm=False,verbose=False):
        self.psi.right_normalize()

        if verbose is True:
            pbar=ProgressBar()
            for n in pbar(range(0,self.length-1)):
                self.psi.node[n].tensor,self.psi.node[n+1].tensor = svd_node_pair.left(self.psi.node[n],self.psi.node[n+1],D_cap=self.D)
        else:
            for n in range(0,self.length-1):
                self.psi.node[n].tensor,self.psi.node[n+1].tensor = svd_node_pair.left(self.psi.node[n],self.psi.node[n+1],D_cap=self.D)

        cross_network = rail_network(self.psi_orig.conj(),self.psi)
        cross_network.contract()
        cross_term = cross_network.contraction
        self.error = np.abs(self.psi_orig.dot(self.psi_orig) - cross_term + np.conj(cross_term) + self.psi.dot(self.psi))**2
        if verbose is True:
            print("SVD Compression error="+str(self.error))
        return self.psi

class var_compress:
    def __init__(self,psi,D,psi_trial=None,verbose=False):
        self.psi = copy.deepcopy(psi)
        self.length = psi.length
        self.psi_norm=self.psi.dot(self.psi)
        self.verbose = verbose
        self.error = None
        self.distance = []

        if psi_trial is None:
            if type(psi) is periodic_MPS:
                self.psi_trial = mps.random(self.length,np.size(psi.node[0].tensor,axis=0),D,boundary="periodic")
            elif type(psi) is open_MPS:
                self.psi_trial = mps.random(self.length,np.size(psi.node[0].tensor,axis=0),D,boundary="open")
        else:
            self.psi_trial = psi_trial
        self.network = rail_network(self.psi_trial,self.psi)

        self.cross_term1 = rail_network(self.psi.conj(),self.psi_trial)
        self.cross_term2 = rail_network(self.psi_trial.conj(),self.psi)

        #Build R matrix of <psi_trial|psi>
        self.overlap_R = dict()
        self.overlap_L = dict()
        temp = layer(self.cross_term1,self.network.length-1)
        self.overlap_R[self.cross_term1.length-1] = collapsed_layer.factory(layer(self.cross_term1,self.cross_term1.length-1))
        for n in range(self.length-2,0,-1):
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.overlap_R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.overlap_R[n+1])


        self.R=dict()
        self.L=dict()
        temp = layer(self.network,self.network.length-1)
        self.R[self.network.length-1] = collapsed_layer.factory(layer(self.network,self.network.length-1))
        for n in range(self.length-2,0,-1):
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.R[n+1])

    def right_sweep(self):
        #first site
        if type(self.psi) is periodic_MPS:
            # self.psi_trial.node[0].tensor = np.einsum('ijk,akbc->iabc',self.psi.node[0].tensor,self.R[1].tensor)
            self.psi_trial.node[0].tensor = np.einsum('ijk,akbj->iba',self.psi.node[0].tensor,self.R[1].tensor)
        else:
            self.psi_trial.node[0].tensor = np.einsum('ij,kj->ik',self.psi.node[0].tensor,self.R[1].tensor)

        #Shift norm with svd + generate left blocks
        self.psi_trial.node[0].tensor, self.psi_trial.node[1].tensor = svd_node_pair.left(self.psi_trial.node[0],self.psi_trial.node[1])
        self.L[0] = collapsed_layer.factory(layer(self.network,0))
        self.overlap_L[0] = collapsed_layer.factory(layer(self.cross_term1,0))

        # if self.verbose is True:
            # pbar=ProgressBar(widgets=[str(self.error)+':',Bar()])
        for n in range(1,self.length-1):
            if type(self.psi) is periodic_MPS:
                self.psi_trial.node[n].tensor = np.einsum('ikja,bac,dcik->bjd',self.L[n-1].tensor, self.psi.node[n].tensor, self.R[n+1].tensor)
            else:
                self.psi_trial.node[n].tensor = np.einsum('ij,ajk,bk->aib',self.L[n-1].tensor, self.psi.node[n].tensor, self.R[n+1].tensor)
            #shift norm + grow left block
            self.psi_trial.node[n].tensor, self.psi_trial.node[n+1].tensor = svd_node_pair.left(self.psi_trial.node[n], self.psi_trial.node[n+1])
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.L[n] = combine_collapsed_layers.new_collapsed_layer(self.L[n-1],clayer)
            clayer = collapsed_layer.factory(layer(self.cross_term1,n))
            self.overlap_L[n] = combine_collapsed_layers.new_collapsed_layer(self.overlap_L[n-1],clayer)

        cross_overlap = np.einsum('ab,ca,cb',self.overlap_L[self.length-2].tensor,self.psi.node[self.length-1].tensor,self.psi_trial.node[self.length-1].tensor)
        psi_trial_norm = np.einsum('ca,ca',self.psi_trial.node[self.length-1].tensor,self.psi_trial.node[self.length-1].tensor)
        self.error = np.abs(self.psi_norm-cross_overlap+np.conj(cross_overlap)+psi_trial_norm)**2
        self.distance = np.append(self.distance,self.error)

    def left_sweep(self):
        #last site
        if type(self.psi) is periodic_MPS:
            self.psi_trial.node[self.length-1].tensor = np.einsum('abcd,udb->uca',self.L[self.length-2].tensor, self.psi.node[self.length-1].tensor)
        else:
            self.psi_trial.node[self.length-1].tensor = np.einsum('ab,cb->ca',self.L[self.length-2].tensor, self.psi.node[self.length-1].tensor)
        #shift norm + grow left block
        self.psi_trial.node[self.length-2].tensor, self.psi_trial.node[self.length-1].tensor = svd_node_pair.right(self.psi_trial.node[self.length-2], self.psi_trial.node[self.length-1])
        self.R[self.length-1] = collapsed_layer.factory(layer(self.network,self.length-1))
        self.overlap_R[self.length-1] = collapsed_layer.factory(layer(self.cross_term1,self.length-1))


        # if self.verbose is True:
            # pbar=ProgressBar(widgets=[str(self.error)+':',ReverseBar()])
        for n in range(self.length-2,0,-1):
            if type(self.psi) is periodic_MPS:
                self.psi_trial.node[n].tensor = np.einsum('ikja,bac,dcik->bjd',self.L[n-1].tensor, self.psi.node[n].tensor, self.R[n+1].tensor)
            else:
                self.psi_trial.node[n].tensor = np.einsum('ij,ajk,bk->aib',self.L[n-1].tensor, self.psi.node[n].tensor, self.R[n+1].tensor)
            #shift norm + grow left block
            self.psi_trial.node[n-1].tensor, self.psi_trial.node[n].tensor = svd_node_pair.right(self.psi_trial.node[n-1], self.psi_trial.node[n])
            clayer = collapsed_layer.factory(layer(self.network,n))
            self.R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.R[n+1])
            clayer = collapsed_layer.factory(layer(self.cross_term1,n))
            self.overlap_R[n] = combine_collapsed_layers.new_collapsed_layer(clayer,self.overlap_R[n+1])

        cross_overlap = np.einsum('ab,ca,cb',self.overlap_R[1].tensor,self.psi.node[0].tensor,self.psi_trial.node[0].tensor)
        psi_trial_norm = np.einsum('ca,ca',self.psi_trial.node[0].tensor,self.psi_trial.node[0].tensor)
        self.error = np.abs(self.psi_norm-cross_overlap+np.conj(cross_overlap)+psi_trial_norm)**2
        self.distance = np.append(self.distance,self.error)

    def compress(self,min_error,max_sweeps=500,convergence=1e-6):
        self.right_sweep()
        sweep_score = np.zeros(2)
        sweep_score = self.distance[int(np.size(self.distance)-1)]
        converged = 0
        sweeps=1
        while converged == 0:
            self.left_sweep()
            self.right_sweep()
            sweeps = sweeps + 2
            new_score = self.distance[int(np.size(self.distance)-1)]
            diff = np.abs(self.distance[int(np.size(self.distance)-1)]-self.distance[int(np.size(self.distance)-2)])
            if sweeps > max_sweeps:
                if self.verbose is True:
                    print("Max sweeps reached, "+str(sweeps)+" sweeps, error="+str(sweep_score))
                break
            elif diff < convergence:
                if self.verbose is True:
                    print("Error converged, "+str(sweeps)+" sweeps, error="+str(sweep_score))
                break
            elif sweep_score < min_error:
                if self.verbose is True:
                    print("Minimum error reached, "+str(sweeps)+" sweeps, error="+str(sweep_score))
                break
        return self.psi_trial
