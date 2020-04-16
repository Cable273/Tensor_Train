#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
from MPS import *
from rail_objects import *
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *
from progressbar import ProgressBar
import copy
from trotter_gate_applicationVidal import *
from copy import deepcopy
from Tensor_Train import *
from combine_rail_objects import *
from collapsed_layers import *
from integrators import expiH

class TDVP:
    def __init__(self,H_mpo,psi):
        self.psi_init = psi
        self.psi_init.right_normalize(norm=True)
        self.H = H_mpo
        self.phys_dim = np.shape(self.psi_init.node[0].tensor)[0]
        self.psi = copy.deepcopy(self.psi_init)
        self.psiConj = self.psi.conj()
        self.length = self.psi.length

        #TDVP combines <psi|H|psi>, <psi|psi> transfer matrices to form projectors
        #<psi|H|psi>
        self.expNetwork = rail_network(self.psi,self.psiConj,self.H)
        self.expR = dict() 
        self.expL = dict()
        #<psi|psi>
        self.overlapNetwork = rail_network(self.psi,self.psiConj)
        self.overlapR = dict() 
        self.overlapL = dict()

        #build initial R transfer matrices for projectors
        print("Build initial right blocks")
        self.expR[self.length-1] = collapsed_layer.factory(layer(self.expNetwork,self.length-1))
        self.overlapR[self.length-1] = collapsed_layer.factory(layer(self.overlapNetwork,self.length-1))
        for n in range(self.length-2,0,-1):
            self.expR[n] = combine_clayer_layer.new_collapsed_layer(layer(self.expNetwork,n),self.expR[n+1])
            self.overlapR[n] = combine_clayer_layer.new_collapsed_layer(layer(self.overlapNetwork,n),self.overlapR[n+1])

    def rightSweep(self,delta_t,integrator):
        #initial site
        #form Heff from product of exp/overlap transfer matrices
        #update site
        Pr = np.einsum('abc,cd->abd',self.expR[1].tensor,self.overlapR[1].tensor)

        H = np.einsum('ijb,abc->jcia',self.H.node[0].tensor,Pr)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        A = self.psi.node[0].tensor.reshape((dims[0]*dims[1]))
        A = integrator(H,A,delta_t)
        A = A.reshape((dims[0],dims[1]))
            
        U,S,Vh = np.linalg.svd(A,full_matrices=False)
        self.psi.node[0].tensor = U
        self.psiConj.node[0].tensor = np.conj(U)
        self.expL[0] = collapsed_layer.factory(layer(self.expNetwork,0))
        self.overlapL[0] = collapsed_layer.factory(layer(self.overlapNetwork,0))

        #update bond with backwards TDVP evolution
        C = np.dot(np.diag(S),Vh)
        Pl = np.einsum('abc,cd->abd',self.expL[0].tensor,self.overlapL[0].tensor)
        H = np.einsum('abc,dbf->cfad',Pl,Pr)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        C = C.reshape((dims[0]*dims[1]))
        C = integrator(H,C,-delta_t)
        C = C.reshape((dims[0],dims[1]))

        #multiply C into next site and continue sweep
        self.psi.node[1].tensor = np.einsum('ab,ibc->iac',C,self.psi.node[1].tensor)
        self.psiConj.node[1].tensor = np.conj(self.psi.node[1].tensor)

        for n in range(1,self.length-1):
            Pr = np.einsum('abc,cd->abd',self.expR[n+1].tensor,self.overlapR[n+1].tensor)
            Pl = np.einsum('abc,cd->abd',self.expL[n-1].tensor,self.overlapL[n-1].tensor)

            H = np.einsum('ijbe,abc->ijace',self.H.node[n].tensor,Pl)
            H = np.einsum('ijace,def->jcfiad',H,Pr)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1]*dims[2],dims[3]*dims[4]*dims[5]))
            A = self.psi.node[n].tensor.reshape((dims[0]*dims[1]*dims[2]))
            A = integrator(H,A,delta_t)
            A = A.reshape((dims[0],dims[1],dims[2]))

            dims = np.shape(A)
            A = A.reshape(dims[0]*dims[1],dims[2])

            U,S,Vh = np.linalg.svd(A,full_matrices=False)
            self.psi.node[n].tensor = U.reshape((dims[0],dims[1],np.size(S)))
            self.psiConj.node[n].tensor = np.conj(self.psi.node[n].tensor)
            self.expL[n] = combine_clayer_layer.new_collapsed_layer(self.expL[n-1],layer(self.expNetwork,n))
            self.overlapL[n] = combine_clayer_layer.new_collapsed_layer(self.overlapL[n-1],layer(self.overlapNetwork,n))

            #update bond with backwards TDVP evolution
            C = np.dot(np.diag(S),Vh)
            Pl = np.einsum('abc,cd->abd',self.expL[n].tensor,self.overlapL[n].tensor)
            H = np.einsum('abc,dbf->cfad',Pl,Pr)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
            C = C.reshape((dims[0]*dims[1]))
            C = integrator(H,C,-delta_t)
            C = C.reshape((dims[0],dims[1]))

            #multiply C into next site and continue sweep
            if n != self.length-2:
                self.psi.node[n+1].tensor = np.einsum('ab,ibc->iac',C,self.psi.node[n+1].tensor)
            else:
                self.psi.node[n+1].tensor = np.einsum('ab,ib->ia',C,self.psi.node[n+1].tensor)
            self.psiConj.node[n+1].tensor = np.conj(self.psi.node[n+1].tensor)

    def leftSweep(self,delta_t,integrator):
        #rhs
        #form Heff from product of exp/overlap transfer matrices
        #update site
        Pl = np.einsum('abc,cd->abd',self.expL[self.length-2].tensor,self.overlapL[self.length-2].tensor)

        H = np.einsum('abc,ijb->jcia',Pl,self.H.node[self.length-1].tensor)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        A = self.psi.node[self.length-1].tensor.reshape((dims[0]*dims[1]))
        A = integrator(H,A,delta_t)
        A = A.reshape((dims[0],dims[1]))

        U,S,Vh = np.linalg.svd(A,full_matrices=False)
        self.psi.node[self.length-1].tensor = np.transpose(Vh)
        self.psiConj.node[self.length-1].tensor = np.conj(self.psi.node[self.length-1].tensor)
        self.expR[self.length-1] = collapsed_layer.factory(layer(self.expNetwork,self.length-1))
        self.overlapR[self.length-1] = collapsed_layer.factory(layer(self.overlapNetwork,self.length-1))

        #update bond with backwards TDVP evolution
        C = np.dot(U,np.diag(S))
        Pr = np.einsum('abc,cd->abd',self.expR[self.length-1].tensor,self.overlapR[self.length-1].tensor)
        H = np.einsum('abc,dbf->cfad',Pl,Pr)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        C = C.reshape((dims[0]*dims[1]))
        C = integrator(H,C,-delta_t)
        C = C.reshape((dims[0],dims[1]))

        #multiply C into next site and continue sweep
        self.psi.node[self.length-2].tensor = np.einsum('iab,bc->iac',self.psi.node[self.length-2].tensor,C)
        self.psiConj.node[self.length-2].tensor = np.conj(self.psi.node[self.length-2].tensor)

        for n in range(self.length-2,0,-1):
            Pr = np.einsum('abc,cd->abd',self.expR[n+1].tensor,self.overlapR[n+1].tensor)
            Pl = np.einsum('abc,cd->abd',self.expL[n-1].tensor,self.overlapL[n-1].tensor)

            H = np.einsum('ijbe,abc->ijace',self.H.node[n].tensor,Pl)
            H = np.einsum('ijace,def->jcfiad',H,Pr)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1]*dims[2],dims[3]*dims[4]*dims[5]))
            A = self.psi.node[n].tensor.reshape((dims[0]*dims[1]*dims[2]))
            A = integrator(H,A,delta_t)
            A = A.reshape((dims[0],dims[1],dims[2]))

            A = np.einsum('ijk->jik',A)
            dims = np.shape(A)
            A = A.reshape(dims[0],dims[1]*dims[2])

            U,S,Vh = np.linalg.svd(A,full_matrices=False)
            self.psi.node[n].tensor  = np.einsum('ijk->jik',Vh.reshape((np.size(S),dims[1],dims[2])))
            self.psiConj.node[n].tensor  = np.conj(self.psi.node[n].tensor)
            self.expR[n] = combine_clayer_layer.new_collapsed_layer(layer(self.expNetwork,n),self.expR[n+1],)
            self.overlapR[n] = combine_clayer_layer.new_collapsed_layer(layer(self.overlapNetwork,n),self.overlapR[n+1])

            #update bond with backwards TDVP evolution
            C = np.dot(U,np.diag(S))
            Pr = np.einsum('abc,cd->abd',self.expR[n].tensor,self.overlapR[n].tensor)
            H = np.einsum('abc,dbf->cfad',Pl,Pr)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
            C = C.reshape((dims[0]*dims[1]))
            C = integrator(H,C,-delta_t)
            C = C.reshape((dims[0],dims[1]))

            #multiply C into next site and continue sweep
            if n != 1:
                self.psi.node[n-1].tensor = np.einsum('iab,bc->iac',self.psi.node[n-1].tensor,C)
            else:
                self.psi.node[n-1].tensor = np.einsum('ia,ab->ib',self.psi.node[n-1].tensor,C)
            self.psiConj.node[n-1].tensor = np.conj(self.psi.node[n-1].tensor)


    def run(self,delta_t,t_max,integrator):
        print("Evolving with TDVP")
        self.t=np.arange(0,t_max+delta_t/2,delta_t/2)
        self.f=np.zeros(np.size(self.t))
        self.energy = np.zeros(np.size(self.t))
        # self.expNetwork.contract()
        # self.energy[0] = self.expNetwork.contraction
        self.f[0] = 1
        pbar=ProgressBar()
        for n in pbar(range(1,np.size(self.t,axis=0))):
            self.rightSweep(delta_t/2,integrator)
            self.leftSweep(delta_t/2,integrator)
            norm = np.abs(self.psi.dot(self.psi))
            # if np.abs(1-norm)>0.1:
                # for m in range(0,self.length):
                    # self.psi.node[m].tensor = self.psi.node[m].tensor / np.power(norm,1/(2*self.length))
                    # self.psiConj.node[m].tensor = self.psiConj.node[m].tensor / np.power(norm,1/(2*self.length))
            self.f[n] = np.abs(self.psi.dot(self.psi_init))**2

            # self.expNetwork.contract()
            # self.energy[n] = self.expNetwork.contraction
            # print(np.abs(self.psi.dot(self.psi)))
