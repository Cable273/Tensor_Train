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

        self.psi_init.right_normalize()

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
        n=0
        Pr = np.einsum('abc,cd->abd',self.expR[n+1].tensor,self.overlapR[n+1].tensor)

        H = np.einsum('ijb,abc->jcia',self.H.node[n].tensor,Pr)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        A = self.psi.node[n].tensor.reshape((dims[2]*dims[3]))
        A = integrator(H,A,delta_t)
        A = A.reshape((dims[0],dims[1]))
            
        U,S,Vh = np.linalg.svd(A,full_matrices=False)

        self.psi.node[n].tensor = U
        self.psiConj.node[n].tensor = np.conj(U)

        self.expL[n] = collapsed_layer.factory(layer(self.expNetwork,n))
        self.overlapL[n] = collapsed_layer.factory(layer(self.overlapNetwork,n))

        #update bond with backwards TDVP evolution
        C = np.dot(np.diag(S),Vh)
        Pl = np.einsum('abc,cd->abd',self.expL[n].tensor,self.overlapL[n].tensor)
        H = np.einsum('abc,dbf->cfad',Pl,Pr)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        C = C.reshape((dims[2]*dims[3]))
        C = integrator(H,C,-delta_t)
        C = C.reshape((dims[0],dims[1]))

        #multiply C into next site and continue sweep
        self.psi.node[n+1].tensor = np.einsum('ab,ibc->iac',C,self.psi.node[n+1].tensor)
        self.psiConj.node[n+1].tensor = np.conj(self.psi.node[n+1].tensor)

        for n in range(1,self.length-1):
            Pr = np.einsum('abc,cd->abd',self.expR[n+1].tensor,self.overlapR[n+1].tensor)
            Pl = np.einsum('abc,cd->abd',self.expL[n-1].tensor,self.overlapL[n-1].tensor)

            H = np.einsum('ijbe,abc->ijace',self.H.node[n].tensor,Pl)
            H = np.einsum('ijace,def->jcfiad',H,Pr)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1]*dims[2],dims[3]*dims[4]*dims[5]))
            A = self.psi.node[n].tensor.reshape((dims[3]*dims[4]*dims[5]))
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
            if n != self.length-2:
                Pl = np.einsum('abc,cd->abd',self.expL[n].tensor,self.overlapL[n].tensor)
                H = np.einsum('abc,dbf->cfad',Pl,Pr)
                dims = np.shape(H)
                H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
                C = C.reshape((dims[2]*dims[3]))
                C = integrator(H,C,-delta_t)
                C = C.reshape((dims[0],dims[1]))
                self.psi.node[n+1].tensor = np.einsum('ab,ibc->iac',C,self.psi.node[n+1].tensor)
            else:
                self.psi.node[n+1].tensor = np.einsum('ab,ib->ia',C,self.psi.node[n+1].tensor)
            self.psiConj.node[n+1].tensor = np.conj(self.psi.node[n+1].tensor)



    def leftSweep(self,delta_t,integrator):
        #rhs
        #form Heff from product of exp/overlap transfer matrices
        #update site
        n=self.length-1
        Pl = np.einsum('abc,cd->abd',self.expL[n-1].tensor,self.overlapL[n-1].tensor)

        H = np.einsum('abc,ijb->jcia',Pl,self.H.node[n].tensor)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        A = self.psi.node[n].tensor.reshape((dims[2]*dims[3]))
        A = integrator(H,A,delta_t)
        A = A.reshape((dims[0],dims[1]))
        A = A.transpose()

        U,S,Vh = np.linalg.svd(A,full_matrices=False)
        self.psi.node[n].tensor = np.transpose(Vh)
        self.psiConj.node[n].tensor = np.conj(self.psi.node[n].tensor)

        self.expR[n] = collapsed_layer.factory(layer(self.expNetwork,n))
        self.overlapR[n] = collapsed_layer.factory(layer(self.overlapNetwork,n))

        #update bond with backwards TDVP evolution
        C = np.dot(U,np.diag(S))
        Pr = np.einsum('abc,cd->abd',self.expR[n].tensor,self.overlapR[n].tensor)
        H = np.einsum('abc,dbf->cfad',Pl,Pr)
        dims = np.shape(H)
        H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        C = C.reshape((dims[2]*dims[3]))
        C = integrator(H,C,-delta_t)
        C = C.reshape((dims[0],dims[1]))

        #multiply C into next site and continue sweep
        self.psi.node[n-1].tensor = np.einsum('iab,bc->iac',self.psi.node[n-1].tensor,C)
        self.psiConj.node[n-1].tensor = np.conj(self.psi.node[n-1].tensor)

        for n in range(self.length-2,0,-1):
            Pr = np.einsum('abc,cd->abd',self.expR[n+1].tensor,self.overlapR[n+1].tensor)
            Pl = np.einsum('abc,cd->abd',self.expL[n-1].tensor,self.overlapL[n-1].tensor)

            H = np.einsum('ijbe,abc->ijace',self.H.node[n].tensor,Pl)
            H = np.einsum('ijace,def->jcfiad',H,Pr)
            dims = np.shape(H)
            H = H.reshape((dims[0]*dims[1]*dims[2],dims[3]*dims[4]*dims[5]))
            A = self.psi.node[n].tensor.reshape((dims[3]*dims[4]*dims[5]))
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
            if n != 1:
                Pr = np.einsum('abc,cd->abd',self.expR[n].tensor,self.overlapR[n].tensor)
                H = np.einsum('abc,dbf->cfad',Pl,Pr)
                dims = np.shape(H)
                H = H.reshape((dims[0]*dims[1],dims[2]*dims[3]))
                C = C.reshape((dims[2]*dims[3]))
                C = integrator(H,C,-delta_t)
                C = C.reshape((dims[0],dims[1]))
                self.psi.node[n-1].tensor = np.einsum('iab,bc->iac',self.psi.node[n-1].tensor,C)
            else:
                self.psi.node[n-1].tensor = np.einsum('ia,ab->ib',self.psi.node[n-1].tensor,C)
            self.psiConj.node[n-1].tensor = np.conj(self.psi.node[n-1].tensor)


    def run(self,delta_t,t_max,integrator,savEvolvedMPS=False,calcFid=False):
        print("Evolving with TDVP")
        self.t=np.arange(0,t_max+delta_t,delta_t)

        if calcFid is True:
            self.f=np.zeros(np.size(self.t))
            self.f[0] = 1
        if savEvolvedMPS is True:
            self.evolvedMPS = dict()
            self.evolvedMPS[0] = self.psi_init

        pbar=ProgressBar()
        for n in pbar(range(1,np.size(self.t,axis=0))):
            self.rightSweep(delta_t/2,integrator)
            self.leftSweep(delta_t/2,integrator)

            if savEvolvedMPS is True:
                self.evolvedMPS[n] = copy.deepcopy(self.psi)
            if calcFid is True:
                # do fidelity overlap by hand (quicker..)
                L_temp = np.einsum('ia,ib->ab',self.psi_init.node[0].tensor,np.conj(self.psi.node[0].tensor))
                for m in range(1,self.length-1):
                    L_temp = np.einsum('ab,iac->ibc',L_temp,self.psi_init.node[m].tensor)
                    L_temp = np.einsum('ibc,ibd->cd',L_temp,np.conj(self.psi.node[m].tensor))
                L_temp = np.einsum('ab,ia->ib',L_temp,self.psi_init.node[self.length-1].tensor)
                scalar = np.abs(np.einsum('ib,ib',L_temp,np.conj(self.psi.node[self.length-1].tensor)))**2
                self.f[n] = scalar

    def eval_fidelity(self):
        print("Fidelity: Contracting evolved states")
        self.f = np.zeros(np.size(self.t))
        pbar=ProgressBar()
        for n in pbar(range(0,np.size(self.f,axis=0))):
            L_temp = np.einsum('ia,ib->ab',self.psi_init.node[0].tensor,np.conj(self.evolvedMPS[n].node[0].tensor))
            for m in range(1,self.length-1):
                L_temp = np.einsum('ab,iac->ibc',L_temp,self.psi_init.node[m].tensor)
                L_temp = np.einsum('ibc,ibd->cd',L_temp,np.conj(self.evolvedMPS[n].node[m].tensor))
            L_temp = np.einsum('ab,ia->ib',L_temp,self.psi_init.node[self.length-1].tensor)
            scalar = np.abs(np.einsum('ib,ib',L_temp,np.conj(self.evolvedMPS[n].node[self.length-1].tensor)))**2
            self.f[n] = scalar

    def eval_exp(self,mpo_object):
        print("Expectation: Contracting MPO with evolved states")
        exp = np.zeros(np.size(self.t),dtype=complex)
        pbar=ProgressBar()
        for n in pbar(range(0,np.size(self.t,axis=0))):
            L_temp = np.einsum('ia,ijb->jab',self.evolvedMPS[n].node[0].tensor,mpo_object.node[0].tensor)
            L_temp = np.einsum('jab,jc->abc',L_temp,np.conj(self.evolvedMPS[n].node[0].tensor))
            for m in range(1,self.length-1):
                L_temp = np.einsum('abc,iad->ibcd',L_temp,self.evolvedMPS[n].node[m].tensor)
                L_temp = np.einsum('ibcd,ijbe->jcde',L_temp,mpo_object.node[m].tensor)
                L_temp = np.einsum('jcde,jcf->def',L_temp,np.conj(self.evolvedMPS[n].node[m].tensor))
            m = self.length-1
            L_temp = np.einsum('abc,ia->ibc',L_temp,self.evolvedMPS[n].node[m].tensor)
            L_temp = np.einsum('ibc,ijb->jc',L_temp,mpo_object.node[m].tensor)
            scalar = np.einsum('jc,jc',L_temp,np.conj(self.evolvedMPS[n].node[m].tensor))
            exp[n] = scalar
        return exp
