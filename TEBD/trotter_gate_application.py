#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
from MPS import mpo,mps
from rail_objects import *
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *
from progressbar import ProgressBar
import copy

class gate_application_method:
    def factory(gate,psi,loc,D,phys_dim):
        gate_length = int(np.size(np.shape(gate))/2)
        if gate_length == 2:
            if psi.node[loc].legs == "right" and psi.node[loc+1].legs == "both":
                return size2GateAppliedLeft(gate,psi,loc,D,phys_dim)
            elif psi.node[loc].legs =="both" and psi.node[loc+1].legs == "both":
                return size2GateAppliedMiddle(gate,psi,loc,D,phys_dim)
            elif psi.node[loc].legs =="both" and psi.node[loc+1].legs == "left":
                return size2GateAppliedRight(gate,psi,loc,D,phys_dim)
        if gate_length == 3:
            if psi.node[loc].legs == 'right' and psi.node[loc+2].legs == 'both':
                return size3GateAppliedLeft(gate,psi,loc,D,phys_dim)
            elif psi.node[loc].legs == 'both' and psi.node[loc+2].legs == 'both':
                return size3GateAppliedMiddle(gate,psi,loc,D,phys_dim)
            elif psi.node[loc].legs =='both' and psi.node[loc+2].legs == 'left':
                return size3GateAppliedRight(gate,psi,loc,D,phys_dim)
            else:
                print("Gate not implemented!")


class size2GateAppliedLeft(gate_application_method):
    def __init__(self,gate,psi,loc,D,phys_dim):
        self.gate = gate
        self.psi = psi
        self.loc = loc
        self.D = D
        self.phys_dim = phys_dim
    def apply(self,truncate = True):
        A = self.psi.node[self.loc].tensor
        B = self.psi.node[self.loc+1].tensor
        M = np.einsum('ae,bef->abf',A,B)
        psi_new = np.einsum('abf,abcd->cdf',M,self.gate)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0],dims[1]*dims[2]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            S = S[:self.D]
            Vh = Vh[:self.D,:]

        A = np.dot(U,np.diag(np.power(S,0.5)))
        Vh = np.dot(np.diag(np.power(S,0.5)),Vh)
        Vh = Vh.reshape((np.size(S),dims[1],dims[2]))
        B = np.einsum('idf->dif',Vh)
        self.psi.set_entry(self.loc,A,"right")
        self.psi.set_entry(self.loc+1,B,"both")

class size2GateAppliedMiddle(gate_application_method):
    def __init__(self,gate,psi,loc,D,phys_dim):
        self.gate = gate
        self.psi = psi
        self.loc = loc
        self.D = D
        self.phys_dim = phys_dim
    def apply(self,truncate = True):
        A = self.psi.node[self.loc].tensor
        B = self.psi.node[self.loc+1].tensor
        M = np.einsum('aef,bfg->abeg',A,B)
        psi_new = np.einsum('abeg,abcd->cedg',M,self.gate)
        dims = np.shape(psi_new)
        psi_new = psi_new.reshape(dims[0]*dims[1],dims[2]*dims[3])
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)

        if truncate is True:
            U = U[:,:self.D]
            S = S[:self.D]
            Vh = Vh[:self.D,:]

        U = np.dot(U,np.power(np.diag(S),0.5))
        Vh = np.dot(np.power(np.diag(S),0.5),Vh)

        A = U.reshape((dims[0],dims[1],np.size(S)))
        B = np.einsum('idg->dig',Vh.reshape((np.size(S),dims[2],dims[3])))

        self.psi.set_entry(self.loc,A,"both")
        self.psi.set_entry(self.loc+1,B,"both")

class size2GateAppliedRight(gate_application_method):
    def __init__(self,gate,psi,loc,D,phys_dim):
        self.gate = gate
        self.psi = psi
        self.loc = loc
        self.D = D
        self.phys_dim = phys_dim
    def apply(self,truncate = True):
        A = self.psi.node[self.loc].tensor
        B = self.psi.node[self.loc+1].tensor
        M = np.einsum('aef,bf->abe',A,B)
        psi_new = np.einsum('abe,abcd->ced',M,self.gate)
        dims = np.shape(psi_new)
        psi_new = psi_new.reshape(dims[0]*dims[1],dims[2])
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)

        if truncate is True:
            U = U[:,:self.D]
            S = S[:self.D]
            Vh = Vh[:self.D,:]

        U = np.dot(U,np.power(np.diag(S),0.5))
        Vh = np.dot(np.power(np.diag(S),0.5),Vh)

        A = U.reshape((dims[0],dims[1],np.size(S)))
        B = np.einsum('id->di',Vh)

        self.psi.set_entry(self.loc,A,"both")
        self.psi.set_entry(self.loc+1,B,"left")

class size3GateAppliedLeft(gate_application_method):
    def __init__(self,gate,psi,loc,D,phys_dim):
        self.gate = gate
        self.psi = psi
        self.loc = loc
        self.D = D
        self.phys_dim = phys_dim
    def apply(self,truncate = True):
        M = np.einsum('ai,bij->abj',self.psi.node[self.loc].tensor,self.psi.node[self.loc+1].tensor)
        M = np.einsum('abj,cjk->abck',M,self.psi.node[self.loc+2].tensor)
        psi_new = np.einsum('abck,abcdef->defk',M,self.gate)

        #reshape/svd to extract new mps tensors
        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1],dims[2]*dims[3]))

        U0,S0,Vh0 = np.linalg.svd(psi_new,full_matrices=False)
        Vh0 = np.dot(np.diag(S0),Vh0)

        Vh0 = Vh0.reshape((np.size(S0),dims[2],dims[3]))
        psi_right = np.einsum('ifk->fik',Vh0)

        #svd to get A S1 B S0 B MPS
        U0 = U0.reshape((dims[0],dims[1]*np.size(U0,axis=1)))
        U1,S1,Vh1 = np.linalg.svd(U0,full_matrices=False)
        U1 = np.dot(U1,np.diag(S1))

        if truncate is True:
            U1 = U1[:,:self.D]
            S1 = S1[:self.D]
            Vh1 = Vh1[:self.D,:]

        psi_left = U1

        Vh1 = Vh1.reshape((np.size(S1),dims[1],np.size(S0)))
        psi_centre = np.einsum('jei->eji',Vh1)

        self.psi.set_entry(self.loc,psi_left,"right")
        self.psi.set_entry(self.loc+1,psi_centre,"both")
        self.psi.set_entry(self.loc+2,psi_right,"both")

class size3GateAppliedMiddle(gate_application_method):
    def __init__(self,gate,psi,loc,D,phys_dim):
        self.gate = gate
        self.psi = psi
        self.loc = loc
        self.D = D
        self.phys_dim = phys_dim
    def apply(self,truncate = True):
        M = np.einsum('aij,bjk->abik',self.psi.node[self.loc].tensor,self.psi.node[self.loc+1].tensor)
        M = np.einsum('abik,ckl->abcil',M,self.psi.node[self.loc+2].tensor)
        psi_new = np.einsum('abcil,abcdef->diefl',M,self.gate)

        #reshape/svd to extract new mps tensors
        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1]*dims[2],dims[3]*dims[4]))

        U0,S0,Vh0 = np.linalg.svd(psi_new,full_matrices=False)
        Vh0 = np.dot(np.diag(S0),Vh0)

        if truncate is True:
            U0 = U0[:,:self.D]
            S0 = S0[:self.D]
            Vh0 = Vh0[:self.D,:]

        Vh0 = Vh0.reshape((np.size(S0),dims[3],dims[4]))
        psi_right = np.einsum('nfl->fnl',Vh0)

        #svd to get A S1 B S0 B MPS
        U0 = U0.reshape((dims[0]*dims[1],dims[2]*np.size(S0)))
        U1,S1,Vh1 = np.linalg.svd(U0,full_matrices=False)
        U1 = np.dot(U1,np.diag(S1))

        if truncate is True:
            U1 = U1[:,:self.D]
            S1 = S1[:self.D]
            Vh1 = Vh1[:self.D,:]

        psi_left = U1.reshape((dims[0],dims[1],np.size(S1)))

        Vh1 = Vh1.reshape((np.size(S1),dims[2],np.size(S0)))
        psi_centre = np.einsum('men->emn',Vh1)

        self.psi.set_entry(self.loc,psi_left,"both")
        self.psi.set_entry(self.loc+1,psi_centre,"both")
        self.psi.set_entry(self.loc+2,psi_right,"both")

class size3GateAppliedRight(gate_application_method):
    def __init__(self,gate,psi,loc,D,phys_dim):
        self.gate = gate
        self.psi = psi
        self.loc = loc
        self.D = D
        self.phys_dim = phys_dim
    def apply(self,truncate = True):
        M = np.einsum('aij,bjk->abik',self.psi.node[self.loc].tensor,self.psi.node[self.loc+1].tensor)
        M = np.einsum('abik,ck->abci',M,self.psi.node[self.loc+2].tensor)
        psi_new = np.einsum('abci,abcdef->dief',M,self.gate)

        #reshape/svd to extract new mps tensors
        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1]*dims[2],dims[3]))

        U0,S0,Vh0 = np.linalg.svd(psi_new,full_matrices=False)

        Vh0 = np.dot(np.diag(S0),Vh0)
        if truncate is True:
            U0 = U0[:,:self.D]
            S0 = S0[:self.D]
            Vh0 = Vh0[:self.D,:]

        Vh0 = Vh0.reshape((np.size(S0),dims[3]))
        psi_right = np.einsum('nf->fn',Vh0)

        #svd to get A S1 B S0 B MPS
        U0 = U0.reshape((dims[0]*dims[1],dims[2]*np.size(S0)))
        U1,S1,Vh1 = np.linalg.svd(U0,full_matrices=False)
        U1 = np.dot(U1,np.diag(S1))

        if truncate is True:
            U1 = U1[:,:self.D]
            S1 = S1[:self.D]
            Vh1 = Vh1[:self.D,:]

        psi_left = U1.reshape((dims[0],dims[1],np.size(S1)))

        Vh1 = Vh1.reshape((np.size(S1),dims[2],np.size(S0)))
        psi_centre = np.einsum('men->emn',Vh1)

        self.psi.set_entry(self.loc,psi_left,"both")
        self.psi.set_entry(self.loc+1,psi_centre,"both")
        self.psi.set_entry(self.loc+2,psi_right,"left")
