#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import *
from rail_objects import *

import sys
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/Classes/'
sys.path.append(file_dir)
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/functions/'
sys.path.append(file_dir)

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

class two_site_trotter:
    def factory(name,Nc=None):
        if name == "xx": return xx_trotter()
    def mpo(self,tau):
        dim=np.size(self.H.sector.matrix(),axis=0)
        U = np.eye(dim) - 1j * tau * self.H.sector.matrix()
        T = U.reshape(np.array((self.base,self.base,self.base,self.base)))
        T = np.einsum('ijkl->ikjl',T)
        shape = np.shape(T)
        T = T.reshape(np.array((shape[0]*shape[1],shape[2]*shape[3])))

        A,S,B = np.linalg.svd(T,full_matrices=False)
        A = np.dot(A,np.power(np.diag(S),0.5))
        B = np.dot(np.power(np.diag(S),0.5),B)
        A=A.reshape(np.array((shape[0],shape[1],np.size(S))))

        B=B.reshape(np.array((np.size(S),shape[2],shape[3])))
        B = np.einsum('ijk->jki',B)
        return A,B

class xx_trotter(two_site_trotter):
    def __init__(self):
        self.base = 2
        self.H = self.xx_2site()

    def xx_2site(self):
        pxp = unlocking_System(np.arange(0,self.base),"open",self.base,2)
        pxp.gen_basis()
        H = Hamiltonian(pxp)
        H.site_ops[1] = np.array([[0,1],[1,0]])
        H.model = np.array([[1,1]])
        H.model_coef=np.array([[1]])
        H.gen()
        H.sector.find_eig()
        return H

class three_site_trotter:
    def factory(name,Nc=None):
        if name == "pxp": return pxp_trotter(Nc)
        if name == "pcp": 
            return pcp_trotter(Nc)

    def mpo(self,tau):
        dim=np.size(self.H,axis=0)
        # H_diag = np.diag(np.exp(-1j*tau*self.e))
        # U=np.dot(np.conj(np.transpose(self.H.sector.eigvectors())),np.dot(H_diag,self.H.sector.eigvectors()))
        # U = sp.linalg.expm(-1j*tau*self.H)
        U = np.eye(dim) - 1j * tau * self.H
        T = U.reshape(np.array((self.base,self.base,self.base,self.base,self.base,self.base)))
        T = np.einsum('ijkunm->iujnkm',T)
        T = T.reshape(np.array((np.power(self.base,4),np.power(self.base,2))))
        U,S0,Vh = np.linalg.svd(T,full_matrices=False)
        from Diagnostics import is_unitary

        #finalize reshaping Vh
        O_right = Vh.reshape(np.array((np.size(S0),self.base,self.base)))
        O_right = np.einsum('ijk->jki',O_right)

        U = U.reshape(np.array((np.power(self.base,2),np.power(self.base,2)*np.size(S0))))
        U,S1,Vh = np.linalg.svd(U,full_matrices=False)

        O_left = U.reshape(np.array((self.base,self.base,np.size(S1))))

        Vh = Vh.reshape(np.array((np.size(S1),self.base,self.base,np.size(S0))))
        O_mid = np.einsum('ij,jkuv,vn->ikun',np.diag(S1),Vh,np.diag(S0))
        O_mid = np.einsum('ijku->jkiu',O_mid)
        return O_left,O_mid,O_right

class pcp_trotter(three_site_trotter):
    def __init__(self,Nc):
        self.base = Nc
        self.H = self.pcp_3site(Nc)

    def pcp_3site(self,Nc):
        pxp = unlocking_System(np.arange(0,self.base),"open",self.base,3)
        pxp.gen_basis()
        C = clock_Hamiltonian(pxp).site_ops[1]
        H = Hamiltonian(pxp)
        H.site_ops[1] = C
        H.model = np.array([[0,1,0]])
        H.model_coef=np.array([[1]])
        H.gen()
        # H.gen_site_H(np.array([1,0]),0)
        # H.gen_site_H(np.array([0,1]),pxp.N-2)
        H.sector.find_eig()
        return H

class pxp_trotter(three_site_trotter):
    def __init__(self,Nc):
        self.base = Nc
        self.H = np.array([[0,1,1,0,1,0,0,0],[1,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,0,0,0,0,1,0,0],[0,1,0,0,1,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
        self.e,self.u = np.linalg.eigh(self.H)
