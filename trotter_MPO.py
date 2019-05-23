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

# class H_three_site:
    # def gen(name,Nc=None):
        # if name is "pxp": return pxp_3site(Nc)
        # if name is "pcp": return pcp_3site(Nc)

# def pxp_3site(Nc):
    # pxp = unlocking_System(np.arange(0,Nc),"open",Nc,3)
    # pxp.gen_basis()
    # X = spin_Hamiltonian(pxp,"x").site_ops[1]
    # H = Hamiltonian(pxp)
    # H.site_ops[1] = X
    # H.model = np.array([[0,1,0]])
    # H.model_coef=np.array([[1]])
    # H.gen()
    # H.sector.find_eig()
    # return H.sector.matrix()

# def pcp_3site(Nc):
    # pxp = unlocking_System(np.arange(0,Nc),"open",Nc,3)
    # pxp.gen_basis()
    # C = clock_Hamiltonian(pxp).site_ops[1]
    # H = Hamiltonian(pxp)
    # H.site_ops[1] = X
    # H.model = np.array([[0,1,0]])
    # H.model_coef=np.array([[1]])
    # H.gen()
    # H.sector.find_eig()
    # return H.sector.matrix()

def pxp_trotter_mpo(tau,Nc):
    pxp = unlocking_System(np.arange(0,Nc),"open",Nc,3)
    pxp.gen_basis()
    X = spin_Hamiltonian(pxp,"x").site_ops[1]
    H = Hamiltonian(pxp)
    H.site_ops[1] = X
    H.model = np.array([[0,1,0]])
    H.model_coef=np.array([[1]])
    H.gen()
    H.sector.find_eig()
    dim=np.size(H.sector.matrix(),axis=0)
    U = np.eye(dim) - 1j * tau * H.sector.matrix()
    T = U.reshape(np.array((pxp.base,pxp.base,pxp.base,pxp.base,pxp.base,pxp.base)))
    T = np.einsum('ijkunm->iujnkm',T)
    T = T.reshape(np.array((np.power(pxp.base,4),np.power(pxp.base,2))))
    U,S0,Vh = np.linalg.svd(T,full_matrices=False)
    from Diagnostics import is_unitary

    #finalize reshaping Vh
    O_right = Vh.reshape(np.array((np.size(S0),pxp.base,pxp.base)))
    O_right = np.einsum('ijk->jki',O_right)

    U = U.reshape(np.array((np.power(pxp.base,2),np.power(pxp.base,2)*np.size(S0))))
    U,S1,Vh = np.linalg.svd(U,full_matrices=False)

    O_left = U.reshape(np.array((pxp.base,pxp.base,np.size(S1))))

    Vh = Vh.reshape(np.array((np.size(S1),pxp.base,pxp.base,np.size(S0))))
    O_mid = np.einsum('ij,jkuv,vn->ikun',np.diag(S1),Vh,np.diag(S0))
    O_mid = np.einsum('ijku->jkiu',O_mid)

    return O_left,O_mid,O_right

def pcp_trotter_mpo(tau,Nc):
    pxp = unlocking_System(np.arange(0,Nc),"open",Nc,3)
    pxp.gen_basis()
    C = clock_Hamiltonian(pxp).site_ops[1]
    H = Hamiltonian(pxp)
    H.site_ops[1] = C
    H.model = np.array([[0,1,0]])
    H.model_coef=np.array([[1]])
    H.gen()
    H.sector.find_eig()
    dim=np.size(H.sector.matrix(),axis=0)
    U = np.eye(dim) - 1j * tau * H.sector.matrix()
    T = U.reshape(np.array((pxp.base,pxp.base,pxp.base,pxp.base,pxp.base,pxp.base)))
    T = np.einsum('ijkunm->iujnkm',T)
    T = T.reshape(np.array((np.power(pxp.base,4),np.power(pxp.base,2))))
    U,S0,Vh = np.linalg.svd(T,full_matrices=False)
    from Diagnostics import is_unitary

    #finalize reshaping Vh
    O_right = Vh.reshape(np.array((np.size(S0),pxp.base,pxp.base)))
    O_right = np.einsum('ijk->jki',O_right)

    U = U.reshape(np.array((np.power(pxp.base,2),np.power(pxp.base,2)*np.size(S0))))
    U,S1,Vh = np.linalg.svd(U,full_matrices=False)

    O_left = U.reshape(np.array((pxp.base,pxp.base,np.size(S1))))

    Vh = Vh.reshape(np.array((np.size(S1),pxp.base,pxp.base,np.size(S0))))
    O_mid = np.einsum('ij,jkuv,vn->ikun',np.diag(S1),Vh,np.diag(S0))
    O_mid = np.einsum('ijku->jkiu',O_mid)

    return O_left,O_mid,O_right

# def free_para(tau):
    # pxp = unlocking_System([0,1],"open",2,3)
    # pxp.gen_basis()
    # H = spin_Hamiltonian(pxp,"x")
    # H.gen()
    # H.sector.find_eig()
    # H_diag = np.exp(-1j*tau*np.diag(H.sector.eigvalues()))
    # U = np.dot(H.sector.eigvectors(),np.dot(H_diag,np.conj(np.transpose(H.sector.eigvectors()))))
    # T = U.reshape(np.array((pxp.base,pxp.base,pxp.base,pxp.base,pxp.base,pxp.base)))
    # T = np.einsum('ijkunm->iujnkm',T)
    # T = T.reshape(np.array((np.power(pxp.base,4),np.power(pxp.base,2))))
    # U,S0,Vh = np.linalg.svd(T,full_matrices=False)

    # #finalize reshaping Vh
    # O_right = Vh.reshape(np.array((np.size(S0),pxp.base,pxp.base)))
    # O_right = np.einsum('ijk->jki',O_right)

    # U = U.reshape(np.array((np.power(pxp.base,2),np.power(pxp.base,2)*np.size(S0))))
    # U,S1,Vh = np.linalg.svd(U,full_matrices=False)

    # O_left = U.reshape(np.array((pxp.base,pxp.base,np.size(S1))))

    # Vh = Vh.reshape(np.array((np.size(S1),pxp.base,pxp.base,np.size(S0))))
    # O_mid = np.einsum('ij,jkuv,vn->ikun',np.diag(S1),Vh,np.diag(S0))
    # O_mid = np.einsum('ijku->jkiu',O_mid)

    # return O_left,O_mid,O_right
