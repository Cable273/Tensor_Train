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
    
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

class intGate:
    def __init__(self,tensor,loc,length):
        self.tensor = tensor
        self.loc = loc
        self.length = length

class swapGate:
    def __init__(self,loc):
        self.loc = loc

class uniformTrotters:
    def gen(N,H,uc_size,phys_dim,delta_t,trotter_order=1):
        if uc_size == 2:
            return twoSite_uniformTrotter(H,N,uc_size,phys_dim,delta_t,trotter_order=1)
        elif uc_size == 3:
            return threeSite_uniformTrotter(H,N,uc_size,phys_dim,delta_t,trotter_order=1)

def twoSite_uniformTrotter(H,N,uc_size,phys_dim,delta_t,trotter_order=1):
    if trotter_order == 1:
        trotter_gate = sp.linalg.expm(-1j*H*delta_t)
        trotter_gate = trotter_gate.reshape(phys_dim,phys_dim,phys_dim,phys_dim)
        trotter_gates = dict()
        trotter_gates[0] = dict()
        trotter_gates[1] = dict()
        loc_even = np.arange(0,N-1,2)
        loc_odd = np.arange(1,N-2,2)
        for n in range(0,np.size(loc_even,axis=0)):
            trotter_gates[0][n] = intGate(trotter_gate,loc_even[n],2)
        for n in range(0,np.size(loc_odd,axis=0)):
            trotter_gates[1][n] = intGate(trotter_gate,loc_odd[n],2)
        return trotter_gates
    elif trotter_order == 2:
        trotter_gate_half = sp.linalg.expm(-1j*H*delta_t/2)
        trotter_gate = sp.linalg.expm(1j*H*delta_t)

        trotter_gate = trotter_gate.reshape(phys_dim,phys_dim,phys_dim,phys_dim)
        trotter_gate_half = trotter_gate_half.reshape(phys_dim,phys_dim,phys_dim,phys_dim)

        trotter_gates = dict()
        trotter_gates[0] = dict()
        trotter_gates[1] = dict()
        trotter_gates[2] = dict()
        loc_even = np.arange(0,N-1,2)
        loc_odd = np.arange(1,N-2,2)
        for n in range(0,np.size(loc_even,axis=0)):
            trotter_gates[1][n] = intGate(trotter_gate,loc_even[n],2)
        for n in range(0,np.size(loc_odd,axis=0)):
            trotter_gates[0][n] = intGate(trotter_gate_half,loc_odd[n],2)
            trotter_gates[2][n] = intGate(trotter_gate_half,loc_odd[n],2)
        return trotter_gates

class common_trotters:
    def PXP(N,phys_dim,delta_t,trotter_order=1):
        s = (phys_dim-1)/2
        m = np.arange(-s,s)
        couplings = np.power(s*(s+1)-m*(m+1),0.5)
        P = np.zeros((phys_dim,phys_dim))
        P[0,0] = 1
        X = (2*(np.diag(couplings,1) + np.diag(couplings,-1)))/2
        # P = np.array([[1,0],[0,0]])
        # X = np.array([[0,1],[1,0]])
        H_XP = np.kron(X,P)
        H_PX = np.kron(P,X)
        H_PXP = np.kron(np.kron(P,X),P)
        if trotter_order == 1:
            XP_trotter = np.reshape(sp.linalg.expm(-1j*H_XP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))
            PXP_trotter = np.reshape(sp.linalg.expm(-1j*H_PXP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            PX_trotter = np.reshape(sp.linalg.expm(-1j*H_PX*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))

            trotter_gates = dict()
            trotter_gates[0] = dict()
            trotter_gates[1] = dict()
            trotter_gates[2] = dict()

            if N % 3 == 0:
                for n in range(0,N-2,3):
                    trotter_gates[0][n] = intGate(PXP_trotter,n,3)
                for n in range(1,N-4,3):
                    trotter_gates[1][n] = intGate(PXP_trotter,n,3)
                trotter_gates[1][N-2] = intGate(PX_trotter,N-2,2)
                for n in range(2,N-3,3):
                    trotter_gates[2][n] = intGate(PXP_trotter,n,3)
                trotter_gates[2][0] = intGate(XP_trotter,0,2)

            elif N % 3 == 1:
                for n in range(0,N-3,3):
                    trotter_gates[0][n] = intGate(PXP_trotter,n,3)
                for n in range(1,N-2,3):
                    trotter_gates[1][n] = intGate(PXP_trotter,n,3)
                for n in range(2,N-4,3):
                    trotter_gates[2][n] = intGate(PXP_trotter,n,3)
                trotter_gates[2][0] = intGate(XP_trotter,0,2)
                trotter_gates[2][N-2] = intGate(PX_trotter,N-2,2)

            elif N % 3 == 2:
                for n in range(0,N-4,3):
                    trotter_gates[0][n] = intGate(PXP_trotter,n,3)
                trotter_gates[0][N-2] = intGate(PX_trotter,N-2,2)
                for n in range(1,N-3,3):
                    trotter_gates[1][n] = intGate(PXP_trotter,n,3)
                for n in range(2,N-2,3):
                    trotter_gates[2][n] = intGate(PXP_trotter,n,3)
                trotter_gates[2][0] = intGate(XP_trotter,0,2)
            return trotter_gates


        elif trotter_order == 2:
            XP_trotter_half = np.reshape(sp.linalg.expm(-1j*H_XP*delta_t/2),(phys_dim,phys_dim,phys_dim,phys_dim))
            PXP_trotter_half = np.reshape(sp.linalg.expm(-1j*H_PXP*delta_t/2),(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            PX_trotter_half = np.reshape(sp.linalg.expm(-1j*H_PX*delta_t/2),(phys_dim,phys_dim,phys_dim,phys_dim))

            XP_trotter= np.reshape(sp.linalg.expm(-1j*H_XP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))
            PXP_trotter= np.reshape(sp.linalg.expm(-1j*H_PXP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            PX_trotter= np.reshape(sp.linalg.expm(-1j*H_PX*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))

            trotter_gates = dict()
            H0_tau = dict()
            H1_tau_half = dict()
            H2_tau_half = dict()

            if N % 3 == 0:
                for n in range(0,N-2,3):
                    H0_tau[n] = intGate(PXP_trotter,n,3)
                for n in range(1,N-4,3):
                    H1_tau_half[n] = intGate(PXP_trotter_half,n,3)
                H1_tau_half[N-2] = intGate(PX_trotter_half,N-2,2)
                for n in range(2,N-3,3):
                    H2_tau_half[n] = intGate(PXP_trotter_half,n,3)
                H2_tau_half[0] = intGate(XP_trotter_half,0,2)

            elif N % 3 == 1:
                for n in range(0,N-3,3):
                    H0_tau[n] = intGate(PXP_trotter,n,3)
                for n in range(1,N-2,3):
                    H1_tau_half[n] = intGate(PXP_trotter_half,n,3)
                for n in range(2,N-4,3):
                    H2_tau_half[n] = intGate(PXP_trotter_half,n,3)

                H2_tau_half[0] = intGate(XP_trotter_half,0,2)
                H2_tau_half[N-2] = intGate(PX_trotter_half,N-2,2)

            elif N % 3 == 2:
                for n in range(0,N-4,3):
                    H0_tau[n] = intGate(PXP_trotter,n,3)
                H0_tau[N-2] = intGate(PX_trotter,N-2,2)
                for n in range(1,N-3,3):
                    H1_tau_half[n] = intGate(PXP_trotter_half,n,3)
                for n in range(2,N-2,3):
                    H2_tau_half[n] = intGate(PXP_trotter_half,n,3)
                H2_tau_half[0] = intGate(XP_trotter_half,0,2)

            trotter_gates[0] = H2_tau_half
            trotter_gates[1] = H1_tau_half
            trotter_gates[2] = H0_tau
            trotter_gates[3] = H1_tau_half
            trotter_gates[4] = H2_tau_half
            return trotter_gates
