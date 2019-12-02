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

class common_trotters:
    def PXP(N,phys_dim,delta_t,trotter_order=1):
        P = np.array([[1,0],[0,0]])
        X = np.array([[0,1],[1,0]])
        H_XP = np.kron(X,P)
        H_PXP = np.kron(np.kron(P,X),P)
        H_PX = np.kron(P,X)
        if trotter_order == 1:
            XP_trotter = np.reshape(sp.linalg.expm(-1j*H_XP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))
            PXP_trotter = np.reshape(sp.linalg.expm(-1j*H_PXP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            PX_trotter = np.reshape(sp.linalg.expm(-1j*H_PX*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))

            # XP_trotter = np.reshape(np.eye(np.size(H_XP,axis=0))-1j*delta_t*H_XP,(phys_dim,phys_dim,phys_dim,phys_dim))
            # PXP_trotter = np.reshape(np.eye(np.size(H_PXP,axis=0))-1j*delta_t*H_PXP,(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            # PX_trotter = np.reshape(np.eye(np.size(H_PX,axis=0))-1j*delta_t*H_PX,(phys_dim,phys_dim,phys_dim,phys_dim))
            trotter_gates = dict()
            trotter_gates[0] = dict()
            trotter_gates[1] = dict()
            trotter_gates[2] = dict()

            if N % 3 == 0:
                for n in range(0,N-2,3):
                    trotter_gates[0][n] = PXP_trotter
                for n in range(1,N-4,3):
                    trotter_gates[1][n] = PXP_trotter
                trotter_gates[1][N-2] = PX_trotter
                for n in range(2,N-3,3):
                    trotter_gates[2][n] = PXP_trotter
                trotter_gates[2][0] = XP_trotter

            elif N % 3 == 1:
                for n in range(0,N-3,3):
                    trotter_gates[0][n] = PXP_trotter
                for n in range(1,N-2,3):
                    trotter_gates[1][n] = PXP_trotter
                for n in range(2,N-4,3):
                    trotter_gates[2][n] = PXP_trotter
                trotter_gates[2][0] = XP_trotter
                trotter_gates[2][N-2] = PX_trotter

            elif N % 3 == 2:
                for n in range(0,N-4,3):
                    trotter_gates[0][n] = PXP_trotter
                trotter_gates[0][N-2] = PX_trotter
                for n in range(1,N-3,3):
                    trotter_gates[1][n] = PXP_trotter
                for n in range(2,N-2,3):
                    trotter_gates[2][n] = PXP_trotter
                trotter_gates[2][0] = XP_trotter
            return trotter_gates

        elif trotter_order == 2:
            XP_trotter_half = np.reshape(sp.linalg.expm(-1j*H_XP*delta_t/2),(phys_dim,phys_dim,phys_dim,phys_dim))
            PXP_trotter_half = np.reshape(sp.linalg.expm(-1j*H_PXP*delta_t/2),(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            PX_trotter_half = np.reshape(sp.linalg.expm(-1j*H_PX*delta_t/2),(phys_dim,phys_dim,phys_dim,phys_dim))

            XP_trotter= np.reshape(sp.linalg.expm(-1j*H_XP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))
            PXP_trotter= np.reshape(sp.linalg.expm(-1j*H_PXP*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            PX_trotter= np.reshape(sp.linalg.expm(-1j*H_PX*delta_t),(phys_dim,phys_dim,phys_dim,phys_dim))

            # XP_trotter_half = np.reshape(np.eye(np.size(H_XP,axis=0))-1j*delta_t/2*H_XP,(phys_dim,phys_dim,phys_dim,phys_dim))
            # PXP_trotter_half = np.reshape(np.eye(np.size(H_PXP,axis=0))-1j*delta_t/2*H_PXP,(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            # PX_trotter_half = np.reshape(np.eye(np.size(H_PX,axis=0))-1j*delta_t/2*H_PX,(phys_dim,phys_dim,phys_dim,phys_dim))

            # XP_trotter = np.reshape(np.eye(np.size(H_XP,axis=0))-1j*delta_t*H_XP,(phys_dim,phys_dim,phys_dim,phys_dim))
            # PXP_trotter = np.reshape(np.eye(np.size(H_PXP,axis=0))-1j*delta_t*H_PXP,(phys_dim,phys_dim,phys_dim,phys_dim,phys_dim,phys_dim))
            # PX_trotter = np.reshape(np.eye(np.size(H_PX,axis=0))-1j*delta_t*H_PX,(phys_dim,phys_dim,phys_dim,phys_dim))


            trotter_gates = dict()
            H0_tau = dict()
            H1_tau_half = dict()
            H2_tau_half = dict()

            if N % 3 == 0:
                for n in range(0,N-2,3):
                    H0_tau[n] = PXP_trotter
                for n in range(1,N-4,3):
                    H1_tau_half[n] = PXP_trotter_half
                H1_tau_half[N-2] = PX_trotter_half
                for n in range(2,N-3,3):
                    H2_tau_half[n] = PXP_trotter_half
                H2_tau_half[0] = XP_trotter_half

            elif N % 3 == 1:
                for n in range(0,N-3,3):
                    H0_tau[n] = PXP_trotter
                for n in range(1,N-2,3):
                    H1_tau_half[n] = PXP_trotter_half
                for n in range(2,N-4,3):
                    H2_tau_half[n] = PXP_trotter_half
                H2_tau_half[0] = XP_trotter_half
                H2_tau_half[N-2] = PX_trotter_half

            elif N % 3 == 2:
                for n in range(0,N-4,3):
                    H0_tau[n] = PXP_trotter
                H0_tau[N-2] = PX_trotter
                for n in range(1,N-3,3):
                    H1_tau_half[n] = PXP_trotter_half
                for n in range(2,N-2,3):
                    H2_tau_half[n] = PXP_trotter_half
                H2_tau_half[0] = XP_trotter_half

            trotter_gates[0] = H2_tau_half
            trotter_gates[1] = H1_tau_half
            trotter_gates[2] = H0_tau
            trotter_gates[3] = H1_tau_half
            trotter_gates[4] = H2_tau_half
            return trotter_gates

    def XX(N,phys_dim,delta_t,trotter_order = 1):
        H_uc = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        if trotter_order == 1:
            trotter_gate = sp.linalg.expm(-1j*H_uc*delta_t)
            trotter_gate = trotter_gate.reshape(phys_dim,phys_dim,phys_dim,phys_dim)
            trotter_gates = dict()
            trotter_gates[0] = dict()
            trotter_gates[1] = dict()
            loc_even = np.arange(0,N-1,2)
            loc_odd = np.arange(1,N-2,2)
            for n in range(0,np.size(loc_even,axis=0)):
                trotter_gates[0][loc_even[n]] = trotter_gate
            for n in range(0,np.size(loc_odd,axis=0)):
                trotter_gates[1][loc_odd[n]] = trotter_gate
            return trotter_gates
        elif trotter_order == 2:
            trotter_gate_half = sp.linalg.expm(-1j*H_uc*delta_t/2)
            trotter_gate = sp.linalg.expm(1j*H_uc*delta_t)

            trotter_gate = trotter_gate.reshape(phys_dim,phys_dim,phys_dim,phys_dim)
            trotter_gate_half = trotter_gate_half.reshape(phys_dim,phys_dim,phys_dim,phys_dim)

            trotter_gates = dict()
            trotter_gates[0] = dict()
            trotter_gates[1] = dict()
            trotter_gates[2] = dict()
            loc_even = np.arange(0,N-1,2)
            loc_odd = np.arange(1,N-2,2)
            for n in range(0,np.size(loc_even,axis=0)):
                trotter_gates[1][loc_even[n]] = trotter_gate
            for n in range(0,np.size(loc_odd,axis=0)):
                trotter_gates[0][loc_odd[n]] = trotter_gate_half
                trotter_gates[2][loc_odd[n]] = trotter_gate_half
            return trotter_gates

    def Heis(J,N,phys_dim,delta_t,trotter_order=1):
        s = 1/2*(phys_dim-1)
        m = np.arange(-s,s)
        couplings = np.power(s*(s+1)-m*(m+1),0.5)
        Hp = np.diag(couplings,1)
        Hm = np.diag(couplings,-1)
        x = 1/2 * (Hp+Hm)
        y = 1/(2j) * (Hp-Hm)
        z = 1/2*(np.dot(Hp,Hm)-np.dot(Hm,Hp))
        H = J*(np.kron(x,x)+np.kron(y,y)+np.kron(z,z))
        if trotter_order == 1:
            trotter_gate = sp.linalg.expm(-1j*H*delta_t)
            trotter_gate = trotter_gate.reshape(phys_dim,phys_dim,phys_dim,phys_dim)
            trotter_gates = dict()
            trotter_gates[0] = dict()
            trotter_gates[1] = dict()
            loc_even = np.arange(0,N-1,2)
            loc_odd = np.arange(1,N-2,2)
            for n in range(0,np.size(loc_even,axis=0)):
                trotter_gates[0][loc_even[n]] = trotter_gate
            for n in range(0,np.size(loc_odd,axis=0)):
                trotter_gates[1][loc_odd[n]] = trotter_gate
            return trotter_gates
        elif trotter_order == 2:
            trotter_gate_half = sp.linalg.expm(-1j*H*delta_t/2)
            trotter_gate = sp.linalg.expm(-1j*H*delta_t)

            trotter_gate = trotter_gate.reshape(phys_dim,phys_dim,phys_dim,phys_dim)
            trotter_gate_half = trotter_gate_half.reshape(phys_dim,phys_dim,phys_dim,phys_dim)

            trotter_gates = dict()
            trotter_gates[0] = dict()
            trotter_gates[1] = dict()
            trotter_gates[2] = dict()
            loc_even = np.arange(0,N-1,2)
            loc_odd = np.arange(1,N-2,2)
            for n in range(0,np.size(loc_even,axis=0)):
                trotter_gates[1][loc_even[n]] = trotter_gate
            for n in range(0,np.size(loc_odd,axis=0)):
                trotter_gates[0][loc_odd[n]] = trotter_gate_half
                trotter_gates[2][loc_odd[n]] = trotter_gate_half
            return trotter_gates
