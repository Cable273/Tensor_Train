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
    def Ising_spinHalf(N,J,hx,hz,delta_t):
        phys_dim = 2
        Z = np.array([[-1,0],[0,1]])
        X = np.array([[0,1],[1,0]])
        I = np.eye(2)

        H_main = J*np.kron(Z,Z) + hx/2*(np.kron(I,X)+np.kron(X,I)) + hz/2*(np.kron(I,Z)+np.kron(Z,I))
        H_boundary = hx/2*X + hz/2*Z

        # H_main = J*np.kron(Z,Z) + hx*np.kron(I,X) + hz*np.kron(Z,I)
        # H_boundary = hx*X + hz*Z

        H_main_trotter = np.reshape(sp.linalg.expm(-1j*H_main*delta_t), (phys_dim,phys_dim,phys_dim,phys_dim))
        boundary_trotter = np.reshape(sp.linalg.expm(-1j*H_boundary*delta_t),(phys_dim,phys_dim))

        trotter_gates = dict()
        trotter_gates[0] = dict()
        trotter_gates[1] = dict()

        if N % 2 == 0:
            for n in range(0,N-1,2):
                trotter_gates[0][n] = intGate(H_main_trotter,n,2)

            trotter_gates[1][0] = intGate(boundary_trotter,0,1)
            for n in range(1,N-2,2):
                trotter_gates[1][n] = intGate(H_main_trotter,n,2)
            trotter_gates[1][N-1] = intGate(boundary_trotter,N-1,1)
        return trotter_gates


    def PXP(N,phys_dim,delta_t,trotter_order=1):
        s = (phys_dim-1)/2
        m = np.arange(-s,s)
        couplings = np.power(s*(s+1)-m*(m+1),0.5)
        P = np.zeros((phys_dim,phys_dim))
        P[0,0] = 1
        X = (2*(np.diag(couplings,1) + np.diag(couplings,-1)))/2
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

    def PPXPP(N,phys_dim,delta_t):
        s = (phys_dim-1)/2
        m = np.arange(-s,s)
        couplings = np.power(s*(s+1)-m*(m+1),0.5)
        P = np.zeros((phys_dim,phys_dim),dtype=complex)
        P[0,0] = 1
        X = (2*(np.diag(couplings,1).astype(complex) + np.diag(couplings,-1)).astype(complex))/2

        H_XPP = np.kron(np.kron(X,P),P)
        H_PXPP = np.kron(np.kron(np.kron(P,X),P),P)
        H_PPXPP = np.kron(np.kron(np.kron(np.kron(P,P),X),P),P)
        H_PPXP = np.kron(np.kron(np.kron(P,P),X),P)
        H_PPX = np.kron(np.kron(P,P),X)

        XPP_trotter = np.reshape(sp.linalg.expm(-1j*H_XPP*delta_t),np.ones(6,dtype=int)*phys_dim)
        PXPP_trotter = np.reshape(sp.linalg.expm(-1j*H_PXPP*delta_t),np.ones(8,dtype=int)*phys_dim)
        PPXPP_trotter = np.reshape(sp.linalg.expm(-1j*H_PPXPP*delta_t),np.ones(10,dtype=int)*phys_dim)
        PPXP_trotter = np.reshape(sp.linalg.expm(-1j*H_PPXP*delta_t),np.ones(8,dtype=int)*phys_dim)
        PPX_trotter = np.reshape(sp.linalg.expm(-1j*H_PPX*delta_t),np.ones(6,dtype=int)*phys_dim)

        trotter_gates = dict()
        trotter_gates[0] = dict()
        trotter_gates[1] = dict()
        trotter_gates[2] = dict()
        trotter_gates[3] = dict()
        trotter_gates[4] = dict()

        if N % 5 == 0:
            for n in range(0,N-4,5):
                trotter_gates[0][n] = intGate(PPXPP_trotter,n,5)

            for n in range(1,N-8,5):
                trotter_gates[1][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[1][N-4] = intGate(PPXP_trotter,N-4,4)

            for n in range(2,N-7,5):
                trotter_gates[2][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[2][N-3] = intGate(PPX_trotter,N-3,3)

            for n in range(3,N-6,5):
                trotter_gates[3][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[3][0] = intGate(XPP_trotter,0,3)

            for n in range(4,N-5,5):
                trotter_gates[4][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[4][0] = intGate(PXPP_trotter,0,4)

        elif N % 5 == 1:
            for n in range(0,N-5,5):
                trotter_gates[0][n] = intGate(PPXPP_trotter,n,5)

            for n in range(1,N-4,5):
                trotter_gates[1][n] = intGate(PPXPP_trotter,n,5)

            for n in range(2,N-8,5):
                trotter_gates[2][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[2][N-4] = intGate(PPXP_trotter,N-4,4)

            for n in range(3,N-7,5):
                trotter_gates[3][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[3][0] = intGate(XPP_trotter,0,3)
            trotter_gates[3][N-3] = intGate(PPX_trotter,N-3,3)

            for n in range(4,N-6,5):
                trotter_gates[4][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[4][0] = intGate(PXPP_trotter,0,4)

        elif N % 5 == 2:
            for n in range(0,N-6,5):
                    trotter_gates[0][n] = intGate(PPXPP_trotter,n,5)

            for n in range(1,N-5,5):
                    trotter_gates[1][n] = intGate(PPXPP_trotter,n,5)

            for n in range(2,N-4,5):
                    trotter_gates[2][n] = intGate(PPXPP_trotter,n,5)

            for n in range(3,N-8,5):
                    trotter_gates[3][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[3][0] = intGate(XPP_trotter,0,3)
            trotter_gates[3][N-4] = intGate(PPXP_trotter,N-4,4)

            for n in range(4,N-7,5):
                    trotter_gates[4][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[4][0] = intGate(PXPP_trotter,0,4)
            trotter_gates[4][N-3] = intGate(PPX_trotter,N-3,3)

        elif N % 5 == 3:
            for n in range(0,N-7,5):
                trotter_gates[0][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[0][N-3] = intGate(PPX_trotter,N-3,3)

            for n in range(1,N-6,5):
                trotter_gates[1][n] = intGate(PPXPP_trotter,n,5)

            for n in range(2,N-5,5):
                trotter_gates[2][n] = intGate(PPXPP_trotter,n,5)

            for n in range(3,N-4,5):
                trotter_gates[3][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[3][0] = intGate(XPP_trotter,0,3)

            for n in range(4,N-8,5):
                trotter_gates[4][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[4][0] = intGate(PXPP_trotter,0,4)
            trotter_gates[4][N-4] = intGate(PPXP_trotter,N-4,4)

        elif N % 5 == 4:
            for n in range(0,N-8,5):
                trotter_gates[0][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[0][N-4] = intGate(PPXP_trotter,N-4,4)

            for n in range(1,N-7,5):
                trotter_gates[1][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[1][N-3] = intGate(PPX_trotter,N-3,3)

            for n in range(2,N-6,5):
                trotter_gates[2][n] = intGate(PPXPP_trotter,n,5)

            for n in range(3,N-5,5):
                trotter_gates[3][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[3][0] = intGate(XPP_trotter,0,3)

            for n in range(4,N-4,5):
                trotter_gates[4][n] = intGate(PPXPP_trotter,n,5)
            trotter_gates[4][0] = intGate(PXPP_trotter,0,4)

        return trotter_gates

    def pxp23(N,phys_dim,delta_t):
        p = np.array([[1,0],[0,0]])
        x = np.array([[0,1],[1,0]])
        i = np.eye(2)
        q = np.array([[0,0],[0,1]])

        def kron3(a,b,c):
            return np.kron(a,np.kron(b,c))
        def kron4(a,b,c,d):
            return np.kron(a,np.kron(b,np.kron(c,d)))
        def kron5(a,b,c,d,e):
            return np.kron(a,np.kron(b,np.kron(c,np.kron(d,e))))

        # H_3bodyLeft = kron3(x,p,p) + kron3(x,p,q) + kron3(x,q,p) + kron3(x,p,p) + kron3(x,p,p) + kron3(x,p,q) + kron3(x,q,p) + kron3(x,p,q)
        # H_3bodyRight = kron3(p,p,x) + kron3(p,p,x) + kron3(p,p,x) + kron3(p,q,x) + kron3(q,p,x) + kron3(p,q,x) + kron3(q,p,x) + kron3(q,p,x)

        # H_4bodyLeft = kron4(p,x,p,p) + kron4(p,x,p,q) + kron4(p,x,q,p) + kron4(q,x,p,p) + kron4(p,x,p,p) + kron4(q,x,p,q) + kron4(p,x,q,p) + kron4(p,x,p,q)
        # H_4bodyRight = kron4(p,p,x,p) + kron4(p,p,x,p) + kron4(p,p,x,q) + kron4(p,q,x,p) + kron4(q,p,x,p) + kron4(p,q,x,p) + kron4(q,p,x,q) + kron4(q,p,x,p)

        # H_5body = kron5(p,p,x,p,p) + kron5(p,p,x,p,q) + kron5(p,p,x,q,p) + kron5(p,q,x,p,p) + kron5(q,p,x,p,p) + kron5(p,q,x,p,q) + kron5(q,p,x,q,p) + kron5(q,p,x,p,q)

        H_3bodyLeft = kron3(x,p,p) + kron3(x,p,q) + kron3(x,q,p) 
        H_4bodyLeft = kron4(p,x,p,p) + kron4(p,x,p,q) + kron4(p,x,q,p) + kron4(q,x,p,p) +kron4(q,x,p,q)
        H_5body = kron5(p,p,x,p,p) + kron5(p,p,x,p,q) + kron5(p,p,x,q,p) + kron5(p,q,x,p,p) + kron5(q,p,x,p,p) + kron5(p,q,x,p,q) + kron5(q,p,x,q,p) + kron5(q,p,x,p,q)
        H_4bodyRight = kron4(p,p,x,p) + kron4(p,p,x,q) + kron4(p,q,x,p) + kron4(q,p,x,p)  + kron4(q,p,x,q)
        H_3bodyRight = kron3(p,p,x) + kron3(p,q,x) + kron3(q,p,x) 

        trotter_3bodyLeft = np.reshape(sp.linalg.expm(-1j*H_3bodyLeft*delta_t),np.ones(6,dtype=int)*phys_dim)
        trotter_4bodyLeft = np.reshape(sp.linalg.expm(-1j*H_4bodyLeft*delta_t),np.ones(8,dtype=int)*phys_dim)
        trotter_5body = np.reshape(sp.linalg.expm(-1j*H_5body*delta_t),np.ones(10,dtype=int)*phys_dim)
        trotter_4bodyRight = np.reshape(sp.linalg.expm(-1j*H_4bodyRight*delta_t),np.ones(8,dtype=int)*phys_dim)
        trotter_3bodyRight = np.reshape(sp.linalg.expm(-1j*H_3bodyRight*delta_t),np.ones(6,dtype=int)*phys_dim)

        trotter_gates = dict()
        trotter_gates[0] = dict()
        trotter_gates[1] = dict()
        trotter_gates[2] = dict()
        trotter_gates[3] = dict()
        trotter_gates[4] = dict()

        if N % 5 == 0:
            for n in range(0,N-4,5):
                trotter_gates[0][n] = intGate(trotter_5body,n,5)

            for n in range(1,N-8,5):
                trotter_gates[1][n] = intGate(trotter_5body,n,5)
            trotter_gates[1][N-4] = intGate(trotter_4bodyRight,N-4,4)

            for n in range(2,N-7,5):
                trotter_gates[2][n] = intGate(trotter_5body,n,5)
            trotter_gates[2][N-3] = intGate(trotter_3bodyRight,N-3,3)

            for n in range(3,N-6,5):
                trotter_gates[3][n] = intGate(trotter_5body,n,5)
            trotter_gates[3][0] = intGate(trotter_3bodyLeft,0,3)

            for n in range(4,N-5,5):
                trotter_gates[4][n] = intGate(trotter_5body,n,5)
            trotter_gates[4][0] = intGate(trotter_4bodyLeft,0,4)

        elif N % 5 == 1:
            for n in range(0,N-5,5):
                trotter_gates[0][n] = intGate(trotter_5body,n,5)

            for n in range(1,N-4,5):
                trotter_gates[1][n] = intGate(trotter_5body,n,5)

            for n in range(2,N-8,5):
                trotter_gates[2][n] = intGate(trotter_5body,n,5)
            trotter_gates[2][N-4] = intGate(trotter_4bodyRight,N-4,4)

            for n in range(3,N-7,5):
                trotter_gates[3][n] = intGate(trotter_5body,n,5)
            trotter_gates[3][0] = intGate(trotter_3bodyLeft,0,3)
            trotter_gates[3][N-3] = intGate(trotter_3bodyRight,N-3,3)

            for n in range(4,N-6,5):
                trotter_gates[4][n] = intGate(trotter_5body,n,5)
            trotter_gates[4][0] = intGate(trotter_4bodyLeft,0,4)

        elif N % 5 == 2:
            for n in range(0,N-6,5):
                    trotter_gates[0][n] = intGate(trotter_5body,n,5)

            for n in range(1,N-5,5):
                    trotter_gates[1][n] = intGate(trotter_5body,n,5)

            for n in range(2,N-4,5):
                    trotter_gates[2][n] = intGate(trotter_5body,n,5)

            for n in range(3,N-8,5):
                    trotter_gates[3][n] = intGate(trotter_5body,n,5)
            trotter_gates[3][0] = intGate(trotter_3bodyLeft,0,3)
            trotter_gates[3][N-4] = intGate(trotter_4bodyRight,N-4,4)

            for n in range(4,N-7,5):
                    trotter_gates[4][n] = intGate(trotter_5body,n,5)
            trotter_gates[4][0] = intGate(trotter_4bodyLeft,0,4)
            trotter_gates[4][N-3] = intGate(trotter_3bodyRight,N-3,3)

        elif N % 5 == 3:
            for n in range(0,N-7,5):
                trotter_gates[0][n] = intGate(trotter_5body,n,5)
            trotter_gates[0][N-3] = intGate(trotter_3bodyRight,N-3,3)

            for n in range(1,N-6,5):
                trotter_gates[1][n] = intGate(trotter_5body,n,5)

            for n in range(2,N-5,5):
                trotter_gates[2][n] = intGate(trotter_5body,n,5)

            for n in range(3,N-4,5):
                trotter_gates[3][n] = intGate(trotter_5body,n,5)
            trotter_gates[3][0] = intGate(trotter_3bodyLeft,0,3)

            for n in range(4,N-8,5):
                trotter_gates[4][n] = intGate(trotter_5body,n,5)
            trotter_gates[4][0] = intGate(trotter_4bodyLeft,0,4)
            trotter_gates[4][N-4] = intGate(trotter_4bodyRight,N-4,4)

        elif N % 5 == 4:
            for n in range(0,N-8,5):
                trotter_gates[0][n] = intGate(trotter_5body,n,5)
            trotter_gates[0][N-4] = intGate(trotter_4bodyRight,N-4,4)

            for n in range(1,N-7,5):
                trotter_gates[1][n] = intGate(trotter_5body,n,5)
            trotter_gates[1][N-3] = intGate(trotter_3bodyRight,N-3,3)

            for n in range(2,N-6,5):
                trotter_gates[2][n] = intGate(trotter_5body,n,5)

            for n in range(3,N-5,5):
                trotter_gates[3][n] = intGate(trotter_5body,n,5)
            trotter_gates[3][0] = intGate(trotter_3bodyLeft,0,3)

            for n in range(4,N-4,5):
                trotter_gates[4][n] = intGate(trotter_5body,n,5)
            trotter_gates[4][0] = intGate(trotter_4bodyLeft,0,4)

        return trotter_gates
