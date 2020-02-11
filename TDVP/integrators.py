#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def orth(v,orthonormalBasis):
    newV = np.copy(v)
    if np.size(np.shape(orthonormalBasis))>1:
        for n in range(0,np.size(orthonormalBasis,axis=0)):
            newV = newV - np.vdot(newV,orthonormalBasis[n,:])*orthonormalBasis[n,:]
    else:
        newV = newV - np.vdot(newV,orthonormalBasis)*orthonormalBasis
    if np.abs(np.vdot(newV,newV))<1e-5:
        return None
    else:
        return newV / np.power(np.vdot(newV,newV),0.5)

class expiH:
    def krylov(H,psi,t):
        psi0 = psi / np.power(np.vdot(psi,psi),0.5)
        kBasis = psi0
        currentState = psi0
        psiLast = np.exp(-1j*np.vdot(psi0,np.dot(H,psi0))*t)*psi
        # for n in range(0,np.size(psi0)):
        for n in range(0,10):
            nextState = np.dot(H,currentState)
            nextStateOrth = orth(nextState,kBasis)
            if nextStateOrth is not None:
                kBasis = np.vstack((kBasis,nextStateOrth))
                currentState = nextStateOrth

                basis = np.transpose(kBasis)
                psik = np.dot(np.conj(np.transpose(basis)),psi)
                Hk = np.dot(np.conj(np.transpose(basis)),np.dot(H,basis))

                expH = sp.linalg.expm(-1j*Hk*t)
                psiNew = np.dot(expH,psik)
                psiNew = np.dot(basis,psiNew)

                if (np.abs(psiNew-psiLast)).all()<1e-2:
                    psiLast = psiNew
                    break
                else:
                    psiLast = psiNew
        # print(n)
        return psiLast

    def euler(H,psi,t):
        U = np.eye(np.size(H,axis=0)) - 1j * H * t
        return np.dot(U,psi)
