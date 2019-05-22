#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import *
class common_mpo:
    def  Ising(length,J,h,boundary):
        #J*ZZ-hX
        Z=np.array([[1,0],[0,-1]])
        I=np.array([[1,0],[0,1]])
        X=np.array([[0,1],[1,0]])

        Q=np.zeros(np.array((3,3,2,2)))
        Q[0,0] = I
        Q[1,0] = Z
        Q[2,0] = h*X

        Q[2,1] = J*Z
        Q[2,2] = I

        V=np.zeros(np.array((3,2,2)))
        V[0] = h*X
        V[1] = J*Z
        V[2] = I
        W=np.zeros(np.array((3,2,2)))
        W[0] = I
        W[1] = Z
        W[2] = h*X

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def  Heis(length,J,boundary):
        #J*dot(S_i,S_i+1)
        Sp=np.array([[0,1],[0,0]])
        Sm=np.array([[0,0],[1,0]])
        I=np.array([[1,0],[0,1]])
        Z=np.array([[1/2,0],[0,-1/2]])

        Q=np.zeros(np.array((5,5,2,2)))
        Q[0,0] = I
        Q[1,0] = Sp
        Q[2,0] = Sm
        Q[3,0] = Z
        Q[4,0] = 0

        Q[4,1] = J/2*Sm
        Q[4,2] = J/2*Sp
        Q[4,3] = J*Z
        Q[4,4] = I

        V=np.zeros(np.array((5,2,2)))
        V[0] = 0
        V[1] = J/2*Sm
        V[2] = J/2*Sp
        V[3] = J*Z
        V[4] = I
        W=np.zeros(np.array((5,2,2)))
        W[0] = I
        W[1] = Sp
        W[2] = Sm
        W[3] = Z
        W[4] = 0

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def  PXP(length,boundary):
        X=np.array([[0,1],[1,0]])
        P = np.array([[0,0],[0,1]])
        I = np.array([[1,0],[0,1]])

        Q=np.zeros(np.array((4,4,2,2)))
        Q[0,0] = I
        Q[1,0] = P
        Q[2,1] = X
        Q[3,2] = P
        Q[3,3] = I

        V=np.zeros(np.array((4,2,2)))
        V[2] = P
        V[3] = I
        W=np.zeros(np.array((4,2,2)))
        W[0] = I
        W[1] = P

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H
