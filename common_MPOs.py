#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import *
class common_mpo:
    def  Heis(length,J,h,boundary):
        #Heisenberg MPO
        Z=np.array([[1,0,0],[0,1,0],[0,0,-1]])
        I=np.array([[1,0,0],[0,1,0],[0,0,1]])
        X=np.array([[0,np.power(2,0.5),0],[np.power(2,0.5),0,np.power(2,0.5)],[0,np.power(2,0.5),0]])

        Q=np.zeros(np.array((3,3,3,3)))
        Q[0,0] = I
        Q[1,0] = Z
        Q[2,0] = -h*X

        Q[2,1] = J*Z
        Q[2,2] = I

        V=np.zeros(np.array((3,3,3)))
        V[0] = -h*X
        V[1] = J*Z
        V[2] = I
        W=np.zeros(np.array((3,3,3)))
        W[0] = I
        W[1] = X
        W[2] = -h*X

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def  PXP(length,boundary):
        #Heisenberg MPO
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
