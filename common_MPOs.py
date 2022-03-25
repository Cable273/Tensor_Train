#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import *
def dsum(a,b):
    M = np.zeros(())


class common_mpo:
    def  Ising(length,J,h,boundary):
        #J*ZZ-hX
        X=np.array([[1/2,0],[0,-1/2]])
        I=np.array([[1,0],[0,1]])
        Z=np.array([[0,1/2],[1/2,0]])

        Q=np.zeros(np.array((3,3,2,2)))
        Q[0,0] = I
        Q[1,0] = Z
        Q[2,0] = -h*X

        Q[2,1] = J*Z
        Q[2,2] = I

        V=np.zeros(np.array((3,2,2)))
        V[0] = -h*X
        V[1] = J*Z
        V[2] = I
        W=np.zeros(np.array((3,2,2)))
        W[0] = I
        W[1] = Z
        W[2] = -h*X

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def  Ising_longi(length,J,hx,hz,boundary):
        #J*ZZ+ hx X +hz Z
        Z=np.array([[-1,0],[0,1]])
        I=np.array([[1,0],[0,1]])
        X=np.array([[0,1],[1,0]])

        Q=np.zeros(np.array((3,3,2,2)))
        Q[0,0] = I
        Q[1,0] = Z
        Q[2,0] = hx*X+hz*Z

        Q[2,1] = J*Z
        Q[2,2] = I

        V=np.zeros(np.array((3,2,2)))
        V[0] = hx*X+hz*Z
        V[1] = J*Z
        V[2] = I

        W=np.zeros(np.array((3,2,2)))
        W[0] = I
        W[1] = Z
        W[2] = hx*X+hz*Z


        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def  Heis(length,J,boundary):
        #J*dot(S_i,S_i+1)
        Sp=np.array([[0,0],[1,0]])
        Sm=np.array([[0,1],[0,0]])
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
        P = np.array([[1,0],[0,0]])
        I = np.array([[1,0],[0,1]])

        Q=np.zeros(np.array((4,4,2,2)))
        Q[0,0] = I
        Q[1,0] = P
        Q[2,1] = X
        Q[3,2] = P
        Q[3,3] = I

        V=np.zeros(np.array((4,2,2)))
        V[1] = X
        V[2] = P
        V[3] = I
        W=np.zeros(np.array((4,2,2)))
        W[0] = I
        W[1] = P
        W[2] = X

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def  XX(length,boundary):
        X=np.array([[0,1],[1,0]])
        I=np.array([[1,0],[0,1]])

        Q=np.zeros(np.array((3,3,2,2)))
        Q[0,0] = I
        Q[1,0] = X

        Q[2,1] = X
        Q[2,2] = I

        V=np.zeros(np.array((3,2,2)))
        V[1] = X
        V[2] = I
        W=np.zeros(np.array((3,2,2)))
        W[0] = I
        W[1] = X

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def zz(length,boundary):
        Z=np.array([[-1,0],[0,1]])
        I=np.array([[1,0],[0,1]])

        Q=np.zeros(np.array((3,3,2,2)))
        Q[0,0] = I
        Q[1,0] = Z

        Q[2,1] = Z
        Q[2,2] = I

        V=np.zeros(np.array((3,2,2)))
        V[1] = Z
        V[2] = I
        W=np.zeros(np.array((3,2,2)))
        W[0] = I
        W[1] = Z

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def Z(length,boundary):
        Z=np.array([[-1,0],[0,1]])
        I=np.array([[1,0],[0,1]])

        Q=np.zeros(np.array((2,2,2,2)))
        Q[0,0] = I
        Q[1,0] = Z
        Q[1,1] = I

        V=np.zeros(np.array((2,2,2)))
        V[0] = Z
        V[1] = I
        W=np.zeros(np.array((2,2,2)))
        W[0] = I
        W[1] = Z

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H

    def X(length,boundary):
        X=np.array([[0,1],[1,0]])
        I=np.array([[1,0],[0,1]])

        Q=np.zeros(np.array((2,2,2,2)))
        Q[0,0] = I
        Q[1,0] = X
        Q[1,1] = I

        V=np.zeros(np.array((2,2,2)))
        V[0] = X
        V[1] = I
        W=np.zeros(np.array((2,2,2)))
        W[0] = I
        W[1] = X

        if boundary == "periodic":
            H = mpo.uniform(length,Q)
        else:
            H = mpo.uniform(length,Q,V,W)
        return H
