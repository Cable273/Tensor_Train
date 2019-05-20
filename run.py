#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import MPS,MPO

N=10000
dim = 3
A=np.zeros(np.array((dim,2,2)))
A[0] = [[0,2],[0,0]]
A[1] = [[-2,0],[0,1]]
A[2] = [[0,0],[-2,0]]

B=np.zeros(np.array((dim,2)))
B[0] = [1/np.power(8,0.25),0]
B[1] = [1/np.power(8,0.25),1/np.power(8,0.25)]
B[2] = [0,1/np.power(8,0.25)]

B[0] = [5,0]
B[1] = [5,5]
B[2] = [0,5]

#Heisenberg MPO
J=-1
h=1 
Z=np.array([[1,0,0],[0,1,0],[0,0,-1]])
I=np.array([[1,0,0],[0,1,0],[0,0,1]])
X=np.array([[0,np.power(2,0.5),0],[np.power(2,0.5),0,np.power(2,0.5)],[0,np.power(2,0.5),0]])

Q=np.zeros(np.array((3,3,3,3)))
Q[0,0] = I
Q[0,1] = Z
Q[0,2] = -h*X
Q[2,1] = J*Z
Q[2,2] = I
V=np.zeros(np.array((3,3,3)))
V[0] = -h*X
V[1] = J*Z
V[2] = I
W=np.zeros(np.array((3,3,3)))
W[0] = I
W[1] = J*Z
W[2] = -h*X

# from MPS import compress_MPS
# H = MPO.uniform(N,Q,V,W)
psi1 = MPS.uniform(N,A)
print(psi1.overlap(psi1))
psi1.left_normalize()
print(psi1.overlap(psi1))
