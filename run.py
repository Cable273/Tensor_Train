#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network

N=400
dim = 3
A=np.zeros(np.array((dim,2,2)))
A[0] = [[0,np.power(2/3,0.5)],[0,0]]
A[1] = [[-np.power(1/3,0.5),0],[0,np.power(1/3,0.5)]]
A[2] = [[0,0],[-np.power(2/3,0.5),0]]

B=np.zeros(np.array((dim,2)))
B[0] = [1/np.power(8,0.25),0]
B[1] = [1/np.power(8,0.25),1/np.power(8,0.25)]
B[2] = [0,1/np.power(8,0.25)]

B[0] = [5,0]
B[1] = [5,5]
B[2] = [0,5]


from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import dmrg
D=4
H = common_mpo.Ising(N,1,2,"open")
# H = common_mpo.Heis(N,1,"open")
# psi = mps.random(N,2,D,boundary="open")
# psi.left_normalize(norm=True)
# print(np.shape(H.node[1].tensor))
# H2 = H.dot(H)
# print(np.shape(H2.node[1].tensor))
# print(psi.dot(psi))

method = dmrg(H,D)
psi = method.run()
print(psi.dot(psi))
# method.plot_convergence()
# method.plot_var()
