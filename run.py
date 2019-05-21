#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network

N=100
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
D=2
H = common_mpo.Heis(N,"open")
# psi = mps.uniform(N,A,B,B)
psi = mps.uniform(N,A,B,B)
# psi = mps.random(N,3,2,boundary="periodic")
psi.left_normalize()
H.act_on(psi)
H.act_on(psi)
psi.left_normalize()
# psi_trial = svd_compress(psi,D).compress()
# compressor = var_compress(psi,D,psi_trial=psi_trial)
compressor = var_compress(psi,D)
psi = compressor.compress(1e-5)
plt.plot(compressor.distance)
plt.show()
    
