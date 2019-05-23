#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
from progressbar import ProgressBar
from common_MPOs import common_mpo
import matplotlib.pyplot as plt
from DMRG import dmrg
from MPS import *
from compression import *
from trotter_MPO import *
tau = 0.1
N=30
D=2
Nc=2
L,W,Q = pcp_trotter_mpo(tau,Nc)
step1=open_MPO(N)
step2=open_MPO(N)
step3=open_MPO(N)

for n in range(0,step1.length-2,3):
    step1.set_entry(n,L,"right")
    step1.set_entry(n+1,W,"both")
    step1.set_entry(n+2,Q,"left")

for n in range(1,step1.length-4,3):
    step2.set_entry(n,L,"right")
    step2.set_entry(n+1,W,"both")
    step2.set_entry(n+2,Q,"left")

for n in range(2,step1.length-3,3):
    step3.set_entry(n,L,"right")
    step3.set_entry(n+1,W,"both")
    step3.set_entry(n+2,Q,"left")

#neel state
I = np.eye(1)
A = np.zeros([Nc,1,1])
A[0] = 1
B=np.zeros([Nc,1,1])
B[Nc-1] = 1
Vl=np.zeros([Nc,1])
Vl[0]=  1
Vr=np.zeros([Nc,1])
Vr[Nc-1]=  1
psi = open_MPS(N)
psi.set_entry(0,Vl,"right")
for n in range(1,N-2,2):
    psi.set_entry(n,A,"both")
for n in range(2,N-1,2):
    psi.set_entry(n,B,"both")
psi.set_entry(N-1,Vr,"left")

psi_orig = copy.deepcopy(psi)

f = []
t_max = 20
t=np.arange(0,t_max,tau)
no_steps = np.size(t)
pbar=ProgressBar()
print("Time evolving")
errors = []
for n in pbar(range(0,no_steps)):
    overlap = np.abs(psi.dot(psi_orig))**2
    f=np.append(f,overlap)
    step1.dot(psi)
    # psi= svd_compress(psi,D).compress(verbose=True)
    c = svd_compress(psi,D)
    psi = c.compress()
    errors = np.append(errors,c.error)
    step2.dot(psi)
    c = svd_compress(psi,D)
    psi = c.compress()
    errors = np.append(errors,c.error)
    # psi= svd_compress(psi,D).compress(verbose=True)
    step3.dot(psi)
    c = svd_compress(psi,D)
    psi = c.compress()
    errors = np.append(errors,c.error)
    # psi= svd_compress(psi,D).compress(verbose=True)
    # print(f[n])

    # psi= svd_compress(psi,D).compress()
    # compressor = var_compress(psi,D,verbose=True)
    # compressor = var_compress(psi,D,psi_trial=psi_trial,verbose=True)
    # psi = compressor.compress(5e-1,max_sweeps=5000)

plt.plot(errors)
plt.show()

plt.plot(t,f)
plt.show()
