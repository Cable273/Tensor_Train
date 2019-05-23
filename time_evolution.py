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
def set_steps(trotter,tau):
    L,W,Q = trotter.mpo(tau)
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
    return step1,step2,step3

def odd_even_gates(trotter,tau):
    A,B = trotter.mpo(tau)
    gate1=open_MPO(N)
    gate2=open_MPO(N)
    for n in range(0,gate1.length-1,2):
        gate1.set_entry(n,A,"right")
        gate1.set_entry(n+1,B,"left")
    for n in range(1,gate2.length-2,2):
        gate2.set_entry(n,A,"right")
        gate2.set_entry(n+1,B,"left")
    return gate1,gate2

def fourth_order_tau(tau):
    t1 = 1/(4-np.power(4,1/3))*tau
    t2 = t1
    t3 = tau - 2*t1 - 2*t2
    return t1,t2,t3

tau = 0.1
N=10
D=2
Nc=2
trotter = two_site_trotter.factory("xx")
step1,step2 = odd_even_gates(trotter,tau)
# step1,step2,step3 = set_steps(trotter,tau)

# neel state
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

from Tensor_Train import rail_network
f = []
t_max = 20
t=np.arange(0,t_max,tau)
pbar=ProgressBar()
print("Time evolving")
errors = []
for n in pbar(range(0,np.size(t))):
    overlap = np.abs(psi.dot(psi_orig))**2
    f=np.append(f,overlap)

    # step1.dot(psi)
    # c = svd_compress(psi,D)
    # psi = c.compress()
    # errors = np.append(errors,c.error)
    # step2.dot(psi)
    # c = svd_compress(psi,D)
    # psi = c.compress()
    # errors = np.append(errors,c.error)
    # step3.dot(psi)
    # c = svd_compress(psi,D)
    # psi = c.compress()
    # errors = np.append(errors,c.error)

    step1.dot(psi)
    c = svd_compress(psi,D)
    psi = c.compress()
    errors = np.append(errors,c.error)
    step2.dot(psi)
    c = svd_compress(psi,D)
    psi = c.compress()
    errors = np.append(errors,c.error)

plt.plot(t,f)
plt.show()
plt.plot(errors)
plt.show()
