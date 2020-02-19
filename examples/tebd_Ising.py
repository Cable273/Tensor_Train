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

N=10
phys_dim = 2
D=100
delta_t = 0.01
t_max = 20

from common_trotters import common_trotters, uniformTrotters, intGate, swapGate

#H = sum_n J x_n x_{n+1} + h z_n

J = 1
h = 1
x = np.array([[0,1],[1,0]])
I = np.eye(2)
z = np.array([[-1,0],[0,1]])

H = J* np.kron(x,x) + h* np.kron(z,I)
trotter_gates = uniformTrotters.gen(N,H,2,phys_dim,delta_t,trotter_order=1)
tempTensor = sp.linalg.expm(-1j*np.kron(I,z)*delta_t).reshape(phys_dim,phys_dim,phys_dim,phys_dim)
trotter_gates[2] = dict()
trotter_gates[2][0] = intGate(tempTensor,N-2,2)

#initial state
# Neel state (init state)
I = np.eye(1)
A = np.zeros([phys_dim,1,1])
A[0] = 1
B=np.zeros([phys_dim,1,1])
B[phys_dim-1] = 1
Vr=np.zeros([phys_dim,1])
Vr[0]=  1
Vl=np.zeros([phys_dim,1])
Vl[phys_dim-1]=  1
psi = open_MPS(N)
psi.set_entry(0,Vl,"right")
for n in range(1,N-2,2):
    psi.set_entry(n,A,"both")
for n in range(2,N-1,2):
    psi.set_entry(n,B,"both")
psi.set_entry(N-1,Vr,"left")
psi.right_normalize()

from TEBD import TEBD
tebd = TEBD(trotter_gates,psi,D)
tebd.run(delta_t,t_max)
tebd.plot_fidelity()
plt.title(r"TEBD $\vert Z_2 \rangle$ Fidelity, $N=$"+str(N)+r", $D=$"+str(D)+r", $\Delta t=$"+str(delta_t))
plt.show()

plt.plot(tebd.t,tebd.error)
plt.xlabel(r"$t$")
plt.ylabel(r"$\sum \sigma_{discarded}^2$")
plt.title(r"TEBD Cumulative Truncation Error")
plt.show()

plt.plot(tebd.t,tebd.entropy)
plt.xlabel(r"$t$")
plt.ylabel(r"$S$")
plt.title(r"TEBD Entropy")
plt.show()
