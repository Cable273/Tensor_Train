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
D=10
delta_t = 0.01
t_max = 20

from common_trotters import common_trotters
trotter_gates = common_trotters.PXP(N,phys_dim,delta_t,trotter_order = 1)
# trotter_gates = common_trotters.XX(N,phys_dim,delta_t,trotter_order=1)
# trotter_gates = common_trotters.Heis(1,N,phys_dim,delta_t,trotter_order = 2)

#initial state
# Neel state (init state)
I = np.eye(1)
A = np.zeros([phys_dim,1,1])
A[0] = 1
B=np.zeros([phys_dim,1,1])
B[phys_dim-1] = 1
Vl=np.zeros([phys_dim,1])
Vl[0]=  1
Vr=np.zeros([phys_dim,1])
Vr[phys_dim-1]=  1
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
plt.title(r"$H=PXP$ TEBD $\vert Z_2 \rangle$ Fidelity, $N=$"+str(N)+r", $D=$"+str(D)+r", $\Delta t=$"+str(delta_t))
plt.show()

# plt.plot(tebd.t,tebd.error)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\sum_n \vert \vert \vert \psi_{exact}^n \rangle - \vert \psi_{approx}^n \rangle \vert \vert^2$")
# plt.title(r"TEBD Cumulative Truncation Error")
# plt.show()
