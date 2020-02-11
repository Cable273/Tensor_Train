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
delta_t = 0.01
t_max = 20

# H = common_mpo.PXP(N,"open")
H = common_mpo.Heis(N,1,"open")
# H = common_mpo.XX(N,"open")

#initial state
# Neel state (init state)
d=10
I = np.eye(d)
A = np.zeros([phys_dim,d,d])
A[0] = I
B=np.zeros([phys_dim,d,d])
B[phys_dim-1] = I

Vr=np.zeros([phys_dim,d])
Vr[0]=  np.zeros(np.size(I,axis=0))
Vr[0][0] = 1

Vl=np.zeros([phys_dim,d])
Vl[phys_dim-1] = np.zeros(np.size(I,axis=0))
Vl[phys_dim-1][0] = 1

psi = open_MPS(N)
psi.set_entry(0,Vl,"right")
for n in range(1,N-2,2):
    psi.set_entry(n,A,"both")
for n in range(2,N-1,2):
    psi.set_entry(n,B,"both")
psi.set_entry(N-1,Vr,"left")
print(np.abs(psi.dot(psi)))

from integrators import expiH
from TDVP import TDVP
tdvp = TDVP(H,psi)
tdvp.run(delta_t,t_max,integrator = expiH.krylov)
# tdvp.run(delta_t,t_max,integrator = expiH.euler)

from integrators import *
plt.plot(tdvp.t,tdvp.f)
plt.show()
# plt.plot(tdvp.t,tdvp.energy)
# plt.show()

