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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state
import math

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N=6
phys_dim = 2
delta_t = 0.01
t_max = 10

# H = common_mpo.PXP(N,"open")
# H = common_mpo.Heis(N,1,"open")
H = common_mpo.XX(N,"open")

#initial state
# Neel state (init state)
d=4
I = np.eye(d)
A = np.zeros([phys_dim,d,d])
A[0] = I
# A[0,0,0] = 1
B=np.zeros([phys_dim,d,d])
B[phys_dim-1] = I
# B[phys_dim-1,0,0] = 1

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

# system = unlocking_System([0,1],"open",2,N)
# system.gen_basis()
# wf = np.zeros(system.dim,dtype=complex)
# for n in range(0,np.size(system.basis_refs,axis=0)):
    # bits = system.basis[n]
    # coef = psi.node[0].tensor[bits[0]]
    # for m in range(1,np.size(bits,axis=0)):
        # coef = np.dot(coef,psi.node[m].tensor[bits[m]])
    # wf[n] = coef
# from Diagnostics import print_wf
# print_wf(wf,system,1e-2)

from integrators import expiH
from TDVP import TDVP
tdvp = TDVP(H,psi)
# tdvp.run(delta_t,t_max,integrator = expiH.krylov)
# tdvp.run(delta_t,t_max,integrator = expiH.euler)
tdvp.run(delta_t,t_max,integrator = expiH.rungeKutta4)

from integrators import *
plt.plot(tdvp.t,tdvp.f)
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.xlabel(r"$t$")
plt.title(r"TDVP Time Evolution, $N=$"+str(N))
plt.show()
# plt.plot(tdvp.t,tdvp.energy)
# plt.show()

