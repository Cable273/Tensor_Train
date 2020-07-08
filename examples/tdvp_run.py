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

N=10
phys_dim = 2
delta_t = 0.01
t_max = 20
d=6

H = common_mpo.PXP(N,"open")

#initial state [must be padded to bond dimension - d comes from initial state not algorithm unlike TEBD!!]
A = np.zeros([phys_dim,d,d])
B=np.zeros([phys_dim,d,d])
Vr=np.zeros([phys_dim,d])
Vl=np.zeros([phys_dim,d])

LR = np.ones(d)
LR = LR / np.power(np.vdot(LR,LR),0.5)

A[phys_dim-1] = np.eye(d)
B[0] = np.eye(d)
Vl[0] = LR
Vr[phys_dim-1] = LR

# Neel state (init state)
psi = open_MPS(N)
psi.set_entry(0,Vl,"right")
for n in range(1,N-2,2):
    psi.set_entry(n,A,"both")
for n in range(2,N-1,2):
    psi.set_entry(n,B,"both")
psi.set_entry(N-1,Vr,"left")


from integrators import expiH
from TDVP import TDVP
tdvp = TDVP(H,psi)
# tdvp.run(delta_t,t_max,integrator = expiH.krylov)
# tdvp.run(delta_t,t_max,integrator = expiH.euler)
tdvp.run(delta_t,t_max,integrator = expiH.rungeKutta4)
# tdvp.run(delta_t,t_max,integrator = expiH.exact_exp)

plt.plot(tdvp.t,tdvp.f)
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.xlabel(r"$t$")
plt.title(r"TDVP Time Evolution, $N$="+str(N))
plt.show()

