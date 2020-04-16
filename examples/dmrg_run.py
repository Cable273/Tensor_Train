#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import mpo,mps
from rail_objects import *
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *
from collapsed_layers import collapsed_layer

N=1000
D=10
# H = common_mpo.Heis(N,1,"open")
H = common_mpo.PXP(N,"open")
# H = common_mpo.XX(N,"open")

# grow chain to desired length
method = idmrg(H,2,D)
psi_trial= method.run(N)

# #finite sweeping until converged
if method.var > 1e-8:
    print("Finite size sweeps")
    method = dmrg(H,D,psi=psi_trial)
    psi = method.run(N)
