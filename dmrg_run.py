#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import dmrg

D=6
N_vals = np.arange(10,300,2)
E0 = np.zeros(np.size(N_vals))
for n in range(0,np.size(N_vals,axis=0)):
    N=N_vals[n]
    H = common_mpo.PXP(N,"open")
    method = dmrg(H,D)
    method.run()
    E0[n] = method.energy
plt.plot(N_vals,np.abs(E0))
plt.show()
