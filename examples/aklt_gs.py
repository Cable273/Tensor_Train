#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *

A = np.zeros([3,2,2])
A[0] = np.array([[0,np.power(2/3,0.5)],[0,0]])
A[1] = np.array([[-1/np.power(3,0.5),0],[0,1/np.power(3,0.5)]])
A[2] = np.array([[0,0],[-np.power(2/3,0.5),0]])

psi = mps.uniform(1000,A)
print(psi.dot(psi))
