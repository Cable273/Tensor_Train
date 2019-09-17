#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *

N=12
D=2
d=2
psi = mps.random(N,d,D,boundary="open")
psi.left_normalize(norm=True,norm_val = 1)

