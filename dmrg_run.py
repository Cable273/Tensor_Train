#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import dmrg
D=2
N=1000
H = common_mpo.PXP(N,"open")
method = dmrg(H,D)
method.run()
