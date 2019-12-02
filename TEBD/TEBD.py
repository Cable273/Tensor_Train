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
from trotter_gate_application import *

class TEBD:
    def __init__(self,trotter_gates,psi_init,D):
        self.trotter_gates = trotter_gates
        self.psi_init = psi_init
        self.D = D
        self.phys_dim = np.shape(psi_init.node[0].tensor)[0]
        self.psi = copy.deepcopy(psi_init)

    def run(self,delta_t,t_max):
        print("Evolving with TEBD")
        self.t=np.arange(0,t_max+delta_t,delta_t)
        self.f=np.zeros(np.size(self.t))
        self.f[0] = 1
        self.error = np.zeros(np.size(self.t))
        pbar=ProgressBar()
        for n in pbar(range(1,np.size(self.t,axis=0))):
            psi_exact = copy.deepcopy(self.psi)
            for row in range(0,len(self.trotter_gates)):
                loc = list(self.trotter_gates[row].keys())
                for m in range(0,np.size(loc,axis=0)):
                    applier = gate_application_method.factory(self.trotter_gates[row][loc[m]],self.psi,loc[m],self.D,self.phys_dim)
                    # applier_exact = gate_application_method.factory(self.trotter_gates[row][loc[m]],psi_exact,loc[m],self.D,self.phys_dim)
                    applier.apply()

            # error = np.abs(self.psi.dot(self.psi)+psi_exact.dot(psi_exact)-self.psi.dot(psi_exact)-psi_exact.dot(self.psi))
            # self.error[n] = self.error[n-1] + error
            self.f[n] = np.abs(self.psi_init.dot(self.psi))**2

    def plot_fidelity(self):
        plt.plot(self.t,self.f)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
