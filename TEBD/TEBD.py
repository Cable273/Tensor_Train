#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
from MPS import *
from rail_objects import *
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *
from progressbar import ProgressBar
import copy
from trotter_gate_applicationVidal import *
from copy import deepcopy
from rw_functions import save_obj,load_obj

class TEBD:
    def __init__(self,trotter_gates,psi_init,D):
        self.trotter_gates = trotter_gates
        self.psi_init = psi_init
        self.D = D
        self.phys_dim = np.shape(psi_init.node[0].tensor)[0]
        self.psi = copy.deepcopy(psi_init)

        #find vidal form of psi (open mps only)
        self.psiVidal = vidalOpenMPS(self.psi)
        self.psiVidalInit = deepcopy(self.psiVidal)

    def run(self,delta_t,t_max,savEvolvedMPS=False,calcFid=False):
        print("Evolving with TEBD")
        self.t=np.arange(0,t_max+delta_t,delta_t)

        if calcFid is True:
            self.f=np.zeros(np.size(self.t))
            self.f[0] = 1
        if savEvolvedMPS is True:
            self.evolvedMPS = dict()
            self.evolvedMPS[0] = self.psiVidalInit

        self.error = np.zeros(np.size(self.t))
        self.entropy = np.zeros(np.size(self.t))

        if self.psiVidal.length % 2 == 0:
            bipartiteCut = int(self.psiVidal.length/2-1)
        else:
            bipartiteCut = int((self.psiVidal.length-1)/2-1)

        S = self.psiVidal.singulars[bipartiteCut]
        self.entropy[0] = -np.sum(S**2*np.log(S**2))

        pbar=ProgressBar()
        for n in pbar(range(1,np.size(self.t,axis=0))):
            error = 0
            for row in range(0,len(self.trotter_gates)):
                keys = list(self.trotter_gates[row].keys())
                for m in range(0,np.size(keys,axis=0)):
                    applier = gate_application_method.factory(self.trotter_gates[row][keys[m]],self.psiVidal,self.D,self.phys_dim)
                    applier.apply()
                    error += applier.error

            #update entropy from middle singular vals
            if self.psiVidal.length % 2 == 0:
                bipartiteCut = int(self.psiVidal.length/2-1)
            else:
                bipartiteCut = int((self.psiVidal.length-1)/2-1)
            S = self.psiVidal.singulars[bipartiteCut]
            self.entropy[n] = -np.sum(S**2*np.log(S**2))

            #update truncation error + fidelity
            self.error[n] = self.error[n-1] + error

            if savEvolvedMPS is True:
                self.evolvedMPS[n] = copy.deepcopy(self.psiVidal)
            if calcFid is True:
                self.f[n] = np.abs(self.psiVidal.vdot(self.psiVidalInit))**2

    def plot_fidelity(self):
        plt.plot(self.t,self.f)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")

    def eval_fidelity(self):
        print("Fidelity: Contracting evolved states")
        self.f = np.zeros(np.size(self.t))
        pbar=ProgressBar()
        for n in pbar(range(0,np.size(self.f,axis=0))):
            self.f[n] = np.abs(self.evolvedMPS[n].vdot(self.psiVidalInit))**2

    def eval_exp(self,mpo_object):
        print("Expectation: Contracting MPO with evolved states")
        exp = np.zeros(np.size(self.t),dtype=complex)
        pbar=ProgressBar()
        for n in pbar(range(0,np.size(self.t,axis=0))):
            exp[n]  = self.evolvedMPS[n].exp(mpo_object)
        return exp
            
            
