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
tol = 1e-15

from common_trotters import intGate, swapGate

class gate_application_method:
    def factory(gate,psiVidal,D,phys_dim):
        if type(gate) is intGate:
            gate_length = gate.length
            loc = gate.loc
            if gate_length == 2:
                if loc == 0:
                    return size2GateAppliedLeft(gate,psiVidal,D,phys_dim)
                elif loc == psiVidal.length-2:
                    return size2GateAppliedRight(gate,psiVidal,D,phys_dim)
                else:
                    return size2GateAppliedMiddle(gate,psiVidal,D,phys_dim)
            if gate_length == 3:
                if loc == 0:
                    return size3GateAppliedLeft(gate,psiVidal,D,phys_dim)
                elif loc == psiVidal.length-3:
                    return size3GateAppliedRight(gate,psiVidal,D,phys_dim)
                else:
                    return size3GateAppliedMiddle(gate,psiVidal,D,phys_dim)
        elif type(gate) is swapGate:
            loc = gate.loc
            if loc == 0:
                return applySwapGateLeft(gate,psiVidal,D,phys_dim)
            elif loc == psiVidal.length - 2:
                return applySwapGateRight(gate,psiVidal,D,phys_dim)
            else:
                return applySwapGateMiddle(gate,psiVidal,D,phys_dim)

class applySwapGateLeft(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error = 0
    def apply(self,truncate = True):
        M = np.einsum('ia,ab->ib',self.psiVidal.node[self.loc],np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('ib,jbc->ijc',M,self.psiVidal.node[self.loc+1])
        M = np.einsum('ijc,cd->ijd',M,np.diag(self.psiVidal.singulars[self.loc+1]))
        psi_new = np.einsum('ijd->jid',M)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0],dims[1]*dims[2]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        gamma0 = U
        B1 = np.einsum('ijk->jik',Vh.reshape((np.size(S),dims[1],dims[2])))
        inverseSingulars = np.diag(np.power(self.psiVidal.singulars[self.loc+1],-1))
        gamma1 = np.einsum('ijk,kd->ijd',B1,inverseSingulars)

        self.psiVidal.node[self.loc] = gamma0
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.node[self.loc+1] = gamma1

class applySwapGateMiddle(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error = 0
    def apply(self,truncate = True):
        M = np.einsum('ab,ibc->iac',np.diag(self.psiVidal.singulars[self.loc-1]),self.psiVidal.node[self.loc])
        M = np.einsum('iac,cd->iad',M,np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('iad,jde->ijae',M,self.psiVidal.node[self.loc+1])
        M = np.einsum('ijae,ef->ijaf',M,np.diag(self.psiVidal.singulars[self.loc+1]))
        psi_new = np.einsum('ijaf->jaif',M)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        A = U.reshape((dims[0],dims[1],np.size(S)))
        B = np.einsum('ijk->jik',Vh.reshape((np.size(S),dims[2],dims[3])))
        inverseSingularsL = np.diag(np.power(self.psiVidal.singulars[self.loc-1],-1))
        inverseSingularsR = np.diag(np.power(self.psiVidal.singulars[self.loc+1],-1))
        gammaL = np.einsum('ab,ibc->iac',inverseSingularsL,A)
        gammaR = np.einsum('iab,bc->iac',B,inverseSingularsR)

        self.psiVidal.node[self.loc] = gammaL
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.node[self.loc+1] = gammaR

class applySwapGateRight(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error=0
    def apply(self,truncate = True):
        M = np.einsum('ab,ibe->iae',np.diag(self.psiVidal.singulars[self.loc-1]),self.psiVidal.node[self.loc])
        M = np.einsum('iae,ef->iaf',M,np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('iaf,jf->ija',M,self.psiVidal.node[self.loc+1])
        psi_new = np.einsum('ija->jai',M)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1],dims[2]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        A = U.reshape((dims[0],dims[1],np.size(S)))
        gammaR = np.transpose(Vh)
        inverseSingulars = np.diag(np.power(self.psiVidal.singulars[self.loc-1],-1))
        gammaL = np.einsum('ab,ibe->iae',inverseSingulars,A)

        self.psiVidal.node[self.loc] = gammaL
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.node[self.loc+1] = gammaR


class size2GateAppliedLeft(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.gate = gate.tensor
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error = 0
    def apply(self,truncate = True):
        M = np.einsum('ia,ab->ib',self.psiVidal.node[self.loc],np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('ib,jbc->ijc',M,self.psiVidal.node[self.loc+1])
        M = np.einsum('ijc,cd->ijd',M,np.diag(self.psiVidal.singulars[self.loc+1]))
        psi_new = np.einsum('ijd,ijab->abd',M,self.gate)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0],dims[1]*dims[2]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        gamma0 = U
        B1 = np.einsum('ijk->jik',Vh.reshape((np.size(S),dims[1],dims[2])))
        inverseSingulars = np.diag(np.power(self.psiVidal.singulars[self.loc+1],-1))
        gamma1 = np.einsum('ijk,kd->ijd',B1,inverseSingulars)

        self.psiVidal.node[self.loc] = gamma0
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.node[self.loc+1] = gamma1

class size2GateAppliedMiddle(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.gate = gate.tensor
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error=0
    def apply(self,truncate = True):
        M = np.einsum('ab,ibc->iac',np.diag(self.psiVidal.singulars[self.loc-1]),self.psiVidal.node[self.loc])
        M = np.einsum('iac,cd->iad',M,np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('iad,jde->ijae',M,self.psiVidal.node[self.loc+1])
        M = np.einsum('ijae,ef->ijaf',M,np.diag(self.psiVidal.singulars[self.loc+1]))
        psi_new = np.einsum('abcd,abef->ecfd',M,self.gate)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        A = U.reshape((dims[0],dims[1],np.size(S)))
        B = np.einsum('ijk->jik',Vh.reshape((np.size(S),dims[2],dims[3])))
        inverseSingularsL = np.diag(np.power(self.psiVidal.singulars[self.loc-1],-1))
        inverseSingularsR = np.diag(np.power(self.psiVidal.singulars[self.loc+1],-1))
        gammaL = np.einsum('ab,ibc->iac',inverseSingularsL,A)
        gammaR = np.einsum('iab,bc->iac',B,inverseSingularsR)

        self.psiVidal.node[self.loc] = gammaL
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.node[self.loc+1] = gammaR

class size2GateAppliedRight(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.gate = gate.tensor
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error=0
    def apply(self,truncate = True):
        M = np.einsum('ab,ibe->iae',np.diag(self.psiVidal.singulars[self.loc-1]),self.psiVidal.node[self.loc])
        M = np.einsum('iae,ef->iaf',M,np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('iaf,jf->ija',M,self.psiVidal.node[self.loc+1])
        psi_new = np.einsum('ija,ijcd->cad',M,self.gate)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1],dims[2]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        A = U.reshape((dims[0],dims[1],np.size(S)))
        gammaR = np.transpose(Vh)
        inverseSingulars = np.diag(np.power(self.psiVidal.singulars[self.loc-1],-1))
        gammaL = np.einsum('ab,ibe->iae',inverseSingulars,A)

        self.psiVidal.node[self.loc] = gammaL
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.node[self.loc+1] = gammaR

class size3GateAppliedLeft(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.gate = gate.tensor
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error = 0
    def apply(self,truncate = True):
        M = np.einsum('ib,bc->ic',self.psiVidal.node[self.loc],np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('ic,jcd->ijd',M,self.psiVidal.node[self.loc+1])
        M = np.einsum('ijd,de->ije',M,np.diag(self.psiVidal.singulars[self.loc+1]))
        M = np.einsum('ije,kef->ijkf',M,self.psiVidal.node[self.loc+2])
        M = np.einsum('ijkf,fg->ijkg',M,np.diag(self.psiVidal.singulars[self.loc+2]))
        psi_new = np.einsum('ijkg,ijkabc->abcg',M,self.gate)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0],dims[1]*dims[2]*dims[3]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        gamma0 = U
        M = np.dot(np.diag(S),Vh)
        M = M.reshape((np.size(S),dims[1],dims[2],dims[3]))
        M = np.einsum('abcd->bacd',M)
        dims2 = np.shape(M)
        M = M.reshape((dims2[0]*dims2[1],dims2[2]*dims2[3]))

        U,S2,Vh = np.linalg.svd(M,full_matrices = False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S2[self.D:]**2)
            S2 = S2[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S2,axis=0)):
            if np.abs(S2[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S2[cut:]**2)
            S2 = S2[:cut]
            Vh = Vh[:cut,:]
        A1 = U.reshape((dims2[0],dims2[1],np.size(S2)))
        B1 = np.einsum('ijk->jik',Vh.reshape((np.size(S2),dims2[2],dims2[3])))

        inverseSingulars0 = np.diag(np.power(S,-1))
        gamma1 = np.einsum('ab,ibc->iac',inverseSingulars0,A1)

        inverseSingulars2 = np.diag(np.power(self.psiVidal.singulars[self.loc+2],-1))
        gamma2 = np.einsum('iab,bc->iac',B1,inverseSingulars2)

        self.psiVidal.node[self.loc] = gamma0
        self.psiVidal.node[self.loc+1] = gamma1
        self.psiVidal.node[self.loc+2] = gamma2
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.singulars[self.loc+1] = S2 / np.power(np.vdot(S2,S2),0.5)

class size3GateAppliedMiddle(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.gate = gate.tensor
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error = 0
    def apply(self,truncate = True):
        M = np.einsum('ab,iac->ibc',np.diag(self.psiVidal.singulars[self.loc-1]),self.psiVidal.node[self.loc])
        M = np.einsum('ibc,cd->ibd',M,np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('ibd,jde->ijbe',M,self.psiVidal.node[self.loc+1])
        M = np.einsum('ijbe,ef->ijbf',M,np.diag(self.psiVidal.singulars[self.loc+1]))
        M = np.einsum('ijbf,kfg->ijkbg',M,self.psiVidal.node[self.loc+2])
        M = np.einsum('ijkbg,gh->ijkbh',M,np.diag(self.psiVidal.singulars[self.loc+2]))
        psi_new = np.einsum('ijkbg,ijkunm->ubnmg',M,self.gate)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1],dims[2]*dims[3]*dims[4]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        A0 = U.reshape((dims[0],dims[1],np.size(S)))
        M = np.dot(np.diag(S),Vh)
        M = M.reshape((np.size(S),dims[2],dims[3],dims[4]))
        M = np.einsum('abcd->bacd',M)
        dims2 = np.shape(M)
        M = M.reshape((dims2[0]*dims2[1],dims2[2]*dims2[3]))

        U,S2,Vh = np.linalg.svd(M,full_matrices = False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S2[self.D:]**2)
            S2 = S2[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S2,axis=0)):
            if np.abs(S2[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S2[cut:]**2)
            S2 = S2[:cut]
            Vh = Vh[:cut,:]
        A1 = U.reshape((dims2[0],dims2[1],np.size(S2)))
        B1 = np.einsum('ijk->jik',Vh.reshape((np.size(S2),dims2[2],dims2[3])))

        inverseSingulars0 = np.diag(np.power(self.psiVidal.singulars[self.loc-1],-1))
        gamma0 = np.einsum('ab,ibc->iac',inverseSingulars0,A0)

        inverseSingulars1 = np.diag(np.power(S,-1))
        gamma1 = np.einsum('ab,ibc->iac',inverseSingulars1,A1)

        inverseSingulars2 = np.diag(np.power(self.psiVidal.singulars[self.loc+2],-1))
        gamma2 = np.einsum('iab,bc->iac',B1,inverseSingulars2)

        self.psiVidal.node[self.loc] = gamma0
        self.psiVidal.node[self.loc+1] = gamma1
        self.psiVidal.node[self.loc+2] = gamma2
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.singulars[self.loc+1] = S2 / np.power(np.vdot(S2,S2),0.5)

class size3GateAppliedRight(gate_application_method):
    def __init__(self,gate,psiVidal,D,phys_dim):
        self.gate = gate.tensor
        self.psiVidal = psiVidal
        self.loc = gate.loc
        self.D = D
        self.phys_dim = phys_dim
        self.error = 0
    def apply(self,truncate = True):
        M = np.einsum('ab,ibc->iac',np.diag(self.psiVidal.singulars[self.loc-1]),self.psiVidal.node[self.loc])
        M = np.einsum('iac,cd->iad',M,np.diag(self.psiVidal.singulars[self.loc]))
        M = np.einsum('iad,jde->ijae',M,self.psiVidal.node[self.loc+1])
        M = np.einsum('ijae,ef->ijaf',M,np.diag(self.psiVidal.singulars[self.loc+1]))
        M = np.einsum('ijaf,kf->ijka',M,self.psiVidal.node[self.loc+2])
        psi_new = np.einsum('ijka,ijkunm->uanm',M,self.gate)

        dims = np.shape(psi_new)
        psi_new = psi_new.reshape((dims[0]*dims[1],dims[2]*dims[3]))
        U,S,Vh = np.linalg.svd(psi_new,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S[self.D:]**2)
            S = S[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S,axis=0)):
            if np.abs(S[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S[cut:]**2)
            S = S[:cut]
            Vh = Vh[:cut,:]

        A0 = U.reshape((dims[0],dims[1],np.size(S)))
        M = np.dot(np.diag(S),Vh)
        M = M.reshape((np.size(S),dims[2],dims[3]))
        M = np.einsum('ijk->jik',M)
        dims2 = np.shape(M)
        M = M.reshape((dims2[0]*dims2[1],dims2[2]))

        U,S2,Vh = np.linalg.svd(M,full_matrices=False)
        if truncate is True:
            U = U[:,:self.D]
            self.error += np.sum(S2[self.D:]**2)
            S2 = S2[:self.D]
            Vh = Vh[:self.D,:]
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for n in range(0,np.size(S2,axis=0)):
            if np.abs(S2[n])<tol:
                cut = n
                break
        if cut is not None:
            U = U[:,:cut]
            self.error += np.sum(S2[cut:]**2)
            S2 = S2[:cut]
            Vh = Vh[:cut,:]
        A1 = U.reshape((dims2[0],dims2[1],np.size(S2)))

        inverseSingulars0 = np.diag(np.power(self.psiVidal.singulars[self.loc-1],-1))
        gamma0 = np.einsum('ab,ibc->iac',inverseSingulars0,A0)

        inverseSingulars0 = np.diag(np.power(S,-1))
        gamma1 = np.einsum('ab,ibc->iac',inverseSingulars0,A1)
        gamma2 = np.transpose(Vh)


        self.psiVidal.node[self.loc] = gamma0
        self.psiVidal.node[self.loc+1] = gamma1
        self.psiVidal.node[self.loc+2] = gamma2
        self.psiVidal.singulars[self.loc] = S / np.power(np.vdot(S,S),0.5)
        self.psiVidal.singulars[self.loc+1] = S2 / np.power(np.vdot(S2,S2),0.5)
