import numpy as np
from rail_objects import *
from progressbar import ProgressBar
from svd_operations import *
from combine_rail_objects import *
from Tensor_Train import *
import copy
trunc_cutoff = 1e-8

class mps:
    def uniform(length,A,V=None,W=None):
        if V is None and W is None: return periodic_MPS(length,A)
        else: return open_MPS(length,A,V,W)

    def random(length,on_site_dim,bond_dim,boundary=None):
        A = np.random.uniform(-1,1,np.array((on_site_dim,bond_dim,bond_dim)))
        V = np.random.uniform(-1,1,np.array((on_site_dim,bond_dim)))
        W = np.random.uniform(-1,1,np.array((on_site_dim,bond_dim)))
        if boundary == "periodic": return periodic_MPS(length,A)
        elif boundary == "open": return open_MPS(length,A,V,W)
        else: #choose open/periodic randomly
            bc = np.random.choice(['periodic','open'])
            if bc == "periodic":
                return periodic_MPS(length,A)
            elif bc == "open":
                return open_MPS(length,A,V,W)

    def set_entry(self,site,tensor,legs):
        self.node[site].tensor = tensor
        self.node[site].legs = legs

    def dot(self,MPS2):
        network = rail_network(self.conj(),MPS2)
        network.contract()
        return network.contraction
    
    def norm(self):
        norm = np.power(np.abs(self.dot(self)),0.5)
        for n in range(0,self.length):
            self.node[n].tensor = self.node[n].tensor/norm
    
    def left_normalize(self,norm=False,verbose=False,D=None,rescale=None):
        norm_val = np.abs(self.dot(self))
        if verbose is True:
            print("Left normalizing MPS")
            pbar=ProgressBar()
            for n in pbar(range(0,self.length-1)):
                self.node[n].tensor,self.node[n+1].tensor = svd_node_pair.left(self.node[n],self.node[n+1],D_cap=D,rescale=rescale)
        else:
            for n in range(0,self.length-1):
                self.node[n].tensor,self.node[n+1].tensor = svd_node_pair.left(self.node[n],self.node[n+1],D_cap=D,rescale=rescale)
        if norm is True:
            self.node[self.length-1].tensor = self.node[self.length-1].tensor / np.power(norm_val,0.5)

    def right_normalize(self,norm=False,verbose=False,D=None,rescale=None):
        pbar=ProgressBar()
        norm_val = np.abs(self.dot(self))
        if verbose is True:
            print("Right normalizing MPS")
            for n in pbar(range(self.length-1,0,-1)):
                self.node[n-1].tensor,self.node[n].tensor = svd_node_pair.right(self.node[n-1],self.node[n],D_cap=D,rescale=rescale)
        else:
            for n in range(self.length-1,0,-1):
                self.node[n-1].tensor,self.node[n].tensor = svd_node_pair.right(self.node[n-1],self.node[n],D_cap=D,rescale=rescale)
        if norm is True:
            self.node[0].tensor = self.node[0].tensor / np.power(norm_val,0.5)

    def mixed_normalize(self,site):
        print("Mixed normalizing MPS")
        print("Left norm:")
        pbar=ProgressBar()
        for n in pbar(range(0,self.length-1)):
            self.node[n].tensor,self.node[n+1].tensor = svd_node_pair.left(self.node[n],self.node[n+1])
        # self.node[self.length-1].tensor = svd_norm_node.left(self.node[self.length-1])
        print("Right norm:")
        pbar=ProgressBar()
        for n in pbar(range(self.length-1,site,-1)):
            self.node[n-1].tensor,self.node[n].tensor = svd_node_pair.right(self.node[n-1],self.node[n])
        # self.node[site].tensor = svd_norm_node(self.node[site])

    def conj(self):
        psi_conj = copy.deepcopy(self)
        for n in range(0,self.length-1):
            if psi_conj.node is not None:
                psi_conj.node[n].tensor = np.conj(psi_conj.node[n].tensor)
        return psi_conj

class mpo:
    def uniform(length,Q,V=None,W=None):
        if V is None and W is None: return periodic_MPO(length,Q)
        else: return open_MPO(length,Q,V,W)
            

    def random(length,on_site_dim,bond_dim,boundary=None):
        O = np.random.uniform(-1,1,np.array((on_site_dim,on_site_dim,bond_dim,bond_dim)))
        V = np.random.uniform(-1,1,np.array((on_site_dim,on_site_dim,bond_dim)))
        W = np.random.uniform(-1,1,np.array((on_site_dim,on_site_dim,bond_dim)))
        if boundary == "periodic": return periodic_MPO(length,O)
        elif boundary == "open": return open_MPO(length,O,V,W)
        else: #choose open/periodic randomly
            bc = np.random.choice(['periodic','open'])
            if bc == "periodic":
                return periodic_MPO(length,O)
            elif bc == "open":
                return open_MPO(length,O,V,W)

    def set_entry(self,site,tensor,legs):
        self.node[site].tensor = tensor
        self.node[site].legs = legs

    def exp(self,psi1,psi2=None):
        if psi2 is None:
            network = rail_network(psi1.conj(),psi1,self)
        else:
            network = rail_network(psi1.conj(),psi2,self)
        network.contract()
        return network.contraction

    def dot(self,psi):
        if type(psi) is periodic_MPS or type(psi) is open_MPS:
            network = rail_network(psi,Q=self)
            pbar=ProgressBar()
            for n in range(0,psi.length):
                psi.node[n] = collapsed_MPO_layer.factory(layer(network,n))
            return psi
        elif type(psi) is periodic_MPO or type(psi) is open_MPO:
            new_mpo = copy.deepcopy(psi)
            for n in range(0,self.length):
                new_mpo.node[n] = combine_mpo_nodes_vertical.factory(self.node[n],psi.node[n])
            return new_mpo

    def left_normalize(self,norm=False,verbose=False,D=None,rescale=None):
        if verbose is True:
            print("Left normalizing MPS")
            pbar=ProgressBar()
            for n in pbar(range(0,self.length-1)):
                self.node[n].tensor,self.node[n+1].tensor = svd_mpo_node_pair.left(self.node[n],self.node[n+1],D_cap=D,rescale=rescale)
        else:
            for n in range(0,self.length-1):
                self.node[n].tensor,self.node[n+1].tensor = svd_mpo_node_pair.left(self.node[n],self.node[n+1],D_cap=D,rescale=rescale)
        if norm is True:
            new_norm = np.abs(self.dot(self))
            self.node[self.length-1].tensor = self.node[self.length-1].tensor * np.power(norm_val/new_norm,0.5)

    def right_normalize(self,norm=False,verbose=False,D=None,rescale=None):
        pbar=ProgressBar()
        if verbose is True:
            print("Right normalizing MPS")
            for n in pbar(range(self.length-1,0,-1)):
                self.node[n-1].tensor,self.node[n].tensor = svd_mpo_node_pair.right(self.node[n-1],self.node[n],D_cap=D,rescale=rescale)
        else:
            for n in range(self.length-1,0,-1):
                self.node[n-1].tensor,self.node[n].tensor = svd_mpo_node_pair.right(self.node[n-1],self.node[n],D_cap=D,rescale=rescale)
        if norm is True:
            new_norm = np.abs(self.dot(self))
            self.node[0].tensor = self.node[0].tensor * np.power(norm_val/new_norm,0.5)

    def herm_conj(self):
        Q_conj = copy.deepcopy(self)
        for n in range(0,self.length-1):
            if Q_conj.node is not None:
                Q_conj.node[n].tensor = np.conj(np.einsum('ij...->ji...',Q_conj.node[n].tensor))
        return Q_conj

class periodic_MPS(mps):
    def __init__(self,length,A=None):
        self.length = length
        self.node = dict()
        for n in range(0,self.length):
            self.node[n] = rail_node()
        if A is not None:
            for n in range(0,self.length):
                self.node[n] = rail_node(A,legs="both")
            
    #form direct sum of each tensor
    def __add__(self,MPS2):
        new_MPS = periodic_MPS(self.length)
        for n in range(0,new_MPS.length):
            shape1 = np.shape(self.node[n].tensor)
            shape2 = np.shape(MPS2.node[n].tensor)
            M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1],shape1[2]+shape2[2],)))
            for physical_index in range(0,shape1[0]):
                M[physical_index][0:shape1[1],0:shape1[2]] = self.node[n].tensor[physical_index]
                M[physical_index][shape1[1]:,shape1[2]:] = MPS2.node[n].tensor[physical_index]
            new_MPS.node[n] = rail_node(M,"both")
        return new_MPS
    
class open_MPS(mps):
    def __init__(self,length,A=None,V=None,W=None):
        self.length = length
        self.node = dict()
        for n in range(0,self.length):
            self.node[n] = rail_node()
        if A is not None:
            self.node[0] = rail_node(V,legs="right")
            for n in range(1,self.length-1):
                self.node[n] = rail_node(A,legs="both")
            self.node[self.length-1] = rail_node(W,legs="left")

    #form direct sum of each tensor
    def __add__(self,MPS2):
        new_MPS = open_MPS(self.length)
        #left edge
        shape1 = np.shape(self.node[0].tensor)
        shape2 = np.shape(MPS2.node[0].tensor)
        M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1])))
        for physical_index in range(0,shape1[0]):
            M[physical_index][0:shape1[1]] = self.node[0].tensor[physical_index]
            M[physical_index][shape1[1]:] = MPS2.node[0].tensor[physical_index]
        new_MPS.node[0] = rail_node(M,"right")

        #middle
        for n in range(1,new_MPS.length-1):
            shape1 = np.shape(self.node[n].tensor)
            shape2 = np.shape(MPS2.node[n].tensor)
            M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1],shape1[2]+shape2[2],)))
            for physical_index in range(0,shape1[0]):
                M[physical_index][0:shape1[1],0:shape1[2]] = self.node[n].tensor[physical_index]
                M[physical_index][shape1[1]:,shape1[2]:] = MPS2.node[n].tensor[physical_index]
            new_MPS.node[n] = rail_node(M,"both")

        #right edge
        shape1 = np.shape(self.node[self.length-1].tensor)
        shape2 = np.shape(MPS2.node[MPS2.length-1].tensor)
        M = np.zeros(np.array((shape1[0],shape1[1]+shape2[1])))
        for physical_index in range(0,shape1[0]):
            M[physical_index][0:shape1[1]] = self.node[self.length-1].tensor[physical_index]
            M[physical_index][shape1[1]:] = MPS2.node[MPS2.length-1].tensor[physical_index]
        new_MPS.node[self.length-1] = rail_node(M,"left")
        return new_MPS


class periodic_MPO(mpo):
    def __init__(self,length,Q=None):
        self.length = length
        self.node = dict()
        for n in range(0,self.length):
            self.node[n] = rail_node()
        if Q is not None:
            #reshape so legs corrspond to: up down left right
            Q=np.einsum('ijkl->klij',Q)
            for n in range(0,self.length):
                self.node[n] = rail_node(Q,legs="both")

class open_MPO(mpo):
    def __init__(self,length,Q=None,V=None,W=None):
        self.length = length
        self.node = dict()
        for n in range(0,self.length):
            self.node[n] = rail_node()
        if Q is not None:
            #reshape so legs corrspond to: up down left right
            Q=np.einsum('ijkl->klij',Q)
            V=np.einsum('ijk->jki',V)
            W=np.einsum('ijk->jki',W)

            self.node[0] = rail_node(V,legs="right")
            for n in range(1,self.length-1):
                self.node[n] = rail_node(Q,legs="both")
            self.node[self.length-1] = rail_node(W,legs="left")

class vidalOpenMPS:
    def __init__(self,initMps):
        self.initMps = initMps
        self.length = self.initMps.length

        #site tensors + bond singular values 
        self.genVidalForm()

    def genVidalForm(self):
        #first site + last sites do sep
        self.singulars = dict()
        self.node = dict()
        M = self.initMps.node[0].tensor

        U,S,Vh = np.linalg.svd(M,full_matrices=False)
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for m in range(0,np.size(S,axis=0)):
            if np.abs(S[m])<trunc_cutoff:
                cut = m
                break
        if cut is not None:
            U = U[:,:cut]
            S = S[:cut]
            Vh = Vh[:cut,:]
        self.node[0] = U #left gamma matrix
        self.singulars[0] = S
        R = np.dot(np.diag(S),Vh)

        #loop through centre states
        for n in range(1,self.length-1):
            RM = np.einsum('ab,ibc->iac',R,self.initMps.node[n].tensor)
            dims = np.shape(RM)
            RM = RM.reshape((dims[0]*dims[1],dims[2]))
            U,S,Vh = np.linalg.svd(RM,full_matrices=False)
            #check for singulars < 1e-7, truncate further, for numerical stability
            cut = None
            for m in range(0,np.size(S,axis=0)):
                if np.abs(S[m])<trunc_cutoff:
                    cut = m
                    break
            if cut is not None:
                U = U[:,:cut]
                S = S[:cut]
                Vh = Vh[:cut,:]
            A = U.reshape((dims[0],dims[1],np.size(S)))
            #inverse of singulars dangerous numerical stability
            prevSingularM = np.diag(np.power(self.singulars[n-1],-1)) 
            gamma = np.einsum('ab,ibc->iac',prevSingularM,A)
            self.node[n] = gamma
            self.singulars[n] = S
            R = np.dot(np.diag(S),Vh)

        RM = np.einsum('ab,ib->ia',R,self.initMps.node[self.length-1].tensor)
        A = np.einsum('ab,ibc->iac',np.diag(self.singulars[self.length-3]),self.node[self.length-2])
        M = np.einsum('iac,jc->iaj',A,RM)
        dims = np.shape(M)
        M = M.reshape((dims[0]*dims[1],dims[2]))
        U,S,Vh = np.linalg.svd(M,full_matrices=False)
        #check for singulars < 1e-7, truncate further, for numerical stability
        cut = None
        for m in range(0,np.size(S,axis=0)):
            if np.abs(S[m])<trunc_cutoff:
                cut = m
                break
        if cut is not None:
            U = U[:,:cut]
            S = S[:cut]
            Vh = Vh[:cut,:]
        A = U.reshape([dims[0],dims[1],np.size(S)])
        prevSingularM = np.diag(np.power(self.singulars[self.length-3],-1)) 
        gamma_Lm2 = np.einsum('ab,ibc->iac',prevSingularM,A)
        gamma_Lm1 = np.transpose(Vh)

        self.node[self.length-2] = gamma_Lm2
        self.node[self.length-1] = gamma_Lm1
        self.singulars[self.length-2] = S

        #alt
        # RM = np.einsum('ab,ib->ia',R,self.initMps.node[self.length-1].tensor)
        # prevSingularM = np.diag(np.power(self.singulars[self.length-2],-1)) 
        # gamma = np.einsum('ab,ib->ia',prevSingularM,RM)
        # self.node[self.length-1] = gamma

    def vdot(self,B):
        #initial left side
        L = np.einsum('ia,ib->ab',self.node[0],np.conj(B.node[0]))
        L = np.einsum('ab,ac->cb',L,np.diag(self.singulars[0]))
        L = np.einsum('cb,bd->cd',L,np.conj(np.diag(B.singulars[0])))

        for n in range(1,self.length-1):
            L = np.einsum('ab,iac->icb',L,self.node[n])
            L = np.einsum('icb,ibd->cd',L,np.conj(B.node[n]))
            L = np.einsum('cd,ce->ed',L,np.diag(self.singulars[n]))
            L = np.einsum('ed,df->ef',L,np.conj(np.diag(B.singulars[n])))
        L = np.einsum('ab,ia->ib',L,self.node[self.length-1])
        scalar = np.einsum('ib,ib',L,np.conj(B.node[self.length-1]))
        return scalar

    def exp(self,mpo_object):
        L = np.einsum('ia,ijb->jab',self.node[0],mpo_object.node[0].tensor)
        L = np.einsum('jab,jc->abc',L,np.conj(self.node[0]))
        L = np.einsum('abc,au->ubc',L,np.diag(self.singulars[0]))
        L = np.einsum('ubc,cv->ubv',L,np.diag(self.singulars[0]))

        for n in range(1,self.length-1):
            M = np.einsum('iau,ijbn->jaubn',self.node[n],mpo_object.node[n].tensor)
            M = np.einsum('abc,jaubn->jcun',L,M)
            L = np.einsum('jcun,jcm->unm',M,np.conj(self.node[n]))
            L = np.einsum('abc,au->ubc',L,np.diag(self.singulars[n]))
            L = np.einsum('ubc,cv->ubv',L,np.diag(self.singulars[n]))

        M = np.einsum('ia,ijb->jab',self.node[self.length-1],mpo_object.node[self.length-1].tensor)
        M = np.einsum('abc,jab->jc',L,M)
        scalar = np.einsum('jc,jc',M,np.conj(self.node[self.length-1]))
        return scalar

            
