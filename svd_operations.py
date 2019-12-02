#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as linalg
class svd_node_pair:
    def left(node1,node2,D_cap=None,rescale=False):
        if node1.legs == "both" and node2.legs == "both" : 
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1],shape[2])))

            A,S,Vh = linalg.svd(reshaped_node,full_matrices=False)
            S0 = np.copy(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    A = A[:,0:D_cap]
                    S = S[0:D_cap]
                    Vh = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    A = np.hstack((A,np.zeros((np.size(A,axis=0),dim_padding))))
                    Vh = np.vstack((Vh,np.zeros((dim_padding,np.size(Vh,axis=1)))))

            if rescale is True:
                l0 = np.power(np.dot(S0,S0),0.5)
                l = np.power(np.dot(S,S),0.5)
                S = S * l0/l

            new_left_node = A.reshape((shape[0],shape[1],np.size(S)))
            new_right_node = np.einsum('ij,jk,ukl->uil',np.diag(S),Vh,node2.tensor)
            return new_left_node, new_right_node

        elif node1.legs == "both" and node2.legs == "left":
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1],shape[2])))

            A,S,Vh = linalg.svd(reshaped_node,full_matrices=False)
            S0 = np.copy(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    A = A[:,0:D_cap]
                    S = S[0:D_cap]
                    Vh = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    A = np.hstack((A,np.zeros((np.size(A,axis=0),dim_padding))))
                    Vh = np.vstack((Vh,np.zeros((dim_padding,np.size(Vh,axis=1)))))

            if rescale is True:
                l0 = np.power(np.dot(S0,S0),0.5)
                l = np.power(np.dot(S,S),0.5)
                S = S * l0/l

            A = A.reshape((shape[0],shape[1],np.size(S)))
            M = np.einsum('ij,jk,uk->ui',np.diag(S),Vh,node2.tensor)
            return A,M

        elif node1.legs == "right" and node2.legs == "both":
            A,S,Vh = linalg.svd(node1.tensor,full_matrices=False)
            S0 = np.copy(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    A = A[:,0:D_cap]
                    S = S[0:D_cap]
                    Vh = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    A = np.hstack((A,np.zeros((np.size(A,axis=0),dim_padding))))
                    Vh = np.vstack((Vh,np.zeros((dim_padding,np.size(Vh,axis=1)))))

            if rescale is True:
                l0 = np.power(np.dot(S0,S0),0.5)
                l = np.power(np.dot(S,S),0.5)
                S = S * l0/l

            new_left_node = A
            new_right_node = np.einsum('ij,jk,ukl->uil',np.diag(S),Vh,node2.tensor)
            return new_left_node, new_right_node

    def right(node1,node2,D_cap=None,rescale=False):
        if node1.legs == "both" and node2.legs == "both" : 
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijk->jki',node2.tensor),np.array((shape[1],shape[0]*shape[2])))

            U,S,B = linalg.svd(reshaped_node,full_matrices=False)
            S0 = np.copy(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    U = A[:,0:D_cap]
                    S = S[0:D_cap]
                    B = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    U = np.hstack((U,np.zeros((np.size(U,axis=0),dim_padding))))
                    B = np.vstack((B,np.zeros((dim_padding,np.size(B,axis=1)))))

            if rescale is True:
                l0 = np.power(np.dot(S0,S0),0.5)
                l = np.power(np.dot(S,S),0.5)
                S = S * l0/l

            B = np.reshape(B,np.array((np.size(S),shape[2],shape[0])))
            new_right_node = np.einsum('ijk->kij',B)
            new_left_node = np.einsum('ijk,kl,lu->iju',node1.tensor,U,np.diag(S))
            return new_left_node,new_right_node

        elif node1.legs == "both" and node2.legs == "left":
            shape = np.shape(node2.tensor)
            reshaped_node = node2.tensor.transpose()

            U,S,B = linalg.svd(reshaped_node,full_matrices=False)
            S0 = np.copy(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    U = A[:,0:D_cap]
                    S = S[0:D_cap]
                    B = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    U = np.hstack((U,np.zeros((np.size(U,axis=0),dim_padding))))
                    B = np.vstack((B,np.zeros((dim_padding,np.size(B,axis=1)))))

            if rescale is True:
                l0 = np.power(np.dot(S0,S0),0.5)
                l = np.power(np.dot(S,S),0.5)
                S = S * l0/l

            new_right_node = B.transpose()
            new_left_node = np.einsum('ijk,kl,lu->iju',node1.tensor,U,np.diag(S))
            return new_left_node, new_right_node

        elif node1.legs == "right" and node2.legs == "both":
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijk->jki',node2.tensor),np.array((shape[1],shape[0]*shape[2])))

            U,S,B = linalg.svd(reshaped_node,full_matrices=False)
            S0 = np.copy(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    U = A[:,0:D_cap]
                    S = S[0:D_cap]
                    B = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    U = np.hstack((U,np.zeros((np.size(U,axis=0),dim_padding))))
                    B = np.vstack((B,np.zeros((dim_padding,np.size(B,axis=1)))))

            if rescale is True:
                l0 = np.power(np.dot(S0,S0),0.5)
                l = np.power(np.dot(S,S),0.5)
                S = S * l0/l

            B = np.reshape(B,np.array((np.size(S),shape[2],shape[0])))
            B = np.einsum('ijk->kij',B)
            new_right_node = B

            new_left_node = np.einsum('ij,jk,kl->il',node1.tensor,U,np.diag(S))
            return new_left_node, new_right_node

class svd_norm_node:
    def left(node,D_cap=None,rescale=None):
        if node.legs == "both":
            shape = np.shape(node.tensor)
            M = node.tensor.reshape(np.array((shape[0]*shape[1],shape[2])))
            A,S,Vh = linalg.svd(M,full_matrices=False)
            A = np.reshape(A,np.array((shape[0],shape[1],np.size(S))))
            if rescale is not None:
                A = A * rescale
            return A

        else:
            shape = np.shape(node.tensor)
            M = node.tensor.reshape(np.array((shape[0]*shape[1],1)))
            A,S,Vh = linalg.svd(M,full_matrices=False)
            A = np.reshape(A,np.array((shape[0],shape[1])))
            if rescale is not None:
                A = A * rescale
            return A

    def right(node,D_cap=None,rescale=None):
        if node.legs == "both":
            shape = np.shape(node.tensor)
            M = np.reshape(np.einsum('ijk->jki',node.tensor),np.array((shape[1],shape[0]*shape[2])))
            U,S,B = linalg.svd(M,full_matrices=False)
            B = np.reshape(B,np.array((np.size(S),shape[2],shape[0])))
            B = np.einsum('ijk->kij',B)
            if rescale is not None:
                B = B*rescale
            return B

        else:
            shape = np.shape(node.tensor)
            M = node.tensor.transpose()
            M = M.reshape(np.array((1,shape[0]*shape[1])))
            U,S,B = linalg.svd(M,full_matrices=False)
            B = np.reshape(B,np.array((shape[1],shape[0])))
            if rescale is not None:
                B = B*rescale
            return B.transpose()

class svd_mpo_node_pair:
    def left(node1,node2,D_cap=None,rescale=False):
        if node1.legs == "both" and node2.legs == "both" : 
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1] * shape[2] , shape[3])))

            A,S,Vh = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    A = A[:,0:D_cap]
                    S = S[0:D_cap]
                    Vh = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    A = np.hstack((A,np.zeros((np.size(A,axis=0),dim_padding))))
                    Vh = np.vstack((Vh,np.zeros((dim_padding,np.size(Vh,axis=1)))))

            new_left_node = A.reshape((shape[0],shape[1],shape[2],np.size(S)))
            new_right_node = np.einsum('ij,jk,uvkl->uvil',np.diag(S),Vh,node2.tensor)
            return new_left_node, new_right_node

        elif node1.legs == "both" and node2.legs == "left":
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1] * shape[2] , shape[3])))

            A,S,Vh = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    A = A[:,0:D_cap]
                    S = S[0:D_cap]
                    Vh = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    A = np.hstack((A,np.zeros((np.size(A,axis=0),dim_padding))))
                    Vh = np.vstack((Vh,np.zeros((dim_padding,np.size(Vh,axis=1)))))

            new_left_node = A.reshape((shape[0],shape[1],shape[2],np.size(S)))
            new_right_node = np.einsum('ij,jk,uvk->uvi',np.diag(S),Vh,node2.tensor)
            return new_left_node,new_right_node

        elif node1.legs == "right" and node2.legs == "both":
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1] , shape[2] )))

            A,S,Vh = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    A = A[:,0:D_cap]
                    S = S[0:D_cap]
                    Vh = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    A = np.hstack((A,np.zeros((np.size(A,axis=0),dim_padding))))
                    Vh = np.vstack((Vh,np.zeros((dim_padding,np.size(Vh,axis=1)))))

            new_left_node = A.reshape((shape[0],shape[1],np.size(S)))
            new_right_node = np.einsum('ij,jk,uvkl->uvil',np.diag(S),Vh,node2.tensor)
            return new_left_node, new_right_node

        elif node1.legs == "right" and node2.legs == "left":
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1] , shape[2] )))

            A,S,Vh = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    A = A[:,0:D_cap]
                    S = S[0:D_cap]
                    Vh = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    A = np.hstack((A,np.zeros((np.size(A,axis=0),dim_padding))))
                    Vh = np.vstack((Vh,np.zeros((dim_padding,np.size(Vh,axis=1)))))

            new_left_node = A.reshape((shape[0],shape[1],np.size(S)))
            new_right_node = np.einsum('ij,jk,uvk->uvi',np.diag(S),Vh,node2.tensor)
            return new_left_node, new_right_node

    def right(node1,node2,D_cap=None,rescale=False):
        if node1.legs == "both" and node2.legs == "both" : 
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijkl->klij',node2.tensor),np.array((shape[2],shape[0]*shape[1]*shape[3])))

            U,S,B = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    U = A[:,0:D_cap]
                    S = S[0:D_cap]
                    B = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    U = np.hstack((U,np.zeros((np.size(U,axis=0),dim_padding))))
                    B = np.vstack((B,np.zeros((dim_padding,np.size(B,axis=1)))))

            B = np.reshape(B,np.array((np.size(S),shape[3],shape[0],shape[1])))
            new_right_node = np.einsum('ijkl->klij',B)
            new_left_node = np.einsum('ijkl,lu,uv->ijkv',node1.tensor,U,np.diag(S))
            return new_left_node,new_right_node

        elif node1.legs == "both" and node2.legs == "left":
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijk->kij',node2.tensor),np.array((shape[2],shape[0]*shape[1])))

            U,S,B = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    U = A[:,0:D_cap]
                    S = S[0:D_cap]
                    B = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    U = np.hstack((U,np.zeros((np.size(U,axis=0),dim_padding))))
                    B = np.vstack((B,np.zeros((dim_padding,np.size(B,axis=1)))))

            B = np.reshape(B,np.array((np.size(S),shape[0],shape[1])))
            new_right_node = np.einsum('ijk->jki',B)
            new_left_node = np.einsum('ijkl,lu,uv->ijkv',node1.tensor,U,np.diag(S))
            return new_left_node, new_right_node

        elif node1.legs == "right" and node2.legs == "both":
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijkl->klij',node2.tensor),np.array((shape[2],shape[0]*shape[1]*shape[3])))

            U,S,B = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    U = A[:,0:D_cap]
                    S = S[0:D_cap]
                    B = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    U = np.hstack((U,np.zeros((np.size(U,axis=0),dim_padding))))
                    B = np.vstack((B,np.zeros((dim_padding,np.size(B,axis=1)))))

            B = np.reshape(B,np.array((np.size(S),shape[3],shape[0],shape[1])))
            new_right_node = np.einsum('ijkl->klij',B)
            new_left_node = np.einsum('ijk,ku,uv->ijv',node1.tensor,U,np.diag(S))
            return new_left_node, new_right_node

        elif node1.legs == "right" and node2.legs == "left":
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijk->kij',node2.tensor),np.array((shape[2],shape[0]*shape[1])))

            U,S,B = linalg.svd(reshaped_node,full_matrices=False)
            if rescale is True:
                S = S / np.max(S)

            if D_cap is not None:
                if np.size(S)>D_cap: #truncate
                    U = A[:,0:D_cap]
                    S = S[0:D_cap]
                    B = Vh[0:D_cap,:]
                elif np.size(S)<D_cap: #pad with zeros
                    dim_padding = int(D_cap - np.size(S))
                    S = np.append(S,np.zeros(dim_padding))
                    U = np.hstack((U,np.zeros((np.size(U,axis=0),dim_padding))))
                    B = np.vstack((B,np.zeros((dim_padding,np.size(B,axis=1)))))

            B = np.reshape(B,np.array((np.size(S),shape[0],shape[1])))
            new_right_node = np.einsum('ijk->jki',B)

            new_left_node = np.einsum('ijk,kl,lu->iju',node1.tensor,U,np.diag(S))
            return new_left_node, new_right_node
