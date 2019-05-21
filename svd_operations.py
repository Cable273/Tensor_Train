#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as linalg
class svd_node_pair:
    def left(node1,node2,D_cap=None):
        if node1.legs == "both" and node2.legs == "both" : 
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1],shape[2])))

            A,S,Vh = linalg.svd(reshaped_node,full_matrices=False)
            if D_cap is not None:
                A = A[:,0:D_cap]
                S = S[0:D_cap]
                Vh = Vh[0:D_cap,:]
            #rescale S
            S = S / np.max(S)
            new_left_node = A.reshape((shape[0],shape[1],np.size(S)))
            new_right_node = np.einsum('ij,jk,ukl->uil',np.diag(S),Vh,node2.tensor)
            return new_left_node, new_right_node

        elif node1.legs == "both" and node2.legs == "left":
            shape = np.shape(node1.tensor)
            reshaped_node = node1.tensor.reshape(np.array((shape[0] * shape[1],shape[2])))

            A,S,Vh = np.linalg.svd(reshaped_node,full_matrices=False)
            if D_cap is not None:
                A = A[:,0:D_cap]
                S = S[0:D_cap]
                Vh = Vh[0:D_cap,:]
            S = S / np.max(S)
            A = A.reshape((shape[0],shape[1],np.size(S)))
            M = np.einsum('ij,jk,uk->ui',np.diag(S),Vh,node2.tensor)
            return A,M

        elif node1.legs == "right" and node2.legs == "both":
            A,S,Vh = np.linalg.svd(node1.tensor,full_matrices=False)
            if D_cap is not None:
                A = A[:,0:D_cap]
                S = S[0:D_cap]
                Vh = Vh[0:D_cap,:]
            S = S / np.max(S)
            new_left_node = A
            new_right_node = np.einsum('ij,jk,ukl->uil',np.diag(S),Vh,node2.tensor)
            return new_left_node, new_right_node

    def right(node1,node2,D_cap=None):
        if node1.legs == "both" and node2.legs == "both" : 
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijk->jki',node2.tensor),np.array((shape[1],shape[0]*shape[2])))

            U,S,B = np.linalg.svd(reshaped_node,full_matrices=False)
            S = S / np.max(S)
            if D_cap is not None:
                U = U[:,0:D_cap]
                S = S[0:D_cap]
                B = B[0:D_cap,:]
            B = np.reshape(B,np.array((np.size(S),shape[2],shape[0])))
            new_right_node = np.einsum('ijk->kij',B)
            new_left_node = np.einsum('ijk,kl,lu->iju',node1.tensor,U,np.diag(S))
            return new_left_node,new_right_node

        elif node1.legs == "both" and node2.legs == "left":
            shape = np.shape(node2.tensor)
            reshaped_node = node2.tensor.transpose()

            U,S,B = np.linalg.svd(reshaped_node,full_matrices=False)
            S = S / np.max(S)
            if D_cap is not None:
                U = U[:,0:D_cap]
                S = S[0:D_cap]
                B = B[0:D_cap,:]
            new_right_node = B.transpose()
            new_left_node = np.einsum('ijk,kl,lu->iju',node1.tensor,U,np.diag(S))
            return new_left_node, new_right_node

        elif node1.legs == "right" and node2.legs == "both":
            shape = np.shape(node2.tensor)
            reshaped_node = np.reshape(np.einsum('ijk->jki',node2.tensor),np.array((shape[1],shape[0]*shape[2])))

            U,S,B = np.linalg.svd(reshaped_node,full_matrices=False)
            S = S / np.max(S)
            if D_cap is not None:
                U = U[:,0:D_cap]
                S = S[0:D_cap]
                B = B[0:D_cap,:]
            B = np.reshape(B,np.array((np.size(S),shape[2],shape[0])))
            B = np.einsum('ijk->kij',B)
            new_right_node = B

            new_left_node = np.einsum('ij,jk,kl->il',node1.tensor,U,np.diag(S))
            return new_left_node, new_right_node

class svd_norm_node:
    def left(node,D_cap=None):
        if node.legs == "both":
            shape = np.shape(node.tensor)
            M = node.tensor.reshape(np.array((shape[0]*shape[1],shape[2])))
            A,S,Vh = np.linalg.svd(M,full_matrices=False)
            A = np.reshape(A,np.array((shape[0],shape[1],shape[2])))
            return A

        else:
            shape = np.shape(node.tensor)
            M = node.tensor.reshape(np.array((shape[0]*shape[1],1)))
            A,S,Vh = np.linalg.svd(M,full_matrices=False)
            A = np.reshape(A,np.array((shape[0],shape[1])))
            return A

    def right(node,D_cap=None):
        if node.legs == "both":
            shape = np.shape(node.tensor)
            M = np.reshape(np.einsum('ijk->jki',node.tensor),np.array((shape[1],shape[0]*shape[2])))
            U,S,B = np.linalg.svd(M,full_matrices=False)
            B = np.reshape(B,np.array((np.size(S),shape[2],shape[0])))
            B = np.einsum('ijk->kij',B)
            return B

        else:
            shape = np.shape(node.tensor)
            M = node.tensor.transpose()
            M = M.reshape(np.array((1,shape[0]*shape[1])))
            U,S,B = np.linalg.svd(M,full_matrices=False)
            B = np.reshape(B,np.array((shape[1],shape[0])))
            return B.transpose()

