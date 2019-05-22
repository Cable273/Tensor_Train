#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from collapsed_layers import *
class combine_collapsed_layers:
    def new_collapsed_layer(clayer1,clayer2):
        #three three
        if type(clayer1) is three_three and type(clayer2) is three_three:
            return three_three(np.einsum('ijkunm,unmabc->ijkabc',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_three and type(clayer2) is three_two:
            return three_two(np.einsum('ijkunm,unmab->ijkab',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_three and type(clayer2) is three_one:
            return three_one(np.einsum('ijkabc,abcu->ijku',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_three and type(clayer2) is three_zero:
            return three_zero(np.einsum('ijkunm,unm->ijk',clayer1.tensor,clayer2.tensor))

        #three two
        if type(clayer1) is three_two and type(clayer2) is two_three:
            return three_three(np.einsum('ijkab,abunm->ijkunm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_two and type(clayer2) is two_two:
            return three_two(np.einsum('ijkab,abun->ijkun',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_two and type(clayer2) is two_one:
            return three_one(np.einsum('ijkab,abc->ijkc',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_two and type(clayer2) is two_zero:
            return three_zero(np.einsum('ijkab,ab->ijk',clayer1.tensor,clayer2.tensor))

        #three one
        if type(clayer1) is three_one and type(clayer2) is one_three:
            return three_three(np.einsum('ijka,aunm->ijkunm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_one and type(clayer2) is one_two:
            return three_two(np.einsum('ijka,aun->ijkun',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_one and type(clayer2) is one_one:
            return three_one(np.einsum('ijka,ab->ijkb',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is three_one and type(clayer2) is one_zero:
            return three_zero(np.einsum('ijka,a->ijk',clayer1.tensor,clayer2.tensor))

        #two three
        if type(clayer1) is two_three and type(clayer2) is three_three:
            return two_three(np.einsum('ijabc,abcunm->ijunm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_three and type(clayer2) is three_two:
            return two_two(np.einsum('ijabc,abcun->ijun',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_three and type(clayer2) is three_one:
            return two_one(np.einsum('ijabc,abcd->ijd',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_three and type(clayer2) is three_zero:
            return two_zero(np.einsum('ijabc,abc->ij',clayer1.tensor,clayer2.tensor))

        #two two
        if type(clayer1) is two_two and type(clayer2) is two_three:
            return two_three(np.einsum('ijab,abunm->ijunm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_two and type(clayer2) is two_two:
            return two_two(np.einsum('ijab,abun->ijun',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_two and type(clayer2) is two_one:
            return two_one(np.einsum('ijab,abc->ijc',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_two and type(clayer2) is two_zero:
            return two_zero(np.einsum('ijab,ab->ij',clayer1.tensor,clayer2.tensor))

        #two one
        if type(clayer1) is two_one and type(clayer2) is one_three:
            return two_three(np.einsum('ija,aunm->ijunm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_one and type(clayer2) is one_two:
            return two_two(np.einsum('ija,aun->ijun',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_one and type(clayer2) is one_one:
            return two_one(np.einsum('ija,ab->ijb',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is two_one and type(clayer2) is one_zero:
            return two_zero(np.einsum('ija,a->ij',clayer1.tensor,clayer2.tensor))

        #one three
        if type(clayer1) is one_three and type(clayer2) is three_three:
            return one_three(np.einsum('iunm,unmabc->iabc',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_three and type(clayer2) is three_two:
            return one_two(np.einsum('iunm,unmab->iab',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_three and type(clayer2) is three_one:
            return one_one(np.einsum('iunm,unma->ia',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_three and type(clayer2) is three_zero:
            return one_zero(np.einsum('iunm,unm->i',clayer1.tensor,clayer2.tensor))

        #one two
        if type(clayer1) is one_two and type(clayer2) is two_three:
            return one_three(np.einsum('iab,abunm->iunm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_two and type(clayer2) is two_two:
            return one_two(np.einsum('iab,abun->iun',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_two and type(clayer2) is two_one:
            return one_one(np.einsum('iab,abu->iu',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_two and type(clayer2) is two_zero:
            return one_zero(np.einsum('iab,ab->i',clayer1.tensor,clayer2.tensor))

        #one one
        if type(clayer1) is one_one and type(clayer2) is one_three:
            return one_three(np.einsum('ia,aunm->iunm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_one and type(clayer2) is one_two:
            return one_two(np.einsum('ia,aun->iun',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_one and type(clayer2) is one_one:
            return one_one(np.einsum('ia,ab->ib',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is one_one and type(clayer2) is one_zero:
            return one_zero(np.einsum('ia,a->i',clayer1.tensor,clayer2.tensor))

        #zero three
        if type(clayer1) is zero_three and type(clayer2) is three_three:
            return zero_three(np.einsum('abc,abcunm->unm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is zero_three and type(clayer2) is three_two:
            return zero_two(np.einsum('abc,abcun->un',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is zero_three and type(clayer2) is three_one:
            return zero_one(np.einsum('abc,abcu->u',clayer1.tensor,clayer2.tensor))

        #zero two
        if type(clayer1) is zero_two and type(clayer2) is two_three:
            return zero_three(np.einsum('ab,abunm->unm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is zero_two and type(clayer2) is two_two:
            return zero_two(np.einsum('ab,abun->un',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is zero_two and type(clayer2) is two_one:
            return zero_one(np.einsum('ab,abu->u',clayer1.tensor,clayer2.tensor))

        #zero one
        if type(clayer1) is zero_one and type(clayer2) is one_three:
            return zero_three(np.einsum('i,iunm->unm',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is zero_one and type(clayer2) is one_two:
            return zero_two(np.einsum('i,iab->ab',clayer1.tensor,clayer2.tensor))
        if type(clayer1) is zero_one and type(clayer2) is one_one:
            return zero_one(np.einsum('i,ia->a',clayer1.tensor,clayer2.tensor))

    def scalar(clayer1,clayer2):
        if type(clayer1) is three_three and type(clayer2) is three_three:
            return np.einsum('ijkabc,abcijk',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is three_two and type(clayer2) is two_three:
            return np.einsum('ijkab,abijk',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is three_one and type(clayer2) is one_three:
            return np.einsum('ijka,aijk',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is three_zero and type(clayer2) is zero_three:
            return np.einsum('ijk,ijk',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is two_three and type(clayer2) is three_two:
            return np.einsum('ijabc,abcij',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is two_two and type(clayer2) is two_two:
            return np.einsum('ijab,abij',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is two_one and type(clayer2) is one_two:
            return np.einsum('ija,aij',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is two_zero and type(clayer2) is zero_two:
            return np.einsum('ij,ij',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is one_three and type(clayer2) is three_one:
            return np.einsum('iabc,abci',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is one_two and type(clayer2) is two_one:
            return np.einsum('iab,abi',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is one_one and type(clayer2) is one_one:
            return np.einsum('ia,ai',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is one_zero and type(clayer2) is zero_one:
            return np.einsum('i,i',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is zero_three and type(clayer2) is three_zero:
            return np.einsum('abc,abc',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is zero_two and type(clayer2) is two_zero:
            return np.einsum('ab,ab',clayer1.tensor,clayer2.tensor)
        if type(clayer1) is zero_one and type(clayer2) is one_zero:
            return np.einsum('i,i',clayer1.tensor,clayer2.tensor)

class combine_mpoNode_clayers:
    def factory(mpo_node,clayer1,clayer2=None):
        if clayer2 is None:
            if mpo_node.legs=="left": 
                H_tensor = np.einsum('iuk,jka->ijua',mpo_node.tensor,clayer1.tensor)
                shape = np.shape(H_tensor)
                H_tensor = H_tensor.reshape(np.array((shape[0]*shape[1],shape[2]*shape[3])))
                return H_tensor

            elif mpo_node.legs=="right":
                H_tensor = np.einsum('iub,abc->iauc',mpo_node.tensor,clayer1.tensor)
                shape = np.shape(H_tensor)
                H_tensor = H_tensor.reshape(np.array((shape[0]*shape[1],shape[2]*shape[3])))
                return H_tensor

        elif type(clayer1) is zero_three and type(clayer2) is three_zero and mpo_node.legs=="both":
            H_tensor =  np.einsum('abc,ijbk,ukn->iaujcn',clayer1.tensor,mpo_node.tensor,clayer2.tensor)
            shape = np.shape(H_tensor)
            H_tensor = H_tensor.reshape(np.array((shape[0]*shape[1]*shape[2],shape[3]*shape[4]*shape[5])))
            return H_tensor

        elif type(clayer1) is three_three and type(clayer2) is three_three and mpo_node.legs=="both":
            H_tensor = np.einsum('abcdef,ijeh,ghzklm->idgjfz',clayer1.tensor,mpo_node.tensor,clayer2.tensor)
            shape = np.shape(H_tensor)
            H_tensor = H_tensor.rehsape(np.array((shape[0]*shape[1]*shape[2],shape[3]*shape[4]*shape[5])))
            return H_tensor

class combine_mpo_nodes_vertical:
    def factory(node1,node2):
        if node1.legs == "both" and node2.legs == "both":
            mpo_tensor =  np.einsum('ijab,jkcd->ikacbd',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2]*shape[3],shape[4]*shape[5])))
            return rail_node(mpo_tensor,"both")
        if node1.legs == "both" and node2.legs == "left":
            mpo_tensor =  np.einsum('ijab,jkc->ikacb',node1,tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2]*shape[3],shape[4])))
            return rail_node(mpo_tensor,"both")
        if node1.legs == "both" and node2.legs == "right":
            mpo_tensor =  np.einsum('ijab,jkc->ikabc',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2],shape[3]*shape[4])))
            return rail_node(mpo_tensor,"both")

        if node1.legs == "right" and node2.legs == "both":
            mpo_tensor =  np.einsum('ija,jkbc->ikbac',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2],shape[3]*shape[4])))
            return rail_node(mpo_tensor,"both")
        if node1.legs == "right" and node2.legs == "left":
            mpo_tensor =  np.einsum('ijb,jka->ikab',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2],shape[3])))
            return rail_node(mpo_tensor,"both")
        if node1.legs == "right" and node2.legs == "right":
            mpo_tensor =  np.einsum('ija,jkb->ikab',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2]*shape[3])))
            return rail_node(mpo_tensor,"right")

        if node1.legs == "left" and node2.legs == "both":
            mpo_tensor =  np.einsum('ija,jkbc->ikabc',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2]*shape[3],shape[4])))
            return rail_node(mpo_tensor,"both")
        if node1.legs == "left" and node2.legs == "left":
            mpo_tensor =  np.einsum('ija,jkb->ikab',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2]*shape[3])))
            return rail_node(mpo_tensor,"left")
        if node1.legs == "left" and node2.legs == "right":
            mpo_tensor =  np.einsum('ika,kjb->ijab',node1.tensor,node2.tensor)
            shape = np.shape(mpo_tensor)
            mpo_tensor =  mpo_tensor.reshape(np.array((shape[0],shape[1],shape[2],shape[3])))
            return rail_node(mpo_tensor,"both")
