#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from collapsed_layers import *
from rail_objects import *
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
            return two_zero(np.einsum('abcd,cd->ab',clayer1.tensor,clayer2.tensor))

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
            return zero_two(np.einsum('ab,abcd->cd',clayer1.tensor,clayer2.tensor))
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

#combine collapsed layer and uncollapsed layer - for optimal contraction strategy when contracting a rail network/dmrg
class combine_clayer_layer:
    def new_collapsed_layer(object1,object2):
        #collapsed layer on the left
        if isinstance(object1,collapsed_layer) is True:
            clayer = object1
            layer = object2

            if type(clayer) is zero_three and layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "both":
                temp = np.einsum('abc,iaj->ijbc',clayer.tensor,layer.top)
                temp = np.einsum('ijbc,ikbv->kjvc',temp,layer.mid)
                temp = np.einsum('kjvc,kcd->jvd',temp,layer.bot)
                return zero_three(temp)

            if type(clayer) is three_three and layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "both":
                temp = np.einsum('abcdef,aidg->iabcgef',clayer.tensor,layer.top)
                temp = np.einsum('iabcgef,ijeu->jabcguf',temp,layer.mid)
                temp = np.einsum('jabcguf,jfv->abcguv',temp,layer.bot)
                return three_three(temp)

            if type(clayer) is three_three and layer.top_legs == "left" and layer.mid_legs == "left" and layer.bot_legs == "left":
                temp = np.einsum('abcdef,id->iabcef',clayer.tensor,layer.top)
                temp = np.einsum('iabcef,ije->jabcf',temp,layer.mid)
                temp = np.einsum('jabcf,jf->abc',temp,layer.bot)
                return three_zero(temp)

            if type(clayer) is zero_two and layer.top_legs == "both" and layer.mid is None and layer.bot_legs == "both":
                temp = np.einsum('ab,iac->icb',clayer.tensor,layer.top)
                temp = np.einsum('icb,ibd->cd',temp,layer.bot)
                return zero_two(temp)

            if type(clayer) is two_two and layer.top_legs == "both" and layer.mid is None and layer.bot_legs == "both":
                temp = np.einsum('abcd,ice->iabed',clayer.tensor,layer.top)
                temp = np.einsum('iabed,idf->abef',temp,layer.bot)
                return two_two(temp)

            if type(clayer) is two_two and layer.top_legs == "left" and layer.mid is None and layer.bot_legs == "left":
                temp = np.einsum('abcd,ic->iabd',clayer.tensor,layer.top)
                temp = np.einsum('iabd,id->ab',temp,layer.bot)
                return two_zero(temp)

        #collapsed layer on the right
        else:
            clayer = object2
            layer = object1

            if type(clayer) is three_zero and layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "both":
                temp = np.einsum('abc,ida->idbc',clayer.tensor,layer.top)
                temp = np.einsum('idbc,ijeb->jdec',temp,layer.mid)
                temp = np.einsum('jdec,jfc->def',temp,layer.bot)
                return three_zero(temp)

            if type(clayer) is three_three and layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "both":
                temp = np.einsum('defabc,iud->iuefabc',clayer.tensor,layer.top)
                temp = np.einsum('iuefabc,ijne->junfabc',temp,layer.mid)
                temp = np.einsum('junfabc,jmf->unmabc',temp,layer.bot)
                return three_three(temp)

            if type(clayer) is three_three and layer.top_legs == "right" and layer.mid_legs == "right" and layer.bot_legs == "right":
                temp = np.einsum('abcdef,ia->ibcdef',clayer.tensor,layer.top)
                temp = np.einsum('ibcdef,ijb->jcdef',temp,layer.mid)
                temp = np.einsum('jcdef,jc->def',temp,layer.bot)
                return zero_three(temp)

            if type(clayer) is two_zero and layer.top_legs == "both" and layer.mid is None and layer.bot_legs == "both":
                temp = np.einsum('ab,ica->icb',clayer.tensor,layer.top)
                temp = np.einsum('icb,idb->cd',temp,layer.bot)
                return two_zero(temp)

            if type(clayer) is two_two and layer.top_legs == "both" and layer.mid is None and layer.bot_legs == "both":
                temp = np.einsum('cdab,iec->iedab',clayer.tensor,layer.top)
                temp = np.einsum('iedab,ifd->efab',temp,layer.bot)
                return two_two(temp)

            if type(clayer) is two_two and layer.top_legs == "right" and layer.mid is None and layer.bot_legs == "right":
                temp = np.einsum('abcd,ia->ibcd',clayer.tensor,layer.top)
                temp = np.einsum('ibcd,ib->cd',temp,layer.bot)
                return zero_two(temp)

    def scalar(object1,object2):
        #collapsed layer on the left
        if isinstance(object1,collapsed_layer) is True:
            clayer = object1
            layer = object2

            if type(clayer) is zero_three and layer.top_legs == "left" and layer.mid_legs == "left" and layer.bot_legs == "left":
                temp = np.einsum('abc,ia->ibc',clayer.tensor,layer.top)
                temp = np.einsum('ibc,ijb->jc',temp,layer.mid)
                temp = np.einsum('jc,jc',temp,layer.bot)
                return temp

            if type(clayer) is three_three and layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "both":
                temp = np.einsum('abcefg,iea->ibcfg',clayer.tensor,layer.top)
                temp = np.einsum('ibcfg,ijfb->jcg',temp,layer.mid)
                temp = np.einsum('jcg,jgc',temp,layer.bot)
                return temp

            if type(clayer) is zero_two and layer.top_legs == "left" and layer.mid is None and layer.bot_legs == "left":
                temp = np.einsum('ab,ia->ib',clayer.tensor,layer.top)
                temp = np.einsum('ib,ib',temp,layer.bot)
                return temp

            if type(clayer) is two_two and layer.top_legs == "both" and layer.mid is None and layer.bot_legs == "both":
                temp = np.einsum('abef,iea->ibf',clayer.tensor,layer.top)
                temp = np.einsum('ibf,ifb',temp,layer.bot)
                return temp
        #collapsed layer on the right
        else:
            clayer = object2
            layer = object1

            if type(clayer) is three_zero and layer.top_legs == "right" and layer.mid_legs == "right" and layer.bot_legs == "right":
                temp = np.einsum('abc,ia->ibc',clayer.tensor,layer.top)
                temp = np.einsum('ibc,ijb->jc',temp,layer.mid)
                temp = np.einsum('jc,jc',temp,layer.bot)
                return temp

            if type(clayer) is three_three and layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "both":
                temp = np.einsum('efgabc,iae->ifgbc',clayer.tensor,layer.top)
                temp = np.einsum('ifgbc,ijbf->jgc',temp,layer.mid)
                temp = np.einsum('jgc,jcg',temp,layer.bot)
                return temp

            if type(clayer) is two_zero and layer.top_legs == "right" and layer.mid is None and layer.bot_legs == "right":
                temp = np.einsum('ab,ia->ib',clayer.tensor,layer.top)
                temp = np.einsum('ib,ib',clayer.tensor,layer.bot)
                return temp

            if type(clayer) is two_two and layer.top_legs == "both" and layer.mid is None and layer.bot_legs == "both":
                temp = np.einsum('efab,iae->ifb',clayer.tensor,layer.top)
                temp = np.einsum('ifb,ibf',clayer.tensor,layer.bot)
                return temp

#to form H matrix for DMRG optimization
class combine_mpoNode_clayers:
    def factory(mpo_node,clayer1,clayer2=None):
        if clayer2 is None:
            if mpo_node.legs=="left": 
                H_tensor = np.einsum('iuk,jka->ijua',mpo_node.tensor,clayer1.tensor)
                shape = np.shape(H_tensor)
                H_tensor = H_tensor.reshape(np.array((shape[0]*shape[1],shape[2]*shape[3])))
                return H_tensor

            elif mpo_node.legs=="right":
                H_tensor = np.einsum('abc,dce->adbe',mpo_node.tensor,clayer1.tensor)
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

#for repeated application of MPOS (time evolution)
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

        if node1.tensor is not None and node2.tensor is None:
            return rail_node(node1.tensor,node1.legs)
        elif node1.tensor is None and node2.tensor is not None:
            return rail_node(node2.tensor,node2.legs)

