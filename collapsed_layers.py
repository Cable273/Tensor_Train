#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from rail_objects import *
class collapsed_layer:
    def factory(layer):
        #three three
        if layer.top_legs == "both" and layer.bot_legs == "both" and layer.mid_legs == "both":
            tensor = np.einsum('ijk,iuab,unm->jankbm',layer.top,layer.mid,layer.bot)
            return three_three(tensor)

        #three two
        elif layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "left":
            tensor = np.einsum('ika,ijbc,jd->kbdac',layer.top,layer.mid,layer.bot)
            return three_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "left" and layer.bot_legs == "both":
            tensor = np.einsum('ijk,iua,ubc->jabkc',layer.top,layer.mid,layer.bot)
            return three_two(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "both" and layer.bot_legs == "both":
            tensor = np.einsum('ij,iuka,ubc->jkbac',layer.top,layer.mid,layer.bot)
            return three_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "left" and layer.bot_legs == "both":
            tensor = np.einsum('iab,iuc,ude->acdbe',layer.top,layer.mid,layer.bot)
            return three_two(tensor)

        #three one
        elif layer.top_legs == "both" and layer.mid_legs == "left" and layer.bot_legs == "left":
            tensor = np.einsum('ijk,iua,ub->jabk',layer.top,layer.mid,layer.bot)
            return three_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "both" and layer.bot_legs == "left":
            tensor = np.einsum('ij,iuka,ub->jkba',layer.top,layer.mid,layer.bot)
            return three_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "left" and layer.bot_legs == "both":
            tensor = np.einsum('ia,iub,ucd->abcd',layer.top,layer.mid,layer.bot)
            return three_one(tensor)

        #three zero
        elif layer.top_legs == "left" and layer.mid_legs == "left" and layer.bot_legs == "left":
            tensor = np.einsum('ij,iuk,ua->jka',layer.top,layer.mid,layer.bot)
            return three_zero(tensor)

        #two three
        elif layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "left":
            tensor = np.einsum('ijk,iuab,un->jankb',layer.top,layer.mid,layer.bot)
            return two_three(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "both" and layer.bot_legs == "both":
            tensor = np.einsum('ij,iuak,unb->jankb',layer.top,layer.mid,layer.bot)
            return two_three(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "right" and layer.bot_legs == "both":
            tensor = np.einsum('iab,iuc,ude->adbce',layer.top,layer.mid,layer.bot)
            return two_three(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs == "right":
            tensor = np.einsum('iab,iujk,uc->ajbkc',layer.top,layer.mid,layer.bot)
            return two_three(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "both" and layer.bot_legs == "both":
            tensor = np.einsum('ij,iuka,ubc->kbjac',layer.top,layer.mid,layer.bot)
            return two_three(tensor)

        #two two
        elif layer.top_legs == "both" and layer.bot_legs == "both" and layer.mid is None:
            tensor = np.einsum('ijk,iab->jakb',layer.top,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot_legs is "no legs":
            tensor = np.einsum('ijk,iuab,u->jakb',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "left" and layer.bot_legs is "right":
            tensor = np.einsum('ijk,iua,ub->jakb',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "both" and layer.bot_legs is "right":
            tensor = np.einsum('ij,iuka,ub->jkab',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "right" and layer.bot_legs is "left":
            tensor = np.einsum('ijk,iua,ub->jbka',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "no legs" and layer.bot_legs is "both":
            tensor = np.einsum('ijk,iu,uab->uakb',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "right" and layer.bot_legs is "both":
            tensor = np.einsum('ij,iuk,uab->jakb',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "both" and layer.bot_legs is "left":
            tensor = np.einsum('ik,iuja,ub->jbka',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "left" and layer.bot_legs is "both":
            tensor = np.einsum('ij,iuk,uab->kajb',layer.top,layer.mid,layer.bot)
            return two_two(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "both" and layer.bot_legs is "both":
            tensor = np.einsum('i,iujk,uab->jakb',layer.top,layer.mid,layer.bot)
            return two_two(tensor)

        #two one
        elif layer.top_legs == "both" and layer.bot_legs == "left" and layer.mid is None:
            tensor = np.einsum('ijk,ia->jak',layer.top,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "left" and layer.bot_legs == "both" and layer.mid is None:
            tensor = np.einsum('ij,ika->jka',layer.top,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "left" and layer.bot_legs == "no legs":
            tensor = np.einsum('ijk,iua,u->jak',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "no legs" and layer.bot_legs == "left":
            tensor = np.einsum('ijk,iua,ua->jak',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "left" and layer.bot_legs == "left":
            tensor = np.einsum('ik,iuj,ua->jak',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "both" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iuka,u->jka',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "right" and layer.bot_legs == "left":
            tensor = np.einsum('ij,iuk,ua->jak',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "both" and layer.bot_legs == "left":
            tensor = np.einsum('i,iujk,ua->jak',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "left" and layer.bot_legs == "right":
            tensor = np.einsum('ij,iuk,ua->jka',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "no legs" and layer.bot_legs == "both":
            tensor = np.einsum('ij,iu,uka->jka',layer.top,layer.mid,layer.bot)
            return two_one(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "left" and layer.bot_legs == "both":
            tensor = np.einsum('i,iuj,uka->jka',layer.top,layer.mid,layer.bot)
            return two_one(tensor)

        #two zero
        elif layer.top_legs == "left" and layer.bot_legs == "left" and layer.mid is None:
            tensor = np.einsum('ij,ik->jk',layer.top,layer.bot)
            return two_zero(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "left" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iku,u->jk',layer.top,layer.mid,layer.bot)
            return two_zero(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "no legs" and layer.bot_legs == "left":
            tensor = np.einsum('j,iu,uk->jk',layer.top,layer.mid,layer.bot)
            return two_zeros(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "left" and layer.bot_legs == "left":
            tensor = np.einsum('i,iuj,uk->jk',layer.top,layer.mid,layer.bot)
            return two_zeros(tensor)

        #one three
        elif layer.top_legs == "both" and layer.mid_legs == "right" and layer.bot_legs == "right":
            tensor = np.einsum('ijk,iua,ub->jkab',layer.top,layer.mid,layer.bot)
            return one_three(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "both" and layer.bot_legs == "right":
            tensor = np.einsum('ij,iuka,ub->kjab',layer.top,layer.mid,layer.bot)
            return one_three(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "right" and layer.bot_legs == "both":
            tensor = np.einsum('ij,iuk,uab->ajkb',layer.top,layer.mid,layer.bot)
            return one_three(tensor)

        #one_two
        elif layer.top_legs == "both" and layer.bot_legs == "right" and layer.mid is None:
            tensor = np.einsum('ijk,ia->jka',layer.top,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "right" and layer.bot_legs == "both" and layer.mid is None:
            tensor = np.einsum('ij,ika->kja',layer.top,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "right" and layer.bot_legs == "no legs":
            tensor = np.einsum('ijk,iua,u->jka',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "no legs" and layer.bot_legs == "right":
            tensor = np.einsum('ijk,iu,ua->jka',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "right" and layer.bot_legs == "right":
            tensor = np.einsum('ij,iuk,ua->jka',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "both" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iuka,u->kja',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "left" and layer.bot_legs == "right":
            tensor = np.einsum('ij,iuk,ua->kja',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "both" and layer.bot_legs == "right":
            tensor = np.einsum('i,iujk,ua->jka',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "right" and layer.bot_legs == "left":
            tensor = np.einsum('ik,iua,uj->jka',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "no legs" and layer.bot_legs == "both":
            tensor = np.einsum('ij,iu,uka->kja',layer.top,layer.mid,layer.bot)
            return one_two(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "right" and layer.bot_legs == "both":
            tensor = np.einsum('i,iuj,uka->kja',layer.top,layer.mid,layer.bot)
            return one_two(tensor)

        #one one
        elif layer.top_legs == "left" and layer.bot_legs == "right" and layer.mid is None:
            tensor = np.einsum('ij,ik->jk',layer.top,layer.bot)
            return one_one(tensor)
        elif layer.top_legs == "right" and layer.bot_legs == "left" and layer.mid is None:
            tensor = np.einsum('ij,ik->kj',layer.top,layer.bot)
            return one_one(tensor)
        elif layer.top_legs == "both" and layer.mid_legs == "no legs" and layer.bot_legs == "no legs":
            tensor = np.einsum('ijk,iu,u->jk',layer.top,layer.mid,layer.bot)
            return one_one(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "left" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iuk,u->kj',layer.top,layer.mid,layer.bot)
            return one_one(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "no legs" and layer.bot_legs == "left":
            tensor = np.einsum('ij,iu,uk->kj',layer.top,layer.mid,layer.bot)
            return one_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "right" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iuk,u->jk',layer.top,layer.mid,layer.bot)
            return one_one(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "no legs" and layer.bot_legs == "right":
            tensor = np.einsum('ij,iu,uk->jk',layer.top,layer.mid,layer.bot)
            return one_one(tensor)

        #one zero
        elif layer.top_legs == "left" and layer.bot_legs == "no legs" and layer.mid is None:
            tensor = np.einsum('iu,i->u',layer.top,layer.bot)
            return one_zero(tensor)
        elif layer.top_legs == "no legs" and layer.bot_legs == "left" and layer.mid is None:
            tensor = np.einsum('i,iu->u',layer.top,layer.bot)
            return one_zero(tensor)
        elif layer.top_legs == "left" and layer.mid_legs == "no legs" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iu,u->j',layer.top,layer.mid,layer.bot)
            return one_zero(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "left" and layer.bot_legs == "no legs":
            tensor = np.einsum('i,iuj,u->j',layer.top,layer.mid,layer.bot)
            return one_zero(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "no legs" and layer.bot_legs == "left":
            tensor = np.einsum('i,iu,uj->j',layer.top,layer.mid,layer.bot)
            return one_zero(tensor)

        #zero three
        elif layer.top_legs == "right" and layer.mid_legs == "right" and layer.bot_legs == "right":
            tensor = np.einsum('ia,iub,uc->abc',layer.top,layer.mid,layer.bot)
            return zero_three(tensor)

        #zero two
        elif layer.top_legs == "right" and layer.bot_legs == "right" and layer.mid is None:
            tensor = np.einsum('ij,ik->jk',layer.top,layer.bot)
            return zero_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "right" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iuk,u->jk',layer.top,layer.mid,layer.bot)
            return zero_two(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "no legs" and layer.bot_legs == "right":
            tensor = np.einsum('ij,iu,uk->jk',layer.top,layer.mid,layer.bot)
            return zero_two(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "right" and layer.bot_legs == "right":
            tensor = np.einsum('i,iuj,uk->jk',layer.top,layer.mid,layer.bot)
            return zero_two(tensor)

        #zero one
        elif layer.top_legs == "right" and layer.bot_legs == "no legs" and layer.mid is None:
            tensor = np.einsum('iu,i->u',layer.top,layer.bot)
            return zero_one(tensor)
        elif layer.top_legs == "no legs" and layer.bot_legs == "right" and layer.mid is None:
            tensor = np.einsum('i,iu->u',layer.top,layer.bot)
            return zero_one(tensor)
        elif layer.top_legs == "right" and layer.mid_legs == "no legs" and layer.bot_legs == "no legs":
            tensor = np.einsum('ij,iu,u->j',layer.top,layer.mid,layer.bot)
            return zero_one(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "right" and layer.bot_legs == "no legs":
            tensor = np.einsum('i,iuj,u->j',layer.top,layer.mid,layer.bot)
            return zero_one(tensor)
        elif layer.top_legs == "no legs" and layer.mid_legs == "no legs" and layer.bot_legs == "right":
            tensor = np.einsum('i,iu,uj->j',layer.top,layer.mid,layer.bot)
            return zero_one(tensor)

class collapsed_MPO_layer:
    def factory(layer):
        if layer.top_legs == "both" and layer.mid_legs == "both" and layer.bot is None:
            N = np.einsum('iab,iecd->eacbd',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1]*shape[2],shape[3]*shape[4])))
            return rail_node(N,"both")
        elif layer.top_legs == "left" and layer.mid_legs == "both" and layer.bot is None:
            N = np.einsum('ij,ibka->bjka',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1]*shape[2],shape[3])))
            return rail_node(N,"both")
        elif layer.top_legs == "right" and layer.mid_legs == "both" and layer.bot is None:
            N = np.einsum('ij,ibka->bkja',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1],shape[2]*shape[3])))
            return rail_node(N,"both")
        elif layer.top_legs == "both" and layer.mid_legs == "left" and layer.bot is None:
            N = np.einsum('ijk,iba->bjak',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1]*shape[2],shape[3])))
            return rail_node(N,"both")
        elif layer.top_legs == "right" and layer.mid_legs == "left" and layer.bot is None:
            N = np.einsum('ij,iak->akj',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1],shape[2])))
            return rail_node(N,"both")
        elif layer.top_legs == "both" and layer.mid_legs == "right" and layer.bot is None:
            N = np.einsum('ijk,iab->ajk',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1],shape[2])))
            return rail_node(N,"both")
        elif layer.top_legs == "left" and layer.mid_legs == "right" and layer.bot is None:
            N = np.einsum('ia,ijk->jak',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1],shape[2])))
            return rail_node(N,"both")

        elif layer.top_legs == "right" and layer.mid_legs == "right" and layer.bot is None:
            N = np.einsum('ia,ijk->jak',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1]*shape[2])))
            return rail_node(N,"right")
        elif layer.top_legs == "left" and layer.mid_legs == "left" and layer.bot is None:
            N = np.einsum('ik,ija->jka',layer.top,layer.mid)
            shape = np.shape(N)
            N = N.reshape(np.array((shape[0],shape[1]*shape[2])))
            return rail_node(N,"left")

class three_three(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class three_two(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class three_one(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class three_zero(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class two_three(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class two_two(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class two_one(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class two_zero(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class one_three(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class one_two(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class one_one(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class one_zero(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class zero_one(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class zero_two(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor

class zero_three(collapsed_layer):
    def __init__(self,tensor):
        self.tensor = tensor
