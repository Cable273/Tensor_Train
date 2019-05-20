#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from combine_rail_objects import *
from collapsed_layers import *
from rail_objects import *

class rail_network:
    def __init__(self,MPS1,MPS2=None,Q=None):
        self.top_row = MPS1
        self.bot_row = MPS2
        self.mid_row = Q
        self.length = MPS1.length

    def contract(self):
        collapsed_edge = collapsed_layer.factory(layer(self,site=0))
        for site in range(1,self.length-1):
            collapsed_next_site = collapsed_layer.factory(layer(self,site))
            collapsed_edge = combine_collapsed_layers.new_collapsed_layer(collapsed_edge,collapsed_next_site)

        collapsed_right_edge = collapsed_layer.factory(layer(self,site=self.length-1))
        self.contraction = combine_collapsed_layers.scalar(collapsed_edge,collapsed_right_edge)
