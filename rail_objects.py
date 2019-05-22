#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
class rail_node:
    def __init__(self,tensor=None,legs=None):
        self.tensor = tensor
        #legs in horizontal direction. Init to "left", "both" or "right"
        self.legs = legs
        #shape
        if self.tensor is not None:
            self.shape = np.shape(tensor)
        else:
            self.shape = None


class layer:
    def __init__(self,rail_network,site):
        if rail_network.top_row.node[site] is None:
            self.top=None
        else:
            self.top=rail_network.top_row.node[site].tensor

        if rail_network.bot_row is None:
            self.bot=None
        else:
            if rail_network.bot_row.node[site] is not None:
                self.bot = rail_network.bot_row.node[site].tensor
            else:
                self.bot = None

        #mid row is MPO
        if rail_network.mid_row is None:
            self.mid = None
        else:
            self.mid = rail_network.mid_row.node[site].tensor

        #leg variables
        if rail_network.top_row.node[site] is None:
            self.top_legs = None
        else:
            self.top_legs = rail_network.top_row.node[site].legs
        if self.bot is None:
            self.bot_legs = None
        else:
            self.bot_legs = rail_network.bot_row.node[site].legs

        if self.mid is None:
            self.mid_legs = None
        else:
            self.mid_legs = rail_network.mid_row.node[site].legs


