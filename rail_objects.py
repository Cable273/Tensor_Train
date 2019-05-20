#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
class rail_node:
    def __init__(self,tensor,legs):
        self.tensor = tensor
        #legs in horizontal direction. Init to "left", "both" or "right"
        self.legs = legs


class layer:
    def __init__(self,rail_network,site):
        self.top = rail_network.top_row.node[site].tensor
        if rail_network.bot_row is None:
            self.bot=None
        else:
            self.bot = rail_network.bot_row.node[site].tensor

        #mid row is MPO
        if rail_network.mid_row is None:
            self.mid = None
        else:
            self.mid = rail_network.mid_row.node[site].tensor

        #leg variables
        self.top_legs = rail_network.top_row.node[site].legs
        if self.bot is None:
            self.bot_legs = None
        else:
            self.bot_legs = rail_network.bot_row.node[site].legs

        if self.mid is None:
            self.mid_legs = None
        else:
            self.mid_legs = rail_network.mid_row.node[site].legs
