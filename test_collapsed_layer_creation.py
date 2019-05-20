#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import unittest
import sys

from combine_rail_objects import combine_collapsed_layers
from collapsed_layers import *
from rail_objects import *
from MPS import *

class dummy_layer:
    def __init__(self,A,B,C,top_legs,mid_legs,bot_legs):
        self.top = A
        self.mid = B
        self.bot = C
        self.top_legs = top_legs
        self.mid_legs = mid_legs
        self.bot_legs = bot_legs

class test_collapsed_layer_creation(unittest.TestCase):
    def setUp(self):
        #dummy data (AKLT gs and heisenberg MPO)
        dim = 3
        self.A=np.zeros(np.array((dim,2,2)))
        self.A[0] = [[0,np.power(2/3,0.5)],[0,0]]
        self.A[1] = [[-np.power(1/3,0.5),0],[0,np.power(1/3,0.5)]]
        self.A[2] = [[0,0],[-np.power(2/3,0.5),0]]

        self.B=np.zeros(np.array((dim,2)))
        self.B[0] = [1/np.power(8,0.25),0]
        self.B[1] = [1/np.power(8,0.25),1/np.power(8,0.25)]
        self.B[2] = [0,1/np.power(8,0.25)]

        #Heisenberg MPO
        Z=np.array([[1,0,0],[0,1,0],[0,0,-1]])
        I=np.array([[1,0,0],[0,1,0],[0,0,1]])
        X=np.array([[0,np.power(2,0.5),0],[np.power(2,0.5),0,np.power(2,0.5)],[0,np.power(2,0.5),0]])

        Z=np.array([[1,0,0],[0,1,0],[0,0,-1]])
        I=np.array([[1,0,0],[0,1,0],[0,0,1]])
        X=np.array([[0,np.power(2,0.5),0],[np.power(2,0.5),0,np.power(2,0.5)],[0,np.power(2,0.5),0]])

        self.Q=np.zeros(np.array((3,3,3,3)))
        self.Q[0,0] = I
        self.Q[0,1] = Z
        self.Q[0,2] = -X
        self.Q[2,1] = Z
        self.Q[2,2] = I
        self.V=np.zeros(np.array((3,3,3)))
        self.V[0] = -X
        self.V[1] = Z
        self.V[2] = I
        self.W=np.zeros(np.array((3,3,3)))
        self.W[0] = I
        self.W[1] = Z
        self.W[2] = X
        self.Q=np.einsum('ijkl->klij',self.Q)
        self.V=np.einsum('ijk->jki',self.V)
        self.W=np.einsum('ijk->jki',self.W)

    #three threes
    def test_three_three_from_BBB(self):
        edge = dummy_layer(self.A,self.Q,self.A,"both","both","both")
        self.assertIsInstance(collapsed_layer.factory(edge),three_three,"Should be a three three collapsed layer")

    #three two
    def test_three_two_from_BBL(self):
        edge = dummy_layer(self.A,self.Q,self.B,"both","both","left")
        self.assertIsInstance(collapsed_layer.factory(edge),three_two,"Should be a three two collapsed layer")
    def test_three_two_from_LBB(self):
        edge = dummy_layer(self.B,self.Q,self.A,"left","both","both")
        self.assertIsInstance(collapsed_layer.factory(edge),three_two,"Should be a three two collapsed layer")
    def test_three_two_from_BLB(self):
        edge = dummy_layer(self.A,self.V,self.A,"both","left","both")
        self.assertIsInstance(collapsed_layer.factory(edge),three_two,"Should be a three two collapsed layer")
    def test_three_two_from_BLB(self):
        edge = dummy_layer(self.A,self.V,self.A,"both","left","both")
        self.assertIsInstance(collapsed_layer.factory(edge),three_two,"Should be a three two collapsed layer")

    #three one
    def test_three_one_from_BLL(self):
        edge = dummy_layer(self.A,self.V,self.B,"both","left","left")
        self.assertIsInstance(collapsed_layer.factory(edge),three_one,"Should be a three one collapsed layer")
    def test_three_one_from_LBL(self):
        edge = dummy_layer(self.B,self.Q,self.B,"left","both","left")
        self.assertIsInstance(collapsed_layer.factory(edge),three_one,"Should be a three one collapsed layer")
    def test_three_one_from_LLB(self):
        edge = dummy_layer(self.B,self.V,self.A,"left","left","both")
        self.assertIsInstance(collapsed_layer.factory(edge),three_one,"Should be a three one collapsed layer")

    #three zero
    def test_three_zero_from_LLL(self):
        edge = dummy_layer(self.B,self.V,self.B,"left","left","left")
        self.assertIsInstance(collapsed_layer.factory(edge),three_zero,"Should be a three zero collapsed layer")

    #two three
    def test_two_three_from_BBR(self):
        edge = dummy_layer(self.A,self.Q,self.B,"both","both","right")
        self.assertIsInstance(collapsed_layer.factory(edge),two_three,"Should be a two three collapsed layer")
    def test_two_three_from_BRB(self):
        edge = dummy_layer(self.A,self.V,self.A,"both","right","both")
        self.assertIsInstance(collapsed_layer.factory(edge),two_three,"Should be a two three collapsed layer")
    def test_two_three_from_RBB(self):
        edge = dummy_layer(self.B,self.Q,self.A,"right","both","both")
        self.assertIsInstance(collapsed_layer.factory(edge),two_three,"Should be a two three collapsed layer")

    #two two
    def test_two_three_from_BNB(self):
        edge = dummy_layer(self.A,None,self.A,"both",None,"both")
        self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")
    # def test_two_three_from_BnlB(self):
        # edge = dummy_layer(self.A,self.V,self.A,"both","no legs","both")
        # self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")
    def test_two_three_from_BRL(self):
        edge = dummy_layer(self.A,self.V,self.B,"both","right","left")
        self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")
    def test_two_three_from_RBL(self):
        edge = dummy_layer(self.B,self.Q,self.B,"right","both","left")
        self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")
    def test_two_three_from_BLR(self):
        edge = dummy_layer(self.A,self.V,self.B,"both","left","right")
        self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")
    def test_two_three_from_RLB(self):
        edge = dummy_layer(self.B,self.V,self.A,"right","left","both")
        self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")
    def test_two_three_from_RBL(self):
        edge = dummy_layer(self.B,self.Q,self.B,"right","both","left")
        self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")
    def test_two_three_from_LRB(self):
        edge = dummy_layer(self.B,self.V,self.A,"left","right","both")
        self.assertIsInstance(collapsed_layer.factory(edge),two_two,"Should be a two two collapsed layer")

    #two one
    def test_two_one_from_LLR(self):
        edge = dummy_layer(self.B,self.V,self.B,"left","left","right")
        self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")
    # def test_two_one_from_BNlL(self):
        # edge = dummy_layer(self.B,self.V,self.B,"left","left","right")
        # self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")
    # def test_two_one_from_LNlB(self):
        # edge = dummy_layer(self.B,self.V,self.B,"left","right","left")
        # self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")
    def test_two_one_from_LRL(self):
        edge = dummy_layer(self.B,self.V,self.B,"left","right","left")
        self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")
    def test_two_one_from_RLL(self):
        edge = dummy_layer(self.B,self.V,self.B,"right","left","left")
        self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")

    def test_two_one_from_BNL(self):
        edge = dummy_layer(self.A,None,self.B,"both",None,"left")
        self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")
    def test_two_one_from_LNB(self):
        edge = dummy_layer(self.B,None,self.A,"left",None,"both")
        self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")

    #two zero
    # def test_two_zero_from_LNlL(self):
        # edge = dummy_layer(self.B,self.V,self.B,"left","left","right")
        # self.assertIsInstance(collapsed_layer.factory(edge),two_one,"Should be a two one collapsed layer")
    def test_two_zero_from_LNL(self):
        edge = dummy_layer(self.B,None,self.B,"left",None,"left")
        self.assertIsInstance(collapsed_layer.factory(edge),two_zero,"Should be a two zero collapsed layer")

    #one three
    def test_one_three_from_BRR(self):
        edge = dummy_layer(self.A,self.V,self.B,"both","right","right")
        self.assertIsInstance(collapsed_layer.factory(edge),one_three,"Should be a one three collapsed layer")
    def test_one_three_from_RBR(self):
        edge = dummy_layer(self.B,self.Q,self.B,"right","both","right")
        self.assertIsInstance(collapsed_layer.factory(edge),one_three,"Should be a one three collapsed layer")
    def test_one_three_from_RRB(self):
        edge = dummy_layer(self.B,self.V,self.A,"right","right","both")
        self.assertIsInstance(collapsed_layer.factory(edge),one_three,"Should be a one three collapsed layer")

    #one two
    def test_one_two_from_RRL(self):
        edge = dummy_layer(self.B,self.V,self.B,"right","right","left")
        self.assertIsInstance(collapsed_layer.factory(edge),one_two,"Should be a one two collapsed layer")
    def test_one_two_from_RLR(self):
        edge = dummy_layer(self.B,self.V,self.B,"right","left","right")
        self.assertIsInstance(collapsed_layer.factory(edge),one_two,"Should be a one two collapsed layer")
    def test_one_two_from_LRR(self):
        edge = dummy_layer(self.B,self.V,self.B,"left","right","right")
        self.assertIsInstance(collapsed_layer.factory(edge),one_two,"Should be a one two collapsed layer")
    # def test_one_two_from_BNlR(self):
        # edge = dummy_layer(self.A,self.V,self.B,"both","no legs","right")
        # self.assertIsInstance(collapsed_layer.factory(edge),one_two,"Should be a one two collapsed layer")
    # def test_one_two_from_RNlB(self):
        # edge = dummy_layer(self.B,self.V,self.A,"right","no legs","both")
        # self.assertIsInstance(collapsed_layer.factory(edge),one_two,"Should be a one two collapsed layer")

    #one one
    # def test_one_one_from_RNlL(self):
        # edge = dummy_layer(self.B,self.V,self.B,"right","no legs","left")
        # self.assertIsInstance(collapsed_layer.factory(edge),one_one,"Should be a one one collapsed layer")
    # def test_one_one_from_LNlR(self):
        # edge = dummy_layer(self.B,self.V,self.B,"left","no legs","right")
        # self.assertIsInstance(collapsed_layer.factory(edge),one_one,"Should be a one one collapsed layer")
    def test_one_one_from_LNR(self):
        edge = dummy_layer(self.B,None,self.B,"left",None,"right")
        self.assertIsInstance(collapsed_layer.factory(edge),one_one,"Should be a one one collapsed layer")
    def test_one_one_from_RNL(self):
        edge = dummy_layer(self.B,None,self.B,"right",None,"left")
        self.assertIsInstance(collapsed_layer.factory(edge),one_one,"Should be a one one collapsed layer")

    #zero two
    # def test_zero_two_from_RNlR(self):
        # edge = dummy_layer(self.B,self.V,self.B,"right","no legs","left")
        # self.assertIsInstance(collapsed_layer.factory(edge),zero_two,"Should be a one one collapsed layer")
    def test_zero_two_from_RNR(self):
        edge = dummy_layer(self.B,None,self.B,"right",None,"right")
        self.assertIsInstance(collapsed_layer.factory(edge),zero_two,"Should be a zero two collapsed layer")

    #zero three
    def test_zero_three_from_RRR(self):
        edge = dummy_layer(self.B,self.V,self.B,"right","right","right")
        self.assertIsInstance(collapsed_layer.factory(edge),zero_three,"Should be a zero three collapsed layer")

if __name__ == '__main__':
    unittest.main()

