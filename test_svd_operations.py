#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import unittest
import sys

from combine_rail_objects import combine_collapsed_layers
from collapsed_layers import *
from rail_objects import *
from MPS import *
from svd_operations import *

class test_svd_pair_shapes(unittest.TestCase):
    def setUp(self):
        dim = 3
        self.A=np.zeros(np.array((dim,2,2)))
        self.A[0] = [[0,2],[0,0]]
        self.A[1] = [[-1,0],[0,1]]
        self.A[2] = [[0,0],[-2,0]]

        self.B=np.zeros(np.array((dim,2)))
        self.B[0] = [1,0]
        self.B[1] = [1,1]
        self.B[2] = [0,1]

    def test_left_svd_on_BL(self):
        node1=rail_node(self.A,"both")
        node2=rail_node(self.B,"left")
        new_left_node , new_right_node = svd_node_pair.left(node1,node2)

        shape_left = np.shape(new_left_node)
        shape_right = np.shape(new_right_node)

        self.assertEqual(shape_left[0],3,"Should equal 3")
        self.assertEqual(shape_left[1],2,"Should equal 2")
        self.assertEqual(shape_right[0],3,"Should equal 3")
        self.assertEqual(shape_left[2],shape_right[1],"Should be equal")

    def test_left_svd_on_BB(self):
        node1=rail_node(self.A,"both")
        node2=rail_node(self.A,"both")
        new_left_node , new_right_node = svd_node_pair.left(node1,node2)

        shape_left = np.shape(new_left_node)
        shape_right = np.shape(new_right_node)

        self.assertEqual(shape_left[0],3,"Should equal 3")
        self.assertEqual(shape_left[1],2,"Should equal 2")
        self.assertEqual(shape_right[0],3,"Should equal 3")
        self.assertEqual(shape_right[2],2,"Should equal 2")
        self.assertEqual(shape_left[2],shape_right[1],"Should be equal")

    def test_right_svd_on_BL(self):
        node1=rail_node(self.A,"both")
        node2=rail_node(self.B,"left")
        new_left_node , new_right_node = svd_node_pair.left(node1,node2)

        shape_left = np.shape(new_left_node)
        shape_right = np.shape(new_right_node)

        self.assertEqual(shape_left[0],3,"Should equal 3")
        self.assertEqual(shape_left[1],2,"Should equal 2")
        self.assertEqual(shape_right[0],3,"Should equal 3")
        self.assertEqual(shape_left[2],shape_right[1],"Should be equal")

    def test_right_svd_on_BB(self):
        node1=rail_node(self.A,"both")
        node2=rail_node(self.A,"both")
        new_left_node , new_right_node = svd_node_pair.left(node1,node2)

        shape_left = np.shape(new_left_node)
        shape_right = np.shape(new_right_node)

        self.assertEqual(shape_left[0],3,"Should equal 3")
        self.assertEqual(shape_left[1],2,"Should equal 2")
        self.assertEqual(shape_right[0],3,"Should equal 3")
        self.assertEqual(shape_right[2],2,"Should equal 3")
        self.assertEqual(shape_left[2],shape_right[1],"Should be equal")

if __name__ == '__main__':
    unittest.main()

