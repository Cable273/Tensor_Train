#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import unittest
import sys
import copy

from combine_rail_objects import combine_collapsed_layers
from collapsed_layers import *
from rail_objects import *
from MPS import *
from svd_operations import *

def equal_matrices(A,B,tol):
    return (np.abs(A-B)<tol).all()

class test_normalizations(unittest.TestCase):
    def setUp(self):
        self.dim = 3
        self.A=np.zeros(np.array((self.dim,2,2)))
        self.A[0] = [[0,2],[0,0]]
        self.A[1] = [[-1,0],[0,1]]
        self.A[2] = [[0,0],[-2,0]]

        self.B=np.zeros(np.array((self.dim,2)))
        self.B[0] = [1,0]
        self.B[1] = [1,1]
        self.B[2] = [0,1]

        self.periodic_MPS = mps.uniform(10,self.A)
        self.open_MPS = mps.uniform(10,self.A,self.B,self.B)

        self.I = np.eye(2)
        self.tol=1e-5

    def test_left_normalization_of_open_MPS(self):
        self.open_MPS.left_normalize()

        A_dagger = np.conj(self.open_MPS.node[0].tensor)
        product = np.einsum('ij,ik->jk',A_dagger,self.open_MPS.node[0].tensor)
        self.assertTrue(equal_matrices(product,self.I,self.tol),"Left norm failing on 0th site")

        for n in range(1,self.open_MPS.length-1):
            A_dagger = np.conj(np.einsum('ijk->ikj',self.open_MPS.node[n].tensor))
            product = np.einsum('ijk,iku->ju',A_dagger,self.open_MPS.node[n].tensor)
            self.assertTrue(equal_matrices(product,self.I,self.tol),"Left norm failing on site "+str(n))

        A_dagger = np.conj(self.open_MPS.node[self.open_MPS.length-1].tensor)
        product = np.einsum('ij,ik->jk',A_dagger,self.open_MPS.node[self.open_MPS.length-1].tensor)
        self.assertTrue(equal_matrices(product,self.I,self.tol),"Left norm failing on last site")

    def test_left_normalization_of_periodic_MPS(self):
        self.periodic_MPS.left_normalize()
        for n in range(0,self.periodic_MPS.length):
            A_dagger = np.conj(np.einsum('ijk->ikj',self.periodic_MPS.node[n].tensor))
            product = np.einsum('ijk,iku->ju',A_dagger,self.periodic_MPS.node[n].tensor)
            self.assertTrue(equal_matrices(product,self.I,self.tol),"Left norm failing on site "+str(n))

    def test_right_normalization_of_open_MPS(self):
        self.open_MPS.right_normalize()

        B_dagger = np.conj(self.open_MPS.node[0].tensor)
        product = np.einsum('ij,ik->jk',self.open_MPS.node[0].tensor,B_dagger)
        self.assertTrue(equal_matrices(product,self.I,self.tol),"Right norm failing on 0th site")

        for n in range(1,self.open_MPS.length-1):
            B_dagger = np.conj(np.einsum('ijk->ikj',self.open_MPS.node[n].tensor))
            product = np.einsum('ijk,iku->ju',self.open_MPS.node[n].tensor,B_dagger)
            self.assertTrue(equal_matrices(product,self.I,self.tol),"Right norm failing on site "+str(n))

        B_dagger = np.conj(self.open_MPS.node[self.open_MPS.length-1].tensor)
        product = np.einsum('ij,ik->jk',self.open_MPS.node[self.open_MPS.length-1].tensor,B_dagger)
        self.assertTrue(equal_matrices(product,self.I,self.tol),"Right norm failing on last site")

    def test_right_normalization_of_periodic_MPS(self):
        self.periodic_MPS.left_normalize()
        for n in range(0,self.periodic_MPS.length):
            B_dagger = np.conj(np.einsum('ijk->ikj',self.periodic_MPS.node[n].tensor))
            product = np.einsum('ijk,iku->ju',self.periodic_MPS.node[n].tensor,B_dagger)
            self.assertTrue(equal_matrices(product,self.I,self.tol),"Right norm failing on site "+str(n))

    def test_right_normalization_of_periodic_MPS(self):
        self.periodic_MPS.left_normalize()
        for n in range(0,self.periodic_MPS.length):
            B_dagger = np.conj(np.einsum('ijk->ikj',self.periodic_MPS.node[n].tensor))
            product = np.einsum('ijk,iku->ju',self.periodic_MPS.node[n].tensor,B_dagger)
            self.assertTrue(equal_matrices(product,self.I,self.tol),"Right norm failing on site "+str(n))

    def test_mixed_normalization_of_periodic_MPS(self):
        for n in range(0,self.periodic_MPS.length):
            orig_MPS = copy.deepcopy(self.periodic_MPS)
            orig_MPS.mixed_normalize(n)
            for m in range(0,n): #check L norm of sites left of cut
                A_dagger = np.conj(np.einsum('ijk->ikj',orig_MPS.node[m].tensor))
                product = np.einsum('ijk,iku->ju',A_dagger,orig_MPS.node[m].tensor)
                self.assertTrue(equal_matrices(product,self.I,self.tol),"Mixed norm site "+str(n)+", L norm failing at site "+str(m))
            for m in range(n+1,orig_MPS.length): #check R norm of sites right of cut
                B_dagger = np.conj(np.einsum('ijk->ikj',orig_MPS.node[m].tensor))
                product = np.einsum('ijk,iku->ju',orig_MPS.node[m].tensor,B_dagger)
                self.assertTrue(equal_matrices(product,self.I,self.tol),"Mixed norm site "+str(n)+", R norm failing at site "+str(m))

    def test_mixed_normalization_of_open_MPS(self):
        for n in range(0,self.periodic_MPS.length):
            orig_MPS = copy.deepcopy(self.open_MPS)
            orig_MPS.mixed_normalize(n)

            #L norm of 0th site
            if n != 0:
                A_dagger = np.conj(orig_MPS.node[0].tensor)
                product = np.einsum('ij,ik->jk',A_dagger,orig_MPS.node[0].tensor)
                self.assertTrue(equal_matrices(product,self.I,self.tol),"Mixed norm site "+str(n)+",L norm failing at first site")

            #L norm of sites left of cut
            for m in range(1,n):
                A_dagger = np.conj(np.einsum('ijk->ikj',orig_MPS.node[m].tensor))
                product = np.einsum('ijk,iku->ju',A_dagger,orig_MPS.node[m].tensor)
                self.assertTrue(equal_matrices(product,self.I,self.tol),"Mixed norm site "+str(n)+", L norm failing at site "+str(m))

            #R norm of sites right of cut
            for m in range(n+1,orig_MPS.length-1):
                B_dagger = np.conj(np.einsum('ijk->ikj',orig_MPS.node[m].tensor))
                product = np.einsum('ijk,iku->ju',orig_MPS.node[m].tensor,B_dagger)
                self.assertTrue(equal_matrices(product,self.I,self.tol),"Mixed norm site "+str(n)+", R norm failing at site "+str(m))

            #R norm of far right site
            if n!= orig_MPS.length-1:
                B_dagger = np.conj(orig_MPS.node[orig_MPS.length-1].tensor)
                product = np.einsum('ij,ik->jk',orig_MPS.node[orig_MPS.length-1].tensor,B_dagger)
                self.assertTrue(equal_matrices(product,self.I,self.tol),"Mixed norm site "+str(n)+", R norm failing at last site ")

if __name__ == '__main__':
    unittest.main()
