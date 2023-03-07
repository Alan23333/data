import os
from random import random

import cv2
import scipy.io as sio
from scipy.sparse import linalg
import torch
import os
import numpy as np
import scipy.io as sio
from scipy.sparse import linalg
import torch
import numpy as geek
import random

from sklearn.decomposition import TruncatedSVD

import common
from common import *


def tprod(A, B):
    # % Tensor - Tensor product of two 3 - way tensor: C = A * B
    # % A - n1 * n2 * n3 tensor
    # % B - n2 * l * n3 tensor
    # % C - n1 * l * n3 tensor
    n1, _, n3 = A.shape
    l = B[2].size()
    A = np.fft(A)
    B = np.fft(B)
    C = np.zeros(n1, l, n3)
    for i  in range(n3):
        C[:,:, i] = np.dot(A[:,:, i], B[:,:, i])

    C = np.real(np.ifft(C))
    return C



def t_svd(A):
    Af = np.fft.fft(A,axis=2)
    U = np.zeros(tuple([A.shape[0],A.shape[0],A.shape[2]]), dtype=complex)
    S = np.zeros(tuple(A.shape), dtype=complex)
    V = np.zeros(tuple([A.shape[1],A.shape[1],A.shape[2]]), dtype=complex)

    for i3 in range(A.shape[2]):
        uf,sf,vf = np.linalg.svd(Af[:,:,i3])
        sf = np.diag(sf)

        U[:,:,i3] = uf
        S[:, :, i3] = sf
        V[:, :, i3] = vf
    U = np.fft.ifft(U,axis=2)
    S = np.fft.ifft(S,axis=2)
    V = np.fft.ifft(V,axis=2)

    return U,S,V


p = 10
mu = 1e-5
sf = 8
MSI = np.random.uniform(1, 10, size=(96, 96, 3))
HSI = np.random.uniform(1, 10, size=(12, 12, 29))
C = np.random.uniform(1, 10, size=(96, 96))
R = np.random.uniform(1, 10, size=(3, 29))
# MSI(96,96,3) HSI(12,12,29) C(96,96) sf=8

# # np.moveaxis(HSI, 2, 0)
# HSI3 = reshape(np.moveaxis(HSI, 2, 0), (HSI.shape[2], -1))
# # HSI3 (29, 144) MSI(96,96,3) HSI(12,12,29) C(96,96)
# D, _, _ = linalg.svds(HSI3, p)  D(29,10)


HSI_up1_twist = HSI.transpose(1, 0) # HSI_up1_twist((12,29,12))
U, _, _ = t_svd(HSI_up1_twist) # U(12,29,12)

D= U[:,0:p,:] # D(12,10,12)
A = tprod(D.transpose, HSI_up1_twist)   # D(10, 12, 12)  HSI_up1_twist((12, 29, 12))  A(10, 29, 12)



# D (29, 10)
# D = D[:, 0:p]

RD = np.dot(R, D) # RD(3,10)  R(3, 29)  D(29, 10) ——> RD (3, 144)
L1 = D.shape[1] # L1 10 ——> L1 144
nr = MSI.shape[0] # nr 96 不变
nc = MSI.shape[1] # nc 96 不变
L = HSI.shape[2] # L 29
HSI_int = np.zeros((nr, nc, L)) # HSI_int(96,96,29)
HSI_int[0::sf, 0::sf, :] = HSI #上采样
FBmC = np.conj(C) # FBmC(96,96)
FBs1 = np.tile(C[:, :, np.newaxis], (1, 1, L1)) # FBs1(96,96,10)
FBCs = np.tile(FBmC[:, :, np.newaxis], (1, 1, L)) # FBs1(96,96,29)
FBCs1 = np.tile(FBmC[:, :, np.newaxis], (1, 1, L1)) # FBCs1(96,96,10)
HHH = ifft2((fft2(HSI_int) * FBCs)) # HHH(96,96,29)
HHH1 = hyperConvert2D(HHH) # HHH1(29,9216)
MSI3 = reshape(np.moveaxis(MSI, 2, 0), (MSI.shape[2], -1)) # MSI3(3, 9216)
n_dr = nr // sf # n_dr 12
n_dc = nc // sf # n_dc 12
HR_load1 = cv2.resize(HSI, None, fx=sf, fy=sf, interpolation=cv2.INTER_CUBIC) # HR_load1(96,96,29)
V2 = np.dot(D.T, hyperConvert2D(HR_load1)) # V2(10,9216)
CCC = np.dot(RD.T, MSI3) + np.dot(D.T, HHH1) # CCC(10,9216)
C1 = np.dot(RD.T, RD) + mu * np.eye(D.shape[1]) # C1(10,10)
Lambda, Q = np.linalg.eig(C1) #Lambda(10)  Q(10,10)
Lambda = reshape(Lambda, (1, 1, L1)) #Lambda(1,1,10)
InvLbd = 1 / np.tile(Lambda, (sf * n_dr, sf * n_dc, 1)) # InvLbd(96,96,10)
B2Sum = PPlus(np.power(np.abs(FBs1), 2) / (sf * sf), n_dr, n_dc) # B2Sum(96,96,10)
InvDI = 1 / (B2Sum[0:n_dr, 0:n_dc, :] + np.tile(Lambda, (n_dr, n_dc, 1))) # InvDI(12,12,10)
HR_HSI3 = mu * V2 # HR_HSI3(10, 9216)
C3 = CCC + HR_HSI3 #C3(10, 9216)
C30 = fft2(reshape(np.dot(np.linalg.inv(Q), C3).T, (nr, nc, L1))) * InvLbd # C30(96,96,10)
temp = PPlus_s(C30 / (sf * sf) * FBs1, n_dr, n_dc) # temp(12,12,10)
invQUF = C30 - np.tile(temp * InvDI, (sf, sf, 1)) * FBCs1 # invQUF(96,96,10)
VXF = np.dot(Q, reshape(invQUF, (nr * nc, L1)).T) # VXF(10, 9216)
A = reshape(np.real(ifft2(reshape(VXF.T, (nr, nc, L1)))), (nr * nc, L1)).T # A(10, 9216)
A = reshape(A.T, (nr, nc, -1)) # A (96,96,10)


# return D, A


