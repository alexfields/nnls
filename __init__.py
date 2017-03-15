#!/usr/bin/env python

import numpy as np


def nnls_predotted(A_dot_A, A_dot_b, tol=1e-8):
    assert A_dot_A.shape[0] == A_dot_A.shape[1] == A_dot_b.shape[0]
    nvar = A_dot_A.shape[0]
    P_bool = np.zeros(nvar, np.bool)
    x = np.zeros(nvar)
    s = np.empty_like(x)
    w = A_dot_b
    while not P_bool.all() and w.max() > tol:
        j_idx = w[~P_bool].argmax()
        P_bool[np.flatnonzero(~P_bool)[j_idx]] = True
        s[:] = 0
        currPs = np.flatnonzero(P_bool)
        if len(currPs) > 1:
            s[currPs] = np.linalg.solve(A_dot_A[currPs[:, None], currPs[None, :]], A_dot_b[currPs])
        else:
            currP = currPs[0]
            s[currP] = A_dot_b[currP]/A_dot_A[currP, currP]
        s_P_le_0 = (s[currPs] <= 0)
        while s_P_le_0.any():
            currPs_s_P_le_0 = currPs[s_P_le_0]
            alpha = (x[currPs_s_P_le_0]/(x[currPs_s_P_le_0] - s[currPs_s_P_le_0])).min()
            x += alpha*(s-x)
            P_bool[currPs] = (x[currPs] > tol)
            s[:] = 0
            currPs = np.flatnonzero(P_bool)
            if len(currPs) > 1:
                s[currPs] = np.linalg.solve(A_dot_A[currPs[:, None], currPs[None, :]], A_dot_b[currPs])
            else:
                currP = currPs[0]
                s[currP] = A_dot_b[currP]/A_dot_A[currP, currP]
            s_P_le_0 = (s[currPs] <= 0)
        x[:] = s[:]
        w = A_dot_b - A_dot_A.dot(x)
    return x