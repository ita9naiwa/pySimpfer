import itertools
import logging
import multiprocessing
import random
import time

import cython
import numpy as np
import scipy.sparse
import tqdm
from cython.parallel import parallel, prange, threadid
from cython cimport floating, integral
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport log10

from libcpp cimport bool
from libcpp.algorithm cimport sort
from libcpp.algorithm cimport binary_search
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
cimport numpy as np
cimport scipy.linalg.cython_blas as cython_blas

log = logging.getLogger("pysimpfer")


cdef inline floating gmemv(int M, int N, floating *A, floating *x, floating *y) nogil:
    cdef char r = 'T'
    cdef int INCX = 1, INCY = 1
    cdef int LDA = M
    cdef floating beta = 0.0
    cdef floating alpha = -1.0
    if floating is double:
        cython_blas.dgemv(&r, &M, &N, &alpha, A, &LDA, x, &INCX, &beta, y, &INCY)
    elif floating is float:
        cython_blas.sgemv(&r, &M, &N, &alpha, A, &LDA, x, &INCX, &beta, y, &INCY)

cdef extern from "util.h" nogil:
    cdef int get_tid()

@cython.cdivision(True)
@cython.boundscheck(False)
cdef floating dot(floating[:] x, floating[:] y, int factors) nogil:
    cdef floating ret = 0;
    for i in range(factors):
        ret += x[i] * y[i]
    return ret

class Simpfer():
    def __init__(self, Q, P, K=50, block_c=3):
        self.org_Q = np.copy(Q)
        self.org_P = np.copy(P)
        self.pre_K = K

        self.Q_indmap, \
        self.Q, self.P, \
        self.Q_norm, self.P_norm, \
        self.L, self.block_Ls, self.inds \
        = self.construct(self.org_Q, self.org_P, K, block_c)

    def construct(self, Q, P, k, block_c, max_chunk_size=64):
        n = Q.shape[0]
        _Q_norm = np.sqrt((Q * Q).sum(-1))
        _P_norm = np.sqrt((P * P).sum(-1))
        _Q_args = np.argsort(-_Q_norm)
        _P_args = np.argsort(-_P_norm)
        _Q = Q[_Q_args]
        _P = P[_P_args]
        _Q_norm = _Q_norm[_Q_args]
        _P_norm = _P_norm[_P_args]
        top_k_P = _P[:k, :]
        Q_indmap = {i: k for i, k in enumerate(_Q_args)}
        # Q_indmap = _Q_args
        r = np.dot(_Q, top_k_P.T)
        L = np.sort(r, axis=-1)[:, ::-1]
        chunk_size = max(max_chunk_size, int(block_c  * (1 + np.log(1 + n))))
        blocks, block_Ls, inds = [], [], []
        for i in range(0, n+1, chunk_size):
            if i == len(L):
                break
            blocks.append(_Q[i:i+chunk_size])
            block_Ls.append(np.min(L[i:i+chunk_size], axis=0))
            inds.append([i, min(len(L), i + chunk_size)])

        return (Q_indmap, _Q, _P, _Q_norm, _P_norm, L,
                np.array(block_Ls), np.array(inds, dtype=np.int32))
    def rmips(self, item_idx, k=5, backend='cython'):
        if backend == 'python':
            return self.rmips_v1(item_idx, k)
        elif backend == 'cython':
            return self.rmips_v2(item_idx, k)
        else:
            raise IndexError("Not implemented")


    def rmips_v1(self, item_idx, k=5):
        ret_arr = set()
        q = self.org_P[item_idx]
        q_norm = np.sqrt(np.dot(q, q))
        r1 = []
        for block_l, inds  in zip(self.block_Ls, self.inds):
            if self.Q_norm[inds[0]] * q_norm <= block_l[k]:
                # print('Skip', inds, 'Smaller than Lower bound for block')
                continue

            for i in range(inds[0], inds[1]):
                u = self.Q[i]
                d = np.dot(u, q)
                if d <= self.L[i, k]:
                    # print("Skip type 2: Smaller than Lower bound for single user")
                    continue
                elif d > self.Q_norm[i] * self.P_norm[k]:
                    ret_arr.add(self.Q_indmap[i])
                    # print("Skip type 3: Force Add")
                else:
                    ret = np.dot(self.org_P, self.Q[i])
                    ret = np.argpartition(-ret, k)[:k]
                    if item_idx in ret:
                        ret_arr.add(self.Q_indmap[i])
        return ret_arr

    def rmips_v2(self, item_idx, k=5):
        _ret = __find(item_idx, self.org_P[item_idx],
                     self.Q, self.P, self.org_P,
                     self.Q_norm, self.P_norm,
                     self.L, self.block_Ls,
                     self.inds, k=k)
        for x in _ret:
            try:
                self.Q_indmap[x]
            except:
                print(f"KEYERROR!! {x}")
        return set(self.Q_indmap[x] for x in _ret)



@cython.cdivision(True)
@cython.boundscheck(False)
def __find(int item_idx, floating[:] q,
           floating[:, :] Q, floating[:, :] P, floating[:, :] org_P,
           floating[:] Q_norm, floating[:] P_norm,
           floating[:, :] L, floating[:, :] block_Ls,
           integral[:, :] inds,
           integral k=10,
           int num_threads=4):
    cdef int tid, block_id, i, j;
    cdef int n_users, n_items, n_blocks, dim ;
    n_users = Q.shape[0]
    n_items = P.shape[0]
    dim = P.shape[1]
    n_blocks = block_Ls.shape[0]
    cdef int block_b, block_e;
    cdef vector[vector[int]] return_values
    cdef floating* _temp

    for i in range(num_threads):
        return_values.push_back(vector[int]())

    cdef floating q_norm = np.sqrt(np.dot(q, q))
    cdef floating d;
    with nogil, parallel(num_threads=num_threads):
        _temp = <floating*> malloc(sizeof(floating) * n_items)
        tid = get_tid()
        for block_id in prange(n_blocks, schedule='static'):
            block_b, block_e = inds[block_id][0], inds[block_id][1]
            if Q_norm[block_b] * q_norm < block_Ls[block_id, k]:
                continue
            for i in range(block_b, block_e):
                d = dot(Q[i], q, dim)
                if d <= L[i, k]:
                    continue
                elif d > Q_norm[i] * P_norm[k]:
                    return_values[tid].push_back(i)
                    continue

                gmemv(dim, n_items, &org_P[0, 0], &Q[i, 0], _temp)
                sort(_temp, _temp + n_items)
                if (d +_temp[k]) >= 1e-8:
                    return_values[tid].push_back(i)
        free(_temp)
    ret = set()
    for tid in range(num_threads):
        for j in return_values[tid]:
            ret.add(j)
    return ret
