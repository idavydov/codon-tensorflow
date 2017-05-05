import numpy as np
import tensorflow as tf

from codon import *

## creates a substitution matrix, cond is a function
def sub_mat(cond):
    size = len(codons)
    mat = np.zeros((size, size))
    for i, c1 in enumerate(codons):
        for j, c2 in enumerate(codons):
            if cond(c1, c2):
                mat[i, j] = 1
    return mat

class Q:
    def __init__(self, w, k, freq):
        self.w = w
        self.k = k
        self.freq = freq
        self._create_matrix()

    def _create_matrix(self):
        q_pre = (self.w * nonsyn + syn) * (self.k * transitions + transversions)

        # multiply Q by frequencies (double check that not transposed)
        q_pre = q_pre * np.tile(self.freq, (len(codons), 1))

        # sum of rows
        
        rowsum = tf.reduce_sum(q_pre, 1)
        self.scale = tf.reduce_sum(rowsum * self.freq)

        # set diagnoal to negative sum
        self.q = q_pre - tf.diag(rowsum)

        # do PAML-style eigendecomposition 
        pi_12 = tf.diag(np.sqrt(self.freq))
        pi_12_inv = tf.diag(1/np.sqrt(self.freq))
        a = tf.matmul(tf.matmul(pi_12, self.q), pi_12_inv)
        self.e_val, e_vec = tf.self_adjoint_eig(a)
        self.u = tf.matmul(pi_12_inv, e_vec)
        self.u_inv = tf.matmul(tf.matrix_transpose(e_vec), pi_12)


    # compute P matrix
    def P(self, t, scale=None):
        if scale is None:
            scale = self.scale
        p = tf.matmul(self.u, tf.diag(tf.exp(self.e_val * t / scale)))
        p = tf.matmul(p, self.u_inv)
        return p


# matrix of allowed subsitutions
dist1 = sub_mat(lambda c1, c2: distance(c1, c2) == 1)
# matrix of nonsynonymous subsitutions
nonsyn = sub_mat(lambda c1, c2: not is_synonymous(c1, c2)) * dist1
# matrix of synonymous subsitutions
syn = dist1 - nonsyn
# matrix of transitions
transitions = sub_mat(is_transition) * dist1
# matrix of transitions
transversions = dist1 - transitions
