#!/usr/bin/env python3
import sys
import dendropy
import scipy.optimize

from Bio import SeqIO

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

import numpy as np

from codon import *
from q import Q

# string to list of codon indexes
def to_codon(s):
    assert len(s) % 3 == 0
    s = str(s)
    return [codon_to_index[s[i:i+3]] for i in range(0, len(s), 3)]

# convert index to vector with all zeros and one 1 at this index
def n2v(n):
    assert n >= 0 and n < len(codons)
    v = np.zeros(size)
    v[n] = 1
    return v


if __name__ == '__main__':
    ali = {r.id: to_codon(r.seq) for r in SeqIO.parse(sys.argv[1], 'fasta')}
    tree = dendropy.Tree.get(path=sys.argv[2], schema='newick')

    size = len(codons)


    # frequency F0
    freq = np.ones(size) / size

    # model parameters
    w0 = tf.Variable(0.5, dtype=tf.float64)
    w2 = tf.Variable(4, dtype=tf.float64)
    p01sum = tf.Variable(0.5, dtype=tf.float64)
    p0prop = tf.Variable(0.867, dtype=tf.float64)
    k = tf.Variable(1.0, dtype=tf.float64)

    q0 = Q(w0, k, freq)
    q1 = Q(tf.constant(1.0, dtype=tf.float64), k, freq)
    q2 = Q(w2, k, freq)

    p0 = p0prop * p01sum
    p1 = p01sum - p0
    p2a = (1 - p0 - p1) * p0 / (p0 + p1)
    p2b = (1 - p0 - p1) * p1 / (p0 + p1)

    prop = [p0, p1, p2a, p2b]

    bg_scale = (p0 + p2a) * q0.scale + (p1 + p2b) * q1.scale
    fg_scale = p0 * q0.scale + p1 * q1.scale + (p2a + p2b) * q2.scale

    # save all the edge lengths
    edges = []
    edge_lengths = []
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            node.pl = [np.transpose(np.array([n2v(c) for c in ali[node.taxon.label]]))] * 4
        else:
            node.pl = [None]  * 4
            for child in node.child_node_iter():
                el = max(child.edge_length, 1e-9)
                edge_lengths.append(el)
                el = tf.Variable(el, dtype=tf.float64)
                
                edges.append(el)
                # if we don't want to optimize branches
                #el = tf.constant(el, dtype=tf.float64)
                if child.label == '#1':
                    p = [q.P(el, fg_scale) for q in (q0, q1, q2, q2)]
                else:
                    p = [q.P(el, bg_scale) for q in (q0, q1, q0, q1)]

                # set partial likelihood
                for i in range(len(p)):
                    delta_pl = tf.matmul(p[i], child.pl[i])
                    if node.pl[i] is None:
                        node.pl[i] = delta_pl
                    else:
                        node.pl[i] *= delta_pl

            if node.level() == 0:
                # compute size-wise likelihoods
                pvec = [pr * tf.matmul(pl, tf.constant(freq, shape=(len(codons),1)),
                                  transpose_a=True) for pr, pl in zip(prop, node.pl)]
                # and full likelihood
                lnL = tf.reduce_sum(tf.log(sum(pvec)))


    # parameter boundaries
    bounds=[(0.05, 1),
            (1, 20),
            (0.01, 1),
            (0.01, 1),
            (0.005, 20)] + [[1e-9, 100.]] * len(edges)
    grad = tf.gradients(lnL, [w0, w2, p01sum, p0prop, k] + edges)
    def f(x):
        d = {w0: x[0], w2: x[1], p01sum: x[2], p0prop: x[3], k: x[4]}
        for i, e in enumerate(edges):
            d[e] = x[5 + i]
        v = session.run(lnL, d)
        return -v
    def df(x):
        d = {w0: x[0], w2: x[1], p01sum: x[2], p0prop: x[3], k: x[4]}
        for i, e in enumerate(edges):
            d[e] = x[5 + i]
        g = session.run(grad, d)
        if np.any(np.isnan(g)):
            #print('x=', x)
            #print('g(x)=', g)
            g = np.nan_to_num(g)
        return -np.array(g)

    config=tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        print(lnL.eval())
        x0 = [0.5, 2.0, 0.5, 0.5, 2.0] + edge_lengths
        res = scipy.optimize.minimize(f, x0, jac=df, method='L-BFGS-B',
                                      bounds=bounds, options={'disp': True})
        print(res.x)
