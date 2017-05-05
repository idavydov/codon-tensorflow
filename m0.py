#!/usr/bin/env python3
import sys
import dendropy

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

    # omega
    w = tf.Variable(0.2, dtype=tf.float64)
    # kappa
    k = tf.Variable(2.0, dtype=tf.float64)

    q = Q(w, k, freq)


    # save all the edge lengths
    edges = []
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            node.pl = tf.constant(np.transpose(np.array([n2v(c) for c in ali[node.taxon.label]])))
        else:
            for child in node.child_node_iter():
                el = tf.Variable(max(child.edge_length, 1e-9), dtype=tf.float64)
                edges.append(el)
                # if we don't want to optimize branches
                #el = tf.constant(max(child.edge_length, 1e-9), dtype=tf.float64)

                # compute p-vatrix
                p = q.P(el)

                delta_pl = tf.matmul(p, child.pl)

                # set partial likelihood
                try:
                    node.pl *= delta_pl
                except AttributeError:
                    node.pl = delta_pl
            if node.level() == 0:
                # compute size-wise likelihoods
                pvec = tf.matmul(node.pl, tf.constant(freq, shape=(len(codons),1)),
                                 transpose_a=True)
                # and full likelihood
                lnL = tf.reduce_sum(tf.log(pvec))


    # parameter boundaries
    bounds=[(0.05, 20), (0.05, 20)] + [[1e-9, 100.]] * len(edges)
    optimizer = ScipyOptimizerInterface(
        -lnL, bounds=bounds, method='L-BFGS-B', options={'disp': True})

    with tf.Session() as session:
        # initialize starting values
        session.run(tf.global_variables_initializer())

        
        print('starting lnL', lnL.eval())
        optimizer.minimize(session)
        print('final lnL', lnL.eval())
        print('w=%s, k=%s' % (w.eval(), k.eval()))
