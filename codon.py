import itertools

from Bio.Seq import Seq


def _init_codons():
    global codons, translate, codon_to_index, index_to_codon
    codons = []
    translate = {}
    codon_to_index = {}
    index_to_codon = {}
    i = 0
    for codon in itertools.product('ATGC', repeat=3):
        codon = ''.join(codon)
        aa = str(Seq(codon).translate())
        if str(aa) != '*':
            codons.append(str(codon))
            translate[codon] = str(aa)
            codon_to_index[codon] = i
            index_to_codon[i] = codon
            i += 1

def distance(codon1, codon2):
    return sum(n1 != n2 for n1, n2 in zip(codon1, codon2))

def substitution_position(codon1, codon2):
    # Returns the first position which differs between codons.
    # returns -1 if no difference is found
    for i, (n1, n2) in enumerate(zip(codon1, codon2)):
        if n1 != n2:
            return i
    return -1

def is_synonymous(codon1, codon2):
    return translate[codon1] == translate[codon2]

def is_transition(codon1, codon2):
    for n1, n2 in zip(codon1, codon2):
        if n1 != n2:
            if (n1 in 'AG' and n2 in 'AG') or \
               (n1 in 'CT' and n2 in 'CT'):
                return True
            else:
                return False
    return False

_init_codons()
