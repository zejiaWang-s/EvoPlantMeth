from __future__ import division, print_function
from collections import OrderedDict
import numpy as np

CHAR_TO_INT = OrderedDict([('A', 0), ('T', 1), ('G', 2), ('C', 3), ('N', 4)])
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

def get_alphabet(special=False, reverse=False):
    alpha = OrderedDict(CHAR_TO_INT)
    if not special: del alpha['N']
    if reverse: alpha = {v: k for k, v in alpha.items()}
    return alpha

def char_to_int(seq):
    return [CHAR_TO_INT[x] for x in seq.upper()]

def int_to_char(seq, join=True):
    t = [INT_TO_CHAR[x] for x in seq]
    return ''.join(t) if join else t

def int_to_onehot(seqs, dim=4):
    seqs = np.atleast_2d(np.asarray(seqs))
    n, l = seqs.shape
    enc_seqs = np.zeros((n, l, dim), dtype='int8')
    for i in range(dim):
        enc_seqs[seqs == i, i] = 1
    return enc_seqs

def onehot_to_int(seqs, axis=-1):
    return seqs.argmax(axis=axis)