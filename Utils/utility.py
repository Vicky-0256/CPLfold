"""
utility.py
provides feature functions.
Note: scalar parameters (multi_*, external_*) are fetched dynamically from
Utils.feature_weight at call time so that training scripts can update them
without re-importing this module.
"""

import numpy as np
from Utils.feature_weight import *

# Constants and parameters
NOTON = 5
NOTOND = 25
NOTONT = 125
SINGLE_MAX_LEN = 30
SINGLE_MIN_LEN = 0
EXPLICIT_MAX_LEN = 4
HAIRPIN_MAX_LEN = 30
INTERNAL_MAX_LEN = 30
SYMMETRIC_MAX_LEN = 15
ASYMMETRY_MAX_LEN = 28
BULGE_MAX_LEN = 30

# Initialize cache_single
cache_single = np.zeros((SINGLE_MAX_LEN + 1, SINGLE_MAX_LEN + 1))

def initialize_cachesingle():
    global cache_single
    cache_single = np.zeros((SINGLE_MAX_LEN + 1, SINGLE_MAX_LEN + 1))
    for l1 in range(SINGLE_MIN_LEN, SINGLE_MAX_LEN + 1):
        for l2 in range(SINGLE_MIN_LEN, SINGLE_MAX_LEN + 1):
            if l1 == 0 and l2 == 0:
                continue

            # bulge
            if l1 == 0:
                cache_single[l1][l2] += bulge_length[l2]
            elif l2 == 0:
                cache_single[l1][l2] += bulge_length[l1]
            else:
                # internal
                cache_single[l1][l2] += internal_length[min(l1 + l2, INTERNAL_MAX_LEN)]

                # internal explicit
                if l1 <= EXPLICIT_MAX_LEN and l2 <= EXPLICIT_MAX_LEN:
                    cache_single[l1][l2] += internal_explicit[(l1 * EXPLICIT_MAX_LEN + l2) if l1 <= l2 else (l2 * EXPLICIT_MAX_LEN + l1)]

                # internal symmetry
                if l1 == l2:
                    cache_single[l1][l2] += internal_symmetric_length[min(l1, SYMMETRIC_MAX_LEN)]
                else:  # internal asymmetry
                    diff = abs(l1 - l2)
                    cache_single[l1][l2] += internal_asymmetry[min(diff, ASYMMETRY_MAX_LEN)]
    return

# ------------- nucs based scores -------------

def base_pair_score(nuci, nucj):
    return base_pair[nucj * NOTON + nuci]
    # + or -， I don't know, of base_pair_score specific to the sequence based on Greg's list

def helix_stacking_score(nuci, nuci1, nucj_1, nucj):
    return helix_stacking[nuci * NOTONT + nucj * NOTOND + nuci1 * NOTON + nucj_1]

def helix_closing_score(nuci, nucj):
    return helix_closing[nuci * NOTON + nucj]

def terminal_mismatch_score(nuci, nuci1, nucj_1, nucj):
    return terminal_mismatch[nuci * NOTONT + nucj * NOTOND + nuci1 * NOTON + nucj_1]

def bulge_nuc_score(nuci):
    return bulge_0x1_nucleotides[nuci]

def internal_nuc_score(nuci, nucj):
    return internal_1x1_nucleotides[nuci * NOTON + nucj]

def dangle_left_score(nuci, nuci1, nucj):
    return dangle_left[nuci * NOTOND + nucj * NOTON + nuci1]

def dangle_right_score(nuci, nucj_1, nucj):
    return dangle_right[nuci * NOTOND + nucj * NOTON + nucj_1]

# ------------- length based scores -------------

def hairpin_score(i, j):
    return hairpin_length[min(j - i - 1, HAIRPIN_MAX_LEN)]

def internal_length_score(l):
    return internal_length[min(l, INTERNAL_MAX_LEN)]

def internal_explicit_score(l1, l2):
    l1_ = min(l1, EXPLICIT_MAX_LEN)
    l2_ = min(l2, EXPLICIT_MAX_LEN)
    return internal_explicit[(l1_ * NOTON + l2_) if l1_ <= l2_ else (l2_ * NOTON + l1_)]

def internal_sym_score(l):
    return internal_symmetric_length[min(l, SYMMETRIC_MAX_LEN)]

def internal_asym_score(l1, l2):
    diff = abs(l1 - l2)
    return internal_asymmetry[min(diff, ASYMMETRY_MAX_LEN)]

def bulge_length_score(l):
    return bulge_length[min(l, BULGE_MAX_LEN)]

def hairpin_at_least_score(l):
    return hairpin_length_at_least[min(l, HAIRPIN_MAX_LEN)]

def buldge_length_at_least_score(l):
    return bulge_length_at_least[min(l, BULGE_MAX_LEN)]

def internal_length_at_least_score(l):
    return internal_length_at_least[min(l, INTERNAL_MAX_LEN)]

#-----------------------------------------------------
def score_junction_A(i, j, nuci, nuci1, nucj_1, nucj, len):
    return helix_closing_score(nuci, nucj) + \
           (0 if i >= len - 1 else dangle_left_score(nuci, nuci1, nucj)) + \
           (0 if j <= 0 else dangle_right_score(nuci, nucj_1, nucj))

def score_junction_B(i, j, nuci, nuci1, nucj_1, nucj):
    return helix_closing_score(nuci, nucj) + terminal_mismatch_score(nuci, nuci1, nucj_1, nucj)

def score_hairpin_length(len):
    return hairpin_length[min(len, HAIRPIN_MAX_LEN)]

def score_hairpin(i, j, nuci, nuci1, nucj_1, nucj):
    return hairpin_length[min(j - i - 1, HAIRPIN_MAX_LEN)] + \
           score_junction_B(i, j, nuci, nuci1, nucj_1, nucj)

def score_helix(nuci, nuci1, nucj_1, nucj):
    return helix_stacking_score(nuci, nuci1, nucj_1, nucj) + base_pair_score(nuci1, nucj_1)

def score_single_nuc(i, j, p, q, nucp_1, nucq1):
    l1 = p - i - 1
    l2 = j - q - 1
    if l1 == 0 and l2 == 1:
        return bulge_nuc_score(nucq1)
    if l1 == 1 and l2 == 0:
        return bulge_nuc_score(nucp_1)
    if l1 == 1 and l2 == 1:
        return internal_nuc_score(nucp_1, nucq1)
    return 0

def score_single(i, j, p, q, len, nuci, nuci1, nucj_1, nucj, nucp_1, nucp, nucq, nucq1):
    l1 = p - i - 1
    l2 = j - q - 1
    return cache_single[l1][l2] + \
           base_pair_score(nucp, nucq) + \
           score_junction_B(i, j, nuci, nuci1, nucj_1, nucj) + \
           score_junction_B(q, p, nucq, nucq1, nucp_1, nucp) + \
           score_single_nuc(i, j, p, q, nucp_1, nucq1)

# score_single without socre_junction_B
def score_single_without_junctionB(i, j, p, q, nucp_1, nucp, nucq, nucq1):
    l1 = p - i - 1
    l2 = j - q - 1
    return cache_single[l1][l2] + \
           base_pair_score(nucp, nucq) + \
           score_single_nuc(i, j, p, q, nucp_1, nucq1)

def score_multi(i, j, nuci, nuci1, nucj_1, nucj, len):
    # fetch dynamic scalars at call-time so updates are visible
    from Utils import feature_weight as _fw
    return score_junction_A(i, j, nuci, nuci1, nucj_1, nucj, len) + \
           _fw.multi_paired + _fw.multi_base

def score_multi_unpaired(i, j):
    from Utils import feature_weight as _fw
    return (j - i + 1) * _fw.multi_unpaired

def score_M1(i, j, k, nuci_1, nuci, nuck, nuck1, len):
    from Utils import feature_weight as _fw
    return score_junction_A(k, i, nuck, nuck1, nuci_1, nuci, len) + \
           score_multi_unpaired(k + 1, j) + base_pair_score(nuci, nuck) + _fw.multi_paired

def score_external_paired(i, j, nuci_1, nuci, nucj, nucj1, len):
    from Utils import feature_weight as _fw
    return score_junction_A(j, i, nucj, nucj1, nuci_1, nuci, len) + \
           _fw.external_paired + base_pair_score(nuci, nucj) # + greg_pair_score(i,j)*alpha

# def greg_pair_score(i,j):
#     read_lookups = greg_table[i][j]

    # return there are different options
    # log(read_lookups)
    # -log(read_lookups)
    # ）

def score_external_unpaired(i, j):
    from Utils import feature_weight as _fw
    return (j - i + 1) * _fw.external_unpaired
