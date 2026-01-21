"""
utility_v.py 
provides feature functions for vienna model.

author: Kai Zhao, Dezhong Deng
edited by: 02/2018  
"""

# pairs: 0:NP 1:CG 2:GC 3:GU 4:UG 5:AU 6:UA 7:NN
# nucleotides: CONTRAfold: 0:A 1:C 2:G 3:U 4:N ; Vienna: 0:N 1:A 2:C 3:G 4:U
from collections import defaultdict

# CONTRAfold nucleotides: 0:A 1:C 2:G 3:U 4:N
# Vienna nucleotides: 0:N 1:A 2:C 3:G 4:U
# Convert CONTRAfold numbering to Vienna numbering
def NUM_TO_NUC(x):
    if x == -1:
        return -1
    elif x == 4:
        return 0
    else:
        return x + 1

# NUM_TO_PAIR: Convert CONTRAfold-encoded nucleotides to pair type
# x,y are in CONTRAfold encoding (0:A, 1:C, 2:G, 3:U, 4:N)
# pairs: 0:NP 1:CG 2:GC 3:GU 4:UG 5:AU 6:UA
def NUM_TO_PAIR(x, y):
    if x == 0:    # A
        return 5 if y == 3 else 0  # AU
    elif x == 1:  # C
        return 1 if y == 2 else 0  # CG
    elif x == 2:  # G
        return 2 if y == 1 else (3 if y == 3 else 0)  # GC or GU
    elif x == 3:  # U
        return 4 if y == 2 else (6 if y == 0 else 0)  # UG or UA
    else:
        return 0

# NUC_TO_PAIR: Convert Vienna-encoded nucleotides to pair type (kept for compatibility)
# x,y are in Vienna encoding (1:A, 2:C, 3:G, 4:U, 0:N)
def NUC_TO_PAIR(x, y):
    if x == 1:
        return 5 if y == 4 else 0
    elif x == 2:
        return 1 if y == 3 else 0
    elif x == 3:
        return 2 if y == 2 else 3 if y == 4 else 0
    elif x == 4:
        return 4 if y == 3 else 6 if y == 1 else 0
    else:
        return 0

import math

from Utils.shared import * # lhuang

from Utils.energy_parameter import * # energy_parameter stuff  
from Utils.intl11 import *
from Utils.intl21 import *
from Utils.intl22 import *

MAXLOOP = 30

def MIN2(a, b):
    return a if a <= b else b

def MAX2(a, b):  
    return a if a >= b else b

def v_init_tetra_hex_tri(seq, seq_length, if_tetraloops, if_hexaloops, if_triloops):
    # TetraLoops
    if_tetraloops.extend([-1] * max(0, seq_length-5))
    for i in range(seq_length-5):
        if not (seq[i] == 'C' and seq[i+5] == 'G'):
            continue
        ts = seq[i:i+6]
        if ts in Tetraloops:
            if_tetraloops[i] = Tetraloops.index(ts)

    # Triloops
    if_triloops.extend([-1] * max(0, seq_length-4))
    for i in range(seq_length-4):
        if not ((seq[i] == 'C' and seq[i+4] == 'G') or (seq[i] == 'G' and seq[i+4] == 'C')):
            continue
        ts = seq[i:i+5]
        if ts in Triloops:
            if_triloops[i] = Triloops.index(ts)

    # Hexaloops
    if_hexaloops.extend([-1] * max(0, seq_length-7))
    for i in range(seq_length-7):
        if not (seq[i] == 'A' and seq[i+7] == 'U'):
            continue
        ts = seq[i:i+8]
        if ts in Hexaloops:
            if_hexaloops[i] = Hexaloops.index(ts)

def v_score_hairpin(i, j, nuci, nuci1, nucj_1, nucj, tetra_hex_tri_index=-1):
    size = j - i - 1
    type = NUM_TO_PAIR(nuci, nucj)
    # Convert to Vienna nucleotide numbering for mismatch lookup
    si1 = NUM_TO_NUC(nuci1)
    sj1 = NUM_TO_NUC(nucj_1)
    
    if size <= 30:
        energy = hairpin37[size]
    else:
        energy = hairpin37[30] + int(lxc37 * math.log(size / 30.0))
        
    if size < 3:
        return energy
    
    if SPECIAL_HP:
        if size == 4 and tetra_hex_tri_index > -1:
            return Tetraloop37[tetra_hex_tri_index]
        elif size == 6 and tetra_hex_tri_index > -1:
            return Hexaloop37[tetra_hex_tri_index]
        elif size == 3:
            if tetra_hex_tri_index > -1:
                return Triloop37[tetra_hex_tri_index]
            return energy + (TerminalAU37 if type > 2 else 0)
        
    energy += mismatchH37[type][si1][sj1]
    
    return energy

def v_score_single(i, j, p, q, nuci, nuci1, nucj_1, nucj, nucp_1, nucp, nucq, nucq1):
    # Convert to Vienna nucleotide numbering for mismatch lookup
    si1 = NUM_TO_NUC(nuci1)
    sj1 = NUM_TO_NUC(nucj_1)
    sp1 = NUM_TO_NUC(nucp_1)
    sq1 = NUM_TO_NUC(nucq1)
    type = NUM_TO_PAIR(nuci, nucj)
    type_2 = NUM_TO_PAIR(nucq, nucp)
    n1 = p - i - 1
    n2 = j - q - 1
    energy = 0
    
    if n1 > n2:
        nl = n1
        ns = n2
    else:
        nl = n2
        ns = n1
        
    if nl == 0:
        return stack37[type][type_2]
        
    if ns == 0:
        energy = bulge37[nl] if nl <= MAXLOOP else (bulge37[30] + int(lxc37 * math.log(nl / 30.0)))
        if nl == 1:
            energy += stack37[type][type_2]
        else:
            if type > 2:
                energy += TerminalAU37
            if type_2 > 2:
                energy += TerminalAU37
        return energy
    else:
        if ns == 1:
            if nl == 1:
                return int11_37[type][type_2][si1][sj1]
            if nl == 2:
                if n1 == 1:
                    energy = int21_37[type][type_2][si1][sq1][sj1]
                else:
                    energy = int21_37[type_2][type][sq1][si1][sp1]
                return energy
            else:
                energy = internal_loop37[nl+1] if nl+1 <= MAXLOOP else (internal_loop37[30] + int(lxc37 * math.log((nl+1) / 30.0)))
                energy += MIN2(MAX_NINIO, (nl-ns) * ninio37)
                energy += mismatch1nI37[type][si1][sj1] + mismatch1nI37[type_2][sq1][sp1]
                return energy
        elif ns == 2:
            if nl == 2:
                return int22_37[type][type_2][si1][sp1][sq1][sj1]
            elif nl == 3:
                energy = internal_loop37[5] + ninio37
                energy += mismatch23I37[type][si1][sj1] + mismatch23I37[type_2][sq1][sp1]
                return energy
                
        u = nl + ns
        energy = internal_loop37[u] if u <= MAXLOOP else (internal_loop37[30] + int(lxc37 * math.log(u / 30.0)))
        
        energy += MIN2(MAX_NINIO, (nl-ns) * ninio37)
        
        energy += mismatchI37[type][si1][sj1] + mismatchI37[type_2][sq1][sp1]
        
    return energy

def E_MLstem(type, si1, sj1):
    energy = 0
    
    if si1 >= 0 and sj1 >= 0:
        energy += mismatchM37[type][si1][sj1]
    elif si1 >= 0:
        energy += dangle5_37[type][si1]
    elif sj1 >= 0:
        energy += dangle3_37[type][sj1]
            
    if type > 2:
        energy += TerminalAU37
        
    energy += ML_intern37
    
    return energy





def v_score_M1(i, j, k, nuci_1, nuci, nuck, nuck1, len, dangle_model):
    tt = NUM_TO_PAIR(nuci, nuck)
    sp1 = NUM_TO_NUC(nuci_1)
    sq1 = NUM_TO_NUC(nuck1)
    
    return E_MLstem(tt, sp1, sq1)

def v_score_multi_unpaired(i, j):
    return 0

def v_score_multi(i, j, nuci, nuci1, nucj_1, nucj, len, dangle_model):
    tt = NUM_TO_PAIR(nucj, nuci) # lhuang: closing pair in multi: reversed
    si1 = NUM_TO_NUC(nuci1)
    sj1 = NUM_TO_NUC(nucj_1)
    
    return E_MLstem(tt, sj1, si1) + ML_closing37

def v_score_external_paired(i, j, nuci_1, nuci, nucj, nucj1, len, dangle_model):
    type = NUM_TO_PAIR(nuci, nucj)
    si1 = NUM_TO_NUC(nuci_1)
    sj1 = NUM_TO_NUC(nucj1)
    energy = 0
    
    if si1 >= 0 and sj1 >= 0:
        energy += mismatchExt37[type][si1][sj1]
    elif si1 >= 0:
        energy += dangle5_37[type][si1]
    elif sj1 >= 0:
        energy += dangle3_37[type][sj1]
            
    if type > 2:
        energy += TerminalAU37
        
    return energy

def v_score_external_unpaired(i, j):
    return 0