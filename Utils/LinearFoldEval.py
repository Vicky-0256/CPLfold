"""
LinearFoldEval.py
Evaluate the energy of a given RNA structure.

author: He Zhang
edited by: 12/2018
converted from C++ to Python by ChatGPT 06/2023
"""

from typing import List, Tuple
from collections import defaultdict

# from LinearFold import State, BeamCKYParser
from Utils.utility_v import *
from Utils.utility import *
import Utils.shared

def eval(seq: str, ref: str, is_verbose: bool, dangle_model: int = 2, lv: bool = True) -> int:
    seq_length = len(seq)

    if_tetraloops = []
    if_hexaloops = []
    if_triloops = []

    v_init_tetra_hex_tri(seq, seq_length, if_tetraloops, if_hexaloops, if_triloops)  # calculate if_tetraloops, if_hexaloops, if_triloops

    # Use appropriate encoding based on lv parameter
    if lv:
        eval_nucs = [Utils.shared.get_acgu_num_v(seq[i]) for i in range(seq_length)]
    else:
        eval_nucs = [Utils.shared.get_acgu_num_c(seq[i]) for i in range(seq_length)]

    total_energy = 0
    external_energy = 0
    M1_energy = [0] * seq_length
    multi_number_unpaired = [0] * seq_length

    stk = []  # stack of (index, page)
    inner_loop = (-1, -1)

    for j in range(seq_length):
        if ref[j] == '.':
            if stk:
                multi_number_unpaired[stk[-1][0]] += 1

        elif ref[j] == '(':
            if stk:  # +1 for outer loop page
                stk[-1] = (stk[-1][0], stk[-1][1] + 1)
            stk.append((j, 0))  # init page=0

        elif ref[j] == ')':
            assert stk
            i, page = stk.pop()

            nuci = eval_nucs[i]
            nucj = eval_nucs[j]
            nuci1 = eval_nucs[i + 1] if (i + 1) < seq_length else -1
            nucj_1 = eval_nucs[j - 1] if (j - 1) > -1 else -1
            nuci_1 = eval_nucs[i - 1] if (i - 1) > -1 else -1  # only for calculating v_score_M1
            nucj1 = eval_nucs[j + 1] if (j + 1) < seq_length else -1  # only for calculating v_score_M1

            if page == 0:  # hairpin
                tetra_hex_tri = -1
                if j - i - 1 == 4:  # 6:tetra
                    tetra_hex_tri = if_tetraloops[i]
                elif j - i - 1 == 6:  # 8:hexa
                    tetra_hex_tri = if_hexaloops[i]
                elif j - i - 1 == 3:  # 5:tri
                    tetra_hex_tri = if_triloops[i]

                if lv:
                    newscore = -v_score_hairpin(i, j, nuci, nuci1, nucj_1, nucj, tetra_hex_tri)
                    if is_verbose:
                        print(f"Hairpin loop ( {i+1}, {j+1}) {seq[i]}{seq[j]} : {newscore / -100.0:.2f}")
                else:
                    newscore = score_hairpin(i, j, nuci, nuci1, nucj_1, nucj)
                    if is_verbose:
                        print(f"Hairpin loop ( {i+1}, {j+1}) {seq[i]}{seq[j]} : {newscore:.2f}")
                total_energy += newscore

            elif page == 1:  # single
                p, q = inner_loop

                nucp_1 = eval_nucs[p - 1]
                nucp = eval_nucs[p]
                nucq = eval_nucs[q]
                nucq1 = eval_nucs[q + 1]
                nucp1 = eval_nucs[p + 1]
                nucq_1 = eval_nucs[q - 1]

                if lv:
                    newscore = -v_score_single(i, j, p, q, nuci, nuci1, nucj_1, nucj,
                                               nucp_1, nucp, nucq, nucq1)
                    if is_verbose:
                        print(f"Interior loop ( {i+1}, {j+1}) {seq[i]}{seq[j]}; ( {p+1}, {q+1}) {seq[p]}{seq[q]} : {newscore / -100.0:.2f}")
                else:
                    newscore = (score_junction_B(i, j, nuci, nuci1, nucj_1, nucj) +
                                score_junction_B(p, q, nucp, nucp1, nucq_1, nucq) +
                                score_single_without_junctionB(i, j, p, q, nuci_1, nuci, nucj, nucj1))
                    if is_verbose:
                        print(f"Interior loop ( {i+1}, {j+1}) {seq[i]}{seq[j]}; ( {p+1}, {q+1}) {seq[p]}{seq[q]} : {newscore:.2f}")
                total_energy += newscore

            else:  # multi
                multi_score = 0
                multi_score += M1_energy[i]
                if lv:
                    multi_score += -v_score_multi(i, j, nuci, nuci1, nucj_1, nucj, seq_length, dangle_model)
                    multi_score += -v_score_multi_unpaired(i + 1, i + multi_number_unpaired[i])
                    if is_verbose:
                        print(f"Multi loop ( {i+1}, {j+1}) {seq[i]}{seq[j]} : {multi_score / -100.0:.2f}")
                else:
                    multi_score += score_multi(i, j, nuci, nuci1, nucj_1, nucj, seq_length)
                    multi_score += score_multi_unpaired(i + 1, i + multi_number_unpaired[i])
                    if is_verbose:
                        print(f"Multi loop ( {i+1}, {j+1}) {seq[i]}{seq[j]} : {multi_score:.2f}")
                total_energy += multi_score

            # update inner_loop
            inner_loop = (i, j)

            # possible M
            if stk:
                if lv:
                    M1_energy[stk[-1][0]] += -v_score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length, dangle_model)
                else:
                    M1_energy[stk[-1][0]] += score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length)

            # check if adding external energy
            if not stk:
                k = i - 1
                nuck = eval_nucs[k] if k > -1 else -1
                nuck1 = eval_nucs[k + 1]
                if lv:
                    external_energy += -v_score_external_paired(k + 1, j, nuck, nuck1,
                                                                nucj, nucj1, seq_length, dangle_model)
                else:
                    external_energy += score_external_paired(k + 1, j, nuck, nuck1, nucj, nucj1, seq_length)

    if is_verbose:
        if lv:
            print(f"External loop : {external_energy / -100.0:.2f}")
        else:
            print(f"External loop : {external_energy:.2f}")
    total_energy += external_energy
    return total_energy