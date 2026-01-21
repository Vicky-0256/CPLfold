"""
CPLfold_parser.py - LinearFold Parser for CPLfold

Ultra-optimized LinearFold implementation with COMRADES/PARIS bonus matrix support.

Based on LinearFold:
- Source: https://github.com/LinearFold/LinearFold
- Reference: Huang et al. (2019) Bioinformatics 35(14):i295-i304

Features:
1. Flat 1D arrays instead of 3D (better cache locality)
2. Inlined scoring functions for zero call overhead
3. O(n log n) beam pruning with deterministic sorting
4. Vienna mode (lv=True) and CONTRAfold mode (lv=False)
5. Bonus matrix integration for experimental data (COMRADES/PARIS)
6. Constraint folding support for two-phase pseudoknot prediction

Key Methods:
- parse(seq, cons=None): Fold sequence with optional constraints
- parse_subopt(seq, energy_delta, max_structures): Generate suboptimal structures
- set_bonus_matrix(matrix, seq_len): Set COMRADES/PARIS bonus matrix
- set_alpha(alpha): Set bonus matrix scaling factor

Author: Ke Wang
"""

import sys
import time
import numpy as np
from numba import njit

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False

import Utils.shared
from Utils.feature_weight import (
    base_pair, helix_stacking, helix_closing, terminal_mismatch,
    dangle_left, dangle_right, hairpin_length, bulge_length,
    internal_length, internal_symmetric_length, internal_asymmetry,
    internal_explicit, bulge_0x1_nucleotides, internal_1x1_nucleotides,
    multi_base, multi_unpaired, multi_paired, external_unpaired, external_paired
)

# Vienna energy parameters
from Utils.energy_parameter import (
    stack37, hairpin37, bulge37, internal_loop37,
    mismatchI37, mismatchH37, mismatchM37, mismatchExt37,
    mismatch1nI37, mismatch23I37,
    dangle5_37, dangle3_37, TerminalAU37,
    ML_intern37, ML_closing37, lxc37, ninio37, MAX_NINIO,
    Tetraloop37, Hexaloop37, Triloop37,
    Tetraloops, Hexaloops, Triloops, SPECIAL_HP
)
from Utils.intl11 import int11_37
from Utils.intl21 import int21_37
from Utils.intl22 import int22_37
from Utils.utility_v import v_init_tetra_hex_tri, NUM_TO_PAIR, NUM_TO_NUC

# ============================================================================
# PARIS Bonus Preprocessing Functions
# ============================================================================

@njit(cache=True, fastmath=True)
def diagonal_smooth_matrix(matrix, n, window=2):
    """
    对 PARIS 支持度矩阵沿对角线方向平滑
    鼓励形成连续的茎结构

    Args:
        matrix: n x n 的扁平化矩阵 (长度 n*n)
        n: 序列长度
        window: 平滑窗口大小 (向上下各延伸 window 个位置)

    Returns:
        smoothed: 平滑后的矩阵
    """
    smoothed = np.zeros(n * n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 4, n):  # j > i + 3 (hairpin constraint)
            # 沿对角线收集值 (i-k, j+k) 和 (i+k, j-k)
            total = np.float32(0.0)
            count = 0

            for k in range(-window, window + 1):
                ni, nj = i + k, j - k
                if 0 <= ni < n and ni + 4 <= nj < n:
                    total += matrix[ni * n + nj]
                    count += 1

            if count > 0:
                smoothed[i * n + j] = total / count
            else:
                smoothed[i * n + j] = matrix[i * n + j]

    return smoothed


@njit(cache=True, fastmath=True)
def compute_stack_bonus_matrix(pair_matrix, n, method=0):
    """
    计算 stack bonus 矩阵

    Args:
        pair_matrix: n x n 的扁平化 PARIS 支持度矩阵
        n: 序列长度
        method: 聚合方法
            0 = min(支持度)  - 保守，要求两个pair都有信号
            1 = mean(支持度) - 平衡
            2 = max(支持度)  - 激进
            3 = geometric_mean - 几何平均

    Returns:
        stack_bonus: stack bonus 矩阵，stack_bonus[i,j] 表示在 (i,j) 上形成 stack 的 bonus
                     即 (i,j) 和 (i+1, j-1) 配对时的 bonus
    """
    stack_bonus = np.zeros(n * n, dtype=np.float32)

    for i in range(n - 1):
        for j in range(i + 5, n):  # j > i + 4 保证 (i+1, j-1) 有效
            # (i, j) 和 (i+1, j-1) 形成 stack
            support_ij = pair_matrix[i * n + j]
            support_inner = pair_matrix[(i + 1) * n + (j - 1)]

            if method == 0:  # min
                stack_bonus[i * n + j] = min(support_ij, support_inner)
            elif method == 1:  # mean
                stack_bonus[i * n + j] = (support_ij + support_inner) / 2
            elif method == 2:  # max
                stack_bonus[i * n + j] = max(support_ij, support_inner)
            else:  # geometric mean
                if support_ij > 0 and support_inner > 0:
                    stack_bonus[i * n + j] = np.sqrt(support_ij * support_inner)
                else:
                    stack_bonus[i * n + j] = 0.0

    return stack_bonus


@njit(cache=True, fastmath=True)
def compute_helix_bonus_matrix(pair_matrix, n, min_helix_len=2, window=1):
    """
    计算 helix-level bonus 矩阵
    只对能形成足够长茎的位置给予 bonus

    Args:
        pair_matrix: n x n 的扁平化 PARIS 支持度矩阵
        n: 序列长度
        min_helix_len: 最小茎长度要求
        window: 检测窗口

    Returns:
        helix_bonus: helix bonus 矩阵
    """
    helix_bonus = np.zeros(n * n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 4, n):
            # 检查沿对角线方向是否有连续的高支持度
            # 计算以 (i,j) 为起点/终点的潜在茎长度

            # 向内延伸 (i+k, j-k)
            inner_support = np.float32(0.0)
            inner_count = 0
            for k in range(min_helix_len):
                ni, nj = i + k, j - k
                if ni < nj - 3 and ni < n and nj >= 0:
                    inner_support += pair_matrix[ni * n + nj]
                    inner_count += 1
                else:
                    break

            # 只有当能形成足够长的茎时才给 bonus
            if inner_count >= min_helix_len:
                # 使用平均值作为 bonus
                helix_bonus[i * n + j] = inner_support / inner_count
            # 否则 bonus 为 0，抑制孤立的碱基对

    return helix_bonus


# Bonus mode constants
BONUS_MODE_PAIR = 0    # 原始单对 bonus
BONUS_MODE_STACK = 1   # Stack-level bonus
BONUS_MODE_HELIX = 2   # Helix-level bonus with minimum length


# Constants
NOTON = 5
NOTOND = 25
NOTONT = 125
SINGLE_MAX_LEN = 30
HAIRPIN_MAX_LEN = 30

VALUE_MIN_FLOAT = np.float64(-1e18)
VALUE_MIN_INT = np.int32(-2147483647)

# Manner constants
MANNER_NONE = 0
MANNER_H = 1
MANNER_HAIRPIN = 2
MANNER_SINGLE = 3
MANNER_HELIX = 4
MANNER_MULTI = 5
MANNER_MULTI_eq_MULTI_plus_U = 6
MANNER_P_eq_MULTI = 7
MANNER_M2_eq_M_plus_P = 8
MANNER_M_eq_M2 = 9
MANNER_M_eq_M_plus_U = 10
MANNER_M_eq_P = 11
MANNER_C_eq_C_plus_U = 12
MANNER_C_eq_C_plus_P = 13

# State types
STATE_H = 0
STATE_P = 1
STATE_M2 = 2
STATE_MULTI = 3
STATE_M = 4
NUM_STATE_TYPES = 5

# ============================================================================
# CONTRAfold feature weights as numpy arrays
# ============================================================================
_base_pair = np.array(base_pair, dtype=np.float64)
_helix_stacking = np.array(helix_stacking, dtype=np.float64)
_helix_closing = np.array(helix_closing, dtype=np.float64)
_terminal_mismatch = np.array(terminal_mismatch, dtype=np.float64)
_dangle_left = np.array(dangle_left, dtype=np.float64)
_dangle_right = np.array(dangle_right, dtype=np.float64)
_hairpin_length = np.array(hairpin_length, dtype=np.float64)
_bulge_length = np.array(bulge_length, dtype=np.float64)
_internal_length = np.array(internal_length, dtype=np.float64)
_internal_symmetric_length = np.array(internal_symmetric_length, dtype=np.float64)
_internal_asymmetry = np.array(internal_asymmetry, dtype=np.float64)
_internal_explicit = np.array(internal_explicit, dtype=np.float64)
_bulge_0x1_nucleotides = np.array(bulge_0x1_nucleotides, dtype=np.float64)
_internal_1x1_nucleotides = np.array(internal_1x1_nucleotides, dtype=np.float64)

_multi_base = np.float64(multi_base)
_multi_unpaired = np.float64(multi_unpaired)
_multi_paired = np.float64(multi_paired)
_external_unpaired = np.float64(external_unpaired)
_external_paired = np.float64(external_paired)

# Pre-compute cache_single for CONTRAfold
_cache_single = np.zeros((SINGLE_MAX_LEN + 1, SINGLE_MAX_LEN + 1), dtype=np.float64)

def _init_cache_single():
    for l1 in range(SINGLE_MAX_LEN + 1):
        for l2 in range(SINGLE_MAX_LEN + 1):
            if l1 == 0 and l2 == 0:
                continue
            if l1 == 0:
                _cache_single[l1, l2] = _bulge_length[min(l2, 30)]
            elif l2 == 0:
                _cache_single[l1, l2] = _bulge_length[min(l1, 30)]
            else:
                _cache_single[l1, l2] = _internal_length[min(l1 + l2, 30)]
                if l1 <= 4 and l2 <= 4:
                    idx = (l1 * 4 + l2) if l1 <= l2 else (l2 * 4 + l1)
                    if idx < len(_internal_explicit):
                        _cache_single[l1, l2] += _internal_explicit[idx]
                if l1 == l2:
                    _cache_single[l1, l2] += _internal_symmetric_length[min(l1, 15)]
                else:
                    _cache_single[l1, l2] += _internal_asymmetry[min(abs(l1 - l2), 28)]

_init_cache_single()

# ============================================================================
# Vienna energy parameters as numpy arrays
# ============================================================================
_v_stack = np.array(stack37, dtype=np.int32)
_v_hairpin = np.array(hairpin37, dtype=np.int32)
_v_bulge = np.array(bulge37, dtype=np.int32)
_v_internal = np.array(internal_loop37, dtype=np.int32)
_v_mismatchI = np.array(mismatchI37, dtype=np.int32)
_v_mismatchH = np.array(mismatchH37, dtype=np.int32)
_v_mismatchM = np.array(mismatchM37, dtype=np.int32)
_v_mismatchExt = np.array(mismatchExt37, dtype=np.int32)
_v_mismatch1nI = np.array(mismatch1nI37, dtype=np.int32)
_v_mismatch23I = np.array(mismatch23I37, dtype=np.int32)
_v_dangle5 = np.array(dangle5_37, dtype=np.int32)
_v_dangle3 = np.array(dangle3_37, dtype=np.int32)
_v_int11 = np.array(int11_37, dtype=np.int32)
_v_int21 = np.array(int21_37, dtype=np.int32)
_v_int22 = np.array(int22_37, dtype=np.int32)
_v_tetra = np.array(Tetraloop37, dtype=np.int32)
_v_hexa = np.array(Hexaloop37, dtype=np.int32)
_v_tri = np.array(Triloop37, dtype=np.int32)

_v_terminalAU = np.int32(TerminalAU37)
_v_ml_intern = np.int32(ML_intern37)
_v_ml_closing = np.int32(ML_closing37)
_v_lxc = np.float64(lxc37)
_v_ninio = np.int32(ninio37)
_v_max_ninio = np.int32(MAX_NINIO)

# Pre-compute pair type lookup table
_pair_type_table = np.zeros(16, dtype=np.int32)
_pair_type_table[0 * 4 + 3] = 5  # A-U
_pair_type_table[1 * 4 + 2] = 1  # C-G
_pair_type_table[2 * 4 + 1] = 2  # G-C
_pair_type_table[2 * 4 + 3] = 3  # G-U
_pair_type_table[3 * 4 + 2] = 4  # U-G
_pair_type_table[3 * 4 + 0] = 6  # U-A

# Vienna nuc conversion table
_nuc_table = np.array([1, 2, 3, 4, 0], dtype=np.int32)

# ============================================================================
# Inlined helper functions
# ============================================================================

@njit(cache=True, fastmath=True, inline='always')
def get_pair_type(nuci, nucj):
    if nuci < 0 or nucj < 0 or nuci > 3 or nucj > 3:
        return 0
    return _pair_type_table[nuci * 4 + nucj]

@njit(cache=True, fastmath=True, inline='always')
def get_vienna_nuc(nuc):
    if nuc < 0 or nuc > 4:
        return -1
    return _nuc_table[nuc]

@njit(cache=True, fastmath=True, inline='always')
def flat_idx(state, j, i, n):
    return state * n * n + j * n + i

# ============================================================================
# Vienna scoring functions (inlined)
# ============================================================================

@njit(cache=True, fastmath=True, inline='always')
def v_score_hairpin(i, j, nuci, nuci1, nucj_1, nucj, tetra_idx,
                    hairpin_arr, mismatchH_arr, terminalAU, tetra_arr, hexa_arr, tri_arr):
    size = j - i - 1
    pt = get_pair_type(nuci, nucj)

    if size <= 30:
        energy = hairpin_arr[size]
    else:
        energy = hairpin_arr[30] + np.int32(107.856 * np.log(size / 30.0))

    if size < 3:
        return -energy

    if size == 4 and tetra_idx >= 0:
        return -tetra_arr[tetra_idx]
    if size == 6 and tetra_idx >= 0:
        return -hexa_arr[tetra_idx]
    if size == 3:
        if tetra_idx >= 0:
            return -tri_arr[tetra_idx]
        if pt > 2:
            energy += terminalAU
        return -energy

    si1 = get_vienna_nuc(nuci1)
    sj1 = get_vienna_nuc(nucj_1)
    if si1 >= 0 and sj1 >= 0:
        energy += mismatchH_arr[pt, si1, sj1]

    return -energy

@njit(cache=True, fastmath=True, inline='always')
def v_score_single(i, j, p, q, nuci, nuci1, nucj_1, nucj, nucp_1, nucp, nucq, nucq1,
                   stack_arr, bulge_arr, internal_arr,
                   int11_arr, int21_arr, int22_arr,
                   mismatchI_arr, mismatch1nI_arr, mismatch23I_arr,
                   terminalAU, ninio, max_ninio):
    pt1 = get_pair_type(nuci, nucj)
    pt2 = get_pair_type(nucq, nucp)
    n1 = p - i - 1
    n2 = j - q - 1

    if n1 > n2:
        nl, ns = n1, n2
    else:
        nl, ns = n2, n1

    if nl == 0:
        return -stack_arr[pt1, pt2]

    if ns == 0:
        if nl <= 30:
            energy = bulge_arr[nl]
        else:
            energy = bulge_arr[30] + np.int32(107.856 * np.log(nl / 30.0))
        if nl == 1:
            energy += stack_arr[pt1, pt2]
        else:
            if pt1 > 2:
                energy += terminalAU
            if pt2 > 2:
                energy += terminalAU
        return -energy

    si1 = get_vienna_nuc(nuci1)
    sj1 = get_vienna_nuc(nucj_1)
    sp1 = get_vienna_nuc(nucp_1)
    sq1 = get_vienna_nuc(nucq1)

    if ns == 1 and nl == 1:
        if si1 >= 0 and sj1 >= 0:
            return -int11_arr[pt1, pt2, si1, sj1]
        return 0

    if ns == 1 and nl == 2:
        if n1 == 1:
            if si1 >= 0 and sq1 >= 0 and sj1 >= 0:
                return -int21_arr[pt1, pt2, si1, sq1, sj1]
        else:
            if sq1 >= 0 and si1 >= 0 and sp1 >= 0:
                return -int21_arr[pt2, pt1, sq1, si1, sp1]
        return 0

    if ns == 2 and nl == 2:
        if si1 >= 0 and sp1 >= 0 and sq1 >= 0 and sj1 >= 0:
            return -int22_arr[pt1, pt2, si1, sp1, sq1, sj1]
        return 0

    if ns == 1:
        u = nl + 1
        if u <= 30:
            energy = internal_arr[u]
        else:
            energy = internal_arr[30] + np.int32(107.856 * np.log(u / 30.0))
        energy += min(max_ninio, (nl - ns) * ninio)
        if si1 >= 0 and sj1 >= 0:
            energy += mismatch1nI_arr[pt1, si1, sj1]
        if sq1 >= 0 and sp1 >= 0:
            energy += mismatch1nI_arr[pt2, sq1, sp1]
        return -energy

    if ns == 2 and nl == 3:
        energy = internal_arr[5] + ninio
        if si1 >= 0 and sj1 >= 0:
            energy += mismatch23I_arr[pt1, si1, sj1]
        if sq1 >= 0 and sp1 >= 0:
            energy += mismatch23I_arr[pt2, sq1, sp1]
        return -energy

    u = nl + ns
    if u <= 30:
        energy = internal_arr[u]
    else:
        energy = internal_arr[30] + np.int32(107.856 * np.log(u / 30.0))
    energy += min(max_ninio, (nl - ns) * ninio)
    if si1 >= 0 and sj1 >= 0:
        energy += mismatchI_arr[pt1, si1, sj1]
    if sq1 >= 0 and sp1 >= 0:
        energy += mismatchI_arr[pt2, sq1, sp1]
    return -energy

@njit(cache=True, fastmath=True, inline='always')
def v_score_multi(nuci, nuci1, nucj_1, nucj,
                  mismatchM_arr, dangle5_arr, dangle3_arr, terminalAU, ml_intern, ml_closing):
    """Multi-loop closing score (matching C's v_score_multi).

    Note: For multi-loop closing, pair type is REVERSED: (nucj, nuci).
    Uses E_MLstem logic with reversed flanking nucleotides.
    C does NOT use dangles for multi-loops (commented out in E_MLstem).
    """
    pt = get_pair_type(nucj, nuci)  # Reversed pair for multi-loop closing
    si1 = get_vienna_nuc(nuci1)
    sj1 = get_vienna_nuc(nucj_1)
    energy = np.int32(0)
    # Only use mismatchM when both neighbors present (matching C's E_MLstem)
    # Note: indices are [pt, sj1, si1] because flanking nucleotides are also reversed
    if si1 >= 0 and sj1 >= 0:
        energy += mismatchM_arr[pt, sj1, si1]
    # No dangle fallback - matching C behavior
    if pt > 2:
        energy += terminalAU
    energy += ml_intern + ml_closing
    return -energy

@njit(cache=True, fastmath=True, inline='always')
def v_score_M1(nuci_1, nuci, nucj, nucj1,
               mismatchM_arr, dangle5_arr, dangle3_arr, terminalAU, ml_intern):
    """Multi-loop stem score (matching C's E_MLstem).

    Note: C does NOT use dangles for multi-loop stems (commented out in E_MLstem).
    Only uses mismatchM when both neighbors are present.
    """
    pt = get_pair_type(nuci, nucj)
    si1 = get_vienna_nuc(nuci_1)
    sj1 = get_vienna_nuc(nucj1)
    energy = np.int32(0)
    # Only use mismatchM when both neighbors present (matching C's E_MLstem)
    # C code has dangles commented out: "multi dangles are never used"
    if si1 >= 0 and sj1 >= 0:
        energy += mismatchM_arr[pt, si1, sj1]
    # No dangle fallback - matching C behavior
    if pt > 2:
        energy += terminalAU
    energy += ml_intern
    return -energy

@njit(cache=True, fastmath=True, inline='always')
def v_score_external(nuci_1, nuci, nucj, nucj1,
                     mismatchExt_arr, dangle5_arr, dangle3_arr, terminalAU):
    pt = get_pair_type(nuci, nucj)
    si1 = get_vienna_nuc(nuci_1)
    sj1 = get_vienna_nuc(nucj1)
    energy = np.int32(0)
    if si1 >= 0 and sj1 >= 0:
        energy += mismatchExt_arr[pt, si1, sj1]
    elif si1 >= 0:
        energy += dangle5_arr[pt, si1]
    elif sj1 >= 0:
        energy += dangle3_arr[pt, sj1]
    if pt > 2:
        energy += terminalAU
    return -energy

# ============================================================================
# CONTRAfold scoring functions (inlined)
# ============================================================================

@njit(cache=True, fastmath=True, inline='always')
def score_base_pair(nuci, nucj, base_pair_arr):
    return base_pair_arr[nucj * NOTON + nuci]

@njit(cache=True, fastmath=True, inline='always')
def score_helix_stacking(nuci, nuci1, nucj_1, nucj, helix_stacking_arr):
    return helix_stacking_arr[nuci * NOTONT + nucj * NOTOND + nuci1 * NOTON + nucj_1]

@njit(cache=True, fastmath=True, inline='always')
def score_helix_closing(nuci, nucj, helix_closing_arr):
    return helix_closing_arr[nuci * NOTON + nucj]

@njit(cache=True, fastmath=True, inline='always')
def score_terminal_mismatch(nuci, nuci1, nucj_1, nucj, terminal_mismatch_arr):
    return terminal_mismatch_arr[nuci * NOTONT + nucj * NOTOND + nuci1 * NOTON + nucj_1]

@njit(cache=True, fastmath=True, inline='always')
def score_dangle_left(nuci, nuci1, nucj, dangle_left_arr):
    return dangle_left_arr[nuci * NOTOND + nucj * NOTON + nuci1]

@njit(cache=True, fastmath=True, inline='always')
def score_dangle_right(nuci, nucj_1, nucj, dangle_right_arr):
    return dangle_right_arr[nuci * NOTOND + nucj * NOTON + nucj_1]

# ============================================================================
# Beam pruning with deterministic sorting
# ============================================================================

@njit(cache=True)
def beam_prune_deterministic(beam_arr, beam_scores, count, k):
    if count <= k:
        return count
    sorted_scores = np.sort(beam_scores[:count])[::-1]
    threshold = sorted_scores[k - 1]
    new_count = 0
    temp_arr = beam_arr[:count].copy()
    temp_scores = beam_scores[:count].copy()
    # Keep candidates >= threshold, but limit to k to prevent buffer overflow
    # When multiple candidates have the same threshold score, we must cap at k
    for idx in range(count):
        if temp_scores[idx] >= threshold and new_count < k:
            beam_arr[new_count] = temp_arr[idx]
            beam_scores[new_count] = temp_scores[idx]
            new_count += 1
    return new_count

@njit(cache=True)
def sort_by_position_desc(beam_arr, count):
    if count <= 1:
        return
    sorted_idx = np.argsort(beam_arr[:count])[::-1]
    temp = beam_arr[:count].copy()
    for k in range(count):
        beam_arr[k] = temp[sorted_idx[k]]

# ============================================================================
# Main parsing loop - Vienna mode
# ============================================================================

@njit(cache=True, fastmath=True)
def parse_loop_vienna(
    n, beam, nucs, allowed_pairs, next_pair, no_sharp_turn,
    scores, manners, splits, l1s, l2s,
    c_scores, c_manners, c_splits,
    beam_arr, beam_scores,
    m_beam_idx, m_beam_count,
    if_tetra, if_hexa, if_tri,
    v_hairpin, v_mismatchH, v_terminalAU, v_tetra, v_hexa, v_tri,
    v_stack, v_bulge, v_internal,
    v_int11, v_int21, v_int22,
    v_mismatchI, v_mismatch1nI, v_mismatch23I,
    v_ninio, v_max_ninio,
    v_mismatchM, v_mismatchExt, v_dangle5, v_dangle3,
    v_ml_intern, v_ml_closing,
    use_constraints, cons_arr, allow_unpaired_position, allow_unpaired_range,
    bonus_matrix, alpha_scaled
):
    n2 = n * n
    value_min = VALUE_MIN_FLOAT

    # Initialize C scores (with constraint awareness)
    if n > 0:
        # Only initialize C[0] as unpaired if position 0 can be unpaired
        if not use_constraints or allow_unpaired_position[0] != 0:
            c_scores[0] = np.float64(0.0)
            c_manners[0] = MANNER_C_eq_C_plus_U
    if n > 1:
        # Only initialize C[1] as unpaired if positions 0 and 1 can both be unpaired
        if not use_constraints or (allow_unpaired_position[0] != 0 and allow_unpaired_position[1] != 0):
            c_scores[1] = np.float64(0.0)
            c_manners[1] = MANNER_C_eq_C_plus_U

    for j in range(n):
        nucj = nucs[j]
        nucj1 = nucs[j + 1] if j + 1 < n else -1
        nucj_1 = nucs[j - 1] if j > 0 else -1

        # Collect H beam
        h_count = 0
        for i in range(j + 1):
            idx = STATE_H * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[h_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[h_count] = prefix + scores[idx]
                h_count += 1

        h_count = beam_prune_deterministic(beam_arr, beam_scores, h_count, beam)
        sort_by_position_desc(beam_arr, h_count)

        # Hairpin initiation
        jnext = next_pair[nucj, j]
        if no_sharp_turn:
            while jnext != -1 and jnext - j < 4:
                jnext = next_pair[nucj, jnext]

        # Constraint check for hairpin initiation
        if use_constraints:
            if allow_unpaired_position[j] == 0:
                # j must be left bracket of a forced pair, jump to cons[j] directly
                jnext = cons_arr[j] if cons_arr[j] > j else -1
            if jnext != -1:
                nucjnext = nucs[jnext]
                # Check if jnext crosses a constrained bracket or pair not allowed
                if jnext > allow_unpaired_range[j]:
                    jnext = -1
                elif not ((cons_arr[j] == -1 or cons_arr[j] == jnext) and
                          (cons_arr[jnext] == -1 or cons_arr[jnext] == j) and
                          allowed_pairs[nucj, nucjnext]):
                    jnext = -1

        if jnext != -1 and jnext < n:
            nucjnext = nucs[jnext]
            nucjnext_1 = nucs[jnext - 1] if jnext > 0 else -1
            tetra_idx = -1
            length = jnext - j - 1
            if length == 4:
                tetra_idx = if_tetra[j]
            elif length == 6:
                tetra_idx = if_hexa[j]
            elif length == 3:
                tetra_idx = if_tri[j]

            newscore = np.float64(v_score_hairpin(j, jnext, nucj, nucj1, nucjnext_1, nucjnext, tetra_idx,
                                       v_hairpin, v_mismatchH, v_terminalAU, v_tetra, v_hexa, v_tri))
            # Add PARIS bonus
            if alpha_scaled != 0.0:
                newscore += bonus_matrix[j * n + jnext] * alpha_scaled

            idx = STATE_H * n2 + jnext * n + j
            if scores[idx] < newscore:
                scores[idx] = newscore
                manners[idx] = MANNER_H

        # Process H states
        for hidx in range(h_count):
            i = beam_arr[hidx]
            h_idx = STATE_H * n2 + j * n + i
            h_score = scores[h_idx]

            p_idx = STATE_P * n2 + j * n + i
            if scores[p_idx] < h_score:
                scores[p_idx] = h_score
                manners[p_idx] = MANNER_HAIRPIN

            nuci = nucs[i]
            jnext_i = next_pair[nuci, j]

            # Constraint check for H state extension (Vienna)
            if jnext_i != -1 and use_constraints:
                nucjnext_check = nucs[jnext_i]
                # Check if jnext_i crosses a constrained bracket or pair not allowed
                if jnext_i > allow_unpaired_range[i]:
                    jnext_i = -1
                elif not ((cons_arr[i] == -1 or cons_arr[i] == jnext_i) and
                          (cons_arr[jnext_i] == -1 or cons_arr[jnext_i] == i) and
                          allowed_pairs[nuci, nucjnext_check]):
                    jnext_i = -1

            if jnext_i != -1 and jnext_i < n:
                nuci1 = nucs[i + 1] if i + 1 < n else -1
                nucjnext = nucs[jnext_i]
                nucjnext_1 = nucs[jnext_i - 1] if jnext_i > 0 else -1
                length = jnext_i - i - 1
                tetra_idx = -1
                if length == 4:
                    tetra_idx = if_tetra[i]
                elif length == 6:
                    tetra_idx = if_hexa[i]
                elif length == 3:
                    tetra_idx = if_tri[i]

                newscore = np.float64(v_score_hairpin(i, jnext_i, nuci, nuci1, nucjnext_1, nucjnext, tetra_idx,
                                           v_hairpin, v_mismatchH, v_terminalAU, v_tetra, v_hexa, v_tri))
                # Add PARIS bonus
                if alpha_scaled != 0.0:
                    newscore += bonus_matrix[i * n + jnext_i] * alpha_scaled

                idx = STATE_H * n2 + jnext_i * n + i
                if scores[idx] < newscore:
                    scores[idx] = newscore
                    manners[idx] = MANNER_H

        if j == 0:
            continue

        # Collect Multi beam
        multi_count = 0
        for i in range(j + 1):
            idx = STATE_MULTI * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[multi_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[multi_count] = prefix + scores[idx]
                multi_count += 1

        multi_count = beam_prune_deterministic(beam_arr, beam_scores, multi_count, beam)
        sort_by_position_desc(beam_arr, multi_count)

        # Process Multi states
        for midx in range(multi_count):
            i = beam_arr[midx]
            m_idx = STATE_MULTI * n2 + j * n + i
            multi_score = scores[m_idx]
            nuci = nucs[i]
            nuci1 = nucs[i + 1] if i + 1 < n else -1

            newscore = multi_score + v_score_multi(nuci, nuci1, nucj_1, nucj,
                                                    v_mismatchM, v_dangle5, v_dangle3,
                                                    v_terminalAU, v_ml_intern, v_ml_closing)
            # Add PARIS bonus for multi-loop closing pair (i, j)
            if alpha_scaled != 0.0:
                newscore += bonus_matrix[i * n + j] * alpha_scaled

            p_idx = STATE_P * n2 + j * n + i
            if scores[p_idx] < newscore:
                scores[p_idx] = newscore
                manners[p_idx] = MANNER_P_eq_MULTI

            jnext_i = next_pair[nuci, j]

            # Constraint check for Multi state extension (Vienna)
            if jnext_i != -1 and use_constraints:
                nucjnext_check = nucs[jnext_i]
                # Check if jnext_i crosses a constrained bracket or pair not allowed
                if jnext_i > allow_unpaired_range[j]:
                    jnext_i = -1
                elif not ((cons_arr[i] == -1 or cons_arr[i] == jnext_i) and
                          (cons_arr[jnext_i] == -1 or cons_arr[jnext_i] == i) and
                          allowed_pairs[nuci, nucjnext_check]):
                    jnext_i = -1

            if jnext_i != -1 and jnext_i < n:
                idx = STATE_MULTI * n2 + jnext_i * n + i
                if scores[idx] < multi_score:
                    scores[idx] = multi_score
                    manners[idx] = MANNER_MULTI_eq_MULTI_plus_U
                    l1s[idx] = l1s[m_idx]
                    l2s[idx] = l2s[m_idx] + jnext_i - j

        # Collect P beam
        p_count = 0
        for i in range(j + 1):
            idx = STATE_P * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[p_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[p_count] = prefix + scores[idx]
                p_count += 1

        p_count = beam_prune_deterministic(beam_arr, beam_scores, p_count, beam)
        sort_by_position_desc(beam_arr, p_count)

        # Process P states
        for pidx in range(p_count):
            i = beam_arr[pidx]
            p_idx_ij = STATE_P * n2 + j * n + i
            p_score = scores[p_idx_ij]
            nuci = nucs[i]
            nuci_1 = nucs[i - 1] if i > 0 else -1

            if i > 0 and j < n - 6:
                newscore = v_score_M1(nuci_1, nuci, nucj, nucj1,
                                      v_mismatchM, v_dangle5, v_dangle3,
                                      v_terminalAU, v_ml_intern) + p_score
                m_idx = STATE_M * n2 + j * n + i
                if scores[m_idx] < newscore:
                    scores[m_idx] = newscore
                    manners[m_idx] = MANNER_M_eq_P

            if i > 1:
                M1_score = v_score_M1(nuci_1, nuci, nucj, nucj1,
                                      v_mismatchM, v_dangle5, v_dangle3,
                                      v_terminalAU, v_ml_intern) + p_score
                k = i - 1
                m_cnt = m_beam_count[k]
                for m_idx_k in range(m_cnt):
                    newi = m_beam_idx[k, m_idx_k]
                    m_score = scores[STATE_M * n2 + k * n + newi]
                    if m_score > value_min:
                        newscore = M1_score + m_score
                        m2_idx = STATE_M2 * n2 + j * n + newi
                        if scores[m2_idx] < newscore:
                            scores[m2_idx] = newscore
                            manners[m2_idx] = MANNER_M2_eq_M_plus_P
                            splits[m2_idx] = k

            ext_score = v_score_external(nuci_1, nuci, nucj, nucj1,
                                         v_mismatchExt, v_dangle5, v_dangle3, v_terminalAU)
            k = i - 1
            if k >= 0:
                newscore = ext_score + c_scores[k] + p_score
                if c_scores[j] < newscore:
                    c_scores[j] = newscore
                    c_manners[j] = MANNER_C_eq_C_plus_P
                    c_splits[j] = k
            else:
                ext_score_0 = v_score_external(-1, nuci, nucj, nucj1,
                                               v_mismatchExt, v_dangle5, v_dangle3, v_terminalAU)
                newscore = ext_score_0 + p_score
                if c_scores[j] < newscore:
                    c_scores[j] = newscore
                    c_manners[j] = MANNER_C_eq_C_plus_P
                    c_splits[j] = -1

            # SINGLE/HELIX
            p_min = max(i - SINGLE_MAX_LEN, 0)
            for p in range(i - 1, p_min - 1, -1):
                # Constraint check for single/helix - outer loop
                if use_constraints:
                    # if p+1 must be paired, break
                    if p < i - 1 and allow_unpaired_position[p + 1] == 0:
                        break
                    # if p must be paired, p must be left bracket
                    if allow_unpaired_position[p] == 0:
                        q = cons_arr[p]
                        if q < p:
                            break
                        # q must be > j for outer pair to enclose inner pair
                        if q <= j:
                            continue
                        # Force q to this constrained pair
                    else:
                        q = -1  # will be set below

                nucp = nucs[p]
                nucp1 = nucs[p + 1]
                if not use_constraints or allow_unpaired_position[p] != 0:
                    q = next_pair[nucp, j]
                l1 = i - p - 1
                max_l2 = SINGLE_MAX_LEN - l1

                while q != -1 and q < n and q - j - 1 <= max_l2:
                    # Constraint check for single/helix - inner loop
                    if use_constraints:
                        # if q-1 must be paired, break
                        if q > j + 1 and q > allow_unpaired_range[j]:
                            break
                        # if p q are not allowed to pair, break
                        nucq = nucs[q]
                        if not ((cons_arr[p] == -1 or cons_arr[p] == q) and
                                (cons_arr[q] == -1 or cons_arr[q] == p) and
                                allowed_pairs[nucp, nucq]):
                            break

                    nucq = nucs[q]
                    nucq_1 = nucs[q - 1] if q > 0 else -1
                    nucq1 = nucs[q + 1] if q + 1 < n else -1
                    nucp_1 = nucs[p - 1] if p > 0 else -1

                    newscore = v_score_single(p, q, i, j, nucp, nucp1, nucq_1, nucq,
                                              nuci_1, nuci, nucj, nucj1,
                                              v_stack, v_bulge, v_internal,
                                              v_int11, v_int21, v_int22,
                                              v_mismatchI, v_mismatch1nI, v_mismatch23I,
                                              v_terminalAU, v_ninio, v_max_ninio) + p_score

                    # Add PARIS bonus for the new base pair (p, q)
                    if alpha_scaled != 0.0:
                        newscore += bonus_matrix[p * n + q] * alpha_scaled

                    target_idx = STATE_P * n2 + q * n + p
                    if scores[target_idx] < newscore:
                        scores[target_idx] = newscore
                        if p == i - 1 and q == j + 1:
                            manners[target_idx] = MANNER_HELIX
                            l1s[target_idx] = 0
                            l2s[target_idx] = 0
                        else:
                            manners[target_idx] = MANNER_SINGLE
                            l1s[target_idx] = i - p
                            l2s[target_idx] = q - j

                    q = next_pair[nucp, q]

        # Collect M2 beam
        m2_count = 0
        for i in range(j + 1):
            idx = STATE_M2 * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[m2_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[m2_count] = prefix + scores[idx]
                m2_count += 1

        m2_count = beam_prune_deterministic(beam_arr, beam_scores, m2_count, beam)
        sort_by_position_desc(beam_arr, m2_count)

        # Process M2 states
        for m2idx in range(m2_count):
            i = beam_arr[m2idx]
            m2_idx = STATE_M2 * n2 + j * n + i
            m2_score = scores[m2_idx]

            m_idx = STATE_M * n2 + j * n + i
            if scores[m_idx] < m2_score:
                scores[m_idx] = m2_score
                manners[m_idx] = MANNER_M_eq_M2

            p_min = max(i - SINGLE_MAX_LEN, 0)
            for p in range(i - 1, p_min - 1, -1):
                nucp = nucs[p]
                q = next_pair[nucp, j]

                # Constraint check for Multi state creation
                if use_constraints:
                    # If p+1 must be paired, break (can't have unpaired between p+1 and i)
                    if p < i - 1 and allow_unpaired_position[p + 1] == 0:
                        break
                    # If p must be paired, jump to cons[p]
                    if allow_unpaired_position[p] == 0:
                        q = cons_arr[p]
                        if q < p:
                            break
                    # Check q doesn't cross constrained positions
                    if q > j + 1 and q > allow_unpaired_range[j]:
                        continue
                    # Check p-q pair is allowed
                    if q != -1 and q < n:
                        nucq = nucs[q]
                        if not ((cons_arr[p] == -1 or cons_arr[p] == q) and
                                (cons_arr[q] == -1 or cons_arr[q] == p) and
                                allowed_pairs[nucp, nucq]):
                            continue

                if q != -1 and q < n and (i - p - 1) <= SINGLE_MAX_LEN:
                    target_idx = STATE_MULTI * n2 + q * n + p
                    if scores[target_idx] < m2_score:
                        scores[target_idx] = m2_score
                        manners[target_idx] = MANNER_MULTI
                        l1s[target_idx] = i - p
                        l2s[target_idx] = q - j

        # Collect M beam
        m_count = 0
        for i in range(j + 1):
            idx = STATE_M * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[m_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[m_count] = prefix + scores[idx]
                m_count += 1

        m_count = beam_prune_deterministic(beam_arr, beam_scores, m_count, beam)
        sort_by_position_desc(beam_arr, m_count)

        m_beam_count[j] = m_count
        for idx in range(m_count):
            m_beam_idx[j, idx] = beam_arr[idx]

        if j < n - 1:
            # Constraint check: don't extend M if j+1 must be paired
            if not use_constraints or allow_unpaired_position[j + 1] != 0:
                for midx in range(m_count):
                    i = beam_arr[midx]
                    m_score = scores[STATE_M * n2 + j * n + i]
                    target_idx = STATE_M * n2 + (j + 1) * n + i
                    if scores[target_idx] < m_score:
                        scores[target_idx] = m_score
                        manners[target_idx] = MANNER_M_eq_M_plus_U

        if j < n - 1:
            # Constraint check: don't extend C if next position must be paired
            if not use_constraints or allow_unpaired_position[j + 1] != 0:
                if c_scores[j + 1] < c_scores[j]:
                    c_scores[j + 1] = c_scores[j]
                    c_manners[j + 1] = MANNER_C_eq_C_plus_U

    return c_scores[n - 1]


# ============================================================================
# Main parsing loop - CONTRAfold mode
# ============================================================================

@njit(cache=True, fastmath=True)
def parse_loop_contrafold(
    n, beam, nucs, allowed_pairs, next_pair, no_sharp_turn,
    scores, manners, splits, l1s, l2s,
    c_scores, c_manners, c_splits,
    beam_arr, beam_scores,
    m_beam_idx, m_beam_count,
    hairpin_arr, helix_closing_arr, terminal_mismatch_arr,
    dangle_left_arr, dangle_right_arr, base_pair_arr, helix_stacking_arr,
    cache_single, bulge_nuc_arr, internal_nuc_arr,
    multi_base, multi_unpaired, multi_paired,
    external_unpaired, external_paired,
    use_constraints, cons_arr, allow_unpaired_position, allow_unpaired_range,
    bonus_matrix, alpha_scaled
):
    n2 = n * n
    value_min = VALUE_MIN_FLOAT

    # Initialize C scores (with constraint awareness)
    if n > 0:
        # Only initialize C[0] as unpaired if position 0 can be unpaired
        if not use_constraints or allow_unpaired_position[0] != 0:
            c_scores[0] = external_unpaired
            c_manners[0] = MANNER_C_eq_C_plus_U
    if n > 1:
        # Only initialize C[1] as unpaired if positions 0 and 1 can both be unpaired
        if not use_constraints or (allow_unpaired_position[0] != 0 and allow_unpaired_position[1] != 0):
            c_scores[1] = external_unpaired * 2
            c_manners[1] = MANNER_C_eq_C_plus_U

    for j in range(n):
        nucj = nucs[j]
        nucj1 = nucs[j + 1] if j + 1 < n else -1
        nucj_1 = nucs[j - 1] if j > 0 else -1

        # Collect H beam
        h_count = 0
        for i in range(j + 1):
            idx = STATE_H * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[h_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[h_count] = prefix + scores[idx]
                h_count += 1

        h_count = beam_prune_deterministic(beam_arr, beam_scores, h_count, beam)

        # Hairpin initiation
        jnext = next_pair[nucj, j]
        if no_sharp_turn:
            while jnext != -1 and jnext - j < 4:
                jnext = next_pair[nucj, jnext]

        # Constraint check for hairpin initiation
        if use_constraints:
            if allow_unpaired_position[j] == 0:
                # j must be left bracket of a forced pair, jump to cons[j] directly
                jnext = cons_arr[j] if cons_arr[j] > j else -1
            if jnext != -1:
                nucjnext = nucs[jnext]
                # Check if jnext crosses a constrained bracket or pair not allowed
                if jnext > allow_unpaired_range[j]:
                    jnext = -1
                elif not ((cons_arr[j] == -1 or cons_arr[j] == jnext) and
                          (cons_arr[jnext] == -1 or cons_arr[jnext] == j) and
                          allowed_pairs[nucj, nucjnext]):
                    jnext = -1

        if jnext != -1 and jnext < n:
            nucjnext = nucs[jnext]
            nucjnext_1 = nucs[jnext - 1] if jnext > 0 else -1
            length = min(jnext - j - 1, HAIRPIN_MAX_LEN)
            newscore = (hairpin_arr[length] +
                       score_helix_closing(nucj, nucjnext, helix_closing_arr) +
                       score_terminal_mismatch(nucj, nucj1, nucjnext_1, nucjnext, terminal_mismatch_arr))
            # Add PARIS bonus
            if alpha_scaled != 0.0:
                newscore += bonus_matrix[j * n + jnext] * alpha_scaled

            idx = STATE_H * n2 + jnext * n + j
            if scores[idx] < newscore:
                scores[idx] = newscore
                manners[idx] = MANNER_H

        # Process H states
        for hidx in range(h_count):
            i = beam_arr[hidx]
            h_idx = STATE_H * n2 + j * n + i
            h_score = scores[h_idx]

            p_idx = STATE_P * n2 + j * n + i
            if scores[p_idx] < h_score:
                scores[p_idx] = h_score
                manners[p_idx] = MANNER_HAIRPIN

            nuci = nucs[i]
            jnext_i = next_pair[nuci, j]

            # Constraint check for H state extension (CONTRAfold)
            if jnext_i != -1 and use_constraints:
                nucjnext_check = nucs[jnext_i]
                # Check if jnext_i crosses a constrained bracket or pair not allowed
                if jnext_i > allow_unpaired_range[i]:
                    jnext_i = -1
                elif not ((cons_arr[i] == -1 or cons_arr[i] == jnext_i) and
                          (cons_arr[jnext_i] == -1 or cons_arr[jnext_i] == i) and
                          allowed_pairs[nuci, nucjnext_check]):
                    jnext_i = -1

            if jnext_i != -1 and jnext_i < n:
                nuci1 = nucs[i + 1] if i + 1 < n else -1
                nucjnext = nucs[jnext_i]
                nucjnext_1 = nucs[jnext_i - 1] if jnext_i > 0 else -1
                length = min(jnext_i - i - 1, HAIRPIN_MAX_LEN)
                newscore = (hairpin_arr[length] +
                           score_helix_closing(nuci, nucjnext, helix_closing_arr) +
                           score_terminal_mismatch(nuci, nuci1, nucjnext_1, nucjnext, terminal_mismatch_arr))
                # Add PARIS bonus
                if alpha_scaled != 0.0:
                    newscore += bonus_matrix[i * n + jnext_i] * alpha_scaled

                idx = STATE_H * n2 + jnext_i * n + i
                if scores[idx] < newscore:
                    scores[idx] = newscore
                    manners[idx] = MANNER_H

        if j == 0:
            continue

        # Collect Multi beam
        multi_count = 0
        for i in range(j + 1):
            idx = STATE_MULTI * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[multi_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[multi_count] = prefix + scores[idx]
                multi_count += 1

        multi_count = beam_prune_deterministic(beam_arr, beam_scores, multi_count, beam)

        # Process Multi states
        for midx in range(multi_count):
            i = beam_arr[midx]
            m_idx = STATE_MULTI * n2 + j * n + i
            multi_score = scores[m_idx]
            nuci = nucs[i]
            nuci1 = nucs[i + 1] if i + 1 < n else -1

            junction_A = score_helix_closing(nuci, nucj, helix_closing_arr)
            if i < n - 1:
                junction_A += score_dangle_left(nuci, nuci1, nucj, dangle_left_arr)
            if j > 0:
                junction_A += score_dangle_right(nuci, nucj_1, nucj, dangle_right_arr)
            newscore = multi_score + junction_A + multi_paired + multi_base
            # Add PARIS bonus for multi-loop closing pair (i, j)
            if alpha_scaled != 0.0:
                newscore += bonus_matrix[i * n + j] * alpha_scaled

            p_idx = STATE_P * n2 + j * n + i
            if scores[p_idx] < newscore:
                scores[p_idx] = newscore
                manners[p_idx] = MANNER_P_eq_MULTI

            jnext_i = next_pair[nuci, j]

            # Constraint check for Multi state extension (CONTRAfold)
            if jnext_i != -1 and use_constraints:
                nucjnext_check = nucs[jnext_i]
                # Check if jnext_i crosses a constrained bracket or pair not allowed
                if jnext_i > allow_unpaired_range[j]:
                    jnext_i = -1
                elif not ((cons_arr[i] == -1 or cons_arr[i] == jnext_i) and
                          (cons_arr[jnext_i] == -1 or cons_arr[jnext_i] == i) and
                          allowed_pairs[nuci, nucjnext_check]):
                    jnext_i = -1

            if jnext_i != -1 and jnext_i < n:
                unpaired_len = jnext_i - j
                new_multi_score = multi_score + unpaired_len * multi_unpaired
                idx = STATE_MULTI * n2 + jnext_i * n + i
                if scores[idx] < new_multi_score:
                    scores[idx] = new_multi_score
                    manners[idx] = MANNER_MULTI_eq_MULTI_plus_U
                    l1s[idx] = l1s[m_idx]
                    l2s[idx] = l2s[m_idx] + jnext_i - j

        # Collect P beam
        p_count = 0
        for i in range(j + 1):
            idx = STATE_P * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[p_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[p_count] = prefix + scores[idx]
                p_count += 1

        p_count = beam_prune_deterministic(beam_arr, beam_scores, p_count, beam)

        # Process P states
        for pidx in range(p_count):
            i = beam_arr[pidx]
            p_idx_ij = STATE_P * n2 + j * n + i
            p_score = scores[p_idx_ij]
            nuci = nucs[i]
            nuci_1 = nucs[i - 1] if i > 0 else -1

            if i > 0 and j < n - 6:
                junction_A = score_helix_closing(nucj, nuci, helix_closing_arr)
                if j < n - 1:
                    junction_A += score_dangle_left(nucj, nucj1, nuci, dangle_left_arr)
                if i > 0:
                    junction_A += score_dangle_right(nucj, nuci_1, nuci, dangle_right_arr)
                newscore = junction_A + score_base_pair(nuci, nucj, base_pair_arr) + multi_paired + p_score

                m_idx = STATE_M * n2 + j * n + i
                if scores[m_idx] < newscore:
                    scores[m_idx] = newscore
                    manners[m_idx] = MANNER_M_eq_P

            if i > 1:
                junction_A = score_helix_closing(nucj, nuci, helix_closing_arr)
                if j < n - 1:
                    junction_A += score_dangle_left(nucj, nucj1, nuci, dangle_left_arr)
                if i > 0:
                    junction_A += score_dangle_right(nucj, nuci_1, nuci, dangle_right_arr)
                M1_score = junction_A + score_base_pair(nuci, nucj, base_pair_arr) + multi_paired + p_score

                k = i - 1
                m_cnt = m_beam_count[k]
                for m_idx_k in range(m_cnt):
                    newi = m_beam_idx[k, m_idx_k]
                    m_score = scores[STATE_M * n2 + k * n + newi]
                    if m_score > value_min:
                        newscore = M1_score + m_score
                        m2_idx = STATE_M2 * n2 + j * n + newi
                        if scores[m2_idx] < newscore:
                            scores[m2_idx] = newscore
                            manners[m2_idx] = MANNER_M2_eq_M_plus_P
                            splits[m2_idx] = k

            # C = C + P
            junction_A = score_helix_closing(nucj, nuci, helix_closing_arr)
            if j < n - 1:
                junction_A += score_dangle_left(nucj, nucj1, nuci, dangle_left_arr)
            if i > 0:
                junction_A += score_dangle_right(nucj, nuci_1, nuci, dangle_right_arr)
            ext_score = junction_A + external_paired + score_base_pair(nuci, nucj, base_pair_arr)

            k = i - 1
            if k >= 0:
                newscore = ext_score + c_scores[k] + p_score
                if c_scores[j] < newscore:
                    c_scores[j] = newscore
                    c_manners[j] = MANNER_C_eq_C_plus_P
                    c_splits[j] = k
            else:
                nuc0 = nucs[0]
                junction_A_0 = score_helix_closing(nucj, nuc0, helix_closing_arr)
                if j < n - 1:
                    junction_A_0 += score_dangle_left(nucj, nucj1, nuc0, dangle_left_arr)
                ext_score_0 = junction_A_0 + external_paired + score_base_pair(nuc0, nucj, base_pair_arr)
                newscore = ext_score_0 + p_score
                if c_scores[j] < newscore:
                    c_scores[j] = newscore
                    c_manners[j] = MANNER_C_eq_C_plus_P
                    c_splits[j] = -1

            # SINGLE/HELIX
            base_pair_ij = score_base_pair(nuci, nucj, base_pair_arr)
            junction_B = (score_helix_closing(nucj, nuci, helix_closing_arr) +
                         score_terminal_mismatch(nucj, nucj1, nuci_1, nuci, terminal_mismatch_arr))
            single_base_score = junction_B + base_pair_ij + p_score

            p_min = max(i - SINGLE_MAX_LEN, 0)
            for p in range(i - 1, p_min - 1, -1):
                # Constraint check for single/helix - outer loop
                if use_constraints:
                    # if p+1 must be paired, break
                    if p < i - 1 and allow_unpaired_position[p + 1] == 0:
                        break
                    # if p must be paired, p must be left bracket
                    if allow_unpaired_position[p] == 0:
                        q = cons_arr[p]
                        if q < p:
                            break
                        # q must be > j for outer pair to enclose inner pair
                        if q <= j:
                            continue
                        # Force q to this constrained pair
                    else:
                        q = -1  # will be set below

                nucp = nucs[p]
                nucp1 = nucs[p + 1]
                if not use_constraints or allow_unpaired_position[p] != 0:
                    q = next_pair[nucp, j]
                l1 = i - p - 1
                max_l2 = SINGLE_MAX_LEN - l1

                while q != -1 and q < n and q - j - 1 <= max_l2:
                    # Constraint check for single/helix - inner loop
                    if use_constraints:
                        # if q-1 must be paired, break
                        if q > j + 1 and q > allow_unpaired_range[j]:
                            break
                        # if p q are not allowed to pair, break
                        nucq = nucs[q]
                        if not ((cons_arr[p] == -1 or cons_arr[p] == q) and
                                (cons_arr[q] == -1 or cons_arr[q] == p) and
                                allowed_pairs[nucp, nucq]):
                            break

                    nucq = nucs[q]
                    nucq_1 = nucs[q - 1] if q > 0 else -1
                    nucq1 = nucs[q + 1] if q + 1 < n else -1

                    if p == i - 1 and q == j + 1:
                        newscore = (score_helix_stacking(nucp, nucp1, nucq_1, nucq, helix_stacking_arr) +
                                   score_base_pair(nucp1, nucq_1, base_pair_arr) + p_score)
                    else:
                        l2 = q - j - 1
                        newscore = cache_single[l1, l2] + single_base_score
                        newscore += (score_helix_closing(nucp, nucq, helix_closing_arr) +
                                    score_terminal_mismatch(nucp, nucp1, nucq_1, nucq, terminal_mismatch_arr))
                        if l1 == 0 and l2 == 1 and nucj1 >= 0:
                            newscore += bulge_nuc_arr[nucj1]
                        elif l1 == 1 and l2 == 0 and nuci_1 >= 0:
                            newscore += bulge_nuc_arr[nuci_1]
                        elif l1 == 1 and l2 == 1 and nuci_1 >= 0 and nucj1 >= 0:
                            newscore += internal_nuc_arr[nuci_1 * NOTON + nucj1]

                    # Add PARIS bonus for the new base pair (p, q)
                    if alpha_scaled != 0.0:
                        newscore += bonus_matrix[p * n + q] * alpha_scaled

                    target_idx = STATE_P * n2 + q * n + p
                    if scores[target_idx] < newscore:
                        scores[target_idx] = newscore
                        if p == i - 1 and q == j + 1:
                            manners[target_idx] = MANNER_HELIX
                            l1s[target_idx] = 0
                            l2s[target_idx] = 0
                        else:
                            manners[target_idx] = MANNER_SINGLE
                            l1s[target_idx] = i - p
                            l2s[target_idx] = q - j

                    q = next_pair[nucp, q]

        # Collect M2 beam
        m2_count = 0
        for i in range(j + 1):
            idx = STATE_M2 * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[m2_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[m2_count] = prefix + scores[idx]
                m2_count += 1

        m2_count = beam_prune_deterministic(beam_arr, beam_scores, m2_count, beam)

        # Process M2 states
        for m2idx in range(m2_count):
            i = beam_arr[m2idx]
            m2_idx = STATE_M2 * n2 + j * n + i
            m2_score = scores[m2_idx]

            m_idx = STATE_M * n2 + j * n + i
            if scores[m_idx] < m2_score:
                scores[m_idx] = m2_score
                manners[m_idx] = MANNER_M_eq_M2

            p_min = max(i - SINGLE_MAX_LEN, 0)
            for p in range(i - 1, p_min - 1, -1):
                nucp = nucs[p]
                q = next_pair[nucp, j]

                # Constraint check for Multi state creation (CONTRAfold)
                if use_constraints:
                    # If p+1 must be paired, break
                    if p < i - 1 and allow_unpaired_position[p + 1] == 0:
                        break
                    # If p must be paired, jump to cons[p]
                    if allow_unpaired_position[p] == 0:
                        q = cons_arr[p]
                        if q < p:
                            break
                    # Check q doesn't cross constrained positions
                    if q > j + 1 and q > allow_unpaired_range[j]:
                        continue
                    # Check p-q pair is allowed
                    if q != -1 and q < n:
                        nucq = nucs[q]
                        if not ((cons_arr[p] == -1 or cons_arr[p] == q) and
                                (cons_arr[q] == -1 or cons_arr[q] == p) and
                                allowed_pairs[nucp, nucq]):
                            continue

                if q != -1 and q < n and (i - p - 1) <= SINGLE_MAX_LEN:
                    len1 = i - p - 1
                    len2 = q - j - 1
                    new_multi_score = m2_score
                    if len1 > 0:
                        new_multi_score += len1 * multi_unpaired
                    if len2 > 0:
                        new_multi_score += len2 * multi_unpaired
                    target_idx = STATE_MULTI * n2 + q * n + p
                    if scores[target_idx] < new_multi_score:
                        scores[target_idx] = new_multi_score
                        manners[target_idx] = MANNER_MULTI
                        l1s[target_idx] = i - p
                        l2s[target_idx] = q - j

        # Collect M beam
        m_count = 0
        for i in range(j + 1):
            idx = STATE_M * n2 + j * n + i
            if scores[idx] > value_min:
                beam_arr[m_count] = i
                prefix = c_scores[i - 1] if i > 0 else np.float64(0.0)
                beam_scores[m_count] = prefix + scores[idx]
                m_count += 1

        m_count = beam_prune_deterministic(beam_arr, beam_scores, m_count, beam)

        m_beam_count[j] = m_count
        for idx in range(m_count):
            m_beam_idx[j, idx] = beam_arr[idx]

        if j < n - 1:
            # Constraint check: don't extend M if j+1 must be paired
            if not use_constraints or allow_unpaired_position[j + 1] != 0:
                for midx in range(m_count):
                    i = beam_arr[midx]
                    m_score = scores[STATE_M * n2 + j * n + i]
                    newscore = multi_unpaired + m_score
                    target_idx = STATE_M * n2 + (j + 1) * n + i
                    if scores[target_idx] < newscore:
                        scores[target_idx] = newscore
                        manners[target_idx] = MANNER_M_eq_M_plus_U

        if j < n - 1:
            # Constraint check: don't extend C if next position must be paired
            if not use_constraints or allow_unpaired_position[j + 1] != 0:
                newscore = external_unpaired + c_scores[j]
                if c_scores[j + 1] < newscore:
                    c_scores[j + 1] = newscore
                    c_manners[j + 1] = MANNER_C_eq_C_plus_U

    return c_scores[n - 1]


# ============================================================================
# Parser class
# ============================================================================

class BeamCKYParserHyper:
    """Ultra-optimized BeamCKY parser with flat arrays and PARIS bonus support."""

    def __init__(self, beam_size=100, nosharpturn=True, is_verbose=False, lv=False, use_constraints=False):
        self.beam = beam_size
        self.no_sharp_turn = nosharpturn
        self.is_verbose = is_verbose
        self.lv = lv
        self.use_constraints = use_constraints
        self._allowed_pairs = np.array(Utils.shared.allowed_pairs, dtype=np.int8)

        # PARIS bonus attributes
        self._alpha = 0.0
        self._alpha_scaled = np.float32(0.0)
        self._bonus_matrix = None
        self._raw_bonus_matrix = None  # 保存原始矩阵用于调试

        # PARIS bonus 模式参数
        self._bonus_mode = BONUS_MODE_PAIR  # 默认使用单对 bonus
        self._stack_method = 0  # stack 聚合方法: 0=min, 1=mean, 2=max, 3=geometric
        self._min_helix_len = 2  # helix 模式最小茎长度
        self._diagonal_smooth = False  # 是否使用对角线平滑
        self._smooth_window = 2  # 平滑窗口大小

    def set_alpha(self, alpha):
        """Set the scaling factor for PARIS bonus."""
        self._alpha = alpha
        self._alpha_scaled = np.float32(alpha * 100.0) if self.lv else np.float32(alpha)

    def set_bonus_mode(self, mode='pair', stack_method=0, min_helix_len=2,
                       diagonal_smooth=False, smooth_window=2):
        """
        设置 PARIS bonus 模式

        Args:
            mode: str, bonus 模式
                - 'pair': 单碱基对 bonus (原始模式)
                - 'stack': Stack-level bonus，鼓励连续堆叠
                - 'helix': Helix-level bonus，要求最小茎长度
            stack_method: int, stack bonus 聚合方法
                - 0: min (保守，要求两个 pair 都有信号)
                - 1: mean (平衡)
                - 2: max (激进)
                - 3: geometric mean
            min_helix_len: int, helix 模式最小茎长度要求
            diagonal_smooth: bool, 是否对支持度矩阵进行对角线平滑
            smooth_window: int, 对角线平滑窗口大小
        """
        mode_map = {'pair': BONUS_MODE_PAIR, 'stack': BONUS_MODE_STACK, 'helix': BONUS_MODE_HELIX}
        self._bonus_mode = mode_map.get(mode.lower(), BONUS_MODE_PAIR)
        self._stack_method = stack_method
        self._min_helix_len = min_helix_len
        self._diagonal_smooth = diagonal_smooth
        self._smooth_window = smooth_window

    def set_bonus_matrix(self, bonus_matrix, seq_length):
        """
        设置并预处理 PARIS bonus 矩阵

        根据 bonus_mode 进行相应的预处理:
        - pair: 直接使用原始矩阵
        - stack: 计算 stack bonus 矩阵
        - helix: 计算 helix bonus 矩阵
        """
        if bonus_matrix is None:
            self._bonus_matrix = np.zeros(seq_length * seq_length, dtype=np.float32)
            self._raw_bonus_matrix = None
            return

        # 转换为扁平数组
        if bonus_matrix.ndim == 2:
            flat_matrix = bonus_matrix.flatten().astype(np.float32)
        else:
            flat_matrix = bonus_matrix.astype(np.float32)

        self._raw_bonus_matrix = flat_matrix.copy()
        n = seq_length

        # Step 1: 可选的对角线平滑
        if self._diagonal_smooth:
            flat_matrix = diagonal_smooth_matrix(flat_matrix, n, self._smooth_window)

        # Step 2: 根据 bonus 模式处理
        if self._bonus_mode == BONUS_MODE_PAIR:
            # 原始单对 bonus
            self._bonus_matrix = flat_matrix
        elif self._bonus_mode == BONUS_MODE_STACK:
            # Stack-level bonus
            self._bonus_matrix = compute_stack_bonus_matrix(flat_matrix, n, self._stack_method)
        elif self._bonus_mode == BONUS_MODE_HELIX:
            # Helix-level bonus with minimum length
            self._bonus_matrix = compute_helix_bonus_matrix(flat_matrix, n, self._min_helix_len)
        else:
            self._bonus_matrix = flat_matrix

    def parse(self, seq, cons=None):
        """
        Parse sequence with optional constraints.

        Args:
            seq: RNA sequence string (automatically converted: lowercase->uppercase, T->U)
            cons: Optional constraint array (list/array of int)
                  -1 = no constraint (free)
                  -2 = forced unpaired
                  j >= 0 = forced pair with position j
        """
        # Preprocess sequence: lowercase->uppercase, T->U
        seq = seq.upper().replace('T', 'U')

        start_time = time.time()
        n = len(seq)
        lv = self.lv

        # Convert sequence
        nucs = np.zeros(n, dtype=np.int32)
        for i, c in enumerate(seq):
            nucs[i] = Utils.shared.get_acgu_num_c(c)

        # Process constraints
        use_constraints = self.use_constraints and cons is not None
        if use_constraints:
            # Parse constraint string if needed
            if isinstance(cons, str):
                # Parse constraint string: ? = free, . = unpaired, ( ) = forced pairs
                cons_arr = np.full(n, -1, dtype=np.int32)
                stack = []
                for i, c in enumerate(cons):
                    if c == '(':
                        stack.append(i)
                    elif c == ')':
                        if stack:
                            left = stack.pop()
                            cons_arr[left] = i
                            cons_arr[i] = left
                    elif c == '.':
                        cons_arr[i] = -2  # forced unpaired
                    # '?' or anything else -> -1 (no constraint)
            else:
                cons_arr = np.array(cons, dtype=np.int32)
            # allow_unpaired_position[i] = True if position i can be unpaired
            allow_unpaired_position = np.zeros(n, dtype=np.int8)
            for i in range(n):
                cons_idx = cons_arr[i]
                allow_unpaired_position[i] = 1 if (cons_idx == -1 or cons_idx == -2) else 0
                # Validate forced pairs
                if cons_idx >= 0:
                    if not self._allowed_pairs[nucs[i], nucs[cons_idx]]:
                        print("Constraints on non-classical base pairs (non AU, CG, GU pairs)")
                        return None, 0, 0, 0

            # allow_unpaired_range[i] = first position >= i that has a forced pair constraint
            allow_unpaired_range = np.full(n, n, dtype=np.int32)
            firstpair = n
            for i in range(n - 1, -1, -1):
                allow_unpaired_range[i] = firstpair
                if cons_arr[i] >= 0:
                    firstpair = i
        else:
            cons_arr = np.full(n, -1, dtype=np.int32)
            allow_unpaired_position = np.ones(n, dtype=np.int8)
            allow_unpaired_range = np.full(n, n, dtype=np.int32)

        # Build next_pair (with constraint awareness)
        next_pair = np.full((NOTON, n), -1, dtype=np.int32)
        for nuc in range(NOTON):
            next_val = -1
            for j in range(n - 1, -1, -1):
                next_pair[nuc, j] = next_val
                # Skip positions that must be unpaired (cons[j] == -2)
                if use_constraints and cons_arr[j] == -2:
                    continue
                if self._allowed_pairs[nuc, nucs[j]]:
                    next_val = j

        # Allocate arrays
        n2 = n * n
        total_size = NUM_STATE_TYPES * n2

        scores = np.full(total_size, VALUE_MIN_FLOAT, dtype=np.float64)
        c_scores = np.full(n, VALUE_MIN_FLOAT, dtype=np.float64)
        beam_scores = np.zeros(n, dtype=np.float64)

        manners = np.zeros(total_size, dtype=np.int8)
        splits = np.zeros(total_size, dtype=np.int32)
        l1s = np.zeros(total_size, dtype=np.int16)
        l2s = np.zeros(total_size, dtype=np.int16)

        c_manners = np.zeros(n, dtype=np.int8)
        c_splits = np.zeros(n, dtype=np.int32)

        beam_arr = np.zeros(n, dtype=np.int32)
        max_beam = min(self.beam, n)
        m_beam_idx = np.zeros((n, max_beam), dtype=np.int32)
        m_beam_count = np.zeros(n, dtype=np.int32)

        # Bonus matrix - initialize if not set
        if self._bonus_matrix is None:
            bonus_matrix = np.zeros(n * n, dtype=np.float32)
        else:
            bonus_matrix = self._bonus_matrix

        if lv:
            # Vienna mode
            if_tetra = np.full(n, -1, dtype=np.int32)
            if_hexa = np.full(n, -1, dtype=np.int32)
            if_tri = np.full(n, -1, dtype=np.int32)

            tetra_list, hexa_list, tri_list = [], [], []
            v_init_tetra_hex_tri(seq, n, tetra_list, hexa_list, tri_list)
            for i, idx in enumerate(tetra_list):
                if i < n and idx >= 0:
                    if_tetra[i] = idx
            for i, idx in enumerate(hexa_list):
                if i < n and idx >= 0:
                    if_hexa[i] = idx
            for i, idx in enumerate(tri_list):
                if i < n and idx >= 0:
                    if_tri[i] = idx

            viterbi = parse_loop_vienna(
                n, self.beam, nucs, self._allowed_pairs, next_pair, self.no_sharp_turn,
                scores, manners, splits, l1s, l2s,
                c_scores, c_manners, c_splits,
                beam_arr, beam_scores,
                m_beam_idx, m_beam_count,
                if_tetra, if_hexa, if_tri,
                _v_hairpin, _v_mismatchH, _v_terminalAU, _v_tetra, _v_hexa, _v_tri,
                _v_stack, _v_bulge, _v_internal,
                _v_int11, _v_int21, _v_int22,
                _v_mismatchI, _v_mismatch1nI, _v_mismatch23I,
                _v_ninio, _v_max_ninio,
                _v_mismatchM, _v_mismatchExt, _v_dangle5, _v_dangle3,
                _v_ml_intern, _v_ml_closing,
                use_constraints, cons_arr, allow_unpaired_position, allow_unpaired_range,
                bonus_matrix, self._alpha_scaled
            )
            score = float(viterbi) / -100.0
        else:
            # CONTRAfold mode
            viterbi = parse_loop_contrafold(
                n, self.beam, nucs, self._allowed_pairs, next_pair, self.no_sharp_turn,
                scores, manners, splits, l1s, l2s,
                c_scores, c_manners, c_splits,
                beam_arr, beam_scores,
                m_beam_idx, m_beam_count,
                _hairpin_length, _helix_closing, _terminal_mismatch,
                _dangle_left, _dangle_right, _base_pair, _helix_stacking,
                _cache_single, _bulge_0x1_nucleotides, _internal_1x1_nucleotides,
                _multi_base, _multi_unpaired, _multi_paired,
                _external_unpaired, _external_paired,
                use_constraints, cons_arr, allow_unpaired_position, allow_unpaired_range,
                bonus_matrix, self._alpha_scaled
            )
            score = float(viterbi)

        # Save for traceback
        self._n = n
        self._n2 = n2
        self._scores = scores
        self._manners = manners
        self._splits = splits
        self._l1s = l1s
        self._l2s = l2s
        self._c_scores = c_scores
        self._c_manners = c_manners
        self._c_splits = c_splits
        self._value_min = VALUE_MIN_FLOAT

        structure = self._traceback()
        elapsed = time.time() - start_time

        if self.is_verbose:
            print(f"Time: {elapsed:.4f}s, Length: {n}, Score: {score:.2f}")

        return structure, score, 0, elapsed

    def _traceback(self):
        n = self._n
        n2 = self._n2
        result = ['.'] * n

        stk = [(0, n - 1, int(self._c_manners[n - 1]), int(self._c_splits[n - 1]), 0, 0, -1)]

        while stk:
            i, j, manner, split, l1, l2, state = stk.pop()

            if manner == MANNER_NONE or manner == MANNER_H:
                continue
            elif manner == MANNER_HAIRPIN:
                result[i] = '('
                result[j] = ')'
            elif manner == MANNER_SINGLE:
                result[i] = '('
                result[j] = ')'
                p, q = i + l1, j - l2
                idx = int(STATE_P * n2 + q * n + p)
                stk.append((p, q, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_P))
            elif manner == MANNER_HELIX:
                result[i] = '('
                result[j] = ')'
                idx = int(STATE_P * n2 + (j - 1) * n + (i + 1))
                stk.append((i + 1, j - 1, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_P))
            elif manner == MANNER_MULTI:
                p, q = i + l1, j - l2
                idx = int(STATE_M2 * n2 + q * n + p)
                stk.append((p, q, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_M2))
            elif manner == MANNER_MULTI_eq_MULTI_plus_U:
                p, q = i + l1, j - l2
                idx = int(STATE_M2 * n2 + q * n + p)
                stk.append((p, q, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_M2))
            elif manner == MANNER_P_eq_MULTI:
                result[i] = '('
                result[j] = ')'
                idx = int(STATE_MULTI * n2 + j * n + i)
                stk.append((i, j, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_MULTI))
            elif manner == MANNER_M2_eq_M_plus_P:
                k = split
                idx_m = int(STATE_M * n2 + k * n + i)
                idx_p = int(STATE_P * n2 + j * n + (k + 1))
                stk.append((i, k, int(self._manners[idx_m]), int(self._splits[idx_m]),
                           int(self._l1s[idx_m]), int(self._l2s[idx_m]), STATE_M))
                stk.append((k + 1, j, int(self._manners[idx_p]), int(self._splits[idx_p]),
                           int(self._l1s[idx_p]), int(self._l2s[idx_p]), STATE_P))
            elif manner == MANNER_M_eq_M2:
                idx = int(STATE_M2 * n2 + j * n + i)
                stk.append((i, j, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_M2))
            elif manner == MANNER_M_eq_M_plus_U:
                idx = int(STATE_M * n2 + (j - 1) * n + i)
                stk.append((i, j - 1, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_M))
            elif manner == MANNER_M_eq_P:
                idx = int(STATE_P * n2 + j * n + i)
                stk.append((i, j, int(self._manners[idx]), int(self._splits[idx]),
                           int(self._l1s[idx]), int(self._l2s[idx]), STATE_P))
            elif manner == MANNER_C_eq_C_plus_U:
                if j > 0:
                    stk.append((0, j - 1, int(self._c_manners[j - 1]), int(self._c_splits[j - 1]), 0, 0, -1))
            elif manner == MANNER_C_eq_C_plus_P:
                k = split
                if k >= 0:
                    stk.append((0, k, int(self._c_manners[k]), int(self._c_splits[k]), 0, 0, -1))
                    idx = int(STATE_P * n2 + j * n + (k + 1))
                    stk.append((k + 1, j, int(self._manners[idx]), int(self._splits[idx]),
                               int(self._l1s[idx]), int(self._l2s[idx]), STATE_P))
                else:
                    idx = int(STATE_P * n2 + j * n + i)
                    stk.append((i, j, int(self._manners[idx]), int(self._splits[idx]),
                               int(self._l1s[idx]), int(self._l2s[idx]), STATE_P))

        return ''.join(result)

    def parse_subopt(self, seq, energy_delta=5.0, max_structures=100, window_size=0):
        """
        Parse sequence and return suboptimal structures within energy_delta of MFE.
        Uses Zuker suboptimal algorithm with recursive backtrace and global caching.
        Follows the C LinearFold implementation for structure generation.

        Args:
            seq: RNA sequence (automatically converted: lowercase->uppercase, T->U)
            energy_delta: Energy range from MFE (kcal/mol for Vienna, score units for CONTRAfold)
            max_structures: Maximum number of structures to return
            window_size: Window size for deduplication (0 = auto, -1 = disabled)

        Returns:
            List of (structure, score) tuples, sorted by energy
        """
        # Preprocess sequence: lowercase->uppercase, T->U
        seq = seq.upper().replace('T', 'U')

        start_time = time.time()
        n = len(seq)
        lv = self.lv
        self._seq = seq  # Store sequence for backtrace

        # Set window size (matching C code exactly)
        if window_size == 0:
            if n < 100:
                window_size = 2
            elif n < 300:
                window_size = 5
            elif n < 500:
                window_size = 7
            elif n < 1200:
                window_size = 9
            else:
                window_size = 12

        # Convert sequence
        nucs = np.zeros(n, dtype=np.int32)
        for i, c in enumerate(seq):
            nucs[i] = Utils.shared.get_acgu_num_c(c)

        # Build next_pair
        next_pair = np.full((NOTON, n), -1, dtype=np.int32)
        for nuc in range(NOTON):
            next_val = -1
            for j in range(n - 1, -1, -1):
                next_pair[nuc, j] = next_val
                if self._allowed_pairs[nuc, nucs[j]]:
                    next_val = j

        # No constraints for subopt mode
        use_constraints = False
        cons_arr = np.full(n, -1, dtype=np.int32)
        allow_unpaired_position = np.ones(n, dtype=np.int8)
        allow_unpaired_range = np.full(n, n, dtype=np.int32)

        # Allocate inside arrays
        n2 = n * n
        total_size = NUM_STATE_TYPES * n2

        scores = np.full(total_size, VALUE_MIN_FLOAT, dtype=np.float64)
        c_scores = np.full(n, VALUE_MIN_FLOAT, dtype=np.float64)
        beam_scores = np.zeros(n, dtype=np.float64)

        manners = np.zeros(total_size, dtype=np.int8)
        splits = np.zeros(total_size, dtype=np.int32)
        l1s = np.zeros(total_size, dtype=np.int16)
        l2s = np.zeros(total_size, dtype=np.int16)

        c_manners = np.zeros(n, dtype=np.int8)
        c_splits = np.zeros(n, dtype=np.int32)

        beam_arr = np.zeros(n, dtype=np.int32)
        max_beam = min(self.beam, n)
        m_beam_idx = np.zeros((n, max_beam), dtype=np.int32)
        m_beam_count = np.zeros(n, dtype=np.int32)

        # Bonus matrix - initialize if not set
        if self._bonus_matrix is None:
            bonus_matrix = np.zeros(n * n, dtype=np.float32)
        else:
            bonus_matrix = self._bonus_matrix

        if lv:
            # Vienna mode - inside pass
            if_tetra = np.full(n, -1, dtype=np.int32)
            if_hexa = np.full(n, -1, dtype=np.int32)
            if_tri = np.full(n, -1, dtype=np.int32)

            tetra_list, hexa_list, tri_list = [], [], []
            v_init_tetra_hex_tri(seq, n, tetra_list, hexa_list, tri_list)
            for i, idx in enumerate(tetra_list):
                if i < n and idx >= 0:
                    if_tetra[i] = idx
            for i, idx in enumerate(hexa_list):
                if i < n and idx >= 0:
                    if_hexa[i] = idx
            for i, idx in enumerate(tri_list):
                if i < n and idx >= 0:
                    if_tri[i] = idx

            viterbi = parse_loop_vienna(
                n, self.beam, nucs, self._allowed_pairs, next_pair, self.no_sharp_turn,
                scores, manners, splits, l1s, l2s,
                c_scores, c_manners, c_splits,
                beam_arr, beam_scores,
                m_beam_idx, m_beam_count,
                if_tetra, if_hexa, if_tri,
                _v_hairpin, _v_mismatchH, _v_terminalAU, _v_tetra, _v_hexa, _v_tri,
                _v_stack, _v_bulge, _v_internal,
                _v_int11, _v_int21, _v_int22,
                _v_mismatchI, _v_mismatch1nI, _v_mismatch23I,
                _v_ninio, _v_max_ninio,
                _v_mismatchM, _v_mismatchExt, _v_dangle5, _v_dangle3,
                _v_ml_intern, _v_ml_closing,
                use_constraints, cons_arr, allow_unpaired_position, allow_unpaired_range,
                bonus_matrix, self._alpha_scaled
            )
            mfe_score = float(viterbi) / -100.0
            delta_threshold = energy_delta * 100.0  # Convert to internal units
        else:
            # CONTRAfold mode - inside pass
            viterbi = parse_loop_contrafold(
                n, self.beam, nucs, self._allowed_pairs, next_pair, self.no_sharp_turn,
                scores, manners, splits, l1s, l2s,
                c_scores, c_manners, c_splits,
                beam_arr, beam_scores,
                m_beam_idx, m_beam_count,
                _hairpin_length, _helix_closing, _terminal_mismatch,
                _dangle_left, _dangle_right, _base_pair, _helix_stacking,
                _cache_single, _bulge_0x1_nucleotides, _internal_1x1_nucleotides,
                _multi_base, _multi_unpaired, _multi_paired,
                _external_unpaired, _external_paired,
                use_constraints, cons_arr, allow_unpaired_position, allow_unpaired_range,
                bonus_matrix, self._alpha_scaled
            )
            mfe_score = float(viterbi)
            delta_threshold = energy_delta

        # Save inside arrays for recursive backtrace
        self._n = n
        self._n2 = n2
        self._nucs = nucs
        self._next_pair = next_pair
        self._scores = scores
        self._manners = manners
        self._splits = splits
        self._l1s = l1s
        self._l2s = l2s
        self._c_scores = c_scores
        self._c_manners = c_manners
        self._c_splits = c_splits
        self._viterbi = viterbi
        self._window_size = window_size

        # Allocate outside (beta) arrays
        beta_scores = np.full(total_size, VALUE_MIN_FLOAT, dtype=np.float64)
        beta_manners = np.zeros(total_size, dtype=np.int8)
        beta_splits = np.zeros(total_size, dtype=np.int32)
        beta_l1s = np.zeros(total_size, dtype=np.int16)
        beta_l2s = np.zeros(total_size, dtype=np.int16)
        beta_c_scores = np.full(n, VALUE_MIN_FLOAT, dtype=np.float64)
        beta_c_manners = np.zeros(n, dtype=np.int8)
        beta_c_splits = np.zeros(n, dtype=np.int32)  # Store j (P's end) for C_eq_C_plus_P

        # Run outside pass
        self._outside(n, nucs, next_pair, lv,
                      scores, manners, splits, l1s, l2s,
                      c_scores, c_manners, c_splits,
                      beta_scores, beta_manners, beta_splits, beta_l1s, beta_l2s,
                      beta_c_scores, beta_c_manners, beta_c_splits)

        # Save beta arrays for recursive backtrace
        self._beta_scores = beta_scores
        self._beta_manners = beta_manners
        self._beta_splits = beta_splits
        self._beta_l1s = beta_l1s
        self._beta_l2s = beta_l2s
        self._beta_c_scores = beta_c_scores
        self._beta_c_manners = beta_c_manners
        self._beta_c_splits = beta_c_splits

        # Collect suboptimal pairs based on inside-outside scores (Zuker criterion)
        # C iterates over bestP_beta[j][i] where manner != 0
        sorted_bestP_beta = []
        for j in range(n):
            for i in range(j):
                idx = STATE_P * n2 + j * n + i
                # Check both inside and outside manners (matching C code)
                if manners[idx] == MANNER_NONE:
                    continue
                if beta_manners[idx] == MANNER_NONE:
                    continue
                alpha = scores[idx]
                beta = beta_scores[idx]
                if alpha <= VALUE_MIN_FLOAT + 1e10 or beta <= VALUE_MIN_FLOAT + 1e10:
                    continue

                # Check energy threshold (matching C code exactly)
                # C code: abs(alpha_inside + beta_outside - float(viterbi.score))/100. > zuker_energy_delta
                if lv:
                    if abs(alpha + beta - float(viterbi)) / 100.0 > energy_delta:
                        continue
                    # Filter out candidates with alpha+beta > viterbi (better than MFE)
                    # This can happen due to beam search inconsistencies
                    if alpha + beta > viterbi + 1:  # Allow small numerical tolerance
                        continue
                else:
                    if abs(alpha + beta - float(viterbi)) > energy_delta:
                        continue

                sorted_bestP_beta.append((-alpha - beta, i, j))

        # Sort by total score (best first = lowest negative sum), then by (i-j)
        # Matching C code's cmp function in LinearFold.h:
        #   if (score_a != score_b) return score_a < score_b
        #   else return (i_a - j_a) < (i_b - j_b)
        sorted_bestP_beta.sort(key=lambda x: (x[0], x[1] - x[2]))

        # Generate structures using inside-outside backtrace following C code pattern
        results = []
        seen_structures = set()
        window_visited = set()
        num_outputs = 0

        # Set member variables for traceback
        self._n = n
        self._n2 = n2
        self._scores = scores
        self._manners = manners
        self._splits = splits
        self._l1s = l1s
        self._l2s = l2s
        self._c_scores = c_scores
        self._c_manners = c_manners
        self._c_splits = c_splits

        # Get MFE structure using the standard _traceback method
        mfe_struct = self._traceback()

        if mfe_struct and mfe_struct not in seen_structures:
            seen_structures.add(mfe_struct)
            results.append((mfe_struct, mfe_score))
            num_outputs += 1

        # Global caches for inside-outside backtrace (following C code exactly)
        global_visited_outside = {}  # (state, i, j) -> (left_str, right_str)
        global_visited_inside = {}   # (state, i, j) -> structure_str

        # Store arrays as member variables for backtrace functions
        self._window_size = window_size
        self._beta_scores = beta_scores
        self._beta_manners = beta_manners
        self._beta_splits = beta_splits
        self._beta_l1s = beta_l1s
        self._beta_l2s = beta_l2s
        self._beta_c_scores = beta_c_scores
        self._beta_c_manners = beta_c_manners
        self._beta_c_splits = beta_c_splits

        # Generate suboptimal structures using Zuker algorithm (C code pattern exactly)
        for item in sorted_bestP_beta:
            if num_outputs >= max_structures:
                break

            i, j = item[1], item[2]

            # Skip if (i, j) is in window of a previous candidate (unless window_size < 0)
            if window_size >= 0:
                if (i, j) in window_visited:
                    continue
                self._window_fill(window_visited, i, j, n, window_size)

            try:
                idx_p = STATE_P * n2 + j * n + i

                # Trace outside: from (i,j) to external loop
                # Pass the beta state for P[i,j] (following C code: bestP_beta[j][i])
                outsider = self._trace_outside_recursive(
                    i, j, STATE_P, n, n2, window_size,
                    global_visited_outside, global_visited_inside, window_visited
                )
                global_visited_outside[(STATE_P, i, j)] = outsider

                # Trace inside: structure for [i, j]
                # Use an empty window_visited for insider (following C code)
                insider_window_visited = set()
                insider = self._trace_inside_recursive(
                    i, j, STATE_P, n, n2, window_size,
                    global_visited_inside, insider_window_visited
                )
                global_visited_inside[(STATE_P, i, j)] = insider

                # Build full-length structure (C code pattern: lines 1562-1579)
                full_structure = ['.'] * n

                # Fill in outsider.first (positions 0 to len-1)
                for pos in range(min(len(outsider[0]), n)):
                    full_structure[pos] = outsider[0][pos]

                # Fill in insider (positions i to i+len-1)
                for pos in range(min(len(insider), n - i)):
                    if i + pos < n:
                        full_structure[i + pos] = insider[pos]

                # Fill in outsider.second (positions n - len to n-1)
                second_start = n - len(outsider[1])
                for pos in range(len(outsider[1])):
                    if second_start + pos < n and second_start + pos >= 0:
                        full_structure[second_start + pos] = outsider[1][pos]

                structure = ''.join(full_structure)

                # Skip if structure length doesn't match
                if len(structure) != n:
                    continue

                # Skip duplicate structures
                if structure in seen_structures:
                    continue

                # Validate structure has balanced brackets
                if structure.count('(') != structure.count(')'):
                    continue

                # Validate that the candidate pair (i, j) appears in the traced structure
                # This ensures traceback consistency - if the structure doesn't contain
                # the starting pair, the inside-outside combination is invalid
                if len(structure) > max(i, j) and (structure[i] != '(' or structure[j] != ')'):
                    # Parse the structure to check if (i, j) is actually paired
                    stack = []
                    pairs_in_struct = {}
                    for pos, c in enumerate(structure):
                        if c == '(':
                            stack.append(pos)
                        elif c == ')' and stack:
                            open_pos = stack.pop()
                            pairs_in_struct[open_pos] = pos
                    # Skip if (i, j) is not a pair in the structure
                    if pairs_in_struct.get(i) != j:
                        continue

                # Use alpha+beta as energy (matching C code exactly)
                # C code: printf("%s (%.2f)\n", ..., (bestP[j][i].score + bestP_beta[j][i].score)/(-100.));
                alpha = scores[idx_p]
                beta_val = beta_scores[idx_p]
                if lv:
                    actual_score = float(alpha + beta_val) / -100.0
                else:
                    actual_score = float(alpha + beta_val)

                # Filter out structures that claim MFE energy but differ from MFE
                # This handles inconsistencies in inside-outside combination due to beam search
                if lv:
                    # For Vienna mode, if a structure has energy within 0.05 of MFE
                    # but is different from MFE, skip it (it's likely a spurious result)
                    if abs(actual_score - mfe_score) < 0.05 and structure != mfe_struct:
                        continue

                seen_structures.add(structure)
                num_outputs += 1
                results.append((structure, actual_score))

            except Exception as e:
                if self.is_verbose:
                    print(f"Backtrace failed for ({i}, {j}): {e}")
                    import traceback
                    traceback.print_exc()
                continue

        # Sort results by energy (best first = most negative for Vienna)
        results.sort(key=lambda x: x[1] if lv else -x[1])

        elapsed = time.time() - start_time
        if self.is_verbose:
            print(f"Subopt: {len(results)} structures in {elapsed:.4f}s")

        return results

    def _window_fill(self, window_visited, i, j, n, window_size):
        """Mark pairs within window as visited."""
        for ii in range(max(0, i - window_size), min(n - 1, i + window_size) + 1):
            for jj in range(max(0, j - window_size), min(n - 1, j + window_size) + 1):
                if ii < jj:
                    window_visited.add((ii, jj))

    def _backtrace_mfe(self, n, n2, scores, manners, splits, l1s, l2s,
                       c_scores, c_manners, c_splits):
        """Standard MFE backtrace from C state."""
        result = ['.'] * n

        # Stack for backtrace: (state, i, j)
        stack = [(-1, 0, n - 1)]  # Start from C state covering [0, n-1]

        while stack:
            state, i, j = stack.pop()

            if state == -1:  # C state (external loop)
                # Trace C from position j backwards
                pos = j
                while pos >= 0:
                    manner = int(c_manners[pos])
                    if manner == MANNER_C_eq_C_plus_U:
                        result[pos] = '.'
                        pos -= 1
                    elif manner == MANNER_C_eq_C_plus_P:
                        k = int(c_splits[pos])
                        # P covers [k+1, pos]
                        stack.append((STATE_P, k + 1, pos))
                        pos = k
                    else:
                        result[pos] = '.'
                        pos -= 1

            elif state == STATE_P:
                if i >= j:
                    continue

                idx = STATE_P * n2 + j * n + i
                manner = int(manners[idx])

                if manner == MANNER_HAIRPIN:
                    result[i] = '('
                    result[j] = ')'
                    for k in range(i + 1, j):
                        result[k] = '.'

                elif manner == MANNER_SINGLE:
                    l1, l2 = int(l1s[idx]), int(l2s[idx])
                    p, q = i + l1, j - l2
                    result[i] = '('
                    result[j] = ')'
                    for k in range(i + 1, p):
                        result[k] = '.'
                    for k in range(q + 1, j):
                        result[k] = '.'
                    stack.append((STATE_P, p, q))

                elif manner == MANNER_HELIX:
                    result[i] = '('
                    result[j] = ')'
                    stack.append((STATE_P, i + 1, j - 1))

                elif manner == MANNER_P_eq_MULTI:
                    result[i] = '('
                    result[j] = ')'
                    stack.append((STATE_MULTI, i + 1, j - 1))

            elif state == STATE_MULTI:
                idx = STATE_MULTI * n2 + j * n + i
                manner = int(manners[idx])

                if manner == MANNER_MULTI:
                    l1, l2 = int(l1s[idx]), int(l2s[idx])
                    p, q = i + l1 - 1, j - l2 + 1
                    for k in range(i, p):
                        result[k] = '.'
                    for k in range(q + 1, j + 1):
                        result[k] = '.'
                    stack.append((STATE_M2, p, q))

                elif manner == MANNER_MULTI_eq_MULTI_plus_U:
                    l1, l2 = int(l1s[idx]), int(l2s[idx])
                    p, q = i + l1 - 1, j - l2 + 1
                    for k in range(i, p):
                        result[k] = '.'
                    for k in range(q + 1, j + 1):
                        result[k] = '.'
                    stack.append((STATE_M2, p, q))

            elif state == STATE_M2:
                idx = STATE_M2 * n2 + j * n + i
                manner = int(manners[idx])

                if manner == MANNER_M2_eq_M_plus_P:
                    k = int(splits[idx])
                    stack.append((STATE_M, i, k))
                    stack.append((STATE_P, k + 1, j))

            elif state == STATE_M:
                idx = STATE_M * n2 + j * n + i
                manner = int(manners[idx])

                if manner == MANNER_M_eq_P:
                    stack.append((STATE_P, i, j))

                elif manner == MANNER_M_eq_M2:
                    stack.append((STATE_M2, i, j))

                elif manner == MANNER_M_eq_M_plus_U:
                    result[j] = '.'
                    stack.append((STATE_M, i, j - 1))

        return ''.join(result)

    def _trace_inside_recursive(self, i, j, state, n, n2, window_size,
                                 global_visited_inside, window_visited):
        """
        Recursive inside backtrace with global caching.
        Returns structure string for region [i, j].
        """
        # Check cache
        if (state, i, j) in global_visited_inside:
            return global_visited_inside[(state, i, j)]

        if state == STATE_P:
            idx = STATE_P * n2 + j * n + i
            manner = int(self._manners[idx])

            if manner == MANNER_H or manner == MANNER_NONE:
                return ""

            elif manner == MANNER_HAIRPIN:
                inner = '.' * (j - i - 1)
                result = '(' + inner + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, window_size)
                return result

            elif manner == MANNER_SINGLE:
                l1, l2 = int(self._l1s[idx]), int(self._l2s[idx])
                p, q = i + l1, j - l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)

                inner = self._trace_inside_recursive(p, q, STATE_P, n, n2, window_size,
                                                     global_visited_inside, window_visited)
                result = '(' + left + inner + right + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, window_size)
                return result

            elif manner == MANNER_HELIX:
                inner = self._trace_inside_recursive(i + 1, j - 1, STATE_P, n, n2, window_size,
                                                     global_visited_inside, window_visited)
                result = '(' + inner + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, window_size)
                return result

            elif manner == MANNER_P_eq_MULTI:
                inner = self._trace_inside_recursive(i, j, STATE_MULTI, n, n2, window_size,
                                                     global_visited_inside, window_visited)
                result = '(' + inner + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, window_size)
                return result

            else:
                return '(' + '.' * (j - i - 1) + ')'

        elif state == STATE_MULTI:
            idx = STATE_MULTI * n2 + j * n + i
            manner = int(self._manners[idx])

            if manner == MANNER_MULTI:
                l1, l2 = int(self._l1s[idx]), int(self._l2s[idx])
                p, q = i + l1, j - l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)
                inner = self._trace_inside_recursive(p, q, STATE_M2, n, n2, window_size,
                                                     global_visited_inside, window_visited)
                result = left + inner + right
                global_visited_inside[(STATE_MULTI, i, j)] = result
                return result

            elif manner == MANNER_MULTI_eq_MULTI_plus_U:
                l1, l2 = int(self._l1s[idx]), int(self._l2s[idx])
                p, q = i + l1, j - l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)
                inner = self._trace_inside_recursive(p, q, STATE_M2, n, n2, window_size,
                                                     global_visited_inside, window_visited)
                result = left + inner + right
                global_visited_inside[(STATE_MULTI, i, j)] = result
                return result

            else:
                return '.' * (j - i + 1)

        elif state == STATE_M2:
            idx = STATE_M2 * n2 + j * n + i
            manner = int(self._manners[idx])

            if manner == MANNER_M2_eq_M_plus_P:
                k = int(self._splits[idx])
                m_part = self._trace_inside_recursive(i, k, STATE_M, n, n2, window_size,
                                                      global_visited_inside, window_visited)
                p_part = self._trace_inside_recursive(k + 1, j, STATE_P, n, n2, window_size,
                                                      global_visited_inside, window_visited)
                result = m_part + p_part
                global_visited_inside[(STATE_M2, i, j)] = result
                return result

            else:
                return '.' * (j - i + 1)

        elif state == STATE_M:
            idx = STATE_M * n2 + j * n + i
            manner = int(self._manners[idx])

            if manner == MANNER_M_eq_P:
                result = self._trace_inside_recursive(i, j, STATE_P, n, n2, window_size,
                                                      global_visited_inside, window_visited)
                global_visited_inside[(STATE_M, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M2:
                result = self._trace_inside_recursive(i, j, STATE_M2, n, n2, window_size,
                                                      global_visited_inside, window_visited)
                global_visited_inside[(STATE_M, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M_plus_U:
                inner = self._trace_inside_recursive(i, j - 1, STATE_M, n, n2, window_size,
                                                     global_visited_inside, window_visited)
                result = inner + '.'
                global_visited_inside[(STATE_M, i, j)] = result
                return result

            else:
                return '.' * (j - i + 1)

        elif state == -1:  # External loop (C state)
            if j < 0:
                return ""
            manner = int(self._c_manners[j])

            if manner == MANNER_C_eq_C_plus_U:
                if j > 0:
                    inner = self._trace_inside_recursive(0, j - 1, -1, n, n2, window_size,
                                                         global_visited_inside, window_visited)
                else:
                    inner = ""
                result = inner + '.'
                global_visited_inside[(-1, 0, j)] = result
                return result

            elif manner == MANNER_C_eq_C_plus_P:
                k = int(self._c_splits[j])
                if k >= 0:
                    c_part = self._trace_inside_recursive(0, k, -1, n, n2, window_size,
                                                          global_visited_inside, window_visited)
                else:
                    c_part = ""
                p_part = self._trace_inside_recursive(k + 1, j, STATE_P, n, n2, window_size,
                                                      global_visited_inside, window_visited)
                result = c_part + p_part
                global_visited_inside[(-1, 0, j)] = result
                return result

            else:
                return '.' * (j + 1)

        return '.' * (j - i + 1)

    def _trace_outside_recursive(self, i, j, state, n, n2, window_size,
                                  global_visited_outside, global_visited_inside, window_visited):
        """
        Recursive outside backtrace with global caching.
        Returns (left_string, right_string) tuple for structure outside [i, j].
        """
        # Check cache
        if (state, i, j) in global_visited_outside:
            return global_visited_outside[(state, i, j)]

        if state == STATE_P:
            idx = STATE_P * n2 + j * n + i
            manner = int(self._beta_manners[idx])

            if manner == MANNER_NONE or manner == 0:
                # Terminal case - at external loop
                # Build external structure
                left_ext = self._trace_inside_recursive(0, i - 1, -1, n, n2, window_size,
                                                        global_visited_inside, window_visited) if i > 0 else ""
                right_ext = self._trace_inside_recursive(0, n - 1, -1, n, n2, window_size,
                                                         global_visited_inside, window_visited)[j + 1:] if j < n - 1 else ""
                # Actually, for terminal case we need to build from C state
                # Let's trace the external loop parts
                c_left = ""
                c_right = ""
                if i > 0:
                    c_left = self._trace_external_left(0, i - 1, n, n2, window_size, global_visited_inside, window_visited)
                if j < n - 1:
                    c_right = self._trace_external_right_recursive(j + 1, n - 1, n, n2, window_size, global_visited_inside, window_visited)
                return (c_left, c_right)

            elif manner == MANNER_HELIX:
                # P[i,j] <- P[i-1, j+1] via HELIX
                p, q = i - 1, j + 1
                outer = self._trace_outside_recursive(p, q, STATE_P, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                result = (outer[0] + '(', ')' + outer[1])
                global_visited_outside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, p, q, n, window_size)
                return result

            elif manner == MANNER_SINGLE:
                # P[i,j] <- P[p,q] via SINGLE
                l1, l2 = int(self._beta_l1s[idx]), int(self._beta_l2s[idx])
                p, q = i - l1, j + l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)
                outer = self._trace_outside_recursive(p, q, STATE_P, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                result = (outer[0] + '(' + left, right + ')' + outer[1])
                global_visited_outside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, p, q, n, window_size)
                return result

            elif manner == MANNER_M_eq_P:
                # P[i,j] <- M[i,j] via M_eq_P
                outer = self._trace_outside_recursive(i, j, STATE_M, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                global_visited_outside[(STATE_P, i, j)] = outer
                return outer

            elif manner == MANNER_M2_eq_M_plus_P:
                # P[i,j] is the right part of M2 = M + P
                split = int(self._beta_splits[idx])
                if split < i:
                    # M is [split, i-1], we are P[i, j], M2 is [split, j]
                    m_inside = self._trace_inside_recursive(split, i - 1, STATE_M, n, n2, window_size,
                                                            global_visited_inside, window_visited)
                    outer = self._trace_outside_recursive(split, j, STATE_M2, n, n2, window_size,
                                                          global_visited_outside, global_visited_inside, window_visited)
                    result = (outer[0] + m_inside, outer[1])
                    global_visited_outside[(STATE_P, i, j)] = result
                    return result
                else:
                    return ("", "")

            elif manner == MANNER_C_eq_C_plus_P:
                # P[i,j] <- C via C_eq_C_plus_P (external loop)
                # C code: when split == -1, trace C[0..i-1] as inside (left part)
                # and C[0..j] as outside (right part comes from C's beta)
                kk = i - 1  # Position before P[i,j]
                jj = j

                if kk >= 0:  # i > 0, there's C[0..kk] to the left
                    # Trace INSIDE C[0..kk] to get structure before P
                    c_inside = self._trace_inside_recursive(0, kk, -1, n, n2, window_size,
                                                            global_visited_inside, window_visited)
                else:
                    c_inside = ""

                # Trace right part of external loop (positions j+1 to n-1)
                # Following C code: trace what comes after C[0..j] in the external loop
                if jj < n - 1:
                    # Get the external loop structure from j+1 to n-1
                    c_right = self._trace_c_outside_right(jj, n, n2, window_size,
                                                          global_visited_outside, global_visited_inside, window_visited)
                else:
                    c_right = ""

                result = (c_inside, c_right)
                global_visited_outside[(STATE_P, i, j)] = result
                return result

            else:
                return ("", "")

        elif state == STATE_M:
            idx = STATE_M * n2 + j * n + i
            manner = int(self._beta_manners[idx])

            if manner == MANNER_M_eq_M_plus_U:
                # M[i,j] <- M[i,j+1]
                outer = self._trace_outside_recursive(i, j + 1, STATE_M, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                result = (outer[0], '.' + outer[1])
                global_visited_outside[(STATE_M, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M2:
                outer = self._trace_outside_recursive(i, j, STATE_M2, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                global_visited_outside[(STATE_M, i, j)] = outer
                return outer

            elif manner == MANNER_M2_eq_M_plus_P:
                # M[i,j] is the left part, P is to the right
                split = int(self._beta_splits[idx])
                if split > j:
                    # P is [j+1, split]
                    p_inside = self._trace_inside_recursive(j + 1, split, STATE_P, n, n2, window_size,
                                                            global_visited_inside, window_visited)
                    outer = self._trace_outside_recursive(i, split, STATE_M2, n, n2, window_size,
                                                          global_visited_outside, global_visited_inside, window_visited)
                    result = (outer[0], p_inside + outer[1])
                    global_visited_outside[(STATE_M, i, j)] = result
                    return result
                else:
                    return ("", "")

            else:
                return ("", "")

        elif state == STATE_M2:
            idx = STATE_M2 * n2 + j * n + i
            manner = int(self._beta_manners[idx])

            if manner == MANNER_MULTI:
                # M2[i,j] <- Multi[p,q]
                l1, l2 = int(self._beta_l1s[idx]), int(self._beta_l2s[idx])
                p, q = i - l1, j + l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)
                outer = self._trace_outside_recursive(p, q, STATE_MULTI, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                result = (outer[0] + left, right + outer[1])
                global_visited_outside[(STATE_M2, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M2:
                outer = self._trace_outside_recursive(i, j, STATE_M, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                global_visited_outside[(STATE_M2, i, j)] = outer
                return outer

            else:
                return ("", "")

        elif state == STATE_MULTI:
            idx = STATE_MULTI * n2 + j * n + i
            manner = int(self._beta_manners[idx])

            if manner == MANNER_MULTI_eq_MULTI_plus_U:
                # Multi[i,j] <- Multi[i, j_next]
                ext = int(self._beta_splits[idx])
                j_next = j + ext
                right = '.' * ext
                outer = self._trace_outside_recursive(i, j_next, STATE_MULTI, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                result = (outer[0], right + outer[1])
                global_visited_outside[(STATE_MULTI, i, j)] = result
                return result

            elif manner == MANNER_P_eq_MULTI:
                # Multi <- P
                outer = self._trace_outside_recursive(i, j, STATE_P, n, n2, window_size,
                                                      global_visited_outside, global_visited_inside, window_visited)
                result = (outer[0] + '(', ')' + outer[1])
                global_visited_outside[(STATE_MULTI, i, j)] = result
                self._window_fill(window_visited, i, j, n, window_size)
                return result

            else:
                return ("", "")

        return ("", "")

    def _trace_c_outside_right(self, j, n, n2, window_size, global_visited_outside, global_visited_inside, window_visited):
        """
        Trace C state's outside (beta) starting from position j to get the structure to the right.
        Following C code's get_parentheses_outside_real_backtrace for TYPE_C.
        Returns the structure string for positions j+1 to n-1.
        """
        # Check cache first
        cache_key = ("C", 0, j)
        if cache_key in global_visited_outside:
            return global_visited_outside[cache_key][1]  # Return right part

        if j >= n - 1:
            # At the end, nothing to the right
            global_visited_outside[cache_key] = ("", "")
            return ""

        # Get the beta manner for C[j]
        manner = int(self._beta_c_manners[j]) if j < n else MANNER_NONE

        if manner == MANNER_C_eq_C_plus_U:
            # C[0,j] <- C[0, j+1]
            # Add '.' to the right and recurse
            right_part = self._trace_c_outside_right(j + 1, n, n2, window_size,
                                                     global_visited_outside, global_visited_inside, window_visited)
            result = '.' + right_part
            global_visited_outside[cache_key] = ("", result)
            return result

        elif manner == MANNER_C_eq_C_plus_P:
            # C[0,j] <- C[0, split] + P[j+1, split]
            # The split stores j (P's end)
            split = int(self._beta_c_splits[j])
            if split > j:
                # Trace P[j+1, split] using inside traceback
                p_inside = self._trace_inside_recursive(j + 1, split, STATE_P, n, n2, window_size,
                                                        global_visited_inside, window_visited)
                # Trace C[0, split]'s outside to get what comes after
                right_after = self._trace_c_outside_right(split, n, n2, window_size,
                                                          global_visited_outside, global_visited_inside, window_visited)
                result = p_inside + right_after
                global_visited_outside[cache_key] = ("", result)
                return result
            else:
                # Invalid split, just return dots
                return '.' * (n - 1 - j) if j < n - 1 else ""

        elif manner == MANNER_NONE or manner == 0:
            # Terminal case - at the end of the sequence
            # Just return dots for remaining positions
            return '.' * (n - 1 - j) if j < n - 1 else ""

        else:
            # Unknown manner, return dots
            return '.' * (n - 1 - j) if j < n - 1 else ""

    def _trace_external_left(self, start, end, n, n2, window_size, global_visited_inside, window_visited):
        """Trace external loop from start to end (inclusive), returning structure string."""
        if start > end or end < 0:
            return ""
        return self._trace_inside_recursive(0, end, -1, n, n2, window_size, global_visited_inside, window_visited)

    def _trace_external_right_recursive(self, start, end, n, n2, window_size, global_visited_inside, window_visited):
        """Trace external loop from start to end, returning structure string."""
        if start > end:
            return ""

        # Build the external structure from start to end
        result = []
        j = end
        while j >= start:
            manner = int(self._c_manners[j]) if j < self._n else MANNER_NONE
            if manner == MANNER_C_eq_C_plus_U:
                result.append('.')
                j -= 1
            elif manner == MANNER_C_eq_C_plus_P:
                k = int(self._c_splits[j])
                if k >= start:
                    # Trace P[k+1, j]
                    p_str = self._trace_inside_recursive(k + 1, j, STATE_P, n, n2, window_size,
                                                         global_visited_inside, window_visited)
                    result.append(p_str)
                    j = k
                else:
                    # k < start, just trace P and stop
                    p_str = self._trace_inside_recursive(k + 1, j, STATE_P, n, n2, window_size,
                                                         global_visited_inside, window_visited)
                    # Only include the portion from start
                    if k + 1 <= start:
                        result.append(p_str[start - (k + 1):])
                    else:
                        result.append(p_str)
                    break
            else:
                result.append('.')
                j -= 1

        result.reverse()
        return ''.join(result)

    def _outside(self, n, nucs, next_pair, lv,
                 scores, manners, splits, l1s, l2s,
                 c_scores, c_manners, c_splits,
                 beta_scores, beta_manners, beta_splits, beta_l1s, beta_l2s,
                 beta_c_scores, beta_c_manners, beta_c_splits):
        """
        Outside pass: compute beta scores for suboptimal structures.
        Iterates backwards from j = n-1 to 0.
        """
        n2 = n * n

        # Initialize: beta[C][n-1] = 0
        beta_c_scores[n - 1] = 0.0
        beta_c_manners[n - 1] = MANNER_NONE

        for j in range(n - 1, -1, -1):
            nucj = nucs[j]
            nucj1 = nucs[j + 1] if j + 1 < n else -1

            # C beam: C[j] <- C[j+1]
            if j < n - 1:
                if beta_c_manners[j + 1] != MANNER_NONE or beta_c_scores[j + 1] > VALUE_MIN_FLOAT + 1e10:
                    newscore = beta_c_scores[j + 1]
                    if not lv:
                        newscore += _external_unpaired
                    if newscore > beta_c_scores[j]:
                        beta_c_scores[j] = newscore
                        beta_c_manners[j] = MANNER_C_eq_C_plus_U

            if j == 0:
                break

            # M beam
            for i in range(j):
                idx_m = STATE_M * n2 + j * n + i
                if manners[idx_m] == MANNER_NONE:
                    continue

                # M[i,j] <- M[i,j+1]
                if j < n - 1:
                    idx_m_next = STATE_M * n2 + (j + 1) * n + i
                    if beta_manners[idx_m_next] != MANNER_NONE:
                        newscore = beta_scores[idx_m_next]
                        if not lv:
                            newscore += _multi_unpaired
                        if newscore > beta_scores[idx_m]:
                            beta_scores[idx_m] = newscore
                            beta_manners[idx_m] = MANNER_M_eq_M_plus_U

            # M2 beam
            for i in range(j):
                idx_m2 = STATE_M2 * n2 + j * n + i
                if manners[idx_m2] == MANNER_NONE:
                    continue

                # M2[i,j] <- Multi[p,q] through MANNER_MULTI
                for p in range(max(0, i - SINGLE_MAX_LEN), i):
                    nucp = nucs[p]
                    q = next_pair[nucp, j]
                    if q != -1 and (i - p - 1) <= SINGLE_MAX_LEN:
                        idx_multi = STATE_MULTI * n2 + q * n + p
                        if beta_manners[idx_multi] != MANNER_NONE:
                            newscore = beta_scores[idx_multi]
                            if not lv:
                                newscore += _multi_unpaired * ((i - p - 1) + (q - j - 1))
                            if newscore > beta_scores[idx_m2]:
                                beta_scores[idx_m2] = newscore
                                beta_manners[idx_m2] = MANNER_MULTI
                                beta_l1s[idx_m2] = i - p
                                beta_l2s[idx_m2] = q - j

                # M2 <- M (M_eq_M2)
                idx_m = STATE_M * n2 + j * n + i
                if beta_manners[idx_m] != MANNER_NONE:
                    if beta_scores[idx_m] > beta_scores[idx_m2]:
                        beta_scores[idx_m2] = beta_scores[idx_m]
                        beta_manners[idx_m2] = MANNER_M_eq_M2

            # P beam
            for i in range(j):
                idx_p = STATE_P * n2 + j * n + i
                if manners[idx_p] == MANNER_NONE:
                    continue

                nuci = nucs[i]
                nuci_1 = nucs[i - 1] if i > 0 else -1

                if i > 0 and j < n - 1:
                    # P[i,j] <- P[p,q] through SINGLE/HELIX
                    for p in range(max(0, i - SINGLE_MAX_LEN), i):
                        nucp = nucs[p]
                        nucp1 = nucs[p + 1]
                        q = next_pair[nucp, j]

                        while q != -1 and (i - p) + (q - j) - 2 <= SINGLE_MAX_LEN:
                            nucq = nucs[q]
                            nucq_1 = nucs[q - 1]
                            idx_pq = STATE_P * n2 + q * n + p

                            if beta_manners[idx_pq] != MANNER_NONE:
                                if p == i - 1 and q == j + 1:
                                    # Helix extension
                                    if lv:
                                        score_ext = self._v_score_single_fast(p, q, i, j, nucs)
                                    else:
                                        score_ext = self._score_helix(nucp, nucp1, nucq_1, nucq)
                                    newscore = beta_scores[idx_pq] + score_ext
                                    if newscore > beta_scores[idx_p]:
                                        beta_scores[idx_p] = newscore
                                        beta_manners[idx_p] = MANNER_HELIX
                                        beta_l1s[idx_p] = 1
                                        beta_l2s[idx_p] = 1
                                else:
                                    # Single loop (bulge/internal)
                                    if lv:
                                        score_ext = self._v_score_single_fast(p, q, i, j, nucs)
                                    else:
                                        score_ext = self._score_single(p, q, i, j, nucs)
                                    newscore = beta_scores[idx_pq] + score_ext
                                    if newscore > beta_scores[idx_p]:
                                        beta_scores[idx_p] = newscore
                                        beta_manners[idx_p] = MANNER_SINGLE
                                        beta_l1s[idx_p] = i - p
                                        beta_l2s[idx_p] = q - j

                            q = next_pair[nucp, q]

                # P <- M (M_eq_P)
                idx_m = STATE_M * n2 + j * n + i
                if i > 0 and j < n - 1 and beta_manners[idx_m] != MANNER_NONE:
                    if lv:
                        score_m1 = self._v_score_M1_fast(i, j, nucs, n)
                    else:
                        score_m1 = self._score_M1(i, j, nuci_1, nuci, nucj, nucj1, n)
                    newscore = beta_scores[idx_m] + score_m1
                    if newscore > beta_scores[idx_p]:
                        beta_scores[idx_p] = newscore
                        beta_manners[idx_p] = MANNER_M_eq_P

                # P contributes to M2 = M + P
                k = i - 1
                if k > 0:
                    if lv:
                        m1_score = self._v_score_M1_fast(i, j, nucs, n)
                    else:
                        m1_score = self._score_M1(i, j, nuci_1, nuci, nucj, nucj1, n)

                    for newi in range(k):
                        idx_m_ki = STATE_M * n2 + k * n + newi
                        if manners[idx_m_ki] == MANNER_NONE:
                            continue

                        idx_m2_ji = STATE_M2 * n2 + j * n + newi
                        if beta_manners[idx_m2_ji] != MANNER_NONE:
                            # Update M[newi, k]
                            m_alpha = scores[idx_m_ki]
                            newscore = beta_scores[idx_m2_ji] + scores[idx_p] + m1_score
                            if newscore > beta_scores[idx_m_ki]:
                                beta_scores[idx_m_ki] = newscore
                                beta_manners[idx_m_ki] = MANNER_M2_eq_M_plus_P
                                beta_splits[idx_m_ki] = j

                            # Update P[i, j]
                            newscore2 = beta_scores[idx_m2_ji] + m_alpha + m1_score
                            if newscore2 > beta_scores[idx_p]:
                                beta_scores[idx_p] = newscore2
                                beta_manners[idx_p] = MANNER_M2_eq_M_plus_P
                                beta_splits[idx_p] = newi

                # P <- C (C_eq_C_plus_P)
                k = i - 1
                if k >= 0:
                    if lv:
                        ext_score = self._v_score_external_paired_fast(k + 1, j, nucs, n)
                    else:
                        ext_score = self._score_external_paired(k + 1, j, nuci_1, nuci, nucj, nucj1, n)

                    newscore = beta_c_scores[j] + ext_score

                    # Update C[k] - no conditional check (gold standard doesn't have one)
                    c_newscore = scores[idx_p] + newscore
                    if c_newscore > beta_c_scores[k]:
                        beta_c_scores[k] = c_newscore
                        beta_c_manners[k] = MANNER_C_eq_C_plus_P
                        beta_c_splits[k] = j  # Store P's end position

                    # Update P[i, j]
                    p_newscore = c_scores[k] + newscore
                    if p_newscore > beta_scores[idx_p]:
                        beta_scores[idx_p] = p_newscore
                        beta_manners[idx_p] = MANNER_C_eq_C_plus_P
                        beta_splits[idx_p] = -1
                elif i == 0:
                    if lv:
                        ext_score = self._v_score_external_paired_fast(0, j, nucs, n)
                    else:
                        ext_score = self._score_external_paired(0, j, -1, nuci, nucj, nucj1, n)
                    newscore = beta_c_scores[j] + ext_score
                    if newscore > beta_scores[idx_p]:
                        beta_scores[idx_p] = newscore
                        beta_manners[idx_p] = MANNER_C_eq_C_plus_P
                        beta_splits[idx_p] = -1

            # Multi beam
            for i in range(j):
                idx_multi = STATE_MULTI * n2 + j * n + i
                if manners[idx_multi] == MANNER_NONE:
                    continue

                nuci = nucs[i]
                nuci1 = nucs[i + 1]
                jnext = next_pair[nuci, j]

                # Multi[i,j] <- Multi[i, jnext]
                if jnext != -1:
                    idx_multi_next = STATE_MULTI * n2 + jnext * n + i
                    if beta_manners[idx_multi_next] != MANNER_NONE:
                        newscore = beta_scores[idx_multi_next]
                        if not lv:
                            newscore += _multi_unpaired * (jnext - j - 1)
                        if newscore > beta_scores[idx_multi]:
                            beta_scores[idx_multi] = newscore
                            beta_manners[idx_multi] = MANNER_MULTI_eq_MULTI_plus_U
                            beta_splits[idx_multi] = jnext - j

                # Multi <- P (P_eq_MULTI)
                idx_p = STATE_P * n2 + j * n + i
                if beta_manners[idx_p] != MANNER_NONE:
                    if lv:
                        score_multi = self._v_score_multi_fast(i, j, nucs, n)
                    else:
                        score_multi = self._score_multi(i, j, nuci, nuci1, nucs[j - 1], nucj, n)
                    newscore = beta_scores[idx_p] + score_multi
                    if newscore > beta_scores[idx_multi]:
                        beta_scores[idx_multi] = newscore
                        beta_manners[idx_multi] = MANNER_P_eq_MULTI

    def _backtrace_subopt(self, start_i, start_j, n, n2,
                          scores, manners, splits, l1s, l2s,
                          c_scores, c_manners, c_splits,
                          beta_scores, beta_manners, beta_splits, beta_l1s, beta_l2s,
                          beta_c_scores, beta_c_manners):
        """
        Backtrace a suboptimal structure starting from base pair (start_i, start_j).
        Uses both inside and outside information.
        """
        result = ['.'] * n

        # First, trace outside from (start_i, start_j) to external loop
        outside_left, outside_right = self._trace_outside(start_i, start_j, n, n2,
                                                          scores, manners, splits, l1s, l2s,
                                                          beta_scores, beta_manners, beta_splits, beta_l1s, beta_l2s,
                                                          beta_c_scores, beta_c_manners)

        # Then, trace inside from (start_i, start_j)
        inside = self._trace_inside(start_i, start_j, n, n2,
                                    scores, manners, splits, l1s, l2s,
                                    c_scores, c_manners, c_splits)

        # Combine: outside_left + inside + outside_right
        combined = outside_left + inside + outside_right

        # Verify length and return
        if len(combined) == n:
            return combined
        else:
            # Fallback: just trace inside
            return self._trace_inside_full(start_i, start_j, n, n2, manners, splits, l1s, l2s)

    def _trace_outside(self, i, j, n, n2,
                       scores, manners, splits, l1s, l2s,
                       beta_scores, beta_manners, beta_splits, beta_l1s, beta_l2s,
                       beta_c_scores, beta_c_manners):
        """Trace outside path from (i,j) to external loop, returning (left_string, right_string)."""
        # Ensure all inputs are Python ints to avoid numpy int16 overflow
        i = int(i)
        j = int(j)
        n = int(n)
        n2 = int(n2)

        left_parts = []
        right_parts = []

        curr_i, curr_j = i, j
        state_type = STATE_P

        while True:
            if state_type == STATE_P:
                idx = STATE_P * n2 + curr_j * n + curr_i
                manner = int(beta_manners[idx])

                if manner == MANNER_NONE or manner == 0:
                    break
                elif manner == MANNER_HELIX:
                    left_parts.append('(')
                    right_parts.append(')')  # append, not insert(0) - builds in correct positional order
                    curr_i = curr_i - 1
                    curr_j = curr_j + 1
                elif manner == MANNER_SINGLE:
                    l1 = int(beta_l1s[idx])
                    l2 = int(beta_l2s[idx])
                    left_parts.append('(' + '.' * (l1 - 1))
                    right_parts.append('.' * (l2 - 1) + ')')  # append, not insert(0)
                    curr_i = curr_i - l1
                    curr_j = curr_j + l2
                elif manner == MANNER_M_eq_P:
                    state_type = STATE_M
                elif manner == MANNER_M2_eq_M_plus_P:
                    # This P is the right part of M2 = M + P
                    # We need to trace M part inside, then continue with M2 outside
                    split = int(beta_splits[idx])
                    if split < curr_i:
                        # M is to the left
                        m_inside = self._trace_inside(int(split), int(curr_i - 1), n, n2,
                                                      scores, manners, splits, l1s, l2s,
                                                      None, None, None, state=STATE_M)
                        left_parts.append(m_inside)
                        state_type = STATE_M2
                        # curr_i stays the same (it's the start of M2)
                        curr_i = int(split)
                    else:
                        break
                elif manner == MANNER_C_eq_C_plus_P:
                    # External loop
                    split = int(beta_splits[idx])
                    if split >= 0:
                        c_inside = self._trace_inside(0, int(split), n, n2,
                                                      scores, manners, splits, l1s, l2s,
                                                      self._c_scores, self._c_manners, self._c_splits,
                                                      state=-1)
                        left_parts.append(c_inside)
                    # Trace remaining external loop to the right
                    if curr_j < n - 1:
                        right_str = self._trace_external_right(int(curr_j + 1), n, n2,
                                                               scores, manners, splits, l1s, l2s,
                                                               self._c_scores, self._c_manners, self._c_splits)
                        right_parts.append(right_str)  # append, not insert(0)
                    break
                else:
                    break

            elif state_type == STATE_M:
                idx = STATE_M * n2 + curr_j * n + curr_i
                manner = int(beta_manners[idx])

                if manner == MANNER_NONE or manner == 0:
                    break
                elif manner == MANNER_M_eq_M_plus_U:
                    right_parts.append('.')  # append, not insert(0)
                    curr_j = curr_j + 1
                elif manner == MANNER_M_eq_M2:
                    state_type = STATE_M2
                elif manner == MANNER_M2_eq_M_plus_P:
                    split = int(beta_splits[idx])
                    # P is to the right
                    p_inside = self._trace_inside(curr_j + 1, split, n, n2,
                                                  scores, manners, splits, l1s, l2s,
                                                  None, None, None, state=STATE_P)
                    right_parts.append(p_inside)  # append, not insert(0)
                    state_type = STATE_M2
                    curr_j = split
                else:
                    break

            elif state_type == STATE_M2:
                idx = STATE_M2 * n2 + curr_j * n + curr_i
                manner = int(beta_manners[idx])

                if manner == MANNER_NONE or manner == 0:
                    break
                elif manner == MANNER_MULTI:
                    l1 = int(beta_l1s[idx])
                    l2 = int(beta_l2s[idx])
                    left_parts.append('.' * (l1 - 1))
                    right_parts.append('.' * (l2 - 1))  # append, not insert(0)
                    curr_i = curr_i - l1
                    curr_j = curr_j + l2
                    state_type = STATE_MULTI
                elif manner == MANNER_M_eq_M2:
                    state_type = STATE_M
                else:
                    break

            elif state_type == STATE_MULTI:
                idx = STATE_MULTI * n2 + curr_j * n + curr_i
                manner = int(beta_manners[idx])

                if manner == MANNER_NONE or manner == 0:
                    break
                elif manner == MANNER_MULTI_eq_MULTI_plus_U:
                    ext = int(beta_splits[idx])
                    right_parts.append('.' * ext)  # append, not insert(0)
                    curr_j = curr_j + ext
                elif manner == MANNER_P_eq_MULTI:
                    left_parts.append('(')
                    right_parts.append(')')  # append, not insert(0)
                    state_type = STATE_P
                else:
                    break

            else:
                break

        # left_parts is built from inside-out (position i-1, i-2, i-3, ...)
        # We need to reverse it to get the correct order (position 0, 1, 2, ...)
        left_parts.reverse()
        return ''.join(left_parts), ''.join(right_parts)

    def _trace_inside(self, i, j, n, n2, scores, manners, splits, l1s, l2s,
                      c_scores, c_manners, c_splits, state=STATE_P):
        """Trace inside path, returning structure string."""
        # Ensure all inputs are Python ints to avoid numpy int16 overflow
        i = int(i)
        j = int(j)
        n = int(n)
        n2 = int(n2)
        state = int(state)

        result = ['.'] * (j - i + 1)

        if state == -1:
            # External loop (C state)
            stk = [(0, j, int(c_manners[j]) if j >= 0 else MANNER_NONE,
                   int(c_splits[j]) if j >= 0 else 0, 0, 0, -1)]
        else:
            idx = state * n2 + j * n + i
            stk = [(i, j, int(manners[idx]), int(splits[idx]),
                   int(l1s[idx]), int(l2s[idx]), state)]

        while stk:
            ci, cj, manner, split, l1, l2, cstate = stk.pop()
            # Ensure all values are Python ints
            ci, cj, manner, split, l1, l2, cstate = int(ci), int(cj), int(manner), int(split), int(l1), int(l2), int(cstate)

            if manner == MANNER_NONE or manner == MANNER_H:
                continue
            elif manner == MANNER_HAIRPIN:
                result[ci - i] = '('
                result[cj - i] = ')'
            elif manner == MANNER_SINGLE:
                result[ci - i] = '('
                result[cj - i] = ')'
                p, q = ci + l1, cj - l2
                idx = STATE_P * n2 + q * n + p
                stk.append((p, q, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_P))
            elif manner == MANNER_HELIX:
                result[ci - i] = '('
                result[cj - i] = ')'
                idx = STATE_P * n2 + (cj - 1) * n + (ci + 1)
                stk.append((ci + 1, cj - 1, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_P))
            elif manner == MANNER_MULTI:
                p, q = ci + l1, cj - l2
                idx = STATE_M2 * n2 + q * n + p
                stk.append((p, q, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M2))
            elif manner == MANNER_MULTI_eq_MULTI_plus_U:
                p, q = ci + l1, cj - l2
                idx = STATE_M2 * n2 + q * n + p
                stk.append((p, q, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M2))
            elif manner == MANNER_P_eq_MULTI:
                result[ci - i] = '('
                result[cj - i] = ')'
                idx = STATE_MULTI * n2 + cj * n + ci
                stk.append((ci, cj, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_MULTI))
            elif manner == MANNER_M2_eq_M_plus_P:
                k = split
                idx_m = STATE_M * n2 + k * n + ci
                idx_p = STATE_P * n2 + cj * n + (k + 1)
                stk.append((ci, k, int(manners[idx_m]), int(splits[idx_m]),
                           int(l1s[idx_m]), int(l2s[idx_m]), STATE_M))
                stk.append((k + 1, cj, int(manners[idx_p]), int(splits[idx_p]),
                           int(l1s[idx_p]), int(l2s[idx_p]), STATE_P))
            elif manner == MANNER_M_eq_M2:
                idx = STATE_M2 * n2 + cj * n + ci
                stk.append((ci, cj, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M2))
            elif manner == MANNER_M_eq_M_plus_U:
                idx = STATE_M * n2 + (cj - 1) * n + ci
                stk.append((ci, cj - 1, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M))
            elif manner == MANNER_M_eq_P:
                idx = STATE_P * n2 + cj * n + ci
                stk.append((ci, cj, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_P))
            elif manner == MANNER_C_eq_C_plus_U:
                if cj > 0:
                    stk.append((0, cj - 1, int(c_manners[cj - 1]), int(c_splits[cj - 1]), 0, 0, -1))
            elif manner == MANNER_C_eq_C_plus_P:
                k = split
                if k >= 0:
                    stk.append((0, k, int(c_manners[k]), int(c_splits[k]), 0, 0, -1))
                    idx = STATE_P * n2 + cj * n + (k + 1)
                    stk.append((k + 1, cj, int(manners[idx]), int(splits[idx]),
                               int(l1s[idx]), int(l2s[idx]), STATE_P))
                else:
                    idx = STATE_P * n2 + cj * n + ci
                    stk.append((ci, cj, int(manners[idx]), int(splits[idx]),
                               int(l1s[idx]), int(l2s[idx]), STATE_P))

        return ''.join(result)

    def _trace_inside_full(self, start_i, start_j, n, n2, manners, splits, l1s, l2s):
        """Full inside trace starting from a specific pair, filling entire sequence."""
        result = ['.'] * n

        # Trace inside from (start_i, start_j)
        idx = STATE_P * n2 + start_j * n + start_i
        stk = [(start_i, start_j, int(manners[idx]), int(splits[idx]),
               int(l1s[idx]), int(l2s[idx]), STATE_P)]

        while stk:
            i, j, manner, split, l1, l2, state = stk.pop()

            if manner == MANNER_NONE or manner == MANNER_H:
                continue
            elif manner == MANNER_HAIRPIN:
                result[i] = '('
                result[j] = ')'
            elif manner == MANNER_SINGLE:
                result[i] = '('
                result[j] = ')'
                p, q = i + l1, j - l2
                idx = int(STATE_P * n2 + q * n + p)
                stk.append((p, q, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_P))
            elif manner == MANNER_HELIX:
                result[i] = '('
                result[j] = ')'
                idx = int(STATE_P * n2 + (j - 1) * n + (i + 1))
                stk.append((i + 1, j - 1, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_P))
            elif manner == MANNER_MULTI:
                p, q = i + l1, j - l2
                idx = int(STATE_M2 * n2 + q * n + p)
                stk.append((p, q, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M2))
            elif manner == MANNER_MULTI_eq_MULTI_plus_U:
                p, q = i + l1, j - l2
                idx = int(STATE_M2 * n2 + q * n + p)
                stk.append((p, q, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M2))
            elif manner == MANNER_P_eq_MULTI:
                result[i] = '('
                result[j] = ')'
                idx = int(STATE_MULTI * n2 + j * n + i)
                stk.append((i, j, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_MULTI))
            elif manner == MANNER_M2_eq_M_plus_P:
                k = split
                idx_m = int(STATE_M * n2 + k * n + i)
                idx_p = int(STATE_P * n2 + j * n + (k + 1))
                stk.append((i, k, int(manners[idx_m]), int(splits[idx_m]),
                           int(l1s[idx_m]), int(l2s[idx_m]), STATE_M))
                stk.append((k + 1, j, int(manners[idx_p]), int(splits[idx_p]),
                           int(l1s[idx_p]), int(l2s[idx_p]), STATE_P))
            elif manner == MANNER_M_eq_M2:
                idx = int(STATE_M2 * n2 + j * n + i)
                stk.append((i, j, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M2))
            elif manner == MANNER_M_eq_M_plus_U:
                idx = int(STATE_M * n2 + (j - 1) * n + i)
                stk.append((i, j - 1, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_M))
            elif manner == MANNER_M_eq_P:
                idx = int(STATE_P * n2 + j * n + i)
                stk.append((i, j, int(manners[idx]), int(splits[idx]),
                           int(l1s[idx]), int(l2s[idx]), STATE_P))

        return ''.join(result)

    def _trace_structure_with_pair(self, target_i, target_j, n, n2, nucs, next_pair,
                                   scores, manners, splits, l1s, l2s,
                                   c_scores, c_manners, c_splits):
        """
        Build a complete structure that contains the specified pair (target_i, target_j).
        This finds the best path through the DP that includes this pair.
        """
        result = ['.'] * n

        # Mark the target pair
        result[target_i] = '('
        result[target_j] = ')'

        # 1. Trace inside the target pair
        idx_p = STATE_P * n2 + target_j * n + target_i
        if manners[idx_p] != MANNER_NONE:
            self._fill_inside(target_i, target_j, n, n2, manners, splits, l1s, l2s, result)

        # 2. Find the best way to reach this pair from external loop
        # Look for enclosing pairs or direct external connection
        best_outer_score = VALUE_MIN_FLOAT
        best_outer_path = None

        # Option A: Direct external connection (pair is in external loop)
        # Check if there's a valid C state that includes this pair
        for k in range(target_i):
            if c_manners[k] != MANNER_NONE or k == 0:
                # Check if we can connect through external loop
                # C[k] -> P[target_i, target_j] -> C[target_j]
                ext_score = c_scores[k] if k >= 0 and c_scores[k] > VALUE_MIN_FLOAT + 1e10 else 0
                if ext_score > best_outer_score or best_outer_path is None:
                    best_outer_score = ext_score
                    best_outer_path = ('external', k)

        # Option B: Enclosing pair
        for p in range(target_i):
            for q in range(target_j + 1, n):
                idx_pq = STATE_P * n2 + q * n + p
                if manners[idx_pq] != MANNER_NONE and scores[idx_pq] > VALUE_MIN_FLOAT + 1e10:
                    # Check if (p,q) can enclose (target_i, target_j)
                    # via helix or single loop
                    if q - p <= SINGLE_MAX_LEN + target_j - target_i + 2:
                        enc_score = scores[idx_pq]
                        if enc_score > best_outer_score:
                            best_outer_score = enc_score
                            best_outer_path = ('enclosing', p, q)

        # 3. Build outer structure based on best path
        if best_outer_path is not None:
            if best_outer_path[0] == 'external':
                # Fill external region before target_i
                k = best_outer_path[1]
                if k > 0:
                    self._fill_external(0, k, n, n2, manners, splits, l1s, l2s,
                                       c_scores, c_manners, c_splits, result)
                # Fill external region after target_j
                if target_j < n - 1:
                    self._fill_external(target_j + 1, n - 1, n, n2, manners, splits, l1s, l2s,
                                       c_scores, c_manners, c_splits, result)
            elif best_outer_path[0] == 'enclosing':
                p, q = best_outer_path[1], best_outer_path[2]
                result[p] = '('
                result[q] = ')'
                # Recursively build structure with enclosing pair
                outer = self._trace_structure_with_pair(p, q, n, n2, nucs, next_pair,
                                                        scores, manners, splits, l1s, l2s,
                                                        c_scores, c_manners, c_splits)
                # Merge outer structure (but keep our inner part)
                for pos in range(n):
                    if pos < target_i or pos > target_j:
                        if outer[pos] != '.':
                            result[pos] = outer[pos]

        return ''.join(result)

    def _fill_inside(self, i, j, n, n2, manners, splits, l1s, l2s, result):
        """Fill the structure inside pair (i,j)."""
        idx = STATE_P * n2 + j * n + i
        manner = manners[idx]

        if manner == MANNER_HAIRPIN or manner == MANNER_H:
            return
        elif manner == MANNER_SINGLE:
            l1, l2 = l1s[idx], l2s[idx]
            p, q = i + l1, j - l2
            result[p] = '('
            result[q] = ')'
            self._fill_inside(p, q, n, n2, manners, splits, l1s, l2s, result)
        elif manner == MANNER_HELIX:
            result[i + 1] = '('
            result[j - 1] = ')'
            self._fill_inside(i + 1, j - 1, n, n2, manners, splits, l1s, l2s, result)
        elif manner == MANNER_MULTI or manner == MANNER_P_eq_MULTI:
            # Multi-loop: need to find the branches
            self._fill_multi(i, j, n, n2, manners, splits, l1s, l2s, result)

    def _fill_multi(self, i, j, n, n2, manners, splits, l1s, l2s, result):
        """Fill multi-loop structure."""
        idx = STATE_MULTI * n2 + j * n + i
        manner = manners[idx]

        if manner == MANNER_MULTI:
            l1, l2 = l1s[idx], l2s[idx]
            p, q = i + l1, j - l2
            self._fill_m2(p, q, n, n2, manners, splits, l1s, l2s, result)
        elif manner == MANNER_MULTI_eq_MULTI_plus_U:
            l1, l2 = l1s[idx], l2s[idx]
            p, q = i + l1, j - l2
            self._fill_m2(p, q, n, n2, manners, splits, l1s, l2s, result)

    def _fill_m2(self, i, j, n, n2, manners, splits, l1s, l2s, result):
        """Fill M2 state (at least 2 branches in multi-loop)."""
        idx = STATE_M2 * n2 + j * n + i
        manner = manners[idx]

        if manner == MANNER_M2_eq_M_plus_P:
            k = splits[idx]
            # M branch [i, k]
            self._fill_m(i, k, n, n2, manners, splits, l1s, l2s, result)
            # P branch [k+1, j]
            result[k + 1] = '('
            result[j] = ')'
            self._fill_inside(k + 1, j, n, n2, manners, splits, l1s, l2s, result)

    def _fill_m(self, i, j, n, n2, manners, splits, l1s, l2s, result):
        """Fill M state."""
        idx = STATE_M * n2 + j * n + i
        manner = manners[idx]

        if manner == MANNER_M_eq_P:
            result[i] = '('
            result[j] = ')'
            self._fill_inside(i, j, n, n2, manners, splits, l1s, l2s, result)
        elif manner == MANNER_M_eq_M2:
            self._fill_m2(i, j, n, n2, manners, splits, l1s, l2s, result)
        elif manner == MANNER_M_eq_M_plus_U:
            self._fill_m(i, j - 1, n, n2, manners, splits, l1s, l2s, result)

    def _fill_external(self, start, end, n, n2, manners, splits, l1s, l2s,
                       c_scores, c_manners, c_splits, result):
        """Fill external loop region [start, end]."""
        if start > end:
            return

        # Trace from C[end]
        j = end
        while j >= start:
            manner = c_manners[j]
            if manner == MANNER_C_eq_C_plus_U:
                j -= 1
            elif manner == MANNER_C_eq_C_plus_P:
                k = c_splits[j]
                if k >= start:
                    # P[k+1, j]
                    result[k + 1] = '('
                    result[j] = ')'
                    self._fill_inside(k + 1, j, n, n2, manners, splits, l1s, l2s, result)
                    j = k
                else:
                    break
            else:
                break

    def _trace_external_right(self, start_j, n, n2, scores, manners, splits, l1s, l2s,
                              c_scores, c_manners, c_splits):
        """Trace external loop from start_j to end of sequence."""
        result = ['.'] * (n - start_j)

        if start_j >= n:
            return ''

        j = n - 1
        stk = [(0, j, int(c_manners[j]), int(c_splits[j]), 0, 0, -1)]

        while stk:
            i, cj, manner, split, l1, l2, state = stk.pop()

            if manner == MANNER_NONE:
                continue
            elif manner == MANNER_C_eq_C_plus_U:
                if cj > start_j:
                    stk.append((0, cj - 1, int(c_manners[cj - 1]), int(c_splits[cj - 1]), 0, 0, -1))
            elif manner == MANNER_C_eq_C_plus_P:
                k = split
                if k >= start_j:
                    stk.append((0, k, int(c_manners[k]), int(c_splits[k]), 0, 0, -1))
                    idx = int(STATE_P * n2 + cj * n + (k + 1))
                    inner = self._trace_inside(k + 1, cj, n, n2, scores, manners, splits, l1s, l2s,
                                               c_scores, c_manners, c_splits, state=STATE_P)
                    for ci, ch in enumerate(inner):
                        if k + 1 + ci >= start_j:
                            result[k + 1 + ci - start_j] = ch
                elif k + 1 >= start_j:
                    idx = int(STATE_P * n2 + cj * n + (k + 1))
                    inner = self._trace_inside(k + 1, cj, n, n2, scores, manners, splits, l1s, l2s,
                                               c_scores, c_manners, c_splits, state=STATE_P)
                    for ci, ch in enumerate(inner):
                        if k + 1 + ci >= start_j:
                            result[k + 1 + ci - start_j] = ch

        return ''.join(result)

    # Scoring helper functions for outside pass
    def _v_score_single_fast(self, p, q, i, j, nucs):
        """Full Vienna single loop score (matching v_score_single numba function).

        Handles stacks, bulges with terminal AU penalties, and all internal loop
        special cases (int11, int21, int22, mismatch tables).
        """
        n = len(nucs)
        nucp = nucs[p]
        nucp1 = nucs[p + 1] if p + 1 < n else -1
        nucq_1 = nucs[q - 1] if q > 0 else -1
        nucq = nucs[q]
        nuci_1 = nucs[i - 1] if i > 0 else -1
        nuci = nucs[i]
        nucj = nucs[j]
        nucj1 = nucs[j + 1] if j + 1 < n else -1

        pt1 = NUM_TO_PAIR(nucp, nucq)  # outer pair type
        pt2 = NUM_TO_PAIR(nucj, nuci)  # inner pair type - REVERSED (as in gold standard)
        n1 = i - p - 1  # left unpaired
        n2 = q - j - 1  # right unpaired

        if n1 > n2:
            nl, ns = n1, n2
        else:
            nl, ns = n2, n1

        # Stack - return -energy (score convention, same as -v_score_single in gold standard)
        if nl == 0:
            return -_v_stack[pt1][pt2]

        # Bulge
        if ns == 0:
            if nl <= 30:
                energy = _v_bulge[nl]
            else:
                energy = _v_bulge[30] + int(107.856 * np.log(nl / 30.0))
            if nl == 1:
                energy += _v_stack[pt1][pt2]
            else:
                if pt1 > 2:  # Not CG or GC
                    energy += _v_terminalAU
                if pt2 > 2:
                    energy += _v_terminalAU
            return -energy

        # Internal loop - get Vienna nucleotide indices
        si1 = _nuc_table[nucp1] if nucp1 >= 0 and nucp1 <= 4 else -1
        sj1 = _nuc_table[nucq_1] if nucq_1 >= 0 and nucq_1 <= 4 else -1
        sp1 = _nuc_table[nuci_1] if nuci_1 >= 0 and nuci_1 <= 4 else -1
        sq1 = _nuc_table[nucj1] if nucj1 >= 0 and nucj1 <= 4 else -1

        # 1x1 internal loop
        if ns == 1 and nl == 1:
            if si1 >= 0 and sj1 >= 0:
                return -_v_int11[pt1][pt2][si1][sj1]
            return 0

        # 1x2 or 2x1 internal loop
        if ns == 1 and nl == 2:
            if n1 == 1:
                if si1 >= 0 and sq1 >= 0 and sj1 >= 0:
                    return -_v_int21[pt1][pt2][si1][sq1][sj1]
            else:
                if sq1 >= 0 and si1 >= 0 and sp1 >= 0:
                    return -_v_int21[pt2][pt1][sq1][si1][sp1]
            return 0

        # 2x2 internal loop
        if ns == 2 and nl == 2:
            if si1 >= 0 and sp1 >= 0 and sq1 >= 0 and sj1 >= 0:
                return -_v_int22[pt1][pt2][si1][sp1][sq1][sj1]
            return 0

        # General internal loop with mismatch energies
        # 1xn internal loop (n > 2)
        if ns == 1:
            u = nl + 1
            if u <= 30:
                energy = _v_internal[u]
            else:
                energy = _v_internal[30] + int(107.856 * np.log(u / 30.0))
            energy += min(_v_max_ninio, (nl - ns) * _v_ninio)
            if si1 >= 0 and sj1 >= 0:
                energy += _v_mismatch1nI[pt1][si1][sj1]
            if sq1 >= 0 and sp1 >= 0:
                energy += _v_mismatch1nI[pt2][sq1][sp1]
            return -energy

        # 2x3 internal loop (special case)
        if ns == 2 and nl == 3:
            energy = _v_internal[5] + _v_ninio
            if si1 >= 0 and sj1 >= 0:
                energy += _v_mismatch23I[pt1][si1][sj1]
            if sq1 >= 0 and sp1 >= 0:
                energy += _v_mismatch23I[pt2][sq1][sp1]
            return -energy

        # General internal loop
        u = nl + ns
        if u <= 30:
            energy = _v_internal[u]
        else:
            energy = _v_internal[30] + int(107.856 * np.log(u / 30.0))
        energy += min(_v_max_ninio, (nl - ns) * _v_ninio)
        if si1 >= 0 and sj1 >= 0:
            energy += _v_mismatchI[pt1][si1][sj1]
        if sq1 >= 0 and sp1 >= 0:
            energy += _v_mismatchI[pt2][sq1][sp1]
        return -energy

    def _score_helix(self, nucp, nucp1, nucq_1, nucq):
        """CONTRAfold helix score."""
        return (_base_pair[nucp * NOTON + nucq] +
                _helix_stacking[nucp * NOTONT + nucp1 * NOTOND + nucq_1 * NOTON + nucq])

    def _score_single(self, p, q, i, j, nucs):
        """CONTRAfold single loop score."""
        l1 = i - p - 1
        l2 = q - j - 1
        len_total = l1 + l2

        score = _base_pair[nucs[p] * NOTON + nucs[q]]

        if l1 == 0 and l2 == 0:
            # Stack
            score += _helix_stacking[nucs[p] * NOTONT + nucs[i] * NOTOND + nucs[j] * NOTON + nucs[q]]
        else:
            score += _internal_length[min(len_total, 30)]
            if l1 == l2:
                score += _internal_symmetric_length[min(l1, 15)]
            score += _internal_asymmetry[min(abs(l1 - l2), 28)]

        return score

    def _score_M1(self, i, j, nuci_1, nuci, nucj, nucj1, n):
        """CONTRAfold M1 score."""
        score = _multi_paired
        if nuci_1 >= 0:
            score += _dangle_left[nuci_1 * NOTOND + nuci * NOTON + nucj]
        if nucj1 >= 0 and j + 1 < n:
            score += _dangle_right[nuci * NOTOND + nucj * NOTON + nucj1]
        return score

    def _v_score_M1_fast(self, i, j, nucs, n):
        """Vienna M1 score (matching gold standard E_MLstem).

        Uses mismatchM37 when both neighbors are present.
        Uses dangles when only one neighbor is present (matching gold standard).
        Note: Uses Vienna nucleotide indices (1-4) for mismatch/dangle lookups.
        """
        nuci = nucs[i]
        nucj = nucs[j]
        type_ij = NUM_TO_PAIR(nuci, nucj)

        score = 0
        nuci_1 = nucs[i - 1] if i > 0 else -1
        nucj1 = nucs[j + 1] if j + 1 < n else -1

        # Convert to Vienna nucleotide indices for mismatch/dangle lookups
        si1 = _nuc_table[nuci_1] if nuci_1 >= 0 and nuci_1 <= 4 else -1
        sj1 = _nuc_table[nucj1] if nucj1 >= 0 and nucj1 <= 4 else -1

        # Use mismatchM when both neighbors present, otherwise use dangles
        if si1 >= 0 and sj1 >= 0:
            score += mismatchM37[type_ij][si1][sj1]
        elif si1 >= 0:
            score += _v_dangle5[type_ij][si1]
        elif sj1 >= 0:
            score += _v_dangle3[type_ij][sj1]

        if type_ij > 2:  # Not CG or GC
            score += _v_terminalAU

        score += _v_ml_intern

        return -score

    def _score_external_paired(self, i, j, nuci_1, nuci, nucj, nucj1, n):
        """CONTRAfold external paired score."""
        score = _external_paired
        if nuci_1 >= 0:
            score += _dangle_left[nuci_1 * NOTOND + nuci * NOTON + nucj]
        if nucj1 >= 0 and j + 1 < n:
            score += _dangle_right[nuci * NOTOND + nucj * NOTON + nucj1]
        return score

    def _v_score_external_paired_fast(self, i, j, nucs, n):
        """Vienna external paired score.

        Uses mismatchExt37 when both neighbors are present (matching C behavior),
        otherwise uses dangle5 or dangle3.
        Note: Uses Vienna nucleotide indices (1-4) for mismatch/dangle lookups.
        """
        nuci = nucs[i]
        nucj = nucs[j]
        type_ij = NUM_TO_PAIR(nuci, nucj)

        score = 0
        if type_ij > 2:  # Not CG or GC
            score += _v_terminalAU

        nuci_1 = nucs[i - 1] if i > 0 else -1
        nucj1 = nucs[j + 1] if j + 1 < n else -1

        # Convert to Vienna nucleotide indices for mismatch/dangle lookups
        si1 = _nuc_table[nuci_1] if nuci_1 >= 0 and nuci_1 <= 4 else -1
        sj1 = _nuc_table[nucj1] if nucj1 >= 0 and nucj1 <= 4 else -1

        # Use mismatchExt when both neighbors present, otherwise dangle (matching C code)
        if si1 >= 0 and sj1 >= 0:
            score += mismatchExt37[type_ij][si1][sj1]
        elif si1 >= 0:
            score += _v_dangle5[type_ij][si1]
        elif sj1 >= 0:
            score += _v_dangle3[type_ij][sj1]

        return -score

    def _score_multi(self, i, j, nuci, nuci1, nucj_1, nucj, n):
        """CONTRAfold multi score."""
        return _multi_base + _multi_paired

    def _v_score_multi_fast(self, i, j, nucs, n):
        """Vienna multi score (matching gold standard v_score_multi).

        For multi-loop closing pair, the pair type is REVERSED: (nucj, nuci).
        Uses E_MLstem logic with reversed flanking nucleotides (nucj_1, nuci1).
        Uses dangles when only one neighbor is present (matching gold standard).
        Note: Uses Vienna nucleotide indices (1-4) for mismatch/dangle lookups.
        """
        nuci = nucs[i]
        nucj = nucs[j]
        nuci1 = nucs[i + 1] if i + 1 < n else -1
        nucj_1 = nucs[j - 1] if j > 0 else -1

        # CRITICAL: For multi-loop closing, the pair type is REVERSED!
        type_ji = NUM_TO_PAIR(nucj, nuci)

        # Convert to Vienna nucleotide indices for mismatch/dangle lookups
        # Note: si1=nucj_1, sj1=nuci1 (reversed order for closing pair)
        si1 = _nuc_table[nucj_1] if nucj_1 >= 0 and nucj_1 <= 4 else -1
        sj1 = _nuc_table[nuci1] if nuci1 >= 0 and nuci1 <= 4 else -1

        # E_MLstem logic with reversed flanking nucleotides
        score = 0
        if si1 >= 0 and sj1 >= 0:
            score += mismatchM37[type_ji][si1][sj1]
        elif si1 >= 0:
            score += _v_dangle5[type_ji][si1]
        elif sj1 >= 0:
            score += _v_dangle3[type_ji][sj1]

        if type_ji > 2:  # Not CG or GC
            score += _v_terminalAU

        score += _v_ml_intern
        score += _v_ml_closing

        return -score

    def _evaluate_structure(self, seq, structure, lv):
        """
        Re-evaluate a structure to get its correct energy.
        Uses ViennaRNA if available, otherwise falls back to simple evaluation.
        """
        try:
            import RNA
            # Use ViennaRNA for accurate evaluation
            fc = RNA.fold_compound(seq)
            energy = fc.eval_structure(structure)
            return energy
        except ImportError:
            # Fallback: use our own evaluation
            return self._simple_eval(seq, structure, lv)

    def _simple_eval(self, seq, structure, lv):
        """
        Simple structure evaluation using Vienna parameters.
        This is a simplified version - may not be 100% accurate.
        """
        n = len(seq)
        nucs = np.zeros(n, dtype=np.int32)
        for i, c in enumerate(seq):
            nucs[i] = Utils.shared.get_acgu_num_c(c)

        # Build pair table
        pairs = [-1] * n
        stack = []
        for i, c in enumerate(structure):
            if c == '(':
                stack.append(i)
            elif c == ')':
                if stack:
                    j = stack.pop()
                    pairs[j] = i
                    pairs[i] = j

        if not lv:
            # CONTRAfold scoring (simplified)
            score = 0.0
            for i in range(n):
                if pairs[i] > i:
                    j = pairs[i]
                    # Base pair score
                    score += _base_pair[nucs[i] * NOTON + nucs[j]]
                    # Check for stacking
                    if i + 1 < j and pairs[i + 1] == j - 1:
                        score += _helix_stacking[nucs[i] * NOTONT + nucs[i+1] * NOTOND + nucs[j-1] * NOTON + nucs[j]]
            return score
        else:
            # Vienna scoring
            energy = 0

            # Find all helices and loops
            i = 0
            while i < n:
                if pairs[i] > i:
                    # Found a base pair, trace the helix
                    j = pairs[i]
                    # Stack energy
                    p, q = i, j
                    while p + 1 < q - 1 and pairs[p + 1] == q - 1:
                        type1 = NUM_TO_PAIR(nucs[p], nucs[q])
                        type2 = NUM_TO_PAIR(nucs[p + 1], nucs[q - 1])
                        if type1 > 0 and type2 > 0:
                            energy += _v_stack[type1][type2]
                        p += 1
                        q -= 1

                    # Check loop type
                    inner_i, inner_j = p, q
                    # Count unpaired bases and inner pairs
                    unpaired = 0
                    inner_pairs = 0
                    for k in range(inner_i + 1, inner_j):
                        if pairs[k] == -1:
                            unpaired += 1
                        elif pairs[k] > k:
                            inner_pairs += 1

                    if inner_pairs == 0:
                        # Hairpin
                        loop_len = inner_j - inner_i - 1
                        if loop_len <= 30:
                            energy += _v_hairpin[loop_len]
                    elif inner_pairs == 1:
                        # Internal or bulge
                        # Find the inner pair
                        for k in range(inner_i + 1, inner_j):
                            if pairs[k] > k:
                                l1 = k - inner_i - 1
                                l2 = inner_j - pairs[k] - 1
                                if l1 == 0 or l2 == 0:
                                    # Bulge
                                    length = l1 + l2
                                    energy += _v_bulge[min(length, 30)]
                                else:
                                    # Internal loop
                                    energy += _v_internal[min(l1 + l2, 30)]
                                break
                    else:
                        # Multi-loop
                        energy += _v_ml_closing
                        energy += _v_ml_intern * inner_pairs
                    i = j + 1
                else:
                    i += 1

            return energy / -100.0  # Convert to kcal/mol

    def _get_parentheses_inside_real_backtrace(self, i, j, state, n, n2,
                                                global_visited_inside, window_visited):
        """
        Inside backtrace with global caching, following C code pattern.
        Returns structure string for region [i, j].

        This follows the C function get_parentheses_inside_real_backtrace exactly.
        """
        # Check cache
        if (state, i, j) in global_visited_inside:
            return global_visited_inside[(state, i, j)]

        scores = self._scores
        manners = self._manners
        splits = self._splits
        l1s = self._l1s
        l2s = self._l2s

        if state == STATE_P:
            idx = STATE_P * n2 + j * n + i
            manner = int(manners[idx])

            if manner == MANNER_H or manner == MANNER_NONE:
                return ""

            elif manner == MANNER_HAIRPIN:
                inner = '.' * (j - i - 1)
                result = '(' + inner + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, self._window_size)
                return result

            elif manner == MANNER_SINGLE:
                l1, l2 = int(l1s[idx]), int(l2s[idx])
                p, q = i + l1, j - l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)

                if (STATE_P, p, q) not in global_visited_inside:
                    global_visited_inside[(STATE_P, p, q)] = self._get_parentheses_inside_real_backtrace(
                        p, q, STATE_P, n, n2, global_visited_inside, window_visited)

                result = '(' + left + global_visited_inside[(STATE_P, p, q)] + right + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, self._window_size)
                return result

            elif manner == MANNER_HELIX:
                if (STATE_P, i + 1, j - 1) not in global_visited_inside:
                    global_visited_inside[(STATE_P, i + 1, j - 1)] = self._get_parentheses_inside_real_backtrace(
                        i + 1, j - 1, STATE_P, n, n2, global_visited_inside, window_visited)

                result = '(' + global_visited_inside[(STATE_P, i + 1, j - 1)] + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, self._window_size)
                return result

            elif manner == MANNER_P_eq_MULTI:
                if (STATE_MULTI, i, j) not in global_visited_inside:
                    global_visited_inside[(STATE_MULTI, i, j)] = self._get_parentheses_inside_real_backtrace(
                        i, j, STATE_MULTI, n, n2, global_visited_inside, window_visited)

                result = '(' + global_visited_inside[(STATE_MULTI, i, j)] + ')'
                global_visited_inside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, i, j, n, self._window_size)
                return result

            else:
                return '(' + '.' * (j - i - 1) + ')'

        elif state == STATE_MULTI:
            idx = STATE_MULTI * n2 + j * n + i
            manner = int(manners[idx])

            if manner == MANNER_MULTI or manner == MANNER_MULTI_eq_MULTI_plus_U:
                l1, l2 = int(l1s[idx]), int(l2s[idx])
                p, q = i + l1, j - l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)

                if (STATE_M2, p, q) not in global_visited_inside:
                    global_visited_inside[(STATE_M2, p, q)] = self._get_parentheses_inside_real_backtrace(
                        p, q, STATE_M2, n, n2, global_visited_inside, window_visited)

                result = left + global_visited_inside[(STATE_M2, p, q)] + right
                global_visited_inside[(STATE_MULTI, i, j)] = result
                return result

            else:
                return '.' * (j - i + 1)

        elif state == STATE_M2:
            idx = STATE_M2 * n2 + j * n + i
            manner = int(manners[idx])

            if manner == MANNER_M2_eq_M_plus_P:
                k = int(splits[idx])

                if (STATE_M, i, k) not in global_visited_inside:
                    global_visited_inside[(STATE_M, i, k)] = self._get_parentheses_inside_real_backtrace(
                        i, k, STATE_M, n, n2, global_visited_inside, window_visited)
                if (STATE_P, k + 1, j) not in global_visited_inside:
                    global_visited_inside[(STATE_P, k + 1, j)] = self._get_parentheses_inside_real_backtrace(
                        k + 1, j, STATE_P, n, n2, global_visited_inside, window_visited)

                result = global_visited_inside[(STATE_M, i, k)] + global_visited_inside[(STATE_P, k + 1, j)]
                global_visited_inside[(STATE_M2, i, j)] = result
                return result

            else:
                return '.' * (j - i + 1)

        elif state == STATE_M:
            idx = STATE_M * n2 + j * n + i
            manner = int(manners[idx])

            if manner == MANNER_M_eq_P:
                if (STATE_P, i, j) not in global_visited_inside:
                    global_visited_inside[(STATE_P, i, j)] = self._get_parentheses_inside_real_backtrace(
                        i, j, STATE_P, n, n2, global_visited_inside, window_visited)

                result = global_visited_inside[(STATE_P, i, j)]
                global_visited_inside[(STATE_M, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M2:
                if (STATE_M2, i, j) not in global_visited_inside:
                    global_visited_inside[(STATE_M2, i, j)] = self._get_parentheses_inside_real_backtrace(
                        i, j, STATE_M2, n, n2, global_visited_inside, window_visited)

                result = global_visited_inside[(STATE_M2, i, j)]
                global_visited_inside[(STATE_M, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M_plus_U:
                if (STATE_M, i, j - 1) not in global_visited_inside:
                    global_visited_inside[(STATE_M, i, j - 1)] = self._get_parentheses_inside_real_backtrace(
                        i, j - 1, STATE_M, n, n2, global_visited_inside, window_visited)

                result = global_visited_inside[(STATE_M, i, j - 1)] + '.'
                global_visited_inside[(STATE_M, i, j)] = result
                return result

            else:
                return '.' * (j - i + 1)

        elif state == -1:  # External loop (C state)
            c_manners = self._c_manners
            c_splits = self._c_splits

            if j < 0:
                return ""

            manner = int(c_manners[j])

            if manner == MANNER_C_eq_C_plus_U:
                if j > 0:
                    if (-1, 0, j - 1) not in global_visited_inside:
                        global_visited_inside[(-1, 0, j - 1)] = self._get_parentheses_inside_real_backtrace(
                            0, j - 1, -1, n, n2, global_visited_inside, window_visited)
                    result = global_visited_inside[(-1, 0, j - 1)] + '.'
                else:
                    result = '.'
                global_visited_inside[(-1, 0, j)] = result
                return result

            elif manner == MANNER_C_eq_C_plus_P:
                k = int(c_splits[j])
                if k >= 0:
                    if (-1, 0, k) not in global_visited_inside:
                        global_visited_inside[(-1, 0, k)] = self._get_parentheses_inside_real_backtrace(
                            0, k, -1, n, n2, global_visited_inside, window_visited)
                    c_part = global_visited_inside[(-1, 0, k)]
                else:
                    c_part = ""

                if (STATE_P, k + 1, j) not in global_visited_inside:
                    global_visited_inside[(STATE_P, k + 1, j)] = self._get_parentheses_inside_real_backtrace(
                        k + 1, j, STATE_P, n, n2, global_visited_inside, window_visited)

                result = c_part + global_visited_inside[(STATE_P, k + 1, j)]
                global_visited_inside[(-1, 0, j)] = result
                return result

            else:
                return '.' * (j + 1)

        return '.' * (j - i + 1)

    def _get_parentheses_outside_real_backtrace(self, i, j, state, n, n2,
                                                 global_visited_outside, global_visited_inside,
                                                 window_visited):
        """
        Outside backtrace with global caching, following C code pattern exactly.
        Returns (left_string, right_string) tuple for structure outside [i, j].

        This follows the C function get_parentheses_outside_real_backtrace exactly.
        """
        # Check cache
        if (state, i, j) in global_visited_outside:
            return global_visited_outside[(state, i, j)]

        # Get beta arrays from member variables
        beta_manners = self._beta_manners
        beta_splits = self._beta_splits
        beta_l1s = self._beta_l1s
        beta_l2s = self._beta_l2s
        window_size = self._window_size

        if state == STATE_P:
            idx = STATE_P * n2 + j * n + i
            manner = int(beta_manners[idx])

            if manner == MANNER_NONE or manner == 0:
                # Terminal case - at external loop
                # Don't build the external right part - leave as dots (matching C behavior)
                c_left = ""
                if i > 0:
                    # Trace external loop left of (i, j)
                    c_left = self._get_parentheses_inside_real_backtrace(
                        0, i - 1, -1, n, n2, global_visited_inside, set())
                # Right part is empty - will be filled with dots in final structure
                return (c_left, "")

            elif manner == MANNER_SINGLE:
                l1, l2 = int(beta_l1s[idx]), int(beta_l2s[idx])
                p, q = i - l1, j + l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)

                if (STATE_P, p, q) not in global_visited_outside:
                    global_visited_outside[(STATE_P, p, q)] = self._get_parentheses_outside_real_backtrace(
                        p, q, STATE_P, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                outsider = global_visited_outside[(STATE_P, p, q)]
                result = (outsider[0] + '(' + left, right + ')' + outsider[1])
                global_visited_outside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, p, q, n, window_size)
                return result

            elif manner == MANNER_HELIX:
                p, q = i - 1, j + 1

                if (STATE_P, p, q) not in global_visited_outside:
                    global_visited_outside[(STATE_P, p, q)] = self._get_parentheses_outside_real_backtrace(
                        p, q, STATE_P, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                outsider = global_visited_outside[(STATE_P, p, q)]
                result = (outsider[0] + '(', ')' + outsider[1])
                global_visited_outside[(STATE_P, i, j)] = result
                self._window_fill(window_visited, p, q, n, window_size)
                return result

            elif manner == MANNER_M_eq_P:
                if (STATE_M, i, j) not in global_visited_outside:
                    global_visited_outside[(STATE_M, i, j)] = self._get_parentheses_outside_real_backtrace(
                        i, j, STATE_M, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                result = global_visited_outside[(STATE_M, i, j)]
                global_visited_outside[(STATE_P, i, j)] = result
                return result

            elif manner == MANNER_M2_eq_M_plus_P:
                split = int(beta_splits[idx])
                if split < i:
                    # P[i,j] is the right part, M is [split, i-1]
                    mm, kk = split, i - 1
                    ii, jj = i, j

                    if (STATE_M, mm, kk) not in global_visited_inside:
                        global_visited_inside[(STATE_M, mm, kk)] = self._get_parentheses_inside_real_backtrace(
                            mm, kk, STATE_M, n, n2, global_visited_inside, window_visited)

                    m_inside = global_visited_inside[(STATE_M, mm, kk)]

                    if (STATE_M2, mm, jj) not in global_visited_outside:
                        global_visited_outside[(STATE_M2, mm, jj)] = self._get_parentheses_outside_real_backtrace(
                            mm, jj, STATE_M2, n, n2,
                            global_visited_outside, global_visited_inside, window_visited)

                    outsider = global_visited_outside[(STATE_M2, mm, jj)]
                    result = (outsider[0] + m_inside, outsider[1])
                    global_visited_outside[(STATE_P, i, j)] = result
                    return result
                else:
                    return ("", "")

            elif manner == MANNER_C_eq_C_plus_P:
                # C = c + P, tracing P[i,j] means we need C outside and c inside
                # Following C code: get_parentheses_outside_real_backtrace lines 667-708
                kk = i - 1  # Position before P's start

                if kk >= 0:
                    # Get C[0..kk] inside (structure before P)
                    if (-1, 0, kk) not in global_visited_inside:
                        global_visited_inside[(-1, 0, kk)] = self._get_parentheses_inside_real_backtrace(
                            0, kk, -1, n, n2, global_visited_inside, set())
                    inside_C = global_visited_inside[(-1, 0, kk)]

                    # Get C[0..j] outside (external loop ending at j)
                    if (-1, 0, j) not in global_visited_outside:
                        if j < n - 1:
                            # Need to trace C's outside using beta_c
                            c_outsider = self._get_c_outside(j, n, n2, global_visited_outside, global_visited_inside)
                        else:
                            c_outsider = ("", "")
                        global_visited_outside[(-1, 0, j)] = c_outsider

                    outsider = global_visited_outside[(-1, 0, j)]
                    result = (outsider[0] + inside_C, outsider[1])
                else:
                    # P starts at position 0, no C before it
                    if (-1, 0, j) not in global_visited_outside:
                        if j < n - 1:
                            c_outsider = self._get_c_outside(j, n, n2, global_visited_outside, global_visited_inside)
                        else:
                            c_outsider = ("", "")
                        global_visited_outside[(-1, 0, j)] = c_outsider

                    result = global_visited_outside[(-1, 0, j)]

                global_visited_outside[(STATE_P, i, j)] = result
                return result

            else:
                return ("", "")

        elif state == STATE_M:
            idx = STATE_M * n2 + j * n + i
            manner = int(beta_manners[idx])

            if manner == MANNER_M_eq_M_plus_U:
                if (STATE_M, i, j + 1) not in global_visited_outside:
                    global_visited_outside[(STATE_M, i, j + 1)] = self._get_parentheses_outside_real_backtrace(
                        i, j + 1, STATE_M, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                outsider = global_visited_outside[(STATE_M, i, j + 1)]
                result = (outsider[0], '.' + outsider[1])
                global_visited_outside[(STATE_M, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M2:
                if (STATE_M2, i, j) not in global_visited_outside:
                    global_visited_outside[(STATE_M2, i, j)] = self._get_parentheses_outside_real_backtrace(
                        i, j, STATE_M2, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                result = global_visited_outside[(STATE_M2, i, j)]
                global_visited_outside[(STATE_M, i, j)] = result
                return result

            elif manner == MANNER_M2_eq_M_plus_P:
                split = int(beta_splits[idx])
                if split > j:
                    # M[i,j] is the left part, P is [j+1, split]
                    ii, jj = j + 1, split

                    if (STATE_P, ii, jj) not in global_visited_inside:
                        global_visited_inside[(STATE_P, ii, jj)] = self._get_parentheses_inside_real_backtrace(
                            ii, jj, STATE_P, n, n2, global_visited_inside, window_visited)

                    p_inside = global_visited_inside[(STATE_P, ii, jj)]

                    if (STATE_M2, i, jj) not in global_visited_outside:
                        global_visited_outside[(STATE_M2, i, jj)] = self._get_parentheses_outside_real_backtrace(
                            i, jj, STATE_M2, n, n2,
                            global_visited_outside, global_visited_inside, window_visited)

                    outsider = global_visited_outside[(STATE_M2, i, jj)]
                    result = (outsider[0], p_inside + outsider[1])
                    global_visited_outside[(STATE_M, i, j)] = result
                    return result
                else:
                    return ("", "")

            else:
                return ("", "")

        elif state == STATE_M2:
            idx = STATE_M2 * n2 + j * n + i
            manner = int(beta_manners[idx])

            if manner == MANNER_MULTI:
                l1, l2 = int(beta_l1s[idx]), int(beta_l2s[idx])
                p, q = i - l1, j + l2
                left = '.' * (l1 - 1)
                right = '.' * (l2 - 1)

                if (STATE_MULTI, p, q) not in global_visited_outside:
                    global_visited_outside[(STATE_MULTI, p, q)] = self._get_parentheses_outside_real_backtrace(
                        p, q, STATE_MULTI, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                outsider = global_visited_outside[(STATE_MULTI, p, q)]
                result = (outsider[0] + left, right + outsider[1])
                global_visited_outside[(STATE_M2, i, j)] = result
                return result

            elif manner == MANNER_M_eq_M2:
                if (STATE_M, i, j) not in global_visited_outside:
                    global_visited_outside[(STATE_M, i, j)] = self._get_parentheses_outside_real_backtrace(
                        i, j, STATE_M, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                result = global_visited_outside[(STATE_M, i, j)]
                global_visited_outside[(STATE_M2, i, j)] = result
                return result

            else:
                return ("", "")

        elif state == STATE_MULTI:
            idx = STATE_MULTI * n2 + j * n + i
            manner = int(beta_manners[idx])

            if manner == MANNER_MULTI_eq_MULTI_plus_U:
                ext = int(beta_splits[idx])
                j_next = j + ext
                right = '.' * ext

                if (STATE_MULTI, i, j_next) not in global_visited_outside:
                    global_visited_outside[(STATE_MULTI, i, j_next)] = self._get_parentheses_outside_real_backtrace(
                        i, j_next, STATE_MULTI, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                outsider = global_visited_outside[(STATE_MULTI, i, j_next)]
                result = (outsider[0], right + outsider[1])
                global_visited_outside[(STATE_MULTI, i, j)] = result
                return result

            elif manner == MANNER_P_eq_MULTI:
                if (STATE_P, i, j) not in global_visited_outside:
                    global_visited_outside[(STATE_P, i, j)] = self._get_parentheses_outside_real_backtrace(
                        i, j, STATE_P, n, n2,
                        global_visited_outside, global_visited_inside, window_visited)

                outsider = global_visited_outside[(STATE_P, i, j)]
                result = (outsider[0] + '(', ')' + outsider[1])
                global_visited_outside[(STATE_MULTI, i, j)] = result
                self._window_fill(window_visited, i, j, n, window_size)
                return result

            else:
                return ("", "")

        return ("", "")

    def _get_c_outside(self, j, n, n2, global_visited_outside, global_visited_inside):
        """
        Trace C[0..j]'s outside contribution.
        Following C code: get_parentheses_outside_real_backtrace for MANNER_C_eq_C_plus_U
        and MANNER_C_eq_C_plus_P cases.

        Returns (left_string, right_string) for the structure outside C[0..j].
        """
        beta_c_manners = self._beta_c_manners
        beta_c_scores = self._beta_c_scores
        window_size = self._window_size

        # Check cache
        if (-1, 0, j) in global_visited_outside:
            return global_visited_outside[(-1, 0, j)]

        # Base case: at the end
        if j >= n - 1:
            return ("", "")

        manner = int(beta_c_manners[j])

        if manner == MANNER_C_eq_C_plus_U:
            # C[0..j] <- C[0..j+1] by adding unpaired position j+1
            if j + 1 < n:
                if (-1, 0, j + 1) not in global_visited_outside:
                    if j + 1 < n - 1:
                        global_visited_outside[(-1, 0, j + 1)] = self._get_c_outside(
                            j + 1, n, n2, global_visited_outside, global_visited_inside)
                    else:
                        global_visited_outside[(-1, 0, j + 1)] = ("", "")
                outsider = global_visited_outside[(-1, 0, j + 1)]
                result = (outsider[0], '.' + outsider[1])
            else:
                result = ("", "")
            global_visited_outside[(-1, 0, j)] = result
            return result

        elif manner == MANNER_C_eq_C_plus_P:
            # C = c + P, tracing from small c, we need C[0..jj] outside and P inside
            # Following C code: lines 711-738
            # kk = j (small c's end), ii = j+1 (P's start), jj = split (P's end)
            beta_c_splits = self._beta_c_splits
            kk = j
            ii = j + 1
            jj = int(beta_c_splits[j])  # P's end position

            if jj > ii and jj < n:
                # Get P[ii, jj] inside
                if (STATE_P, ii, jj) not in global_visited_inside:
                    global_visited_inside[(STATE_P, ii, jj)] = self._get_parentheses_inside_real_backtrace(
                        ii, jj, STATE_P, n, n2, global_visited_inside, set())
                inside_P = global_visited_inside[(STATE_P, ii, jj)]

                # Get C[0..jj] outside
                if (-1, 0, jj) not in global_visited_outside:
                    if jj < n - 1:
                        global_visited_outside[(-1, 0, jj)] = self._get_c_outside(
                            jj, n, n2, global_visited_outside, global_visited_inside)
                    else:
                        global_visited_outside[(-1, 0, jj)] = ("", "")

                outsider = global_visited_outside[(-1, 0, jj)]
                result = (outsider[0], inside_P + outsider[1])
            else:
                # Fallback: leave right part as dots (matching C behavior)
                result = ("", "")

            global_visited_outside[(-1, 0, j)] = result
            return result

        else:
            # Default: leave right part as dots (matching C behavior)
            result = ("", "")
            global_visited_outside[(-1, 0, j)] = result
            return result

    def _build_external_right(self, start, end, n, n2, global_visited_inside):
        """Build structure for external loop from start to end."""
        if start > end:
            return ""

        c_manners = self._c_manners
        c_splits = self._c_splits

        result = ['.'] * (end - start + 1)

        j = end
        while j >= start:
            manner = int(c_manners[j]) if j < n else MANNER_NONE

            if manner == MANNER_C_eq_C_plus_U:
                j -= 1
            elif manner == MANNER_C_eq_C_plus_P:
                k = int(c_splits[j])
                if k >= start - 1:
                    # Trace P[k+1, j]
                    if (STATE_P, k + 1, j) not in global_visited_inside:
                        global_visited_inside[(STATE_P, k + 1, j)] = self._get_parentheses_inside_real_backtrace(
                            k + 1, j, STATE_P, n, n2, global_visited_inside, set())
                    p_str = global_visited_inside[(STATE_P, k + 1, j)]

                    # Copy P structure into result
                    p_start = k + 1 - start
                    for ci, ch in enumerate(p_str):
                        if p_start + ci >= 0 and p_start + ci < len(result):
                            result[p_start + ci] = ch

                    j = k
                else:
                    break
            else:
                j -= 1

        return ''.join(result)


def main():
    import argparse
    parser_args = argparse.ArgumentParser(description='LinearFold Hyper - Ultra-optimized RNA Folding (No Bonus)')
    parser_args.add_argument('--beamsize', '-b', type=int, default=100)
    parser_args.add_argument('--verbose', '-v', action='store_true')
    parser_args.add_argument('--V', action='store_true', help='Use Vienna energy model')
    parser_args.add_argument('--zuker', action='store_true', help='Output Zuker suboptimal structures')
    parser_args.add_argument('--delta', type=float, default=5.0, help='Energy delta for suboptimal structures')

    args = parser_args.parse_args()

    parser = BeamCKYParserHyper(beam_size=args.beamsize, is_verbose=args.verbose, lv=args.V)

    for line in sys.stdin:
        seq = line.strip()
        if not seq or seq.startswith(';') or seq.startswith('>'):
            continue
        seq = seq.upper().replace('T', 'U')
        structure, score, num_states, elapsed = parser.parse(seq)
        print(seq)
        print(f"{structure} ({score:.2f})")

        if args.zuker:
            print("Zuker suboptimal structures...")
            subopt_structures = parser.parse_subopt(seq, energy_delta=args.delta)
            for struct, energy in subopt_structures:
                print(f"{struct} ({energy:.2f})")


if __name__ == '__main__':
    main()
