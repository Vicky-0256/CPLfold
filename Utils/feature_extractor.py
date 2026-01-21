"""
feature_extractor.py
Extract CONTRAfold-style feature counts from a sequence and dot-bracket structure.

This mirrors the decomposition used by Utils.utility scoring functions and
Utils.LinearFoldEval.eval (lv=False), but records feature counts instead of
energies. Counts align with arrays in Utils.feature_weight so updates can be
applied via structured perceptron or similar.
"""

from typing import Dict, Any, Tuple, List

import numpy as np

import Utils.shared as shared
from Utils import feature_weight as fw
from Utils.utility import (
    NOTON,
    EXPLICIT_MAX_LEN,
    SINGLE_MAX_LEN,
    HAIRPIN_MAX_LEN,
    INTERNAL_MAX_LEN,
    SYMMETRIC_MAX_LEN,
    ASYMMETRY_MAX_LEN,
)


def _mk_counts() -> Dict[str, Any]:
    """Initialize zeroed counts as numpy arrays/scalars for all used features."""
    counts: Dict[str, Any] = {}

    # Scalars
    counts["multi_base"] = 0.0
    counts["multi_unpaired"] = 0.0
    counts["multi_paired"] = 0.0
    counts["external_unpaired"] = 0.0
    counts["external_paired"] = 0.0

    # Arrays: allocate same length as weight arrays
    counts["base_pair"] = np.zeros(len(fw.base_pair), dtype=np.float64)
    counts["helix_stacking"] = np.zeros(len(fw.helix_stacking), dtype=np.float64) if hasattr(fw, "helix_stacking") else None
    counts["helix_closing"] = np.zeros(len(fw.helix_closing), dtype=np.float64)
    counts["terminal_mismatch"] = np.zeros(len(fw.terminal_mismatch), dtype=np.float64)
    counts["bulge_0x1_nucleotides"] = np.zeros(len(fw.bulge_0x1_nucleotides), dtype=np.float64)
    counts["internal_1x1_nucleotides"] = np.zeros(len(fw.internal_1x1_nucleotides), dtype=np.float64)
    counts["dangle_left"] = np.zeros(len(fw.dangle_left), dtype=np.float64)
    counts["dangle_right"] = np.zeros(len(fw.dangle_right), dtype=np.float64)
    counts["hairpin_length"] = np.zeros(len(fw.hairpin_length), dtype=np.float64)
    counts["internal_length"] = np.zeros(len(fw.internal_length), dtype=np.float64)
    counts["internal_explicit"] = np.zeros(len(fw.internal_explicit), dtype=np.float64)
    counts["internal_symmetric_length"] = np.zeros(len(fw.internal_symmetric_length), dtype=np.float64)
    counts["internal_asymmetry"] = np.zeros(len(fw.internal_asymmetry), dtype=np.float64)
    counts["bulge_length"] = np.zeros(len(fw.bulge_length), dtype=np.float64)

    return counts


def _add_junction_B_counts(nuci: int, nuci1: int, nucj_1: int, nucj: int, counts: Dict[str, Any]):
    """score_junction_B(i,j,...) = helix_closing + terminal_mismatch"""
    # helix_closing index
    idx_hc = nuci * NOTON + nucj
    counts["helix_closing"][idx_hc] += 1.0

    # terminal_mismatch index: nuci, nuci1, nucj_1, nucj (flattened as in utility)
    idx_tm = nuci * (NOTON * NOTON * NOTON) + nucj * (NOTON * NOTON) + nuci1 * NOTON + nucj_1
    counts["terminal_mismatch"][idx_tm] += 1.0


def _add_junction_A_counts(nuci: int, nuci1: int, nucj_1: int, nucj: int, i: int, j: int, seqlen: int, counts: Dict[str, Any]):
    """score_junction_A(i,j,...) = helix_closing + optional dangles."""
    # helix closing always
    idx_hc = nuci * NOTON + nucj
    counts["helix_closing"][idx_hc] += 1.0

    # dangle left
    if i < seqlen - 1 and nuci1 >= 0:
        idx_dl = nuci * (NOTON * NOTON) + nucj * NOTON + nuci1
        counts["dangle_left"][idx_dl] += 1.0
    # dangle right
    if j > 0 and nucj_1 >= 0:
        idx_dr = nuci * (NOTON * NOTON) + nucj * NOTON + nucj_1
        counts["dangle_right"][idx_dr] += 1.0


def _add_base_pair_count(nuci: int, nucj: int, counts: Dict[str, Any]):
    idx_bp = nucj * NOTON + nuci
    counts["base_pair"][idx_bp] += 1.0


def _add_helix_stacking_count(nuci: int, nuci1: int, nucj_1: int, nucj: int, counts: Dict[str, Any]):
    if counts.get("helix_stacking") is None:
        return
    idx = nuci * (NOTON * NOTON * NOTON) + nucj * (NOTON * NOTON) + nuci1 * NOTON + nucj_1
    counts["helix_stacking"][idx] += 1.0


def extract_counts(seq: str, ref: str, dangle_model: int = 2) -> Dict[str, Any]:
    """Extract feature counts for CONTRAfold-style model from a sequence and structure.

    Pseudoknots/restraint symbols are ignored (treated as unpaired).
    """
    n = len(seq)
    # Vienna encoding is not used; use CONTRAfold mapping
    nucs = [shared.get_acgu_num_c(ch) for ch in seq]

    counts = _mk_counts()

    # Pre-clean structure: map non () . to .
    clean_map = {'[': '.', ']': '.', '{': '.', '}': '.', '<': '.', '>': '.', ' ': '.'}
    ref_clean = ''.join(clean_map.get(c, c) for c in ref)

    # Track multi-loop unpaired per opening index (similar to eval)
    M1_energy_dummy = [0] * n  # placeholder to match flow, not used numerically
    multi_unpaired_counter = [0] * n

    stk: List[Tuple[int, int]] = []  # (index, page)
    inner_loop = (-1, -1)

    # Helper to get nuc with bounds
    def _get(idx: int) -> int:
        return nucs[idx] if 0 <= idx < n else -1

    # Pass 1: handle hairpin/single/multi and external paired
    for j, ch in enumerate(ref_clean):
        if ch == '.':
            if stk:
                multi_unpaired_counter[stk[-1][0]] += 1
        elif ch == '(':
            if stk:
                i0, page0 = stk[-1]
                stk[-1] = (i0, page0 + 1)
            stk.append((j, 0))
        elif ch == ')':
            assert stk
            i, page = stk.pop()

            nuci = _get(i)
            nucj = _get(j)
            nuci1 = _get(i + 1)
            nucj_1 = _get(j - 1)
            nuci_1 = _get(i - 1)
            nucj1 = _get(j + 1)

            if page == 0:
                # hairpin
                size = j - i - 1
                counts["hairpin_length"][min(size, HAIRPIN_MAX_LEN)] += 1.0
                _add_junction_B_counts(nuci, nuci1, nucj_1, nucj, counts)
            elif page == 1:
                # single loop (interior/bulge)
                p, q = inner_loop
                # base-pair for inner pair
                nucp_1 = _get(p - 1)
                nucp = _get(p)
                nucq = _get(q)
                nucq1 = _get(q + 1)

                # add base pair for inner closing pair (p,q)
                _add_base_pair_count(nucp, nucq, counts)

                # add junction_B for outer and inner pair
                _add_junction_B_counts(nuci, nuci1, nucj_1, nucj, counts)
                _add_junction_B_counts(nucp, nucp_1, nucq1, nucq, counts)

                l1 = p - i - 1
                l2 = j - q - 1
                if l1 == 0 and l2 > 0:
                    counts["bulge_length"][min(l2, SINGLE_MAX_LEN)] += 1.0
                    if l2 == 1 and nucq1 >= 0:
                        counts["bulge_0x1_nucleotides"][nucq1] += 1.0
                elif l2 == 0 and l1 > 0:
                    counts["bulge_length"][min(l1, SINGLE_MAX_LEN)] += 1.0
                    if l1 == 1 and nucp_1 >= 0:
                        counts["bulge_0x1_nucleotides"][nucp_1] += 1.0
                else:
                    # internal loop
                    counts["internal_length"][min(l1 + l2, INTERNAL_MAX_LEN)] += 1.0
                    if l1 <= EXPLICIT_MAX_LEN and l2 <= EXPLICIT_MAX_LEN:
                        l1_ = min(l1, EXPLICIT_MAX_LEN)
                        l2_ = min(l2, EXPLICIT_MAX_LEN)
                        idx = (l1_ * EXPLICIT_MAX_LEN + l2_) if l1_ <= l2_ else (l2_ * EXPLICIT_MAX_LEN + l1_)
                        counts["internal_explicit"][idx] += 1.0
                    if l1 == l2:
                        counts["internal_symmetric_length"][min(l1, SYMMETRIC_MAX_LEN)] += 1.0
                    else:
                        diff = abs(l1 - l2)
                        counts["internal_asymmetry"][min(diff, ASYMMETRY_MAX_LEN)] += 1.0
                    if l1 == 1 and l2 == 1 and nucp_1 >= 0 and nucq1 >= 0:
                        counts["internal_1x1_nucleotides"][nucp_1 * NOTON + nucq1] += 1.0

            else:
                # multi loop contribution: add closing pair multi score later; here nothing yet.
                pass

            # record for next inner loop
            inner_loop = (i, j)

            # M1 contributions and external paired
            if stk:
                k = j  # same as eval usage of j in score_M1(..., k=j)
                # score_M1 adds: junction_A(k,i, ...), multi_unpaired(k+1, j), base_pair(nuci, nuck), multi_paired
                # nuck is nucs[k], nuck1 is nucs[k+1]
                nuck = _get(k)
                nuck1 = _get(k + 1)
                # junction_A between (k,i)
                _add_junction_A_counts(nuck, nuck1, nuci_1, nuci, k, i, n, counts)
                # multi unpaired inside this M1 segment
                # In eval they added score_multi_unpaired(k+1, j) where j is current close index
                if k + 1 <= j:
                    counts["multi_unpaired"] += float(j - (k + 1) + 1)
                # base pair at (i,k)
                _add_base_pair_count(nuci, nuck, counts)
                # multi_paired scalar
                counts["multi_paired"] += 1.0
            else:
                # top-level pair: external paired contribution
                k = i - 1
                nuck = _get(k)
                nuck1 = _get(k + 1)
                _add_junction_A_counts(nuck, nuck1, _get(j - 1), nucj, k + 1, j, n, counts)
                counts["external_paired"] += 1.0

    # After processing all pairs: add multi closing pair contributions for each multi loop
    # We can approximate by scanning pairs again to find those with page>1 at close time.
    # Re-parse with a stack to detect multi loop closers and count unpaired at outer level.
    stk = []
    multi_unpaired_counter = [0] * n
    for j, ch in enumerate(ref_clean):
        if ch == '.':
            if stk:
                multi_unpaired_counter[stk[-1][0]] += 1
        elif ch == '(':
            if stk:
                i0, page0 = stk[-1]
                stk[-1] = (i0, page0 + 1)
            stk.append((j, 0))
        elif ch == ')':
            i, page = stk.pop()
            if page >= 2:
                # multi loop closed by (i,j)
                nuci = _get(i)
                nucj = _get(j)
                nuci1 = _get(i + 1)
                nucj_1 = _get(j - 1)
                _add_junction_A_counts(nuci, nuci1, nucj_1, nucj, i, j, n, counts)
                counts["multi_paired"] += 1.0  # multi closing includes one paired
                counts["multi_base"] += 1.0
                # unpaired counted earlier per M1; multi_unpaired for outer (i+1..i+num_unpaired)
                num_unpaired = multi_unpaired_counter[i]
                if num_unpaired > 0:
                    counts["multi_unpaired"] += float(num_unpaired)

    # External unpaired: count dots at depth 0
    depth = 0
    for ch in ref_clean:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '.' and depth == 0:
            counts["external_unpaired"] += 1.0

    # Helix stacking: count stacked adjacent pairs (i,j) and (i+1,j-1)
    # Build pair map
    stack = []
    pair = {}
    for idx, ch in enumerate(ref_clean):
        if ch == '(':
            stack.append(idx)
        elif ch == ')':
            if stack:
                i = stack.pop()
                pair[i] = idx
                pair[idx] = i
    for i, j in list(pair.items()):
        if i < j:
            i1 = i + 1
            j1 = j - 1
            if i1 in pair and pair[i1] == j1:
                # stacked
                nuci = _get(i)
                nucj = _get(j)
                nuci1 = _get(i1)
                nucj_1 = _get(j1)
                _add_helix_stacking_count(nuci, nuci1, nucj_1, nucj, counts)

    return counts
