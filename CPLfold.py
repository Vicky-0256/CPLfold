#!/usr/bin/env python3
"""
CPLfold - COMRADES-guided Pseudoknot LinearFold

A two-phase algorithm for RNA secondary structure prediction with pseudoknots,
integrating experimental COMRADES/PARIS data as folding constraints.

Algorithm:
1. Phase 1: Generate suboptimal structures using LinearFold with bonus matrix
2. Phase 2: For each Phase 1 structure, mask paired positions and fold again
3. Merge Phase 1 and Phase 2 structures to form pseudoknots
4. Calculate energy with the HotKnots-derived Python DP/CC/RE evaluator and rank by effective energy

Key Parameters:
- alpha: Scaling factor for COMRADES/PARIS bonus matrix (0.0-1.0)
- beta: Pseudoknot preference factor for ranking (0.0-1.0)

Dependencies:
- LinearFold: https://github.com/LinearFold/LinearFold
  Huang et al. (2019) Bioinformatics 35(14):i295-i304
- HotKnots 2.0: source code and parameters for the Python energy-calculation port;
  its structure-search algorithm and native executable are not runtime dependencies
  https://www.cs.ubc.ca/labs/algorithms/Software/HotKnots/
  Ren et al. (2005) RNA 11(10):1494-1504

Author: Ke Wang
"""

import sys
import os
import argparse
from typing import List, Dict, Tuple, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CPLfold parser
from CPLfold_parser import BeamCKYParserHyper

from Utils.hotknots_energy import HotKnotsEnergy


def structure_to_constraint(structure: str) -> str:
    """
    Convert a structure to constraint string for Phase 2 folding.

    In the constraint string:
    - Paired positions ('(' or ')') become '.' (forced unpaired - don't re-pair)
    - Unpaired positions ('.') become '?' (free to form new pairs)

    This allows Phase 2 to find new base pairs in unpaired regions that
    may form pseudoknots with Phase 1 pairs.

    Example:
        Input:  ((((((((......)))))))).....................(((((....)))))...
        Output: ........??????........?????????????????????.....[????]....???

    Args:
        structure: Phase 1 structure in dot-bracket notation

    Returns:
        Constraint string for Phase 2 folding
    """
    constraint = []
    for c in structure:
        if c == '(' or c == ')':
            constraint.append('.')  # Force unpaired - don't re-pair these positions
        else:
            constraint.append('?')  # Free to form new pairs
    return ''.join(constraint)


def get_pairs_from_structure(structure: str) -> List[Tuple[int, int]]:
    """
    Extract base pairs from a structure string.

    Args:
        structure: Dot-bracket structure (supports (), [], {})

    Returns:
        List of (i, j) pairs where i < j
    """
    pairs = []
    stacks = {'(': [], '[': [], '{': []}
    closing = {')': '(', ']': '[', '}': '{'}

    for i, c in enumerate(structure):
        if c in stacks:
            stacks[c].append(i)
        elif c in closing:
            opener = closing[c]
            if stacks[opener]:
                j = stacks[opener].pop()
                pairs.append((j, i))

    return pairs


def has_new_pairs(structure: str) -> bool:
    """
    Check if a structure has any base pairs.

    Args:
        structure: Dot-bracket structure

    Returns:
        True if structure contains at least one pair
    """
    return '(' in structure or '[' in structure or '{' in structure


def pairs_conflict(pairs1: List[Tuple[int, int]], pairs2: List[Tuple[int, int]]) -> bool:
    """
    Check if two sets of pairs conflict (same position paired differently).

    Args:
        pairs1: First set of pairs
        pairs2: Second set of pairs

    Returns:
        True if there's a conflict
    """
    paired1 = {}
    for i, j in pairs1:
        paired1[i] = j
        paired1[j] = i

    for i, j in pairs2:
        if i in paired1 and paired1[i] != j:
            return True
        if j in paired1 and paired1[j] != i:
            return True

    return False


def is_pseudoknot(pairs1: List[Tuple[int, int]], pairs2: List[Tuple[int, int]]) -> bool:
    """
    Check if two sets of pairs form a pseudoknot.
    A pseudoknot occurs when pairs cross: i < k < j < l where (i,j) and (k,l) are pairs.

    Args:
        pairs1: First set of pairs (from Phase 1)
        pairs2: Second set of pairs (from Phase 2)

    Returns:
        True if pairs form a pseudoknot
    """
    for i1, j1 in pairs1:
        for i2, j2 in pairs2:
            # Check for crossing: i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1
            if i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1:
                return True
    return False


def merge_structures(struct1: str, struct2: str) -> Optional[str]:
    """
    Merge Phase 1 and Phase 2 structures into a pseudoknotted structure.

    Phase 1 structure uses () notation.
    Phase 2 structure's pairs are converted to [] notation.

    Args:
        struct1: Phase 1 structure (uses '(' and ')')
        struct2: Phase 2 structure (uses '(' and ')')

    Returns:
        Merged structure with pseudoknot notation, or None if invalid
    """
    if len(struct1) != len(struct2):
        return None

    n = len(struct1)

    # Get pairs from both structures
    pairs1 = get_pairs_from_structure(struct1)
    pairs2 = get_pairs_from_structure(struct2)

    # Check for conflicts
    if pairs_conflict(pairs1, pairs2):
        return None

    # Build merged structure
    result = list(struct1)  # Start with Phase 1 structure

    # Add Phase 2 pairs using [] notation
    for i, j in pairs2:
        if result[i] == '.' and result[j] == '.':
            result[i] = '['
            result[j] = ']'

    return ''.join(result)


def phase1_fold(seq: str, parser: BeamCKYParserHyper,
                energy_delta: float = 5.0,
                max_structures: int = 10) -> List[Tuple[str, float]]:
    """
    Phase 1: Generate suboptimal structures using LinearFold.

    Args:
        seq: RNA sequence
        parser: LinearFold parser instance
        energy_delta: Energy range for suboptimal structures
        max_structures: Maximum number of structures to return

    Returns:
        List of (structure, score) tuples
    """
    results = parser.parse_subopt(seq, energy_delta=energy_delta, max_structures=max_structures)
    return results[:max_structures]


def phase2_fold(seq: str, parser: BeamCKYParserHyper,
                constraint: str) -> Tuple[Optional[str], float]:
    """
    Phase 2: Fold with constraints to find complementary structure.

    Args:
        seq: RNA sequence
        parser: LinearFold parser instance (must have use_constraints=True)
        constraint: Constraint string from structure_to_constraint()

    Returns:
        Tuple of (structure, score) or (None, 0) if failed
    """
    result = parser.parse(seq, cons=constraint)
    if result and result[0]:
        return result[0], result[1]
    return None, 0.0


def compute_energy_hotknots(seq: str, structure: str, hk: HotKnotsEnergy,
                             model: str = "DP09") -> float:
    """
    Compute structure energy using the Python port of HotKnots' energy evaluator.

    The implementation is derived from HotKnots 2.0 energy-calculation code and
    parameters, but it does not invoke the HotKnots package or executable.

    Args:
        seq: RNA sequence
        structure: Dot-bracket structure (can include [] for pseudoknots)
        hk: HotKnotsEnergy instance
        model: Energy model (DP03, DP09, CC06, CC09, RE)

    Returns:
        Energy in kcal/mol
    """
    result = hk.compute_energy(seq, structure, model=model)
    return float(result['energy'])


def output_results(all_structures: List[Dict], seq: str,
                   output_file: Optional[str], verbose: bool = True) -> None:
    """
    Output results to stdout and optionally to file.

    Args:
        all_structures: List of structure dictionaries
        seq: Original RNA sequence
        output_file: Optional file path for output
        verbose: Whether to print to stdout
    """
    lines = []

    lines.append(f"Sequence: {seq}")
    lines.append(f"Length: {len(seq)} nt")
    lines.append("=" * 60)
    lines.append("")

    # Group by type
    phase1_structs = [s for s in all_structures if s['type'] == 'phase1']
    pk_structs = [s for s in all_structures if s['type'] == 'pseudoknot']

    lines.append(f"Phase 1 Structures: {len(phase1_structs)}")
    lines.append(f"Pseudoknot Structures: {len(pk_structs)}")
    lines.append("")

    # Output Phase 1 structures
    if phase1_structs:
        lines.append("--- Phase 1 Structures ---")
        for i, s in enumerate(phase1_structs, 1):
            energy_str = f"{s['energy']:.2f}" if s['energy'] is not None else "N/A"
            lines.append(f"P1-{i}: {s['structure']}")
            lines.append(f"       Energy: {energy_str} kcal/mol")
        lines.append("")

    # Output pseudoknot structures
    if pk_structs:
        lines.append("--- Pseudoknot Structures ---")
        for i, s in enumerate(pk_structs, 1):
            energy_str = f"{s['energy']:.2f}" if s['energy'] is not None else "N/A"
            lines.append(f"PK-{i}: {s['structure']}")
            lines.append(f"       Energy: {energy_str} kcal/mol")
            if s.get('phase1_struct'):
                lines.append(f"       Phase 1: {s['phase1_struct']}")
            if s.get('phase2_struct'):
                lines.append(f"       Phase 2: {s['phase2_struct']}")
        lines.append("")

    # Best structures summary
    lines.append("=" * 60)
    lines.append("FINAL RESULTS (sorted by effective energy)")
    lines.append("=" * 60)

    if all_structures:
        # Best overall structure
        best = all_structures[0]
        energy_str = f"{best['energy']:.2f}" if best['energy'] is not None else "N/A"
        lines.append("")
        lines.append("Best Overall Structure:")
        lines.append(f"  {best['structure']}")
        lines.append(f"  Energy: {energy_str} kcal/mol")
        # Show effective energy if different from actual energy
        if best.get('effective_energy') is not None and best['effective_energy'] != best['energy']:
            eff_str = f"{best['effective_energy']:.2f}"
            lines.append(f"  Effective Energy: {eff_str} kcal/mol")
        lines.append(f"  Type: {best['type']}")

        # Best pseudoknot structure (if different from best overall)
        if pk_structs:
            # Sort pseudoknots by effective energy
            pk_sorted = sorted(pk_structs, key=lambda x: x['effective_energy'] if x.get('effective_energy') is not None else float('inf'))
            best_pk = pk_sorted[0]
            pk_energy_str = f"{best_pk['energy']:.2f}" if best_pk['energy'] is not None else "N/A"

            lines.append("")
            lines.append("Best Pseudoknot Structure:")
            lines.append(f"  {best_pk['structure']}")
            lines.append(f"  Energy: {pk_energy_str} kcal/mol")
            # Show effective energy if different
            if best_pk.get('effective_energy') is not None and best_pk['effective_energy'] != best_pk['energy']:
                pk_eff_str = f"{best_pk['effective_energy']:.2f}"
                lines.append(f"  Effective Energy: {pk_eff_str} kcal/mol")

            # Compare with best overall
            if best['type'] == 'pseudoknot':
                lines.append("  ** This IS the best overall structure **")
            else:
                if best['energy'] is not None and best_pk['energy'] is not None:
                    diff = best_pk['energy'] - best['energy']
                    lines.append(f"  Energy difference from best: +{diff:.2f} kcal/mol")
                    if diff > 0:
                        lines.append("  (Pseudoknot is less stable than non-pseudoknot structure)")
                    else:
                        lines.append("  (Pseudoknot is more stable)")
    else:
        lines.append("No structures found.")

    output_text = '\n'.join(lines)

    # Print to stdout
    if verbose:
        print(output_text)

    # Write to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
            f.write('\n')
        if verbose:
            print(f"\nResults saved to: {output_file}")


def two_phase_pseudoknot_fold(seq: str,
                               beam_size: int = 100,
                               energy_delta: float = 5.0,
                               max_phase1: int = 10,
                               max_phase2: int = 5,
                               energy_model: str = "DP09",
                               output_file: Optional[str] = None,
                               verbose: bool = True,
                               lv: bool = True,
                               bonus_matrix = None,
                               alpha: float = 0.0,
                               beta: float = 0.0) -> List[Dict]:
    """
    Main function: Two-phase pseudoknot prediction algorithm.

    Args:
        seq: RNA sequence
        beam_size: LinearFold beam size
        energy_delta: Energy range for suboptimal structures
        max_phase1: Maximum Phase 1 structures to process
        max_phase2: Maximum Phase 2 structures per Phase 1
        energy_model: Pseudoknot energy model (DP03, DP09, CC06, CC09, RE)
        output_file: Optional output file path
        verbose: Whether to print progress
        lv: Whether to use Vienna mode (True) or CONTRAfold mode (False)
        bonus_matrix: Optional PARIS/COMRADES bonus matrix (numpy array)
        alpha: Scaling factor for bonus matrix (0.0-1.0)
        beta: Pseudoknot energy bonus factor (0.0-1.0). For pseudoknot structures,
              the sorting energy = pseudoknot_energy + beta * phase1_energy.
              Since RNA energies are negative, this makes pseudoknots more
              favorable in ranking, improving recall of pseudoknot structures.

    Returns:
        List of structure dictionaries sorted by effective energy
    """
    # Preprocess sequence
    seq = seq.upper().replace('T', 'U')

    if verbose:
        print(f"Running Two-Phase Pseudoknot Algorithm...")
        print(f"Sequence length: {len(seq)}")
        print(f"Beam size: {beam_size}")
        print(f"Energy delta: {energy_delta}")
        print(f"Energy model: {energy_model}")
        print(f"LinearFold mode: {'Vienna' if lv else 'CONTRAfold'}")
        if bonus_matrix is not None:
            print(f"Using bonus matrix with alpha: {alpha}")
        if beta != 0.0:
            print(f"Pseudoknot energy bonus (beta): {beta}")
        print("")

    # Initialize parser with constraints enabled
    parser = BeamCKYParserHyper(
        beam_size=beam_size,
        lv=lv,  # Vienna mode (True) or CONTRAfold mode (False)
        use_constraints=True,
        is_verbose=False
    )
    
    # Set bonus matrix if provided
    if bonus_matrix is not None:
        parser.set_alpha(alpha)
        parser.set_bonus_matrix(bonus_matrix, len(seq))

    # Load CPLfold's packaged DP/CC/RE parameters once.
    hk = HotKnotsEnergy()

    all_structures = []
    seen_structures = set()  # Avoid duplicates

    # Phase 1: Get suboptimal structures
    if verbose:
        print("Phase 1: Generating suboptimal structures...")

    phase1_results = phase1_fold(seq, parser, energy_delta, max_phase1)

    if verbose:
        print(f"  Found {len(phase1_results)} Phase 1 structures")

    for idx, (struct1, score1) in enumerate(phase1_results):
        if verbose:
            print(f"\nProcessing Phase 1 structure {idx + 1}/{len(phase1_results)}")
            print(f"  Structure: {struct1}")

        # Skip if already seen
        if struct1 in seen_structures:
            continue
        seen_structures.add(struct1)

        # Calculate Phase 1 structure energy
        energy1 = compute_energy_hotknots(seq, struct1, hk, energy_model)

        all_structures.append({
            'structure': struct1,
            'energy': energy1,
            'effective_energy': energy1,  # For phase1, effective = actual energy
            'type': 'phase1',
            'phase1_struct': struct1,
            'phase1_energy': energy1,
            'phase2_struct': None,
            'score': score1
        })

        if verbose:
            print(f"  Energy: {energy1:.2f} kcal/mol")

        # Phase 2: Generate constraint and fold
        constraint = structure_to_constraint(struct1)

        if verbose:
            print(f"  Phase 2: Folding with constraints...")
            print(f"  Constraint: {constraint}")

        struct2, score2 = phase2_fold(seq, parser, constraint)

        if struct2 and has_new_pairs(struct2):
            if verbose:
                print(f"  Phase 2 structure: {struct2}")

            # Get pairs for pseudoknot check
            pairs1 = get_pairs_from_structure(struct1)
            pairs2 = get_pairs_from_structure(struct2)

            if verbose:
                print(f"  Phase 1 pairs: {len(pairs1)}, Phase 2 pairs: {len(pairs2)}")

            # Check if it forms a pseudoknot
            if is_pseudoknot(pairs1, pairs2):
                # Merge structures
                merged = merge_structures(struct1, struct2)

                if verbose:
                    print(f"  Merged structure: {merged}")

                if (
                    merged
                    and merged not in seen_structures
                ):
                    # Calculate merged structure energy
                    energy_merged = compute_energy_hotknots(seq, merged, hk, energy_model)
                    seen_structures.add(merged)

                    # Calculate effective energy for sorting
                    # effective_energy = pseudoknot_energy + beta * phase1_energy
                    # Since energies are negative, this makes pseudoknots more favorable
                    effective_energy = energy_merged + beta * energy1

                    all_structures.append({
                        'structure': merged,
                        'energy': energy_merged,
                        'effective_energy': effective_energy,
                        'type': 'pseudoknot',
                        'phase1_struct': struct1,
                        'phase1_energy': energy1,
                        'phase2_struct': struct2,
                        'score': score1 + score2
                    })

                    if verbose:
                        energy_str = f"{energy_merged:.2f}" if energy_merged is not None else "N/A"
                        print(f"  ** Pseudoknot found! **")
                        print(f"  Pseudoknot energy: {energy_str} kcal/mol")
                        if beta != 0.0 and effective_energy is not None:
                            print(f"  Effective energy (with beta={beta}): {effective_energy:.2f} kcal/mol")
                elif merged in seen_structures:
                    if verbose:
                        print(f"  (Duplicate structure, skipped)")
            else:
                if verbose:
                    print(f"  No pseudoknot formed (pairs don't cross)")
        else:
            if verbose:
                if struct2:
                    print(f"  Phase 2 structure: {struct2}")
                    print(f"  No new pairs found in Phase 2 (all positions unpaired)")
                else:
                    print(f"  Phase 2 folding returned no structure")

    # Sort by effective energy (None values go to end)
    # For pseudoknots with beta > 0, effective_energy = energy + beta * phase1_energy
    # This makes pseudoknots more favorable since energies are negative
    all_structures.sort(key=lambda x: x['effective_energy'] if x['effective_energy'] is not None else float('inf'))

    # Output results
    if verbose:
        print("\n")
    output_results(all_structures, seq, output_file, verbose)

    return all_structures


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Two-Phase LinearFold Pseudoknot Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CPLfold.py -s GGCGCGGCACCGUCCGCGGAACAAACGG
  python CPLfold.py -s GGCGCGGCACCGUCCGCGGAACAAACGG -o results.txt
  python CPLfold.py -s GGCGCGGCACCGUCCGCGGAACAAACGG -b 200 -d 10.0

Energy Models:
  DP03 - Dirks & Pierce 2003
  DP09 - Dirks & Pierce 2009 (recommended)
  CC06 - Cao & Chen 2006
  CC09 - Cao & Chen 2009
  RE   - Rivas & Eddy

Energy Implementation:
  Python port/refactoring based on HotKnots 2.0 energy-calculation code
  and parameter files; no HotKnots executable is invoked at runtime.

Beta Parameter:
  The --beta parameter controls pseudoknot preference in ranking.
  For pseudoknot structures:
    effective_energy = pseudoknot_energy + beta * phase1_energy
  Since RNA energies are negative, higher beta makes pseudoknots
  appear more favorable, improving recall of pseudoknot structures.

  Example: beta=0.5, phase1=-30, pseudoknot=-25
    effective = -25 + 0.5*(-30) = -40 kcal/mol
        """
    )

    parser.add_argument('-s', '--sequence', required=True,
                        help='RNA sequence (ACGU or ACGT)')
    parser.add_argument('-b', '--beam', type=int, default=100,
                        help='Beam size for LinearFold (default: 100)')
    parser.add_argument('-d', '--delta', type=float, default=5.0,
                        help='Energy delta for suboptimal structures (default: 5.0)')
    parser.add_argument('-n1', '--max-phase1', type=int, default=10,
                        help='Maximum Phase 1 structures (default: 10)')
    parser.add_argument('-n2', '--max-phase2', type=int, default=5,
                        help='Maximum Phase 2 structures per Phase 1 (default: 5)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('-m', '--model', type=str, default='DP09',
                        choices=['DP03', 'DP09', 'CC06', 'CC09', 'RE'],
                        help='Pseudoknot energy model (default: DP09)')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='Pseudoknot energy bonus factor (default: 0.0). '
                             'For pseudoknots, effective_energy = pk_energy + beta * phase1_energy. '
                             'Higher beta favors pseudoknots in ranking (improves recall).')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Run algorithm
    results = two_phase_pseudoknot_fold(
        seq=args.sequence,
        beam_size=args.beam,
        energy_delta=args.delta,
        max_phase1=args.max_phase1,
        max_phase2=args.max_phase2,
        energy_model=args.model,
        output_file=args.output,
        verbose=not args.quiet,
        beta=args.beta
    )

    # Return exit code based on whether pseudoknots were found
    pk_count = sum(1 for r in results if r['type'] == 'pseudoknot')
    if pk_count > 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
