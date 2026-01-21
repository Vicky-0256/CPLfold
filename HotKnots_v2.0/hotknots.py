#!/usr/bin/env python3
"""
HotKnots v2.0 Python Wrapper
RNA secondary structure prediction with pseudoknots

Supported energy models:
  - DP: Dirks & Pierce (DP03, DP09)
  - CC: Cao & Chen (CC06, CC09)
  - RE: Rivas & Eddy

Usage:
    from hotknots import HotKnots

    hk = HotKnots()
    results = hk.predict("GGCGCGGCACCGUCCGCGGAACAAACGG", model="DP09")
    energy = hk.compute_energy("GGCGCGGCACCGUCCGCGGAACAAACGG", "..(((((..[[[[)))))......]]]]", model="DP09")
"""

import subprocess
import os
import re
from typing import List, Dict, Optional, Tuple

class HotKnots:
    """HotKnots v2.0 wrapper for pseudoknot RNA structure prediction"""

    # Available models and parameter files
    MODELS = {
        "DP03": ("DP", "parameters_DP03.txt", "Dirks & Pierce 2003"),
        "DP09": ("DP", "parameters_DP09.txt", "Dirks & Pierce 2009"),
        "CC06": ("CC", "parameters_CC06.txt", "Cao & Chen 2006"),
        "CC09": ("CC", "parameters_CC09.txt", "Cao & Chen 2009"),
        "RE": ("RE", None, "Rivas & Eddy"),
    }

    def __init__(self, hotknots_dir: str = None):
        """
        Initialize HotKnots wrapper

        Args:
            hotknots_dir: Path to HotKnots_v2.0 directory. If None, uses the directory containing this script.
        """
        if hotknots_dir is None:
            hotknots_dir = os.path.dirname(os.path.abspath(__file__))

        self.hotknots_dir = hotknots_dir
        self.bin_dir = os.path.join(hotknots_dir, "bin")
        self.hotknots_bin = os.path.join(self.bin_dir, "HotKnots")
        self.compute_energy_bin = os.path.join(self.bin_dir, "computeEnergy")
        self.params_dir = os.path.join(self.bin_dir, "params")

        # Verify installation
        if not os.path.exists(self.hotknots_bin):
            raise FileNotFoundError(f"HotKnots binary not found at {self.hotknots_bin}. Please compile first.")

    def predict(self, sequence: str, model: str = "DP09", top_n: int = 10) -> List[Dict]:
        """
        Predict RNA secondary structure with pseudoknots

        Args:
            sequence: RNA sequence (ACGU)
            model: Energy model (DP03, DP09, CC06, CC09, RE)
            top_n: Number of top structures to return

        Returns:
            List of dicts with 'structure', 'energy', 'has_pseudoknot'
        """
        sequence = sequence.upper().replace('T', 'U')

        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")

        model_type, param_file, _ = self.MODELS[model]

        cmd = [self.hotknots_bin, "-s", sequence, "-m", model_type, "-noPS"]
        if param_file:
            cmd.extend(["-p", os.path.join("params", param_file)])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.bin_dir)
        output = result.stdout + result.stderr

        return self._parse_prediction_output(output)[:top_n]

    def compute_energy(self, sequence: str, structure: str, model: str = "DP09") -> Dict:
        """
        Compute free energy of a given structure

        Args:
            sequence: RNA sequence
            structure: Dot-bracket structure (can include [ ] for pseudoknots)
            model: Energy model

        Returns:
            Dict with 'energy', 'energy_no_dangling'
        """
        sequence = sequence.upper().replace('T', 'U')

        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")

        model_type, param_file, _ = self.MODELS[model]

        cmd = [self.compute_energy_bin, "-m", model_type, "-s", sequence, structure]
        if param_file:
            cmd.extend(["-p", os.path.join("params", param_file)])

        result = subprocess.run(cmd, capture_output=True, cwd=self.bin_dir)
        output = result.stdout.decode('utf-8', errors='ignore') + result.stderr.decode('utf-8', errors='ignore')

        return self._parse_energy_output(output)

    def compare_models(self, sequence: str) -> Dict[str, List[Dict]]:
        """
        Compare predictions from all energy models

        Args:
            sequence: RNA sequence

        Returns:
            Dict mapping model names to prediction results
        """
        results = {}
        for model in self.MODELS:
            try:
                results[model] = self.predict(sequence, model=model, top_n=5)
            except Exception as e:
                results[model] = [{"error": str(e)}]
        return results

    def _parse_prediction_output(self, output: str) -> List[Dict]:
        """Parse HotKnots prediction output"""
        results = []
        for line in output.strip().split('\n'):
            match = re.match(r'S(\d+):\s+(\S+)\s+(-?\d+\.?\d*)', line)
            if match:
                structure = match.group(2)
                energy = float(match.group(3))
                has_pk = '[' in structure or '{' in structure
                results.append({
                    'rank': int(match.group(1)),
                    'structure': structure,
                    'energy': energy,
                    'has_pseudoknot': has_pk
                })
        return results

    def _parse_energy_output(self, output: str) -> Dict:
        """Parse computeEnergy output"""
        lines = output.strip().split('\n')
        for i, line in enumerate(lines):
            # 标准格式: Dirks&Pierce/Cao&Chen/Rivas&Eddy  能量1  能量2
            match = re.search(r'(Dirks|Cao|Rivas).*?\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*$', line)
            if match:
                return {
                    'energy': float(match.group(2)),
                    'energy_no_dangling': float(match.group(3))
                }
            
            # RE 模型可能有 WARNING，能量在下一行
            # 格式: Rivas&Eddy     WARNING: ...
            #       -416.97                  -412.57
            if 'Rivas' in line and 'WARNING' in line:
                # 找下一行的能量
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    energy_match = re.search(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', next_line)
                    if energy_match:
                        return {
                            'energy': float(energy_match.group(1)),
                            'energy_no_dangling': float(energy_match.group(2))
                        }
        return {'energy': None, 'energy_no_dangling': None}

    @staticmethod
    def has_pseudoknot(structure: str) -> bool:
        """Check if structure contains pseudoknot"""
        return '[' in structure or '{' in structure

    @staticmethod
    def available_models() -> List[str]:
        """Return list of available models"""
        return list(HotKnots.MODELS.keys())


def predict_structure(sequence: str, model: str = "DP09") -> Tuple[str, float]:
    """
    Convenience function to predict MFE structure

    Args:
        sequence: RNA sequence
        model: Energy model (default: DP09)

    Returns:
        Tuple of (structure, energy)
    """
    hk = HotKnots()
    results = hk.predict(sequence, model=model, top_n=1)
    if results:
        return results[0]['structure'], results[0]['energy']
    return None, None


def compute_energy(sequence: str, structure: str, model: str = "DP09") -> float:
    """
    Convenience function to compute structure energy

    Args:
        sequence: RNA sequence
        structure: Dot-bracket structure
        model: Energy model (default: DP09)

    Returns:
        Free energy in kcal/mol
    """
    hk = HotKnots()
    result = hk.compute_energy(sequence, structure, model=model)
    return result['energy']


# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HotKnots v2.0 - RNA pseudoknot prediction')
    parser.add_argument('-s', '--sequence', required=True, help='RNA sequence')
    parser.add_argument('-m', '--model', default='DP09',
                        choices=HotKnots.MODELS.keys(),
                        help='Energy model (default: DP09)')
    parser.add_argument('--structure', help='Structure for energy calculation')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('-n', '--top-n', type=int, default=5, help='Show top N structures')

    args = parser.parse_args()

    hk = HotKnots()
    seq = args.sequence.upper().replace('T', 'U')

    print(f"Sequence: {seq}")
    print(f"Length: {len(seq)} nt")
    print("=" * 60)

    if args.structure:
        # Compute energy mode
        result = hk.compute_energy(seq, args.structure, model=args.model)
        print(f"Model: {args.model} ({HotKnots.MODELS[args.model][2]})")
        print(f"Structure: {args.structure}")
        print(f"Energy: {result['energy']:.2f} kcal/mol")

    elif args.compare:
        # Compare all models
        results = hk.compare_models(seq)
        for model, preds in results.items():
            print(f"\n--- {model}: {HotKnots.MODELS[model][2]} ---")
            for p in preds[:3]:
                if 'error' in p:
                    print(f"  Error: {p['error']}")
                else:
                    pk = " [PK]" if p['has_pseudoknot'] else ""
                    print(f"  {p['structure']}  {p['energy']:>7.2f}{pk}")
    else:
        # Predict mode
        print(f"Model: {args.model} ({HotKnots.MODELS[args.model][2]})")
        print("-" * 60)
        results = hk.predict(seq, model=args.model, top_n=args.top_n)
        for r in results:
            pk = " [PK]" if r['has_pseudoknot'] else ""
            print(f"S{r['rank']}: {r['structure']}  {r['energy']:>7.2f} kcal/mol{pk}")
