# CPLfold: Chimeric and Pseudoknot-capable Linear-time RNA Secondary Structure Prediction

Two-phase pseudoknot prediction algorithm using LinearFold with experimental COMRADES/PARIS data support.

Energy scoring is provided by a Python port/refactoring built from the HotKnots
2.0 energy-calculation source code and parameter files. CPLfold does not use the
HotKnots heuristic search and does not require HotKnots binaries or libraries at
runtime; this runtime independence does not change the code and data provenance.

## Directory Structure

```
CPLfold/
├── CPLfold.py                       # Main algorithm
├── CPLfold_parser.py                # LinearFold parser with bonus matrix support
├── extract_paris_scores.py          # PARIS BAM file processing (IRIS method)
├── example_with_paris.py            # Example using PARIS data
├── data/                            # Example data files
│   ├── bpRNA_RFAM_5220.dbn          # Example RNA sequence/structure
│   ├── bpRNA_RFAM_5220_paris_scores.txt  # PARIS support scores
│   └── bpRNA_RFAM_5220_paris_scores.npy  # PARIS matrix (numpy)
└── Utils/                           # Energy parameters and utilities
    ├── hotknots_energy.py           # Python port of HotKnots energy calculation
    ├── energy_params/               # Parameter tables from HotKnots 2.0
    ├── energy_parameter.py
    ├── feature_weight.py
    ├── intl11.py, intl21.py, intl22.py
    └── ...
```

## Algorithm

### Two-Phase Approach

1. **Phase 1**: Generate suboptimal structures using LinearFold
2. **Phase 2**: For each Phase 1 structure, mask paired positions and fold again
3. **Merge**: Combine Phase 1 and Phase 2 pairs to form pseudoknots
4. **Score**: Calculate DP/CC/RE energy with the HotKnots-derived Python evaluator and rank structures

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `beam_size` | LinearFold beam size | 100 |
| `energy_delta` | Energy range for suboptimal structures | 5.0 |
| `max_phase1` | Maximum Phase 1 structures | 10 |
| `alpha` | Bonus matrix scaling factor | 0.0 |
| `beta` | Pseudoknot ranking bonus | 0.0 |

### Beta Parameter

For pseudoknot structures:
```
effective_energy = pseudoknot_energy + beta × phase1_energy
```

Since RNA energies are negative, higher beta favors pseudoknots in ranking.

## Usage

### Command Line

```bash
# Basic usage
python CPLfold.py -s GGCGCGGCACCGUCCGCGGAACAAACGG

# With parameters
python CPLfold.py -s SEQUENCE -b 200 -d 10.0 --beta 0.3

# With output file
python CPLfold.py -s SEQUENCE -o results.txt
```

### Python API

```python
from CPLfold import two_phase_pseudoknot_fold

# Basic usage
results = two_phase_pseudoknot_fold(sequence, beam_size=100)

# With PARIS bonus matrix
import numpy as np
bonus_matrix = np.load("bonus.npy")
results = two_phase_pseudoknot_fold(
    sequence,
    bonus_matrix=bonus_matrix,
    alpha=0.5,
    beta=0.3
)
```

### Energy-only API

CPLfold's energy calculator is based on the **HotKnots 2.0 source code and
parameter distribution**. `Utils/hotknots_energy.py` ports and refactors the
energy-relevant `Stack`/`Loop`/`LoopList`/`Bands` structure decomposition,
SimFold secondary-structure terms, and DP/CC/RE pseudoknot scoring behavior into
Python. The eight files in `Utils/energy_params` come from the parameter tables
distributed with HotKnots 2.0; only formatting such as line endings or trailing
whitespace was normalized where needed, not the numerical values.

Here, **standalone** means runtime-independent, not independently invented or
clean-room implemented. The current scoring path does not import the HotKnots
wrapper, start `computeEnergy`, load a compiled HotKnots library, or require the
former `Utils/HotKnots_v2.0` source tree. It does, however, remain a Python port
derived from HotKnots' energy-calculation code and data.

| CPLfold component | Relationship to HotKnots |
|---|---|
| Candidate generation | CPLfold's two-phase LinearFold workflow; it does not use the HotKnots hotspot/heuristic search |
| Structure energy | Python port/refactoring based on HotKnots 2.0 energy-calculation code |
| Energy parameters | Required tables taken from the HotKnots 2.0 distribution |
| Runtime | No HotKnots package, source tree, shared library, or executable is required |

The Python API evaluates the complete closed-region/Loop/Bands tree directly:

```python
from Utils.hotknots_energy import HotKnotsEnergy

result = HotKnotsEnergy().compute_energy(
    "GGCGCGGCACCGUCCGCGGAACAAACGG",
    "..(((((..[[[[)))))......]]]]",
    model="CC09",
)
print(result["energy"], result["breakdown"])
```

The evaluator supports pseudoknot-free and H-type structures as well as nested
secondary structures inside pseudoloops, nested pseudoknots, multi-band chains,
kissing pseudoknots, and multiloops spanning a band. DP09, CC09, and RE follow
the original Loop/Bands scoring behavior; CC uses its defined DP fallback for
topologies outside its two-stem entropy table.

Numerical parity applies to valid canonical RNA structures, including a minimum
of three unpaired nucleotides in every hairpin. For a shorter, non-physical
hairpin the original C program exposes its `16000 kcal/mol` internal `INF`
sentinel; the Python API raises `ValueError` instead of treating that sentinel
as a meaningful energy.

See [the port, provenance, and formula analysis](docs/hotknots_energy_analysis.md)
for the source-to-Python relationship, parameter layout, DP09/CC09/RE equations,
supported topology, and reference validation results.

### PARIS Data Processing

Extract PARIS support matrix from BAM files (based on IRIS method):

```bash
python extract_paris_scores.py paris_reads.bam 207 output_scores.txt
```

The script:
1. Reads chimeric reads from BAM file (reads with exactly one gap)
2. Uses Normal distribution to spread support around interval centers
3. Creates outer product for pairwise support
4. Applies log transformation

Reference: IRIS (https://github.com/qczhang/IRIS)

### Example with PARIS Data

```bash
python example_with_paris.py
```

This runs CPLfold on bpRNA_RFAM_5220 (snoRNA, 207nt) with PARIS data from 9592 chimeric reads.

## Dependencies and provenance

### LinearFold

Linear-time RNA secondary structure prediction algorithm.

- Source: https://github.com/LinearFold/LinearFold
- Reference: Huang, L., Zhang, H., Deng, D., Zhao, K., Liu, K., Hendrix, D. A., & Mathews, D. H. (2019). LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search. Bioinformatics, 35(14), i295-i304.

### Relationship to HotKnots 2.0 and attribution

The energy calculator in this branch is derived from the energy-calculation
portion of HotKnots 2.0 and uses a packaged subset of its parameter files. It is
not a new independently fitted DP/CC/RE model. The former complete
`Utils/HotKnots_v2.0` tree was removed only because the HotKnots search code,
wrapper, native libraries, and executable are not needed at runtime; that
removal does not change the implementation or parameter provenance described
above.

The HotKnots 2.0 README credits the original implementation to Jihong Ren and
Baharak Rastegari, with modifications by Cristina Pop and Mirela Andronescu.
Relevant upstream source headers contain GNU GPL version 2-or-later notices;
redistribution of this derived implementation should preserve the upstream
attribution and comply with the applicable upstream license terms.

- Source: https://www.cs.ubc.ca/labs/algorithms/Software/HotKnots/
- Reference: Ren, J., Rastegari, B., Condon, A., & Hoos, H. H. (2005). HotKnots: Heuristic prediction of RNA secondary structures including pseudoknots. RNA, 11(10), 1494-1504.

## Pseudoknot Energy Models

Available energy models:

- `DP09` - Dirks & Pierce 2009 (recommended)
- `DP03` - Dirks & Pierce 2003
- `CC06` - Cao & Chen 2006
- `CC09` - Cao & Chen 2009
- `RE` - Rivas & Eddy

`DP09`, `CC09`, and `RE` are covered by regression vectors from the original
`computeEnergy`; `DP03` and `CC06` remain available for compatibility.

## Requirements

- Python 3.7+
- NumPy
- Numba
- SciPy
- pysam (for PARIS BAM file processing)
- ViennaRNA (optional, for comparison)

## Author

Ke Wang
