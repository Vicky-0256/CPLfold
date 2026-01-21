# CPLfold - COMRADES-guided Pseudoknot LinearFold

Two-phase pseudoknot prediction algorithm using LinearFold with experimental COMRADES/PARIS data support.

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
├── Utils/                           # Energy parameters and utilities
│   ├── energy_parameter.py
│   ├── feature_weight.py
│   ├── intl11.py, intl21.py, intl22.py
│   └── ...
└── HotKnots_v2.0/                   # Pseudoknot energy calculation
    ├── hotknots.py
    ├── bin/
    └── ...
```

## Algorithm

### Two-Phase Approach

1. **Phase 1**: Generate suboptimal structures using LinearFold
2. **Phase 2**: For each Phase 1 structure, mask paired positions and fold again
3. **Merge**: Combine Phase 1 and Phase 2 pairs to form pseudoknots
4. **Score**: Calculate energy using HotKnots and rank structures

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

## Dependencies

### LinearFold
Linear-time RNA secondary structure prediction algorithm.
- Source: https://github.com/LinearFold/LinearFold
- Reference: Huang, L., Zhang, H., Deng, D., Zhao, K., Liu, K., Hendrix, D. A., & Mathews, D. H. (2019). LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search. Bioinformatics, 35(14), i295-i304.

### HotKnots
Pseudoknot energy calculation.
- Source: https://www.cs.ubc.ca/labs/algorithms/Software/HotKnots/
- Reference: Ren, J., Rastegari, B., Condon, A., & Hoos, H. H. (2005). HotKnots: Heuristic prediction of RNA secondary structures including pseudoknots. RNA, 11(10), 1494-1504.

## Energy Models (HotKnots)

Available energy models:
- `DP09` - Dirks & Pierce 2009 (recommended)
- `DP03` - Dirks & Pierce 2003
- `CC06` - Cao & Chen 2006
- `CC09` - Cao & Chen 2009
- `RE` - Rivas & Eddy

## Requirements

- Python 3.7+
- NumPy
- Numba
- SciPy
- pysam (for PARIS BAM file processing)
- ViennaRNA (optional, for comparison)

## Author

Ke Wang
