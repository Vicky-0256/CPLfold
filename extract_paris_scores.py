#!/usr/bin/env python3
"""
Extract PARIS support matrix from BAM files.

Based on the IRIS method (https://github.com/qczhang/IRIS):
1. Read PARIS blocks (pairs of intervals from chimeric reads)
2. Use Normal distribution to spread support around interval centers
3. Create outer product for pairwise support
4. Apply log transformation

Author: Ke Wang
"""

import pysam
import numpy as np
from scipy.stats import norm


def read_paris_blocks(bam_file):
    """
    Read PARIS blocks from BAM file.

    Each block consists of two intervals representing a chimeric PARIS read.
    Based on IRIS/Info_PARIS.py
    """
    blocks = []

    with pysam.AlignmentFile(bam_file, 'rb') as f:
        for read in f:
            block = read_to_block(read)
            if block is not None:
                blocks.append(block)

    return blocks


def read_to_block(read):
    """
    Convert a read to a block (two intervals).
    Based on IRIS/Info_PARIS.py
    """
    # Only one gap is allowed
    if read.cigarstring is None or read.cigarstring.count('N') != 1:
        return None

    # Refine read (remove indels and soft clips)
    block = read.get_blocks()
    if len(block) > 2:
        i, updated_block = 0, []
        j, cigar = 0, read.cigartuples
        while j < len(cigar):
            if cigar[j][0] == 0:
                updated_block.append(block[i])
                i += 1
            elif cigar[j][0] == 1 or cigar[j][0] == 2:
                updated_block[-1] = (updated_block[-1][0], block[i][1])
                i += 1
                j += 1
            j += 1
        block = updated_block

    if len(block) != 2:
        return None

    # Only retain reads with two intervals longer than 15nt
    (ll, lr), (rl, rr) = block
    if lr - ll <= 15 or rr - rl <= 15:
        return None

    return block


def compute_paris_support(blocks, seq_len):
    """
    Convert PARIS blocks to support matrix.
    Based on IRIS/IRIS_Core.py Scoring.compute_PARIS_support()
    """
    PARIS_support = np.zeros((seq_len, seq_len), dtype=np.float32)

    if len(blocks) == 0:
        return PARIS_support

    # Calculate standard deviation using the 3-sigma rule
    arm_len = []
    for (ll, lr), (rl, rr) in blocks:
        arm_len.append(lr - ll)
        arm_len.append(rr - rl)

    sd = np.mean(arm_len) / 6

    # Create position vectors
    l_vec = np.arange(seq_len)
    r_vec = np.arange(seq_len)

    for (ll, lr), (rl, rr) in blocks:
        # The center of intervals as the mean value
        l_mu = (ll + lr) / 2
        r_mu = (rl + rr) / 2

        # Transform via Normal distribution
        l_support = norm.pdf(l_vec, l_mu, sd)
        r_support = norm.pdf(r_vec, r_mu, sd)

        # Normalize by the peak of the distribution
        l_support /= np.max(l_support)
        r_support /= np.max(r_support)

        # Sum up to the matrix
        PARIS_support += np.outer(l_support, r_support)
        PARIS_support += np.outer(r_support, l_support)

    # Trim infinitesimal values as 0
    EPS = 1e-6
    PARIS_support[PARIS_support < EPS] = 0

    # The logarithm transformation
    PARIS_support = np.log(PARIS_support + 1)

    return PARIS_support


def save_paris_scores(scores, output_file, min_score=0.01):
    """
    Save PARIS support matrix to text file.
    Format: i j score (1-indexed)
    """
    with open(output_file, 'w') as f:
        n = scores.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if scores[i, j] >= min_score:
                    f.write(f"{i+1}\t{j+1}\t{scores[i, j]:.4f}\n")


def extract_paris_matrix(bam_file, seq_len):
    """
    Main function to extract PARIS support matrix from BAM file.

    Args:
        bam_file: Path to PARIS BAM file
        seq_len: Length of RNA sequence

    Returns:
        numpy array of shape (seq_len, seq_len) with PARIS support scores
    """
    print(f"Reading PARIS blocks from {bam_file}...")
    blocks = read_paris_blocks(bam_file)
    print(f"  Found {len(blocks)} valid PARIS blocks")

    print("Computing PARIS support matrix...")
    support = compute_paris_support(blocks, seq_len)
    print(f"  Non-zero entries: {np.count_nonzero(support)}")
    print(f"  Max score: {support.max():.4f}")

    return support


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python extract_paris_scores.py <bam_file> <seq_len> <output_file>")
        print("\nExample:")
        print("  python extract_paris_scores.py paris_reads.bam 207 paris_scores.txt")
        sys.exit(1)

    bam_file = sys.argv[1]
    seq_len = int(sys.argv[2])
    output_file = sys.argv[3]

    support = extract_paris_matrix(bam_file, seq_len)
    save_paris_scores(support, output_file)

    # Also save as numpy array
    npy_file = output_file.replace('.txt', '.npy')
    np.save(npy_file, support)

    print(f"\nSaved to:")
    print(f"  Text: {output_file}")
    print(f"  Numpy: {npy_file}")
