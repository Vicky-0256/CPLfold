# HotKnots-derived CPLfold energy parameter subset

These are the eight data tables consumed by `Utils/hotknots_energy.py` for the
FM363, DP03/DP09, CC06/CC09, and RE energy models. Their numerical/table
contents come from `HotKnots_v2.0/bin/params` in the HotKnots 2.0 distribution;
some files have only line-ending or trailing-whitespace normalization. These
parameters were not independently fitted by CPLfold.

They are packaged here because `Utils/hotknots_energy.py` is a Python port and
refactoring of the energy-calculation portion of HotKnots. Keeping only the
required tables makes the evaluator runtime-independent: it does not mean the
energy implementation or parameters are unrelated to HotKnots, and it does not
require the full HotKnots source tree, wrapper, shared libraries, or executable.

The model provenance and parameter layout are documented in
[`docs/hotknots_energy_analysis.md`](../../docs/hotknots_energy_analysis.md).
The upstream package is available from the
[HotKnots project page](https://www.cs.ubc.ca/labs/algorithms/Software/HotKnots/).
