#!/usr/bin/env python3
"""Python port of the energy-calculation portion of HotKnots 2.0.

The HotKnots executable contains both a structure search algorithm and an
energy evaluator.  CPLfold only needs the latter.  This module was built by
porting and refactoring the energy-relevant HotKnots ``Stack``/``Loop``/
``LoopList``/``Bands`` logic, its bundled SimFold behavior, and its DP, CC, and
RE scoring paths into Python.  The accompanying tables in ``energy_params``
come from the HotKnots 2.0 parameter distribution; this is not an independently
fitted or clean-room energy model.

The closed-region, loop, and band decomposition is implemented here as well,
so nested secondary structures, chained/multi-band pseudoknots, kissing
pseudoknots, and pseudoknots nested inside other pseudoknots do not require the
HotKnots source tree or executable at runtime.  "Standalone" therefore refers
only to runtime dependencies, not to the implementation's provenance.

The HotKnots 2.0 README credits Jihong Ren and Baharak Rastegari for the
original implementation and Cristina Pop and Mirela Andronescu for subsequent
modifications.  Relevant upstream source headers carry GNU GPL version 2 or
later notices; see the repository README for attribution and provenance.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


A, C, G, U = range(4)
NUC_TO_INT = {"A": A, "C": C, "G": G, "U": U}
CANONICAL_PAIRS = {(A, U), (U, A), (C, G), (G, C), (G, U), (U, G)}
WATSON_CRICK_PAIRS = {(A, U), (U, A), (C, G), (G, C)}
CG_PAIRS = {(C, G), (G, C)}
KB = 0.001987
TEMPERATURE_K = 310.15
INF = float("inf")

Pair = Tuple[int, int]


class UnsupportedTopologyError(ValueError):
    """Backward-compatible exception type from the former H-type-only API."""


@dataclass(frozen=True)
class ParsedPair:
    left: int
    right: int
    level: str

    @property
    def pair(self) -> Pair:
        return self.left, self.right


@dataclass
class _LoopNode:
    pair: Pair
    children: List["_LoopNode"]
    placeholder: bool = False


@dataclass(frozen=True)
class _HType:
    stem1: Tuple[Pair, ...]
    stem2: Tuple[Pair, ...]
    loop1: int
    loop2: int
    loop3: int
    unpaired: int
    begin: int
    end: int


@dataclass(frozen=True)
class _Band:
    """One HotKnots band, represented by its outer-to-inner base pairs."""

    pairs: Tuple[Pair, ...]

    @property
    def first_arm(self) -> Pair:
        return self.pairs[0][0], self.pairs[-1][0]

    @property
    def second_arm(self) -> Pair:
        return self.pairs[-1][1], self.pairs[0][1]

    @property
    def regions(self) -> Tuple[Pair, Pair]:
        return self.first_arm, self.second_arm


@dataclass(frozen=True)
class _SpanLoop:
    """A stack/interior/multiloop whose closing pairs belong to a band."""

    outer: Pair
    inner: Pair
    children: Tuple["_ClosedRegion", ...]
    unpaired: int

    @property
    def is_multi(self) -> bool:
        return bool(self.children)


@dataclass
class _ClosedRegion:
    """A node in the closed-region tree defined by HotKnots' Stack parser."""

    begin: int
    end: int
    children: List["_ClosedRegion"] = field(default_factory=list)
    parent: Optional["_ClosedRegion"] = None
    kind: str = "external"
    nested: str = "nothing"
    bands: Tuple[_Band, ...] = ()
    band_regions: Tuple[Pair, ...] = ()
    span_loops: Tuple[_SpanLoop, ...] = ()
    number_unpaired: int = 0
    number_unpaired_in_pseudo: int = 0

    @property
    def pair(self) -> Pair:
        return self.begin, self.end


class _LoopBandsTree:
    """Pure-Python port of HotKnots' ``Stack``/``Loop``/``Bands`` parser.

    The original implementation mutates a global linked list of paired
    positions while closed regions are discovered.  Once the closed-region
    tree is known, the same surface is obtained by excluding every direct
    child interval.  Building it this way is deterministic and avoids the
    pointer-order dependence of the C++ representation while preserving its
    loop and band definitions.
    """

    def __init__(self, pair_table: Sequence[int]):
        self.pair_table = tuple(pair_table)
        self.length = len(pair_table)
        intervals = self._closed_intervals()
        self.root = _ClosedRegion(-1, self.length, kind="external")
        self.nodes = [_ClosedRegion(begin, end) for begin, end in intervals]
        self.by_begin: Dict[int, _ClosedRegion] = {node.begin: node for node in self.nodes}
        self._attach_nodes()
        self._annotate_nodes()

    def _closed_intervals(self) -> Tuple[Pair, ...]:
        stack: List[List[int]] = []
        intervals: List[Pair] = []
        for position, partner in enumerate(self.pair_table):
            if partner < 0:
                continue
            if position < partner:
                stack.append([position, partner])
                continue

            if not stack:
                raise ValueError("Invalid pair table: closing endpoint without an open region")
            expanded_end = position
            while stack and stack[-1][0] > partner:
                expanded_end = max(expanded_end, stack.pop()[1])
            if not stack:
                raise ValueError("Invalid crossing pattern: no region can contain a closing endpoint")
            stack[-1][1] = max(expanded_end, stack[-1][1])
            if position == stack[-1][1]:
                begin, end = stack.pop()
                intervals.append((begin, end))

        if stack:
            raise ValueError("Invalid pair table: unclosed region")
        return tuple(intervals)

    def _attach_nodes(self) -> None:
        # Closed-region intervals are laminar even when their constituent base
        # pairs cross.  The smallest strict containing interval is the parent.
        for node in self.nodes:
            candidates = [
                other
                for other in self.nodes
                if other.begin < node.begin and node.end < other.end
            ]
            parent = min(candidates, key=lambda item: item.end - item.begin) if candidates else self.root
            node.parent = parent
            parent.children.append(node)
        for node in [self.root] + self.nodes:
            node.children.sort(key=lambda item: item.begin)

    def _kind(self, node: _ClosedRegion) -> str:
        if self.pair_table[node.begin] != node.end:
            return "pseudo"
        if not node.children:
            return "hairpin"
        if len(node.children) == 1 and node.children[0].kind != "pseudo":
            child = node.children[0]
            if child.begin == node.begin + 1 and child.end == node.end - 1:
                return "stack"
            return "interior"
        return "multi"

    @staticmethod
    def _outside_children(position: int, children: Sequence[_ClosedRegion]) -> bool:
        return not any(child.begin <= position <= child.end for child in children)

    def _surface_positions(self, node: _ClosedRegion) -> List[int]:
        return [
            position
            for position in range(node.begin, node.end + 1)
            if self.pair_table[position] >= 0 and self._outside_children(position, node.children)
        ]

    def _find_bands(self, node: _ClosedRegion) -> Tuple[_Band, ...]:
        surface = self._surface_positions(node)
        if not surface:
            raise ValueError(f"Pseudoknotted region [{node.begin}, {node.end}] has no surface pairs")

        before = {-1: -1}
        after = {self.length: self.length}
        last = -1
        for position in surface:
            after[last] = position
            before[position] = last
            last = position
        after[last] = self.length
        before[self.length] = last

        left_borders = set()
        bands: List[_Band] = []
        position = surface[0]
        safety = 0
        while position <= node.end:
            while position in left_borders:
                position = after[position]
            if position > node.end:
                break
            partner = self.pair_table[position]
            if partner < position or partner not in before:
                raise ValueError(
                    f"Could not form a HotKnots band at pair endpoint {position} in "
                    f"region [{node.begin}, {node.end}]"
                )

            first_end = position
            second_start = partner
            band_pairs: List[Pair] = [(position, partner)]
            while True:
                next_first = after[first_end]
                previous_second = before[second_start]
                if (
                    next_first <= node.end
                    and previous_second >= node.begin
                    and self.pair_table[next_first] == previous_second
                ):
                    first_end = next_first
                    second_start = previous_second
                    band_pairs.append((first_end, second_start))
                else:
                    break

            bands.append(_Band(tuple(band_pairs)))
            left_borders.update((position, second_start))
            if first_end != position:
                next_after_first = after[first_end]
                after[position] = next_after_first
                before[next_after_first] = position

                next_after_partner = after[partner]
                after[second_start] = next_after_partner
                before[next_after_partner] = second_start

            position = after[position]
            safety += 1
            if safety > len(surface):
                raise RuntimeError("Band discovery did not make progress")

        assigned = {pair for band in bands for pair in band.pairs}
        expected = {
            (position, partner)
            for position, partner in enumerate(self.pair_table)
            if node.begin <= position < partner <= node.end
            and self._outside_children(position, node.children)
            and self._outside_children(partner, node.children)
        }
        if assigned != expected:
            missing = sorted(expected - assigned)
            raise ValueError(
                f"Band decomposition for [{node.begin}, {node.end}] missed surface pairs {missing}"
            )
        return tuple(bands)

    @staticmethod
    def _top_level(candidates: Sequence[_ClosedRegion]) -> Tuple[_ClosedRegion, ...]:
        return tuple(
            node
            for node in sorted(candidates, key=lambda item: item.begin)
            if not any(
                other.begin < node.begin and node.end < other.end
                for other in candidates
            )
        )

    def _span_loop(self, node: _ClosedRegion, outer: Pair, inner: Pair) -> _SpanLoop:
        i, j = outer
        ip, jp = inner
        faces = ((i + 1, ip - 1), (jp + 1, j - 1))
        candidates = [
            child
            for child in node.children
            if any(start <= child.begin and child.end <= end for start, end in faces)
        ]
        children = self._top_level(candidates)
        face_size = sum(max(0, end - start + 1) for start, end in faces)
        unpaired = face_size - sum(child.end - child.begin + 1 for child in children)
        if unpaired < 0:
            raise RuntimeError("Negative unpaired count in a loop spanning a band")
        return _SpanLoop(outer, inner, children, unpaired)

    def _annotate_nodes(self) -> None:
        # Children occur earlier in the closure order; annotating by interval
        # width therefore also makes their types available to their parents.
        for node in sorted(self.nodes, key=lambda item: item.end - item.begin):
            node.kind = self._kind(node)
            node.number_unpaired = sum(
                1
                for position in range(node.begin, node.end)
                if self.pair_table[position] < 0
                and self._outside_children(position, node.children)
            )
            if node.kind != "pseudo":
                continue

            node.bands = self._find_bands(node)
            node.band_regions = tuple(
                sorted(
                    (region for band in node.bands for region in band.regions),
                    key=lambda region: region[0],
                )
            )
            node.span_loops = tuple(
                self._span_loop(node, outer, inner)
                for band in node.bands
                for outer, inner in zip(band.pairs, band.pairs[1:])
            )

        for node in self.nodes:
            if node.parent is None or node.parent.kind != "pseudo":
                continue
            parent = node.parent
            if any(start <= node.begin and node.end <= end for start, end in parent.band_regions):
                node.nested = "in_band"
            else:
                node.nested = "un_band"

        for node in self.nodes:
            if node.kind != "pseudo":
                continue
            gaps = sum(
                right[0] - left[1] - 1
                for left, right in zip(node.band_regions, node.band_regions[1:])
            )
            excluded = sum(
                child.end - child.begin + 1
                for child in node.children
                if child.nested == "un_band"
            )
            node.number_unpaired_in_pseudo = gaps - excluded
            if node.number_unpaired_in_pseudo < 0:
                raise RuntimeError(
                    f"Negative pseudoloop unpaired count in [{node.begin}, {node.end}]"
                )

    @property
    def pseudoknots(self) -> Tuple[_ClosedRegion, ...]:
        return tuple(node for node in self.nodes if node.kind == "pseudo")


def _can_pair(i: int, j: int) -> bool:
    return (i, j) in CANONICAL_PAIRS


def _watson_crick(i: int, j: int) -> bool:
    return (i, j) in WATSON_CRICK_PAIRS


def _has_au_penalty(i: int, j: int) -> bool:
    return (i, j) not in CG_PAIRS


def _crosses(first: Pair, second: Pair) -> bool:
    i, j = first
    k, l = second
    return i < k < j < l or k < i < l < j


def _read_numbers(path: Path) -> List[float]:
    values: List[float] = []
    for raw in path.read_text().splitlines():
        text = raw.strip()
        if text and not text.startswith("#"):
            values.append(float(text.split()[0]))
    return values


def _read_data_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    for raw in path.read_text().splitlines():
        text = raw.strip()
        if text and not text.startswith("#"):
            rows.append(text.split())
    return rows


def parse_dot_bracket(structure: str) -> Tuple[List[ParsedPair], List[int]]:
    """Parse the bracket alphabet accepted by HotKnots.

    Besides ``()``, ``[]``, ``{}``, and ``<>``, HotKnots uses uppercase and
    lowercase letters as additional opening/closing levels, and treats both
    ``.`` and ``_`` as unpaired positions.
    """

    close_to_open = {")": "(", "]": "[", "}": "{", ">": "<"}
    close_to_open.update({chr(ord("a") + i): chr(ord("A") + i) for i in range(26)})
    openers = set(close_to_open.values())
    stacks: Dict[str, List[int]] = {opener: [] for opener in openers}
    pairs: List[ParsedPair] = []
    pair_table = [-1] * len(structure)

    for position, symbol in enumerate(structure):
        if symbol in "._":
            continue
        if symbol in openers:
            stacks[symbol].append(position)
            continue
        opener = close_to_open.get(symbol)
        if opener is None:
            raise ValueError(f"Unsupported dot-bracket symbol {symbol!r} at position {position}")
        if not stacks[opener]:
            raise ValueError(f"Unmatched closing symbol {symbol!r} at position {position}")
        left = stacks[opener].pop()
        pairs.append(ParsedPair(left, position, opener))
        pair_table[left] = position
        pair_table[position] = left

    unmatched = [(symbol, positions) for symbol, positions in stacks.items() if positions]
    if unmatched:
        symbol, positions = min(unmatched, key=lambda item: item[1][0])
        raise ValueError(f"Unmatched opening symbol {symbol!r} at position {positions[-1]}")

    pairs.sort(key=lambda item: (item.left, -item.right))
    return pairs, pair_table


class FM363Parameters:
    """The 363-parameter SimFold model embedded in HotKnots."""

    def __init__(self, values: Sequence[float], parameter_dir: Path):
        if len(values) < 363:
            raise ValueError(f"FM363 parameter vector has {len(values)} values; expected at least 363")
        # SimFold stores these parameters in integer hundredths of kcal/mol.
        # Its legacy loader uses a direct C++ cast after multiplying by 100
        # (without the epsilon used by the PK-specific loader), so values such
        # as -0.58 become -0.57 because of binary floating-point truncation.
        # Reproducing that quirk is necessary for bit-level HotKnots parity.
        self.values = tuple(math.trunc(float(value) * 100.0) / 100.0 for value in values[:363])
        self.parameter_dir = parameter_dir
        self._load()

    def _load(self) -> None:
        values = self.values
        index = 0

        self.stack: Dict[Tuple[int, int, int, int], float] = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        if not (_can_pair(i, j) and _can_pair(k, l)):
                            continue
                        if 1000 * i + 100 * j + 10 * k + l > 1000 * l + 100 * k + 10 * j + i:
                            continue
                        value = values[index]
                        index += 1
                        self.stack[i, j, k, l] = value
                        self.stack[l, k, j, i] = value

        self.tstackh: Dict[Tuple[int, int, int, int], float] = {}
        for i in range(4):
            for j in range(4):
                if not _can_pair(i, j):
                    continue
                for k in range(4):
                    for l in range(4):
                        self.tstackh[i, j, k, l] = values[index]
                        index += 1

        self.internal_au_closure = values[index]
        self.internal_ag_mismatch = values[index + 1]
        self.internal_uu_mismatch = values[index + 2]
        index += 3

        self.int11_explicit: Dict[Tuple[int, int, int, int, int, int], float] = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        for m in range(4):
                            for n in range(4):
                                selected = (
                                    ((i, j) in CG_PAIRS and (m, n) in CG_PAIRS and not _can_pair(k, l))
                                    or (_watson_crick(i, j) and _watson_crick(m, n) and k == U and l == U)
                                )
                                if not selected:
                                    continue
                                forward = 100000 * i + 10000 * j + 1000 * k + 100 * l + 10 * m + n
                                reverse = 100000 * n + 10000 * m + 1000 * l + 100 * k + 10 * j + i
                                if forward > reverse:
                                    continue
                                value = values[index]
                                index += 1
                                self.int11_explicit[i, j, k, l, m, n] = value
                                self.int11_explicit[n, m, l, k, j, i] = value
        self.internal11_basic_mismatch = values[index]
        self.internal11_gg_mismatch = values[index + 1]
        index += 2

        self.int21_explicit: Dict[Tuple[int, int, int, int, int, int, int], float] = {}
        for i, j in ((C, G), (G, C)):
            m, n = i, j
            for k in range(4):
                for l in range(4):
                    for o in range(4):
                        if not _can_pair(k, l) and not _can_pair(k, o):
                            self.int21_explicit[i, j, k, l, m, n, o] = values[index]
                            index += 1
        self.internal21_match = values[index]
        self.internal21_au_closure = values[index + 1]
        index += 2

        self.int22_explicit: Dict[Tuple[int, int, int, int, int, int, int, int], float] = {}
        for i in range(4):
            for j in range(4):
                if not _watson_crick(i, j):
                    continue
                for k in range(4):
                    for l in range(4):
                        if _watson_crick(k, l):
                            continue
                        m, n, o, p = j, i, l, k
                        forward = (
                            i * 10000000 + j * 1000000 + k * 100000 + l * 10000
                            + m * 1000 + n * 100 + o * 10 + p
                        )
                        reverse = (
                            n * 10000000 + m * 1000000 + p * 100000 + o * 10000
                            + j * 1000 + i * 100 + l * 10 + k
                        )
                        if forward > reverse:
                            continue
                        value = values[index]
                        index += 1
                        key = (i, j, k, l, m, n, o, p)
                        reverse_key = (n, m, p, o, j, i, l, k)
                        self.int22_explicit[key] = value
                        self.int22_explicit[reverse_key] = value
        self.internal22_delta_same_size = values[index]
        self.internal22_delta_different_size = values[index + 1]
        self.internal22_delta_one_stable = values[index + 2]
        self.internal22_delta_ac = values[index + 3]
        self.internal22_match = values[index + 4]
        index += 5

        self.dangle_top: Dict[Tuple[int, int, int], float] = {}
        self.dangle_bot: Dict[Tuple[int, int, int], float] = {}
        for target in (self.dangle_top, self.dangle_bot):
            for i in range(4):
                for j in range(4):
                    if not _can_pair(i, j):
                        continue
                    for k in range(4):
                        target[i, j, k] = values[index]
                        index += 1

        self.internal_size = {size: values[index + size - 4] for size in range(4, 7)}
        index += 3
        self.bulge_size = {size: values[index + size - 1] for size in range(1, 7)}
        index += 6
        self.hairpin_size = {size: values[index + size - 3] for size in range(3, 10)}
        index += 7

        self.terminal_au = values[index]
        self.hairpin_ggg = values[index + 1]
        self.hairpin_c1 = values[index + 2]
        self.hairpin_c2 = values[index + 3]
        self.hairpin_c3 = values[index + 4]
        index += 5
        self.multi_offset = values[index]
        self.multi_helix = values[index + 1]
        self.multi_free = values[index + 2]
        self.intermolecular = values[index + 3]
        index += 4

        tloop_rows = _read_data_rows(self.parameter_dir / "turner-tloop-rna.dat")
        self.tetraloops: Dict[str, float] = {}
        for row in tloop_rows:
            if len(row) >= 2:
                self.tetraloops[row[0].upper().replace("T", "U")] = values[index]
                index += 1

        if index != 363:
            raise RuntimeError(f"Internal FM363 mapping consumed {index} parameters instead of 363")

    def au_penalty(self, i: int, j: int) -> float:
        return self.terminal_au if _has_au_penalty(i, j) else 0.0

    @staticmethod
    def _large_loop(base: float, size: int, measured_max: int) -> float:
        cents = math.trunc(100.0 * 1.079 * math.log(size / measured_max))
        return base + cents / 100.0

    def size_penalty(self, size: int, loop_type: str) -> float:
        if loop_type == "H":
            if size in self.hairpin_size:
                return self.hairpin_size[size]
            return self._large_loop(self.hairpin_size[9], size, 9)
        if loop_type == "I":
            if size in self.internal_size:
                return self.internal_size[size]
            return self._large_loop(self.internal_size[6], size, 6)
        if loop_type == "B":
            if size in self.bulge_size:
                return self.bulge_size[size]
            return self._large_loop(self.bulge_size[6], size, 6)
        raise ValueError(f"Unknown loop type: {loop_type}")

    def hairpin_energy(
        self,
        sequence: str,
        encoded: Sequence[int],
        i: int,
        j: int,
        include_ggg: bool = True,
    ) -> float:
        size = j - i - 1
        if size < 3:
            raise ValueError(f"Hairpin ({i}, {j}) contains fewer than three unpaired bases")
        terminal = self.au_penalty(encoded[i], encoded[j]) if size == 3 else 0.0
        mismatch = 0.0 if size == 3 else self.tstackh[encoded[i], encoded[j], encoded[i + 1], encoded[j - 1]]
        bonus = self.tetraloops.get(sequence[i : j + 1], 0.0) if size == 4 else 0.0
        special = 0.0
        if include_ggg and i >= 2 and sequence[i - 2 : i + 1] == "GGG" and sequence[j] == "U":
            special += self.hairpin_ggg
        if all(base == "C" for base in sequence[i + 1 : j]):
            special += self.hairpin_c3 if size == 3 else self.hairpin_c2 + self.hairpin_c1 * size
        return self.size_penalty(size, "H") + mismatch + bonus + special + terminal

    def tstacki(self, i: int, j: int, k: int, l: int) -> float:
        value = self.internal_au_closure if _has_au_penalty(i, j) else 0.0
        if (k, l) in ((A, G), (G, A)):
            value += self.internal_ag_mismatch
        if k == U and l == U:
            value += self.internal_uu_mismatch
        return value

    def int11_energy(self, i: int, j: int, k: int, l: int, m: int, n: int) -> float:
        key = (i, j, k, l, m, n)
        if (i, j) in CG_PAIRS and (m, n) in CG_PAIRS:
            if _can_pair(k, l):
                return self.internal11_basic_mismatch
            return self.int11_explicit[key]
        if _watson_crick(i, j) and _watson_crick(m, n) and k == U and l == U:
            return self.int11_explicit[key]
        value = self.internal11_gg_mismatch if k == G and l == G else self.internal11_basic_mismatch
        if _has_au_penalty(i, j):
            value += self.internal_au_closure
        if _has_au_penalty(m, n):
            value += self.internal_au_closure
        return value

    @staticmethod
    def _half_sum(first: float, second: float) -> float:
        return (math.trunc(first * 50.0) + math.trunc(second * 50.0)) / 100.0

    def int21_energy(self, i: int, j: int, k: int, l: int, m: int, n: int, o: int) -> float:
        key = (i, j, k, l, m, n, o)
        if ((i, j, m, n) == (C, G, C, G) or (i, j, m, n) == (G, C, G, C)):
            if _can_pair(k, l) or _can_pair(k, o):
                return self.internal21_match
            return self.int21_explicit[key]
        if _can_pair(k, l) or _can_pair(k, o):
            value = self.internal21_match
        else:
            first = self.int21_explicit[C, G, k, l, C, G, o]
            second = self.int21_explicit[G, C, k, l, G, C, o]
            value = self._half_sum(first, second)
        if _has_au_penalty(i, j):
            value += self.internal21_au_closure
        if _has_au_penalty(m, n):
            value += self.internal21_au_closure
        return value

    @staticmethod
    def _int22_delta_group(k: int, l: int, o: int, p: int) -> int:
        if (k, l) in ((A, C), (C, A)) or (o, p) in ((A, C), (C, A)):
            return 4
        purine = {A, G}
        pyrimidine = {C, U}
        pairs = ((k, l), (o, p))
        same_classes = (
            all(x in purine and y in purine for x, y in pairs)
            or all(x in pyrimidine and y in pyrimidine for x, y in pairs)
            or all((x in purine) != (y in purine) for x, y in pairs)
        )
        unstable = {(A, A), (C, C), (C, U), (U, C), (G, G)}
        if same_classes or ((k, l) in unstable and (o, p) in unstable):
            return 1
        stable = {(G, U), (U, G), (G, A), (A, G), (U, U)}
        if (k, l) in stable and (o, p) in stable:
            return 2
        if ((k, l) in stable and (o, p) in unstable) or ((o, p) in stable and (k, l) in unstable):
            return 3
        raise ValueError(f"Cannot classify 2x2 internal-loop mismatches {(k, l)} and {(o, p)}")

    def int22_energy(
        self, i: int, j: int, k: int, l: int, m: int, n: int, o: int, p: int
    ) -> float:
        ii = A if (i, j) == (G, U) else i
        jj = A if (i, j) == (U, G) else j
        mm = A if (m, n) == (G, U) else m
        nn = A if (m, n) == (U, G) else n
        if _watson_crick(k, l) or _watson_crick(o, p):
            return self.internal22_match
        if nn == ii and mm == jj and p == k and o == l:
            return self.int22_explicit[ii, jj, k, l, mm, nn, o, p]
        first = self.int22_explicit[ii, jj, k, l, jj, ii, l, k]
        second = self.int22_explicit[nn, mm, p, o, mm, nn, o, p]
        value = self._half_sum(first, second)
        group = self._int22_delta_group(k, l, o, p)
        return value + {
            1: self.internal22_delta_same_size,
            2: self.internal22_delta_different_size,
            3: self.internal22_delta_one_stable,
            4: self.internal22_delta_ac,
        }[group]

    def single_energy(self, encoded: Sequence[int], i: int, j: int, ip: int, jp: int) -> float:
        branch1 = ip - i - 1
        branch2 = j - jp - 1
        if branch1 == 0 and branch2 == 0:
            return self.stack[encoded[i], encoded[j], encoded[ip], encoded[jp]]
        if branch1 == branch2 == 1:
            return self.int11_energy(
                encoded[i], encoded[j], encoded[i + 1], encoded[j - 1], encoded[ip], encoded[jp]
            )
        if branch1 == 1 and branch2 == 2:
            return self.int21_energy(
                encoded[i], encoded[j], encoded[i + 1], encoded[j - 1], encoded[ip], encoded[jp], encoded[jp + 1]
            )
        if branch1 == 2 and branch2 == 1:
            return self.int21_energy(
                encoded[jp], encoded[ip], encoded[j - 1], encoded[ip - 1], encoded[j], encoded[i], encoded[i + 1]
            )
        if branch1 == branch2 == 2:
            return self.int22_energy(
                encoded[i], encoded[j], encoded[i + 1], encoded[j - 1], encoded[ip], encoded[jp],
                encoded[ip - 1], encoded[jp + 1],
            )
        if branch1 == 0 or branch2 == 0:
            size = branch1 + branch2
            value = self.size_penalty(size, "B")
            if size == 1:
                return value + self.stack[encoded[i], encoded[j], encoded[ip], encoded[jp]]
            return value + self.au_penalty(encoded[i], encoded[j]) + self.au_penalty(encoded[ip], encoded[jp])

        size = branch1 + branch2
        value = self.size_penalty(size, "I") + min(3.0, 0.5 * abs(branch1 - branch2))
        if branch1 == 1 or branch2 == 1:  # GAIL rule is fixed to one in HotKnots.
            value += self.tstacki(encoded[i], encoded[j], A, A)
            value += self.tstacki(encoded[jp], encoded[ip], A, A)
        else:
            value += self.tstacki(encoded[i], encoded[j], encoded[i + 1], encoded[j - 1])
            value += self.tstacki(encoded[jp], encoded[ip], encoded[jp + 1], encoded[ip - 1])
        return value


class _SecondaryStructureEvaluator:
    def __init__(
        self,
        sequence: str,
        pair_table: Sequence[int],
        parameters: FM363Parameters,
        include_exterior_ggg: bool = True,
        context_start: int = 0,
    ):
        self.sequence = sequence
        self.encoded = tuple(NUC_TO_INT[base] for base in sequence)
        self.pair_table = tuple(pair_table)
        self.parameters = parameters
        self.include_exterior_ggg = include_exterior_ggg
        self.context_start = context_start

    @staticmethod
    def _make_forest(
        pairs: Iterable[Pair], placeholder_pairs: Iterable[Pair] = ()
    ) -> List[_LoopNode]:
        placeholders = set(placeholder_pairs)
        all_pairs = set(pairs) | placeholders
        nodes: List[_LoopNode] = []
        stack: List[_LoopNode] = []
        for pair in sorted(all_pairs, key=lambda item: (item[0], -item[1])):
            i, j = pair
            while stack and i > stack[-1].pair[1]:
                stack.pop()
            if stack and j > stack[-1].pair[1]:
                raise ValueError("Crossing pairs cannot be evaluated as a secondary-structure loop tree")
            node = _LoopNode(pair, [], pair in placeholders)
            if stack:
                stack[-1].children.append(node)
            else:
                nodes.append(node)
            stack.append(node)
        return nodes

    @staticmethod
    def _unpaired_on_face(i: int, j: int, children: Sequence[_LoopNode]) -> int:
        cursor = i + 1
        count = 0
        for child in children:
            count += child.pair[0] - cursor
            cursor = child.pair[1] + 1
        return count + j - cursor

    def _top(self, pair: Pair, base: int) -> float:
        i, j = pair
        return self.parameters.dangle_top[self.encoded[j], self.encoded[i], self.encoded[base]]

    def _bot(self, pair: Pair, base: int) -> float:
        i, j = pair
        return self.parameters.dangle_bot[self.encoded[j], self.encoded[i], self.encoded[base]]

    @staticmethod
    def _join_dangles(left: float, right: float, gap: int) -> float:
        if gap <= 0:
            return 0.0
        if gap == 1:
            return left  # HotKnots sets simple_dangling_ends=1 (take the 3' dangle).
        return left + right

    def _multi_dangles(self, pair: Pair, children: Sequence[_LoopNode]) -> float:
        i, j = pair
        first_node = children[0]
        first = first_node.pair
        first_gap = first[0] - i - 1
        if first_node.placeholder:
            # SimFold represents a nested pseudoknot as <xxx>.  No dangle is
            # allowed onto the angle branch; the outer pair may retain its 3'
            # dangle only when at least two free bases separate the branches.
            value = (
                self.parameters.dangle_top[
                    self.encoded[i], self.encoded[j], self.encoded[i + 1]
                ]
                if first_gap >= 2
                else 0.0
            )
        else:
            value = self._join_dangles(
                self.parameters.dangle_top[
                    self.encoded[i], self.encoded[j], self.encoded[i + 1]
                ],
                self._bot(first, first[0] - 1),
                first_gap,
            )
        for left_node, right_node in zip(children, children[1:]):
            left, right = left_node.pair, right_node.pair
            gap = right[0] - left[1] - 1
            if left_node.placeholder and right_node.placeholder:
                continue
            if left_node.placeholder:
                if gap >= 2:
                    value += self._bot(right, right[0] - 1)
                continue
            if right_node.placeholder:
                if gap >= 2:
                    value += self._top(left, left[1] + 1)
                continue
            value += self._join_dangles(
                self._top(left, left[1] + 1),
                self._bot(right, right[0] - 1),
                gap,
            )
        last_node = children[-1]
        last = last_node.pair
        last_gap = j - last[1] - 1
        if last_node.placeholder:
            if last_gap >= 2:
                value += self.parameters.dangle_bot[
                    self.encoded[i], self.encoded[j], self.encoded[j - 1]
                ]
        else:
            value += self._join_dangles(
                self._top(last, last[1] + 1),
                self.parameters.dangle_bot[
                    self.encoded[i], self.encoded[j], self.encoded[j - 1]
                ],
                last_gap,
            )
        return value

    def _score_node(self, node: _LoopNode, exterior: bool = False) -> float:
        if node.placeholder:
            return 0.0
        i, j = node.pair
        subtotal = sum(self._score_node(child) for child in node.children)
        if not node.children:
            return subtotal + self.parameters.hairpin_energy(
                self.sequence,
                self.encoded,
                i,
                j,
                include_ggg=(self.include_exterior_ggg or not exterior)
                and i >= self.context_start + 2,
            )
        if len(node.children) == 1 and not node.children[0].placeholder:
            ip, jp = node.children[0].pair
            return subtotal + self.parameters.single_energy(self.encoded, i, j, ip, jp)

        unpaired = self._unpaired_on_face(i, j, node.children)
        local = (
            self.parameters.multi_offset
            + self.parameters.multi_helix * (len(node.children) + 1)
            + self.parameters.multi_free * unpaired
            + self.parameters.au_penalty(self.encoded[i], self.encoded[j])
        )
        for child in node.children:
            if child.placeholder:
                continue
            p, q = child.pair
            local += self.parameters.au_penalty(self.encoded[p], self.encoded[q])
        # HotKnots' ``energy_no_dangling`` label is slightly misleading:
        # computeEnergy subtracts only EnergyDanglingViaSimfold(), which contains
        # exterior-loop dangles.  Multiloop dangles remain in both columns.
        local += self._multi_dangles(node.pair, node.children)
        return subtotal + local

    def score(
        self,
        pairs: Iterable[Pair],
        include_dangles: bool,
        include_exterior_au: bool = True,
        placeholder_pairs: Iterable[Pair] = (),
    ) -> float:
        forest = self._make_forest(pairs, placeholder_pairs)
        value = sum(self._score_node(node, exterior=True) for node in forest)
        if not forest:
            return value

        if include_exterior_au:
            for node in forest:
                if node.placeholder:
                    continue
                i, j = node.pair
                value += self.parameters.au_penalty(self.encoded[i], self.encoded[j])

        if include_dangles:
            first_node = forest[0]
            first = first_node.pair
            if first[0] > 0 and not first_node.placeholder:
                value += self._bot(first, first[0] - 1)
            for left_node, right_node in zip(forest, forest[1:]):
                left, right = left_node.pair, right_node.pair
                gap = right[0] - left[1] - 1
                if left_node.placeholder and right_node.placeholder:
                    continue
                if left_node.placeholder:
                    if gap >= 2:
                        value += self._bot(right, right[0] - 1)
                elif right_node.placeholder:
                    if gap >= 2:
                        value += self._top(left, left[1] + 1)
                else:
                    value += self._join_dangles(
                        self._top(left, left[1] + 1),
                        self._bot(right, right[0] - 1),
                        gap,
                    )
            last_node = forest[-1]
            last = last_node.pair
            if last[1] + 1 < len(self.sequence) and not last_node.placeholder:
                value += self._top(last, last[1] + 1)
        return value


class _CCParameters:
    def __init__(self, values: Sequence[float], parameter_dir: Path):
        if len(values) != 923:
            raise ValueError(f"CC parameter vector has {len(values)} values; expected 923")
        self.values = tuple(values)
        self.fm = FM363Parameters(values[:363], parameter_dir)
        self.dp = _dp_parameters(values)
        index = 377

        self.coax_flush: Dict[Tuple[int, int, int, int], float] = {}
        for i in range(4):
            for j in range(4):
                if not _can_pair(i, j):
                    continue
                for k in range(4):
                    for l in range(4):
                        if _can_pair(k, l):
                            self.coax_flush[i, j, k, l] = values[index]
                            index += 1

        self.coax_m1: Dict[Tuple[int, int, int, int], float] = {}
        for i in range(4):
            for j in range(4):
                if not _can_pair(i, j):
                    continue
                for k in range(4):
                    for l in range(4):
                        self.coax_m1[i, j, k, l] = values[index]
                        index += 1

        self.coax_m2: Dict[Tuple[int, int, int, int], float] = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        if _can_pair(i, j) or _can_pair(k, l):
                            self.coax_m2[i, j, k, l] = values[index]
                            index += 1

        rows = _read_data_rows(parameter_dir / "pkmodelCC2006.dat")
        if len(rows) < 30:
            raise ValueError("Could not parse pkmodelCC2006.dat")
        s2_mask, s1_mask = rows[:11], rows[11:22]
        s2_formula_rows, s1_formula_rows = rows[22:26], rows[26:30]
        self.s2_l1: Dict[Tuple[int, int], float] = {}
        self.s1_l2: Dict[Tuple[int, int], float] = {}
        for target, mask in ((self.s2_l1, s2_mask), (self.s1_l2, s1_mask)):
            for i, row in enumerate(mask):
                for j, token in enumerate(row):
                    if token != ".":
                        target[i, j] = values[index]
                        index += 1
        self.s2_formula = [[float(token) for token in row] for row in s2_formula_rows]
        self.s1_formula = [[float(token) for token in row] for row in s1_formula_rows]
        for target in (self.s2_formula, self.s1_formula):
            for row in range(1, 4):
                for column in range(11):
                    target[row][column] = values[index]
                    index += 1
        if index != 923:
            raise RuntimeError(f"Internal CC mapping consumed {index} parameters instead of 923")

    @staticmethod
    def entropy_penalty(loop: int, stem: int, table: Mapping[Tuple[int, int], float], formula: Sequence[Sequence[float]]) -> Optional[float]:
        row = stem - 2
        if loop <= 0 or row < 0 or row >= len(formula[0]):
            return None
        if loop <= 12:
            value = table.get((row, loop - 1))
            return None if value is None else KB * TEMPERATURE_K * value
        l_min = formula[0][row]
        shifted = loop - l_min + 1
        if shifted <= 0:
            return None
        ln_coil = 2.14 * loop + 0.10
        ln_folded = formula[1][row] * math.log(shifted) + formula[2][row] * shifted + formula[3][row]
        return KB * TEMPERATURE_K * (ln_coil - ln_folded)


def _dp_parameters(values: Sequence[float]) -> Dict[str, float]:
    names = ("Ps", "Psm", "Psp", "Pb", "Pup", "Pps", "stP", "intP", "a", "b", "c", "a_p", "b_p", "c_p")
    return dict(zip(names, values[363:377]))


class HotKnotsEnergy:
    """Runtime-standalone Python port of ``HotKnots.compute_energy`` behavior."""

    MODELS = {
        "DP03": ("parameters_DP03.txt", "DP"),
        "DP09": ("parameters_DP09.txt", "DP"),
        "CC06": ("parameters_CC06.txt", "CC"),
        "CC09": ("parameters_CC09.txt", "CC"),
        "RE": (None, "RE"),
    }

    def __init__(self, parameter_dir: Optional[str] = None):
        """Load the packaged energy tables.

        ``parameter_dir`` normally points directly at a directory containing
        the eight data files used by this module.  For compatibility with
        earlier callers, a HotKnots source root containing ``bin/params`` is
        accepted too; the default has no dependency on that source tree.
        """

        if parameter_dir is None:
            resolved_parameter_dir = Path(__file__).resolve().parent / "energy_params"
        else:
            supplied = Path(parameter_dir)
            legacy_parameter_dir = supplied / "bin" / "params"
            resolved_parameter_dir = (
                legacy_parameter_dir if legacy_parameter_dir.is_dir() else supplied
            )
        self.parameter_dir = resolved_parameter_dir
        baseline_path = self.parameter_dir / "turner_parameters_fm363_constrdangles.txt"
        self.baseline = FM363Parameters(_read_numbers(baseline_path), self.parameter_dir)
        self._cache: Dict[str, object] = {}
        self.re_parameters = self._load_re_parameters()

    def _load_re_parameters(self) -> Dict[str, float]:
        values = _read_numbers(self.parameter_dir / "pkmodelRE.dat")
        names = (
            "g_interiorPseudo", "P_tilda", "P_i", "Q_tilda", "M_tilda",
            "Gw", "Gwh", "Gwi", "q_unpairedMultiPseudo", "p_pairedMultiPseudo",
        )
        if len(values) != len(names):
            raise ValueError("Could not parse pkmodelRE.dat")
        return dict(zip(names, values))

    def _parameters_for(self, model: str) -> object:
        cached = self._cache.get(model)
        if cached is not None:
            return cached
        filename, family = self.MODELS[model]
        if family == "RE":
            result: object = self.baseline
        else:
            values = _read_numbers(self.parameter_dir / str(filename))
            result = _CCParameters(values, self.parameter_dir) if family == "CC" else (
                FM363Parameters(values[:363], self.parameter_dir), _dp_parameters(values)
            )
        self._cache[model] = result
        return result

    @staticmethod
    def _validate(sequence: str, structure: str, pairs: Sequence[ParsedPair]) -> None:
        if len(sequence) != len(structure):
            raise ValueError(f"Sequence and structure lengths differ ({len(sequence)} != {len(structure)})")
        invalid = sorted(set(sequence) - set("ACGU"))
        if invalid:
            raise ValueError(f"Sequence contains unsupported bases: {''.join(invalid)}")
        encoded = [NUC_TO_INT[base] for base in sequence]
        for pair in pairs:
            if not _can_pair(encoded[pair.left], encoded[pair.right]):
                bases = sequence[pair.left] + sequence[pair.right]
                raise ValueError(f"Non-canonical pair {bases} at ({pair.left}, {pair.right})")

    @staticmethod
    def _band_energy(stem: Sequence[Pair], encoded: Sequence[int], fm: FM363Parameters, stack_scale: float, loop_scale: float) -> float:
        value = 0.0
        for outer, inner in zip(stem, stem[1:]):
            i, j = outer
            ip, jp = inner
            local = fm.single_energy(encoded, i, j, ip, jp)
            # DP's original band traversal labels the term from the distance
            # on the 5' arm only (``ap == a + 1``).  Consequently a right-side
            # bulge with adjacent 5' pairs receives stP, not intP.
            value += local * (stack_scale if ip == i + 1 else loop_scale)
        return value

    @staticmethod
    def _h_terminal_au(h_type: _HType, encoded: Sequence[int], fm: FM363Parameters) -> float:
        au = 0.0
        for stem in (h_type.stem1, h_type.stem2):
            i, j = stem[0]
            au += fm.au_penalty(encoded[i], encoded[j])
        return au

    def _score_dp(
        self, h_type: _HType, encoded: Sequence[int], pair_table: Sequence[int], fm: FM363Parameters,
        dp: Mapping[str, float],
    ) -> Tuple[float, float, Dict[str, float]]:
        band = self._band_energy(h_type.stem1, encoded, fm, dp["stP"], dp["intP"])
        band += self._band_energy(h_type.stem2, encoded, fm, dp["stP"], dp["intP"])
        penalty = dp["Ps"] + 2 * dp["Pb"] + h_type.unpaired * dp["Pup"]
        au = self._h_terminal_au(h_type, encoded, fm)
        no_dangling = band + penalty + au
        return no_dangling, no_dangling, {
            "band_energy": band, "pseudoknot_penalty": penalty, "terminal_au": au
        }

    def _cc_coaxial(
        self,
        h_type: _HType,
        encoded: Sequence[int],
        pair_table: Sequence[int],
        cc: _CCParameters,
        restrictions: set,
    ) -> float:
        ap, bp = h_type.stem1[-1]
        cp, dp = h_type.stem2[-1]

        # LEcoax_stack_energy_{flush_b,mismatch} compares a possible coaxial
        # stack with the dangling ends that would otherwise occupy the same
        # junction.  A non-zero coaxial term is used only when it is strictly
        # more favourable.  The two outside dangles can already have been
        # reserved by a nested loop, so mirror cannot_add_dangling as well.
        outside_first = ap + 1
        outside_second = dp - 1
        dangle_first = (
            outside_first
            if pair_table[outside_first] < 0 and outside_first not in restrictions
            else None
        )
        dangle_second = (
            outside_second
            if pair_table[outside_second] < 0 and outside_second not in restrictions
            else None
        )

        dangle_energy_first = 0.0
        if dangle_first is not None:
            dangle_energy_first = min(
                0.0,
                cc.fm.dangle_bot[encoded[cp], encoded[dp], encoded[dangle_first]],
            )
            # With HotKnots' simple_dangling_ends=1, an immediately adjacent
            # pair on the other side owns its own shared base instead.  The C
            # expression uses j-1 (outside_second), not dangle_i.
            adjacent = dp - 2
            adjacent_partner = pair_table[adjacent]
            if adjacent_partner >= 0:
                dangle_energy_first = min(
                    0.0,
                    cc.fm.dangle_top[
                        encoded[adjacent],
                        encoded[adjacent_partner],
                        encoded[outside_second],
                    ],
                )

        dangle_energy_second = 0.0
        if dangle_second is not None:
            dangle_energy_second = min(
                0.0,
                cc.fm.dangle_top[encoded[ap], encoded[bp], encoded[dangle_second]],
            )

        if h_type.loop3 == 0:
            coaxial = cc.coax_flush[
                encoded[cp], encoded[dp], encoded[bp], encoded[ap]
            ]
            competing_dangles = dangle_energy_first + dangle_energy_second
        elif h_type.loop3 == 1:
            middle = cp + 1
            coaxial_first = 0.0
            if dangle_first is not None:
                coaxial_first = (
                    cc.coax_m1[
                        encoded[cp], encoded[dp], encoded[middle], encoded[dangle_first]
                    ]
                    + cc.coax_m2[
                        encoded[middle], encoded[dangle_first], encoded[bp], encoded[ap]
                    ]
                )
            coaxial_second = 0.0
            if dangle_second is not None:
                coaxial_second = (
                    cc.coax_m1[
                        encoded[ap], encoded[bp], encoded[dangle_second], encoded[middle]
                    ]
                    + cc.coax_m2[
                        encoded[cp], encoded[dp], encoded[middle], encoded[dangle_second]
                    ]
                )
            coaxial = min(coaxial_first, coaxial_second)
            middle_dangle = min(
                0.0,
                cc.fm.dangle_top[encoded[cp], encoded[dp], encoded[middle]],
            )
            competing_dangles = (
                middle_dangle + dangle_energy_first + dangle_energy_second
            )
        else:
            return 0.0

        if competing_dangles <= coaxial or coaxial >= INF:
            return 0.0
        if dangle_first is not None:
            restrictions.add(dangle_first)
        if dangle_second is not None:
            restrictions.add(dangle_second)
        return coaxial

    def _score_cc(
        self,
        h_type: _HType,
        encoded: Sequence[int],
        pair_table: Sequence[int],
        cc: _CCParameters,
        restrictions: set,
    ) -> Tuple[float, float, Dict[str, float]]:
        stem1, stem2 = len(h_type.stem1), len(h_type.stem2)
        # The CC entropy rows exist only for stems 2--12.  HotKnots checks the
        # applicability bounds before indexing them and otherwise uses DP.
        if (
            stem1 <= 1 or stem2 <= 1 or stem1 > 12 or stem2 > 12
            or h_type.loop1 == 0 or h_type.loop2 == 0
        ):
            return self._score_dp(h_type, encoded, pair_table, cc.fm, cc.dp)
        entropy1 = cc.entropy_penalty(h_type.loop1, stem2, cc.s2_l1, cc.s2_formula)
        entropy2 = cc.entropy_penalty(h_type.loop2, stem1, cc.s1_l2, cc.s1_formula)
        if entropy1 is None or entropy2 is None:
            return self._score_dp(h_type, encoded, pair_table, cc.fm, cc.dp)

        band = self._band_energy(h_type.stem1, encoded, cc.fm, 1.0, 1.0)
        band += self._band_energy(h_type.stem2, encoded, cc.fm, 1.0, 1.0)
        assembly = KB * TEMPERATURE_K * math.log(9.0)
        coaxial = self._cc_coaxial(h_type, encoded, pair_table, cc, restrictions)
        au = self._h_terminal_au(h_type, encoded, cc.fm)
        no_dangling = band + assembly + float(entropy1) + float(entropy2) + coaxial + au
        return no_dangling, no_dangling, {
            "band_energy": band,
            "assembly": assembly,
            "loop1_entropy": float(entropy1),
            "loop2_entropy": float(entropy2),
            "coaxial": coaxial,
            "terminal_au": au,
        }

    @staticmethod
    def _add_term(target: Dict[str, float], name: str, value: float) -> None:
        target[name] = target.get(name, 0.0) + value

    @staticmethod
    def _is_pk_free(node: _ClosedRegion) -> bool:
        return node.kind != "pseudo" and all(
            HotKnotsEnergy._is_pk_free(child) for child in node.children
        )

    @staticmethod
    def _top_pseudoknots(node: _ClosedRegion) -> Tuple[_ClosedRegion, ...]:
        result: List[_ClosedRegion] = []

        def visit(current: _ClosedRegion) -> None:
            for child in current.children:
                if child.kind == "pseudo":
                    result.append(child)
                else:
                    visit(child)

        visit(node)
        return tuple(result)

    @staticmethod
    def _pairs_in_interval(
        pair_table: Sequence[int], begin: int, end: int,
        excluded: Sequence[_ClosedRegion] = (),
    ) -> Tuple[Pair, ...]:
        return tuple(
            (left, right)
            for left, right in enumerate(pair_table)
            if begin <= left < right <= end
            and not any(region.begin <= left and right <= region.end for region in excluded)
        )

    @staticmethod
    def _actual_branch_pair(node: _ClosedRegion, pair_table: Sequence[int]) -> Pair:
        return node.begin, pair_table[node.begin]

    @staticmethod
    def _region_has_direct_pk_child(node: _ClosedRegion) -> bool:
        return any(child.kind == "pseudo" for child in node.children)

    @staticmethod
    def _descendants(node: _ClosedRegion) -> Iterable[_ClosedRegion]:
        yield node
        for child in node.children:
            yield from HotKnotsEnergy._descendants(child)

    @staticmethod
    def _ancestors(node: _ClosedRegion) -> Iterable[_ClosedRegion]:
        current = node.parent
        while current is not None:
            yield current
            current = current.parent

    @staticmethod
    def _mark_branch_restrictions(
        restrictions: set, children: Sequence[_ClosedRegion], pair_table: Sequence[int]
    ) -> None:
        for child in children:
            restrictions.add(child.begin - 1)
            restrictions.add(pair_table[child.begin] + 1)

    @staticmethod
    def _multi_face_unpaired(
        outer: Pair, children: Sequence[_ClosedRegion], pair_table: Sequence[int]
    ) -> int:
        i, j = outer
        if not children:
            return max(0, j - i - 1)
        count = children[0].begin - i - 1
        for left, right in zip(children, children[1:]):
            count += right.begin - pair_table[left.begin] - 1
        count += j - pair_table[children[-1].begin] - 1
        return count

    @staticmethod
    def _multi_dangles_for_regions(
        sequence: str,
        pair_table: Sequence[int],
        fm: FM363Parameters,
        outer: Pair,
        children: Sequence[_ClosedRegion],
    ) -> float:
        if not children:
            return 0.0
        evaluator = _SecondaryStructureEvaluator(sequence, pair_table, fm)
        branches = [
            _LoopNode((child.begin, pair_table[child.begin]), [])
            for child in children
        ]
        return evaluator._multi_dangles(outer, branches)

    def _score_pkfree_dpcc(
        self,
        sequence: str,
        pair_table: Sequence[int],
        node: _ClosedRegion,
        fm: FM363Parameters,
    ) -> float:
        pairs = self._pairs_in_interval(pair_table, node.begin, node.end)
        # A closed region is passed to SimFold as its own subsequence.  Its
        # exterior AU penalty is included, while exterior dangles are left for
        # HotKnots' final EnergyDangling scan.  Multiloop dangles are always
        # included by _score_node, matching get_feature_counts_restricted().
        return _SecondaryStructureEvaluator(
            sequence,
            pair_table,
            fm,
            include_exterior_ggg=False,
            context_start=node.begin,
        ).score(pairs, include_dangles=False, include_exterior_au=True)

    def _score_nested_scaffold(
        self,
        sequence: str,
        pair_table: Sequence[int],
        node: _ClosedRegion,
        pseudoknots: Sequence[_ClosedRegion],
        fm: FM363Parameters,
        restrictions: set,
    ) -> float:
        actual = self._pairs_in_interval(
            pair_table, node.begin, node.end, excluded=pseudoknots
        )
        placeholders = tuple((pk.begin, pk.end) for pk in pseudoknots)
        value = _SecondaryStructureEvaluator(
            sequence,
            pair_table,
            fm,
            include_exterior_ggg=False,
            context_start=node.begin,
        ).score(
            actual,
            include_dangles=False,
            include_exterior_au=True,
            placeholder_pairs=placeholders,
        )

        # HotKnots supplements the angle-bracket SimFold scaffold with the
        # actual-structure dangles whenever a multiloop with a direct PK branch
        # occurs below it.  The 2.0 code intentionally keeps i_pt at the current
        # scaffold root, so the root dangle pattern is added once per such loop.
        # Reproducing this detail is required for exact nested-PK parity.
        repeats = sum(
            1
            for descendant in self._descendants(node)
            if descendant.kind == "multi" and self._region_has_direct_pk_child(descendant)
        )
        if repeats and node.children:
            extra = self._multi_dangles_for_regions(
                sequence, pair_table, fm, (node.begin, pair_table[node.begin]), node.children
            )
            value += repeats * extra
            self._mark_branch_restrictions(restrictions, node.children, pair_table)
        return value

    def _score_span_multi_dp(
        self,
        sequence: str,
        pair_table: Sequence[int],
        loop: _SpanLoop,
        fm: FM363Parameters,
        dp: Mapping[str, float],
        restrictions: set,
    ) -> Tuple[float, Dict[str, float]]:
        children = loop.children
        penalty = dp["a_p"] + dp["b_p"] * (len(children) + 2) + dp["c_p"] * loop.unpaired
        au = fm.au_penalty(
            NUC_TO_INT[sequence[loop.outer[0]]], NUC_TO_INT[sequence[loop.outer[1]]]
        )
        encoded = tuple(NUC_TO_INT[base] for base in sequence)
        for child in children:
            if child.kind == "pseudo":
                left, right = self._actual_branch_pair(child, pair_table)
                au += fm.au_penalty(encoded[left], encoded[right])
        dangling = self._multi_dangles_for_regions(
            sequence, pair_table, fm, loop.outer, children
        )
        self._mark_branch_restrictions(restrictions, children, pair_table)
        return penalty + au + dangling, {
            "band_multiloop": penalty,
            "band_multiloop_au": au,
            "band_multiloop_dangling": dangling,
        }

    def _score_pseudo_dp_region(
        self,
        sequence: str,
        pair_table: Sequence[int],
        node: _ClosedRegion,
        fm: FM363Parameters,
        dp: Mapping[str, float],
        restrictions: set,
        breakdown: Dict[str, float],
    ) -> float:
        encoded = tuple(NUC_TO_INT[base] for base in sequence)
        band_energy = 0.0
        reported_band_energy = 0.0
        for loop in node.span_loops:
            if loop.is_multi:
                value, terms = self._score_span_multi_dp(
                    sequence, pair_table, loop, fm, dp, restrictions
                )
                band_energy += value
                for name, term in terms.items():
                    self._add_term(breakdown, name, term)
            else:
                local = fm.single_energy(
                    encoded, loop.outer[0], loop.outer[1], loop.inner[0], loop.inner[1]
                )
                scale = dp["stP"] if loop.inner[0] == loop.outer[0] + 1 else dp["intP"]
                scaled = scale * local
                band_energy += scaled
                reported_band_energy += scaled
                restrictions.update((loop.inner[0] - 1, loop.inner[1] + 1))

        unband_children = sum(child.nested == "un_band" for child in node.children)
        if node.parent is not None and (
            node.parent.kind == "multi" or node.nested == "in_band"
        ):
            initiation = dp["Psm"]
        elif node.nested == "un_band":
            initiation = dp["Psp"]
        else:
            initiation = dp["Ps"]
        penalty = (
            initiation
            + dp["Pb"] * len(node.bands)
            + dp["Pps"] * unband_children
            + dp["Pup"] * node.number_unpaired_in_pseudo
        )
        terminal_au = sum(
            fm.au_penalty(encoded[band.pairs[0][0]], encoded[band.pairs[0][1]])
            for band in node.bands
        )
        # In HotKnots' Loop::pseudoEnergyDP(), a multiloop that interrupts a
        # band creates another helix end: after scoring multiPseudoEnergyDP(),
        # the first band-spanning pair inside that multiloop receives an AU/GU
        # penalty.  In the loop tree this is exactly the inner pair of each
        # spanning multiloop.  It is separate from both the multiloop's own AU
        # terms and the outermost-pair penalty for each band above.
        terminal_au += sum(
            fm.au_penalty(encoded[loop.inner[0]], encoded[loop.inner[1]])
            for loop in node.span_loops
            if loop.is_multi
        )
        self._add_term(breakdown, "band_energy", reported_band_energy)
        self._add_term(breakdown, "pseudoknot_penalty", penalty)
        self._add_term(breakdown, "terminal_au", terminal_au)
        return band_energy + penalty + terminal_au

    def _score_dp_node(
        self,
        sequence: str,
        pair_table: Sequence[int],
        node: _ClosedRegion,
        fm: FM363Parameters,
        dp: Mapping[str, float],
        restrictions: set,
        breakdown: Dict[str, float],
    ) -> float:
        if self._is_pk_free(node):
            value = self._score_pkfree_dpcc(sequence, pair_table, node, fm)
            term = (
                "secondary_structure_outside_pseudoknot"
                if node.parent is not None and node.parent.kind == "external"
                else "secondary_structure"
            )
            self._add_term(breakdown, term, value)
            return value
        if node.kind != "pseudo":
            pseudoknots = self._top_pseudoknots(node)
            value = sum(
                self._score_dp_node(
                    sequence, pair_table, pk, fm, dp, restrictions, breakdown
                )
                for pk in pseudoknots
            )
            scaffold = self._score_nested_scaffold(
                sequence, pair_table, node, pseudoknots, fm, restrictions
            )
            self._add_term(breakdown, "secondary_structure_scaffold", scaffold)
            return value + scaffold

        value = sum(
            self._score_dp_node(
                sequence, pair_table, child, fm, dp, restrictions, breakdown
            )
            for child in node.children
        )
        return value + self._score_pseudo_dp_region(
            sequence, pair_table, node, fm, dp, restrictions, breakdown
        )

    @staticmethod
    def _as_h_type(node: _ClosedRegion) -> _HType:
        first, second = node.bands
        a, b = first.pairs[0]
        c, d = second.pairs[0]
        ap, bp = first.pairs[-1]
        cp, dp = second.pairs[-1]
        return _HType(
            first.pairs,
            second.pairs,
            c - ap - 1,
            dp - b - 1,
            bp - cp - 1,
            node.number_unpaired_in_pseudo,
            node.begin,
            node.end,
        )

    def _score_cc_node(
        self,
        sequence: str,
        pair_table: Sequence[int],
        node: _ClosedRegion,
        cc: _CCParameters,
        restrictions: set,
        breakdown: Dict[str, float],
        metadata: Dict[str, object],
    ) -> float:
        if self._is_pk_free(node):
            value = self._score_pkfree_dpcc(sequence, pair_table, node, cc.fm)
            term = (
                "secondary_structure_outside_pseudoknot"
                if node.parent is not None and node.parent.kind == "external"
                else "secondary_structure"
            )
            self._add_term(breakdown, term, value)
            return value
        if node.kind != "pseudo":
            pseudoknots = self._top_pseudoknots(node)
            value = sum(
                self._score_cc_node(
                    sequence, pair_table, pk, cc, restrictions, breakdown, metadata
                )
                for pk in pseudoknots
            )
            scaffold = self._score_nested_scaffold(
                sequence, pair_table, node, pseudoknots, cc.fm, restrictions
            )
            self._add_term(breakdown, "secondary_structure_scaffold", scaffold)
            return value + scaffold

        value = sum(
            self._score_cc_node(
                sequence, pair_table, child, cc, restrictions, breakdown, metadata
            )
            for child in node.children
        )
        if len(node.bands) != 2 or any(loop.is_multi for loop in node.span_loops):
            metadata["cc_fallback_to_dp"] = True
            metadata["cc_fallback_components"] = int(
                metadata.get("cc_fallback_components", 0)
            ) + 1
            return value + self._score_pseudo_dp_region(
                sequence, pair_table, node, cc.fm, cc.dp, restrictions, breakdown
            )

        h_type = self._as_h_type(node)
        _, local, terms = self._score_cc(
            h_type,
            tuple(NUC_TO_INT[b] for b in sequence),
            pair_table,
            cc,
            restrictions,
        )
        if "assembly" not in terms:
            metadata["cc_fallback_to_dp"] = True
            metadata["cc_fallback_components"] = int(
                metadata.get("cc_fallback_components", 0)
            ) + 1
            # _score_cc already evaluated the DP fallback for a simple H-type;
            # use the general routine so nested/unband counts and restrictions
            # remain correct for the full tree.
            local = self._score_pseudo_dp_region(
                sequence, pair_table, node, cc.fm, cc.dp, restrictions, breakdown
            )
            return value + local
        for name, term in terms.items():
            self._add_term(breakdown, name, term)
        for loop in node.span_loops:
            restrictions.update((loop.inner[0] - 1, loop.inner[1] + 1))
        return value + local

    def _score_re_span_multi(
        self,
        sequence: str,
        pair_table: Sequence[int],
        loop: _SpanLoop,
        restrictions: set,
    ) -> Tuple[float, Dict[str, float]]:
        fm = self.baseline
        encoded = tuple(NUC_TO_INT[base] for base in sequence)
        pseudo_branches = sum(2 if child.kind == "pseudo" else 1 for child in loop.children)
        penalty = (
            self.re_parameters["M_tilda"]
            + fm.multi_helix * pseudo_branches
            + 2 * self.re_parameters["p_pairedMultiPseudo"]
        )
        au = fm.au_penalty(encoded[loop.outer[0]], encoded[loop.outer[1]])
        for child in loop.children:
            left, right = self._actual_branch_pair(child, pair_table)
            au += fm.au_penalty(encoded[left], encoded[right])
        dangling = self._multi_dangles_for_regions(
            sequence, pair_table, fm, loop.outer, loop.children
        )
        self._mark_branch_restrictions(restrictions, loop.children, pair_table)
        return penalty + au + dangling, {
            "band_multiloop": penalty,
            "band_multiloop_au": au,
            "band_multiloop_dangling": dangling,
        }

    def _score_re_node(
        self,
        sequence: str,
        pair_table: Sequence[int],
        node: _ClosedRegion,
        restrictions: set,
        breakdown: Dict[str, float],
    ) -> float:
        fm = self.baseline
        encoded = tuple(NUC_TO_INT[base] for base in sequence)
        secondary_term = (
            "secondary_structure_outside_pseudoknot"
            if node.parent is not None and node.parent.kind == "external"
            else "secondary_structure"
        )
        subtotal = sum(
            self._score_re_node(sequence, pair_table, child, restrictions, breakdown)
            for child in node.children
        )
        if node.kind == "hairpin":
            local = fm.hairpin_energy(sequence, encoded, node.begin, node.end, include_ggg=True)
            self._add_term(breakdown, secondary_term, local)
            return subtotal + local
        if node.kind in ("stack", "interior"):
            child = node.children[0]
            local = fm.single_energy(
                encoded, node.begin, node.end, child.begin, child.end
            )
            restrictions.update((child.begin - 1, child.end + 1))
            self._add_term(breakdown, secondary_term, local)
            return subtotal + local
        if node.kind == "multi":
            branches = node.children
            pseudo_branches = sum(2 if child.kind == "pseudo" else 1 for child in branches)
            unpaired = self._multi_face_unpaired(
                (node.begin, node.end), branches, pair_table
            )
            local = (
                fm.multi_free * unpaired
                + fm.multi_offset
                + fm.multi_helix * (pseudo_branches + 1)
                + fm.au_penalty(encoded[node.begin], encoded[node.end])
            )
            for child in branches:
                left, right = self._actual_branch_pair(child, pair_table)
                local += fm.au_penalty(encoded[left], encoded[right])
            local += self._multi_dangles_for_regions(
                sequence, pair_table, fm, (node.begin, node.end), branches
            )
            self._mark_branch_restrictions(restrictions, branches, pair_table)
            self._add_term(breakdown, secondary_term, local)
            return subtotal + local
        if node.kind == "pseudo":
            unband_children = sum(child.nested == "un_band" for child in node.children)
            penalty = (
                self.re_parameters["Gw"]
                + self.re_parameters["Gwh"] * (len(node.bands) - 2)
                + self.re_parameters["Q_tilda"] * node.number_unpaired
                + self.re_parameters["P_tilda"] * 2 * len(node.bands)
                + self.re_parameters["P_i"] * unband_children
            )
            band = 0.0
            reported_band = 0.0
            for loop in node.span_loops:
                if loop.is_multi:
                    local, terms = self._score_re_span_multi(
                        sequence, pair_table, loop, restrictions
                    )
                    band += local
                    for name, term in terms.items():
                        self._add_term(breakdown, name, term)
                else:
                    local = fm.single_energy(
                        encoded,
                        loop.outer[0], loop.outer[1], loop.inner[0], loop.inner[1],
                    )
                    scaled = self.re_parameters["g_interiorPseudo"] * local
                    band += scaled
                    reported_band += scaled
                    restrictions.update((loop.inner[0] - 1, loop.inner[1] + 1))
            self._add_term(breakdown, "band_energy", reported_band)
            self._add_term(breakdown, "pseudoknot_penalty", penalty)
            return subtotal + band + penalty
        raise RuntimeError(f"Unknown closed-region kind {node.kind!r}")

    @staticmethod
    def _find_containing_region(tree: _LoopBandsTree, position: int) -> Optional[_ClosedRegion]:
        for begin in range(position, -1, -1):
            region = tree.by_begin.get(begin)
            if region is not None and region.end >= position:
                return region
        return None

    @staticmethod
    def _nested_band_start(region: _ClosedRegion, position: int, pair_table: Sequence[int]) -> Optional[int]:
        for start, end in region.band_regions:
            if start <= position <= end:
                partner = pair_table[position]
                if partner < 0 or start <= partner <= end:
                    return start
                return None
        return None

    @staticmethod
    def _is_adjacent_to_pk(
        tree: _LoopBandsTree, position: int, band_start: int = -1
    ) -> bool:
        if position + 1 < tree.length:
            next_region = tree.by_begin.get(position + 1)
            if next_region is not None and next_region.kind == "pseudo":
                return True
        if position > 0:
            partner = tree.pair_table[position - 1]
            if (
                0 <= partner < position - 1
                and partner not in tree.by_begin
                and partner > band_start
            ):
                return True
        return False

    @staticmethod
    def _pk_multiloop_dangle_positions(region: _ClosedRegion) -> set:
        return {
            position
            for loop in region.span_loops
            if loop.is_multi
            for position in (loop.inner[0] - 1, loop.inner[1] + 1)
        }

    def _residual_dangling(
        self,
        sequence: str,
        pair_table: Sequence[int],
        tree: _LoopBandsTree,
        fm: FM363Parameters,
        restrictions: set,
    ) -> float:
        encoded = tuple(NUC_TO_INT[base] for base in sequence)
        pk_multi_dangles = {
            node.begin: self._pk_multiloop_dangle_positions(node)
            for node in tree.pseudoknots
        }

        def should_add(position: int) -> bool:
            region = self._find_containing_region(tree, position)
            if region is None:
                return True
            if region.kind == "pseudo" and position in pk_multi_dangles.get(region.begin, set()):
                return True
            if self._is_pk_free(region):
                return False
            if region.kind != "pseudo":
                return self._is_adjacent_to_pk(tree, position)
            band_start = self._nested_band_start(region, position, pair_table)
            if band_start is not None and not self._is_adjacent_to_pk(
                tree, position, band_start
            ):
                return False
            return True

        value = 0.0
        for position, partner in enumerate(pair_table):
            if partner >= 0 or position in restrictions:
                continue
            previous = pair_table[position - 1] if position > 0 else -1
            following = pair_table[position + 1] if position + 1 < len(pair_table) else -1
            if (
                (position == 0 or previous < 0)
                and position + 1 < len(pair_table)
                and following > position + 1
            ):
                if should_add(position):
                    opener = position + 1
                    value += min(
                        0.0,
                        fm.dangle_bot[encoded[following], encoded[opener], encoded[position]],
                    )
            elif (
                (position + 1 == len(pair_table) or following < 0)
                and position > 0
                and 0 <= previous < position - 1
            ):
                if should_add(position):
                    closer = position - 1
                    value += min(
                        0.0,
                        fm.dangle_top[encoded[closer], encoded[previous], encoded[position]],
                    )
            elif (
                position > 0
                and position + 1 < len(pair_table)
                and 0 <= previous < position - 1
                and following > position + 1
            ):
                if should_add(position):
                    closer = position - 1
                    value += min(
                        0.0,
                        fm.dangle_top[encoded[closer], encoded[previous], encoded[position]],
                    )
        return value

    def compute_energy(self, sequence: str, structure: str, model: str = "DP09") -> Dict[str, object]:
        sequence = sequence.upper().replace("T", "U")
        if model not in self.MODELS:
            raise ValueError(f"Unknown model {model!r}; choose from {', '.join(self.MODELS)}")
        pairs, pair_table = parse_dot_bracket(structure)
        self._validate(sequence, structure, pairs)
        crossing = any(
            _crosses(first.pair, second.pair)
            for index, first in enumerate(pairs)
            for second in pairs[index + 1 :]
        )

        parameters = self._parameters_for(model)
        pair_list = [pair.pair for pair in pairs]
        metadata: Dict[str, object] = {}
        if not crossing:
            fm = self.baseline if model == "RE" else (
                parameters.fm if isinstance(parameters, _CCParameters) else parameters[0]
            )
            evaluator = _SecondaryStructureEvaluator(
                sequence,
                pair_table,
                fm,
                include_exterior_ggg=model == "RE",
            )
            include_exterior_au = model != "RE"
            no_dangling = evaluator.score(
                pair_list,
                include_dangles=False,
                include_exterior_au=include_exterior_au,
            )
            total = evaluator.score(
                pair_list,
                include_dangles=True,
                include_exterior_au=include_exterior_au,
            )
            breakdown = {"secondary_structure": no_dangling, "dangling": total - no_dangling}
        else:
            tree = _LoopBandsTree(pair_table)
            family = self.MODELS[model][1]
            fm = self.baseline if model == "RE" else (
                parameters.fm if isinstance(parameters, _CCParameters) else parameters[0]
            )
            breakdown: Dict[str, float] = {}
            restrictions: set = set()
            if family == "DP":
                no_dangling = sum(
                    self._score_dp_node(
                        sequence,
                        pair_table,
                        node,
                        fm,
                        parameters[1],
                        restrictions,
                        breakdown,
                    )
                    for node in tree.root.children
                )
            elif family == "CC":
                no_dangling = sum(
                    self._score_cc_node(
                        sequence,
                        pair_table,
                        node,
                        parameters,
                        restrictions,
                        breakdown,
                        metadata,
                    )
                    for node in tree.root.children
                )
            else:
                no_dangling = sum(
                    self._score_re_node(
                        sequence, pair_table, node, restrictions, breakdown
                    )
                    for node in tree.root.children
                )

            dangling = self._residual_dangling(
                sequence, pair_table, tree, fm, restrictions
            )
            total = no_dangling + dangling
            breakdown["dangling"] = dangling
            metadata["pseudoknot_components"] = len(tree.pseudoknots)
            metadata["bands_per_pseudoknot"] = [
                len(node.bands) for node in tree.pseudoknots
            ]
            metadata["nested_pseudoknots"] = sum(
                any(ancestor.kind == "pseudo" for ancestor in self._ancestors(node))
                for node in tree.pseudoknots
            )
            metadata["topology"] = [
                "h_type" if len(node.bands) == 2 else "chain_or_kissing"
                for node in tree.pseudoknots
            ]

        return {
            "energy": total,
            "energy_no_dangling": no_dangling,
            "model": model,
            "breakdown": breakdown,
            "metadata": metadata,
        }


def compute_energy(sequence: str, structure: str, model: str = "DP09") -> float:
    """Convenience API returning only the free energy in kcal/mol."""

    return float(HotKnotsEnergy().compute_energy(sequence, structure, model)["energy"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runtime-standalone Python port of the HotKnots DP/CC/RE energy evaluator"
    )
    parser.add_argument("-s", "--sequence", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("-m", "--model", default="DP09", choices=tuple(HotKnotsEnergy.MODELS))
    args = parser.parse_args()
    result = HotKnotsEnergy().compute_energy(args.sequence, args.structure, args.model)
    print(f"{result['model']}\t{result['energy']:.6f}\t{result['energy_no_dangling']:.6f}")


if __name__ == "__main__":
    main()
