"""Tests for CPLfold's runtime-standalone, HotKnots-derived energy evaluator."""

import importlib
import sys
import types
import unittest
from unittest import mock


def _load_cplfold_with_numba_stub():
    """Import CPLfold without paying the JIT startup cost in this unit test."""

    if "numba" not in sys.modules:
        numba = types.ModuleType("numba")

        def njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda function: function

        numba.njit = njit
        sys.modules["numba"] = numba
    return importlib.import_module("CPLfold")


class _BrokenEvaluator:
    def compute_energy(self, sequence, structure, model):
        raise RuntimeError("unexpected evaluator defect")


class _MissingEnergyEvaluator:
    def compute_energy(self, sequence, structure, model):
        return {"model": model}


class _SelectiveEvaluator:
    def compute_energy(self, sequence, structure, model):
        return {"energy": -2.0 if "[" in structure else -1.0}


class CPLfoldEnergyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cplfold = _load_cplfold_with_numba_stub()

    def test_unexpected_evaluator_error_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected evaluator defect"):
            self.cplfold.compute_energy_hotknots(
                "GAAAC", "(...)", _BrokenEvaluator(), "DP09"
            )

        with self.assertRaises(KeyError):
            self.cplfold.compute_energy_hotknots(
                "GAAAC", "(...)", _MissingEnergyEvaluator(), "DP09"
            )

    def test_complex_merged_candidate_enters_ranking(self):
        merged = "((.(...).[[...))...]]"
        phase1 = "".join(symbol if symbol in "()" else "." for symbol in merged)
        phase2 = "".join(
            "(" if symbol == "[" else ")" if symbol == "]" else "."
            for symbol in merged
        )
        with mock.patch.object(self.cplfold, "BeamCKYParserHyper"), mock.patch.object(
            self.cplfold, "HotKnotsEnergy", return_value=_SelectiveEvaluator()
        ), mock.patch.object(
            self.cplfold, "phase1_fold", return_value=[(phase1, -1.0)]
        ), mock.patch.object(
            self.cplfold, "phase2_fold", return_value=(phase2, -1.0)
        ):
            results = self.cplfold.two_phase_pseudoknot_fold(
                "G" * len(merged), verbose=False
            )

        self.assertEqual(
            {item["type"] for item in results}, {"phase1", "pseudoknot"}
        )
        pseudoknot = next(item for item in results if item["type"] == "pseudoknot")
        self.assertEqual(pseudoknot["structure"], merged)
        self.assertEqual(pseudoknot["energy"], -2.0)
        self.assertTrue(all(item["energy"] is not None for item in results))

    def test_pseudoknot_free_mode_skips_phase2(self):
        with mock.patch.object(self.cplfold, "BeamCKYParserHyper") as parser_cls, mock.patch.object(
            self.cplfold, "HotKnotsEnergy", return_value=_SelectiveEvaluator()
        ), mock.patch.object(
            self.cplfold, "phase1_fold", return_value=[("(...)", -1.0)]
        ), mock.patch.object(self.cplfold, "phase2_fold") as phase2:
            results = self.cplfold.two_phase_pseudoknot_fold(
                "G" * 5,
                verbose=False,
                allow_pseudoknot=False,
                beta=0.5,
            )

        phase2.assert_not_called()
        parser_cls.assert_called_once_with(
            beam_size=100,
            lv=True,
            use_constraints=False,
            is_verbose=False,
        )
        self.assertEqual([item["type"] for item in results], ["phase1"])
        self.assertEqual(results[0]["structure"], "(...)")


if __name__ == "__main__":
    unittest.main()
