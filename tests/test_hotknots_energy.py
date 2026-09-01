"""Regression tests against the original HotKnots 2.0 computeEnergy binary."""

import unittest
from pathlib import Path

from Utils.hotknots_energy import HotKnotsEnergy


REFERENCE_SEQUENCE = "GGCGCGGCACCGUCCGCGGAACAAACGG"


class HotKnotsEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = HotKnotsEnergy()

    def assert_energy(self, sequence, structure, model, expected, places=4):
        result = self.evaluator.compute_energy(sequence, structure, model)
        self.assertAlmostEqual(result["energy"], expected[0], places=places)
        self.assertAlmostEqual(result["energy_no_dangling"], expected[1], places=places)
        self.assertAlmostEqual(
            sum(result["breakdown"].values()), result["energy"], places=8
        )
        return result

    def test_default_parameters_are_packaged_outside_hotknots(self):
        expected = {
            "parameters_CC06.txt",
            "parameters_CC09.txt",
            "parameters_DP03.txt",
            "parameters_DP09.txt",
            "pkmodelCC2006.dat",
            "pkmodelRE.dat",
            "turner-tloop-rna.dat",
            "turner_parameters_fm363_constrdangles.txt",
        }
        self.assertEqual(self.evaluator.parameter_dir.name, "energy_params")
        self.assertNotIn("HotKnots_v2.0", str(self.evaluator.parameter_dir))
        self.assertTrue(
            expected.issubset(
                {path.name for path in Path(self.evaluator.parameter_dir).iterdir()}
            )
        )

    def test_pseudoknot_free_reference(self):
        structure = "..(((((......))))).........."
        expected = {
            "DP09": (-5.14, -4.19),
            "CC09": (-5.14, -4.19),
            "RE": (-8.10, -6.60),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                self.assert_energy(REFERENCE_SEQUENCE, structure, model, energy)

    def test_h_type_reference(self):
        structure = "..(((((..[[[[)))))......]]]]"
        expected = {
            "DP09": (-8.0204, -6.9804),
            "CC09": (-8.79466, -7.75466),
            "RE": (-9.102, -7.102),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                self.assert_energy(REFERENCE_SEQUENCE, structure, model, energy)

    def test_internal_loop_inside_h_type_band(self):
        sequence = "GAAGCGAACCACUUUGGAGGGU"
        structure = "(.((...[[[..)))....]]]"
        expected = {
            "DP09": (2.2648, 3.4448),
            "CC09": (8.48822, 9.66822),
            "RE": (5.412, 6.412),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                self.assert_energy(sequence, structure, model, energy)

    def test_cc09_mismatch_coaxial_stack(self):
        sequence = "AGUUGGUAAUUGCGGUUCCCACGAGUU"
        structure = "(((((..[[[[.)))))......]]]]"
        result = self.assert_energy(sequence, structure, "CC09", (1.54534, 2.02534))
        self.assertAlmostEqual(result["breakdown"]["coaxial"], -4.21, places=6)

    def test_outside_secondary_structure_component(self):
        structure = "[[(((((..]]..)))))...(....)."
        expected = {
            "DP09": (-5.1728, -2.2028),
            "CC09": (-5.1728, -2.2028),
            "RE": (-4.784, 0.0160004),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                result = self.assert_energy(REFERENCE_SEQUENCE, structure, model, energy)
                self.assertIn("secondary_structure_outside_pseudoknot", result["breakdown"])

    def test_dp03_and_cc06_are_kept_for_compatibility(self):
        structure = "..(((((..[[[[)))))......]]]]"
        self.assert_energy(REFERENCE_SEQUENCE, structure, "DP03", (-7.302, -5.302))
        self.assert_energy(REFERENCE_SEQUENCE, structure, "CC06", (-13.6949, -11.6949))

    def test_exterior_ggg_hairpin_matches_model_specific_hotknots_context(self):
        sequence = "GGGAAAAU"
        structure = "..(....)"
        self.assert_energy(sequence, structure, "DP09", (3.93, 3.93))
        self.assert_energy(sequence, structure, "CC09", (3.93, 3.93))
        self.assert_energy(sequence, structure, "RE", (3.20, 3.60))

        # DP/CC suppress the upstream-GGG term only for an exterior root.  The
        # original SimFold path still applies it to a nested hairpin.
        nested_sequence = "GGGGGAAAAUAACC"
        nested_structure = "((..(....)..))"
        self.assert_energy(nested_sequence, nested_structure, "DP09", (1.84, 1.84))
        self.assert_energy(nested_sequence, nested_structure, "CC09", (1.84, 1.84))
        self.assert_energy(nested_sequence, nested_structure, "RE", (-0.65, -0.65))

    def test_multiple_independent_h_type_components(self):
        sequence = "GAGAACACAAGAGAACACAA"
        structure = "(.[..).]..(.[..).].."
        expected = {
            "DP09": (5.50, 7.56),
            "CC09": (5.50, 7.56),
            "RE": (12.80, 16.40),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                result = self.assert_energy(sequence, structure, model, energy)
                self.assertEqual(result["metadata"]["pseudoknot_components"], 2)
                self.assertAlmostEqual(
                    sum(result["breakdown"].values()), result["energy"], places=8
                )

        # Exercise two components through CC's entropy/coaxial path as well as
        # the short-stem DP fallback above.
        sequence = REFERENCE_SEQUENCE + "A" + REFERENCE_SEQUENCE
        component = "..(((((..[[[[)))))......]]]]"
        structure = component + "." + component
        expected = {
            "DP09": (-16.5008, -13.9608),
            "CC09": (-18.04931, -15.50931),
            "RE": (-19.304, -14.204),
        }
        for model, energy in expected.items():
            with self.subTest(model=model, path="cc-entropy"):
                result = self.assert_energy(sequence, structure, model, energy)
                self.assertEqual(result["metadata"]["pseudoknot_components"], 2)
        self.assertNotIn(
            "cc_fallback_to_dp",
            self.evaluator.compute_energy(sequence, structure, "CC09")["metadata"],
        )

    def test_cc_fallback_is_metadata_not_an_energy_term(self):
        structure = "[[(((((..]]..)))))...(....)."
        result = self.assert_energy(
            REFERENCE_SEQUENCE, structure, "CC09", (-5.1728, -2.2028)
        )
        self.assertTrue(result["metadata"]["cc_fallback_to_dp"])
        self.assertNotIn("cc_fallback_to_dp", result["breakdown"])
        self.assertAlmostEqual(sum(result["breakdown"].values()), result["energy"], places=8)

    def test_secondary_structure_nested_inside_pseudoknot_loop(self):
        sequence = "CCACAAAGACCAAAGGAAAGG"
        structure = "((.(...).[[...))...]]"
        expected = {
            "DP09": (3.9052, 4.8252),
            "CC09": (3.9052, 4.8252),
            "RE": (7.122, 9.322),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                result = self.assert_energy(sequence, structure, model, energy)
                self.assertIn("secondary_structure", result["breakdown"])

    def test_nested_pseudoknots_inside_band_and_pseudoloop_gap(self):
        fixtures = (
            (
                "inside-band",
                "CACACAGAGACACAAGAAAAGAAAAG",
                "(.(.[.).].(.[..)....)....]",
                {
                    "DP09": (21.91, 23.29),
                    "CC09": (21.91, 23.29),
                    "RE": (21.13, 24.43),
                },
            ),
            (
                "inside-pseudoloop-gap",
                "ACACAAACCGAAAGAGAAG",
                ".(.[...([)...].)..]",
                {
                    "DP09": (23.92, 25.02),
                    "CC09": (23.92, 25.02),
                    "RE": (13.70, 16.90),
                },
            ),
        )
        for name, sequence, structure, expected in fixtures:
            for model, energy in expected.items():
                with self.subTest(topology=name, model=model):
                    result = self.assert_energy(sequence, structure, model, energy)
                    self.assertEqual(result["metadata"]["pseudoknot_components"], 2)
                    self.assertEqual(result["metadata"]["nested_pseudoknots"], 1)
                    self.assertEqual(result["metadata"]["bands_per_pseudoknot"], [2, 2])

    def test_cc_entropy_path_inside_nested_pseudoknot_tree(self):
        sequence = "CCCCCCCAACCCCGGGGGAAAAAAGGGGAAACCAAAGGAAAGG"
        structure = "(({{{{{..<<<<}}}}}......>>>>...[[...))...]]"
        expected = {
            "DP09": (6.5784, 8.1384),
            "CC09": (-8.40946, -6.84946),
            "RE": (-10.651, -6.351),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                result = self.assert_energy(sequence, structure, model, energy)
                self.assertEqual(result["metadata"]["pseudoknot_components"], 2)
                self.assertEqual(result["metadata"]["nested_pseudoknots"], 1)
                if model == "CC09":
                    # The outer short-stem component falls back to DP while
                    # the nested component exercises the CC entropy table.
                    self.assertEqual(result["metadata"]["cc_fallback_components"], 1)
                    self.assertIn("assembly", result["breakdown"])

    def test_cc_entropy_path_nested_in_secondary_multiloop(self):
        sequence = "CACCCCCAACCCCGGGGGAAAAAAGGGGAG"
        structure = "(.{{{{{..<<<<}}}}}......>>>>.)"
        expected = {
            "DP09": (3.7632, 3.8532),
            "CC09": (-6.29466, -6.20466),
            "RE": (-9.073, -8.573),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                result = self.assert_energy(sequence, structure, model, energy)
                if model == "CC09":
                    self.assertNotIn("cc_fallback_to_dp", result["metadata"])
                    self.assertIn("assembly", result["breakdown"])
                if model in ("DP09", "CC09"):
                    self.assertIn("secondary_structure_scaffold", result["breakdown"])

    def test_three_band_kissing_or_chain_pseudoknot(self):
        sequence = "CCCAAAAAACCCAAAGGGAAACCCAAAGGGAAAAAAGGG"
        structure = "(((......[[[...)))...{{{...]]]......}}}"
        expected = {
            "DP03": (-7.334, -4.134),
            "DP09": (-5.3744, -4.2744),
            "CC06": (-7.334, -4.134),
            "CC09": (-5.3744, -4.2744),
            "RE": (-1.834, 1.366),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                result = self.assert_energy(sequence, structure, model, energy)
                self.assertEqual(result["metadata"]["bands_per_pseudoknot"], [3])
                self.assertEqual(result["metadata"]["topology"], ["chain_or_kissing"])

    def test_multiloop_spanning_a_band(self):
        sequence = "CCAAAGCACCAAAGCAACGGGGAAAG"
        structure = "((...)(.((...)(..[))))...]"
        expected = {
            "DP09": (20.331, 20.881),
            "CC09": (20.331, 20.881),
            "RE": (36.075, 37.675),
        }
        for model, energy in expected.items():
            with self.subTest(model=model):
                result = self.assert_energy(sequence, structure, model, energy)
                self.assertIn("band_multiloop", result["breakdown"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluator.compute_energy("AAAAA", "(...)", "DP09")
        with self.assertRaises(ValueError):
            self.evaluator.compute_energy("GAAAC", "((...)", "DP09")


if __name__ == "__main__":
    unittest.main()
