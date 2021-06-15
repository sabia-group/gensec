import unittest
from gensec.modules import (
    create_connectivity_matrix,
    detect_rotatble,
    detect_cycles,
    exclude_rotatable_from_cycles,
    measure_quaternion,
    carried_atoms,
    quaternion_set,
    all_right,
    internal_clashes,
)

from ase.io import read, write
import os
import numpy as np
import json
from gensec.structure import Structure, Fixed_frame


class TestStructure(unittest.TestCase):
    def test_read_hexane(self):
        """Import of Hexane molecule in xyz format
        Checking of the identified torsions for the molecule
        """

        list_of_torsions_ref = [[1, 0, 2, 4], [2, 0, 1, 3], [0, 1, 3, 5]]
        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(dirname, "supporting", "molecules", "hexane.xyz"),
            format="xyz",
        )
        connectivity = create_connectivity_matrix(atoms, bothways=False)
        list_of_torsions = detect_rotatble(connectivity, atoms)
        self.assertEqual(list_of_torsions_ref, list_of_torsions)

    def test_find_cycles(self):
        """Import the molecule with cycles in sdf format
        Checking of the identified torsions for the molecule
        """

        list_of_torsions_all_ref = [
            [5, 3, 4, 8],
            [4, 3, 5, 7],
            [2, 4, 8, 0],
            [6, 5, 7, 10],
            [7, 5, 6, 9],
            [5, 6, 9, 11],
            [5, 7, 10, 11],
            [6, 9, 11, 10],
            [7, 10, 11, 9],
        ]

        list_of_torsions_no_cycles_ref = [
            [5, 3, 4, 8],
            [4, 3, 5, 7],
            [2, 4, 8, 0],
        ]

        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(
                dirname, "supporting", "molecules", "phenylalanine.sdf"
            ),
            format="sdf",
        )
        connectivity = create_connectivity_matrix(atoms, bothways=False)
        list_of_torsions = detect_rotatble(connectivity, atoms)
        self.assertEqual(list_of_torsions_all_ref, list_of_torsions)
        cycles = detect_cycles(connectivity)
        torsions_no_cycles = exclude_rotatable_from_cycles(
            list_of_torsions, cycles
        )
        self.assertEqual(list_of_torsions_no_cycles_ref, torsions_no_cycles)

    def test_assign_the_torsions(self):
        """Test of assignment of the torsions

        Detect torsion of the molecule
        Apply different configuration to the molecule
        """

        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(dirname, "supporting", "molecules", "hexane.xyz"),
            format="xyz",
        )
        atoms_ref = read(
            os.path.join(
                dirname,
                "supporting",
                "molecules",
                "hexane_torsions_applied.xyz",
            ),
            format="xyz",
        )
        connectivity = create_connectivity_matrix(atoms, bothways=False)
        list_of_torsions = detect_rotatble(connectivity, atoms)
        angles = [10, 60, 250]
        # Apply one configuration
        for t in range(len(list_of_torsions)):
            fixed_indices = carried_atoms(connectivity, list_of_torsions[t])
            atoms.set_dihedral(
                angle=angles[t],
                a1=list_of_torsions[t][0],
                a2=list_of_torsions[t][1],
                a3=list_of_torsions[t][2],
                a4=list_of_torsions[t][3],
                indices=fixed_indices,
            )

        coords_ref = atoms_ref.get_positions()
        coords = atoms.get_positions()
        np.testing.assert_array_almost_equal(coords_ref, coords)

    def test_find_the_orientation(self):
        """Identify the orientation of the molecule

        Calculate the orientation of the molecule.
        """
        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(
                dirname, "supporting", "molecules", "phenylalanine.sdf"
            ),
            format="sdf",
        )

        orientation_ref = [
            -179.844909,
            0.00599313651,
            -0.0211136661,
            0.999759119,
        ]
        orientation = measure_quaternion(atoms, 0, len(atoms) - 1)
        np.testing.assert_array_almost_equal(
            np.array(orientation_ref), np.array(list(orientation)), decimal=6
        )

    def test_assign_the_orientation(self):
        """Assign orientation to the molecule

        Compare orientation with other molecule, for which is
        the orientation was preset.
        """
        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(
                dirname, "supporting", "molecules", "phenylalanine.sdf"
            ),
            format="sdf",
        )
        atoms_ref = read(
            os.path.join(
                dirname,
                "supporting",
                "molecules",
                "phenylalanine_orientation_applied.xyz",
            ),
            format="xyz",
        )

        orientation = [0, 0, 0, 1]
        quaternion_set(atoms, orientation, 0, len(atoms) - 1)
        coords_ref = atoms_ref.get_positions()
        coords = atoms.get_positions()
        np.testing.assert_array_almost_equal(coords, coords_ref)

    def test_find_center_of_mass(self):
        """Identify the orientation of the molecule

        Calculate the orientation of the molecule.
        """
        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(
                dirname, "supporting", "molecules", "phenylalanine.sdf"
            ),
            format="sdf",
        )

        com_ref = [-0.136086, -0.016372, -0.010462]
        com = atoms.get_center_of_mass()
        np.testing.assert_array_almost_equal(com_ref, com)

    def test_set_center_of_mass(self):
        """Assign center of mass to the molecule

        Compare com with other molecule, for which is
        the com was preset.
        """
        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(
                dirname, "supporting", "molecules", "phenylalanine.sdf"
            ),
            format="sdf",
        )
        atoms_ref = read(
            os.path.join(
                dirname,
                "supporting",
                "molecules",
                "phenylalanine_com_applied.xyz",
            ),
            format="xyz",
        )

        com = [0, 0, 0]
        atoms.set_center_of_mass(com)
        coords_ref = atoms_ref.get_positions()
        coords = atoms.get_positions()
        np.testing.assert_array_almost_equal(coords, coords_ref)

    def test_for_internal_clashes(self):
        """Test different conformations

        Apply different conformations and check
        if they have internal clashes or not.
        """

        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(
            os.path.join(dirname, "supporting", "molecules", "hexane.xyz"),
            format="xyz",
        )

        with open(os.path.join(dirname, "parameters_generate.json")) as f:
            parameters = json.load(f)
        parameters["geometry"][0] = os.path.join(
            dirname, "supporting", "molecules", "hexane.xyz"
        )
        parameters["geometry"][1] = "xyz"

        parameters["fixed_frame"]["activate"] = False
        parameters["fixed_frame"]["filename"] = os.path.join(
            dirname, "supporting", "surface", "Rh.in"
        )
        parameters["fixed_frame"]["format"] = "aims"
        structure = Structure(parameters)
        fixed_frame = Fixed_frame(parameters)

        configs = [
            {
                "m0t0": 201,
                "m0t1": 325,
                "m0t2": 274,
                "m0q0": 39.0,
                "m0q1": 0.035786671430549165,
                "m0q2": 0.9542295189830069,
                "m0q3": 0.2969264879551523,
                "m0c0": 8.222222222222221,
                "m0c1": 5.555555555555555,
                "m0c2": 44.44444444444444,
            },
            {
                "m0t0": 219,
                "m0t1": 287,
                "m0t2": 20,
                "m0q0": 163.0,
                "m0q1": 0.6651765708470119,
                "m0q2": 0.5916682332808936,
                "m0q3": 0.45548746560413317,
                "m0c0": 5.555555555555555,
                "m0c1": 9.11111111111111,
                "m0c2": 40.0,
            },
            {
                "m0t0": 63,
                "m0t1": 214,
                "m0t2": 124,
                "m0q0": 268.0,
                "m0q1": 0.5702903019832661,
                "m0q2": 0.6919576355504017,
                "m0q3": 0.44267776324018415,
                "m0c0": 3.7777777777777777,
                "m0c1": 8.222222222222221,
                "m0c2": 42.22222222222222,
            },
            {
                "m0t0": 350,
                "m0t1": 271,
                "m0t2": 150,
                "m0q0": 352.0,
                "m0q1": 0.9164925540916642,
                "m0q2": 0.2644159240879181,
                "m0q3": 0.3002092893020973,
                "m0c0": 10.0,
                "m0c1": 9.11111111111111,
                "m0c2": 45.0,
            },
            {
                "m0t0": 9,
                "m0t1": 52,
                "m0t2": 280,
                "m0q0": 103.0,
                "m0q1": 0.6465420456505039,
                "m0q2": 0.4537166791461405,
                "m0q3": 0.613289946330983,
                "m0c0": 4.666666666666666,
                "m0c1": 2.888888888888889,
                "m0c2": 44.44444444444444,
            },
            {
                "m0t0": 315,
                "m0t1": 43,
                "m0t2": 291,
                "m0q0": 190.0,
                "m0q1": 0.6549387415883375,
                "m0q2": 0.42436960443964467,
                "m0q3": 0.625272487476,
                "m0c0": 3.7777777777777777,
                "m0c1": 6.444444444444445,
                "m0c2": 40.0,
            },
            {
                "m0t0": 208,
                "m0t1": 144,
                "m0t2": 194,
                "m0q0": 231.0,
                "m0q1": 0.46088090726971404,
                "m0q2": 0.06512217694908007,
                "m0q3": 0.8850694274369992,
                "m0c0": 5.555555555555555,
                "m0c1": 5.555555555555555,
                "m0c2": 40.0,
            },
            {
                "m0t0": 125,
                "m0t1": 17,
                "m0t2": 186,
                "m0q0": 327.0,
                "m0q1": 0.5577774326124594,
                "m0q2": 0.6344697919232174,
                "m0q3": 0.5351003819893634,
                "m0c0": 3.7777777777777777,
                "m0c1": 2.0,
                "m0c2": 41.111111111111114,
            },
            {
                "m0t0": 22,
                "m0t1": 127,
                "m0t2": 194,
                "m0q0": 209.0,
                "m0q1": 0.030319548354817788,
                "m0q2": 0.8496698445978416,
                "m0q3": 0.5264426656043749,
                "m0c0": 7.333333333333333,
                "m0c1": 9.11111111111111,
                "m0c2": 42.77777777777778,
            },
            {
                "m0t0": 73,
                "m0t1": 38,
                "m0t2": 49,
                "m0q0": 319.0,
                "m0q1": 0.7775785882885461,
                "m0q2": 0.6055896662517776,
                "m0q3": 0.16921198292157763,
                "m0c0": 7.333333333333333,
                "m0c1": 9.11111111111111,
                "m0c2": 43.888888888888886,
            },
            {
                "m0t0": 322,
                "m0t1": 318,
                "m0t2": 34,
                "m0q0": 167.0,
                "m0q1": 0.6857361347527214,
                "m0q2": 0.11271595694555324,
                "m0q3": 0.7190695839376364,
                "m0c0": 3.7777777777777777,
                "m0c1": 6.444444444444445,
                "m0c2": 43.888888888888886,
            },
            {
                "m0t0": 95,
                "m0t1": 153,
                "m0t2": 250,
                "m0q0": 71.0,
                "m0q1": 0.43085826315894077,
                "m0q2": 0.8108275579301744,
                "m0q3": 0.3961310747324057,
                "m0c0": 10.0,
                "m0c1": 2.0,
                "m0c2": 44.44444444444444,
            },
            {
                "m0t0": 39,
                "m0t1": 325,
                "m0t2": 259,
                "m0q0": 271.0,
                "m0q1": 0.194156883279832,
                "m0q2": 0.7484700825955319,
                "m0q3": 0.6341101167261874,
                "m0c0": 2.888888888888889,
                "m0c1": 7.333333333333333,
                "m0c2": 42.22222222222222,
            },
            {
                "m0t0": 135,
                "m0t1": 71,
                "m0t2": 345,
                "m0q0": 276.0,
                "m0q1": 0.9935466098257036,
                "m0q2": 0.10488267650406222,
                "m0q3": 0.04318284697880838,
                "m0c0": 10.0,
                "m0c1": 10.0,
                "m0c2": 41.111111111111114,
            },
            {
                "m0t0": 192,
                "m0t1": 21,
                "m0t2": 192,
                "m0q0": 127.0,
                "m0q1": 0.39085730213628483,
                "m0q2": 0.9103924892753021,
                "m0q3": 0.13570587620977934,
                "m0c0": 5.555555555555555,
                "m0c1": 3.7777777777777777,
                "m0c2": 41.111111111111114,
            },
            {
                "m0t0": 96,
                "m0t1": 200,
                "m0t2": 272,
                "m0q0": 26.0,
                "m0q1": 0.9375427206582388,
                "m0q2": 0.0979847446744857,
                "m0q3": 0.33378531536276906,
                "m0c0": 7.333333333333333,
                "m0c1": 2.0,
                "m0c2": 43.333333333333336,
            },
            {
                "m0t0": 30,
                "m0t1": 286,
                "m0t2": 57,
                "m0q0": 122.0,
                "m0q1": 0.18443343039631432,
                "m0q2": 0.8513375106996769,
                "m0q3": 0.4911300770955953,
                "m0c0": 8.222222222222221,
                "m0c1": 2.0,
                "m0c2": 41.666666666666664,
            },
        ]

        results_ref = [
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
        ]

        results = []
        for conf in configs:
            structure.apply_conf(conf)
            results.append(not internal_clashes(structure))

        self.assertEqual(results_ref, results)


if __name__ == "__main__":
    unittest.main()
