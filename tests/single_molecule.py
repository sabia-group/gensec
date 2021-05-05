import unittest
from gensec.modules import create_connectivity_matrix, detect_rotatble

from ase.io import read
import os


class TestStructure(unittest.TestCase):
    def test_read_hexane(self):
        """Import of Hexane molecule in xyz format
        Checking of the identified torsions for the molecule
        """

        list_of_torsions_ref = [[1, 0, 2, 4], [2, 0, 1, 3], [0, 1, 3, 5]]
        dirname, filename = os.path.split(os.path.abspath(__file__))
        atoms = read(os.path.join(dirname, "supporting", "molecules", "hexane.xyz"), format="xyz")
        connectivity = create_connectivity_matrix(atoms, bothways=False)
        list_of_torsions = detect_rotatble(connectivity, atoms)
        self.assertEqual(list_of_torsions_ref, list_of_torsions)


if __name__ == '__main__':
    unittest.main()
