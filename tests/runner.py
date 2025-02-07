import unittest
import os
import json
import glob
import time

import context  # noqa: F401
import mas_autocomplete
from ansyas import Ansyas


class EddyCurrent(unittest.TestCase):
    output_path = f'{os.path.dirname(os.path.abspath(__file__))}/../output/'

    @classmethod
    def setUpClass(cls):

        files = glob.glob(f"{cls.output_path}/*")
        for f in files:
            os.remove(f)
        print("Starting tests for builder")

    @classmethod
    def tearDownClass(cls):
        print("\nFinishing tests for builder")

    def test_simple_inductor_rectangular_column(self):
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/mas_files/simple_inductor_rectangular_column.json', 'r') as f:
            mas_dict = json.load(f)
            mas = mas_autocomplete.autocomplete(mas_dict)

            ansyas = Ansyas(number_segments_arcs=12, initial_mesh_configuration=2, maximum_error_percent=20, refinement_percent=5, maximum_passes=100, scale=1)

            project = ansyas.create_project(
                outputs_folder=self.output_path,
                project_name=f"test_simple_inductor_rectangular_column_{time.time()}",
                non_graphical=False,
                solution_type="EddyCurrent",
                new_desktop_session=False
            )
            
            ansyas.set_units("meter")
            ansyas.create_magnetic(mas=mas)

            expected_magnetizing_inductance = 102e-9

            ansyas.analyze()
            impedance = ansyas.outputs_extractor.get_results()
            magnetizing_inductance = impedance.inductanceMatrix[0].matrix[0][0].nominal
            print(magnetizing_inductance)
            self.assertAlmostEqual(magnetizing_inductance, expected_magnetizing_inductance)
            project.release_desktop(close_projects=False, close_desktop=False)

    def test_simple_inductor_rectangular_column_stacked(self):
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/mas_files/simple_inductor_rectangular_column_stacked.json', 'r') as f:
            mas_dict = json.load(f)
            mas = mas_autocomplete.autocomplete(mas_dict)

            ansyas = Ansyas(number_segments_arcs=12, initial_mesh_configuration=2, maximum_error_percent=20, refinement_percent=5, maximum_passes=100, scale=1)

            project = ansyas.create_project(
                outputs_folder=self.output_path,
                project_name=f"test_simple_inductor_rectangular_column_stacked_{time.time()}",
                non_graphical=False,
                solution_type="EddyCurrent",
                new_desktop_session=False
            )
            
            ansyas.set_units("meter")
            ansyas.create_magnetic(mas=mas)

            expected_magnetizing_inductance = 702e-9

            ansyas.analyze()
            impedance = ansyas.outputs_extractor.get_results()
            magnetizing_inductance = impedance.inductanceMatrix[0].matrix[0][0].nominal
            print(magnetizing_inductance)
            self.assertAlmostEqual(magnetizing_inductance, expected_magnetizing_inductance)
            project.release_desktop(close_projects=False, close_desktop=False)

    def test_simple_inductor_rectangular_column_toroidal(self):
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/mas_files/simple_inductor_rectangular_column_toroidal.json', 'r') as f:
            mas_dict = json.load(f)
            mas = mas_autocomplete.autocomplete(mas_dict)

            ansyas = Ansyas(number_segments_arcs=12, initial_mesh_configuration=2, maximum_error_percent=20, refinement_percent=5, maximum_passes=100, scale=1)

            project = ansyas.create_project(
                outputs_folder=self.output_path,
                project_name=f"test_simple_inductor_rectangular_column_toroidal_{time.time()}",
                non_graphical=False,
                solution_type="EddyCurrent",
                new_desktop_session=False
            )
            
            ansyas.set_units("meter")
            ansyas.create_magnetic(mas=mas)

            expected_magnetizing_inductance = 10.3e-6

            ansyas.analyze()
            impedance = ansyas.outputs_extractor.get_results()
            magnetizing_inductance = impedance.inductanceMatrix[0].matrix[0][0].nominal
            print(magnetizing_inductance)
            self.assertAlmostEqual(magnetizing_inductance, expected_magnetizing_inductance, places=4)
            project.release_desktop(close_projects=False, close_desktop=False)

    def test_simple_inductor_round_column(self):
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/mas_files/simple_inductor_round_column.json', 'r') as f:
            mas_dict = json.load(f)
            mas = mas_autocomplete.autocomplete(mas_dict)

            ansyas = Ansyas(number_segments_arcs=12, initial_mesh_configuration=2, maximum_error_percent=20, refinement_percent=5, maximum_passes=100, scale=1)

            project = ansyas.create_project(
                outputs_folder=self.output_path,
                project_name=f"test_simple_inductor_rectangular_column_{time.time()}",
                non_graphical=False,
                solution_type="EddyCurrent",
                new_desktop_session=False
            )
            
            ansyas.set_units("meter")
            ansyas.create_magnetic(mas=mas)

            expected_magnetizing_inductance = 102e-9

            ansyas.analyze()
            impedance = ansyas.outputs_extractor.get_results()
            magnetizing_inductance = impedance.inductanceMatrix[0].matrix[0][0].nominal
            print(magnetizing_inductance)
            self.assertAlmostEqual(magnetizing_inductance, expected_magnetizing_inductance)
            project.release_desktop(close_projects=False, close_desktop=False)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
