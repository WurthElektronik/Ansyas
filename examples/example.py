import json
import os
import time
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from ansyas import Ansyas

if __name__ == "__main__":
    non_graphical = False
    new_desktop_session = False
    ansyas = Ansyas(number_segments_arcs=12, initial_mesh_configuration=8, maximum_error_percent=5, refinement_percent=15, scale=1)

    # f = open(os.path.dirname(__file__) + "/example.mas.json")
    # f = open(os.path.dirname(__file__) + "/example_basic.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_round_column_rectangular_wire.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_rectangular_wire.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_litz_round_column.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_round_column.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_litz.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_stacked.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_2_winding_transformer.json")
    # f = open(os.path.dirname(__file__) + "/example_basic_3_winding_transformer.json")
    # f = open(os.path.dirname(__file__) + "/example_filter.json")
    f = open(os.path.dirname(__file__) + "/../examples/example_cmc.json")
    # f = open(os.path.dirname(__file__) + "./../tests/Test_Flyback_Simulation.mas.json")
    # f = open(os.path.dirname(__file__) + "./../tests/Test_Flyback_Simulation_Csv.mas.json")

    mas_dict = json.load(f)

    outputs_folder = os.path.dirname(__file__) + "/../outputs"
    try:
        project_name = f"{mas_dict['magnetic']['manufacturerInfo']['reference']}_{time.time()}"
    except TypeError:
        project_name = f"Unnamed_design_{time.time()}"


    project = ansyas.create_project(
        outputs_folder=outputs_folder,
        project_name=project_name,
        non_graphical=non_graphical,
        solution_type="EddyCurrent",
        # solution_type="Transient",
        # solution_type="Electrostatic",
        new_desktop_session=new_desktop_session
    )
    ansyas.set_units("meter")
    ansyas.set_coordinate_system()
    ansyas.set_working_coordinate_system("Global")
    ansyas.create_magnetic(
        magnetic=mas_dict['magnetic'],
        inputs=mas_dict['inputs'],
    )
