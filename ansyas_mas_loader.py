import json
import os
import time
import sys
import mas_autocomplete

sys.path.append(os.path.dirname(__file__))

from src.ansyas import Ansyas


if __name__ == "__main__":
    non_graphical = False
    new_desktop_session = False
    ansyas = Ansyas(number_segments_arcs=12, initial_mesh_configuration=2, maximum_error_percent=5, refinement_percent=5, scale=1)

    f = open(sys.argv[1])
    mas_dict = json.load(f)

    mas = mas_autocomplete.autocomplete(mas_dict)

    outputs_folder = os.path.dirname(__file__) + "/outputs"

    try:
        project_name = f"{mas_dict['magnetic']['manufacturerInfo']['reference']}_{time.time()}"
    except TypeError:
        project_name = f"Unnamed_design_{time.time()}"

    project = ansyas.create_project(
        outputs_folder=outputs_folder,
        project_name=project_name,
        # specified_version="2023.2",
        non_graphical=non_graphical,
        # solution_type="SteadyState",
        solution_type="EddyCurrent",
        # solution_type="Transient",
        # solution_type="TransientAPhiFormulation",
        # solution_type="Electrostatic",
        new_desktop_session=new_desktop_session
    )
    ansyas.set_units("meter")
    ansyas.create_magnetic_simulation(
        mas=mas,
        simulate=True
    )
