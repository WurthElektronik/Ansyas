from typing import Optional
import pyaedt
import ansyas_utils
import math
import os
from OpenMagneticsVirtualBuilder.builder import Builder as ShapeBuilder  # noqa: E402
import MAS_models as MAS


class Cooling:

    def __init__(self, project, number_segments_arcs=12):
        self.project = project
        self.number_segments_arcs = number_segments_arcs

    def create_cooling(self, core, cooling, material="Al-Extruded"):
        pass
        # if cooling.maximumTemperature is not None:
        #     # Cold plate
        #     dimensions = ansyas_utils.convert_axis(cooling.dimensions)
        #     core_height = core.processedDescription.height
        #     cold_plate = self.project.modeler.create_box(
        #         origin=[-dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] - core_height / 2],
        #         sizes=dimensions,
        #         name="cold_plate",
        #         material=material
        #     )
        #     self.project.assign_source(
        #         assignment=[cold_plate.name],
        #         thermal_condition="Temperature",
        #         assignment_value=f"{cooling.maximumTemperature}cel",
        #         boundary_name="cold_plate"
        #     )
        # else:
        #     raise NotImplementedError("Only cold plate implemented for now")
