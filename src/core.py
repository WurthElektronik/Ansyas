from typing import Optional
import pyaedt
import ansyas_utils
import math
import os
from OpenMagneticsVirtualBuilder.builder import Builder as ShapeBuilder  # noqa: E402
import MAS_models as MAS


class Core:

    def __init__(self, project, number_segments_arcs=12):
        self.project = project
        self.number_segments_arcs = number_segments_arcs

    def load_material(self, material: MAS.CoreMaterial, frequency, temperature=25, magneticFieldDcBias=0):
        # material.name = material.name.replace(' ', '')
        aedt_material = self.project.materials.add_material(material.name)

        if self.project.solution_type not in ["TransientAPhiFormulation"] and "complex" in dir(material.permeability) and "imaginary" in dir(material.permeability.complex) and "real" in dir(material.permeability.complex):
            if isinstance(material.permeability.complex.imaginary, list) and len(material.permeability.complex.imaginary) > 1:
                frequency_values = []
                complex_imaginary_permeability_values = []
                for element in material.permeability.complex.imaginary:
                    frequency_values.append(element.frequency)
                    complex_imaginary_permeability_values.append(element.value)

                dataset = self.project.create_dataset(
                    name=f"material_{material.name}_complexImaginaryPermeability",
                    x=frequency_values,
                    y=complex_imaginary_permeability_values,
                    x_unit="Hz",
                    y_unit="",
                )
            else:
                assert 0, "Complex imag Perm"
            if isinstance(material.permeability.complex.real, list) and len(material.permeability.complex.real) > 1:
                frequency_values = []
                complex_real_permeability_values = []
                for element in material.permeability.complex.real:
                    frequency_values.append(element.frequency)
                    complex_real_permeability_values.append(element.value)

                dataset = self.project.create_dataset(
                    name=f"material_{material.name}_complexRealPermeability",
                    x=frequency_values,
                    y=complex_real_permeability_values,
                    x_unit="Hz",
                    y_unit="",
                )

            else:
                assert 0, "Complex imag Perm"

            aedt_material.permeability = f"sqrt(pow(pwl($material_{material.name}_complexRealPermeability, Freq), 2) + pow(pwl($material_{material.name}_complexImaginaryPermeability, Freq), 2))"
            aedt_material.permeability = f"pwl($material_{material.name}_complexRealPermeability, Freq)"

        else:
            if isinstance(material.permeability.initial, list) and len(material.permeability.initial) > 1:
                temperature_values = []
                initial_permeability_values = []
                for element in material.permeability.initial:
                    temperature_values.append(element.temperature)
                    initial_permeability_values.append(element.value)

                dataset = self.project.create_dataset(
                    name=f"material_{material.name}_initialPermeability",
                    x=temperature_values,
                    y=initial_permeability_values
                )
                aedt_material.permeability.add_thermal_modifier_dataset(dataset.name)
            elif isinstance(material.permeability.initial, list) and len(material.permeability.initial) == 1:
                conductivity = material.permeability.initial[0].value
                aedt_material.conductivity = conductivity
            else:
                aedt_material.permeability = material.permeability.initial.value
                assert 0, "Perm"

        if isinstance(material.resistivity, list) and len(material.resistivity) > 1:
            temperature_values = []
            conductivity_values = []
            for element in material.resistivity:
                temperature_values.append(element.temperature)
                conductivity_values.append(1.0 / element.value)

            dataset = self.project.create_dataset(
                name=f"material_{material.name}_conductivity",
                x=temperature_values,
                y=conductivity_values
            )
            aedt_material.conductivity.add_thermal_modifier_dataset(dataset.name)
        elif isinstance(material.resistivity, list) and len(material.resistivity) == 1:
            conductivity = 1.0 / material.resistivity[0].value
            aedt_material.conductivity = conductivity
        else:
            assert 0, "Unknown resistivity data type"

        for methodData in material.volumetricLosses["default"]:
            if methodData.method == MAS.CoreLossesMethodType.lossFactor:
                closest_temperature = math.inf
                target_temperature = 25
                initial_permeability = None
                for point in material.permeability.initial:
                    if (point.temperature - target_temperature) < (closest_temperature - target_temperature):
                        closest_temperature = point.temperature
                        initial_permeability = point.value

                frequency_values = []
                loss_factor_values = []
                for element in methodData.factors:
                    frequency_values.append(element.frequency)
                    loss_factor_values.append(element.value * initial_permeability)

                dataset = self.project.create_dataset(
                    name=f"material_{material.name}_lossFactor",
                    x=frequency_values,
                    y=loss_factor_values
                )
                aedt_material.magnetic_loss_tangent = f"pwl($material_{material.name}_lossFactor, Freq)"
                break

            if methodData.method == MAS.CoreLossesMethodType.steinmetz:
                for steinmetz_range in methodData.ranges:
                    if steinmetz_range.minimumFrequency <= frequency <= steinmetz_range.maximumFrequency:
                        steinmetz_coefficients = steinmetz_range.to_dict()
                        break
                aedt_material.set_power_ferrite_coreloss(
                    cm=steinmetz_coefficients["k"],
                    x=steinmetz_coefficients["alpha"],
                    y=steinmetz_coefficients["beta"]
                )
        aedt_material.mass_density = material.density

    def import_core(self, step_path: str = None, core: Optional[MAS.MagneticCore] = None, operating_point: Optional[MAS.OperatingPoint] = None, name="core"):

        temperature = operating_point.conditions.ambientTemperature
        magneticFieldDcBias = 0
        frequency = operating_point.excitationsPerWinding[0].frequency

        self.load_material(
            material=core.functionalDescription.material,
            frequency=frequency, 
            temperature=temperature, 
            magneticFieldDcBias=magneticFieldDcBias
        )

        if step_path is None and core is not None:
            step_path, obj_path = ShapeBuilder().get_core(project_name=f"core_{core.name}",
                                                          geometrical_description=[x.to_dict() for x in core.geometricalDescription],
                                                          output_path=os.path.abspath(os.path.dirname(__file__) + "/outputs/"))

        self.project.modeler.import_3d_cad(
            input_file=step_path.replace("/", os.sep),
            healing=True
        )

        core_parts = [self.project.modeler.get_object_from_name(x) for x in self.project.modeler.get_objects_w_string("core_part")]
        if len(core_parts) == 0:
            core_parts = [self.project.modeler.get_object_from_name(x) for x in self.project.modeler.get_objects_w_string("Piece")]
        if len(core_parts) == 0:
            core_parts = [self.project.modeler.get_object_from_name(x) for x in self.project.modeler.get_objects_w_string("STEP")]
        if len(core_parts) == 0:
            core_parts = [self.project.modeler.get_object_from_name(x) for x in self.project.modeler.get_objects_w_string("core")]
        if len(core_parts) == 0:
            assert 0, "Core not found or not imported"

        for core_part_index, core_part in enumerate(core_parts):
            self.project.assign_material(
                assignment=core_part,
                material=core.functionalDescription.material.name
            )
            core_part.color = ansyas_utils.color_ferrite
            if name is not None:
                if len(core_parts) == 1:
                    core_part.name = name
                else:
                    core_part.name = f"{name}_{core_part_index}"

        if self.project.solution_type in ["EddyCurrent", "Transient"]:
            self.project.set_core_losses(
                assignment=[x.name for x in core_parts],
                core_loss_on_field=True
            )
        if self.project.solution_type == "Electrostatic":

            self.project.assign_floating(
                assignment=[x.name for x in core_parts],
                charge_value=0,
                name="FloatingCore"
            )
        if self.project.solution_type == "SteadyState":

            self.project.assign_surface_material(
                obj=[x.name for x in core_parts],
                mat="Steel-oxidised-surface"
            )

        if core.functionalDescription.type == MAS.CoreType.toroidal:
            for core_part in core_parts:
                core_part.rotate(pyaedt.constants.AXIS.X, 90)
        else:
            for core_part in core_parts:
                core_part.rotate(pyaedt.constants.AXIS.Z, 90)

        return core_parts

    def assign_core_losses_as_heat_source(self, core_parts, total_core_losses):
        total_volume = sum([x.volume for x in core_parts])
        for part_index, part in enumerate(core_parts):
            part_losses = total_core_losses * core_parts[0].volume / total_volume

            self.project.assign_source(
                assignment=[part.name],
                thermal_condition="Total Power",
                assignment_value=f"{part_losses}W",
                boundary_name=f"{part.name}_losses"
            )
