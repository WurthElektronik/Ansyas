import typing
import ansyas_utils
import math
import time
import os
import MAS_models as MAS


class Excitation:

    def __init__(self, project, number_segments_arcs=12):
        self.number_segments_arcs = number_segments_arcs
        self.project = project

    def create_matrix(self, excitations: typing.List[str], name):
        matrix = self.project.assign_matrix(assignment=excitations, matrix_name=name)
        return matrix

    def add_excitation(self, coil: MAS.Coil, turns_and_terminals, operating_point: MAS.OperatingPoint = None, primary_current: float = None):

        parallel_per_winding = {}
        for winding_index, _ in enumerate(coil.functionalDescription):
            coil.functionalDescription[winding_index].name = coil.functionalDescription[winding_index].name.replace(" ", "_")

        if self.project.solution_type in ["Transient", "TransientAPhiFormulation"]:
            if operating_point is None:
                raise AttributeError("Operating point is missing")
            for winding_index, winding_data in enumerate(coil.functionalDescription):
                excitation = operating_point.excitationsPerWinding[winding_index]
                current = excitation.current
                voltage = excitation.voltage
                if current.waveform is None or current.waveform.data is None or current.waveform.time is None:
                    raise AttributeError("Current waveform is missing")
                if voltage.waveform is None or voltage.waveform.data is None or voltage.waveform.time is None:
                    raise AttributeError("Current waveform is missing")

                data = []
                for datum in current.waveform.data:
                    data.append(datum / winding_data.numberParallels)

                self.project.create_dataset(
                    name=f"winding_{winding_index}_current",
                    x=current.waveform.time,
                    y=data,
                    is_project_dataset=False
                )
                self.project.create_dataset(
                    name=f"winding_{winding_index}_voltage",
                    x=voltage.waveform.time,
                    y=voltage.waveform.data,
                    is_project_dataset=False
                )

            terminals_per_parallel = {}
            for winding_index, winding_data in enumerate(coil.functionalDescription):
                current_per_parallel = f"pwl_periodic(winding_{winding_index}_current, Time)"
                voltage = f"pwl_periodic(winding_{winding_index}_voltage, Time)"
                current_rms_per_parallel = operating_point.excitationsPerWinding[winding_index].current.processed.rms / coil.functionalDescription[winding_index].numberParallels
                voltage_rms = operating_point.excitationsPerWinding[winding_index].voltage.processed.rms

                parallel_per_winding[winding_data.name.title()] = []
                for parallel_index in range(0, winding_data.numberParallels):

                    winding_this_parallel = self.create_winding(
                        amplitude=current_per_parallel if len(coil.functionalDescription) == 1 else voltage,
                        winding_type="Current" if len(coil.functionalDescription) == 1 else "Voltage",
                        resistance=0 if winding_index == 0 else voltage_rms / current_rms_per_parallel,
                        name=f"{winding_data.name.title()}_winding_parallel_{parallel_index}",
                        is_solid=coil.functionalDescription[winding_index].wire.type != MAS.WireType.litz
                    )

                    terminals_per_parallel[winding_this_parallel.name] = []
                    parallel_per_winding[winding_data.name.title()].append(winding_this_parallel)

            for turn_index, turn_data in enumerate(coil.turnsDescription):
                terminals_per_parallel[parallel_per_winding[ansyas_utils.clean_name(turn_data.winding)][turn_data.parallel].name].append(turns_and_terminals[turn_index][1])
                self.project.add_winding_coils(
                    assignment=parallel_per_winding[ansyas_utils.clean_name(turn_data.winding)][turn_data.parallel].name,
                    coils=turns_and_terminals[turn_index][1]
                )

            if self.project.solution_type in ["TransientAPhiFormulation"]:

                name = os.path.dirname(__file__) + f"/outputs/temp_{time.time()}.aedt"
                old_name = self.project.project_name
                old_project_path = self.project.project_path
                self.project.save_project(file_name=name)

                indexes = []
                f = open(name, "r", encoding="latin-1")
                for line in f:
                    if "CoilOrderList" in line:
                        index = int(line.split(": ")[1].split("]")[0])
                        indexes.append(index)
                indexes.append(len(coil.turnsDescription))
                current_coil_order_index = 1

                f = open(name, "r", encoding="latin-1")
                new_f = ""
                for line in f:
                    if "CoilOrderList" in line:
                        new_line = f"\t\t\t\t\tCoilOrderList[{indexes[current_coil_order_index] - indexes[current_coil_order_index - 1]}: "
                        for x in range(indexes[current_coil_order_index - 1], indexes[current_coil_order_index]):
                            if x == indexes[current_coil_order_index] - 1:
                                new_line += f"{x}"
                            else:
                                new_line += f"{x}, "
                        new_line += "]"

                        new_f += new_line + "\n"
                        current_coil_order_index += 1
                    else:
                        new_f += line

                f = open(name, "w", encoding="latin-1")
                f.write(new_f)
                f.close()

                self.project.close_project(save=False)
                self.project.load_project(
                    file_name=name,
                    close_active=True,
                    set_active=True,
                )

                self.project.save_project(file_name=old_project_path + old_name + ".aedt")
                self.project.close_project(name=name, save=False)

                for winding_name, parallels_in_this_winding in parallel_per_winding.items():
                    if len(parallels_in_this_winding) == 1:
                        parallels_in_this_winding[0].name = winding_name

                all_parallel_excitations = []
                for winding_name, parallels_in_this_winding in parallel_per_winding.items():
                    all_parallel_excitations.extend([x.name for x in parallels_in_this_winding])
                matrix = self.create_matrix(all_parallel_excitations, "solution_matrix")

                for winding_name, parallels_in_this_winding in parallel_per_winding.items():
                    if len(parallels_in_this_winding) > 1:
                        matrix.join_parallel(
                            sources=[x.name for x in parallels_in_this_winding],
                            matrix_name="windings",
                            join_name=winding_name
                        )
                    else:
                        parallels_in_this_winding[0].name = winding_name

        elif self.project.solution_type == "EddyCurrent":
            if operating_point is not None:
                magnetizing_current_peak = operating_point.excitationsPerWinding[0].magnetizingCurrent.processed.rms * math.sqrt(2)

            for winding_index, winding_data in enumerate(coil.functionalDescription):
                current_per_parallel = operating_point.excitationsPerWinding[winding_index].current.processed.rms * math.sqrt(2) / coil.functionalDescription[winding_index].numberParallels
                voltage = operating_point.excitationsPerWinding[winding_index].voltage.processed.rms * math.sqrt(2)
                current_rms_per_parallel = operating_point.excitationsPerWinding[winding_index].current.processed.rms / coil.functionalDescription[winding_index].numberParallels
                voltage_rms = operating_point.excitationsPerWinding[winding_index].voltage.processed.rms
                if winding_index == 0:
                    current_per_parallel += magnetizing_current_peak
                else:
                    current_per_parallel = -current_per_parallel

                parallel_per_winding[winding_data.name.title()] = []
                for parallel_index in range(0, winding_data.numberParallels):
                    winding_this_parallel = self.create_winding(
                        amplitude=current_per_parallel if len(coil.functionalDescription) == 1 else voltage,
                        winding_type="Current" if len(coil.functionalDescription) == 1 else "Voltage",
                        resistance=0 if winding_index == 0 else voltage_rms / current_rms_per_parallel,
                        name=f"{winding_data.name.title()}_winding_parallel_{parallel_index}",
                        is_solid=coil.functionalDescription[winding_index].wire.type != MAS.WireType.litz
                    )
                    parallel_per_winding[winding_data.name.title()].append(winding_this_parallel)

            for turn_index, turn_data in enumerate(coil.turnsDescription):
                self.project.add_winding_coils(
                    assignment=parallel_per_winding[ansyas_utils.clean_name(turn_data.winding)][turn_data.parallel].name,
                    coils=turns_and_terminals[turn_index][1]
                )

            all_parallel_excitations = []
            for winding_name, parallels_in_this_winding in parallel_per_winding.items():
                all_parallel_excitations.extend([x.name for x in parallels_in_this_winding])
            matrix = self.create_matrix(all_parallel_excitations, "solution_matrix")

            for winding_name, parallels_in_this_winding in parallel_per_winding.items():
                if len(parallels_in_this_winding) > 1:
                    matrix.join_parallel(
                        sources=[x.name for x in parallels_in_this_winding],
                        matrix_name="windings",
                        join_name=winding_name
                    )
                else:
                    parallels_in_this_winding[0].name = winding_name

    def create_winding(self, amplitude=0, winding_type="current", name="Winding", is_solid=True, resistance=0):
        winding = self.project.assign_winding(
            winding_type=winding_type.title(),
            is_solid=is_solid,
            current=amplitude,
            resistance=resistance,
            inductance=0,
            voltage=amplitude,
            parallel_branches=1,
            phase=0,
            name=name,
        )
        return winding
