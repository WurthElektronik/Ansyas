import pyaedt
import math
import ansyas_utils
from pyaedt.modeler.cad.object3d import Object3d
import hashlib
import MAS_models as MAS


class Coil:

    def __init__(self, project, number_segments_arcs=12, add_insulation=False):
        self.number_segments_arcs = number_segments_arcs
        self.number_segments_arcs_corners = int(number_segments_arcs / 4)
        self.project = project
        self.add_insulation = add_insulation
        self.terminals_plane = pyaedt.constants.PLANE.YZ

    def add_coil_terminals_to_turn(self, turn_object: Object3d, is_input: bool = True, swap_direction: bool = False):
        turn_object.section(
            plane=self.terminals_plane,
            create_new=True
        )

        conducting_section_names = self.project.modeler.get_objects_w_string(f"{turn_object.name}_Section")
        if len(conducting_section_names) != 1:
            raise AttributeError("Conducting section not found")

        conducting_section = self.project.modeler.get_object_from_name(conducting_section_names[0])

        sections = self.project.modeler.separate_bodies(conducting_section)
        conducting_section = sections[0]
        self.project.modeler.delete(sections[1])

        terminal = self.project.assign_coil(
            assignment=conducting_section,
            conductors_number=1,
            polarity="Negative" if swap_direction else "Positive",
            name=f"{turn_object.name}_terminal"
        )

        return terminal.name

    def get_turn_terminal(self, turn_object: Object3d):
        turn_object.section(
            plane=self.terminals_plane,
            create_new=True
        )

        conducting_section_names = self.project.modeler.get_objects_w_string(f"{turn_object.name}_Section")
        if len(conducting_section_names) != 1:
            conducting_section_names = self.project.modeler.get_objects_w_string(f"{turn_object.name}_Section1")
        if len(conducting_section_names) != 1:
            conducting_section_names = self.project.modeler.get_objects_w_string(f"{turn_object.name}_Section2")
        if len(conducting_section_names) != 1:
            raise AttributeError("Conducting section not found")

        conducting_section = self.project.modeler.get_object_from_name(conducting_section_names[0])

        sections = self.project.modeler.separate_bodies(conducting_section)
        conducting_section = sections[0]
        self.project.modeler.delete(sections[1])

        return conducting_section

    def load_litz_wire(self, wire: MAS.Wire):

        if wire.name is None and wire.type is MAS.WireType.litz:
            wire.name = f"Litz {int(wire.numberConductors)}x{ansyas_utils.resolve_dimensional_values(wire.strand.conductingDiameter) * 1000}"

        wire_material = self.project.materials.duplicate_material(
            material="copper",
            name=wire.name
        )
        wire_material.stacking_type = "Litz Wire"
        wire_material.wire_type = "Round"
        wire_material.strand_number = wire.numberConductors
        wire_material.wire_diameter = f"{ansyas_utils.resolve_dimensional_values(wire.strand.conductingDiameter)}meter"

    def get_wire_material(self, wire, is_insulation):
        if is_insulation:
            wire_material = "Polyurethane 155"
        elif wire.type is MAS.WireType.litz:
            self.load_litz_wire(wire)
            wire_material = wire.name
        else:
            wire_material = wire.material
        return wire_material

    def get_wire_object_radius(self, wire, is_insulation):
        if wire.type is MAS.WireType.round or wire.type is MAS.WireType.litz:
            if wire.type is MAS.WireType.round:
                wire_minus_insulation_radius = ansyas_utils.resolve_dimensional_values(wire.conductingDiameter) / 2
                outer_radius = ansyas_utils.resolve_dimensional_values(wire.outerDiameter) / 2
            elif wire.type is MAS.WireType.litz:
                wire_minus_insulation_radius = ansyas_utils.resolve_dimensional_values(wire.outerDiameter) / 2
                if wire.coating is not None and wire.coating.thickness is not None:
                    wire_minus_insulation_radius -= ansyas_utils.resolve_dimensional_values(wire.coating.thickness)
                if wire.coating is not None and wire.coating.numberLayers is not None and wire.coating.thicknessLayers is not None:
                    wire_minus_insulation_radius -= wire.coating.numberLayers * wire.coating.thicknessLayers
                if wire.coating.type is MAS.InsulationWireCoatingType.served:
                    wire_minus_insulation_radius -= 0.000035
                outer_radius = ansyas_utils.resolve_dimensional_values(wire.outerDiameter) / 2

            if is_insulation and outer_radius <= wire_minus_insulation_radius:
                return None

            if is_insulation:
                return outer_radius
            else:
                return wire_minus_insulation_radius
        else:
            if wire.type is MAS.WireType.rectangular or wire.type is MAS.WireType.foil or wire.type is MAS.WireType.planar:
                wire_minus_insulation_width = ansyas_utils.resolve_dimensional_values(wire.conductingWidth)
                wire_minus_insulation_height = ansyas_utils.resolve_dimensional_values(wire.conductingHeight)
                outer_width = ansyas_utils.resolve_dimensional_values(wire.outerWidth)
                outer_height = ansyas_utils.resolve_dimensional_values(wire.outerHeight)
            else:
                raise RuntimeError("Only rectangular and foil wire should use this method")

            if is_insulation and (outer_width <= wire_minus_insulation_width or outer_height <= wire_minus_insulation_height):
                return None

            if is_insulation:
                return outer_width, outer_height
            else:
                return wire_minus_insulation_width, wire_minus_insulation_height

    def load_insulation_material(self, wire: MAS.Wire):
        if wire.coating.type == MAS.InsulationWireCoatingType.served:
            return 

        material = wire.coating.material
        aedt_material = self.project.materials.add_material(material.name)
        aedt_material.permittivity = material.relativePermittivity

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
            raise AttributeError(f"Material {material.name} is missing resistivity")

    def assign_turn_losses_as_heat_source(self, turns, windingLossesPerTurn):
        for turn_index, turn in enumerate(turns):
            windingLossesThisTurn = windingLossesPerTurn[turn_index]
            turn_losses = windingLossesThisTurn.ohmicLosses.losses + sum(windingLossesThisTurn.skinEffectLosses.lossesPerHarmonic) + sum(windingLossesThisTurn.proximityEffectLosses.lossesPerHarmonic)

            self.project.assign_source(
                assignment=[turn.name],
                thermal_condition="Total Power",
                assignment_value=f"{turn_losses}W",
                boundary_name=f"{turn.name}_losses"
            )


class ConcentricCoil(Coil):
    def create_coil(self, coil: MAS.Coil):
        turns_and_terminals = []

        prev_turn_data_hash = -1
        prev_turn = None
        prev_turn_coordinates = None
        prev_turn_insulation = None
        for winding_index, winding_data in enumerate(coil.functionalDescription):
            self.load_insulation_material(winding_data.wire)

        # for layer_index, layer_data in enumerate(coil.layersDescription):
        #     print(layer_data.type)
        #     if layer_data.type == MAS.ElectricalType.insulation:
        #         insulation_layer = self.create_insulation_layer(
        #             coil=coil,
        #             layer=layer_data,
        #             bobbin=coil.bobbin,
        #         )

        for turn_index, turn_data in enumerate(coil.turnsDescription):
            for winding in coil.functionalDescription:
                if winding.name == turn_data.winding:
                    wire = winding.wire

            turn_data_raw = {
                "wire": wire,
                "length": turn_data.length,
                "section": turn_data.section,
                "additionalCoordinates": turn_data.additionalCoordinates,
            }
            turn_data_hash = hashlib.sha256(str(turn_data_raw).encode()).hexdigest()

            # print(turn_data)
            cloned = False
            if turn_data_hash == prev_turn_data_hash:
                try:
                    print(f"Cloning previous turn into {turn_data.name}")
                    turn = prev_turn.clone()
                    coordinates = ansyas_utils.convert_axis(turn_data.coordinates)
                    turn.move([coordinates[0] - prev_turn_coordinates[0], coordinates[1] - prev_turn_coordinates[1], coordinates[2] - prev_turn_coordinates[2]])
                    turn.name = f"{ansyas_utils.clean_name(turn_data.name)}_copper"
                    if prev_turn_insulation is not None:
                        turn_insulation = prev_turn_insulation.clone()
                        turn_insulation.move([coordinates[0] - prev_turn_coordinates[0], coordinates[1] - prev_turn_coordinates[1], coordinates[2] - prev_turn_coordinates[2]])
                        turn_insulation.name = f"{ansyas_utils.clean_name(turn_data.name)}_insulation"
                    cloned = True
                except AttributeError:
                    pass

            if not cloned:
                print(f"Creating turn {turn_data.name}")
                turn, turn_insulation = self.create_turn(
                    turn=turn_data,
                    wire=wire,
                    bobbin=coil.bobbin,
                )
                prev_turn_data_hash = turn_data_hash
                prev_turn_coordinates = ansyas_utils.convert_axis(turn_data.coordinates)
                prev_turn = turn
                prev_turn_insulation = turn_insulation

            if self.project.solution_type in ["EddyCurrent", "AC Magnetic", "Transient", "TransientAPhiFormulation"]:
                turn_terminal = self.add_coil_terminals_to_turn(
                    turn_object=turn, 
                    is_input=True, 
                    swap_direction=False
                )

                turns_and_terminals.append((turn, turn_terminal))
            elif self.project.solution_type == "Electrostatic":
                turn_terminal = self.get_turn_terminal(
                    turn_object=turn
                )
                turns_and_terminals.append((turn, turn_terminal))
            elif self.project.solution_type == "SteadyState":
                turns_and_terminals.append(turn)

        return turns_and_terminals

    def create_turn(self, turn: MAS.Turn, wire: MAS.Wire, bobbin: MAS.Bobbin):
        def create_primitive_rectangular_turn(turn: MAS.Turn, wire: MAS.Wire, bobbin: MAS.Bobbin, is_insulation=False):
            converted_turn_coordinates = ansyas_utils.convert_axis(turn.coordinates)
            wire_material = self.get_wire_material(wire, is_insulation)

            if wire.type is MAS.WireType.round or wire.type is MAS.WireType.litz:
                wire_object_radius = self.get_wire_object_radius(wire, is_insulation)

                if wire_object_radius is None:
                    return None
                    
                turn_width_half_side_section = self.project.modeler.create_circle(
                    orientation=pyaedt.constants.PLANE.YZ,
                    origin=converted_turn_coordinates,
                    radius=wire_object_radius,
                    num_sides=self.number_segments_arcs,
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_insulation'}",
                    material=wire_material,
                    non_model=False
                )

            else:
                wire_object_width, wire_object_height = self.get_wire_object_radius(wire, is_insulation)

                if wire_object_width is None or wire_object_height is None:
                    return None

                origin = converted_turn_coordinates
                origin[1] -= wire_object_width / 2
                origin[2] -= wire_object_height / 2
                    
                turn_width_half_side_section = self.project.modeler.create_rectangle(
                    orientation=pyaedt.constants.PLANE.YZ,
                    origin=origin,
                    sizes=[wire_object_width, wire_object_height],
                    num_sides=self.number_segments_arcs,
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_insulation'}",
                    material=wire_material,
                    non_model=False
                )

            if turn_width_half_side_section is None:
                raise RuntimeError("Turn not created, check your turn description")

            turn_width_half_side = self.project.modeler.sweep_along_vector(
                assignment=turn_width_half_side_section,
                sweep_vector=[bobbin.processedDescription.columnDepth, 0, 0]
            )

            turn_radius = turn.coordinates[0]
            turn_turn_radius = turn_radius - bobbin.processedDescription.columnWidth

            result = False
            while not result:
                if wire.type is MAS.WireType.round or wire.type is MAS.WireType.litz:
                    turn_corner_circle = self.project.modeler.create_circle(
                        orientation=pyaedt.constants.PLANE.YZ,
                        origin=[0, turn_turn_radius, 0],
                        radius=wire_object_radius,
                        num_sides=self.number_segments_arcs,
                        is_covered=True,
                        name=f"{ansyas_utils.clean_name(turn.name)}_internal_corner{'_copper' if not is_insulation else '_insulation'}",
                        material=wire_material,
                        non_model=False
                    )
                else:
                    origin = [0, turn_turn_radius - wire_object_width / 2, -wire_object_height / 2]
                        
                    turn_corner_circle = self.project.modeler.create_rectangle(
                        orientation=pyaedt.constants.PLANE.YZ,
                        origin=origin,
                        sizes=[wire_object_width, wire_object_height],
                        is_covered=True,
                        name=f"{ansyas_utils.clean_name(turn.name)}_internal_corner{'_copper' if not is_insulation else '_insulation'}",
                        material=wire_material,
                        non_model=False
                    )

                turn_corner = self.project.modeler.sweep_around_axis(
                    assignment=turn_corner_circle,
                    axis=pyaedt.constants.AXIS.Z,
                    sweep_angle=-90,
                    draft_angle=0,
                    number_of_segments=self.number_segments_arcs_corners
                )
                result = turn_corner.move([bobbin.processedDescription.columnDepth, -turn_turn_radius, 0])

                if not result:
                    self.project.modeler.delete(turn_corner)
                    if self.number_segments_arcs_corners == 0:
                        raise RuntimeError("Something went wrong even with rounds corners :(")
                    self.number_segments_arcs_corners = 0
                else:
                    turn_corner.move(ansyas_utils.convert_axis(turn.coordinates))

            turn_depth_half_side_section = ansyas_utils.get_closest_face(turn_corner, position=[bobbin.processedDescription.columnDepth + turn_turn_radius, 0, ansyas_utils.convert_axis(turn.coordinates)[2]]).create_object()

            turn_depth_half_side = self.project.modeler.sweep_along_vector(
                assignment=turn_depth_half_side_section,
                sweep_vector=[0, -bobbin.processedDescription.columnWidth, 0]
            )
            turn_00 = self.project.modeler.get_object_from_name(self.project.modeler.unite([turn_width_half_side, turn_corner, turn_depth_half_side]))

            turn_01 = self.project.modeler.get_object_from_name(self.project.modeler.duplicate_and_mirror(
                assignment=turn_00,
                origin=[0, 0, 0],
                vector=[0, -1, 0],
            )[0]
            )
            turn_0 = self.project.modeler.get_object_from_name(self.project.modeler.unite([turn_00, turn_01]))

            turn_1 = self.project.modeler.get_object_from_name(self.project.modeler.duplicate_and_mirror(
                assignment=turn_0,
                origin=[0, 0, 0],
                vector=[-1, 0, 0],
            )[0]
            )
            turn_object = self.project.modeler.get_object_from_name(self.project.modeler.unite([turn_0, turn_1]))

            return turn_object

        def create_primitive_round_turn(turn: MAS.Turn, wire: MAS.Wire, bobbin: MAS.Bobbin, is_insulation=False):
            wire_material = self.get_wire_material(wire, is_insulation)
            wire_object_height = 0
            if wire.type is MAS.WireType.round or wire.type is MAS.WireType.litz:
                wire_object_radius = self.get_wire_object_radius(wire, is_insulation)
                wire_object_height = wire_object_radius * 2
                if wire_object_radius is None:
                    return None

                name = f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_insulation'}"

                turn_section = self.project.modeler.create_circle(
                    orientation=pyaedt.constants.PLANE.YZ,
                    origin=ansyas_utils.convert_axis(turn.coordinates),
                    radius=wire_object_radius,
                    num_sides=self.number_segments_arcs,
                    is_covered=True,
                    name=name,
                    material=wire_material,
                    non_model=False
                )
            else:
                wire_object_width, wire_object_height = self.get_wire_object_radius(wire, is_insulation)

                if wire_object_width is None or wire_object_height is None:
                    return None

                origin = ansyas_utils.convert_axis(turn.coordinates)
                origin[1] -= wire_object_width / 2
                origin[2] -= wire_object_height / 2
                turn_section = self.project.modeler.create_rectangle(
                    orientation=pyaedt.constants.PLANE.YZ,
                    origin=origin,
                    sizes=[wire_object_width, wire_object_height],
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_insulation'}",
                    material=wire_material,
                    non_model=False
                )

            turn_object = self.project.modeler.sweep_around_axis(
                assignment=turn_section,
                axis=pyaedt.constants.AXIS.Z,
                sweep_angle=360,
                draft_angle=0,
                number_of_segments=self.number_segments_arcs
            )
            return turn_object

        if bobbin.processedDescription.columnShape is MAS.ColumnShape.round:
            turn_object = create_primitive_round_turn(turn, wire, bobbin, is_insulation=False)
        else:
            turn_object = create_primitive_rectangular_turn(turn, wire, bobbin, is_insulation=False)

        turn_insulation_object = None
        if self.add_insulation:
            if bobbin.processedDescription.columnShape is MAS.ColumnShape.round:
                turn_insulation_object = create_primitive_round_turn(turn, wire, bobbin, is_insulation=True)
            else:
                turn_insulation_object = create_primitive_rectangular_turn(turn, wire, bobbin, is_insulation=True)
            if turn_insulation_object is not None:
                turn_insulation_object.subtract(turn_object, True)

        if wire.type is MAS.WireType.litz:
            turn_object.color = ansyas_utils.color_litz
        else:
            turn_object.color = ansyas_utils.color_copper

        if turn_insulation_object is not None:
            turn_insulation_object.color = ansyas_utils.color_teflon
        return turn_object, turn_insulation_object

    def create_insulation_layer(self, coil: MAS.Coil, layer: MAS.Layer, bobbin: MAS.Bobbin):
        def create_primitive_rectangular_layer(layer: MAS.Layer, bobbin: MAS.Bobbin):
            converted_layer_coordinates = ansyas_utils.convert_axis(layer.coordinates)

            if isinstance(layer.insulationMaterial, str):
                layer_material = layer.insulationMaterial
            else:
                layer_material = layer.insulationMaterial.name

            layer_object_width = layer.dimensions[0]
            layer_object_height = layer.dimensions[1]

            if layer_object_width is None or layer_object_height is None:
                return None

            origin = converted_layer_coordinates
            origin[1] -= layer_object_width / 2
            origin[2] -= layer_object_height / 2
                
            layer_width_half_side_section = self.project.modeler.create_rectangle(
                orientation=pyaedt.constants.PLANE.YZ,
                origin=origin,
                sizes=[layer_object_width, layer_object_height],
                is_covered=True,
                name=f"{ansyas_utils.clean_name(layer.name)}_insulation",
                material=layer_material,
                non_model=False
            )

            if layer_width_half_side_section is None:
                raise RuntimeError("Layer not created, check your turn description")

            layer_width_half_side = self.project.modeler.sweep_along_vector(
                assignment=layer_width_half_side_section,
                sweep_vector=[bobbin.processedDescription.columnDepth, 0, 0]
            )

            layer_radius = layer.coordinates[0]
            layer_layer_radius = layer_radius - bobbin.processedDescription.columnWidth

            result = False
            while not result:
                wire_object_width = layer.dimensions[0]
                wire_object_height = layer.dimensions[1]
                origin = [0, layer_layer_radius - wire_object_width / 2, -wire_object_height / 2]
                    
                layer_corner_circle = self.project.modeler.create_rectangle(
                    orientation=pyaedt.constants.PLANE.YZ,
                    origin=origin,
                    sizes=[wire_object_width, wire_object_height],
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(layer.name)}_internal_corner_insulation",
                    material="wire_material",
                    non_model=False
                )

                layer_corner = self.project.modeler.sweep_around_axis(
                    assignment=layer_corner_circle,
                    axis=pyaedt.constants.AXIS.Z,
                    sweep_angle=-90,
                    draft_angle=0,
                    number_of_segments=self.number_segments_arcs_corners
                )
                result = layer_corner.move([bobbin.processedDescription.columnDepth, -layer_layer_radius, 0])

                if not result:
                    self.project.modeler.delete(layer_corner)
                    if self.number_segments_arcs_corners == 0:
                        raise RuntimeError("Something went wrong even with rounds corners :(")
                    self.number_segments_arcs_corners = 0
                else:
                    layer_corner.move(ansyas_utils.convert_axis(layer.coordinates))

            layer_depth_half_side_section = ansyas_utils.get_closest_face(layer_corner, position=[bobbin.processedDescription.columnDepth + layer_layer_radius, 0, ansyas_utils.convert_axis(layer.coordinates)[2]]).create_object()

            layer_depth_half_side = self.project.modeler.sweep_along_vector(
                assignment=layer_depth_half_side_section,
                sweep_vector=[0, -bobbin.processedDescription.columnWidth, 0]
            )
            layer_00 = self.project.modeler.get_object_from_name(self.project.modeler.unite([layer_width_half_side, layer_corner, layer_depth_half_side]))

            layer_01 = self.project.modeler.get_object_from_name(self.project.modeler.duplicate_and_mirror(
                assignment=layer_00,
                origin=[0, 0, 0],
                vector=[0, -1, 0],
            )[0]
            )
            layer_0 = self.project.modeler.get_object_from_name(self.project.modeler.unite([layer_00, layer_01]))

            layer_1 = self.project.modeler.get_object_from_name(self.project.modeler.duplicate_and_mirror(
                assignment=layer_0,
                origin=[0, 0, 0],
                vector=[-1, 0, 0],
            )[0]
            )
            layer_object = self.project.modeler.get_object_from_name(self.project.modeler.unite([layer_0, layer_1]))

            return layer_object

        def create_primitive_round_layer(layer: MAS.Layer, bobbin: MAS.Bobbin,):
            layer_material = layer.material

            layer_object_width = layer.dimensions[0]
            layer_object_height = layer.dimensions[1]

            if layer_object_width is None or layer_object_height is None:
                return None

            origin = ansyas_utils.convert_axis(layer.coordinates)
            origin[1] -= layer_object_width / 2
            origin[2] -= layer_object_height / 2
            layer_section = self.project.modeler.create_rectangle(
                orientation=pyaedt.constants.PLANE.YZ,
                origin=origin,
                sizes=[layer_object_width, layer_object_height],
                is_covered=True,
                name=f"{ansyas_utils.clean_name(layer.name)}",
                material=layer_material,
                non_model=False
            )

            layer_object = self.project.modeler.sweep_around_axis(
                assignment=layer_section,
                axis=pyaedt.constants.AXIS.Z,
                sweep_angle=360,
                draft_angle=0,
                number_of_segments=self.number_segments_arcs
            )
            return layer_object

        if bobbin.processedDescription.columnShape is MAS.ColumnShape.round:
            layer_object = create_primitive_round_layer(layer, bobbin)
        else:
            layer_object = create_primitive_rectangular_layer(layer, bobbin)

        layer_object.color = ansyas_utils.color_tape
        return layer_object


class ToroidalCoil(Coil):
    def __init__(self, project, number_segments_arcs=12, add_insulation=False):
        super().__init__(
            project=project,
            number_segments_arcs=number_segments_arcs,
            add_insulation=add_insulation
        )

        self.terminals_plane = pyaedt.constants.PLANE.XY

    def create_coil(self, coil: MAS.Coil):
        turns_and_terminals = []

        prev_turn_data_hash = -1
        prev_turn = None
        prev_turn_polar_coordinates = None
        prev_turn_insulation = None

        for turn_index, turn_data in enumerate(coil.turnsDescription):
            for winding in coil.functionalDescription:
                if winding.name == turn_data.winding:
                    wire = winding.wire

            internal_turn_angle = 0
            turn_angle = 0
            if turn_data.additionalCoordinates is not None:
                turn_angle = math.atan2(turn_data.coordinates[1], turn_data.coordinates[0])
                internal_turn_angle = round(math.fabs(turn_angle - math.atan2(turn_data.additionalCoordinates[0][1] - turn_data.coordinates[1], turn_data.additionalCoordinates[0][0] - turn_data.coordinates[0])), 12)

            turn_data_raw = {
                "wire": wire,
                "length": round(turn_data.length, 9),
                "section": turn_data.section,
                "internal_turn_angle": internal_turn_angle,
            }
            turn_data_hash = hashlib.sha256(str(turn_data_raw).encode()).hexdigest()

            cloned = False
            if turn_data_hash == prev_turn_data_hash:
                try:
                    print(f"Cloning previous turn into {turn_data.name}")
                    turn = prev_turn.clone()
                    polar_coordinates = ansyas_utils.cartesian_to_polar(turn_data.coordinates, coil.bobbin.processedDescription.windingWindows[0].radialHeight)
                    rotation = polar_coordinates[1] - prev_turn_polar_coordinates[1]
                    turn.rotate(pyaedt.constants.AXIS.Z, rotation)

                    # turn.move([coordinates[0] - prev_turn_coordinates[0], coordinates[1] - prev_turn_coordinates[1], coordinates[2] - prev_turn_coordinates[2]])
                    turn.name = f"{ansyas_utils.clean_name(turn_data.name)}_copper"
                    if prev_turn_insulation is not None:
                        turn_insulation = prev_turn_insulation.clone()
                        turn_insulation.rotate(pyaedt.constants.AXIS.Z, rotation)
                        # turn_insulation.move([coordinates[0] - prev_turn_coordinates[0], coordinates[1] - prev_turn_coordinates[1], coordinates[2] - prev_turn_coordinates[2]])
                        turn_insulation.name = f"{ansyas_utils.clean_name(turn_data.name)}_insulation"
                    cloned = True
                except AttributeError:
                    pass

            if not cloned:
                print(f"Creating turn {turn_data.name}")
                turn, turn_insulation = self.create_rectangular_turn(
                    turn=turn_data,
                    wire=wire,
                    bobbin=coil.bobbin,
                )
                prev_turn_data_hash = turn_data_hash
                prev_turn_polar_coordinates = ansyas_utils.cartesian_to_polar(turn_data.coordinates, coil.bobbin.processedDescription.windingWindows[0].radialHeight)
                prev_turn = turn
                prev_turn_insulation = turn_insulation

            if self.project.solution_type in ["EddyCurrent", "AC Magnetic", "Transient", "TransientAPhiFormulation"]:
                turn_terminal = self.add_coil_terminals_to_turn(
                    turn_object=turn, 
                    is_input=True, 
                    swap_direction=False
                )
                turns_and_terminals.append((turn, turn_terminal))

            elif self.project.solution_type == "Electrostatic":
                turn_terminal = self.get_turn_terminal(
                    turn_object=turn
                )
                turns_and_terminals.append((turn, turn_terminal))

        return turns_and_terminals

    def create_rectangular_turn(self, turn: MAS.Turn, wire: MAS.Wire, bobbin: MAS.Bobbin):
        turn_insulation_section = None
        def create_primitive_turn(turn: MAS.Turn, wire: MAS.Wire, bobbin: MAS.Bobbin, is_insulation=False):

            converted_turn_coordinates = ansyas_utils.convert_axis_toroidal_core(turn.coordinates)
            converted_additional_turn_coordinates = ansyas_utils.convert_axis_toroidal_core(turn.additionalCoordinates[0])
            wire_material = self.get_wire_material(wire, is_insulation)
            if wire.type is MAS.WireType.round or wire.type is MAS.WireType.litz:
                wire_object_radius = self.get_wire_object_radius(wire, is_insulation)
                wire_turn_center_height = wire_object_radius

                turn_section = self.project.modeler.create_circle(
                    orientation=pyaedt.constants.PLANE.XY,
                    origin=converted_turn_coordinates,
                    radius=wire_object_radius,
                    num_sides=self.number_segments_arcs,
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_insulation'}",
                    material=wire_material,
                    non_model=False
                )

                if turn.additionalCoordinates is None:
                    raise RuntimeError("Turn is missing additional coordinates")

                additional_turn_section = self.project.modeler.create_circle(
                    orientation=pyaedt.constants.PLANE.XY,
                    origin=ansyas_utils.convert_axis_toroidal_core(turn.additionalCoordinates[0]),
                    radius=wire_object_radius,
                    num_sides=self.number_segments_arcs,
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(turn.name)}_additional{'_copper' if not is_insulation else '_insulation'}",
                    material=wire_material,
                    non_model=False
                )

                if turn_section is None:
                    raise RuntimeError("Turn not created, check your turn description")

            else:
                wire_object_width, wire_object_height = self.get_wire_object_radius(wire, is_insulation)

                if wire_object_width is None or wire_object_height is None:
                    return None

                origin = converted_turn_coordinates
                origin[0] -= wire_object_height / 2
                origin[1] -= wire_object_width / 2
                turn_section = self.project.modeler.create_rectangle(
                    orientation=pyaedt.constants.PLANE.XY,
                    origin=origin,
                    sizes=[wire_object_height, wire_object_width],
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_insulation'}",
                    material=wire_material,
                    non_model=False
                )

                if turn.additionalCoordinates is None:
                    raise RuntimeError("Turn is missing additional coordinates")

                origin = ansyas_utils.convert_axis_toroidal_core(turn.additionalCoordinates[0])
                origin[0] -= wire_object_height / 2
                origin[1] -= wire_object_width / 2
                additional_turn_section = self.project.modeler.create_rectangle(
                    orientation=pyaedt.constants.PLANE.XY,
                    origin=origin,
                    sizes=[wire_object_height, wire_object_width],
                    is_covered=True,
                    name=f"{ansyas_utils.clean_name(turn.name)}_additional{'_copper' if not is_insulation else '_insulation'}",
                    material=wire_material,
                    non_model=False
                )

                if turn.additionalCoordinates is None:
                    raise RuntimeError("Turn is missing additional coordinates")

            turn_object = self.project.modeler.sweep_along_vector(
                assignment=turn_section,
                sweep_vector=[0, 0, bobbin.processedDescription.columnDepth]
            )
            turn_radius = math.hypot(turn.coordinates[0], turn.coordinates[1])
            turn_radial_height = bobbin.processedDescription.windingWindows[0].radialHeight - turn_radius

            additional_turn_radius = math.hypot(turn.additionalCoordinates[0][0], turn.additionalCoordinates[0][1])
            additional_turn_radial_height = additional_turn_radius - (bobbin.processedDescription.windingWindows[0].radialHeight + bobbin.processedDescription.columnWidth * 2)

            additional_turn_object = self.project.modeler.sweep_along_vector(
                assignment=additional_turn_section,
                sweep_vector=[0, 0, bobbin.processedDescription.columnDepth + (turn_radial_height - additional_turn_radial_height)]
            )

            real_turn_rotation = 180.0 / math.pi * math.atan2((turn.additionalCoordinates[0][1] - turn.coordinates[1]), (turn.additionalCoordinates[0][0] - turn.coordinates[0]))
            turn_object.move([-x for x in ansyas_utils.convert_axis_toroidal_core(turn.coordinates)])
            turn_object.rotate(pyaedt.constants.AXIS.Z, real_turn_rotation)
            turn_object.move(ansyas_utils.convert_axis_toroidal_core(turn.coordinates))

            additional_turn_object.move([-x for x in ansyas_utils.convert_axis_toroidal_core(turn.additionalCoordinates[0])])
            additional_turn_object.rotate(pyaedt.constants.AXIS.Z, real_turn_rotation)
            additional_turn_object.move(ansyas_utils.convert_axis_toroidal_core(turn.additionalCoordinates[0]))
            center_point_aprox = [(converted_turn_coordinates[0] + converted_additional_turn_coordinates[0]) / 2,
                                  (converted_turn_coordinates[1] + converted_additional_turn_coordinates[1]) / 2,
                                  (converted_turn_coordinates[2] + converted_additional_turn_coordinates[2]) / 2 + bobbin.processedDescription.columnDepth + turn_radial_height]

            result = False
            while not result:

                if wire.type is MAS.WireType.round or wire.type is MAS.WireType.litz:
                    turn_top_internal_corner = self.project.modeler.create_circle(
                        orientation=pyaedt.constants.PLANE.XY,
                        origin=[0, -turn_radial_height, 0],
                        radius=wire_object_radius,
                        num_sides=self.number_segments_arcs,
                        is_covered=True,
                        name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_internal_corner'}",
                        material=wire_material,
                        non_model=False
                    )
                else:
                    origin = [-wire_object_height / 2, -(turn_radial_height + wire_object_width / 2), 0]
                    turn_top_internal_corner = self.project.modeler.create_rectangle(
                        orientation=pyaedt.constants.PLANE.XY,
                        origin=origin,
                        sizes=[wire_object_height, wire_object_width],
                        is_covered=True,
                        name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_internal_corner'}",
                        material=wire_material,
                        non_model=False
                    )

                turn_top_internal_corner = self.project.modeler.sweep_around_axis(
                    assignment=turn_top_internal_corner,
                    axis=pyaedt.constants.AXIS.X,
                    sweep_angle=-90,
                    draft_angle=0,
                    number_of_segments=self.number_segments_arcs_corners
                )

                result = turn_top_internal_corner.move([0, turn_radial_height, bobbin.processedDescription.columnDepth])

                if not result:
                    self.project.modeler.delete(turn_top_internal_corner)
                    if self.number_segments_arcs_corners == 0:
                        raise RuntimeError("Something went wrong even with rounds corners :(")
                    self.number_segments_arcs_corners = 0
                else:
                    turn_internal_face = turn_top_internal_corner.faces[1]

                    turn_top_internal_corner.rotate(pyaedt.constants.AXIS.Z, real_turn_rotation)
                    turn_top_internal_corner.move(ansyas_utils.convert_axis_toroidal_core(turn.coordinates))
                    turn_internal_face = ansyas_utils.get_closest_face(turn_top_internal_corner, center_point_aprox)

            result = False
            while not result:

                if wire.type is MAS.WireType.round or wire.type is MAS.WireType.litz:
                    turn_top_external_corner = self.project.modeler.create_circle(
                        orientation=pyaedt.constants.PLANE.XY,
                        origin=[0, -additional_turn_radial_height, 0],
                        radius=wire_object_radius,
                        num_sides=self.number_segments_arcs,
                        is_covered=True,
                        name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_external_corner'}",
                        material=wire_material,
                        non_model=False
                    )
                else:
                    origin = [-wire_object_height / 2, -(additional_turn_radial_height + wire_object_width / 2), 0]
                    turn_top_external_corner = self.project.modeler.create_rectangle(
                        orientation=pyaedt.constants.PLANE.XY,
                        origin=origin,
                        sizes=[wire_object_height, wire_object_width],
                        is_covered=True,
                        name=f"{ansyas_utils.clean_name(turn.name)}{'_copper' if not is_insulation else '_internal_corner'}",
                        material=wire_material,
                        non_model=False
                    )

                turn_top_external_corner = self.project.modeler.sweep_around_axis(
                    assignment=turn_top_external_corner,
                    axis=pyaedt.constants.AXIS.X,
                    sweep_angle=-90,
                    draft_angle=0,
                    number_of_segments=self.number_segments_arcs_corners
                )

                real_turn_rotation = 180.0 / math.pi * math.atan2((turn.additionalCoordinates[0][1] - turn.coordinates[1]), (turn.additionalCoordinates[0][0] - turn.coordinates[0]))

                result = turn_top_external_corner.move([0, additional_turn_radial_height, bobbin.processedDescription.columnDepth + (turn_radial_height - additional_turn_radial_height)])
                if not result:
                    self.project.modeler.delete(turn_top_internal_corner)
                    if self.number_segments_arcs_corners == 0:
                        raise RuntimeError("Something went wrong even with rounds corners :(")
                    self.number_segments_arcs_corners = 0
                else:
                    additional_turn_internal_face = turn_top_external_corner.faces[1]

                    turn_top_external_corner.rotate(pyaedt.constants.AXIS.Z, 180)
                    turn_top_external_corner.rotate(pyaedt.constants.AXIS.Z, real_turn_rotation)
                    turn_top_external_corner.move(ansyas_utils.convert_axis_toroidal_core(turn.additionalCoordinates[0]))
                    additional_turn_internal_face = ansyas_utils.get_closest_face(turn_top_external_corner, center_point_aprox)

            top_part_vector = [additional_turn_internal_face.center[0] - turn_internal_face.center[0],
                               additional_turn_internal_face.center[1] - turn_internal_face.center[1],
                               additional_turn_internal_face.center[2] - turn_internal_face.center[2]]
            turn_internal_face = turn_internal_face.create_object()

            turn_top = self.project.modeler.sweep_along_vector(
                assignment=turn_internal_face,
                sweep_vector=top_part_vector
            )
            turn_top = self.project.modeler.get_object_from_name(self.project.modeler.unite([turn_top, turn_object, turn_top_internal_corner, additional_turn_object, turn_top_external_corner]))
            turn_bottom = self.project.modeler.get_object_from_name(
                self.project.modeler.duplicate_and_mirror(
                    assignment=turn_top,
                    origin=[0, 0, 0],
                    vector=[0, 0, -1],
                )[0]
            )
            turn_object = self.project.modeler.get_object_from_name(self.project.modeler.unite([turn_top, turn_bottom]))

            return turn_object

        if bobbin.processedDescription.columnShape is MAS.ColumnShape.round:
            raise NotImplementedError
        else:
            turn_object = create_primitive_turn(turn, wire, bobbin, is_insulation=False)

        turn_insulation_object = None
        if self.add_insulation:
            if bobbin.processedDescription.columnShape is MAS.ColumnShape.round:
                raise NotImplementedError
            else:
                turn_insulation_object = create_primitive_turn(turn, wire, bobbin, is_insulation=True)
            if turn_insulation_object is not None:
                turn_insulation_object.subtract(turn_object, True)

        if wire.type is MAS.WireType.litz:
            turn_object.color = ansyas_utils.color_litz
        else:
            turn_object.color = ansyas_utils.color_copper

        if turn_insulation_object is not None:
            turn_insulation_object.color = ansyas_utils.color_teflon
        return turn_object, turn_insulation_object

        return turn_object, None
