"""
Ansyas - Decoupled FEM Simulation Framework for Magnetics.

This module provides the main Ansyas class that orchestrates FEM simulations
using pluggable backends for geometry, meshing, solving, and excitation.

The architecture allows swapping backends:
- Geometry: Ansys (PyAEDT), CadQuery, etc.
- Meshing: Ansys, Gmsh, etc.
- Solver: Ansys Maxwell/Icepak, MFEM, etc.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

try:
    from . import MAS_models as MAS
    from . import ansyas_utils
except ImportError:
    import MAS_models as MAS
    import ansyas_utils

try:
    from .backends.base import (
        GeometryBackend,
        MaterialBackend,
        MeshingBackend,
        ExcitationBackend,
        SolverBackend,
        SolverSetup,
        BackendRegistry,
        Axis,
        Plane,
    )
except ImportError:
    from backends.base import (
        GeometryBackend,
        MaterialBackend,
        MeshingBackend,
        ExcitationBackend,
        SolverBackend,
        SolverSetup,
        BackendRegistry,
        Axis,
        Plane,
    )


class Ansyas:
    """
    Main orchestrator for FEM magnetic simulations.

    This class coordinates the various backends (geometry, meshing, solving)
    to perform complete magnetic simulations. By default, it uses Ansys backends
    but can be configured to use alternative backends.

    Attributes:
        geometry_backend: Backend for 3D geometry creation
        material_backend: Backend for material definition
        meshing_backend: Backend for mesh generation
        excitation_backend: Backend for excitation setup
        solver_backend: Backend for FEM solving
    """

    def __init__(
        self,
        geometry_backend: str = "ansys",
        meshing_backend: str = "ansys",
        solver_backend: str = "ansys",
        number_segments_arcs: int = 12,
        initial_mesh_configuration: int = 5,
        maximum_error_percent: float = 3,
        maximum_passes: int = 40,
        refinement_percent: float = 30,
        scale: float = 1,
    ):
        """
        Initialize Ansyas with specified backends.

        Args:
            geometry_backend: Name of geometry backend ("ansys", "cadquery")
            meshing_backend: Name of meshing backend ("ansys", "gmsh")
            solver_backend: Name of solver backend ("ansys", "mfem")
            number_segments_arcs: Number of segments for arc approximation
            initial_mesh_configuration: Initial mesh slider (1-5)
            maximum_error_percent: Maximum error percentage for convergence
            maximum_passes: Maximum adaptive passes
            refinement_percent: Refinement percentage per pass
            scale: Scale factor for geometry
        """
        self.initial_mesh_configuration = initial_mesh_configuration
        self.number_segments_arcs = number_segments_arcs
        self.maximum_error_percent = maximum_error_percent
        self.refinement_percent = refinement_percent
        self.maximum_passes = maximum_passes
        self.scale = scale

        if not 1 <= initial_mesh_configuration <= 5:
            raise ValueError(
                f"initial_mesh_configuration must be 1-5, got {initial_mesh_configuration}"
            )
        if maximum_error_percent <= 0:
            raise ValueError(
                f"maximum_error_percent must be > 0, got {maximum_error_percent}"
            )
        if maximum_passes < 1:
            raise ValueError(f"maximum_passes must be >= 1, got {maximum_passes}")
        if not 0 < refinement_percent <= 100:
            raise ValueError(
                f"refinement_percent must be in (0, 100], got {refinement_percent}"
            )
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if number_segments_arcs != 0 and number_segments_arcs < 3:
            raise ValueError(
                f"number_segments_arcs must be 0 (round) or >= 3, got {number_segments_arcs}"
            )

        # Backend names (for lazy initialization)
        self._geometry_backend_name = geometry_backend
        self._meshing_backend_name = meshing_backend
        self._solver_backend_name = solver_backend

        # Backend instances (initialized when project is created)
        self.geometry_backend: Optional[GeometryBackend] = None
        self.material_backend: Optional[MaterialBackend] = None
        self.meshing_backend: Optional[MeshingBackend] = None
        self.excitation_backend: Optional[ExcitationBackend] = None
        self.solver_backend: Optional[SolverBackend] = None

        # Domain-specific builders (these use the backends)
        self.bobbin_builder = None
        self.core_builder = None
        self.cooling_builder = None
        self.coil_builder = None
        self.outputs_extractor = None

        # Region padding settings
        self.padding = {
            "x_pos": 100,
            "y_pos": 100,
            "z_pos": 100,
            "x_neg": 100,
            "y_neg": 100,
            "z_neg": 100,
        }

        # Project state
        self.project = None
        self.project_path = None
        self.project_name = None
        self.solution_type = None

    def _initialize_backends(self, project):
        """Initialize backends with the created project."""
        # For now, we only have Ansys backends implemented
        # Future: use BackendRegistry to get the appropriate backend class

        try:
            from .backends.ansys import (
                AnsysGeometryBackend,
                AnsysMaterialBackend,
                AnsysMeshingBackend,
                AnsysExcitationBackend,
                AnsysSolverBackend,
            )
        except ImportError:
            from backends.ansys import (
                AnsysGeometryBackend,
                AnsysMaterialBackend,
                AnsysMeshingBackend,
                AnsysExcitationBackend,
                AnsysSolverBackend,
            )

        self.geometry_backend = AnsysGeometryBackend()
        self.geometry_backend.initialize(project=project)

        self.material_backend = AnsysMaterialBackend()
        self.material_backend.initialize(project=project)

        self.meshing_backend = AnsysMeshingBackend()
        self.meshing_backend.initialize(project=project)

        self.excitation_backend = AnsysExcitationBackend()
        self.excitation_backend.initialize(project=project)

        self.solver_backend = AnsysSolverBackend()
        self.solver_backend.initialize(solver_type=self.solution_type, project=project)

    def fit(self):
        """Fit the view to show all objects."""
        if self.geometry_backend:
            self.geometry_backend.fit_all()
        elif self.project:
            self.project.modeler.fit_all()

    def analyze(self):
        """Run the simulation."""
        if self.solver_backend:
            return self.solver_backend.analyze()
        elif self.project:
            self.project.analyze()

    def save(self):
        """Save the project."""
        if self.solver_backend:
            self.solver_backend.save_project()
        elif self.project:
            self.project.oeditor.CleanUpModel()
            self.project.save_project()

    def set_units(self, units: str):
        """
        Set the units of the modeler.

        Parameters
        ----------
        units : str
            Units to use in the model of the magnetic.

        Examples
        --------
        Sets units to meters.

        >>> ansyas.set_units("meter")
        """
        if self.geometry_backend:
            self.geometry_backend.set_units(units)
        elif self.project:
            self.project.modeler.model_units = units

    def create_project(
        self,
        outputs_folder: str,
        project_name: str,
        non_graphical: bool = False,
        new_desktop_session: bool = False,
        solution_type: str = "EddyCurrent",
        specified_version: str = None,
    ):
        """
        Create a project for the given inputs.

        Configures Ansyas and creates an Ansys project with the requested inputs.

        Parameters
        ----------
        outputs_folder : str
            Path to store the output project.
        project_name : str
            Name of the project.
        non_graphical : bool
            Whether to launch AEDT in non-graphical mode.
        new_desktop_session : bool, optional
            Whether to launch an instance of AEDT in a new thread.
        solution_type : str, optional
            Solution type to apply to the design. The default is
            ``EddyCurrent``.
        specified_version : str
            Version of AEDT to use.

        Returns
        -------
        project
            The created PyAEDT project instance.
        """
        from ansys.aedt.core import Maxwell3d, Icepak, Hfss

        project_name = f"{project_name}.aedt"
        self.project_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), outputs_folder
        )

        os.makedirs(self.project_path, exist_ok=True)
        self.solution_type = solution_type
        self.project_name = project_name
        self.project_name = (
            self.project_name.replace(" ", "_")
            .replace(",", "_")
            .replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        # Directory already created above with exist_ok=True

        self.project_name = os.path.join(self.project_path, self.project_name)

        if self.solution_type == "SteadyState":
            icepak_result = Icepak(
                project=self.project_name,
                solution_type="SteadyState",
                non_graphical=non_graphical,
                version=specified_version,
                new_desktop=new_desktop_session,
                close_on_exit=False,
            )
            # Handle case where Icepak returns boolean instead of object (license error)
            if isinstance(icepak_result, bool):
                raise RuntimeError(
                    "Failed to initialize Icepak project. This is likely due to license limitations "
                    "(Student version may not support Icepak). Error: Icepak initialization returned boolean."
                )
            self.project = icepak_result

            # Verify the Icepak object is properly initialized
            if not hasattr(self.project, "_odesign") or self.project._odesign is None:
                raise RuntimeError(
                    "Failed to initialize Icepak project. AEDT connection may have failed. "
                    "Ensure no other AEDT instances are running and try again."
                )
        elif self.solution_type == "Terminal":
            self.project = Hfss(
                project=self.project_name,
                solution_type=solution_type,
                non_graphical=non_graphical,
                version=specified_version,
                new_desktop=new_desktop_session,
                close_on_exit=False
            )
        else:
            self.project = Maxwell3d(
                project=self.project_name,
                solution_type=solution_type,
                non_graphical=non_graphical,
                version=specified_version,
                new_desktop=new_desktop_session,
                close_on_exit=False,
            )

            # Verify the Maxwell3d object is properly initialized
            if not hasattr(self.project, "_odesign") or self.project._odesign is None:
                raise RuntimeError(
                    "Failed to initialize Maxwell3d project. AEDT connection may have failed. "
                    "Ensure no other AEDT instances are running and try again."
                )

            self.project.change_design_settings({"ComputeTransientCapacitance": True})
            self.project.change_design_settings({"ComputeTransientInductance": True})
            if hasattr(self.project, "_odesign") and self.project._odesign is not None:
                self.project.mesh.assign_initial_mesh_from_slider(
                    self.initial_mesh_configuration, curvilinear=True
                )

        self.project.autosave_disable()

        # Initialize backends with the created project
        self._initialize_backends(self.project)

        return self.project

    def get_project_location(self) -> str:
        """Get the project file location."""
        return self.project_name

    def create_builders(self, magnetic: MAS.Magnetic):
        """Create domain-specific builders for core, coil, etc."""
        try:
            from . import core as core_builder
            from . import bobbin as bobbin_builder
            from . import cooling as cooling_builder
            from . import coil as coil_builder
            from . import excitation as excitation_builder
            from . import outputs
        except ImportError:
            import core as core_builder
            import bobbin as bobbin_builder
            import cooling as cooling_builder
            import coil as coil_builder
            import excitation as excitation_builder
            import outputs

        self.core_builder = core_builder.Core(
            project=self.project,
        )
        self.cooling_builder = cooling_builder.Cooling(
            project=self.project,
        )
        self.bobbin_builder = bobbin_builder.Bobbin(
            project=self.project, number_segments_arcs=self.number_segments_arcs
        )

        if self.solution_type == "Electrostatic":
            if (
                magnetic.core.functionalDescription.shape.family
                is MAS.CoreShapeFamily.t
            ):
                self.coil_builder = coil_builder.ToroidalCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=True,
                )
            else:
                self.coil_builder = coil_builder.ConcentricSpiralCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=True,
                )
        else:
            if (
                magnetic.core.functionalDescription.shape.family
                is MAS.CoreShapeFamily.t
            ):
                self.coil_builder = coil_builder.ToroidalCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=False,
                )
            else:
                self.coil_builder = coil_builder.ConcentricCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=False,
                )

        self.excitation_builder = excitation_builder.Excitation(
            project=self.project,
        )
        self.outputs_extractor = outputs.Outputs(
            project=self.project,
        )

    def create_boundary_region(self, padding: Dict[str, float] = None):
        """Create the simulation boundary region."""
        if padding is None:
            if self.geometry_backend:
                return self.geometry_backend.create_air_region(
                    {
                        "x_pos": 50,
                        "y_pos": 50,
                        "z_pos": 50,
                        "x_neg": 50,
                        "y_neg": 50,
                        "z_neg": 50,
                    }
                )
            else:
                region = self.project.modeler.create_air_region(
                    x_pos=50, y_pos=50, z_pos=50, x_neg=50, y_neg=50, z_neg=50
                )
                if region is None or region is False:
                    region = self.project.modeler.get_objects_w_string("Region")
                    region = self.project.modeler.get_object_from_name(region[0])
                return region
        else:
            padding_list = [
                padding["x_pos"],
                padding["x_neg"],
                padding["y_pos"],
                padding["y_neg"],
                padding["z_pos"],
                padding["z_neg"],
            ]
            if self.geometry_backend:
                return self.geometry_backend.create_region(
                    padding_list, is_percentage=True
                )
            else:
                region = self.project.modeler.create_region(
                    pad_percent=padding_list, is_percentage=True
                )
                if region is None or region is False:
                    region = self.project.modeler.get_objects_w_string("Region")
                    region = self.project.modeler.get_object_from_name(region[0])
                return region

    def create_boundary_conditions(self, conditions):
        """Create thermal boundary conditions."""
        region_names = self.project.modeler.get_objects_w_string("Region")
        region = self.project.modeler.get_object_from_name(region_names[0])

        if conditions.cooling is None or conditions.cooling.velocity is None:
            faces = region.faces
            self.excitation_backend.assign_pressure_free_opening(
                faces=[int(str(x)) for x in faces], name="AirOpening"
            )
        else:
            velocity = ansyas_utils.convert_axis(conditions.cooling.velocity)
            velocity_strs = [
                f"{velocity[0]}m_per_sec",
                f"{velocity[1]}m_per_sec",
                f"{velocity[2]}m_per_sec",
            ]
            temperature = (
                "AmbientTemp"
                if conditions.cooling.temperature is None
                else conditions.cooling.temperature
            )

            self.excitation_backend.assign_free_opening(
                faces=region.bottom_face_x,
                flow_type="Pressure",
                velocity=velocity_strs,
                temperature=temperature,
                name="AirOpening",
            )

    def create_setup(self, frequency: float = 100000, single_frequency: bool = False):
        """Create solver setup with frequency sweeps."""
        setup_config = SolverSetup(
            solver_type=self.solution_type,
            frequency=frequency,
            max_passes=self.maximum_passes,
            max_error_percent=self.maximum_error_percent,
            refinement_percent=self.refinement_percent,
        )

        if self.solution_type in ["Transient", "TransientAPhiFormulation"]:
            setup_config.stop_time = 2 / frequency
            setup_config.time_step = 2 / frequency / 10

        setup = self.solver_backend.create_setup(setup_config)

        if self.solution_type in ["EddyCurrent", "AC Magnetic"]:
            if single_frequency:
                self.solver_backend.add_default_frequency_sweeps(
                    setup, single_frequency=frequency
                )
            else:
                self.solver_backend.add_default_frequency_sweeps(setup)

        return setup

    def add_skin_effect(
        self, wires_faces, skin_depth: float = 0.0002, number_layers: int = 2
    ):
        """Add skin effect mesh refinement to wire faces."""
        if not isinstance(wires_faces, list):
            wires_faces = [wires_faces]

        self.meshing_backend.assign_skin_depth(
            faces=wires_faces, skin_depth=skin_depth, num_layers=number_layers
        )

    def create_magnetic_simulation(
        self,
        mas: Union[MAS.Mas, dict],
        simulate: bool = False,
        operating_point_index: int = 0,
        single_frequency: bool = False,
    ):
        """
        Create an automatic simulation from the MAS file.

        Parameters
        ----------
        mas : MAS.Mas, dict
            MAS file or dict containing the information about the magnetic,
            its inputs, and outputs.
        simulate : bool, optional
            Runs the simulation and captures outputs if true. Default is False.
        operating_point_index : int, optional
            Index of the operating point to simulate.
        """
        if isinstance(mas, dict):
            mas = MAS.Mas.from_dict(mas)

        magnetic = mas.magnetic
        inputs = mas.inputs
        outputs = mas.outputs

        # TODO: move conditional
        if magnetic.core.functionalDescription.type.value.lower() == "toroidal":
            self.generate_choke_descriptor(magnetic)
            return

        self.create_builders(magnetic)

        core_parts = self.core_builder.import_core(
            core=magnetic.core,
            operating_point=inputs.operatingPoints[operating_point_index],
        )

        if self.solution_type in "SteadyState":
            core_losses = outputs[operating_point_index].coreLosses.coreLosses
            self.core_builder.assign_core_losses_as_heat_source(core_parts, core_losses)
            self.cooling_builder.create_cooling(
                magnetic.core,
                inputs.operatingPoints[operating_point_index].conditions.cooling,
            )

        self.fit()

        bobbin = self.bobbin_builder.create_simple_bobbin(
            bobbin=magnetic.coil.bobbin,
            material="Plastic"
            if self.solution_type == "SteadyState"
            else "PVC plastic",
        )

        turns_and_terminals = self.coil_builder.create_coil(
            coil=magnetic.coil,
        )

        for core_part in core_parts:
            if bobbin is not None:
                bobbin.subtract(core_part, True)
                for aux in turns_and_terminals:
                    if isinstance(aux, tuple):
                        [turn, terminal] = aux
                    else:
                        turn = aux
                    bobbin.subtract(turn, True)

            for aux in turns_and_terminals:
                if isinstance(aux, tuple):
                    [turn, terminal] = aux
                else:
                    turn = aux
                core_part.subtract(turn, True)

        winding_losses = outputs[
            operating_point_index
        ].windingLosses.windingLossesPerTurn

        if self.solution_type in "SteadyState":
            self.coil_builder.assign_turn_losses_as_heat_source(
                turns_and_terminals, winding_losses
            )

        if self.solution_type in [
            "EddyCurrent",
            "AC Magnetic",
            "Transient",
            "TransientAPhiFormulation",
        ]:
            self.excitation_builder.add_excitation(
                coil=magnetic.coil,
                turns_and_terminals=turns_and_terminals,
                operating_point=inputs.operatingPoints[operating_point_index],
            )

        if self.solution_type in "SteadyState":
            self.create_boundary_conditions(
                inputs.operatingPoints[operating_point_index].conditions
            )
        else:
            self.create_boundary_region(self.padding)

        self.create_setup(
            inputs.operatingPoints[operating_point_index]
            .excitationsPerWinding[0]
            .frequency,
            single_frequency=single_frequency,
        )

        if self.solution_type in "SteadyState":
            self.meshing_backend.set_global_mesh_settings_icepak(meshtype=1)
            # Enable radiation after setup is created
            if hasattr(self, "cooling_builder") and self.cooling_builder:
                self.cooling_builder.enable_radiation()

        # Skip fit() for Icepak to avoid gRPC issues - not essential for simulation
        if self.solution_type not in "SteadyState":
            self.fit()

        if simulate:
            self.analyze()
            self.outputs_extractor.get_results()

        self.save()

    def generate_choke_descriptor(self, magnetic: MAS.Magnetic):

        import json
        from pathlib import Path

        values = {
            "Number of Windings": {"1": False, "2": True, "3": False, "4": False},
            "Layer": {"Simple": False, "Double": True, "Triple": False},
            "Layer Type": {"Separate": False, "Linked": True},
            "Similar Layer": {"Similar": False, "Different": True},
            "Mode": {"Differential": False, "Common": True},
            "Wire Section": {"None": False, "Hexagon": True, "Octagon": False, "Circle": False},
            "Core": {
                "Name": "Core",
                "Material": "ferrite",
                "Inner Radius": magnetic.core.functionalDescription.shape.dimensions["B"]/2,
                "Outer Radius": 30,
                "Height": 10,
                "Chamfer": 0.8,
            },
            "Outer Winding": {
                "Name": "Winding",
                "Material": "copper",
                "Inner Radius": 20,
                "Outer Radius": 30,
                "Height": 10,
                "Wire Diameter": 1.5,
                "Turns": 20,
                "Coil Pit(deg)": 0.1,
                "Occupation(%)": 0,
            },
            "Mid Winding": {"Turns": 25, "Coil Pit(deg)": 0.1, "Occupation(%)": 0},
            "Inner Winding": {"Turns": 4, "Coil Pit(deg)": 0.1, "Occupation(%)": 0},
        }

        # ## Convert dictionary to JSON file
        #
        # Convert the dictionary to a JSON file. You must supply the path of the
        # JSON file as an argument.

        json_path = Path(__file__).parents[2] / "toolkit_choke.json"
        with json_path.open("w") as outfile:
            json.dump(values, outfile)

        # ## Verify parameters of JSON file
        #
        # Verify parameters of the JSON file. The ``check_choke_values()`` method takes
        # the JSON file path as an argument and does the following:
        #
        # - Checks if the JSON file is correctly written (as explained earlier).
        # - Checks equations on windings parameters to avoid having unintended intersections.

        dictionary_values = self.project.modeler.check_choke_values(json_path, create_another_file=False)
        print(dictionary_values)

        # ## Create choke
        #
        # Create the choke. The ``Hfss.modeler.create_choke()`` method takes the JSON file path as an
        # argument.

        list_object = self.project.modeler.create_choke(json_path)
        print(list_object)
        core = list_object[1]
        first_winding_list = list_object[2]
        second_winding_list = list_object[3]

        # ## Create ground

        ground_radius = 1.2 * dictionary_values[1]["Outer Winding"]["Outer Radius"]
        ground_position = [0, 0, first_winding_list[1][0][2] - 2]
        ground = self.project.modeler.create_circle("XY", ground_position, ground_radius, name="GND", material="copper")
        coat = self.project.assign_finite_conductivity(ground, is_infinite_ground=True)
        ground.transparency = 0.9

        # ## Create lumped ports

        port_position_list = [
            [
                first_winding_list[1][0][0],
                first_winding_list[1][0][1],
                first_winding_list[1][0][2] - 1,
            ],
            [
                first_winding_list[1][-1][0],
                first_winding_list[1][-1][1],
                first_winding_list[1][-1][2] - 1,
            ],
            [
                second_winding_list[1][0][0],
                second_winding_list[1][0][1],
                second_winding_list[1][0][2] - 1,
            ],
            [
                second_winding_list[1][-1][0],
                second_winding_list[1][-1][1],
                second_winding_list[1][-1][2] - 1,
            ],
        ]
        port_dimension_list = [2, dictionary_values[1]["Outer Winding"]["Wire Diameter"]]
        for position in port_position_list:
            sheet = self.project.modeler.create_rectangle("XZ", position, port_dimension_list, name="sheet_port")
            sheet.move([-dictionary_values[1]["Outer Winding"]["Wire Diameter"] / 2, 0, -1])
            self.project.lumped_port(
                assignment=sheet.name,
                name="port_" + str(port_position_list.index(position) + 1),
                reference=[ground],
            )

        # ## Create mesh

        # +
        cylinder_height = 2.5 * dictionary_values[1]["Outer Winding"]["Height"]
        cylinder_position = [0, 0, first_winding_list[1][0][2] - 4]
        mesh_operation_cylinder = self.project.modeler.create_cylinder(
            "XY",
            cylinder_position,
            ground_radius,
            cylinder_height,
            num_sides=36,
            name="mesh_cylinder",
        )

        self.project.mesh.assign_length_mesh(
            [mesh_operation_cylinder],
            maximum_length=15,
            maximum_elements=None,
            name="choke_mesh",
        )
        # -

        # ## Create boundaries
        #
        # Create the boundaries. A region with openings is needed to run the analysis.

        region = self.project.modeler.create_region(pad_percent=1000)

        # ## Create setup
        #
        # Create a setup with a sweep to run the simulation. Depending on your machine's
        # computing power, the simulation can take some time to run.

        setup = self.project.create_setup("MySetup")
        setup.props["Frequency"] = "50MHz"
        setup["MaximumPasses"] = 10
        self.project.create_linear_count_sweep(
            setup=setup.name,
            unit="MHz",
            start_frequency=0.1,
            stop_frequency=100,
            num_of_freq_points=100,
            name="sweep1",
            sweep_type="Interpolating",
            save_fields=False,
        )

        # ## Plot objects

        self.project.modeler.fit_all()


