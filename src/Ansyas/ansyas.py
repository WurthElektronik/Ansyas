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
from typing import Optional, List, Dict, Any, Union

import MAS_models as MAS
import ansyas_utils

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
        scale: float = 1
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
        specified_version: str = None
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
        from ansys.aedt.core import Maxwell3d, Icepak
        
        project_name = f"{project_name}.aedt"
        self.project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), outputs_folder)

        os.makedirs(self.project_path, exist_ok=True)
        self.solution_type = solution_type
        self.project_name = project_name
        self.project_name = self.project_name.replace(" ", "_").replace(",", "_").replace(":", "_").replace("/", "_").replace("\\", "_")

        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)

        self.project_name = f"{self.project_path}/{self.project_name}"

        if self.solution_type == "SteadyState":
            self.project = Icepak(
                project=self.project_name,
                solution_type="SteadyState",
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
                close_on_exit=False
            )
            
            # Verify the Maxwell3d object is properly initialized
            if not hasattr(self.project, '_odesign') or self.project._odesign is None:
                raise RuntimeError(
                    "Failed to initialize Maxwell3d project. AEDT connection may have failed. "
                    "Ensure no other AEDT instances are running and try again."
                )
            
            self.project.change_design_settings({"ComputeTransientCapacitance": True})
            self.project.change_design_settings({"ComputeTransientInductance": True})
            if hasattr(self.project, '_odesign') and self.project._odesign is not None:
                self.project.mesh.assign_initial_mesh_from_slider(
                    self.initial_mesh_configuration,
                    curvilinear=True
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
            project=self.project,
            number_segments_arcs=self.number_segments_arcs
        )

        if self.solution_type == "Electrostatic":
            if magnetic.core.functionalDescription.shape.family is MAS.CoreShapeFamily.t:
                self.coil_builder = coil_builder.ToroidalCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=True
                )
            else:
                self.coil_builder = coil_builder.ConcentricSpiralCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=True
                )
        else:
            if magnetic.core.functionalDescription.shape.family is MAS.CoreShapeFamily.t:
                self.coil_builder = coil_builder.ToroidalCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=False
                )
            else:
                self.coil_builder = coil_builder.ConcentricCoil(
                    project=self.project,
                    number_segments_arcs=self.number_segments_arcs,
                    add_insulation=False
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
                return self.geometry_backend.create_air_region({
                    "x_pos": 50, "y_pos": 50, "z_pos": 50,
                    "x_neg": 50, "y_neg": 50, "z_neg": 50
                })
            else:
                region = self.project.modeler.create_air_region(
                    x_pos=50, y_pos=50, z_pos=50,
                    x_neg=50, y_neg=50, z_neg=50
                )
                if region is None or region is False:
                    region = self.project.modeler.get_objects_w_string("Region")
                    region = self.project.modeler.get_object_from_name(region[0])
                return region
        else:
            padding_list = [
                padding["x_pos"], padding["x_neg"],
                padding["y_pos"], padding["y_neg"],
                padding["z_pos"], padding["z_neg"]
            ]
            if self.geometry_backend:
                return self.geometry_backend.create_region(padding_list, is_percentage=True)
            else:
                region = self.project.modeler.create_region(
                    pad_percent=padding_list,
                    is_percentage=True
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
                faces=[int(str(x)) for x in faces],
                name="AirOpening"
            )
        else:
            velocity = ansyas_utils.convert_axis(conditions.cooling.velocity)
            velocity_strs = [
                f"{velocity[0]}m_per_sec",
                f"{velocity[1]}m_per_sec",
                f"{velocity[2]}m_per_sec"
            ]
            temperature = (
                "AmbientTemp" if conditions.cooling.temperature is None
                else conditions.cooling.temperature
            )
            
            self.excitation_backend.assign_free_opening(
                faces=region.bottom_face_x,
                flow_type="Pressure",
                velocity=velocity_strs,
                temperature=temperature,
                name="AirOpening"
            )

    def create_setup(self, frequency: float = 100000):
        """Create solver setup with frequency sweeps."""
        setup_config = SolverSetup(
            solver_type=self.solution_type,
            frequency=frequency,
            max_passes=self.maximum_passes,
            max_error_percent=self.maximum_error_percent,
            refinement_percent=self.refinement_percent
        )
        
        if self.solution_type in ["Transient", "TransientAPhiFormulation"]:
            setup_config.stop_time = 2 / frequency
            setup_config.time_step = 2 / frequency / 10
        
        setup = self.solver_backend.create_setup(setup_config)
        
        if self.solution_type in ["EddyCurrent", "AC Magnetic"]:
            self.solver_backend.add_default_frequency_sweeps(setup)
        
        return setup

    def add_skin_effect(
        self,
        wires_faces,
        skin_depth: float = 0.0002,
        number_layers: int = 2
    ):
        """Add skin effect mesh refinement to wire faces."""
        if not isinstance(wires_faces, list):
            wires_faces = [wires_faces]
        
        self.meshing_backend.assign_skin_depth(
            faces=wires_faces,
            skin_depth=skin_depth,
            num_layers=number_layers
        )

    def create_magnetic_simulation(
        self,
        mas: Union[MAS.Mas, dict],
        simulate: bool = False,
        operating_point_index: int = 0
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
                inputs.operatingPoints[operating_point_index].conditions.cooling
            )

        self.fit()

        bobbin = self.bobbin_builder.create_simple_bobbin(
            bobbin=magnetic.coil.bobbin,
            material="Plastic" if self.solution_type == "SteadyState" else "PVC plastic",
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

        winding_losses = outputs[operating_point_index].windingLosses.windingLossesPerTurn

        if self.solution_type in "SteadyState":
            self.coil_builder.assign_turn_losses_as_heat_source(
                turns_and_terminals,
                winding_losses
            )

        if self.solution_type in ["EddyCurrent", "AC Magnetic", "Transient", "TransientAPhiFormulation"]:
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
            inputs.operatingPoints[operating_point_index].excitationsPerWinding[0].frequency
        )
        
        if self.solution_type in "SteadyState":
            self.meshing_backend.set_global_mesh_settings_icepak(meshtype=1)

        self.fit()

        if simulate:
            self.analyze()
            self.outputs_extractor.get_results()

        self.save()
