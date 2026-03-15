"""
Ansyas - Decoupled FEM Simulation Framework for Magnetics.

This module provides the main Ansyas class that orchestrates FEM simulations
using pluggable backends for geometry, meshing, solving, and excitation.

The architecture allows swapping backends:
- Geometry: Ansys (PyAEDT), CadQuery, etc.
- Meshing: Ansys, Gmsh, etc.
- Solver: Ansys Maxwell/Icepak, Elmer, MFEM, etc.
"""

import os
import logging
import tempfile
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

# Backend availability flags
HAS_ANSYS = False
HAS_ELMER = False

try:
    from ansys.aedt.core import Maxwell3d, Icepak
    HAS_ANSYS = True
except ImportError:
    pass

try:
    from pyelmer import elmer
    import gmsh
    import cadquery
    HAS_ELMER = True
except ImportError:
    pass

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
        
        # Elmer-specific state
        self._sim_dir = None
        self._elmer_postprocessor = None
        self._is_elmer = False

    def _initialize_backends(self, project=None, sim_dir: str = None):
        """
        Initialize backends with the created project or simulation directory.
        
        For Ansys backends, `project` is the PyAEDT project instance.
        For Elmer backends, `sim_dir` is the simulation directory path.
        """
        backend_name = self._solver_backend_name.lower()
        
        if backend_name == "elmer":
            self._initialize_elmer_backends(sim_dir)
        else:
            self._initialize_ansys_backends(project)
    
    def _initialize_ansys_backends(self, project):
        """Initialize Ansys backends with the PyAEDT project."""
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
    
    def _initialize_elmer_backends(self, sim_dir: str):
        """Initialize Elmer backends for open-source FEM simulation."""
        try:
            from .backends.elmer import (
                ElmerGeometryBackend,
                ElmerMaterialBackend,
                ElmerMeshingBackend,
                ElmerExcitationBackend,
                ElmerSolverBackend,
                ElmerPostprocessor,
            )
        except ImportError:
            from backends.elmer import (
                ElmerGeometryBackend,
                ElmerMaterialBackend,
                ElmerMeshingBackend,
                ElmerExcitationBackend,
                ElmerSolverBackend,
                ElmerPostprocessor,
            )
        
        if sim_dir is None:
            sim_dir = self._sim_dir
        
        # Initialize geometry backend (CadQuery + gmsh)
        self.geometry_backend = ElmerGeometryBackend()
        self.geometry_backend.initialize(temp_dir=sim_dir)
        
        # Initialize meshing backend (gmsh + ElmerGrid)
        self.meshing_backend = ElmerMeshingBackend()
        self.meshing_backend.initialize(sim_dir=sim_dir)
        
        # Solver and material/excitation backends need the pyelmer Simulation object
        # These will be fully initialized when create_elmer_simulation() is called
        self.solver_backend = ElmerSolverBackend()
        self.solver_backend.initialize(
            solver_type=self.solution_type or "EddyCurrent",
            sim_dir=sim_dir
        )
        
        # Get simulation object for material and excitation backends
        simulation = self.solver_backend.get_simulation()
        
        self.material_backend = ElmerMaterialBackend()
        self.material_backend.initialize(simulation=simulation)
        
        self.excitation_backend = ElmerExcitationBackend()
        self.excitation_backend.initialize(simulation=simulation)
        
        # Store postprocessor for result extraction
        self._elmer_postprocessor = ElmerPostprocessor(sim_dir=sim_dir)

    def fit(self):
        """Fit the view to show all objects."""
        if self.geometry_backend:
            self.geometry_backend.fit_all()
        elif self.project:
            self.project.modeler.fit_all()

    def analyze(self):
        """Run the simulation."""
        if self.solver_backend:
            result = self.solver_backend.analyze()
            
            # For Elmer, load results after analysis
            if self._is_elmer and self._elmer_postprocessor:
                self._elmer_postprocessor.load_results()
            
            return result
        elif self.project:
            self.project.analyze()

    def save(self):
        """Save the project."""
        if self._is_elmer:
            # For Elmer, write the SIF file
            if self.solver_backend:
                self.solver_backend.save_project()
            return
        
        if self.solver_backend:
            self.solver_backend.save_project()
        elif self.project:
            self.project.oeditor.CleanUpModel()
            self.project.save_project()
    
    def get_elmer_results(
        self, 
        winding_currents: List[float] = None,
        frequency: float = 100000
    ) -> Dict[str, Any]:
        """
        Get results from Elmer simulation.
        
        Parameters
        ----------
        winding_currents : List[float], optional
            List of currents for each winding (A). Required for
            inductance/resistance calculation.
        frequency : float, optional
            Simulation frequency (Hz). Default is 100kHz.
        
        Returns
        -------
        Dict
            Dictionary containing simulation results:
            - inductance_matrix: Self/mutual inductance values
            - resistance_matrix: AC resistance values
            - magnetic_energy: Stored magnetic energy (J)
            - joule_losses: Winding losses (W)
        """
        if not self._is_elmer:
            raise RuntimeError("get_elmer_results() is only available for Elmer simulations")
        
        if self._elmer_postprocessor is None:
            raise RuntimeError("Postprocessor not initialized. Run analyze() first.")
        
        results = {}
        
        # Get basic field quantities
        try:
            magnetic_energy = self._elmer_postprocessor.calculate_magnetic_energy()
            results["magnetic_energy"] = magnetic_energy
        except Exception as e:
            logger.warning(f"Could not calculate magnetic energy: {e}")
        
        try:
            joule_losses = self._elmer_postprocessor.calculate_joule_losses()
            results["joule_losses"] = joule_losses
        except Exception as e:
            logger.warning(f"Could not calculate Joule losses: {e}")
        
        # Calculate inductance and resistance if currents are provided
        if winding_currents:
            try:
                l_matrix = self._elmer_postprocessor.calculate_inductance_matrix(winding_currents)
                results["inductance_matrix"] = l_matrix
            except Exception as e:
                logger.warning(f"Could not calculate inductance matrix: {e}")
            
            try:
                r_matrix = self._elmer_postprocessor.calculate_resistance_matrix(winding_currents)
                results["resistance_matrix"] = r_matrix
            except Exception as e:
                logger.warning(f"Could not calculate resistance matrix: {e}")
            
            # Single winding convenience values
            if len(winding_currents) == 1:
                try:
                    results["inductance"] = self._elmer_postprocessor.calculate_inductance(
                        winding_currents[0]
                    )
                    results["resistance"] = self._elmer_postprocessor.calculate_resistance(
                        winding_currents[0]
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate single-winding L/R: {e}")
        
        return results
    
    def get_elmer_impedance_output(
        self,
        frequencies: List[float],
        currents: List[float]
    ) -> Any:
        """
        Get full MAS-format impedance output.
        
        Parameters
        ----------
        frequencies : List[float]
            List of frequencies (Hz).
        currents : List[float]
            List of winding currents (A).
            
        Returns
        -------
        MAS.ImpedanceOutput
            Full impedance output in MAS format.
        """
        if not self._is_elmer:
            raise RuntimeError("Only available for Elmer simulations")
        
        if self._elmer_postprocessor is None:
            raise RuntimeError("Postprocessor not initialized")
        
        return self._elmer_postprocessor.get_impedance_output(frequencies, currents)

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
        from ansys.aedt.core import Maxwell3d, Icepak

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

    def create_elmer_project(
        self,
        outputs_folder: str,
        project_name: str,
        solution_type: str = "EddyCurrent",
    ):
        """
        Create an Elmer simulation project (open-source alternative to Ansys).
        
        This method sets up the simulation directory and initializes Elmer backends
        without requiring an Ansys license.
        
        Parameters
        ----------
        outputs_folder : str
            Path to store the output files.
        project_name : str
            Name of the project (used for directory naming).
        solution_type : str, optional
            Solution type. Supported: "EddyCurrent", "AC Magnetic", "Magnetostatic",
            "Transient", "TransientAPhiFormulation", "SteadyState", "Thermal",
            "Electrostatic". Default is "EddyCurrent".
            
        Returns
        -------
        str
            Path to the simulation directory.
        """
        if not HAS_ELMER:
            raise ImportError(
                "Elmer dependencies not found. Install with: "
                "pip install ansyas[elmer] or pip install pyelmer gmsh cadquery"
            )
        
        # Sanitize project name
        project_name = (
            project_name.replace(" ", "_")
            .replace(",", "_")
            .replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        
        # Create simulation directory
        self.project_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), outputs_folder
        )
        os.makedirs(self.project_path, exist_ok=True)
        
        self._sim_dir = os.path.join(self.project_path, project_name)
        os.makedirs(self._sim_dir, exist_ok=True)
        
        self.project_name = project_name
        self.solution_type = solution_type
        self._is_elmer = True
        
        # Initialize Elmer backends
        self._initialize_backends(sim_dir=self._sim_dir)
        
        logger.info(f"Created Elmer project in: {self._sim_dir}")
        return self._sim_dir

    def get_project_location(self) -> str:
        """Get the project file location."""
        if self._is_elmer:
            return self._sim_dir
        return self.project_name

    def create_builders(self, magnetic: MAS.Magnetic):
        """
        Create domain-specific builders for core, coil, etc.
        
        Note: For Elmer backend, the existing builders (Core, Coil, etc.) are
        Ansys-specific. Use create_elmer_magnetic_simulation() for Elmer or
        build geometry directly with the geometry_backend.
        """
        if self._is_elmer:
            logger.warning(
                "Domain builders (Core, Coil, Bobbin) are Ansys-specific. "
                "For Elmer simulations, use create_elmer_magnetic_simulation() "
                "or build geometry directly with geometry_backend."
            )
            return
        
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

    def create_elmer_magnetic_simulation(
        self,
        mas: Union[MAS.Mas, dict],
        simulate: bool = False,
        operating_point_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Create an Elmer-based magnetic simulation from MAS data.
        
        This method provides a simplified workflow for Elmer simulations,
        building geometry using CadQuery and gmsh rather than the Ansys-specific
        domain builders.
        
        Parameters
        ----------
        mas : MAS.Mas or dict
            MAS file containing magnetic component description.
        simulate : bool, optional
            Run simulation immediately if True. Default is False.
        operating_point_index : int, optional
            Index of operating point to simulate. Default is 0.
            
        Returns
        -------
        Dict
            Simulation results if simulate=True, otherwise empty dict.
        """
        if not self._is_elmer:
            raise RuntimeError(
                "create_elmer_magnetic_simulation() requires Elmer backend. "
                "Use geometry_backend='elmer', solver_backend='elmer' in constructor "
                "and call create_elmer_project() instead of create_project()."
            )
        
        if isinstance(mas, dict):
            mas = MAS.Mas.from_dict(mas)
        
        magnetic = mas.magnetic
        inputs = mas.inputs
        operating_point = inputs.operatingPoints[operating_point_index]
        
        # Get frequency and excitation
        excitation = operating_point.excitationsPerWinding[0]
        frequency = excitation.frequency
        current = excitation.current.processed.rms if hasattr(excitation.current, 'processed') else 1.0
        
        # TODO: Build geometry from MAS using geometry_backend
        # For now, we provide a template for manual geometry creation
        # Full implementation would import core/coil shapes using OpenMagneticsVirtualBuilder
        
        logger.info(f"Elmer simulation setup for {magnetic.manufacturer_info.name if hasattr(magnetic, 'manufacturer_info') else 'component'}")
        logger.info(f"Frequency: {frequency} Hz, Current: {current} A")
        
        # Configure solver for this frequency
        setup_config = SolverSetup(
            solver_type=self.solution_type,
            frequency=frequency,
            max_passes=self.maximum_passes,
            max_error_percent=self.maximum_error_percent,
            refinement_percent=self.refinement_percent,
        )
        self.solver_backend.create_setup(setup_config)
        
        results = {}
        
        if simulate:
            self.analyze()
            
            # Get winding currents for result extraction
            winding_currents = []
            for exc in operating_point.excitationsPerWinding:
                if hasattr(exc.current, 'processed'):
                    winding_currents.append(exc.current.processed.rms)
                else:
                    winding_currents.append(1.0)
            
            results = self.get_elmer_results(
                winding_currents=winding_currents,
                frequency=frequency
            )
        
        self.save()
        return results
    
    def generate_elmer_mesh(self) -> bool:
        """
        Generate mesh for Elmer simulation using gmsh.
        
        Call this after creating geometry to generate the mesh
        and convert it to Elmer format.
        
        Returns
        -------
        bool
            True if mesh generation succeeded.
        """
        if not self._is_elmer:
            raise RuntimeError("generate_elmer_mesh() requires Elmer backend")
        
        if self.meshing_backend is None:
            raise RuntimeError("Meshing backend not initialized")
        
        # Generate mesh
        success = self.meshing_backend.generate_mesh()
        
        if success:
            # Convert to Elmer format
            self.meshing_backend.convert_to_elmer()
        
        return success
    
    def visualize_elmer_results(
        self,
        field: str = "magnetic flux density",
        save_path: str = None
    ) -> Any:
        """
        Visualize Elmer simulation results using PyVista.
        
        Parameters
        ----------
        field : str, optional
            Field to visualize. Options: "magnetic flux density", 
            "current density", "magnetic vector potential", "temperature".
            Default is "magnetic flux density".
        save_path : str, optional
            Path to save screenshot. If None, displays interactively.
            
        Returns
        -------
        pyvista.Plotter or None
            PyVista plotter object if available, otherwise None.
        """
        if not self._is_elmer:
            raise RuntimeError("visualize_elmer_results() requires Elmer backend")
        
        if self._elmer_postprocessor is None:
            raise RuntimeError("Postprocessor not initialized")
        
        return self._elmer_postprocessor.visualize_field(
            field_name=field,
            save_path=save_path
        )
