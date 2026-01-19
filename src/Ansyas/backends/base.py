"""
Abstract base classes for backend interfaces.

These define the contract that all backend implementations must follow,
allowing Ansyas to be decoupled from specific FEM software.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import MAS_models as MAS


class Axis(Enum):
    """Axis enumeration for rotation and orientation."""
    X = "x"
    Y = "y"
    Z = "z"


class Plane(Enum):
    """Plane enumeration for orientation."""
    XY = "xy"
    YZ = "yz"
    ZX = "zx"


@dataclass
class GeometryObject:
    """
    Abstract representation of a 3D geometry object.
    
    This is a backend-agnostic wrapper around geometry primitives.
    Each backend maps this to its native object type.
    """
    id: str
    name: str
    native_object: Any  # Backend-specific object
    volume: Optional[float] = None
    faces: Optional[List[Any]] = None
    
    def __repr__(self):
        return f"GeometryObject(id={self.id}, name={self.name})"


@dataclass
class MaterialDefinition:
    """
    Backend-agnostic material definition.
    
    Contains all physical properties needed for simulation.
    """
    name: str
    permeability: Union[float, Dict] = 1.0
    conductivity: Union[float, Dict] = 0.0
    permittivity: float = 1.0
    density: float = 1000.0
    thermal_conductivity: float = 1.0
    specific_heat: float = 1000.0
    
    # Magnetic loss properties
    loss_tangent: Optional[Union[float, Dict]] = None
    steinmetz_coefficients: Optional[Dict] = None
    
    # For litz wire
    is_litz: bool = False
    strand_count: Optional[int] = None
    strand_diameter: Optional[float] = None


@dataclass
class MeshSettings:
    """
    Backend-agnostic mesh settings.
    """
    max_element_size: Optional[float] = None
    min_element_size: Optional[float] = None
    growth_rate: float = 1.5
    curvature_refinement: bool = True
    
    # Skin depth settings for eddy current
    skin_depth: Optional[float] = None
    skin_layers: int = 2


@dataclass
class SolverSetup:
    """
    Backend-agnostic solver setup configuration.
    """
    solver_type: str  # "EddyCurrent", "Transient", "Electrostatic", "Thermal"
    frequency: Optional[float] = None
    max_passes: int = 40
    max_error_percent: float = 3.0
    refinement_percent: float = 30.0
    
    # Transient settings
    stop_time: Optional[float] = None
    time_step: Optional[float] = None
    
    # Frequency sweep settings
    sweep_frequencies: Optional[List[float]] = None


@dataclass
class ExcitationDefinition:
    """
    Backend-agnostic excitation definition.
    """
    name: str
    excitation_type: str  # "Current", "Voltage"
    amplitude: Union[float, str, List[float]]  # Can be formula string
    phase: float = 0.0
    resistance: float = 0.0
    inductance: float = 0.0
    
    # For waveform excitation
    waveform_time: Optional[List[float]] = None
    waveform_data: Optional[List[float]] = None


class GeometryBackend(ABC):
    """
    Abstract interface for 3D geometry creation.
    
    Implementations:
    - AnsysGeometryBackend: Uses PyAEDT modeler
    - CadQueryGeometryBackend: Uses CadQuery for geometry
    """
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the geometry backend."""
        pass
    
    @abstractmethod
    def set_units(self, units: str) -> None:
        """Set the model units (e.g., 'meter', 'mm')."""
        pass
    
    # Primitive creation
    @abstractmethod
    def create_box(
        self,
        origin: List[float],
        sizes: List[float],
        name: str,
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a box primitive."""
        pass
    
    @abstractmethod
    def create_cylinder(
        self,
        axis: Axis,
        origin: List[float],
        radius: float,
        height: float,
        num_sides: int = 0,
        name: str = "cylinder",
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a cylinder primitive."""
        pass
    
    @abstractmethod
    def create_circle(
        self,
        plane: Plane,
        origin: List[float],
        radius: float,
        num_sides: int = 12,
        name: str = "circle",
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a 2D circle (for sweeping)."""
        pass
    
    @abstractmethod
    def create_rectangle(
        self,
        plane: Plane,
        origin: List[float],
        sizes: List[float],
        name: str = "rectangle",
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a 2D rectangle (for sweeping)."""
        pass
    
    # Operations
    @abstractmethod
    def move(self, obj: GeometryObject, vector: List[float]) -> bool:
        """Move an object by a vector."""
        pass
    
    @abstractmethod
    def rotate(self, obj: GeometryObject, axis: Axis, angle: float) -> bool:
        """Rotate an object around an axis."""
        pass
    
    @abstractmethod
    def clone(self, obj: GeometryObject) -> Optional[GeometryObject]:
        """Clone an object."""
        pass
    
    @abstractmethod
    def subtract(self, obj: GeometryObject, tool: GeometryObject, keep_tool: bool = True) -> bool:
        """Subtract tool from obj."""
        pass
    
    @abstractmethod
    def unite(self, objects: List[GeometryObject]) -> GeometryObject:
        """Unite multiple objects into one."""
        pass
    
    @abstractmethod
    def mirror(
        self,
        obj: GeometryObject,
        origin: List[float],
        normal: List[float]
    ) -> GeometryObject:
        """Mirror an object across a plane."""
        pass
    
    @abstractmethod
    def sweep_along_vector(
        self,
        profile: GeometryObject,
        vector: List[float]
    ) -> GeometryObject:
        """Sweep a 2D profile along a vector."""
        pass
    
    @abstractmethod
    def sweep_around_axis(
        self,
        profile: GeometryObject,
        axis: Axis,
        angle: float,
        num_segments: int = 12
    ) -> GeometryObject:
        """Sweep a 2D profile around an axis."""
        pass
    
    # Import/Export
    @abstractmethod
    def import_step(self, file_path: str, healing: bool = True) -> List[GeometryObject]:
        """Import geometry from a STEP file."""
        pass
    
    @abstractmethod
    def export_step(self, objects: List[GeometryObject], file_path: str) -> bool:
        """Export geometry to a STEP file."""
        pass
    
    # Object queries
    @abstractmethod
    def get_objects_by_name(self, pattern: str) -> List[GeometryObject]:
        """Get objects matching a name pattern."""
        pass
    
    @abstractmethod
    def get_object_volume(self, obj: GeometryObject) -> float:
        """Get the volume of an object."""
        pass
    
    @abstractmethod
    def get_object_faces(self, obj: GeometryObject) -> List[Any]:
        """Get the faces of an object."""
        pass
    
    @abstractmethod
    def section(
        self,
        obj: GeometryObject,
        plane: Plane,
        create_new: bool = True
    ) -> Optional[GeometryObject]:
        """Create a cross-section of an object."""
        pass
    
    @abstractmethod
    def create_region(
        self,
        padding_percent: List[float],
        is_percentage: bool = True
    ) -> GeometryObject:
        """Create an air/boundary region around the model."""
        pass
    
    @abstractmethod
    def fit_all(self) -> None:
        """Fit the view to show all objects."""
        pass


class MaterialBackend(ABC):
    """
    Abstract interface for material management.
    
    Handles material creation, assignment, and property definition.
    """
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the material backend."""
        pass
    
    @abstractmethod
    def add_material(self, definition: MaterialDefinition) -> str:
        """Add a new material and return its ID/name."""
        pass
    
    @abstractmethod
    def assign_material(self, obj: GeometryObject, material_name: str) -> bool:
        """Assign a material to a geometry object."""
        pass
    
    @abstractmethod
    def create_dataset(
        self,
        name: str,
        x_values: List[float],
        y_values: List[float],
        x_unit: str = "",
        y_unit: str = ""
    ) -> Any:
        """Create a dataset for frequency/temperature dependent properties."""
        pass
    
    @abstractmethod
    def get_material_property(self, material_name: str, property_name: str) -> Any:
        """Get a material property value."""
        pass


class MeshingBackend(ABC):
    """
    Abstract interface for mesh generation.
    
    Implementations:
    - AnsysMeshingBackend: Uses PyAEDT meshing
    - GmshMeshingBackend: Uses Gmsh for meshing
    """
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the meshing backend."""
        pass
    
    @abstractmethod
    def set_global_settings(self, settings: MeshSettings) -> None:
        """Set global mesh settings."""
        pass
    
    @abstractmethod
    def assign_mesh_size(
        self,
        objects: List[GeometryObject],
        max_size: float,
        min_size: Optional[float] = None
    ) -> bool:
        """Assign mesh size to specific objects."""
        pass
    
    @abstractmethod
    def assign_skin_depth(
        self,
        faces: List[Any],
        skin_depth: float,
        num_layers: int = 2
    ) -> bool:
        """Assign skin depth mesh refinement for eddy current simulations."""
        pass
    
    @abstractmethod
    def assign_curvature_refinement(
        self,
        objects: List[GeometryObject],
        num_elements_per_curvature: int = 6
    ) -> bool:
        """Assign curvature-based mesh refinement."""
        pass
    
    @abstractmethod
    def generate_mesh(self) -> bool:
        """Generate the mesh."""
        pass
    
    @abstractmethod
    def export_mesh(self, file_path: str, format: str = "msh") -> bool:
        """Export mesh to file (for use with external solvers)."""
        pass
    
    @abstractmethod
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics (element count, quality, etc.)."""
        pass


class ExcitationBackend(ABC):
    """
    Abstract interface for excitation and boundary condition setup.
    
    Handles winding definitions, current/voltage sources, and boundaries.
    """
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the excitation backend."""
        pass
    
    @abstractmethod
    def create_winding(
        self,
        name: str,
        winding_type: str,  # "Current" or "Voltage"
        amplitude: Union[float, str],
        is_solid: bool = True,
        resistance: float = 0.0,
        inductance: float = 0.0,
        phase: float = 0.0
    ) -> Any:
        """Create a winding excitation."""
        pass
    
    @abstractmethod
    def assign_coil(
        self,
        face: Any,
        conductors_number: int = 1,
        polarity: str = "Positive",
        name: str = "coil"
    ) -> str:
        """Assign a coil terminal to a face."""
        pass
    
    @abstractmethod
    def add_coils_to_winding(
        self,
        winding_name: str,
        coil_names: List[str]
    ) -> bool:
        """Add coil terminals to a winding."""
        pass
    
    @abstractmethod
    def create_matrix(
        self,
        excitations: List[str],
        name: str
    ) -> Any:
        """Create an excitation matrix for multi-winding setups."""
        pass
    
    @abstractmethod
    def join_parallel(
        self,
        matrix: Any,
        sources: List[str],
        matrix_name: str,
        join_name: str
    ) -> bool:
        """Join parallel windings in a matrix."""
        pass
    
    @abstractmethod
    def create_waveform_dataset(
        self,
        name: str,
        time: List[float],
        data: List[float]
    ) -> Any:
        """Create a waveform dataset for transient excitation."""
        pass
    
    @abstractmethod
    def assign_floating(
        self,
        objects: List[GeometryObject],
        charge: float = 0.0,
        name: str = "floating"
    ) -> bool:
        """Assign floating boundary condition (for electrostatics)."""
        pass
    
    @abstractmethod
    def assign_heat_source(
        self,
        objects: List[GeometryObject],
        power: float,
        name: str = "heat_source"
    ) -> bool:
        """Assign heat source (for thermal simulations)."""
        pass
    
    @abstractmethod
    def set_core_losses(
        self,
        objects: List[GeometryObject],
        on_field: bool = True
    ) -> bool:
        """Enable core loss calculation for magnetic cores."""
        pass


class SolverBackend(ABC):
    """
    Abstract interface for FEM solving.
    
    Implementations:
    - AnsysSolverBackend: Uses PyAEDT Maxwell/Icepak
    - MFEMSolverBackend: Uses MFEM for solving
    """
    
    @abstractmethod
    def initialize(self, solver_type: str, **kwargs) -> None:
        """
        Initialize the solver backend.
        
        Args:
            solver_type: Type of simulation ("EddyCurrent", "Transient", 
                        "Electrostatic", "Thermal", etc.)
        """
        pass
    
    @abstractmethod
    def create_setup(self, setup: SolverSetup) -> Any:
        """Create a solver setup with the given configuration."""
        pass
    
    @abstractmethod
    def add_frequency_sweep(
        self,
        setup: Any,
        start_freq: float,
        stop_freq: float,
        step_size: float,
        sweep_type: str = "LinearStep"
    ) -> bool:
        """Add a frequency sweep to the setup."""
        pass
    
    @abstractmethod
    def analyze(self) -> bool:
        """Run the simulation."""
        pass
    
    @abstractmethod
    def get_solution_data(
        self,
        expressions: List[str],
        context: Optional[Dict] = None
    ) -> Any:
        """Get solution data for post-processing."""
        pass
    
    @abstractmethod
    def get_impedance_matrix(self) -> Dict:
        """Get impedance matrix results."""
        pass
    
    @abstractmethod
    def get_inductance_matrix(self) -> Dict:
        """Get inductance matrix results."""
        pass
    
    @abstractmethod
    def get_resistance_matrix(self) -> Dict:
        """Get resistance matrix results."""
        pass
    
    @abstractmethod
    def get_field_data(
        self,
        field_type: str,  # "B", "H", "J", "E", etc.
        objects: Optional[List[GeometryObject]] = None
    ) -> Any:
        """Get field data for visualization or export."""
        pass
    
    @abstractmethod
    def export_results(self, file_path: str, format: str = "csv") -> bool:
        """Export simulation results to file."""
        pass
    
    @abstractmethod
    def save_project(self, file_path: Optional[str] = None) -> bool:
        """Save the project/simulation."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the solver and release resources."""
        pass


class BackendRegistry:
    """
    Registry for backend implementations.
    
    Allows registration and retrieval of different backend implementations.
    """
    
    _geometry_backends: Dict[str, type] = {}
    _material_backends: Dict[str, type] = {}
    _meshing_backends: Dict[str, type] = {}
    _excitation_backends: Dict[str, type] = {}
    _solver_backends: Dict[str, type] = {}
    
    @classmethod
    def register_geometry(cls, name: str, backend_class: type) -> None:
        """Register a geometry backend."""
        if not issubclass(backend_class, GeometryBackend):
            raise TypeError(f"{backend_class} must be a subclass of GeometryBackend")
        cls._geometry_backends[name] = backend_class
    
    @classmethod
    def register_material(cls, name: str, backend_class: type) -> None:
        """Register a material backend."""
        if not issubclass(backend_class, MaterialBackend):
            raise TypeError(f"{backend_class} must be a subclass of MaterialBackend")
        cls._material_backends[name] = backend_class
    
    @classmethod
    def register_meshing(cls, name: str, backend_class: type) -> None:
        """Register a meshing backend."""
        if not issubclass(backend_class, MeshingBackend):
            raise TypeError(f"{backend_class} must be a subclass of MeshingBackend")
        cls._meshing_backends[name] = backend_class
    
    @classmethod
    def register_excitation(cls, name: str, backend_class: type) -> None:
        """Register an excitation backend."""
        if not issubclass(backend_class, ExcitationBackend):
            raise TypeError(f"{backend_class} must be a subclass of ExcitationBackend")
        cls._excitation_backends[name] = backend_class
    
    @classmethod
    def register_solver(cls, name: str, backend_class: type) -> None:
        """Register a solver backend."""
        if not issubclass(backend_class, SolverBackend):
            raise TypeError(f"{backend_class} must be a subclass of SolverBackend")
        cls._solver_backends[name] = backend_class
    
    @classmethod
    def get_geometry(cls, name: str) -> type:
        """Get a geometry backend by name."""
        if name not in cls._geometry_backends:
            raise KeyError(f"Geometry backend '{name}' not found. Available: {list(cls._geometry_backends.keys())}")
        return cls._geometry_backends[name]
    
    @classmethod
    def get_material(cls, name: str) -> type:
        """Get a material backend by name."""
        if name not in cls._material_backends:
            raise KeyError(f"Material backend '{name}' not found. Available: {list(cls._material_backends.keys())}")
        return cls._material_backends[name]
    
    @classmethod
    def get_meshing(cls, name: str) -> type:
        """Get a meshing backend by name."""
        if name not in cls._meshing_backends:
            raise KeyError(f"Meshing backend '{name}' not found. Available: {list(cls._meshing_backends.keys())}")
        return cls._meshing_backends[name]
    
    @classmethod
    def get_excitation(cls, name: str) -> type:
        """Get an excitation backend by name."""
        if name not in cls._excitation_backends:
            raise KeyError(f"Excitation backend '{name}' not found. Available: {list(cls._excitation_backends.keys())}")
        return cls._excitation_backends[name]
    
    @classmethod
    def get_solver(cls, name: str) -> type:
        """Get a solver backend by name."""
        if name not in cls._solver_backends:
            raise KeyError(f"Solver backend '{name}' not found. Available: {list(cls._solver_backends.keys())}")
        return cls._solver_backends[name]
    
    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """List all available backends."""
        return {
            "geometry": list(cls._geometry_backends.keys()),
            "material": list(cls._material_backends.keys()),
            "meshing": list(cls._meshing_backends.keys()),
            "excitation": list(cls._excitation_backends.keys()),
            "solver": list(cls._solver_backends.keys()),
        }
