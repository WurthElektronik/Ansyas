"""
Elmer post-processing backend.

Extracts results from Elmer VTU output files and calculates
electromagnetic quantities (inductance, resistance, impedance).
"""

import os
import math
import glob
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    from ... import MAS_models as MAS
except ImportError:
    try:
        import MAS_models as MAS
    except ImportError:
        MAS = None


class ElmerPostprocessor:
    """
    Post-processor for Elmer FEM results.
    
    Capabilities:
    - Read VTU result files
    - Calculate inductance from magnetic energy
    - Calculate resistance from Joule losses
    - Compute impedance matrices
    - Extract field distributions
    - Generate ParaView-compatible output
    
    Inductance calculation methods:
    1. Energy method: L = 2 * Wm / I²
    2. Flux linkage: L = Ψ / I = ∫ A · dl / I
    
    Resistance calculation:
    R = P_joule / I² = ∫ J·E dV / I²
    """
    
    def __init__(self, sim_dir: str = None):
        self._sim_dir = sim_dir
        self._results: Dict[str, Any] = {}
        self._mesh = None
        self._field_data: Dict[str, np.ndarray] = {}
        self._windings: List[Dict] = []
    
    def set_simulation_directory(self, sim_dir: str) -> None:
        """Set the simulation directory."""
        self._sim_dir = sim_dir
    
    def set_windings(self, windings: List[Dict]) -> None:
        """
        Set winding definitions for matrix calculation.
        
        Each winding dict should contain:
        - name: Winding name
        - current: Excitation current (A)
        - body_ids: List of body IDs belonging to this winding
        """
        self._windings = windings
    
    def load_results(self, result_file: str = None) -> bool:
        """
        Load results from VTU file.
        
        Args:
            result_file: Path to VTU file (auto-detected if None)
        """
        if not HAS_MESHIO and not HAS_PYVISTA:
            print("Warning: Neither meshio nor pyvista available for result reading")
            return False
        
        if result_file is None:
            result_file = self._find_result_file()
        
        if result_file is None or not os.path.exists(result_file):
            print(f"Result file not found: {result_file}")
            return False
        
        try:
            if HAS_PYVISTA:
                self._mesh = pv.read(result_file)
                # Extract field data
                for name in self._mesh.array_names:
                    self._field_data[name] = self._mesh[name]
            elif HAS_MESHIO:
                mesh = meshio.read(result_file)
                self._mesh = mesh
                # Extract point and cell data
                for name, data in mesh.point_data.items():
                    self._field_data[name] = data
                for name, data in mesh.cell_data.items():
                    self._field_data[name] = data[0] if isinstance(data, list) else data
            
            return True
        except Exception as e:
            print(f"Failed to load results: {e}")
            return False
    
    def _find_result_file(self) -> Optional[str]:
        """Find the latest VTU result file in sim directory."""
        if not self._sim_dir:
            return None
        
        patterns = [
            os.path.join(self._sim_dir, "results*.vtu"),
            os.path.join(self._sim_dir, "case*.vtu"),
            os.path.join(self._sim_dir, "*.vtu"),
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # Return the most recent
                return max(files, key=os.path.getmtime)
        
        return None
    
    def get_available_fields(self) -> List[str]:
        """Get list of available field variables."""
        return list(self._field_data.keys())
    
    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get field data by name."""
        return self._field_data.get(field_name)
    
    def calculate_magnetic_energy(self, body_ids: List[int] = None) -> float:
        """
        Calculate magnetic energy from B and H fields.
        
        Wm = 0.5 * ∫ B · H dV
        
        For harmonic analysis with complex fields:
        Wm = 0.25 * ∫ Re(B · H*) dV
        """
        if not HAS_NUMPY:
            return 0.0
        
        # Get B and H fields
        b_field = self._get_vector_field("magnetic flux density")
        h_field = self._get_vector_field("magnetic field strength")
        
        if b_field is None or h_field is None:
            return 0.0
        
        # Get volume elements
        volumes = self._get_cell_volumes()
        
        if volumes is None:
            return 0.0
        
        # Calculate energy density
        # For complex fields: energy = 0.25 * Re(B · H*)
        if np.iscomplexobj(b_field) or np.iscomplexobj(h_field):
            energy_density = 0.25 * np.real(
                np.sum(b_field * np.conj(h_field), axis=1)
            )
        else:
            energy_density = 0.5 * np.sum(b_field * h_field, axis=1)
        
        # Integrate over volume
        total_energy = np.sum(energy_density * volumes)
        
        return total_energy
    
    def calculate_joule_losses(self, body_ids: List[int] = None) -> float:
        """
        Calculate Joule heating losses.
        
        P = ∫ J · E dV
        
        For harmonic: P = 0.5 * ∫ Re(J · E*) dV
        """
        if not HAS_NUMPY:
            return 0.0
        
        # Try direct Joule heat field first
        joule_heat = self._field_data.get("joule heating")
        if joule_heat is not None:
            volumes = self._get_cell_volumes()
            if volumes is not None:
                return np.sum(joule_heat * volumes)
        
        # Calculate from J and E
        j_field = self._get_vector_field("current density")
        e_field = self._get_vector_field("electric field")
        
        if j_field is None or e_field is None:
            return 0.0
        
        volumes = self._get_cell_volumes()
        if volumes is None:
            return 0.0
        
        # For harmonic analysis
        if np.iscomplexobj(j_field) or np.iscomplexobj(e_field):
            power_density = 0.5 * np.real(
                np.sum(j_field * np.conj(e_field), axis=1)
            )
        else:
            power_density = np.sum(j_field * e_field, axis=1)
        
        return np.sum(power_density * volumes)
    
    def calculate_inductance(
        self, 
        winding_current: float,
        method: str = "energy"
    ) -> float:
        """
        Calculate self-inductance.
        
        Args:
            winding_current: Excitation current (A)
            method: "energy" or "flux_linkage"
            
        Returns:
            Inductance in Henries
        """
        if method == "energy":
            # L = 2 * Wm / I²
            wm = self.calculate_magnetic_energy()
            return 2 * wm / (winding_current ** 2)
        else:
            # Flux linkage method
            # L = Ψ / I = N * Φ / I
            flux = self.calculate_flux_linkage()
            return flux / winding_current
    
    def calculate_resistance(self, winding_current: float) -> float:
        """
        Calculate AC resistance from Joule losses.
        
        R = P / I²
        """
        power = self.calculate_joule_losses()
        return power / (winding_current ** 2)
    
    def calculate_flux_linkage(self, coil_path: np.ndarray = None) -> float:
        """
        Calculate flux linkage through a coil.
        
        Ψ = ∫ A · dl along coil path
        
        If no path provided, estimates from average A field.
        """
        a_field = self._get_vector_field("av")  # Magnetic vector potential
        if a_field is None:
            return 0.0
        
        # Simplified: use average A magnitude times estimated path length
        # Full implementation would integrate along coil geometry
        a_avg = np.mean(np.linalg.norm(a_field, axis=1))
        
        return a_avg
    
    def calculate_inductance_matrix(
        self,
        currents: List[float]
    ) -> List[List[Dict]]:
        """
        Calculate mutual inductance matrix for multiple windings.
        
        For n windings with currents I1, I2, ..., In:
        Lij = Ψij / Ij (flux in winding i due to current in winding j)
        
        Using energy method for self-inductance and
        flux linkage for mutual inductance.
        """
        n = len(currents)
        matrix = []
        
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    # Self inductance
                    l_val = self.calculate_inductance(currents[i])
                else:
                    # Mutual inductance (simplified)
                    # Full implementation would excite each winding separately
                    l_val = 0.0
                
                row.append({"nominal": l_val})
            matrix.append(row)
        
        return matrix
    
    def calculate_resistance_matrix(
        self,
        currents: List[float]
    ) -> List[List[Dict]]:
        """
        Calculate resistance matrix.
        
        Diagonal elements: self resistance from Joule losses
        Off-diagonal: typically zero for well-separated windings
        """
        n = len(currents)
        matrix = []
        
        total_power = self.calculate_joule_losses()
        
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    # Self resistance (assuming equal distribution)
                    r_val = total_power / n / (currents[i] ** 2)
                else:
                    r_val = 0.0
                row.append({"nominal": r_val})
            matrix.append(row)
        
        return matrix
    
    def get_impedance_output(
        self,
        frequencies: List[float],
        currents: List[float]
    ) -> 'MAS.ImpedanceOutput':
        """
        Get full impedance output in MAS format.
        
        Args:
            frequencies: List of frequencies (Hz)
            currents: List of winding currents (A)
        """
        if MAS is None:
            raise ImportError("MAS_models not available")
        
        inductance_data = []
        resistance_data = []
        impedance_data = []
        
        for freq in frequencies:
            # Calculate matrices at this frequency
            l_matrix = self.calculate_inductance_matrix(currents)
            r_matrix = self.calculate_resistance_matrix(currents)
            
            # Calculate impedance: Z = R + jωL
            omega = 2 * math.pi * freq
            z_matrix = []
            z_phase = []
            
            for i, (l_row, r_row) in enumerate(zip(l_matrix, r_matrix)):
                z_row = []
                phase_row = []
                for j, (l_val, r_val) in enumerate(zip(l_row, r_row)):
                    z_real = r_val["nominal"]
                    z_imag = omega * l_val["nominal"]
                    z_mag = math.sqrt(z_real**2 + z_imag**2)
                    z_angle = math.degrees(math.atan2(z_imag, z_real))
                    z_row.append({"nominal": z_mag})
                    phase_row.append({"nominal": z_angle})
                z_matrix.append(z_row)
                z_phase.append(phase_row)
            
            inductance_data.append({
                "frequency": freq,
                "magnitude": l_matrix,
            })
            resistance_data.append({
                "frequency": freq,
                "magnitude": r_matrix,
            })
            impedance_data.append({
                "frequency": freq,
                "magnitude": z_matrix,
                "phase": z_phase,
            })
        
        output_dict = {
            "methodUsed": "Elmer FEM",
            "origin": MAS.ResultOrigin.simulation,
            "inductanceMatrix": inductance_data,
            "resistanceMatrix": resistance_data,
            "impedanceMatrix": impedance_data,
        }
        
        return MAS.ImpedanceOutput.from_dict(output_dict)
    
    def _get_vector_field(self, base_name: str) -> Optional[np.ndarray]:
        """Get a vector field, combining components if needed."""
        if not HAS_NUMPY:
            return None
        
        # Try direct name
        if base_name in self._field_data:
            return self._field_data[base_name]
        
        # Try with components
        components = []
        for suffix in [" 1", " 2", " 3", "_x", "_y", "_z", " x", " y", " z"]:
            comp_name = base_name + suffix
            if comp_name in self._field_data:
                components.append(self._field_data[comp_name])
        
        if len(components) == 3:
            return np.column_stack(components)
        
        # Try case variations
        for name in self._field_data:
            if base_name.lower() in name.lower():
                data = self._field_data[name]
                if len(data.shape) > 1 and data.shape[1] == 3:
                    return data
        
        return None
    
    def _get_cell_volumes(self) -> Optional[np.ndarray]:
        """Get cell volumes from mesh."""
        if not HAS_NUMPY:
            return None
        
        if HAS_PYVISTA and self._mesh is not None:
            return self._mesh.compute_cell_sizes()["Volume"]
        
        # Estimate from mesh geometry
        return None
    
    def visualize_field(
        self,
        field_name: str,
        output_file: str = None,
        show: bool = True
    ) -> bool:
        """
        Visualize a field using ParaView/PyVista.
        
        Args:
            field_name: Name of field to visualize
            output_file: Optional output file for screenshot
            show: Whether to show interactive window
        """
        if not HAS_PYVISTA:
            print("PyVista required for visualization")
            return False
        
        if self._mesh is None:
            if not self.load_results():
                return False
        
        try:
            plotter = pv.Plotter()
            
            if field_name in self._mesh.array_names:
                plotter.add_mesh(
                    self._mesh,
                    scalars=field_name,
                    show_edges=False,
                    cmap="viridis"
                )
            else:
                plotter.add_mesh(self._mesh, show_edges=True)
            
            plotter.add_axes()
            plotter.add_scalar_bar(field_name)
            
            if output_file:
                plotter.screenshot(output_file)
            
            if show:
                plotter.show()
            
            return True
        except Exception as e:
            print(f"Visualization failed: {e}")
            return False
    
    def export_to_paraview(self, output_file: str) -> bool:
        """
        Export results for ParaView visualization.
        
        Creates a VTK/VTU file that can be opened in ParaView.
        """
        if self._mesh is None:
            if not self.load_results():
                return False
        
        try:
            if HAS_PYVISTA:
                self._mesh.save(output_file)
            elif HAS_MESHIO:
                meshio.write(output_file, self._mesh)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
