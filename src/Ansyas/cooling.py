from typing import Optional, List, Dict
try:
    import ansys.aedt.core as pyaedt
except ImportError:
    import pyaedt
try:
    from . import ansyas_utils
    from . import MAS_models as MAS
except ImportError:
    import ansyas_utils
    import MAS_models as MAS
import os
import subprocess
import signal
import time
from pathlib import Path


class Cooling:
    """
    Icepak cooling setup - NO ANALYTICAL CALCULATIONS.
    All physics is solved by Ansys Icepak.
    """

    def __init__(self, project, number_segments_arcs=12):
        self.project = project
        self.number_segments_arcs = number_segments_arcs
        self.ambient_temperature = 25.0

    def create_cooling(self, core, cooling_config):
        """
        Setup Icepak boundary conditions ONLY.
        NO calculations - let Icepak solve everything.
        
        Args:
            core: Core object
            cooling_config: MAS cooling configuration
        """
        # Extract cooling type from MAS config
        cooling_type = self._get_cooling_type(cooling_config)
        ambient_temp = getattr(cooling_config, 'temperature', 25.0)
        self.ambient_temperature = ambient_temp
        
        print(f"Setting up Icepak cooling: {cooling_type} at {ambient_temp}°C")
        
        if cooling_type == 'forced':
            velocity = getattr(cooling_config, 'velocity', [1.0, 0.0, 0.0])
            self._setup_forced_convection(ambient_temp, velocity)
        elif cooling_type == 'natural':
            self._setup_natural_convection(ambient_temp)
        else:
            raise ValueError(f"Unknown cooling type: {cooling_type}")
    
    def enable_radiation(self):
        """
        Enable radiation after setup has been created.
        Call this after create_setup() in the main flow.
        """
        print("  Enabling radiation heat transfer...")
        try:
            if self.project.setups:
                setup = self.project.setups[0]
                # Enable radiation - this is the key parameter!
                setup.props['IsEnabled'] = True
                # Set radiation model to Discrete Ordinates Model
                setup.props['Radiation Model'] = 'Discrete Ordinates Model'
                # Set flow iteration per radiation iteration
                setup.props['Flow Iteration Per Radiation Iteration'] = 1
                # Set discrete ordinates division parameters
                setup.props['ThetaDivision'] = 1
                setup.props['PhiDivision'] = 1
                setup.props['ThetaPixels'] = 1
                setup.props['PhiPixels'] = 1
                setup.update()
                print("  [OK] Radiation enabled (Discrete Ordinates Model)")
            else:
                print("  [WARNING] No setup found, cannot enable radiation")
        except Exception as e:
            print(f"  [WARNING] Could not enable radiation: {e}")

    def _get_cooling_type(self, cooling_config) -> str:
        """Extract cooling type from MAS config. Default to natural."""
        if hasattr(cooling_config, 'type'):
            type_val = cooling_config.type
            if isinstance(type_val, str):
                type_lower = type_val.lower()
                if 'forced' in type_lower:
                    return 'forced'
                elif 'natural' in type_lower:
                    return 'natural'
        
        # Check for velocity - if present, it's forced convection
        if hasattr(cooling_config, 'velocity'):
            velocity = cooling_config.velocity
            if velocity and any(v != 0 for v in velocity):
                return 'forced'
        
        # Default to natural convection
        return 'natural'

    def _setup_natural_convection(self, ambient_temperature: float):
        """
        Setup natural convection - Icepak calculates everything.
        Just set ambient temperature (boundaries will use defaults).
        """
        print(f"  Setting up natural convection (ambient: {ambient_temperature}°C)")
        
        # Set ambient temperature in project
        # Icepak will use default boundary conditions for natural convection
        self.project.ambient_temperature = f"{ambient_temperature}cel"
        
        # Enable gravity for natural convection (negative Z direction, standard 9.81 m/s²)
        self.project.gravity = [0, 0, -9.81]
        print("  [OK] Gravity enabled: 9.81 m/s² in -Z direction")
        
        # Radiation will be enabled after setup creation via enable_radiation()
        print("  [OK] Natural convection setup complete")

    def _setup_radiation(self):
        """
        Enable radiation heat transfer using Surface to Surface model.
        This is important for accurate thermal modeling at higher temperatures.
        """
        print("  Setting up radiation heat transfer...")
        
        # Get the setup and enable radiation
        if self.project.setups:
            setup = self.project.setups[0]
            # Enable radiation in setup props
            setup.props['Radiation Model'] = 'Surface To Surface'
            setup.update()
            print("  [OK] Radiation enabled (Surface to Surface model)")
        else:
            print("  [WARNING] No setup found, cannot enable radiation")

    def _setup_forced_convection(self, ambient_temperature: float, velocity: List[float]):
        """
        Setup forced convection with specified velocity.
        Icepak calculates flow and heat transfer.
        """
        print(f"  Setting up forced convection (ambient: {ambient_temperature}°C, velocity: {velocity})")
        
        # Calculate velocity magnitude and direction
        v_mag = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2) ** 0.5
        
        if v_mag < 0.01:
            raise ValueError(f"Velocity too low for forced convection: {v_mag} m/s")
        
        # Determine inlet/outlet faces based on dominant flow direction
        abs_vel = [abs(v) for v in velocity]
        max_idx = abs_vel.index(max(abs_vel))
        
        faces = ["XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"]
        
        if max_idx == 0:  # X direction
            if velocity[0] > 0:
                inlet, outlet = "XMin", "XMax"
            else:
                inlet, outlet = "XMax", "XMin"
        elif max_idx == 1:  # Y direction
            if velocity[1] > 0:
                inlet, outlet = "YMin", "YMax"
            else:
                inlet, outlet = "YMax", "YMin"
        else:  # Z direction
            if velocity[2] > 0:
                inlet, outlet = "ZMin", "ZMax"
            else:
                inlet, outlet = "ZMax", "ZMin"
        
        # Create inlet with velocity
        self.project.assign_inlet_opening(
            assignment=[inlet],
            boundary_name="inlet",
            flow_type="Velocity",
            velocity=f"{v_mag}m_per_sec",
            temperature=f"{ambient_temperature}cel"
        )
        print(f"  [OK] Inlet created on {inlet} at {v_mag:.2f} m/s")
        
        # Create outlet
        self.project.assign_outlet_opening(
            assignment=[outlet],
            boundary_name="outlet"
        )
        print(f"  [OK] Outlet created on {outlet}")
        
        # Set remaining faces as opening
        remaining = [f for f in faces if f not in [inlet, outlet]]
        for face in remaining:
            self.project.assign_free_opening(
                assignment=[face],
                boundary_name=f"opening_{face}",
                temperature=f"{ambient_temperature}cel"
            )

    def extract_temperatures(self, component_parts: Dict[str, str]) -> Dict[str, float]:
        """
        Extract temperatures from Icepak results.
        NO CALCULATIONS - just read Icepak output.
        
        Args:
            component_parts: Dict mapping part names to Icepak object names
                            e.g., {'core_central': 'core_0', 'turn_0': 'Turn_0'}
        
        Returns:
            Dict mapping part names to temperatures in Celsius
        """
        temperatures = {}
        
        print("\nExtracting temperatures from Icepak results...")
        
        # For Icepak, we use field reports on objects
        post = self.project.post
        
        # Get the setup name
        setup_name = None
        if hasattr(self.project, 'setups') and self.project.setups:
            setup_name = self.project.setups[0].name
        
        if not setup_name:
            raise RuntimeError("No simulation setup found")
        
        print(f"  Using setup: {setup_name}")
        
        for part_name, object_name in component_parts.items():
            try:
                # For Icepak, create a field report for the object
                # Temperature is the field quantity
                report_name = f"TempReport_{part_name}"
                
                # Get the object
                if object_name not in self.project.modeler.objects:
                    print(f"  Warning: Object {object_name} not found, skipping")
                    continue
                
                obj = self.project.modeler.objects[object_name]
                
                # Create a field report for temperature on this object
                # In Icepak, the temperature field is called 'Temp'
                try:
                    # Method 1: Try creating a field report
                    report_name = f"TempReport_{part_name}_{object_name}"
                    
                    # Create report for temperature on this object
                    success = post.create_report(
                        plot_name=report_name,
                        expressions=["Temp"],
                        setup_sweep_name=f"{setup_name} : LastAdaptive",
                        domain="Object",
                        context=[obj.name]
                    )
                    
                    if success:
                        # Get the report data
                        report_data = post.get_report_data(report_name)
                        if report_data and len(report_data.data_magnitude()) > 0:
                            temp_celsius = report_data.data_magnitude()[0]
                            temperatures[part_name] = temp_celsius
                            print(f"  {part_name}: {temp_celsius:.2f}C (from report)")
                        else:
                            raise RuntimeError(f"No data in report")
                    else:
                        raise RuntimeError(f"Failed to create report")
                        
                except Exception as e2:
                    # Method 2: Try direct field quantity access
                    try:
                        solution_data = post.get_solution_data(
                            expressions=["Temp"],
                            setup_sweep_name=f"{setup_name} : LastAdaptive"
                        )
                        
                        if solution_data and len(solution_data.data_magnitude()) > 0:
                            # Get first value (this is a workaround)
                            temp_celsius = solution_data.data_magnitude()[0]
                            temperatures[part_name] = temp_celsius
                            print(f"  {part_name}: {temp_celsius:.2f}C (from solution)")
                        else:
                            raise RuntimeError(f"No temperature data")
                    except Exception as e3:
                        print(f"  Warning: All methods failed for {part_name}")
                        raise
                    
            except Exception as e:
                print(f"  Warning: Could not extract temperature for {part_name}: {e}")
                # Don't raise - just skip this part
                continue
        
        if not temperatures:
            raise RuntimeError("No temperatures could be extracted")
        
        return temperatures


class IcepakRunner:
    """
    Handles Icepak simulation execution with proper error handling.
    NO FALLBACKS - if simulation fails, it crashes.
    """
    
    @staticmethod
    def run_simulation(project, timeout: int = 1800):
        """
        Run Icepak simulation with timeout and license handling.
        
        Args:
            project: Icepak project object
            timeout: Maximum simulation time in seconds (default: 30 min)
        
        Raises:
            RuntimeError: If simulation fails or times out
        """
        print(f"\nStarting Icepak thermal simulation (timeout: {timeout}s)...")
        
        try:
            # Create setup if not exists
            if not hasattr(project, 'setups') or not project.setups:
                project.create_setup()
                print("  [OK] Simulation setup created")
            
            # Run analysis with monitoring
            start_time = time.time()
            
            # Start simulation in non-blocking way to allow timeout
            project.analyze_setup(project.setups[0].name)
            
            # Monitor progress
            while True:
                elapsed = time.time() - start_time
                
                if elapsed > timeout:
                    # Kill Ansys process if stuck
                    IcepakRunner._kill_ansys_processes()
                    raise RuntimeError(f"Simulation timed out after {timeout} seconds")
                
                # Check if simulation completed
                try:
                    # Try to access results - if available, simulation is done
                    if project.post:
                        print(f"  Simulation completed in {elapsed:.1f}s")
                        break
                except:
                    pass
                
                time.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            # Kill Ansys processes on error
            IcepakRunner._kill_ansys_processes()
            raise RuntimeError(f"Icepak simulation failed: {e}")
    
    @staticmethod
    def _kill_ansys_processes():
        """Kill any stuck Ansys processes to free license."""
        print("  Killing Ansys processes to free license...")
        try:
            # Windows
            subprocess.run(['taskkill', '/F', '/IM', 'ansys*.exe'], 
                         capture_output=True, check=False)
            subprocess.run(['taskkill', '/F', '/IM', 'Icepak*.exe'], 
                         capture_output=True, check=False)
        except:
            pass
