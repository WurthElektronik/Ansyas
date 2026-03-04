"""
Integration tests for Ansys Maxwell simulations.

These tests require an Ansys license and will launch AEDT to run
actual FEM simulations. They are marked with @pytest.mark.integration
and @pytest.mark.ansys so they can be skipped during quick test runs.

To run only integration tests:
    pytest tests/test_integration_ansys.py -v -m "integration"

To skip integration tests:
    pytest tests/ -v -m "not integration"
"""

import unittest
import os
import json
import glob
import time
import shutil
import subprocess
import pytest

import context  # noqa: F401
import mas_autocomplete
from ansyas import Ansyas


def is_ansys_available():
    """Check if Ansys AEDT is available."""
    try:
        from ansys.aedt.core import Maxwell3d
        return True
    except ImportError:
        return False


def kill_aedt_processes(wait_for_exit=True, max_wait=30):
    """Force kill any lingering AEDT processes.
    
    Parameters
    ----------
    wait_for_exit : bool
        If True, wait until processes are actually terminated.
    max_wait : int
        Maximum seconds to wait for processes to terminate.
    """
    # Kill all AEDT-related processes
    processes_to_kill = [
        "ansysedtsv",
        "ansysedt", 
        "ansysedt.exe",
        "ANS.AEDT.*",
        "grpc_aedt_server"
    ]
    for proc in processes_to_kill:
        subprocess.run(
            ["powershell", "-Command", f"Get-Process -Name '{proc}' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"],
            capture_output=True
        )
    # Also clean up any orphaned gRPC ports and Ansys processes
    subprocess.run(
        ["powershell", "-Command", "Get-Process | Where-Object {$_.ProcessName -like '*aedt*' -or $_.ProcessName -like '*ansys*'} | Stop-Process -Force -ErrorAction SilentlyContinue"],
        capture_output=True
    )
    
    if wait_for_exit:
        # Wait for all AEDT-related processes to actually terminate
        start_time = time.time()
        while (time.time() - start_time) < max_wait:
            result = subprocess.run(
                ["powershell", "-Command", 
                 "Get-Process | Where-Object {$_.ProcessName -like '*aedt*' -or $_.ProcessName -like '*ansys*'} | Measure-Object | Select-Object -ExpandProperty Count"],
                capture_output=True,
                text=True
            )
            try:
                count = int(result.stdout.strip())
                if count == 0:
                    break
            except (ValueError, AttributeError):
                break
            time.sleep(1)
        # Extra wait after processes are gone
        time.sleep(2)


def clean_pyaedt_temp_files():
    """Clean up PyAEDT temporary files that may cause connection issues."""
    import tempfile
    temp_dir = tempfile.gettempdir()
    
    # Clean up .aedt_pipe* files
    patterns = ['aedt_pipe*', 'pyaedt*', 'aedt_port*']
    for pattern in patterns:
        for file in glob.glob(os.path.join(temp_dir, pattern)):
            try:
                os.remove(file)
            except (PermissionError, OSError):
                pass


# Skip all tests in this module if Ansys is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.ansys,
    pytest.mark.skipif(not is_ansys_available(), reason="Ansys AEDT not available")
]


class TestAnsysIntegration(unittest.TestCase):
    """Integration tests for Ansys Maxwell simulations.
    
    All tests are in a single class to ensure sequential execution
    and proper cleanup between tests.
    """
    
    output_path = f'{os.path.dirname(os.path.abspath(__file__))}/../output/'
    mas_files_path = f'{os.path.dirname(os.path.abspath(__file__))}/../examples/'
    external_data_path = f'{os.path.dirname(os.path.abspath(__file__))}/../external_data/'

    @classmethod
    def setUpClass(cls):
        """Clean output directory and kill any lingering AEDT processes."""
        # Kill any existing AEDT processes and wait for full cleanup
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        
        # Clear any lingering PyAEDT sessions
        try:
            from ansys.aedt.core.internal.desktop_sessions import _desktop_sessions, _edb_sessions
            _desktop_sessions.clear()
            _edb_sessions.clear()
        except Exception:
            pass
        
        # Clean output directory
        os.makedirs(cls.output_path, exist_ok=True)
        files = glob.glob(f"{cls.output_path}/*")
        for f in files:
            if os.path.isdir(f):
                shutil.rmtree(f, ignore_errors=True)
            else:
                try:
                    os.remove(f)
                except PermissionError:
                    pass
        print("\n=== Starting Ansys Integration Tests ===")

    @classmethod
    def tearDownClass(cls):
        """Final cleanup after all tests - close the desktop."""
        # Try to properly release any desktop sessions
        try:
            from ansys.aedt.core.internal.desktop_sessions import _desktop_sessions, _edb_sessions
            for session_key in list(_desktop_sessions.keys()):
                try:
                    desktop = _desktop_sessions[session_key]
                    if hasattr(desktop, 'release_desktop'):
                        desktop.release_desktop(close_projects=True, close_desktop=True)
                except Exception:
                    pass
            _desktop_sessions.clear()
            _edb_sessions.clear()
        except Exception:
            pass
        
        # Force kill all AEDT processes
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        print("\n=== Finished Ansys Integration Tests ===")

    def setUp(self):
        """Setup for each test - prepare for new test with fresh state."""
        self._project = None
        self._ansyas = None
        
        # Ensure no AEDT processes are running before starting a new test
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        
        # Clean up temp files that may cause connection issues
        clean_pyaedt_temp_files()
        
        # Wait for system to fully settle after killing processes
        time.sleep(10)
        
        # Completely unload and reimport PyAEDT modules to get fresh state
        # This is necessary because PyAEDT keeps internal module-level state
        import sys
        modules_to_remove = [key for key in sys.modules.keys() if 'ansys.aedt' in key or 'pyaedt' in key]
        for module_name in modules_to_remove:
            try:
                del sys.modules[module_name]
            except KeyError:
                pass
        
        # Also remove ansyas module which imports PyAEDT
        if 'ansyas' in sys.modules:
            del sys.modules['ansyas']
        if 'Ansyas' in sys.modules:
            del sys.modules['Ansyas']
        if 'ansyas.Ansyas' in sys.modules:
            del sys.modules['ansyas.Ansyas']
        
        # Force garbage collection
        import gc
        gc.collect()

    def tearDown(self):
        """Ensure project and desktop are fully released between tests."""
        # Step 1: Release desktop if we have a project
        if self._project is not None:
            try:
                # Close everything - projects AND desktop
                self._project.release_desktop(close_projects=True, close_desktop=True)
            except Exception:
                pass
        
        # Step 2: Reset internal state 
        self._ansyas = None
        self._project = None
        
        # Step 3: Clear PyAEDT internal session cache BEFORE killing processes
        try:
            from ansys.aedt.core.internal.desktop_sessions import _desktop_sessions, _edb_sessions
            _desktop_sessions.clear()
            _edb_sessions.clear()
        except Exception:
            pass
        
        # Step 4: Clear any PyAEDT global state
        try:
            import ansys.aedt.core.generic.settings as settings
            if hasattr(settings, 'settings'):
                settings.settings.enable_logger = True  # Reset logger state
        except Exception:
            pass
        
        # Step 5: Force garbage collection
        import gc
        gc.collect()
        
        # Step 6: Wait for desktop to close gracefully before killing
        time.sleep(5)
        
        # Step 7: Kill any lingering AEDT processes and wait for full exit
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        
        # Step 8: Extra wait for system cleanup after process termination
        time.sleep(5)
        
        # Step 9: Clear PyAEDT sessions again after process kill
        try:
            from ansys.aedt.core.internal.desktop_sessions import _desktop_sessions, _edb_sessions
            _desktop_sessions.clear()
            _edb_sessions.clear()
        except Exception:
            pass
        
        # Final garbage collection
        gc.collect()

    def _run_eddy_current_simulation(self, mas_file: str, project_name: str, non_graphical: bool = True):
        """Helper to run an EddyCurrent simulation and return results."""
        # Reimport modules that may have been unloaded in setUp
        from ansyas import Ansyas as AnsyasClass
        import mas_autocomplete as mas_auto
        
        with open(os.path.join(self.mas_files_path, mas_file), 'r') as f:
            mas_dict = json.load(f)
            mas = mas_auto.autocomplete(mas_dict)

        self._ansyas = AnsyasClass(
            number_segments_arcs=12,
            initial_mesh_configuration=2,
            maximum_error_percent=5,
            refinement_percent=30,
            maximum_passes=15,
            scale=1
        )

        self._project = self._ansyas.create_project(
            outputs_folder=self.output_path,
            project_name=f"{project_name}_{int(time.time())}",
            non_graphical=non_graphical,
            solution_type="EddyCurrent",
            new_desktop_session=True  # Fresh desktop for each test with full cleanup
        )

        self._ansyas.set_units("meter")
        self._ansyas.create_magnetic_simulation(mas=mas, single_frequency=True)
        self._ansyas.analyze()

        return self._ansyas.outputs_extractor.get_results()

    # =========================================================================
    # Example Tests - Using actual example files
    # =========================================================================

    def test_01_concentric_transformer(self):
        """Test concentric transformer simulation."""
        mas_file = "concentric_transformer.json"
        mas_file_path = os.path.join(self.mas_files_path, mas_file)
        if not os.path.exists(mas_file_path):
            self.skipTest(f"{mas_file} not found")

        impedance = self._run_eddy_current_simulation(
            mas_file=mas_file,
            project_name=f"test_concentric_transformer_{int(time.time())}"
        )
        
        # Verify we got valid results
        self.assertIsNotNone(impedance)
        self.assertIsNotNone(impedance.inductanceMatrix)
        self.assertGreater(len(impedance.inductanceMatrix), 0, "Simulation produced no inductance data")

        magnetizing_inductance = impedance.inductanceMatrix[0].magnitude[0][0].nominal
        print(f"Concentric transformer magnetizing inductance: {magnetizing_inductance:.3e} H")

    def test_02_concentric_flyback_rectangular_column(self):
        """Test concentric flyback with rectangular column simulation."""
        mas_file = "concentric_flyback_rectangular_column.json"
        mas_file_path = os.path.join(self.mas_files_path, mas_file)
        if not os.path.exists(mas_file_path):
            self.skipTest(f"{mas_file} not found")
            
        impedance = self._run_eddy_current_simulation(
            mas_file=mas_file,
            project_name=f"test_concentric_flyback_{int(time.time())}"
        )

        magnetizing_inductance = impedance.inductanceMatrix[0].magnitude[0][0].nominal
        print(f"Concentric flyback magnetizing inductance: {magnetizing_inductance:.3e} H")
        
        # Verify we got valid results
        self.assertIsNotNone(impedance)
        self.assertIsNotNone(impedance.inductanceMatrix)
        self.assertGreater(len(impedance.inductanceMatrix), 0)

    def test_03_concentric_transformer_contiguous_rectangular_wire(self):
        """Test concentric transformer with contiguous rectangular wire simulation."""
        mas_file = "concentric_transformer_contiguous_rectangular_wire.json"
        mas_file_path = os.path.join(self.mas_files_path, mas_file)
        if not os.path.exists(mas_file_path):
            self.skipTest(f"{mas_file} not found")
            
        impedance = self._run_eddy_current_simulation(
            mas_file=mas_file,
            project_name=f"test_concentric_rect_wire_{int(time.time())}"
        )

        magnetizing_inductance = impedance.inductanceMatrix[0].magnitude[0][0].nominal
        print(f"Concentric transformer (rectangular wire) magnetizing inductance: {magnetizing_inductance:.3e} H")
        
        # Verify we got valid results
        self.assertIsNotNone(impedance)
        self.assertIsNotNone(impedance.inductanceMatrix)
        self.assertGreater(len(impedance.inductanceMatrix), 0)


if __name__ == '__main__':
    # Run with pytest for better output, sequential execution
    pytest.main([__file__, '-v', '-s', '-x'])
