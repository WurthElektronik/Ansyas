"""
Test that runs all example files in the examples/ directory.
Verifies that simulations complete without errors for each example.

These tests require an Ansys license and are marked with:
- @pytest.mark.ansys - Requires Ansys AEDT
- @pytest.mark.integration - Integration tests  
- @pytest.mark.slow - Slow-running tests

To run these tests:
    pytest tests/test_all_examples.py -v --run-ansys

To skip these tests:
    pytest tests/ -m "not ansys"
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
    """Force kill any lingering AEDT processes."""
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
    subprocess.run(
        ["powershell", "-Command", "Get-Process | Where-Object {$_.ProcessName -like '*aedt*' -or $_.ProcessName -like '*ansys*'} | Stop-Process -Force -ErrorAction SilentlyContinue"],
        capture_output=True
    )
    
    if wait_for_exit:
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
        time.sleep(2)


def clean_pyaedt_temp_files():
    """Clean up PyAEDT temporary files that may cause connection issues."""
    import tempfile
    temp_dir = tempfile.gettempdir()
    patterns = ['aedt_pipe*', 'pyaedt*', 'aedt_port*']
    for pattern in patterns:
        for file in glob.glob(os.path.join(temp_dir, pattern)):
            try:
                os.remove(file)
            except (PermissionError, OSError):
                pass


def get_example_files():
    """Get all example JSON files from the examples directory."""
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    example_files = []
    
    for file_path in glob.glob(os.path.join(examples_dir, 'example*.json')):
        # Skip non-MAS files or metadata files
        filename = os.path.basename(file_path)
        if filename.endswith('.json') and not filename.startswith('.'):
            example_files.append((filename, file_path))
    
    return sorted(example_files)


# Skip all tests if Ansys is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.ansys,
    pytest.mark.slow,
    pytest.mark.skipif(not is_ansys_available(), reason="Ansys AEDT not available")
]


class TestAllExamples(unittest.TestCase):
    """Test all example files in the examples/ directory."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
        cls.output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'examples_test')
        
        # Clean and create output directory
        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path, ignore_errors=True)
        os.makedirs(cls.output_path, exist_ok=True)
        
        # Kill any existing AEDT processes
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        
        # Note: PyMKF has built-in materials, loading external materials
        # overwrites them and can break compatibility
        
        print("\n" + "="*70)
        print("RUNNING ALL EXAMPLES TESTS")
        print("="*70)
        print()
        
        cls.example_files = get_example_files()
        print(f"Found {len(cls.example_files)} example files to test:")
        for filename, _ in cls.example_files:
            print(f"  - {filename}")
        print()

    @classmethod
    def tearDownClass(cls):
        """Final cleanup."""
        # Clean up PyAEDT sessions
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
        
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        print("\n" + "="*70)
        print("FINISHED ALL EXAMPLES TESTS")
        print("="*70 + "\n")

    def setUp(self):
        """Setup for each test."""
        self._project = None
        self._ansyas = None
        
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        clean_pyaedt_temp_files()
        time.sleep(5)
        
        # Clear PyAEDT modules
        import sys
        modules_to_remove = [key for key in sys.modules.keys() if 'ansys.aedt' in key or 'pyaedt' in key]
        for module_name in modules_to_remove:
            try:
                del sys.modules[module_name]
            except KeyError:
                pass
        
        if 'ansyas' in sys.modules:
            del sys.modules['ansyas']
        if 'Ansyas' in sys.modules:
            del sys.modules['Ansyas']
        
        import gc
        gc.collect()

    def tearDown(self):
        """Cleanup after each test."""
        if self._project is not None:
            try:
                self._project.release_desktop(close_projects=True, close_desktop=True)
            except Exception:
                pass
        
        self._ansyas = None
        self._project = None
        
        try:
            from ansys.aedt.core.internal.desktop_sessions import _desktop_sessions, _edb_sessions
            _desktop_sessions.clear()
            _edb_sessions.clear()
        except Exception:
            pass
        
        import gc
        gc.collect()
        time.sleep(3)
        kill_aedt_processes(wait_for_exit=True, max_wait=30)
        gc.collect()

    def _run_example_simulation(self, example_file: str, file_path: str):
        """Run a single example simulation and verify it completes."""
        print(f"\n{'='*70}")
        print(f"Testing: {example_file}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Load example file
        with open(file_path, 'r') as f:
            mas_dict = json.load(f)
        
        # Process with autocomplete to validate structure
        try:
            mas = mas_autocomplete.autocomplete(mas_dict)
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ {example_file} - AUTOCOMPLETE FAILED after {elapsed_time:.1f}s")
            print(f"   Error: {e}")
            raise
        
        # Create Ansyas instance with test-optimized settings
        self._ansyas = Ansyas(
            number_segments_arcs=12,
            initial_mesh_configuration=2,
            maximum_error_percent=5,
            refinement_percent=30,
            maximum_passes=15,
            scale=1
        )
        
        # Create project
        project_name = f"example_{os.path.splitext(example_file)[0]}_{int(time.time())}"
        self._project = self._ansyas.create_project(
            outputs_folder=self.output_path,
            project_name=project_name,
            non_graphical=True,
            solution_type="EddyCurrent",
            new_desktop_session=True
        )
        
        try:
            # Run simulation using the current API
            self._ansyas.set_units("meter")
            self._ansyas.create_magnetic_simulation(mas=mas, single_frequency=True)
            self._ansyas.analyze()
            
            # Get results
            results = self._ansyas.outputs_extractor.get_results()
            
            # Verify results exist
            self.assertIsNotNone(results, "Results should not be None")
            has_impedance = hasattr(results, 'impedanceMatrix') and results.impedanceMatrix is not None
            has_inductance = hasattr(results, 'inductanceMatrix') and results.inductanceMatrix is not None
            self.assertTrue(has_impedance or has_inductance, 
                          "Results should have either impedanceMatrix or inductanceMatrix")
            
            elapsed_time = time.time() - start_time
            print(f"✅ {example_file} - PASSED (took {elapsed_time:.1f}s)")
            return True
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ {example_file} - FAILED after {elapsed_time:.1f}s: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def test_concentric_transformer(self):
        """Test concentric transformer example."""
        file_path = os.path.join(self.examples_dir, 'concentric_transformer.json')
        if os.path.exists(file_path):
            self._run_example_simulation('concentric_transformer.json', file_path)
        else:
            self.skipTest("concentric_transformer.json not found")

    def test_concentric_flyback_rectangular_column(self):
        """Test concentric flyback with rectangular column example."""
        file_path = os.path.join(self.examples_dir, 'concentric_flyback_rectangular_column.json')
        if os.path.exists(file_path):
            self._run_example_simulation('concentric_flyback_rectangular_column.json', file_path)
        else:
            self.skipTest("concentric_flyback_rectangular_column.json not found")

    def test_concentric_transformer_contiguous_rectangular_wire(self):
        """Test concentric transformer with contiguous rectangular wire example."""
        file_path = os.path.join(self.examples_dir, 'concentric_transformer_contiguous_rectangular_wire.json')
        if os.path.exists(file_path):
            self._run_example_simulation('concentric_transformer_contiguous_rectangular_wire.json', file_path)
        else:
            self.skipTest("concentric_transformer_contiguous_rectangular_wire.json not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
