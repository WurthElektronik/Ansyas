"""
Comprehensive test that runs all MAS examples.
Verifies that simulations complete without errors.
"""
import unittest
import os
import json
import time
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


@pytest.mark.ansys
@pytest.mark.integration
@pytest.mark.slow
class TestAllMasExamples(unittest.TestCase):
    """Test all MAS example files run successfully."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.mas_files_path = os.path.join(os.path.dirname(__file__), 'mas_files')
        cls.external_data_path = os.path.join(os.path.dirname(__file__), '..', 'external_data')
        cls.output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(cls.output_path, exist_ok=True)
        
        # Load core materials once
        material_file_path = os.path.join(cls.external_data_path, 'core_materials.ndjson')
        if os.path.exists(material_file_path):
            with open(material_file_path, 'r') as material_file:
                material_data = material_file.read()
                from PyOpenMagnetics import PyOpenMagnetics as PyOM
                PyOM.load_core_materials(material_data)
    
    def _run_simulation(self, mas_file: str, project_name: str):
        """Run a simulation and verify it completes."""
        print(f"\n{'='*60}")
        print(f"Testing: {mas_file}")
        print(f"{'='*60}")
        
        mas_path = os.path.join(self.mas_files_path, mas_file)
        if not os.path.exists(mas_path):
            self.skipTest(f"{mas_file} not found")
        
        # Load and process MAS
        with open(mas_path, 'r') as f:
            mas_dict = json.load(f)
        
        # Replace unavailable materials with available ones
        available_materials = ['WE_A', 'WE_B', 'WE_C', 'WE_D', 'WE_E', 'WE_F']
        if 'magnetic' in mas_dict and 'core' in mas_dict['magnetic']:
            core = mas_dict['magnetic']['core']
            if isinstance(core, dict) and 'functionalDescription' in core:
                func_desc = core['functionalDescription']
                if isinstance(func_desc, list) and len(func_desc) > 0:
                    material = func_desc[0].get('material', '')
                    if material and material not in available_materials:
                        print(f"  Replacing material {material} with WE_A")
                        func_desc[0]['material'] = 'WE_A'
                elif isinstance(func_desc, dict):
                    material = func_desc.get('material', '')
                    if material and material not in available_materials:
                        print(f"  Replacing material {material} with WE_A")
                        func_desc['material'] = 'WE_A'
        
        try:
            mas = mas_autocomplete.autocomplete(mas_dict)
        except Exception as e:
            print(f"Autocomplete error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create Ansyas instance with test-optimized settings
        ansyas = Ansyas(
            number_segments_arcs=12,
            initial_mesh_configuration=2,
            maximum_error_percent=5,
            refinement_percent=30,
            maximum_passes=10,  # Reduced for speed
            scale=1
        )
        
        # Create project
        project = ansyas.create_project(
            outputs_folder=self.output_path,
            project_name=f"{project_name}_{int(time.time())}",
            non_graphical=True,
            solution_type="EddyCurrent",
            new_desktop_session=True
        )
        
        try:
            # Run simulation
            ansyas.set_units("meter")
            ansyas.create_magnetic_simulation(mas=mas, single_frequency=True)
            ansyas.analyze()
            
            # Get results
            results = ansyas.outputs_extractor.get_results()
            
            # Verify results exist
            self.assertIsNotNone(results)
            self.assertTrue(hasattr(results, 'impedanceMatrix') or hasattr(results, 'inductanceMatrix'))
            
            print(f"✅ {mas_file} - PASSED")
            return True
            
        except Exception as e:
            print(f"❌ {mas_file} - FAILED: {str(e)}")
            raise
        finally:
            # Cleanup
            try:
                project.release_desktop(close_projects=True, close_desktop=True)
            except:
                pass
    
    def test_concentric_transformer(self):
        """Test concentric transformer."""
        self._run_simulation("concentric_transformer.json", "test_concentric_transformer")
    
    def test_concentric_flyback_rectangular_column(self):
        """Test concentric flyback with rectangular column."""
        self._run_simulation("concentric_flyback_rectangular_column.json", "test_concentric_flyback")
    
    def test_concentric_transformer_contiguous_rectangular_wire(self):
        """Test concentric transformer with contiguous rectangular wire."""
        self._run_simulation("concentric_transformer_contiguous_rectangular_wire.json", "test_concentric_rect_wire")


if __name__ == '__main__':
    unittest.main()
