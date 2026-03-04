import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from Ansyas.ansyas import Ansyas

class TestDefaults:
    def test_mesh(self): assert Ansyas().initial_mesh_configuration == 5
    def test_segs(self): assert Ansyas().number_segments_arcs == 12
    def test_err(self): assert Ansyas().maximum_error_percent == 3
    def test_pass(self): assert Ansyas().maximum_passes == 40
    def test_ref(self): assert Ansyas().refinement_percent == 30
    def test_scale(self): assert Ansyas().scale == 1
    def test_backend(self): assert Ansyas()._geometry_backend_name == "ansys"

class TestCustom:
    def test_segs(self): assert Ansyas(number_segments_arcs=24).number_segments_arcs == 24
    def test_scale(self): assert Ansyas(scale=0.001).scale == 0.001
    def test_be(self): assert Ansyas(geometry_backend="cadquery")._geometry_backend_name == "cadquery"

class TestState:
    def test_backends_none(self):
        a = Ansyas()
        for x in ["geometry_backend","material_backend","meshing_backend","excitation_backend","solver_backend"]:
            assert getattr(a, x) is None
    def test_project_none(self): a = Ansyas(); assert a.project is None
    def test_padding(self):
        a = Ansyas()
        assert set(a.padding.keys()) == {"x_pos","y_pos","z_pos","x_neg","y_neg","z_neg"}
        assert all(v == 100 for v in a.padding.values())

class TestValidation:
    @pytest.mark.parametrize("v",[0,-1,6,10])
    def test_mesh(self,v):
        with pytest.raises(ValueError, match="initial_mesh_configuration"): Ansyas(initial_mesh_configuration=v)
    @pytest.mark.parametrize("v",[0,-5])
    def test_err(self,v):
        with pytest.raises(ValueError, match="maximum_error_percent"): Ansyas(maximum_error_percent=v)
    @pytest.mark.parametrize("v",[0,-1])
    def test_pass(self,v):
        with pytest.raises(ValueError, match="maximum_passes"): Ansyas(maximum_passes=v)
    @pytest.mark.parametrize("v",[0,-10,101])
    def test_ref(self,v):
        with pytest.raises(ValueError, match="refinement_percent"): Ansyas(refinement_percent=v)
    @pytest.mark.parametrize("v",[0,-1])
    def test_scale(self,v):
        with pytest.raises(ValueError, match="scale"): Ansyas(scale=v)
    @pytest.mark.parametrize("v",[0,1,2])
    def test_segs(self,v):
        with pytest.raises(ValueError, match="number_segments_arcs"): Ansyas(number_segments_arcs=v)

class TestBoundary:
    @pytest.mark.parametrize("v",[1,2,3,4,5])
    def test_mesh_ok(self,v): assert Ansyas(initial_mesh_configuration=v).initial_mesh_configuration == v
    def test_segs_min(self): assert Ansyas(number_segments_arcs=3).number_segments_arcs == 3
    def test_ref_edges(self):
        assert Ansyas(refinement_percent=0.01).refinement_percent == 0.01
        assert Ansyas(refinement_percent=100).refinement_percent == 100
