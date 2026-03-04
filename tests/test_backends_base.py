import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from Ansyas.backends.base import (GeometryBackend, MaterialBackend, MeshingBackend, ExcitationBackend, SolverBackend, BackendRegistry, Axis, Plane)

class TestEnums:
    def test_axis(self): assert len({Axis.X, Axis.Y, Axis.Z}) == 3
    def test_plane(self): assert len({Plane.XY, Plane.YZ, Plane.ZX}) == 3

class TestAbstract:
    @pytest.mark.parametrize("cls",[GeometryBackend,MaterialBackend,MeshingBackend,ExcitationBackend,SolverBackend])
    def test_not_instantiable(self, cls):
        with pytest.raises(TypeError): cls()

class TestRegistry:
    def test_exists(self): assert isinstance(BackendRegistry, type)
