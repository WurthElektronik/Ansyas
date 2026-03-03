import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

class TestExports:
    def test_version(self):
        from Ansyas import __version__; assert isinstance(__version__, str)
    def test_class(self):
        from Ansyas import Ansyas; assert callable(Ansyas)
    def test_all(self):
        import Ansyas; assert "Ansyas" in Ansyas.__all__

class TestSubmodules:
    def test_utils(self):
        from Ansyas import ansyas_utils; assert ansyas_utils is not None
    def test_backends(self):
        from Ansyas import backends; assert backends is not None
