import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

class TestCooling:
    def test_loads(self):
        from Ansyas import cooling; assert cooling is not None
    def test_file(self):
        p = os.path.join(os.path.dirname(__file__), "..", "src", "Ansyas", "cooling.py")
        assert os.path.exists(p)
