import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from Ansyas import ansyas_utils

class TestUtils:
    def test_loads(self): assert ansyas_utils is not None
    def test_public_api(self): assert len([a for a in dir(ansyas_utils) if not a.startswith("_")]) > 0
