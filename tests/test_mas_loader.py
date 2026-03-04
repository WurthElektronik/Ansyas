import pytest, json, os
MAS_DIR = os.path.join(os.path.dirname(__file__), "mas_files")
def _all():
    return sorted(f for f in os.listdir(MAS_DIR) if f.endswith(".json")) if os.path.isdir(MAS_DIR) else []
def _small():
    return [f for f in _all() if os.path.getsize(os.path.join(MAS_DIR,f)) < 50000]

class TestValidity:
    @pytest.fixture(params=_all())
    def fp(self, request): return os.path.join(MAS_DIR, request.param)
    def test_not_empty(self,fp): assert os.path.getsize(fp) > 0
    def test_valid_json(self,fp):
        with open(fp) as f: assert isinstance(json.load(f), dict)

class TestStructure:
    @pytest.fixture(params=_small())
    def data(self, request):
        with open(os.path.join(MAS_DIR, request.param)) as f: return json.load(f)
    def test_has_inputs(self,data): assert "inputs" in data
    def test_has_op(self,data):
        inp = data.get("inputs",{})
        assert "operatingPoints" in inp or "operationPoints" in inp

class TestErrors:
    def test_missing(self):
        with pytest.raises(FileNotFoundError): open("/no/file.json")
    def test_bad(self, tmp_path):
        (tmp_path/"b.json").write_text("{bad}")
        with pytest.raises(json.JSONDecodeError):
            with open(str(tmp_path/"b.json")) as f: json.load(f)
    def test_empty(self, tmp_path):
        (tmp_path/"e.json").write_text("")
        with pytest.raises(json.JSONDecodeError):
            with open(str(tmp_path/"e.json")) as f: json.load(f)
