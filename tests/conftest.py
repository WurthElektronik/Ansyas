import os, json, pytest

def pytest_addoption(parser):
    parser.addoption("--run-ansys", action="store_true", default=False, help="Run Ansys tests.")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-ansys"): return
    skip = pytest.mark.skip(reason="Need --run-ansys")
    for item in items:
        if "ansys" in item.keywords or "integration" in item.keywords: item.add_marker(skip)

@pytest.fixture
def mas_files_dir(): return os.path.join(os.path.dirname(__file__), "mas_files")

@pytest.fixture
def load_mas_file(mas_files_dir):
    def _l(fn):
        with open(os.path.join(mas_files_dir, fn)) as f: return json.load(f)
    return _l
