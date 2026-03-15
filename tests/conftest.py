import os, json, pytest

def pytest_addoption(parser):
    parser.addoption("--run-ansys", action="store_true", default=False, help="Run Ansys tests.")
    parser.addoption("--run-elmer", action="store_true", default=False, help="Run Elmer FEM tests.")

def pytest_collection_modifyitems(config, items):
    run_ansys = config.getoption("--run-ansys")
    run_elmer = config.getoption("--run-elmer")
    
    skip_ansys = pytest.mark.skip(reason="Need --run-ansys")
    skip_elmer = pytest.mark.skip(reason="Need --run-elmer")
    
    for item in items:
        is_elmer = "elmer" in item.keywords
        is_ansys = "ansys" in item.keywords
        is_integration = "integration" in item.keywords

        # Skip elmer tests if --run-elmer not provided
        if not run_elmer and is_elmer:
            item.add_marker(skip_elmer)
        # Skip ansys tests (or non-elmer integration tests) if --run-ansys not provided
        elif not run_ansys and (is_ansys or (is_integration and not is_elmer)):
            item.add_marker(skip_ansys)

@pytest.fixture
def mas_files_dir(): return os.path.join(os.path.dirname(__file__), "mas_files")

@pytest.fixture
def examples_dir(): return os.path.join(os.path.dirname(__file__), "..", "examples")

@pytest.fixture
def output_dir(): 
    path = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(path, exist_ok=True)
    return path

@pytest.fixture
def load_mas_file(mas_files_dir):
    def _l(fn):
        with open(os.path.join(mas_files_dir, fn)) as f: return json.load(f)
    return _l
