import os
import sys
sys.path.append(os.path.dirname(__file__) + "./src/")
import MAS_models as MAS
import PyMKF


def autocomplete(mas: [MAS.Mas, dict]):
    """Checks and autocompletes any missing field in the input MAS file.

    MAS files have many optional fields, as it must be easy to specify a working
    minimum design, but must be able to unambiguously define a magnetic. Because
    of that it is possible that a file is missing some optional fields. This
    method autocompletes them and make sure all the information needed by Ansyas
    is available, keeping the dependency of PyMKF contained here.

    Parameters
    ----------
    mas : MAS.Mas, dict
        Mas file or dict containing the information about the magnetic, its
        inputs, and outputs.

    Examples
    --------
    Load a minimum MAS file and autocompletes it.

    >>> import json
    >>> import mas_autocomplete

    >>> f = open("tests/mas_files/simple_inductor_round_column.json)
    >>> mas_dict = json.load(f)
    >>> mas = mas_autocomplete.autocomplete(mas_dict)

    """
    if isinstance(mas, MAS.Mas):
        mas = mas.to_dict()

    if "outputs" not in mas:
        mas["outputs"] = []

    configuration = {}

    if "interleavingLevel" in mas["magnetic"]["coil"]:
        configuration["interleavingLevel"] = mas["magnetic"]["coil"]["interleavingLevel"]
    if "interleavingPattern" in mas["magnetic"]["coil"]:
        configuration["interleavingPattern"] = mas["magnetic"]["coil"]["interleavingPattern"]
    if "windingOrientation" in mas["magnetic"]["coil"]:
        configuration["windingOrientation"] = mas["magnetic"]["coil"]["windingOrientation"]
    if "layersOrientation" in mas["magnetic"]["coil"]:
        configuration["layersOrientation"] = mas["magnetic"]["coil"]["layersOrientation"]
    if "turnsAlignment" in mas["magnetic"]["coil"]:
        configuration["turnsAlignment"] = mas["magnetic"]["coil"]["turnsAlignment"]
    if "sectionAlignment" in mas["magnetic"]["coil"]:
        configuration["sectionAlignment"] = mas["magnetic"]["coil"]["sectionAlignment"]

    mas = PyMKF.mas_autocomplete(mas, configuration)
    mas = MAS.Mas.from_dict(mas)
    return mas
