import os
import sys
sys.path.append(os.path.dirname(__file__) + "./src/")
import MAS_models as MAS
import PyMKF


def autocomplete(mas):

    if isinstance(mas, dict):
        if "outputs" not in mas:
            mas["outputs"] = []
        mas = MAS.Mas.from_dict(mas)

    external_core_materials_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./external_data/core_materials.ndjson")
    with open(external_core_materials_data, "r") as f:
        external_core_materials_string = f.read()

    PyMKF.load_core_materials(external_core_materials_string)
    PyMKF.load_core_materials("")

    magnetic = mas.magnetic
    inputs = mas.inputs

    try:
        frequency = inputs.operatingPoints[0].excitationsPerWinding[0].frequency
    except:
        frequency = 100000
        inputs.operatingPoints[0].excitationsPerWinding[0].frequency = frequency

    if isinstance(magnetic.core.functionalDescription.shape, str):
        magnetic.core.functionalDescription.shape = MAS.CoreShape.from_dict(PyMKF.find_core_shape_by_name(magnetic.core.functionalDescription.shape))

    if magnetic.coil.bobbin == "Basic":
        core_data = PyMKF.calculate_core_data(magnetic.core.to_dict(), False)
        magnetic.coil.bobbin = MAS.Bobbin.from_dict(PyMKF.create_basic_bobbin(core_data, False))
    elif magnetic.coil.bobbin == "Dummy":
        core_data = PyMKF.calculate_core_data(magnetic.core.to_dict(), False)
        magnetic.coil.bobbin = MAS.Bobbin.from_dict(PyMKF.create_basic_bobbin(core_data, True))

    if magnetic.coil.bobbin.processedDescription.windingWindows[0].sectionsOrientation is None:
        if magnetic.core.functionalDescription.type == MAS.CoreType.twopieceset:
            magnetic.coil.bobbin.processedDescription.windingWindows[0].sectionsAlignment = MAS.CoilAlignment.centered
            magnetic.coil.bobbin.processedDescription.windingWindows[0].sectionsOrientation = MAS.WindingOrientation.overlapping
        else:
            magnetic.coil.bobbin.processedDescription.windingWindows[0].sectionsAlignment = MAS.CoilAlignment.spread
            magnetic.coil.bobbin.processedDescription.windingWindows[0].sectionsOrientation = MAS.WindingOrientation.contiguous

    for winding_index, winding in enumerate(magnetic.coil.functionalDescription):

        if isinstance(winding.wire, str):
            magnetic.coil.functionalDescription[winding_index].wire = MAS.Wire.from_dict(PyMKF.find_wire_by_name(winding.wire))

        if winding.wire.coating is None:
            winding.wire.coating = PyMKF.get_coating(winding.wire.to_dict())

        if winding.wire.coating.material is None:
            defaults = PyMKF.get_defaults()
            if magnetic.coil.functionalDescription[winding_index].wire.coating == MAS.InsulationWireCoatingType.enamelled:
                winding.wire.coating.material = defaults["defaultEnamelledInsulationMaterial"]
            else:
                winding.wire.coating.material = defaults["defaultInsulationMaterial"]
        if isinstance(winding.wire.coating.material, str):
            magnetic.coil.functionalDescription[winding_index].wire.coating.material = MAS.InsulationMaterial.from_dict(PyMKF.find_insulation_material_by_name(winding.wire.coating.material))

        if isinstance(winding.wire.strand, str):
            magnetic.coil.functionalDescription[winding_index].wire.strand = MAS.WireRound.from_dict(PyMKF.find_wire_by_name(winding.wire.strand))

    if isinstance(magnetic.core.functionalDescription.material, str):
        magnetic.core.functionalDescription.material = magnetic.core.functionalDescription.material.replace(' ', '')
        magnetic.core.functionalDescription.material = MAS.CoreMaterial.from_dict(PyMKF.find_core_material_by_name(magnetic.core.functionalDescription.material))
    else:
        magnetic.core.functionalDescription.material.name = magnetic.core.functionalDescription.material.name.replace(' ', '')

    if magnetic.core.processedDescription is None:
        core_data = PyMKF.calculate_core_data(magnetic.core.to_dict(), False)
        magnetic.core = MAS.MagneticCore.from_dict(core_data)

    if magnetic.core.geometricalDescription is None:
        geometricalDescription = PyMKF.calculate_core_geometrical_description(magnetic.core.to_dict())

        magnetic.core.geometricalDescription = []
        for description in geometricalDescription:
            magnetic.core.geometricalDescription.append(MAS.CoreGeometricalDescriptionElement.from_dict(description))

    if magnetic.coil.turnsDescription is None:
        coil_json = magnetic.coil.to_dict()
        if magnetic.core.functionalDescription.type == MAS.CoreType.twopieceset:
            coil_json["_turnsAlignment"] = "spread"

        wind_result = PyMKF.wind(coil_json, 1, [], [], [[0, 0]])
        if isinstance(wind_result, str) and wind_result.startswith("Exception"):
            raise Exception("Error while winding")
        magnetic.coil = MAS.Coil.from_dict(wind_result)

    if magnetic.coil.layersDescription is not None:
        for layer_index, layer in enumerate(magnetic.coil.layersDescription):
            insulation_material = PyMKF.get_insulation_layer_insulation_material(magnetic.coil.to_dict(), layer.name)
            magnetic.coil.layersDescription[layer_index].insulationMaterial = MAS.InsulationMaterial.from_dict(insulation_material)

    number_windings = len(magnetic.coil.functionalDescription)
    for operating_point_index in range(0, len(inputs.operatingPoints)):
        for excitation_index in range(0, len(inputs.operatingPoints[operating_point_index].excitationsPerWinding)):
            if inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.waveform is None:
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.waveform = MAS.Waveform.from_dict(PyMKF.create_waveform(
                    inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.processed.to_dict(),
                    frequency
                ))
            if inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.harmonics is None:
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.harmonics = MAS.Harmonics.from_dict(PyMKF.calculate_harmonics(
                    inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.waveform.to_dict(),
                    frequency
                ))
            inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.processed = MAS.Processed.from_dict(PyMKF.calculate_processed(
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.harmonics.to_dict(),
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].current.waveform.to_dict(),
            ))

            if inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.waveform is None:
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.waveform = MAS.Waveform.from_dict(PyMKF.create_waveform(
                    inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.processed.to_dict(),
                    frequency
                ))

            if inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.harmonics is None:
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.harmonics = MAS.Harmonics.from_dict(PyMKF.calculate_harmonics(
                    inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.waveform.to_dict(),
                    frequency
                ))

            inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.processed = MAS.Processed.from_dict(PyMKF.calculate_processed(
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.harmonics.to_dict(),
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].voltage.waveform.to_dict(),
            ))

            magnetizingInductance = PyMKF.calculate_inductance_from_number_turns_and_gapping(
                magnetic.core.to_dict(),
                magnetic.coil.to_dict(),
                inputs.operatingPoints[operating_point_index].to_dict(),
                {},
            )

            inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].magnetizingCurrent = MAS.SignalDescriptor.from_dict(PyMKF.calculate_induced_current(
                inputs.operatingPoints[operating_point_index].excitationsPerWinding[operating_point_index].to_dict(),
                magnetizingInductance)
            )

            if excitation_index == 0 and number_windings == 2 and len(inputs.operatingPoints[operating_point_index].excitationsPerWinding) == 1:
                primaryExcitationJson = inputs.operatingPoints[operating_point_index].excitationsPerWinding[0].to_dict()
                turnRatio = magnetic.coil.functionalDescription[0].numberTurns / magnetic.coil.functionalDescription[1].numberTurns
                secondary_excitation = MAS.OperatingPointExcitation.from_dict(PyMKF.calculate_reflected_secondary(primaryExcitationJson, turnRatio))
                inputs.operatingPoints[operating_point_index].excitationsPerWinding.append(secondary_excitation)

    mas = MAS.Mas.from_dict(PyMKF.simulate(inputs.to_dict(), magnetic.to_dict(), {}))
    mas.inputs = inputs

    return mas
