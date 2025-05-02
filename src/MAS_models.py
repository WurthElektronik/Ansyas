from dataclasses import dataclass
from typing import Optional, Any, List, Dict, Union, TypeVar, Callable, Type, cast
from enum import Enum


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, (int, float))
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


@dataclass
class DimensionWithTolerance:
    """Required values for the altitude
    
    Voltage RMS of the main supply to which this transformer is connected to.
    
    Required values for the magnetizing inductance
    
    Required values for the temperature that the magnetic can reach under operating
    
    The maximum thickness of the insulation around the wire, in m
    
    The conducting area of the wire, in m². Used for some rectangular shapes where the area
    is smaller than expected due to rounded corners
    
    The conducting diameter of the wire, in m
    
    The outer diameter of the wire, in m
    
    The conducting height of the wire, in m
    
    The conducting width of the wire, in m
    
    The outer height of the wire, in m
    
    The outer width of the wire, in m
    
    The radius of the edge, in case of rectangular wire, in m
    
    Heat capacity value according to manufacturer, in J/Kg/K
    
    Heat conductivity value according to manufacturer, in W/m/K
    
    Data a two dimensional matrix, created as an array of array, where the first coordinate
    in the X and the second the Y
    
    Value of the leakage inductance between the primary and a secondary winding given by the
    position in the array
    
    Value of the magnetizing inductance
    
    A dimension of with minimum, nominal, and maximum values
    """
    excludeMaximum: Optional[bool] = None
    """True is the maximum value must be excluded from the range"""

    excludeMinimum: Optional[bool] = None
    """True is the minimum value must be excluded from the range"""

    maximum: Optional[float] = None
    """The maximum value of the dimension"""

    minimum: Optional[float] = None
    """The minimum value of the dimension"""

    nominal: Optional[float] = None
    """The nominal value of the dimension"""

    @staticmethod
    def from_dict(obj: Any) -> 'DimensionWithTolerance':
        assert isinstance(obj, dict)
        excludeMaximum = from_union([from_bool, from_none], obj.get("excludeMaximum"))
        excludeMinimum = from_union([from_bool, from_none], obj.get("excludeMinimum"))
        maximum = from_union([from_float, from_none], obj.get("maximum"))
        minimum = from_union([from_float, from_none], obj.get("minimum"))
        nominal = from_union([from_float, from_none], obj.get("nominal"))
        return DimensionWithTolerance(excludeMaximum, excludeMinimum, maximum, minimum, nominal)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.excludeMaximum is not None:
            result["excludeMaximum"] = from_union([from_bool, from_none], self.excludeMaximum)
        if self.excludeMinimum is not None:
            result["excludeMinimum"] = from_union([from_bool, from_none], self.excludeMinimum)
        if self.maximum is not None:
            result["maximum"] = from_union([to_float, from_none], self.maximum)
        if self.minimum is not None:
            result["minimum"] = from_union([to_float, from_none], self.minimum)
        if self.nominal is not None:
            result["nominal"] = from_union([to_float, from_none], self.nominal)
        return result


class CTI(Enum):
    """Required CTI"""

    GroupI = "Group I"
    GroupII = "Group II"
    GroupIIIA = "Group IIIA"
    GroupIIIB = "Group IIIB"


class InsulationType(Enum):
    """Required type of insulation"""

    Basic = "Basic"
    Double = "Double"
    Functional = "Functional"
    Reinforced = "Reinforced"
    Supplementary = "Supplementary"


class OvervoltageCategory(Enum):
    """Required overvoltage category"""

    OVCI = "OVC-I"
    OVCII = "OVC-II"
    OVCIII = "OVC-III"
    OVCIV = "OVC-IV"


class PollutionDegree(Enum):
    """Required pollution for the magnetic to work under"""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class InsulationStandards(Enum):
    IEC603351 = "IEC 60335-1"
    IEC606641 = "IEC 60664-1"
    IEC615581 = "IEC 61558-1"
    IEC623681 = "IEC 62368-1"


@dataclass
class InsulationRequirements:
    altitude: Optional[DimensionWithTolerance] = None
    """Required values for the altitude"""

    cti: Optional[CTI] = None
    """Required CTI"""

    insulationType: Optional[InsulationType] = None
    """Required type of insulation"""

    mainSupplyVoltage: Optional[DimensionWithTolerance] = None
    """Voltage RMS of the main supply to which this transformer is connected to."""

    overvoltageCategory: Optional[OvervoltageCategory] = None
    """Required overvoltage category"""

    pollutionDegree: Optional[PollutionDegree] = None
    """Required pollution for the magnetic to work under"""

    standards: Optional[List[InsulationStandards]] = None
    """VList of standards that will be taken into account for insulation."""

    @staticmethod
    def from_dict(obj: Any) -> 'InsulationRequirements':
        assert isinstance(obj, dict)
        altitude = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("altitude"))
        cti = from_union([CTI, from_none], obj.get("cti"))
        insulationType = from_union([InsulationType, from_none], obj.get("insulationType"))
        mainSupplyVoltage = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("mainSupplyVoltage"))
        overvoltageCategory = from_union([OvervoltageCategory, from_none], obj.get("overvoltageCategory"))
        pollutionDegree = from_union([PollutionDegree, from_none], obj.get("pollutionDegree"))
        standards = from_union([lambda x: from_list(InsulationStandards, x), from_none], obj.get("standards"))
        return InsulationRequirements(altitude, cti, insulationType, mainSupplyVoltage, overvoltageCategory, pollutionDegree, standards)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.altitude is not None:
            result["altitude"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.altitude)
        if self.cti is not None:
            result["cti"] = from_union([lambda x: to_enum(CTI, x), from_none], self.cti)
        if self.insulationType is not None:
            result["insulationType"] = from_union([lambda x: to_enum(InsulationType, x), from_none], self.insulationType)
        if self.mainSupplyVoltage is not None:
            result["mainSupplyVoltage"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.mainSupplyVoltage)
        if self.overvoltageCategory is not None:
            result["overvoltageCategory"] = from_union([lambda x: to_enum(OvervoltageCategory, x), from_none], self.overvoltageCategory)
        if self.pollutionDegree is not None:
            result["pollutionDegree"] = from_union([lambda x: to_enum(PollutionDegree, x), from_none], self.pollutionDegree)
        if self.standards is not None:
            result["standards"] = from_union([lambda x: from_list(lambda x: to_enum(InsulationStandards, x), x), from_none], self.standards)
        return result


class IsolationSide(Enum):
    """Tag to identify windings that are sharing the same ground"""

    denary = "denary"
    duodenary = "duodenary"
    nonary = "nonary"
    octonary = "octonary"
    primary = "primary"
    quaternary = "quaternary"
    quinary = "quinary"
    secondary = "secondary"
    senary = "senary"
    septenary = "septenary"
    tertiary = "tertiary"
    undenary = "undenary"


class Market(Enum):
    """Market where the magnetic will end up being used"""

    Commercial = "Commercial"
    Industrial = "Industrial"
    Medical = "Medical"
    Military = "Military"
    Space = "Space"


@dataclass
class MaximumDimensions:
    """Maximum dimensions, width, height, and depth, for the designed magnetic, in m"""

    depth: Optional[float] = None
    height: Optional[float] = None
    width: Optional[float] = None

    @staticmethod
    def from_dict(obj: Any) -> 'MaximumDimensions':
        assert isinstance(obj, dict)
        depth = from_union([from_float, from_none], obj.get("depth"))
        height = from_union([from_float, from_none], obj.get("height"))
        width = from_union([from_float, from_none], obj.get("width"))
        return MaximumDimensions(depth, height, width)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.depth is not None:
            result["depth"] = from_union([to_float, from_none], self.depth)
        if self.height is not None:
            result["height"] = from_union([to_float, from_none], self.height)
        if self.width is not None:
            result["width"] = from_union([to_float, from_none], self.width)
        return result


@dataclass
class ImpedancePoint:
    """Data describing one impendance value"""

    magnitude: float
    """Magnitude of the impedance, in Ohm"""

    imaginaryPart: Optional[float] = None
    phase: Optional[float] = None
    realPart: Optional[float] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ImpedancePoint':
        assert isinstance(obj, dict)
        magnitude = from_float(obj.get("magnitude"))
        imaginaryPart = from_union([from_float, from_none], obj.get("imaginaryPart"))
        phase = from_union([from_float, from_none], obj.get("phase"))
        realPart = from_union([from_float, from_none], obj.get("realPart"))
        return ImpedancePoint(magnitude, imaginaryPart, phase, realPart)

    def to_dict(self) -> dict:
        result: dict = {}
        result["magnitude"] = to_float(self.magnitude)
        if self.imaginaryPart is not None:
            result["imaginaryPart"] = from_union([to_float, from_none], self.imaginaryPart)
        if self.phase is not None:
            result["phase"] = from_union([to_float, from_none], self.phase)
        if self.realPart is not None:
            result["realPart"] = from_union([to_float, from_none], self.realPart)
        return result


@dataclass
class ImpedanceAtFrequency:
    frequency: float
    impedance: ImpedancePoint

    @staticmethod
    def from_dict(obj: Any) -> 'ImpedanceAtFrequency':
        assert isinstance(obj, dict)
        frequency = from_float(obj.get("frequency"))
        impedance = ImpedancePoint.from_dict(obj.get("impedance"))
        return ImpedanceAtFrequency(frequency, impedance)

    def to_dict(self) -> dict:
        result: dict = {}
        result["frequency"] = to_float(self.frequency)
        result["impedance"] = to_class(ImpedancePoint, self.impedance)
        return result


class ConnectionType(Enum):
    """Type of the terminal"""

    FlyingLead = "Flying Lead"
    Pin = "Pin"
    SMT = "SMT"
    Screw = "Screw"


class Topologies(Enum):
    """Topology that will use the magnetic"""

    ActiveClampForwardConverter = "Active Clamp Forward Converter"
    BoostConverter = "Boost Converter"
    BuckConverter = "Buck Converter"
    CukConverter = "Cuk Converter"
    FlybackConverter = "Flyback Converter"
    FullBridgeConverter = "Full-Bridge Converter"
    HalfBridgeConverter = "Half-Bridge Converter"
    InvertingBuckBoostConverter = "Inverting Buck-Boost Converter"
    PhaseShiftedFullBridgeConverter = "Phase-Shifted Full-Bridge Converter"
    PushPullConverter = "Push-Pull Converter"
    SEPIC = "SEPIC"
    SingleSwitchForwardConverter = "Single Switch Forward Converter"
    TwoSwitchFlybackConverter = "Two Switch Flyback Converter"
    TwoSwitchForwardConverter = "Two Switch Forward Converter"
    WeinbergConverter = "Weinberg Converter"
    ZetaConverter = "Zeta Converter"


class WiringTechnology(Enum):
    """Technology that must be used to create the wiring"""

    Deposition = "Deposition"
    Printed = "Printed"
    Wound = "Wound"


@dataclass
class DesignRequirements:
    """Data describing the design requirements
    
    The list of requirement that must comply a given magnetic
    """
    magnetizingInductance: DimensionWithTolerance
    """Required values for the magnetizing inductance"""

    turnsRatios: List[DimensionWithTolerance]
    """Required turns ratios between primary and the rest of windings"""

    insulation: Optional[InsulationRequirements] = None
    isolationSides: Optional[List[IsolationSide]] = None
    """Isolation side where each winding is connected to."""

    leakageInductance: Optional[List[DimensionWithTolerance]] = None
    """Required values for the leakage inductance"""

    market: Optional[Market] = None
    """Market where the magnetic will end up being used"""

    maximumDimensions: Optional[MaximumDimensions] = None
    """Maximum dimensions, width, height, and depth, for the designed magnetic, in m"""

    maximumWeight: Optional[float] = None
    """Maximum weight for the designed magnetic, in Kg"""

    minimumImpedance: Optional[List[ImpedanceAtFrequency]] = None
    """List of minimum impedance at given frequency in the primary"""

    name: Optional[str] = None
    """A label that identifies these Design Requirements"""

    operatingTemperature: Optional[DimensionWithTolerance] = None
    """Required values for the temperature that the magnetic can reach under operating"""

    strayCapacitance: Optional[List[DimensionWithTolerance]] = None
    """Required values for the stray capacitance"""

    terminalType: Optional[List[ConnectionType]] = None
    """Type of the terminal that must be used, per winding"""

    topology: Optional[Topologies] = None
    """Topology that will use the magnetic"""

    wiringTechnology: Optional[WiringTechnology] = None
    """Technology that must be used to create the wiring"""

    @staticmethod
    def from_dict(obj: Any) -> 'DesignRequirements':
        assert isinstance(obj, dict)
        magnetizingInductance = DimensionWithTolerance.from_dict(obj.get("magnetizingInductance"))
        turnsRatios = from_list(DimensionWithTolerance.from_dict, obj.get("turnsRatios"))
        insulation = from_union([InsulationRequirements.from_dict, from_none], obj.get("insulation"))
        isolationSides = from_union([lambda x: from_list(IsolationSide, x), from_none], obj.get("isolationSides"))
        leakageInductance = from_union([lambda x: from_list(DimensionWithTolerance.from_dict, x), from_none], obj.get("leakageInductance"))
        market = from_union([Market, from_none], obj.get("market"))
        maximumDimensions = from_union([MaximumDimensions.from_dict, from_none], obj.get("maximumDimensions"))
        maximumWeight = from_union([from_float, from_none], obj.get("maximumWeight"))
        minimumImpedance = from_union([lambda x: from_list(ImpedanceAtFrequency.from_dict, x), from_none], obj.get("minimumImpedance"))
        name = from_union([from_str, from_none], obj.get("name"))
        operatingTemperature = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("operatingTemperature"))
        strayCapacitance = from_union([lambda x: from_list(DimensionWithTolerance.from_dict, x), from_none], obj.get("strayCapacitance"))
        terminalType = from_union([lambda x: from_list(ConnectionType, x), from_none], obj.get("terminalType"))
        topology = from_union([Topologies, from_none], obj.get("topology"))
        wiringTechnology = from_union([WiringTechnology, from_none], obj.get("wiringTechnology"))
        return DesignRequirements(magnetizingInductance, turnsRatios, insulation, isolationSides, leakageInductance, market, maximumDimensions, maximumWeight, minimumImpedance, name, operatingTemperature, strayCapacitance, terminalType, topology, wiringTechnology)

    def to_dict(self) -> dict:
        result: dict = {}
        result["magnetizingInductance"] = to_class(DimensionWithTolerance, self.magnetizingInductance)
        result["turnsRatios"] = from_list(lambda x: to_class(DimensionWithTolerance, x), self.turnsRatios)
        if self.insulation is not None:
            result["insulation"] = from_union([lambda x: to_class(InsulationRequirements, x), from_none], self.insulation)
        if self.isolationSides is not None:
            result["isolationSides"] = from_union([lambda x: from_list(lambda x: to_enum(IsolationSide, x), x), from_none], self.isolationSides)
        if self.leakageInductance is not None:
            result["leakageInductance"] = from_union([lambda x: from_list(lambda x: to_class(DimensionWithTolerance, x), x), from_none], self.leakageInductance)
        if self.market is not None:
            result["market"] = from_union([lambda x: to_enum(Market, x), from_none], self.market)
        if self.maximumDimensions is not None:
            result["maximumDimensions"] = from_union([lambda x: to_class(MaximumDimensions, x), from_none], self.maximumDimensions)
        if self.maximumWeight is not None:
            result["maximumWeight"] = from_union([to_float, from_none], self.maximumWeight)
        if self.minimumImpedance is not None:
            result["minimumImpedance"] = from_union([lambda x: from_list(lambda x: to_class(ImpedanceAtFrequency, x), x), from_none], self.minimumImpedance)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.operatingTemperature is not None:
            result["operatingTemperature"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.operatingTemperature)
        if self.strayCapacitance is not None:
            result["strayCapacitance"] = from_union([lambda x: from_list(lambda x: to_class(DimensionWithTolerance, x), x), from_none], self.strayCapacitance)
        if self.terminalType is not None:
            result["terminalType"] = from_union([lambda x: from_list(lambda x: to_enum(ConnectionType, x), x), from_none], self.terminalType)
        if self.topology is not None:
            result["topology"] = from_union([lambda x: to_enum(Topologies, x), from_none], self.topology)
        if self.wiringTechnology is not None:
            result["wiringTechnology"] = from_union([lambda x: to_enum(WiringTechnology, x), from_none], self.wiringTechnology)
        return result


@dataclass
class Cooling:
    """Relative Humidity of the ambient where the magnetic will operate
    
    Data describing a natural convection cooling
    
    Data describing a forced convection cooling
    
    Data describing a heatsink cooling
    
    Data describing a cold plate cooling
    """
    fluid: Optional[str] = None
    """Name of the fluid used"""

    temperature: Optional[float] = None
    """Temperature of the fluid. To be used only if different from ambient temperature"""

    flowDiameter: Optional[float] = None
    """Diameter of the fluid flow, normally defined as a fan diameter"""

    velocity: Optional[List[float]] = None
    dimensions: Optional[List[float]] = None
    """Dimensions of the cube defining the heatsink
    
    Dimensions of the cube defining the cold plate
    """
    interfaceThermalResistance: Optional[float] = None
    """Bulk thermal resistance of the thermal interface used to connect the device to the
    heatsink, in W/mK
    
    Bulk thermal resistance of the thermal interface used to connect the device to the cold
    plate, in W/mK
    """
    interfaceThickness: Optional[float] = None
    """Thickness of the thermal interface used to connect the device to the heatsink, in m
    
    Thickness of the thermal interface used to connect the device to the cold plate, in m
    """
    thermalResistance: Optional[float] = None
    """Bulk thermal resistance of the heat sink, in W/K
    
    Bulk thermal resistance of the cold plate, in W/K
    """
    maximumTemperature: Optional[float] = None
    """Maximum temperature of the cold plate"""

    @staticmethod
    def from_dict(obj: Any) -> 'Cooling':
        assert isinstance(obj, dict)
        fluid = from_union([from_str, from_none], obj.get("fluid"))
        temperature = from_union([from_float, from_none], obj.get("temperature"))
        flowDiameter = from_union([from_float, from_none], obj.get("flowDiameter"))
        velocity = from_union([lambda x: from_list(from_float, x), from_none], obj.get("velocity"))
        dimensions = from_union([lambda x: from_list(from_float, x), from_none], obj.get("dimensions"))
        interfaceThermalResistance = from_union([from_float, from_none], obj.get("interfaceThermalResistance"))
        interfaceThickness = from_union([from_float, from_none], obj.get("interfaceThickness"))
        thermalResistance = from_union([from_float, from_none], obj.get("thermalResistance"))
        maximumTemperature = from_union([from_float, from_none], obj.get("maximumTemperature"))
        return Cooling(fluid, temperature, flowDiameter, velocity, dimensions, interfaceThermalResistance, interfaceThickness, thermalResistance, maximumTemperature)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.fluid is not None:
            result["fluid"] = from_union([from_str, from_none], self.fluid)
        if self.temperature is not None:
            result["temperature"] = from_union([to_float, from_none], self.temperature)
        if self.flowDiameter is not None:
            result["flowDiameter"] = from_union([to_float, from_none], self.flowDiameter)
        if self.velocity is not None:
            result["velocity"] = from_union([lambda x: from_list(to_float, x), from_none], self.velocity)
        if self.dimensions is not None:
            result["dimensions"] = from_union([lambda x: from_list(to_float, x), from_none], self.dimensions)
        if self.interfaceThermalResistance is not None:
            result["interfaceThermalResistance"] = from_union([to_float, from_none], self.interfaceThermalResistance)
        if self.interfaceThickness is not None:
            result["interfaceThickness"] = from_union([to_float, from_none], self.interfaceThickness)
        if self.thermalResistance is not None:
            result["thermalResistance"] = from_union([to_float, from_none], self.thermalResistance)
        if self.maximumTemperature is not None:
            result["maximumTemperature"] = from_union([to_float, from_none], self.maximumTemperature)
        return result


@dataclass
class OperatingConditions:
    """The description of a magnetic operating conditions"""

    ambientTemperature: float
    """Temperature of the ambient where the magnetic will operate"""

    ambientRelativeHumidity: Optional[float] = None
    """Relative Humidity of the ambient where the magnetic will operate"""

    cooling: Optional[Cooling] = None
    """Relative Humidity of the ambient where the magnetic will operate"""

    name: Optional[str] = None
    """A label that identifies this Operating Conditions"""

    @staticmethod
    def from_dict(obj: Any) -> 'OperatingConditions':
        assert isinstance(obj, dict)
        ambientTemperature = from_float(obj.get("ambientTemperature"))
        ambientRelativeHumidity = from_union([from_float, from_none], obj.get("ambientRelativeHumidity"))
        cooling = from_union([Cooling.from_dict, from_none], obj.get("cooling"))
        name = from_union([from_str, from_none], obj.get("name"))
        return OperatingConditions(ambientTemperature, ambientRelativeHumidity, cooling, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["ambientTemperature"] = to_float(self.ambientTemperature)
        if self.ambientRelativeHumidity is not None:
            result["ambientRelativeHumidity"] = from_union([to_float, from_none], self.ambientRelativeHumidity)
        if self.cooling is not None:
            result["cooling"] = from_union([lambda x: to_class(Cooling, x), from_none], self.cooling)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        return result


@dataclass
class Harmonics:
    """Data containing the harmonics of the waveform, defined by a list of amplitudes and a list
    of frequencies
    """
    amplitudes: List[float]
    """List of amplitudes of the harmonics that compose the waveform"""

    frequencies: List[float]
    """List of frequencies of the harmonics that compose the waveform"""

    @staticmethod
    def from_dict(obj: Any) -> 'Harmonics':
        assert isinstance(obj, dict)
        amplitudes = from_list(from_float, obj.get("amplitudes"))
        frequencies = from_list(from_float, obj.get("frequencies"))
        return Harmonics(amplitudes, frequencies)

    def to_dict(self) -> dict:
        result: dict = {}
        result["amplitudes"] = from_list(to_float, self.amplitudes)
        result["frequencies"] = from_list(to_float, self.frequencies)
        return result


class WaveformLabel(Enum):
    """Label of the waveform, if applicable. Used for common waveforms"""

    BipolarRectangular = "Bipolar Rectangular"
    BipolarTriangular = "Bipolar Triangular"
    Custom = "Custom"
    FlybackPrimary = "Flyback Primary"
    FlybackSecondary = "Flyback Secondary"
    FlybackSecondaryWithDeadtime = "Flyback Secondary With Deadtime"
    Rectangular = "Rectangular"
    RectangularDCM = "RectangularDCM"
    RectangularWithDeadtime = "Rectangular With Deadtime"
    SecondaryRectangular = "Secondary Rectangular"
    SecondaryRectangularWithDeadtime = "Secondary Rectangular With Deadtime"
    Sinusoidal = "Sinusoidal"
    Triangular = "Triangular"
    UnipolarRectangular = "Unipolar Rectangular"
    UnipolarTriangular = "Unipolar Triangular"


@dataclass
class Processed:
    label: WaveformLabel
    offset: float
    """The offset value of the waveform, referred to 0"""

    acEffectiveFrequency: Optional[float] = None
    """The effective frequency value of the AC component of the waveform, according to
    https://sci-hub.wf/https://ieeexplore.ieee.org/document/750181, Appendix C
    """
    average: Optional[float] = None
    """The average value of the waveform, referred to 0"""

    deadTime: Optional[float] = None
    """The dead time after TOn and Toff, in seconds, if applicable"""

    dutyCycle: Optional[float] = None
    """The duty cycle of the waveform, if applicable"""

    effectiveFrequency: Optional[float] = None
    """The effective frequency value of the waveform, according to
    https://sci-hub.wf/https://ieeexplore.ieee.org/document/750181, Appendix C
    """
    peak: Optional[float] = None
    """The maximum positive value of the waveform"""

    peakToPeak: Optional[float] = None
    """The peak to peak value of the waveform"""

    phase: Optional[float] = None
    """The phase of the waveform, in degrees"""

    rms: Optional[float] = None
    """The RMS value of the waveform"""

    thd: Optional[float] = None
    """The Total Harmonic Distortion of the waveform, according to
    https://en.wikipedia.org/wiki/Total_harmonic_distortion
    """

    @staticmethod
    def from_dict(obj: Any) -> 'Processed':
        assert isinstance(obj, dict)
        label = WaveformLabel(obj.get("label"))
        offset = from_float(obj.get("offset"))
        acEffectiveFrequency = from_union([from_float, from_none], obj.get("acEffectiveFrequency"))
        average = from_union([from_float, from_none], obj.get("average"))
        deadTime = from_union([from_float, from_none], obj.get("deadTime"))
        dutyCycle = from_union([from_float, from_none], obj.get("dutyCycle"))
        effectiveFrequency = from_union([from_float, from_none], obj.get("effectiveFrequency"))
        peak = from_union([from_float, from_none], obj.get("peak"))
        peakToPeak = from_union([from_float, from_none], obj.get("peakToPeak"))
        phase = from_union([from_float, from_none], obj.get("phase"))
        rms = from_union([from_float, from_none], obj.get("rms"))
        thd = from_union([from_float, from_none], obj.get("thd"))
        return Processed(label, offset, acEffectiveFrequency, average, deadTime, dutyCycle, effectiveFrequency, peak, peakToPeak, phase, rms, thd)

    def to_dict(self) -> dict:
        result: dict = {}
        result["label"] = to_enum(WaveformLabel, self.label)
        result["offset"] = to_float(self.offset)
        if self.acEffectiveFrequency is not None:
            result["acEffectiveFrequency"] = from_union([to_float, from_none], self.acEffectiveFrequency)
        if self.average is not None:
            result["average"] = from_union([to_float, from_none], self.average)
        if self.deadTime is not None:
            result["deadTime"] = from_union([to_float, from_none], self.deadTime)
        if self.dutyCycle is not None:
            result["dutyCycle"] = from_union([to_float, from_none], self.dutyCycle)
        if self.effectiveFrequency is not None:
            result["effectiveFrequency"] = from_union([to_float, from_none], self.effectiveFrequency)
        if self.peak is not None:
            result["peak"] = from_union([to_float, from_none], self.peak)
        if self.peakToPeak is not None:
            result["peakToPeak"] = from_union([to_float, from_none], self.peakToPeak)
        if self.phase is not None:
            result["phase"] = from_union([to_float, from_none], self.phase)
        if self.rms is not None:
            result["rms"] = from_union([to_float, from_none], self.rms)
        if self.thd is not None:
            result["thd"] = from_union([to_float, from_none], self.thd)
        return result


@dataclass
class Waveform:
    """Data containing the points that define an arbitrary waveform with equidistant points
    
    Data containing the points that define an arbitrary waveform with non-equidistant points
    paired with their time in the period
    """
    data: List[float]
    """List of values that compose the waveform, at equidistant times form each other"""

    numberPeriods: Optional[int] = None
    """The number of periods covered by the data"""

    ancillaryLabel: Optional[WaveformLabel] = None
    time: Optional[List[float]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Waveform':
        assert isinstance(obj, dict)
        data = from_list(from_float, obj.get("data"))
        numberPeriods = from_union([from_int, from_none], obj.get("numberPeriods"))
        ancillaryLabel = from_union([WaveformLabel, from_none], obj.get("ancillaryLabel"))
        time = from_union([lambda x: from_list(from_float, x), from_none], obj.get("time"))
        return Waveform(data, numberPeriods, ancillaryLabel, time)

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = from_list(to_float, self.data)
        if self.numberPeriods is not None:
            result["numberPeriods"] = from_union([from_int, from_none], self.numberPeriods)
        if self.ancillaryLabel is not None:
            result["ancillaryLabel"] = from_union([lambda x: to_enum(WaveformLabel, x), from_none], self.ancillaryLabel)
        if self.time is not None:
            result["time"] = from_union([lambda x: from_list(to_float, x), from_none], self.time)
        return result


@dataclass
class SignalDescriptor:
    """Excitation of the B field that produced the core losses
    
    Structure definining one electromagnetic parameters: current, voltage, magnetic flux
    density
    """
    harmonics: Optional[Harmonics] = None
    """Data containing the harmonics of the waveform, defined by a list of amplitudes and a list
    of frequencies
    """
    processed: Optional[Processed] = None
    waveform: Optional[Waveform] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SignalDescriptor':
        assert isinstance(obj, dict)
        harmonics = from_union([Harmonics.from_dict, from_none], obj.get("harmonics"))
        processed = from_union([Processed.from_dict, from_none], obj.get("processed"))
        waveform = from_union([Waveform.from_dict, from_none], obj.get("waveform"))
        return SignalDescriptor(harmonics, processed, waveform)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.harmonics is not None:
            result["harmonics"] = from_union([lambda x: to_class(Harmonics, x), from_none], self.harmonics)
        if self.processed is not None:
            result["processed"] = from_union([lambda x: to_class(Processed, x), from_none], self.processed)
        if self.waveform is not None:
            result["waveform"] = from_union([lambda x: to_class(Waveform, x), from_none], self.waveform)
        return result


@dataclass
class OperatingPointExcitation:
    """Data describing the excitation of the winding
    
    The description of a magnetic operating point
    """
    frequency: float
    """Frequency of the waveform, common for all electromagnetic parameters, in Hz"""

    current: Optional[SignalDescriptor] = None
    magneticFieldStrength: Optional[SignalDescriptor] = None
    magneticFluxDensity: Optional[SignalDescriptor] = None
    magnetizingCurrent: Optional[SignalDescriptor] = None
    name: Optional[str] = None
    """A label that identifies this Operating Point"""

    voltage: Optional[SignalDescriptor] = None

    @staticmethod
    def from_dict(obj: Any) -> 'OperatingPointExcitation':
        assert isinstance(obj, dict)
        frequency = from_float(obj.get("frequency"))
        current = from_union([SignalDescriptor.from_dict, from_none], obj.get("current"))
        magneticFieldStrength = from_union([SignalDescriptor.from_dict, from_none], obj.get("magneticFieldStrength"))
        magneticFluxDensity = from_union([SignalDescriptor.from_dict, from_none], obj.get("magneticFluxDensity"))
        magnetizingCurrent = from_union([SignalDescriptor.from_dict, from_none], obj.get("magnetizingCurrent"))
        name = from_union([from_str, from_none], obj.get("name"))
        voltage = from_union([SignalDescriptor.from_dict, from_none], obj.get("voltage"))
        return OperatingPointExcitation(frequency, current, magneticFieldStrength, magneticFluxDensity, magnetizingCurrent, name, voltage)

    def to_dict(self) -> dict:
        result: dict = {}
        result["frequency"] = to_float(self.frequency)
        if self.current is not None:
            result["current"] = from_union([lambda x: to_class(SignalDescriptor, x), from_none], self.current)
        if self.magneticFieldStrength is not None:
            result["magneticFieldStrength"] = from_union([lambda x: to_class(SignalDescriptor, x), from_none], self.magneticFieldStrength)
        if self.magneticFluxDensity is not None:
            result["magneticFluxDensity"] = from_union([lambda x: to_class(SignalDescriptor, x), from_none], self.magneticFluxDensity)
        if self.magnetizingCurrent is not None:
            result["magnetizingCurrent"] = from_union([lambda x: to_class(SignalDescriptor, x), from_none], self.magnetizingCurrent)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.voltage is not None:
            result["voltage"] = from_union([lambda x: to_class(SignalDescriptor, x), from_none], self.voltage)
        return result


@dataclass
class OperatingPoint:
    """Data describing one operating point, including the operating conditions and the
    excitations for all ports
    
    Excitation of the current per winding that produced the winding losses
    """
    conditions: OperatingConditions
    excitationsPerWinding: List[OperatingPointExcitation]
    name: Optional[str] = None
    """Name describing this operating point"""

    @staticmethod
    def from_dict(obj: Any) -> 'OperatingPoint':
        assert isinstance(obj, dict)
        conditions = OperatingConditions.from_dict(obj.get("conditions"))
        excitationsPerWinding = from_list(OperatingPointExcitation.from_dict, obj.get("excitationsPerWinding"))
        name = from_union([from_str, from_none], obj.get("name"))
        return OperatingPoint(conditions, excitationsPerWinding, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["conditions"] = to_class(OperatingConditions, self.conditions)
        result["excitationsPerWinding"] = from_list(lambda x: to_class(OperatingPointExcitation, x), self.excitationsPerWinding)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        return result


@dataclass
class Inputs:
    """The description of the inputs that can be used to design a Magnetic"""

    designRequirements: DesignRequirements
    """Data describing the design requirements"""

    operatingPoints: List[OperatingPoint]
    """Data describing the operating points"""

    @staticmethod
    def from_dict(obj: Any) -> 'Inputs':
        assert isinstance(obj, dict)
        designRequirements = DesignRequirements.from_dict(obj.get("designRequirements"))
        operatingPoints = from_list(OperatingPoint.from_dict, obj.get("operatingPoints"))
        return Inputs(designRequirements, operatingPoints)

    def to_dict(self) -> dict:
        result: dict = {}
        result["designRequirements"] = to_class(DesignRequirements, self.designRequirements)
        result["operatingPoints"] = from_list(lambda x: to_class(OperatingPoint, x), self.operatingPoints)
        return result


@dataclass
class DistributorInfo:
    """Data from the distributor for a given part"""

    name: str
    """The name of the distributor of the part"""

    quantity: float
    """The number of individual pieces available in the distributor"""

    reference: str
    """The distributor's reference of this part"""

    cost: Optional[float] = None
    """The distributor's price for this part"""

    country: Optional[str] = None
    """The country of the distributor of the part"""

    distributedArea: Optional[str] = None
    """The area where the distributor doistributes"""

    email: Optional[str] = None
    """The distributor's email"""

    link: Optional[str] = None
    """The distributor's link"""

    phone: Optional[str] = None
    """The distributor's phone"""

    updatedAt: Optional[str] = None
    """The date that this information was updated"""

    @staticmethod
    def from_dict(obj: Any) -> 'DistributorInfo':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        quantity = from_float(obj.get("quantity"))
        reference = from_str(obj.get("reference"))
        cost = from_union([from_float, from_none], obj.get("cost"))
        country = from_union([from_str, from_none], obj.get("country"))
        distributedArea = from_union([from_str, from_none], obj.get("distributedArea"))
        email = from_union([from_str, from_none], obj.get("email"))
        link = from_union([from_str, from_none], obj.get("link"))
        phone = from_union([from_str, from_none], obj.get("phone"))
        updatedAt = from_union([from_str, from_none], obj.get("updatedAt"))
        return DistributorInfo(name, quantity, reference, cost, country, distributedArea, email, link, phone, updatedAt)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["quantity"] = to_float(self.quantity)
        result["reference"] = from_str(self.reference)
        if self.cost is not None:
            result["cost"] = from_union([to_float, from_none], self.cost)
        if self.country is not None:
            result["country"] = from_union([from_str, from_none], self.country)
        if self.distributedArea is not None:
            result["distributedArea"] = from_union([from_str, from_none], self.distributedArea)
        if self.email is not None:
            result["email"] = from_union([from_str, from_none], self.email)
        if self.link is not None:
            result["link"] = from_union([from_str, from_none], self.link)
        if self.phone is not None:
            result["phone"] = from_union([from_str, from_none], self.phone)
        if self.updatedAt is not None:
            result["updatedAt"] = from_union([from_str, from_none], self.updatedAt)
        return result


@dataclass
class PinWIndingConnection:
    pin: Optional[str] = None
    """The name of the connected pin"""

    winding: Optional[str] = None
    """The name of the connected winding"""

    @staticmethod
    def from_dict(obj: Any) -> 'PinWIndingConnection':
        assert isinstance(obj, dict)
        pin = from_union([from_str, from_none], obj.get("pin"))
        winding = from_union([from_str, from_none], obj.get("winding"))
        return PinWIndingConnection(pin, winding)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.pin is not None:
            result["pin"] = from_union([from_str, from_none], self.pin)
        if self.winding is not None:
            result["winding"] = from_union([from_str, from_none], self.winding)
        return result


class BobbinFamily(Enum):
    """The family of a bobbin"""

    e = "e"
    ec = "ec"
    efd = "efd"
    el = "el"
    ep = "ep"
    er = "er"
    etd = "etd"
    p = "p"
    pm = "pm"
    pq = "pq"
    rm = "rm"
    u = "u"


class PinShape(Enum):
    """The shape of the pin"""

    irregular = "irregular"
    rectangular = "rectangular"
    round = "round"


class PinDescriptionType(Enum):
    """Type of pin"""

    smd = "smd"
    tht = "tht"


@dataclass
class Pin:
    """Data describing one pin in a bobbin"""

    dimensions: List[float]
    """Dimensions of the rectangle defining the pin"""

    shape: PinShape
    """The shape of the pin"""

    type: PinDescriptionType
    """Type of pin"""

    coordinates: Optional[List[float]] = None
    """The coordinates of the center of the pin, referred to the center of the main column"""

    name: Optional[str] = None
    """Name given to the pin"""

    rotation: Optional[List[float]] = None
    """The rotation of the pin, default is vertical"""

    @staticmethod
    def from_dict(obj: Any) -> 'Pin':
        assert isinstance(obj, dict)
        dimensions = from_list(from_float, obj.get("dimensions"))
        shape = PinShape(obj.get("shape"))
        type = PinDescriptionType(obj.get("type"))
        coordinates = from_union([lambda x: from_list(from_float, x), from_none], obj.get("coordinates"))
        name = from_union([from_str, from_none], obj.get("name"))
        rotation = from_union([lambda x: from_list(from_float, x), from_none], obj.get("rotation"))
        return Pin(dimensions, shape, type, coordinates, name, rotation)

    def to_dict(self) -> dict:
        result: dict = {}
        result["dimensions"] = from_list(to_float, self.dimensions)
        result["shape"] = to_enum(PinShape, self.shape)
        result["type"] = to_enum(PinDescriptionType, self.type)
        if self.coordinates is not None:
            result["coordinates"] = from_union([lambda x: from_list(to_float, x), from_none], self.coordinates)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.rotation is not None:
            result["rotation"] = from_union([lambda x: from_list(to_float, x), from_none], self.rotation)
        return result


@dataclass
class Pinout:
    """Data describing the pinout of a bobbin"""

    numberPins: int
    """The number of pins"""

    pinDescription: Pin
    pitch: List[float]
    """The distance between pins, per row, by pin order"""

    rowDistance: float
    """The distance between a row of pins and the center of the bobbin"""

    centralPitch: Optional[float] = None
    """The distance between central pins"""

    numberPinsPerRow: Optional[List[int]] = None
    """List of pins per row"""

    numberRows: Optional[int] = None
    """The number of rows of a bobbin, typically 2"""

    @staticmethod
    def from_dict(obj: Any) -> 'Pinout':
        assert isinstance(obj, dict)
        numberPins = from_int(obj.get("numberPins"))
        pinDescription = Pin.from_dict(obj.get("pinDescription"))
        pitch = from_list(from_float, obj.get("pitch"))
        rowDistance = from_float(obj.get("rowDistance"))
        centralPitch = from_union([from_float, from_none], obj.get("centralPitch"))
        numberPinsPerRow = from_union([lambda x: from_list(from_int, x), from_none], obj.get("numberPinsPerRow"))
        numberRows = from_union([from_int, from_none], obj.get("numberRows"))
        return Pinout(numberPins, pinDescription, pitch, rowDistance, centralPitch, numberPinsPerRow, numberRows)

    def to_dict(self) -> dict:
        result: dict = {}
        result["numberPins"] = from_int(self.numberPins)
        result["pinDescription"] = to_class(Pin, self.pinDescription)
        result["pitch"] = from_list(to_float, self.pitch)
        result["rowDistance"] = to_float(self.rowDistance)
        if self.centralPitch is not None:
            result["centralPitch"] = from_union([to_float, from_none], self.centralPitch)
        if self.numberPinsPerRow is not None:
            result["numberPinsPerRow"] = from_union([lambda x: from_list(from_int, x), from_none], self.numberPinsPerRow)
        if self.numberRows is not None:
            result["numberRows"] = from_union([from_int, from_none], self.numberRows)
        return result


class FunctionalDescriptionType(Enum):
    """The type of a bobbin
    
    The type of a magnetic shape
    """
    custom = "custom"
    standard = "standard"


@dataclass
class BobbinFunctionalDescription:
    """The data from the bobbin based on its function, in a way that can be used by analytical
    models.
    """
    dimensions: Dict[str, Union[DimensionWithTolerance, float]]
    """The dimensions of a bobbin, keys must be as defined in EN 62317"""

    family: BobbinFamily
    """The family of a bobbin"""

    shape: str
    """The name of a bobbin that this bobbin belongs to"""

    type: FunctionalDescriptionType
    """The type of a bobbin"""

    connections: Optional[List[PinWIndingConnection]] = None
    """List of connections between windings and pins"""

    familySubtype: Optional[str] = None
    """The subtype of the shape, in case there are more than one"""

    pinout: Optional[Pinout] = None

    @staticmethod
    def from_dict(obj: Any) -> 'BobbinFunctionalDescription':
        assert isinstance(obj, dict)
        dimensions = from_dict(lambda x: from_union([DimensionWithTolerance.from_dict, from_float], x), obj.get("dimensions"))
        family = BobbinFamily(obj.get("family"))
        shape = from_str(obj.get("shape"))
        type = FunctionalDescriptionType(obj.get("type"))
        connections = from_union([lambda x: from_list(PinWIndingConnection.from_dict, x), from_none], obj.get("connections"))
        familySubtype = from_union([from_str, from_none], obj.get("familySubtype"))
        pinout = from_union([Pinout.from_dict, from_none], obj.get("pinout"))
        return BobbinFunctionalDescription(dimensions, family, shape, type, connections, familySubtype, pinout)

    def to_dict(self) -> dict:
        result: dict = {}
        result["dimensions"] = from_dict(lambda x: from_union([lambda x: to_class(DimensionWithTolerance, x), to_float], x), self.dimensions)
        result["family"] = to_enum(BobbinFamily, self.family)
        result["shape"] = from_str(self.shape)
        result["type"] = to_enum(FunctionalDescriptionType, self.type)
        if self.connections is not None:
            result["connections"] = from_union([lambda x: from_list(lambda x: to_class(PinWIndingConnection, x), x), from_none], self.connections)
        if self.familySubtype is not None:
            result["familySubtype"] = from_union([from_str, from_none], self.familySubtype)
        if self.pinout is not None:
            result["pinout"] = from_union([lambda x: to_class(Pinout, x), from_none], self.pinout)
        return result


class Status(Enum):
    """The production status of a part according to its manufacturer"""

    obsolete = "obsolete"
    production = "production"
    prototype = "prototype"


@dataclass
class ManufacturerInfo:
    """Data from the manufacturer for a given part"""

    name: str
    """The name of the manufacturer of the part"""

    cost: Optional[str] = None
    """The manufacturer's price for this part"""

    datasheetUrl: Optional[str] = None
    """The manufacturer's URL to the datasheet of the product"""

    family: Optional[str] = None
    """The family of a magnetic, as defined by the manufacturer"""

    orderCode: Optional[str] = None
    """The manufacturer's order code of this part"""

    reference: Optional[str] = None
    """The manufacturer's reference of this part"""

    status: Optional[Status] = None
    """The production status of a part according to its manufacturer"""

    @staticmethod
    def from_dict(obj: Any) -> 'ManufacturerInfo':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        cost = from_union([from_str, from_none], obj.get("cost"))
        datasheetUrl = from_union([from_str, from_none], obj.get("datasheetUrl"))
        family = from_union([from_str, from_none], obj.get("family"))
        orderCode = from_union([from_str, from_none], obj.get("orderCode"))
        reference = from_union([from_str, from_none], obj.get("reference"))
        status = from_union([Status, from_none], obj.get("status"))
        return ManufacturerInfo(name, cost, datasheetUrl, family, orderCode, reference, status)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        if self.cost is not None:
            result["cost"] = from_union([from_str, from_none], self.cost)
        if self.datasheetUrl is not None:
            result["datasheetUrl"] = from_union([from_str, from_none], self.datasheetUrl)
        if self.family is not None:
            result["family"] = from_union([from_str, from_none], self.family)
        if self.orderCode is not None:
            result["orderCode"] = from_union([from_str, from_none], self.orderCode)
        if self.reference is not None:
            result["reference"] = from_union([from_str, from_none], self.reference)
        if self.status is not None:
            result["status"] = from_union([lambda x: to_enum(Status, x), from_none], self.status)
        return result


class ColumnShape(Enum):
    """Shape of the column, also used for gaps"""

    irregular = "irregular"
    oblong = "oblong"
    rectangular = "rectangular"
    round = "round"


class CoilAlignment(Enum):
    """Way in which the sections are aligned inside the winding window
    
    Way in which the turns are aligned inside the layer
    
    Way in which the layers are aligned inside the section
    """
    centered = "centered"
    innerortop = "inner or top"
    outerorbottom = "outer or bottom"
    spread = "spread"


class WindingOrientation(Enum):
    """Way in which the sections are oriented inside the winding window
    
    Way in which the layer is oriented inside the section
    
    Way in which the layers are oriented inside the section
    """
    contiguous = "contiguous"
    overlapping = "overlapping"


class WindingWindowShape(Enum):
    rectangular = "rectangular"
    round = "round"


@dataclass
class WindingWindowElement:
    """List of rectangular winding windows
    
    It is the area between the winding column and the closest lateral column, and it
    represents the area where all the wires of the magnetic will have to fit, and
    equivalently, where all the current must circulate once, in the case of inductors, or
    twice, in the case of transformers
    
    List of radial winding windows
    
    It is the area between the delimited between a height from the surface of the toroidal
    core at a given angle, and it represents the area where all the wires of the magnetic
    will have to fit, and equivalently, where all the current must circulate once, in the
    case of inductors, or twice, in the case of transformers
    """
    area: Optional[float] = None
    """Area of the winding window"""

    coordinates: Optional[List[float]] = None
    """The coordinates of the center of the winding window, referred to the center of the main
    column. In the case of half-sets, the center will be in the top point, where it would
    join another half-set
    
    The coordinates of the point of the winding window where the middle height touches the
    main column, referred to the center of the main column. In the case of half-sets, the
    center will be in the top point, where it would join another half-set
    """
    height: Optional[float] = None
    """Vertical height of the winding window"""

    sectionsAlignment: Optional[CoilAlignment] = None
    """Way in which the sections are aligned inside the winding window"""

    sectionsOrientation: Optional[WindingOrientation] = None
    """Way in which the sections are oriented inside the winding window"""

    shape: Optional[WindingWindowShape] = None
    """Shape of the winding window"""

    width: Optional[float] = None
    """Horizontal width of the winding window"""

    angle: Optional[float] = None
    """Total angle of the window"""

    radialHeight: Optional[float] = None
    """Radial height of the winding window"""

    @staticmethod
    def from_dict(obj: Any) -> 'WindingWindowElement':
        assert isinstance(obj, dict)
        area = from_union([from_float, from_none], obj.get("area"))
        coordinates = from_union([lambda x: from_list(from_float, x), from_none], obj.get("coordinates"))
        height = from_union([from_float, from_none], obj.get("height"))
        sectionsAlignment = from_union([CoilAlignment, from_none], obj.get("sectionsAlignment"))
        sectionsOrientation = from_union([WindingOrientation, from_none], obj.get("sectionsOrientation"))
        shape = from_union([WindingWindowShape, from_none], obj.get("shape"))
        width = from_union([from_float, from_none], obj.get("width"))
        angle = from_union([from_float, from_none], obj.get("angle"))
        radialHeight = from_union([from_float, from_none], obj.get("radialHeight"))
        return WindingWindowElement(area, coordinates, height, sectionsAlignment, sectionsOrientation, shape, width, angle, radialHeight)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.area is not None:
            result["area"] = from_union([to_float, from_none], self.area)
        if self.coordinates is not None:
            result["coordinates"] = from_union([lambda x: from_list(to_float, x), from_none], self.coordinates)
        if self.height is not None:
            result["height"] = from_union([to_float, from_none], self.height)
        if self.sectionsAlignment is not None:
            result["sectionsAlignment"] = from_union([lambda x: to_enum(CoilAlignment, x), from_none], self.sectionsAlignment)
        if self.sectionsOrientation is not None:
            result["sectionsOrientation"] = from_union([lambda x: to_enum(WindingOrientation, x), from_none], self.sectionsOrientation)
        if self.shape is not None:
            result["shape"] = from_union([lambda x: to_enum(WindingWindowShape, x), from_none], self.shape)
        if self.width is not None:
            result["width"] = from_union([to_float, from_none], self.width)
        if self.angle is not None:
            result["angle"] = from_union([to_float, from_none], self.angle)
        if self.radialHeight is not None:
            result["radialHeight"] = from_union([to_float, from_none], self.radialHeight)
        return result


@dataclass
class CoreBobbinProcessedDescription:
    columnDepth: float
    """The depth of the central column wall, including thickness, in the z axis"""

    columnShape: ColumnShape
    columnThickness: float
    """The thicknes of the central column wall, where the wire is wound, in the X axis"""

    wallThickness: float
    """The thicknes of the walls that hold the wire on both sides of the column"""

    windingWindows: List[WindingWindowElement]
    """List of winding windows, all elements in the list must be of the same type"""

    columnWidth: Optional[float] = None
    """The width of the central column wall, including thickness, in the x axis"""

    coordinates: Optional[List[float]] = None
    """The coordinates of the center of the bobbin central wall, whre the wires are wound,
    referred to the center of the main column.
    """
    pins: Optional[List[Pin]] = None
    """List of pins, geometrically defining how and where it is"""

    @staticmethod
    def from_dict(obj: Any) -> 'CoreBobbinProcessedDescription':
        assert isinstance(obj, dict)
        columnDepth = from_float(obj.get("columnDepth"))
        columnShape = ColumnShape(obj.get("columnShape"))
        columnThickness = from_float(obj.get("columnThickness"))
        wallThickness = from_float(obj.get("wallThickness"))
        windingWindows = from_list(WindingWindowElement.from_dict, obj.get("windingWindows"))
        columnWidth = from_union([from_float, from_none], obj.get("columnWidth"))
        coordinates = from_union([lambda x: from_list(from_float, x), from_none], obj.get("coordinates"))
        pins = from_union([lambda x: from_list(Pin.from_dict, x), from_none], obj.get("pins"))
        return CoreBobbinProcessedDescription(columnDepth, columnShape, columnThickness, wallThickness, windingWindows, columnWidth, coordinates, pins)

    def to_dict(self) -> dict:
        result: dict = {}
        result["columnDepth"] = to_float(self.columnDepth)
        result["columnShape"] = to_enum(ColumnShape, self.columnShape)
        result["columnThickness"] = to_float(self.columnThickness)
        result["wallThickness"] = to_float(self.wallThickness)
        result["windingWindows"] = from_list(lambda x: to_class(WindingWindowElement, x), self.windingWindows)
        if self.columnWidth is not None:
            result["columnWidth"] = from_union([to_float, from_none], self.columnWidth)
        if self.coordinates is not None:
            result["coordinates"] = from_union([lambda x: from_list(to_float, x), from_none], self.coordinates)
        if self.pins is not None:
            result["pins"] = from_union([lambda x: from_list(lambda x: to_class(Pin, x), x), from_none], self.pins)
        return result


@dataclass
class Bobbin:
    """The description of a bobbin"""

    distributorsInfo: Optional[List[DistributorInfo]] = None
    """The lists of distributors of the magnetic bobbin"""

    functionalDescription: Optional[BobbinFunctionalDescription] = None
    """The data from the bobbin based on its function, in a way that can be used by analytical
    models.
    """
    manufacturerInfo: Optional[ManufacturerInfo] = None
    name: Optional[str] = None
    """The name of bobbin"""

    processedDescription: Optional[CoreBobbinProcessedDescription] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Bobbin':
        assert isinstance(obj, dict)
        distributorsInfo = from_union([lambda x: from_list(DistributorInfo.from_dict, x), from_none], obj.get("distributorsInfo"))
        functionalDescription = from_union([BobbinFunctionalDescription.from_dict, from_none], obj.get("functionalDescription"))
        manufacturerInfo = from_union([ManufacturerInfo.from_dict, from_none], obj.get("manufacturerInfo"))
        name = from_union([from_str, from_none], obj.get("name"))
        processedDescription = from_union([CoreBobbinProcessedDescription.from_dict, from_none], obj.get("processedDescription"))
        return Bobbin(distributorsInfo, functionalDescription, manufacturerInfo, name, processedDescription)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.distributorsInfo is not None:
            result["distributorsInfo"] = from_union([lambda x: from_list(lambda x: to_class(DistributorInfo, x), x), from_none], self.distributorsInfo)
        if self.functionalDescription is not None:
            result["functionalDescription"] = from_union([lambda x: to_class(BobbinFunctionalDescription, x), from_none], self.functionalDescription)
        if self.manufacturerInfo is not None:
            result["manufacturerInfo"] = from_union([lambda x: to_class(ManufacturerInfo, x), from_none], self.manufacturerInfo)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.processedDescription is not None:
            result["processedDescription"] = from_union([lambda x: to_class(CoreBobbinProcessedDescription, x), from_none], self.processedDescription)
        return result


@dataclass
class ConnectionElement:
    """Data describing the connection of the a wire"""

    length: Optional[float] = None
    """Length of the connection, counted from the exit of the last turn until the terminal, in m"""

    metric: Optional[int] = None
    """Metric of the terminal, if applicable"""

    pinName: Optional[str] = None
    """Name of the pin where it is connected, if applicable"""

    type: Optional[ConnectionType] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ConnectionElement':
        assert isinstance(obj, dict)
        length = from_union([from_float, from_none], obj.get("length"))
        metric = from_union([from_int, from_none], obj.get("metric"))
        pinName = from_union([from_str, from_none], obj.get("pinName"))
        type = from_union([ConnectionType, from_none], obj.get("type"))
        return ConnectionElement(length, metric, pinName, type)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.length is not None:
            result["length"] = from_union([to_float, from_none], self.length)
        if self.metric is not None:
            result["metric"] = from_union([from_int, from_none], self.metric)
        if self.pinName is not None:
            result["pinName"] = from_union([from_str, from_none], self.pinName)
        if self.type is not None:
            result["type"] = from_union([lambda x: to_enum(ConnectionType, x), from_none], self.type)
        return result


@dataclass
class DielectricStrengthElement:
    """data for describing one point of dieletric strength"""

    value: float
    """Dieletric strength value, in V / m"""

    humidity: Optional[float] = None
    """Humidity for the field value, in proportion over 1"""

    temperature: Optional[float] = None
    """Temperature for the field value, in Celsius"""

    thickness: Optional[float] = None
    """Thickness of the material"""

    @staticmethod
    def from_dict(obj: Any) -> 'DielectricStrengthElement':
        assert isinstance(obj, dict)
        value = from_float(obj.get("value"))
        humidity = from_union([from_float, from_none], obj.get("humidity"))
        temperature = from_union([from_float, from_none], obj.get("temperature"))
        thickness = from_union([from_float, from_none], obj.get("thickness"))
        return DielectricStrengthElement(value, humidity, temperature, thickness)

    def to_dict(self) -> dict:
        result: dict = {}
        result["value"] = to_float(self.value)
        if self.humidity is not None:
            result["humidity"] = from_union([to_float, from_none], self.humidity)
        if self.temperature is not None:
            result["temperature"] = from_union([to_float, from_none], self.temperature)
        if self.thickness is not None:
            result["thickness"] = from_union([to_float, from_none], self.thickness)
        return result


@dataclass
class ResistivityPoint:
    """data for describing one point of resistivity"""

    value: float
    """Resistivity value, in Ohm * m"""

    temperature: Optional[float] = None
    """temperature for the field value, in Celsius"""

    @staticmethod
    def from_dict(obj: Any) -> 'ResistivityPoint':
        assert isinstance(obj, dict)
        value = from_float(obj.get("value"))
        temperature = from_union([from_float, from_none], obj.get("temperature"))
        return ResistivityPoint(value, temperature)

    def to_dict(self) -> dict:
        result: dict = {}
        result["value"] = to_float(self.value)
        if self.temperature is not None:
            result["temperature"] = from_union([to_float, from_none], self.temperature)
        return result


@dataclass
class InsulationMaterial:
    """A material for insulation"""

    dielectricStrength: List[DielectricStrengthElement]
    name: str
    """The name of a insulation material"""

    aliases: Optional[List[str]] = None
    """Alternative names of the material"""

    composition: Optional[str] = None
    """The composition of a insulation material"""

    manufacturer: Optional[str] = None
    """The manufacturer of the insulation material"""

    meltingPoint: Optional[float] = None
    """The melting temperature of the insulation material, in Celsius"""

    relativePermittivity: Optional[float] = None
    """The dielectric constant of the insulation material"""

    resistivity: Optional[List[ResistivityPoint]] = None
    """Resistivity value according to manufacturer"""

    specificHeat: Optional[float] = None
    """The specific heat of the insulation material, in J / (Kg * K)"""

    temperatureClass: Optional[float] = None
    """The temperature class of the insulation material, in Celsius"""

    thermalConductivity: Optional[float] = None
    """The thermal conductivity of the insulation material, in W / (m * K)"""

    @staticmethod
    def from_dict(obj: Any) -> 'InsulationMaterial':
        assert isinstance(obj, dict)
        dielectricStrength = from_list(DielectricStrengthElement.from_dict, obj.get("dielectricStrength"))
        name = from_str(obj.get("name"))
        aliases = from_union([lambda x: from_list(from_str, x), from_none], obj.get("aliases"))
        composition = from_union([from_str, from_none], obj.get("composition"))
        manufacturer = from_union([from_str, from_none], obj.get("manufacturer"))
        meltingPoint = from_union([from_float, from_none], obj.get("meltingPoint"))
        relativePermittivity = from_union([from_float, from_none], obj.get("relativePermittivity"))
        resistivity = from_union([lambda x: from_list(ResistivityPoint.from_dict, x), from_none], obj.get("resistivity"))
        specificHeat = from_union([from_float, from_none], obj.get("specificHeat"))
        temperatureClass = from_union([from_float, from_none], obj.get("temperatureClass"))
        thermalConductivity = from_union([from_float, from_none], obj.get("thermalConductivity"))
        return InsulationMaterial(dielectricStrength, name, aliases, composition, manufacturer, meltingPoint, relativePermittivity, resistivity, specificHeat, temperatureClass, thermalConductivity)

    def to_dict(self) -> dict:
        result: dict = {}
        result["dielectricStrength"] = from_list(lambda x: to_class(DielectricStrengthElement, x), self.dielectricStrength)
        result["name"] = from_str(self.name)
        if self.aliases is not None:
            result["aliases"] = from_union([lambda x: from_list(from_str, x), from_none], self.aliases)
        if self.composition is not None:
            result["composition"] = from_union([from_str, from_none], self.composition)
        if self.manufacturer is not None:
            result["manufacturer"] = from_union([from_str, from_none], self.manufacturer)
        if self.meltingPoint is not None:
            result["meltingPoint"] = from_union([to_float, from_none], self.meltingPoint)
        if self.relativePermittivity is not None:
            result["relativePermittivity"] = from_union([to_float, from_none], self.relativePermittivity)
        if self.resistivity is not None:
            result["resistivity"] = from_union([lambda x: from_list(lambda x: to_class(ResistivityPoint, x), x), from_none], self.resistivity)
        if self.specificHeat is not None:
            result["specificHeat"] = from_union([to_float, from_none], self.specificHeat)
        if self.temperatureClass is not None:
            result["temperatureClass"] = from_union([to_float, from_none], self.temperatureClass)
        if self.thermalConductivity is not None:
            result["thermalConductivity"] = from_union([to_float, from_none], self.thermalConductivity)
        return result


class InsulationWireCoatingType(Enum):
    """The type of the coating"""

    bare = "bare"
    enamelled = "enamelled"
    extruded = "extruded"
    insulated = "insulated"
    served = "served"
    taped = "taped"


@dataclass
class InsulationWireCoating:
    """A coating for a wire"""

    breakdownVoltage: Optional[float] = None
    """The minimum voltage that causes a portion of an insulator to experience electrical
    breakdown and become electrically conductive, in V
    """
    grade: Optional[int] = None
    """The grade of the insulation around the wire"""

    material: Optional[Union[InsulationMaterial, str]] = None
    numberLayers: Optional[int] = None
    """The number of layers of the insulation around the wire"""

    temperatureRating: Optional[float] = None
    """The maximum temperature that the wire coating can withstand"""

    thickness: Optional[DimensionWithTolerance] = None
    """The maximum thickness of the insulation around the wire, in m"""

    thicknessLayers: Optional[float] = None
    """The thickness of the layers of the insulation around the wire, in m"""

    type: Optional[InsulationWireCoatingType] = None
    """The type of the coating"""

    @staticmethod
    def from_dict(obj: Any) -> 'InsulationWireCoating':
        assert isinstance(obj, dict)
        breakdownVoltage = from_union([from_float, from_none], obj.get("breakdownVoltage"))
        grade = from_union([from_int, from_none], obj.get("grade"))
        material = from_union([InsulationMaterial.from_dict, from_str, from_none], obj.get("material"))
        numberLayers = from_union([from_int, from_none], obj.get("numberLayers"))
        temperatureRating = from_union([from_float, from_none], obj.get("temperatureRating"))
        thickness = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("thickness"))
        thicknessLayers = from_union([from_float, from_none], obj.get("thicknessLayers"))
        type = from_union([InsulationWireCoatingType, from_none], obj.get("type"))
        return InsulationWireCoating(breakdownVoltage, grade, material, numberLayers, temperatureRating, thickness, thicknessLayers, type)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.breakdownVoltage is not None:
            result["breakdownVoltage"] = from_union([to_float, from_none], self.breakdownVoltage)
        if self.grade is not None:
            result["grade"] = from_union([from_int, from_none], self.grade)
        if self.material is not None:
            result["material"] = from_union([lambda x: to_class(InsulationMaterial, x), from_str, from_none], self.material)
        if self.numberLayers is not None:
            result["numberLayers"] = from_union([from_int, from_none], self.numberLayers)
        if self.temperatureRating is not None:
            result["temperatureRating"] = from_union([to_float, from_none], self.temperatureRating)
        if self.thickness is not None:
            result["thickness"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.thickness)
        if self.thicknessLayers is not None:
            result["thicknessLayers"] = from_union([to_float, from_none], self.thicknessLayers)
        if self.type is not None:
            result["type"] = from_union([lambda x: to_enum(InsulationWireCoatingType, x), from_none], self.type)
        return result


@dataclass
class Resistivity:
    """data for describing the resistivity of a wire"""

    referenceTemperature: float
    """Temperature reference value, in Celsius"""

    referenceValue: float
    """Resistivity reference value, in Ohm * m"""

    temperatureCoefficient: float
    """Temperature coefficient value, alpha, in 1 / Celsius"""

    @staticmethod
    def from_dict(obj: Any) -> 'Resistivity':
        assert isinstance(obj, dict)
        referenceTemperature = from_float(obj.get("referenceTemperature"))
        referenceValue = from_float(obj.get("referenceValue"))
        temperatureCoefficient = from_float(obj.get("temperatureCoefficient"))
        return Resistivity(referenceTemperature, referenceValue, temperatureCoefficient)

    def to_dict(self) -> dict:
        result: dict = {}
        result["referenceTemperature"] = to_float(self.referenceTemperature)
        result["referenceValue"] = to_float(self.referenceValue)
        result["temperatureCoefficient"] = to_float(self.temperatureCoefficient)
        return result


@dataclass
class ThermalConductivityElement:
    """data for describing one point of thermal conductivity"""

    temperature: float
    """Temperature for the field value, in Celsius"""

    value: float
    """Thermal conductivity value, in W / m * K"""

    @staticmethod
    def from_dict(obj: Any) -> 'ThermalConductivityElement':
        assert isinstance(obj, dict)
        temperature = from_float(obj.get("temperature"))
        value = from_float(obj.get("value"))
        return ThermalConductivityElement(temperature, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["temperature"] = to_float(self.temperature)
        result["value"] = to_float(self.value)
        return result


@dataclass
class WireMaterial:
    """A material for wire"""

    name: str
    """The name of a wire material"""

    permeability: float
    """The permeability of a wire material"""

    resistivity: Resistivity
    thermalConductivity: Optional[List[ThermalConductivityElement]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'WireMaterial':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        permeability = from_float(obj.get("permeability"))
        resistivity = Resistivity.from_dict(obj.get("resistivity"))
        thermalConductivity = from_union([lambda x: from_list(ThermalConductivityElement.from_dict, x), from_none], obj.get("thermalConductivity"))
        return WireMaterial(name, permeability, resistivity, thermalConductivity)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["permeability"] = to_float(self.permeability)
        result["resistivity"] = to_class(Resistivity, self.resistivity)
        if self.thermalConductivity is not None:
            result["thermalConductivity"] = from_union([lambda x: from_list(lambda x: to_class(ThermalConductivityElement, x), x), from_none], self.thermalConductivity)
        return result


class WireStandard(Enum):
    """The standard of wire"""

    IEC60317 = "IEC 60317"
    IPC6012 = "IPC-6012"
    NEMAMW1000C = "NEMA MW 1000 C"


class WireType(Enum):
    """The type of wire"""

    foil = "foil"
    litz = "litz"
    planar = "planar"
    rectangular = "rectangular"
    round = "round"


@dataclass
class WireRound:
    """The description of a solid round magnet wire
    
    The description of a basic magnet wire
    """
    conductingDiameter: DimensionWithTolerance
    """The conducting diameter of the wire, in m"""

    type: WireType
    material: Optional[Union[WireMaterial, str]] = None
    outerDiameter: Optional[DimensionWithTolerance] = None
    """The outer diameter of the wire, in m"""

    coating: Optional[Union[InsulationWireCoating, str]] = None
    conductingArea: Optional[DimensionWithTolerance] = None
    """The conducting area of the wire, in m². Used for some rectangular shapes where the area
    is smaller than expected due to rounded corners
    """
    manufacturerInfo: Optional[ManufacturerInfo] = None
    name: Optional[str] = None
    """The name of wire"""

    numberConductors: Optional[int] = None
    """The number of conductors in the wire"""

    standard: Optional[WireStandard] = None
    """The standard of wire"""

    standardName: Optional[str] = None
    """Name according to the standard of wire"""

    @staticmethod
    def from_dict(obj: Any) -> 'WireRound':
        assert isinstance(obj, dict)
        conductingDiameter = DimensionWithTolerance.from_dict(obj.get("conductingDiameter"))
        type = WireType(obj.get("type"))
        material = from_union([WireMaterial.from_dict, from_str, from_none], obj.get("material"))
        outerDiameter = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("outerDiameter"))
        coating = from_union([InsulationWireCoating.from_dict, from_str, from_none], obj.get("coating"))
        conductingArea = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("conductingArea"))
        manufacturerInfo = from_union([ManufacturerInfo.from_dict, from_none], obj.get("manufacturerInfo"))
        name = from_union([from_str, from_none], obj.get("name"))
        numberConductors = from_union([from_int, from_none], obj.get("numberConductors"))
        standard = from_union([WireStandard, from_none], obj.get("standard"))
        standardName = from_union([from_str, from_none], obj.get("standardName"))
        return WireRound(conductingDiameter, type, material, outerDiameter, coating, conductingArea, manufacturerInfo, name, numberConductors, standard, standardName)

    def to_dict(self) -> dict:
        result: dict = {}
        result["conductingDiameter"] = to_class(DimensionWithTolerance, self.conductingDiameter)
        result["type"] = to_enum(WireType, self.type)
        if self.material is not None:
            result["material"] = from_union([lambda x: to_class(WireMaterial, x), from_str, from_none], self.material)
        if self.outerDiameter is not None:
            result["outerDiameter"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.outerDiameter)
        if self.coating is not None:
            result["coating"] = from_union([lambda x: to_class(InsulationWireCoating, x), from_str, from_none], self.coating)
        if self.conductingArea is not None:
            result["conductingArea"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.conductingArea)
        if self.manufacturerInfo is not None:
            result["manufacturerInfo"] = from_union([lambda x: to_class(ManufacturerInfo, x), from_none], self.manufacturerInfo)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.numberConductors is not None:
            result["numberConductors"] = from_union([from_int, from_none], self.numberConductors)
        if self.standard is not None:
            result["standard"] = from_union([lambda x: to_enum(WireStandard, x), from_none], self.standard)
        if self.standardName is not None:
            result["standardName"] = from_union([from_str, from_none], self.standardName)
        return result


@dataclass
class Wire:
    """The description of a solid round magnet wire
    
    The description of a basic magnet wire
    
    The description of a solid foil magnet wire
    
    The description of a solid rectangular magnet wire
    
    The description of a stranded litz magnet wire
    
    The description of a solid planar magnet wire
    """
    type: WireType
    conductingDiameter: Optional[DimensionWithTolerance] = None
    """The conducting diameter of the wire, in m"""

    material: Optional[Union[WireMaterial, str]] = None
    outerDiameter: Optional[DimensionWithTolerance] = None
    """The outer diameter of the wire, in m"""

    coating: Optional[Union[InsulationWireCoating, str]] = None
    conductingArea: Optional[DimensionWithTolerance] = None
    """The conducting area of the wire, in m². Used for some rectangular shapes where the area
    is smaller than expected due to rounded corners
    """
    manufacturerInfo: Optional[ManufacturerInfo] = None
    name: Optional[str] = None
    """The name of wire"""

    numberConductors: Optional[int] = None
    """The number of conductors in the wire"""

    standard: Optional[WireStandard] = None
    """The standard of wire"""

    standardName: Optional[str] = None
    """Name according to the standard of wire"""

    conductingHeight: Optional[DimensionWithTolerance] = None
    """The conducting height of the wire, in m"""

    conductingWidth: Optional[DimensionWithTolerance] = None
    """The conducting width of the wire, in m"""

    outerHeight: Optional[DimensionWithTolerance] = None
    """The outer height of the wire, in m"""

    outerWidth: Optional[DimensionWithTolerance] = None
    """The outer width of the wire, in m"""

    edgeRadius: Optional[DimensionWithTolerance] = None
    """The radius of the edge, in case of rectangular wire, in m"""

    strand: Optional[Union[WireRound, str]] = None
    """The wire used as strands"""

    @staticmethod
    def from_dict(obj: Any) -> 'Wire':
        assert isinstance(obj, dict)
        type = WireType(obj.get("type"))
        conductingDiameter = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("conductingDiameter"))
        material = from_union([WireMaterial.from_dict, from_str, from_none], obj.get("material"))
        outerDiameter = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("outerDiameter"))
        coating = from_union([InsulationWireCoating.from_dict, from_str, from_none], obj.get("coating"))
        conductingArea = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("conductingArea"))
        manufacturerInfo = from_union([ManufacturerInfo.from_dict, from_none], obj.get("manufacturerInfo"))
        name = from_union([from_str, from_none], obj.get("name"))
        numberConductors = from_union([from_int, from_none], obj.get("numberConductors"))
        standard = from_union([WireStandard, from_none], obj.get("standard"))
        standardName = from_union([from_str, from_none], obj.get("standardName"))
        conductingHeight = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("conductingHeight"))
        conductingWidth = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("conductingWidth"))
        outerHeight = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("outerHeight"))
        outerWidth = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("outerWidth"))
        edgeRadius = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("edgeRadius"))
        strand = from_union([WireRound.from_dict, from_str, from_none], obj.get("strand"))
        return Wire(type, conductingDiameter, material, outerDiameter, coating, conductingArea, manufacturerInfo, name, numberConductors, standard, standardName, conductingHeight, conductingWidth, outerHeight, outerWidth, edgeRadius, strand)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = to_enum(WireType, self.type)
        if self.conductingDiameter is not None:
            result["conductingDiameter"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.conductingDiameter)
        if self.material is not None:
            result["material"] = from_union([lambda x: to_class(WireMaterial, x), from_str, from_none], self.material)
        if self.outerDiameter is not None:
            result["outerDiameter"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.outerDiameter)
        if self.coating is not None:
            result["coating"] = from_union([lambda x: to_class(InsulationWireCoating, x), from_str, from_none], self.coating)
        if self.conductingArea is not None:
            result["conductingArea"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.conductingArea)
        if self.manufacturerInfo is not None:
            result["manufacturerInfo"] = from_union([lambda x: to_class(ManufacturerInfo, x), from_none], self.manufacturerInfo)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.numberConductors is not None:
            result["numberConductors"] = from_union([from_int, from_none], self.numberConductors)
        if self.standard is not None:
            result["standard"] = from_union([lambda x: to_enum(WireStandard, x), from_none], self.standard)
        if self.standardName is not None:
            result["standardName"] = from_union([from_str, from_none], self.standardName)
        if self.conductingHeight is not None:
            result["conductingHeight"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.conductingHeight)
        if self.conductingWidth is not None:
            result["conductingWidth"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.conductingWidth)
        if self.outerHeight is not None:
            result["outerHeight"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.outerHeight)
        if self.outerWidth is not None:
            result["outerWidth"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.outerWidth)
        if self.edgeRadius is not None:
            result["edgeRadius"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.edgeRadius)
        if self.strand is not None:
            result["strand"] = from_union([lambda x: to_class(WireRound, x), from_str, from_none], self.strand)
        return result


@dataclass
class CoilFunctionalDescription:
    """Data describing one winding associated with a magnetic"""

    isolationSide: IsolationSide
    name: str
    """Name given to the winding"""

    numberParallels: int
    """Number of parallels in winding"""

    numberTurns: int
    """Number of turns in winding"""

    wire: Union[Wire, str]
    connections: Optional[List[ConnectionElement]] = None
    """Array on elements, representing the all the pins this winding is connected to"""

    @staticmethod
    def from_dict(obj: Any) -> 'CoilFunctionalDescription':
        assert isinstance(obj, dict)
        isolationSide = IsolationSide(obj.get("isolationSide"))
        name = from_str(obj.get("name"))
        numberParallels = from_int(obj.get("numberParallels"))
        numberTurns = from_int(obj.get("numberTurns"))
        wire = from_union([Wire.from_dict, from_str], obj.get("wire"))
        connections = from_union([lambda x: from_list(ConnectionElement.from_dict, x), from_none], obj.get("connections"))
        return CoilFunctionalDescription(isolationSide, name, numberParallels, numberTurns, wire, connections)

    def to_dict(self) -> dict:
        result: dict = {}
        result["isolationSide"] = to_enum(IsolationSide, self.isolationSide)
        result["name"] = from_str(self.name)
        result["numberParallels"] = from_int(self.numberParallels)
        result["numberTurns"] = from_int(self.numberTurns)
        result["wire"] = from_union([lambda x: to_class(Wire, x), from_str], self.wire)
        if self.connections is not None:
            result["connections"] = from_union([lambda x: from_list(lambda x: to_class(ConnectionElement, x), x), from_none], self.connections)
        return result


class CoordinateSystem(Enum):
    """System in which dimension and coordinates are in"""

    cartesian = "cartesian"
    polar = "polar"


@dataclass
class PartialWinding:
    """Data describing one part of winding, described by a list with the proportion of each
    parallel in the winding that is contained here
    """
    parallelsProportion: List[float]
    """Number of parallels in winding"""

    winding: str
    """The name of the winding that this part belongs to"""

    connections: Optional[List[ConnectionElement]] = None
    """Array on two elements, representing the input and output connection for this partial
    winding
    """

    @staticmethod
    def from_dict(obj: Any) -> 'PartialWinding':
        assert isinstance(obj, dict)
        parallelsProportion = from_list(from_float, obj.get("parallelsProportion"))
        winding = from_str(obj.get("winding"))
        connections = from_union([lambda x: from_list(ConnectionElement.from_dict, x), from_none], obj.get("connections"))
        return PartialWinding(parallelsProportion, winding, connections)

    def to_dict(self) -> dict:
        result: dict = {}
        result["parallelsProportion"] = from_list(to_float, self.parallelsProportion)
        result["winding"] = from_str(self.winding)
        if self.connections is not None:
            result["connections"] = from_union([lambda x: from_list(lambda x: to_class(ConnectionElement, x), x), from_none], self.connections)
        return result


class ElectricalType(Enum):
    """Type of the layer"""

    conduction = "conduction"
    insulation = "insulation"
    shielding = "shielding"


class WindingStyle(Enum):
    """Defines if the layer is wound by consecutive turns or parallels
    
    Defines if the section is wound by consecutive turns or parallels
    """
    windByConsecutiveParallels = "windByConsecutiveParallels"
    windByConsecutiveTurns = "windByConsecutiveTurns"


@dataclass
class Layer:
    """Data describing one layer in a magnetic"""

    coordinates: List[float]
    """The coordinates of the center of the layer, referred to the center of the main column"""

    dimensions: List[float]
    """Dimensions of the rectangle defining the layer"""

    name: str
    """Name given to the layer"""

    orientation: WindingOrientation
    """Way in which the layer is oriented inside the section"""

    partialWindings: List[PartialWinding]
    """List of partial windings in this layer"""

    type: ElectricalType
    """Type of the layer"""

    additionalCoordinates: Optional[List[List[float]]] = None
    """List of additional coordinates of the center of the layer, referred to the center of the
    main column, in case the layer is not symmetrical, as in toroids
    """
    coordinateSystem: Optional[CoordinateSystem] = None
    """System in which dimension and coordinates are in"""

    fillingFactor: Optional[float] = None
    """How much space in this layer is used by wires compared to the total"""

    insulationMaterial: Optional[Union[InsulationMaterial, str]] = None
    """In case of insulating layer, the material used"""

    section: Optional[str] = None
    """The name of the section that this layer belongs to"""

    turnsAlignment: Optional[CoilAlignment] = None
    """Way in which the turns are aligned inside the layer"""

    windingStyle: Optional[WindingStyle] = None
    """Defines if the layer is wound by consecutive turns or parallels"""

    @staticmethod
    def from_dict(obj: Any) -> 'Layer':
        assert isinstance(obj, dict)
        coordinates = from_list(from_float, obj.get("coordinates"))
        dimensions = from_list(from_float, obj.get("dimensions"))
        name = from_str(obj.get("name"))
        orientation = WindingOrientation(obj.get("orientation"))
        partialWindings = from_list(PartialWinding.from_dict, obj.get("partialWindings"))
        type = ElectricalType(obj.get("type"))
        additionalCoordinates = from_union([lambda x: from_list(lambda x: from_list(from_float, x), x), from_none], obj.get("additionalCoordinates"))
        coordinateSystem = from_union([CoordinateSystem, from_none], obj.get("coordinateSystem"))
        fillingFactor = from_union([from_float, from_none], obj.get("fillingFactor"))
        insulationMaterial = from_union([InsulationMaterial.from_dict, from_str, from_none], obj.get("insulationMaterial"))
        section = from_union([from_str, from_none], obj.get("section"))
        turnsAlignment = from_union([CoilAlignment, from_none], obj.get("turnsAlignment"))
        windingStyle = from_union([WindingStyle, from_none], obj.get("windingStyle"))
        return Layer(coordinates, dimensions, name, orientation, partialWindings, type, additionalCoordinates, coordinateSystem, fillingFactor, insulationMaterial, section, turnsAlignment, windingStyle)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["dimensions"] = from_list(to_float, self.dimensions)
        result["name"] = from_str(self.name)
        result["orientation"] = to_enum(WindingOrientation, self.orientation)
        result["partialWindings"] = from_list(lambda x: to_class(PartialWinding, x), self.partialWindings)
        result["type"] = to_enum(ElectricalType, self.type)
        if self.additionalCoordinates is not None:
            result["additionalCoordinates"] = from_union([lambda x: from_list(lambda x: from_list(to_float, x), x), from_none], self.additionalCoordinates)
        if self.coordinateSystem is not None:
            result["coordinateSystem"] = from_union([lambda x: to_enum(CoordinateSystem, x), from_none], self.coordinateSystem)
        if self.fillingFactor is not None:
            result["fillingFactor"] = from_union([to_float, from_none], self.fillingFactor)
        if self.insulationMaterial is not None:
            result["insulationMaterial"] = from_union([lambda x: to_class(InsulationMaterial, x), from_str, from_none], self.insulationMaterial)
        if self.section is not None:
            result["section"] = from_union([from_str, from_none], self.section)
        if self.turnsAlignment is not None:
            result["turnsAlignment"] = from_union([lambda x: to_enum(CoilAlignment, x), from_none], self.turnsAlignment)
        if self.windingStyle is not None:
            result["windingStyle"] = from_union([lambda x: to_enum(WindingStyle, x), from_none], self.windingStyle)
        return result


@dataclass
class Section:
    """Data describing one section in a magnetic"""

    coordinates: List[float]
    """The coordinates of the center of the section, referred to the center of the main column"""

    dimensions: List[float]
    """Dimensions of the rectangle defining the section"""

    layersOrientation: WindingOrientation
    """Way in which the layers are oriented inside the section"""

    name: str
    """Name given to the winding"""

    partialWindings: List[PartialWinding]
    """List of partial windings in this section"""

    type: ElectricalType
    """Type of the layer"""

    coordinateSystem: Optional[CoordinateSystem] = None
    """System in which dimension and coordinates are in"""

    fillingFactor: Optional[float] = None
    """How much space in this section is used by wires compared to the total"""

    layersAlignment: Optional[CoilAlignment] = None
    """Way in which the layers are aligned inside the section"""

    margin: Optional[List[float]] = None
    """Defines the distance in extremes of the section that is reserved to be filled with margin
    tape. It is an array os two elements from inner or top, to outer or bottom
    """
    windingStyle: Optional[WindingStyle] = None
    """Defines if the section is wound by consecutive turns or parallels"""

    @staticmethod
    def from_dict(obj: Any) -> 'Section':
        assert isinstance(obj, dict)
        coordinates = from_list(from_float, obj.get("coordinates"))
        dimensions = from_list(from_float, obj.get("dimensions"))
        layersOrientation = WindingOrientation(obj.get("layersOrientation"))
        name = from_str(obj.get("name"))
        partialWindings = from_list(PartialWinding.from_dict, obj.get("partialWindings"))
        type = ElectricalType(obj.get("type"))
        coordinateSystem = from_union([CoordinateSystem, from_none], obj.get("coordinateSystem"))
        fillingFactor = from_union([from_float, from_none], obj.get("fillingFactor"))
        layersAlignment = from_union([CoilAlignment, from_none], obj.get("layersAlignment"))
        margin = from_union([lambda x: from_list(from_float, x), from_none], obj.get("margin"))
        windingStyle = from_union([WindingStyle, from_none], obj.get("windingStyle"))
        return Section(coordinates, dimensions, layersOrientation, name, partialWindings, type, coordinateSystem, fillingFactor, layersAlignment, margin, windingStyle)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["dimensions"] = from_list(to_float, self.dimensions)
        result["layersOrientation"] = to_enum(WindingOrientation, self.layersOrientation)
        result["name"] = from_str(self.name)
        result["partialWindings"] = from_list(lambda x: to_class(PartialWinding, x), self.partialWindings)
        result["type"] = to_enum(ElectricalType, self.type)
        if self.coordinateSystem is not None:
            result["coordinateSystem"] = from_union([lambda x: to_enum(CoordinateSystem, x), from_none], self.coordinateSystem)
        if self.fillingFactor is not None:
            result["fillingFactor"] = from_union([to_float, from_none], self.fillingFactor)
        if self.layersAlignment is not None:
            result["layersAlignment"] = from_union([lambda x: to_enum(CoilAlignment, x), from_none], self.layersAlignment)
        if self.margin is not None:
            result["margin"] = from_union([lambda x: from_list(to_float, x), from_none], self.margin)
        if self.windingStyle is not None:
            result["windingStyle"] = from_union([lambda x: to_enum(WindingStyle, x), from_none], self.windingStyle)
        return result


class TurnOrientation(Enum):
    """Way in which the turn is wound"""

    clockwise = "clockwise"
    counterClockwise = "counterClockwise"


@dataclass
class Turn:
    """Data describing one turn in a magnetic"""

    coordinates: List[float]
    """The coordinates of the center of the turn, referred to the center of the main column"""

    length: float
    """The length of the turn, referred from the center of its cross section, in m"""

    name: str
    """Name given to the turn"""

    parallel: int
    """The index of the parallel that this turn belongs to"""

    winding: str
    """The name of the winding that this turn belongs to"""

    additionalCoordinates: Optional[List[List[float]]] = None
    """List of additional coordinates of the center of the turn, referred to the center of the
    main column, in case the turn is not symmetrical, as in toroids
    """
    angle: Optional[float] = None
    """The angle that the turn does, useful for partial turns, in degrees"""

    coordinateSystem: Optional[CoordinateSystem] = None
    """System in which dimension and coordinates are in"""

    dimensions: Optional[List[float]] = None
    """Dimensions of the rectangle defining the turn"""

    layer: Optional[str] = None
    """The name of the layer that this turn belongs to"""

    orientation: Optional[TurnOrientation] = None
    """Way in which the turn is wound"""

    rotation: Optional[float] = None
    """Rotation of the rectangle defining the turn, in degrees"""

    section: Optional[str] = None
    """The name of the section that this turn belongs to"""

    @staticmethod
    def from_dict(obj: Any) -> 'Turn':
        assert isinstance(obj, dict)
        coordinates = from_list(from_float, obj.get("coordinates"))
        length = from_float(obj.get("length"))
        name = from_str(obj.get("name"))
        parallel = from_int(obj.get("parallel"))
        winding = from_str(obj.get("winding"))
        additionalCoordinates = from_union([lambda x: from_list(lambda x: from_list(from_float, x), x), from_none], obj.get("additionalCoordinates"))
        angle = from_union([from_float, from_none], obj.get("angle"))
        coordinateSystem = from_union([CoordinateSystem, from_none], obj.get("coordinateSystem"))
        dimensions = from_union([lambda x: from_list(from_float, x), from_none], obj.get("dimensions"))
        layer = from_union([from_str, from_none], obj.get("layer"))
        orientation = from_union([TurnOrientation, from_none], obj.get("orientation"))
        rotation = from_union([from_float, from_none], obj.get("rotation"))
        section = from_union([from_str, from_none], obj.get("section"))
        return Turn(coordinates, length, name, parallel, winding, additionalCoordinates, angle, coordinateSystem, dimensions, layer, orientation, rotation, section)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["length"] = to_float(self.length)
        result["name"] = from_str(self.name)
        result["parallel"] = from_int(self.parallel)
        result["winding"] = from_str(self.winding)
        if self.additionalCoordinates is not None:
            result["additionalCoordinates"] = from_union([lambda x: from_list(lambda x: from_list(to_float, x), x), from_none], self.additionalCoordinates)
        if self.angle is not None:
            result["angle"] = from_union([to_float, from_none], self.angle)
        if self.coordinateSystem is not None:
            result["coordinateSystem"] = from_union([lambda x: to_enum(CoordinateSystem, x), from_none], self.coordinateSystem)
        if self.dimensions is not None:
            result["dimensions"] = from_union([lambda x: from_list(to_float, x), from_none], self.dimensions)
        if self.layer is not None:
            result["layer"] = from_union([from_str, from_none], self.layer)
        if self.orientation is not None:
            result["orientation"] = from_union([lambda x: to_enum(TurnOrientation, x), from_none], self.orientation)
        if self.rotation is not None:
            result["rotation"] = from_union([to_float, from_none], self.rotation)
        if self.section is not None:
            result["section"] = from_union([from_str, from_none], self.section)
        return result


@dataclass
class Coil:
    """Data describing the coil
    
    The description of a magnetic coil
    """
    bobbin: Union[Bobbin, str]
    functionalDescription: List[CoilFunctionalDescription]
    """The data from the coil based on its function, in a way that can be used by analytical
    models of only Magnetism.
    """
    layersDescription: Optional[List[Layer]] = None
    """The data from the coil at the layer level, in a way that can be used by more advanced
    analytical and finite element models
    """
    sectionsDescription: Optional[List[Section]] = None
    """The data from the coil at the section level, in a way that can be used by more advanced
    analytical and finite element models
    """
    turnsDescription: Optional[List[Turn]] = None
    """The data from the coil at the turn level, in a way that can be used by the most advanced
    analytical and finite element models
    """

    @staticmethod
    def from_dict(obj: Any) -> 'Coil':
        assert isinstance(obj, dict)
        bobbin = from_union([Bobbin.from_dict, from_str], obj.get("bobbin"))
        functionalDescription = from_list(CoilFunctionalDescription.from_dict, obj.get("functionalDescription"))
        layersDescription = from_union([lambda x: from_list(Layer.from_dict, x), from_none], obj.get("layersDescription"))
        sectionsDescription = from_union([lambda x: from_list(Section.from_dict, x), from_none], obj.get("sectionsDescription"))
        turnsDescription = from_union([lambda x: from_list(Turn.from_dict, x), from_none], obj.get("turnsDescription"))
        return Coil(bobbin, functionalDescription, layersDescription, sectionsDescription, turnsDescription)

    def to_dict(self) -> dict:
        result: dict = {}
        result["bobbin"] = from_union([lambda x: to_class(Bobbin, x), from_str], self.bobbin)
        result["functionalDescription"] = from_list(lambda x: to_class(CoilFunctionalDescription, x), self.functionalDescription)
        if self.layersDescription is not None:
            result["layersDescription"] = from_union([lambda x: from_list(lambda x: to_class(Layer, x), x), from_none], self.layersDescription)
        if self.sectionsDescription is not None:
            result["sectionsDescription"] = from_union([lambda x: from_list(lambda x: to_class(Section, x), x), from_none], self.sectionsDescription)
        if self.turnsDescription is not None:
            result["turnsDescription"] = from_union([lambda x: from_list(lambda x: to_class(Turn, x), x), from_none], self.turnsDescription)
        return result


class Coating(Enum):
    """The coating of the core"""

    epoxy = "epoxy"
    parylene = "parylene"


class GapType(Enum):
    """The type of a gap"""

    additive = "additive"
    residual = "residual"
    subtractive = "subtractive"


@dataclass
class CoreGap:
    """A gap for the magnetic cores"""

    length: float
    """The length of the gap"""

    type: GapType
    """The type of a gap"""

    area: Optional[float] = None
    """Geometrical area of the gap"""

    coordinates: Optional[List[float]] = None
    """The coordinates of the center of the gap, referred to the center of the main column"""

    distanceClosestNormalSurface: Optional[float] = None
    """The distance where the closest perpendicular surface is. This usually is half the winding
    height
    """
    distanceClosestParallelSurface: Optional[float] = None
    """The distance where the closest parallel surface is. This usually is the opposite side of
    the winnding window
    """
    sectionDimensions: Optional[List[float]] = None
    """Dimension of the section normal to the magnetic flux"""

    shape: Optional[ColumnShape] = None

    @staticmethod
    def from_dict(obj: Any) -> 'CoreGap':
        assert isinstance(obj, dict)
        length = from_float(obj.get("length"))
        type = GapType(obj.get("type"))
        area = from_union([from_float, from_none], obj.get("area"))
        coordinates = from_union([lambda x: from_list(from_float, x), from_none], obj.get("coordinates"))
        distanceClosestNormalSurface = from_union([from_float, from_none], obj.get("distanceClosestNormalSurface"))
        distanceClosestParallelSurface = from_union([from_float, from_none], obj.get("distanceClosestParallelSurface"))
        sectionDimensions = from_union([lambda x: from_list(from_float, x), from_none], obj.get("sectionDimensions"))
        shape = from_union([ColumnShape, from_none], obj.get("shape"))
        return CoreGap(length, type, area, coordinates, distanceClosestNormalSurface, distanceClosestParallelSurface, sectionDimensions, shape)

    def to_dict(self) -> dict:
        result: dict = {}
        result["length"] = to_float(self.length)
        result["type"] = to_enum(GapType, self.type)
        if self.area is not None:
            result["area"] = from_union([to_float, from_none], self.area)
        if self.coordinates is not None:
            result["coordinates"] = from_union([lambda x: from_list(to_float, x), from_none], self.coordinates)
        if self.distanceClosestNormalSurface is not None:
            result["distanceClosestNormalSurface"] = from_union([to_float, from_none], self.distanceClosestNormalSurface)
        if self.distanceClosestParallelSurface is not None:
            result["distanceClosestParallelSurface"] = from_union([to_float, from_none], self.distanceClosestParallelSurface)
        if self.sectionDimensions is not None:
            result["sectionDimensions"] = from_union([lambda x: from_list(to_float, x), from_none], self.sectionDimensions)
        if self.shape is not None:
            result["shape"] = from_union([lambda x: to_enum(ColumnShape, x), from_none], self.shape)
        return result


@dataclass
class SaturationElement:
    """data for describing one point of the BH cycle"""

    magneticField: float
    """magnetic field value, in A/m"""

    magneticFluxDensity: float
    """magnetic flux density value, in T"""

    temperature: float
    """temperature for the field value, in Celsius"""

    @staticmethod
    def from_dict(obj: Any) -> 'SaturationElement':
        assert isinstance(obj, dict)
        magneticField = from_float(obj.get("magneticField"))
        magneticFluxDensity = from_float(obj.get("magneticFluxDensity"))
        temperature = from_float(obj.get("temperature"))
        return SaturationElement(magneticField, magneticFluxDensity, temperature)

    def to_dict(self) -> dict:
        result: dict = {}
        result["magneticField"] = to_float(self.magneticField)
        result["magneticFluxDensity"] = to_float(self.magneticFluxDensity)
        result["temperature"] = to_float(self.temperature)
        return result


@dataclass
class MassLossesPoint:
    """List of mass losses points
    
    data for describing the mass losses at a given point of magnetic flux density, frequency
    and temperature
    """
    magneticFluxDensity: OperatingPointExcitation
    origin: str
    """origin of the data"""

    temperature: float
    """temperature value, in Celsius"""

    value: float
    """mass losses value, in W/Kg"""

    @staticmethod
    def from_dict(obj: Any) -> 'MassLossesPoint':
        assert isinstance(obj, dict)
        magneticFluxDensity = OperatingPointExcitation.from_dict(obj.get("magneticFluxDensity"))
        origin = from_str(obj.get("origin"))
        temperature = from_float(obj.get("temperature"))
        value = from_float(obj.get("value"))
        return MassLossesPoint(magneticFluxDensity, origin, temperature, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["magneticFluxDensity"] = to_class(OperatingPointExcitation, self.magneticFluxDensity)
        result["origin"] = from_str(self.origin)
        result["temperature"] = to_float(self.temperature)
        result["value"] = to_float(self.value)
        return result


class MassCoreLossesMethodType(Enum):
    magnetec = "magnetec"


@dataclass
class MagneticsCoreLossesMethodData:
    """Magnetic method for estimating mass losses"""

    method: MassCoreLossesMethodType
    """Name of this method"""

    @staticmethod
    def from_dict(obj: Any) -> 'MagneticsCoreLossesMethodData':
        assert isinstance(obj, dict)
        method = MassCoreLossesMethodType(obj.get("method"))
        return MagneticsCoreLossesMethodData(method)

    def to_dict(self) -> dict:
        result: dict = {}
        result["method"] = to_enum(MassCoreLossesMethodType, self.method)
        return result


class MaterialEnum(Enum):
    """The composition of a magnetic material"""

    amorphous = "amorphous"
    electricalSteel = "electricalSteel"
    ferrite = "ferrite"
    nanocrystalline = "nanocrystalline"
    powder = "powder"


class MaterialComposition(Enum):
    """The composition of a magnetic material"""

    CarbonylIron = "Carbonyl Iron"
    FeMo = "FeMo"
    FeNi = "FeNi"
    FeNiMo = "FeNiMo"
    FeSi = "FeSi"
    FeSiAl = "FeSiAl"
    Iron = "Iron"
    MgZn = "MgZn"
    MnZn = "MnZn"
    NiZn = "NiZn"
    Proprietary = "Proprietary"


@dataclass
class FrequencyFactor:
    """Field with the coefficients used to calculate how much the permeability decreases with
    the frequency, as factor = a + b * f + c * pow(f, 2) + d * pow(f, 3) + e * pow(f, 4)
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the frequency, as factor = 1 / (a + b * pow(f, c) ) + d
    """
    a: float
    b: float
    c: float
    d: float
    e: Optional[float] = None

    @staticmethod
    def from_dict(obj: Any) -> 'FrequencyFactor':
        assert isinstance(obj, dict)
        a = from_float(obj.get("a"))
        b = from_float(obj.get("b"))
        c = from_float(obj.get("c"))
        d = from_float(obj.get("d"))
        e = from_union([from_float, from_none], obj.get("e"))
        return FrequencyFactor(a, b, c, d, e)

    def to_dict(self) -> dict:
        result: dict = {}
        result["a"] = to_float(self.a)
        result["b"] = to_float(self.b)
        result["c"] = to_float(self.c)
        result["d"] = to_float(self.d)
        if self.e is not None:
            result["e"] = from_union([to_float, from_none], self.e)
        return result


@dataclass
class MagneticFieldDcBiasFactor:
    """Field with the coefficients used to calculate how much the permeability decreases with
    the H DC bias, as factor = a + b * pow(H, c)
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the H DC bias, as factor = a + b * pow(H, c) + d
    """
    a: float
    b: float
    c: float
    d: Optional[float] = None

    @staticmethod
    def from_dict(obj: Any) -> 'MagneticFieldDcBiasFactor':
        assert isinstance(obj, dict)
        a = from_float(obj.get("a"))
        b = from_float(obj.get("b"))
        c = from_float(obj.get("c"))
        d = from_union([from_float, from_none], obj.get("d"))
        return MagneticFieldDcBiasFactor(a, b, c, d)

    def to_dict(self) -> dict:
        result: dict = {}
        result["a"] = to_float(self.a)
        result["b"] = to_float(self.b)
        result["c"] = to_float(self.c)
        if self.d is not None:
            result["d"] = from_union([to_float, from_none], self.d)
        return result


@dataclass
class MagneticFluxDensityFactor:
    """Field with the coefficients used to calculate how much the permeability decreases with
    the B field, as factor = = 1 / ( 1 / ( a + b * pow(B,c)) + 1 / (d * pow(B, e) ) + 1 / f )
    """
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @staticmethod
    def from_dict(obj: Any) -> 'MagneticFluxDensityFactor':
        assert isinstance(obj, dict)
        a = from_float(obj.get("a"))
        b = from_float(obj.get("b"))
        c = from_float(obj.get("c"))
        d = from_float(obj.get("d"))
        e = from_float(obj.get("e"))
        f = from_float(obj.get("f"))
        return MagneticFluxDensityFactor(a, b, c, d, e, f)

    def to_dict(self) -> dict:
        result: dict = {}
        result["a"] = to_float(self.a)
        result["b"] = to_float(self.b)
        result["c"] = to_float(self.c)
        result["d"] = to_float(self.d)
        result["e"] = to_float(self.e)
        result["f"] = to_float(self.f)
        return result


class InitialPermeabilitModifierMethod(Enum):
    fairrite = "fair-rite"
    magnetics = "magnetics"
    micrometals = "micrometals"


@dataclass
class TemperatureFactor:
    """Field with the coefficients used to calculate how much the permeability decreases with
    the temperature, as factor = a + b * T + c * pow(T, 2) + d * pow(T, 3) + e * pow(T, 4)
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the temperature, as either factor = a * (T -20) * 0.0001 or factor = (a + c * T + e *
    pow(T, 2)) / (1 + b * T + d * pow(T, 2))
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the temperature, as either factor = a
    """
    a: float
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None
    e: Optional[float] = None

    @staticmethod
    def from_dict(obj: Any) -> 'TemperatureFactor':
        assert isinstance(obj, dict)
        a = from_float(obj.get("a"))
        b = from_union([from_float, from_none], obj.get("b"))
        c = from_union([from_float, from_none], obj.get("c"))
        d = from_union([from_float, from_none], obj.get("d"))
        e = from_union([from_float, from_none], obj.get("e"))
        return TemperatureFactor(a, b, c, d, e)

    def to_dict(self) -> dict:
        result: dict = {}
        result["a"] = to_float(self.a)
        if self.b is not None:
            result["b"] = from_union([to_float, from_none], self.b)
        if self.c is not None:
            result["c"] = from_union([to_float, from_none], self.c)
        if self.d is not None:
            result["d"] = from_union([to_float, from_none], self.d)
        if self.e is not None:
            result["e"] = from_union([to_float, from_none], self.e)
        return result


@dataclass
class InitialPermeabilitModifier:
    """Object where keys are shape families for which this permeability is valid. If missing,
    the variant is valid for all shapes
    
    Coefficients given by Magnetics in order to calculate the permeability of their cores
    
    Coefficients given by Micrometals in order to calculate the permeability of their cores
    
    Coefficients given by Fair-Rite in order to calculate the permeability of their materials
    """
    frequencyFactor: Optional[FrequencyFactor] = None
    """Field with the coefficients used to calculate how much the permeability decreases with
    the frequency, as factor = a + b * f + c * pow(f, 2) + d * pow(f, 3) + e * pow(f, 4)
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the frequency, as factor = 1 / (a + b * pow(f, c) ) + d
    """
    magneticFieldDcBiasFactor: Optional[MagneticFieldDcBiasFactor] = None
    """Field with the coefficients used to calculate how much the permeability decreases with
    the H DC bias, as factor = a + b * pow(H, c)
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the H DC bias, as factor = a + b * pow(H, c) + d
    """
    method: Optional[InitialPermeabilitModifierMethod] = None
    """Name of this method"""

    temperatureFactor: Optional[TemperatureFactor] = None
    """Field with the coefficients used to calculate how much the permeability decreases with
    the temperature, as factor = a + b * T + c * pow(T, 2) + d * pow(T, 3) + e * pow(T, 4)
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the temperature, as either factor = a * (T -20) * 0.0001 or factor = (a + c * T + e *
    pow(T, 2)) / (1 + b * T + d * pow(T, 2))
    
    Field with the coefficients used to calculate how much the permeability decreases with
    the temperature, as either factor = a
    """
    magneticFluxDensityFactor: Optional[MagneticFluxDensityFactor] = None
    """Field with the coefficients used to calculate how much the permeability decreases with
    the B field, as factor = = 1 / ( 1 / ( a + b * pow(B,c)) + 1 / (d * pow(B, e) ) + 1 / f )
    """

    @staticmethod
    def from_dict(obj: Any) -> 'InitialPermeabilitModifier':
        assert isinstance(obj, dict)
        frequencyFactor = from_union([FrequencyFactor.from_dict, from_none], obj.get("frequencyFactor"))
        magneticFieldDcBiasFactor = from_union([MagneticFieldDcBiasFactor.from_dict, from_none], obj.get("magneticFieldDcBiasFactor"))
        method = from_union([InitialPermeabilitModifierMethod, from_none], obj.get("method"))
        temperatureFactor = from_union([TemperatureFactor.from_dict, from_none], obj.get("temperatureFactor"))
        magneticFluxDensityFactor = from_union([MagneticFluxDensityFactor.from_dict, from_none], obj.get("magneticFluxDensityFactor"))
        return InitialPermeabilitModifier(frequencyFactor, magneticFieldDcBiasFactor, method, temperatureFactor, magneticFluxDensityFactor)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.frequencyFactor is not None:
            result["frequencyFactor"] = from_union([lambda x: to_class(FrequencyFactor, x), from_none], self.frequencyFactor)
        if self.magneticFieldDcBiasFactor is not None:
            result["magneticFieldDcBiasFactor"] = from_union([lambda x: to_class(MagneticFieldDcBiasFactor, x), from_none], self.magneticFieldDcBiasFactor)
        if self.method is not None:
            result["method"] = from_union([lambda x: to_enum(InitialPermeabilitModifierMethod, x), from_none], self.method)
        if self.temperatureFactor is not None:
            result["temperatureFactor"] = from_union([lambda x: to_class(TemperatureFactor, x), from_none], self.temperatureFactor)
        if self.magneticFluxDensityFactor is not None:
            result["magneticFluxDensityFactor"] = from_union([lambda x: to_class(MagneticFluxDensityFactor, x), from_none], self.magneticFluxDensityFactor)
        return result


@dataclass
class PermeabilityPoint:
    """data for describing one point of permebility"""

    value: float
    """Permeability value"""

    frequency: Optional[float] = None
    """Frequency of the Magnetic field, in Hz"""

    magneticFieldDcBias: Optional[float] = None
    """DC bias in the magnetic field, in A/m"""

    magneticFluxDensityPeak: Optional[float] = None
    """magnetic flux density peak for the field value, in T"""

    modifiers: Optional[Dict[str, InitialPermeabilitModifier]] = None
    """The initial permeability of a magnetic material according to its manufacturer"""

    temperature: Optional[float] = None
    """temperature for the field value, in Celsius"""

    tolerance: Optional[float] = None
    """tolerance for the field value"""

    @staticmethod
    def from_dict(obj: Any) -> 'PermeabilityPoint':
        assert isinstance(obj, dict)
        value = from_float(obj.get("value"))
        frequency = from_union([from_float, from_none], obj.get("frequency"))
        magneticFieldDcBias = from_union([from_float, from_none], obj.get("magneticFieldDcBias"))
        magneticFluxDensityPeak = from_union([from_float, from_none], obj.get("magneticFluxDensityPeak"))
        modifiers = from_union([lambda x: from_dict(InitialPermeabilitModifier.from_dict, x), from_none], obj.get("modifiers"))
        temperature = from_union([from_float, from_none], obj.get("temperature"))
        tolerance = from_union([from_float, from_none], obj.get("tolerance"))
        return PermeabilityPoint(value, frequency, magneticFieldDcBias, magneticFluxDensityPeak, modifiers, temperature, tolerance)

    def to_dict(self) -> dict:
        result: dict = {}
        result["value"] = to_float(self.value)
        if self.frequency is not None:
            result["frequency"] = from_union([to_float, from_none], self.frequency)
        if self.magneticFieldDcBias is not None:
            result["magneticFieldDcBias"] = from_union([to_float, from_none], self.magneticFieldDcBias)
        if self.magneticFluxDensityPeak is not None:
            result["magneticFluxDensityPeak"] = from_union([to_float, from_none], self.magneticFluxDensityPeak)
        if self.modifiers is not None:
            result["modifiers"] = from_union([lambda x: from_dict(lambda x: to_class(InitialPermeabilitModifier, x), x), from_none], self.modifiers)
        if self.temperature is not None:
            result["temperature"] = from_union([to_float, from_none], self.temperature)
        if self.tolerance is not None:
            result["tolerance"] = from_union([to_float, from_none], self.tolerance)
        return result


@dataclass
class ComplexPermeabilityData:
    """The data regarding the complex permeability of a magnetic material"""

    imaginary: Union[PermeabilityPoint, List[PermeabilityPoint]]
    real: Union[PermeabilityPoint, List[PermeabilityPoint]]

    @staticmethod
    def from_dict(obj: Any) -> 'ComplexPermeabilityData':
        assert isinstance(obj, dict)
        imaginary = from_union([PermeabilityPoint.from_dict, lambda x: from_list(PermeabilityPoint.from_dict, x)], obj.get("imaginary"))
        real = from_union([PermeabilityPoint.from_dict, lambda x: from_list(PermeabilityPoint.from_dict, x)], obj.get("real"))
        return ComplexPermeabilityData(imaginary, real)

    def to_dict(self) -> dict:
        result: dict = {}
        result["imaginary"] = from_union([lambda x: to_class(PermeabilityPoint, x), lambda x: from_list(lambda x: to_class(PermeabilityPoint, x), x)], self.imaginary)
        result["real"] = from_union([lambda x: to_class(PermeabilityPoint, x), lambda x: from_list(lambda x: to_class(PermeabilityPoint, x), x)], self.real)
        return result


@dataclass
class Permeabilities:
    """The data regarding the relative permeability of a magnetic material"""

    initial: Union[PermeabilityPoint, List[PermeabilityPoint]]
    amplitude: Optional[Union[PermeabilityPoint, List[PermeabilityPoint]]] = None
    complex: Optional[ComplexPermeabilityData] = None
    """The data regarding the complex permeability of a magnetic material"""

    @staticmethod
    def from_dict(obj: Any) -> 'Permeabilities':
        assert isinstance(obj, dict)
        initial = from_union([PermeabilityPoint.from_dict, lambda x: from_list(PermeabilityPoint.from_dict, x)], obj.get("initial"))
        amplitude = from_union([PermeabilityPoint.from_dict, lambda x: from_list(PermeabilityPoint.from_dict, x), from_none], obj.get("amplitude"))
        complex = from_union([ComplexPermeabilityData.from_dict, from_none], obj.get("complex"))
        return Permeabilities(initial, amplitude, complex)

    def to_dict(self) -> dict:
        result: dict = {}
        result["initial"] = from_union([lambda x: to_class(PermeabilityPoint, x), lambda x: from_list(lambda x: to_class(PermeabilityPoint, x), x)], self.initial)
        if self.amplitude is not None:
            result["amplitude"] = from_union([lambda x: to_class(PermeabilityPoint, x), lambda x: from_list(lambda x: to_class(PermeabilityPoint, x), x), from_none], self.amplitude)
        if self.complex is not None:
            result["complex"] = from_union([lambda x: to_class(ComplexPermeabilityData, x), from_none], self.complex)
        return result


class CoreMaterialType(Enum):
    """The type of a magnetic material"""

    commercial = "commercial"
    custom = "custom"


@dataclass
class VolumetricLossesPoint:
    """data for describing the volumetric losses at a given point of magnetic flux density,
    frequency and temperature
    
    List of volumetric losses points
    """
    magneticFluxDensity: OperatingPointExcitation
    origin: str
    """origin of the data"""

    temperature: float
    """temperature value, in Celsius"""

    value: float
    """volumetric losses value, in W/m3"""

    @staticmethod
    def from_dict(obj: Any) -> 'VolumetricLossesPoint':
        assert isinstance(obj, dict)
        magneticFluxDensity = OperatingPointExcitation.from_dict(obj.get("magneticFluxDensity"))
        origin = from_str(obj.get("origin"))
        temperature = from_float(obj.get("temperature"))
        value = from_float(obj.get("value"))
        return VolumetricLossesPoint(magneticFluxDensity, origin, temperature, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["magneticFluxDensity"] = to_class(OperatingPointExcitation, self.magneticFluxDensity)
        result["origin"] = from_str(self.origin)
        result["temperature"] = to_float(self.temperature)
        result["value"] = to_float(self.value)
        return result


@dataclass
class RoshenAdditionalCoefficients:
    """List of coefficients for taking into account the excess losses and the dependencies of
    the resistivity
    """
    excessLossesCoefficient: float
    resistivityFrequencyCoefficient: float
    resistivityMagneticFluxDensityCoefficient: float
    resistivityOffset: float
    resistivityTemperatureCoefficient: float

    @staticmethod
    def from_dict(obj: Any) -> 'RoshenAdditionalCoefficients':
        assert isinstance(obj, dict)
        excessLossesCoefficient = from_float(obj.get("excessLossesCoefficient"))
        resistivityFrequencyCoefficient = from_float(obj.get("resistivityFrequencyCoefficient"))
        resistivityMagneticFluxDensityCoefficient = from_float(obj.get("resistivityMagneticFluxDensityCoefficient"))
        resistivityOffset = from_float(obj.get("resistivityOffset"))
        resistivityTemperatureCoefficient = from_float(obj.get("resistivityTemperatureCoefficient"))
        return RoshenAdditionalCoefficients(excessLossesCoefficient, resistivityFrequencyCoefficient, resistivityMagneticFluxDensityCoefficient, resistivityOffset, resistivityTemperatureCoefficient)

    def to_dict(self) -> dict:
        result: dict = {}
        result["excessLossesCoefficient"] = to_float(self.excessLossesCoefficient)
        result["resistivityFrequencyCoefficient"] = to_float(self.resistivityFrequencyCoefficient)
        result["resistivityMagneticFluxDensityCoefficient"] = to_float(self.resistivityMagneticFluxDensityCoefficient)
        result["resistivityOffset"] = to_float(self.resistivityOffset)
        result["resistivityTemperatureCoefficient"] = to_float(self.resistivityTemperatureCoefficient)
        return result


@dataclass
class LossFactorPoint:
    """Data for describing the loss factor at a given frequency and temperature"""

    value: float
    """Loss Factor value"""

    frequency: Optional[float] = None
    """Frequency of the field, in Hz"""

    temperature: Optional[float] = None
    """temperature for the value, in Celsius"""

    @staticmethod
    def from_dict(obj: Any) -> 'LossFactorPoint':
        assert isinstance(obj, dict)
        value = from_float(obj.get("value"))
        frequency = from_union([from_float, from_none], obj.get("frequency"))
        temperature = from_union([from_float, from_none], obj.get("temperature"))
        return LossFactorPoint(value, frequency, temperature)

    def to_dict(self) -> dict:
        result: dict = {}
        result["value"] = to_float(self.value)
        if self.frequency is not None:
            result["frequency"] = from_union([to_float, from_none], self.frequency)
        if self.temperature is not None:
            result["temperature"] = from_union([to_float, from_none], self.temperature)
        return result


class VolumetricCoreLossesMethodType(Enum):
    lossFactor = "lossFactor"
    magnetics = "magnetics"
    micrometals = "micrometals"
    roshen = "roshen"
    steinmetz = "steinmetz"


@dataclass
class SteinmetzCoreLossesMethodRangeDatum:
    alpha: float
    """frequency power coefficient alpha"""

    beta: float
    """magnetic flux density power coefficient beta"""

    k: float
    """Proportional coefficient k"""

    ct0: Optional[float] = None
    """Constant temperature coefficient ct0"""

    ct1: Optional[float] = None
    """Proportional negative temperature coefficient ct1"""

    ct2: Optional[float] = None
    """Square temperature coefficient ct2"""

    maximumFrequency: Optional[float] = None
    """maximum frequency for which the coefficients are valid, in Hz"""

    minimumFrequency: Optional[float] = None
    """minimum frequency for which the coefficients are valid, in Hz"""

    @staticmethod
    def from_dict(obj: Any) -> 'SteinmetzCoreLossesMethodRangeDatum':
        assert isinstance(obj, dict)
        alpha = from_float(obj.get("alpha"))
        beta = from_float(obj.get("beta"))
        k = from_float(obj.get("k"))
        ct0 = from_union([from_float, from_none], obj.get("ct0"))
        ct1 = from_union([from_float, from_none], obj.get("ct1"))
        ct2 = from_union([from_float, from_none], obj.get("ct2"))
        maximumFrequency = from_union([from_float, from_none], obj.get("maximumFrequency"))
        minimumFrequency = from_union([from_float, from_none], obj.get("minimumFrequency"))
        return SteinmetzCoreLossesMethodRangeDatum(alpha, beta, k, ct0, ct1, ct2, maximumFrequency, minimumFrequency)

    def to_dict(self) -> dict:
        result: dict = {}
        result["alpha"] = to_float(self.alpha)
        result["beta"] = to_float(self.beta)
        result["k"] = to_float(self.k)
        if self.ct0 is not None:
            result["ct0"] = from_union([to_float, from_none], self.ct0)
        if self.ct1 is not None:
            result["ct1"] = from_union([to_float, from_none], self.ct1)
        if self.ct2 is not None:
            result["ct2"] = from_union([to_float, from_none], self.ct2)
        if self.maximumFrequency is not None:
            result["maximumFrequency"] = from_union([to_float, from_none], self.maximumFrequency)
        if self.minimumFrequency is not None:
            result["minimumFrequency"] = from_union([to_float, from_none], self.minimumFrequency)
        return result


@dataclass
class CoreLossesMethodData:
    """Steinmetz coefficients for estimating volumetric losses in a given frequency range
    
    Roshen coefficients for estimating volumetric losses
    
    Micrometals method for estimating volumetric losses
    
    Magnetics method for estimating volumetric losses
    
    Loss factor method for estimating volumetric losses
    """
    method: VolumetricCoreLossesMethodType
    """Name of this method"""

    ranges: Optional[List[SteinmetzCoreLossesMethodRangeDatum]] = None
    coefficients: Optional[RoshenAdditionalCoefficients] = None
    """List of coefficients for taking into account the excess losses and the dependencies of
    the resistivity
    """
    referenceVolumetricLosses: Optional[List[VolumetricLossesPoint]] = None
    """List of reference volumetric losses used to estimate excess eddy current losses"""

    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None
    factors: Optional[List[LossFactorPoint]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'CoreLossesMethodData':
        assert isinstance(obj, dict)
        method = VolumetricCoreLossesMethodType(obj.get("method"))
        ranges = from_union([lambda x: from_list(SteinmetzCoreLossesMethodRangeDatum.from_dict, x), from_none], obj.get("ranges"))
        coefficients = from_union([RoshenAdditionalCoefficients.from_dict, from_none], obj.get("coefficients"))
        referenceVolumetricLosses = from_union([lambda x: from_list(VolumetricLossesPoint.from_dict, x), from_none], obj.get("referenceVolumetricLosses"))
        a = from_union([from_float, from_none], obj.get("a"))
        b = from_union([from_float, from_none], obj.get("b"))
        c = from_union([from_float, from_none], obj.get("c"))
        d = from_union([from_float, from_none], obj.get("d"))
        factors = from_union([lambda x: from_list(LossFactorPoint.from_dict, x), from_none], obj.get("factors"))
        return CoreLossesMethodData(method, ranges, coefficients, referenceVolumetricLosses, a, b, c, d, factors)

    def to_dict(self) -> dict:
        result: dict = {}
        result["method"] = to_enum(VolumetricCoreLossesMethodType, self.method)
        if self.ranges is not None:
            result["ranges"] = from_union([lambda x: from_list(lambda x: to_class(SteinmetzCoreLossesMethodRangeDatum, x), x), from_none], self.ranges)
        if self.coefficients is not None:
            result["coefficients"] = from_union([lambda x: to_class(RoshenAdditionalCoefficients, x), from_none], self.coefficients)
        if self.referenceVolumetricLosses is not None:
            result["referenceVolumetricLosses"] = from_union([lambda x: from_list(lambda x: to_class(VolumetricLossesPoint, x), x), from_none], self.referenceVolumetricLosses)
        if self.a is not None:
            result["a"] = from_union([to_float, from_none], self.a)
        if self.b is not None:
            result["b"] = from_union([to_float, from_none], self.b)
        if self.c is not None:
            result["c"] = from_union([to_float, from_none], self.c)
        if self.d is not None:
            result["d"] = from_union([to_float, from_none], self.d)
        if self.factors is not None:
            result["factors"] = from_union([lambda x: from_list(lambda x: to_class(LossFactorPoint, x), x), from_none], self.factors)
        return result


@dataclass
class CoreMaterial:
    """A material for the magnetic cores"""

    manufacturerInfo: ManufacturerInfo
    material: MaterialEnum
    """The composition of a magnetic material"""

    name: str
    """The name of a magnetic material"""

    permeability: Permeabilities
    """The data regarding the relative permeability of a magnetic material"""

    resistivity: List[ResistivityPoint]
    """Resistivity value according to manufacturer"""

    saturation: List[SaturationElement]
    """BH Cycle points where a non-negligible increase in magnetic field produces a negligible
    increase of magnetic flux density
    """
    type: CoreMaterialType
    """The type of a magnetic material"""

    volumetricLosses: Dict[str, List[Union[CoreLossesMethodData, List[VolumetricLossesPoint]]]]
    """The data regarding the volumetric losses of a magnetic material"""

    bhCycle: Optional[List[SaturationElement]] = None
    coerciveForce: Optional[List[SaturationElement]] = None
    """BH Cycle points where the magnetic flux density is 0"""

    commercialName: Optional[str] = None
    """The name of a magnetic material together its manufacturer"""

    curieTemperature: Optional[float] = None
    """The temperature at which this material losses all ferromagnetism"""

    density: Optional[float] = None
    """Density value according to manufacturer, in kg/m3"""

    family: Optional[str] = None
    """The family of a magnetic material according to its manufacturer"""

    heatCapacity: Optional[DimensionWithTolerance] = None
    """Heat capacity value according to manufacturer, in J/Kg/K"""

    heatConductivity: Optional[DimensionWithTolerance] = None
    """Heat conductivity value according to manufacturer, in W/m/K"""

    massLosses: Optional[Dict[str, List[Union[MagneticsCoreLossesMethodData, List[MassLossesPoint]]]]] = None
    """The data regarding the mass losses of a magnetic material"""

    materialComposition: Optional[MaterialComposition] = None
    """The composition of a magnetic material"""

    remanence: Optional[List[SaturationElement]] = None
    """BH Cycle points where the magnetic field is 0"""

    @staticmethod
    def from_dict(obj: Any) -> 'CoreMaterial':
        assert isinstance(obj, dict)
        manufacturerInfo = ManufacturerInfo.from_dict(obj.get("manufacturerInfo"))
        material = MaterialEnum(obj.get("material"))
        name = from_str(obj.get("name"))
        permeability = Permeabilities.from_dict(obj.get("permeability"))
        resistivity = from_list(ResistivityPoint.from_dict, obj.get("resistivity"))
        saturation = from_list(SaturationElement.from_dict, obj.get("saturation"))
        type = CoreMaterialType(obj.get("type"))
        volumetricLosses = from_dict(lambda x: from_list(lambda x: from_union([CoreLossesMethodData.from_dict, lambda x: from_list(VolumetricLossesPoint.from_dict, x)], x), x), obj.get("volumetricLosses"))
        bhCycle = from_union([lambda x: from_list(SaturationElement.from_dict, x), from_none], obj.get("bhCycle"))
        coerciveForce = from_union([lambda x: from_list(SaturationElement.from_dict, x), from_none], obj.get("coerciveForce"))
        commercialName = from_union([from_str, from_none], obj.get("commercialName"))
        curieTemperature = from_union([from_float, from_none], obj.get("curieTemperature"))
        density = from_union([from_float, from_none], obj.get("density"))
        family = from_union([from_str, from_none], obj.get("family"))
        heatCapacity = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("heatCapacity"))
        heatConductivity = from_union([DimensionWithTolerance.from_dict, from_none], obj.get("heatConductivity"))
        massLosses = from_union([lambda x: from_dict(lambda x: from_list(lambda x: from_union([MagneticsCoreLossesMethodData.from_dict, lambda x: from_list(MassLossesPoint.from_dict, x)], x), x), x), from_none], obj.get("massLosses"))
        materialComposition = from_union([MaterialComposition, from_none], obj.get("materialComposition"))
        remanence = from_union([lambda x: from_list(SaturationElement.from_dict, x), from_none], obj.get("remanence"))
        return CoreMaterial(manufacturerInfo, material, name, permeability, resistivity, saturation, type, volumetricLosses, bhCycle, coerciveForce, commercialName, curieTemperature, density, family, heatCapacity, heatConductivity, massLosses, materialComposition, remanence)

    def to_dict(self) -> dict:
        result: dict = {}
        result["manufacturerInfo"] = to_class(ManufacturerInfo, self.manufacturerInfo)
        result["material"] = to_enum(MaterialEnum, self.material)
        result["name"] = from_str(self.name)
        result["permeability"] = to_class(Permeabilities, self.permeability)
        result["resistivity"] = from_list(lambda x: to_class(ResistivityPoint, x), self.resistivity)
        result["saturation"] = from_list(lambda x: to_class(SaturationElement, x), self.saturation)
        result["type"] = to_enum(CoreMaterialType, self.type)
        result["volumetricLosses"] = from_dict(lambda x: from_list(lambda x: from_union([lambda x: to_class(CoreLossesMethodData, x), lambda x: from_list(lambda x: to_class(VolumetricLossesPoint, x), x)], x), x), self.volumetricLosses)
        if self.bhCycle is not None:
            result["bhCycle"] = from_union([lambda x: from_list(lambda x: to_class(SaturationElement, x), x), from_none], self.bhCycle)
        if self.coerciveForce is not None:
            result["coerciveForce"] = from_union([lambda x: from_list(lambda x: to_class(SaturationElement, x), x), from_none], self.coerciveForce)
        if self.commercialName is not None:
            result["commercialName"] = from_union([from_str, from_none], self.commercialName)
        if self.curieTemperature is not None:
            result["curieTemperature"] = from_union([to_float, from_none], self.curieTemperature)
        if self.density is not None:
            result["density"] = from_union([to_float, from_none], self.density)
        if self.family is not None:
            result["family"] = from_union([from_str, from_none], self.family)
        if self.heatCapacity is not None:
            result["heatCapacity"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.heatCapacity)
        if self.heatConductivity is not None:
            result["heatConductivity"] = from_union([lambda x: to_class(DimensionWithTolerance, x), from_none], self.heatConductivity)
        if self.massLosses is not None:
            result["massLosses"] = from_union([lambda x: from_dict(lambda x: from_list(lambda x: from_union([lambda x: to_class(MagneticsCoreLossesMethodData, x), lambda x: from_list(lambda x: to_class(MassLossesPoint, x), x)], x), x), x), from_none], self.massLosses)
        if self.materialComposition is not None:
            result["materialComposition"] = from_union([lambda x: to_enum(MaterialComposition, x), from_none], self.materialComposition)
        if self.remanence is not None:
            result["remanence"] = from_union([lambda x: from_list(lambda x: to_class(SaturationElement, x), x), from_none], self.remanence)
        return result


class CoreShapeFamily(Enum):
    """The family of a magnetic shape"""

    c = "c"
    drum = "drum"
    e = "e"
    ec = "ec"
    efd = "efd"
    ei = "ei"
    el = "el"
    elp = "elp"
    ep = "ep"
    epx = "epx"
    eq = "eq"
    er = "er"
    etd = "etd"
    h = "h"
    lp = "lp"
    p = "p"
    planare = "planar e"
    planarel = "planar el"
    planarer = "planar er"
    pm = "pm"
    pq = "pq"
    pqi = "pqi"
    rm = "rm"
    rod = "rod"
    t = "t"
    u = "u"
    ui = "ui"
    ur = "ur"
    ut = "ut"


class MagneticCircuit(Enum):
    """Describes if the magnetic circuit of the shape is open, and can be combined with others;
    or closed, and has to be used by itself
    """
    closed = "closed"
    open = "open"


@dataclass
class CoreShape:
    """A shape for the magnetic cores"""

    family: CoreShapeFamily
    """The family of a magnetic shape"""

    type: FunctionalDescriptionType
    """The type of a magnetic shape"""

    aliases: Optional[List[str]] = None
    """Alternative names of a magnetic shape"""

    dimensions: Optional[Dict[str, Union[DimensionWithTolerance, float]]] = None
    """The dimensions of a magnetic shape, keys must be as defined in EN 62317"""

    familySubtype: Optional[str] = None
    """The subtype of the shape, in case there are more than one"""

    magneticCircuit: Optional[MagneticCircuit] = None
    """Describes if the magnetic circuit of the shape is open, and can be combined with others;
    or closed, and has to be used by itself
    """
    name: Optional[str] = None
    """The name of a magnetic shape"""

    @staticmethod
    def from_dict(obj: Any) -> 'CoreShape':
        assert isinstance(obj, dict)
        family = CoreShapeFamily(obj.get("family"))
        type = FunctionalDescriptionType(obj.get("type"))
        aliases = from_union([lambda x: from_list(from_str, x), from_none], obj.get("aliases"))
        dimensions = from_union([lambda x: from_dict(lambda x: from_union([DimensionWithTolerance.from_dict, from_float], x), x), from_none], obj.get("dimensions"))
        familySubtype = from_union([from_str, from_none], obj.get("familySubtype"))
        magneticCircuit = from_union([MagneticCircuit, from_none], obj.get("magneticCircuit"))
        name = from_union([from_str, from_none], obj.get("name"))
        return CoreShape(family, type, aliases, dimensions, familySubtype, magneticCircuit, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["family"] = to_enum(CoreShapeFamily, self.family)
        result["type"] = to_enum(FunctionalDescriptionType, self.type)
        if self.aliases is not None:
            result["aliases"] = from_union([lambda x: from_list(from_str, x), from_none], self.aliases)
        if self.dimensions is not None:
            result["dimensions"] = from_union([lambda x: from_dict(lambda x: from_union([lambda x: to_class(DimensionWithTolerance, x), to_float], x), x), from_none], self.dimensions)
        if self.familySubtype is not None:
            result["familySubtype"] = from_union([from_str, from_none], self.familySubtype)
        if self.magneticCircuit is not None:
            result["magneticCircuit"] = from_union([lambda x: to_enum(MagneticCircuit, x), from_none], self.magneticCircuit)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        return result


class CoreType(Enum):
    """The type of core"""

    closedshape = "closed shape"
    pieceandplate = "piece and plate"
    toroidal = "toroidal"
    twopieceset = "two-piece set"


@dataclass
class CoreFunctionalDescription:
    """The data from the core based on its function, in a way that can be used by analytical
    models.
    """
    gapping: List[CoreGap]
    """The lists of gaps in the magnetic core"""

    material: Union[CoreMaterial, str]
    shape: Union[CoreShape, str]
    type: CoreType
    """The type of core"""

    coating: Optional[Coating] = None
    """The coating of the core"""

    numberStacks: Optional[int] = None
    """The number of stacked cores"""

    @staticmethod
    def from_dict(obj: Any) -> 'CoreFunctionalDescription':
        assert isinstance(obj, dict)
        gapping = from_list(CoreGap.from_dict, obj.get("gapping"))
        material = from_union([CoreMaterial.from_dict, from_str], obj.get("material"))
        shape = from_union([CoreShape.from_dict, from_str], obj.get("shape"))
        type = CoreType(obj.get("type"))
        coating = from_union([Coating, from_none], obj.get("coating"))
        numberStacks = from_union([from_int, from_none], obj.get("numberStacks"))
        return CoreFunctionalDescription(gapping, material, shape, type, coating, numberStacks)

    def to_dict(self) -> dict:
        result: dict = {}
        result["gapping"] = from_list(lambda x: to_class(CoreGap, x), self.gapping)
        result["material"] = from_union([lambda x: to_class(CoreMaterial, x), from_str], self.material)
        result["shape"] = from_union([lambda x: to_class(CoreShape, x), from_str], self.shape)
        result["type"] = to_enum(CoreType, self.type)
        if self.coating is not None:
            result["coating"] = from_union([lambda x: to_enum(Coating, x), from_none], self.coating)
        if self.numberStacks is not None:
            result["numberStacks"] = from_union([from_int, from_none], self.numberStacks)
        return result


@dataclass
class Machining:
    """Data describing the machining applied to a piece"""

    coordinates: List[float]
    """The coordinates of the start of the machining, referred to the top of the main column of
    the piece
    """
    length: float
    """Length of the machining"""

    @staticmethod
    def from_dict(obj: Any) -> 'Machining':
        assert isinstance(obj, dict)
        coordinates = from_list(from_float, obj.get("coordinates"))
        length = from_float(obj.get("length"))
        return Machining(coordinates, length)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["length"] = to_float(self.length)
        return result


class CoreGeometricalDescriptionElementType(Enum):
    """The type of piece
    
    The type of spacer
    """
    closed = "closed"
    halfset = "half set"
    plate = "plate"
    sheet = "sheet"
    spacer = "spacer"
    toroidal = "toroidal"


@dataclass
class CoreGeometricalDescriptionElement:
    """The data from the core based on its geometrical description, in a way that can be used by
    CAD models.
    
    Data describing the a piece of a core
    
    Data describing the spacer used to separate cores in additive gaps
    """
    coordinates: List[float]
    """The coordinates of the top of the piece, referred to the center of the main column
    
    The coordinates of the center of the gap, referred to the center of the main column
    """
    type: CoreGeometricalDescriptionElementType
    """The type of piece
    
    The type of spacer
    """
    machining: Optional[List[Machining]] = None
    material: Optional[Union[CoreMaterial, str]] = None
    rotation: Optional[List[float]] = None
    """The rotation of the top of the piece from its original state, referred to the center of
    the main column
    """
    shape: Optional[Union[CoreShape, str]] = None
    dimensions: Optional[List[float]] = None
    """Dimensions of the cube defining the spacer"""

    insulationMaterial: Optional[Union[InsulationMaterial, str]] = None
    """Material of the spacer"""

    @staticmethod
    def from_dict(obj: Any) -> 'CoreGeometricalDescriptionElement':
        assert isinstance(obj, dict)
        coordinates = from_list(from_float, obj.get("coordinates"))
        type = CoreGeometricalDescriptionElementType(obj.get("type"))
        machining = from_union([lambda x: from_list(Machining.from_dict, x), from_none], obj.get("machining"))
        material = from_union([CoreMaterial.from_dict, from_str, from_none], obj.get("material"))
        rotation = from_union([lambda x: from_list(from_float, x), from_none], obj.get("rotation"))
        shape = from_union([CoreShape.from_dict, from_str, from_none], obj.get("shape"))
        dimensions = from_union([lambda x: from_list(from_float, x), from_none], obj.get("dimensions"))
        insulationMaterial = from_union([InsulationMaterial.from_dict, from_str, from_none], obj.get("insulationMaterial"))
        return CoreGeometricalDescriptionElement(coordinates, type, machining, material, rotation, shape, dimensions, insulationMaterial)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["type"] = to_enum(CoreGeometricalDescriptionElementType, self.type)
        if self.machining is not None:
            result["machining"] = from_union([lambda x: from_list(lambda x: to_class(Machining, x), x), from_none], self.machining)
        if self.material is not None:
            result["material"] = from_union([lambda x: to_class(CoreMaterial, x), from_str, from_none], self.material)
        if self.rotation is not None:
            result["rotation"] = from_union([lambda x: from_list(to_float, x), from_none], self.rotation)
        if self.shape is not None:
            result["shape"] = from_union([lambda x: to_class(CoreShape, x), from_str, from_none], self.shape)
        if self.dimensions is not None:
            result["dimensions"] = from_union([lambda x: from_list(to_float, x), from_none], self.dimensions)
        if self.insulationMaterial is not None:
            result["insulationMaterial"] = from_union([lambda x: to_class(InsulationMaterial, x), from_str, from_none], self.insulationMaterial)
        return result


class ColumnType(Enum):
    """Name of the column"""

    central = "central"
    lateral = "lateral"


@dataclass
class ColumnElement:
    """Data describing a column of the core"""

    area: float
    """Area of the section column, normal to the magnetic flux direction"""

    coordinates: List[float]
    """The coordinates of the center of the column, referred to the center of the main column.
    In the case of half-sets, the center will be in the top point, where it would join
    another half-set
    """
    depth: float
    """Depth of the column"""

    height: float
    """Height of the column"""

    shape: ColumnShape
    type: ColumnType
    """Name of the column"""

    width: float
    """Width of the column"""

    minimumDepth: Optional[float] = None
    """Minimum depth of the column, if irregular"""

    minimumWidth: Optional[float] = None
    """Minimum width of the column, if irregular"""

    @staticmethod
    def from_dict(obj: Any) -> 'ColumnElement':
        assert isinstance(obj, dict)
        area = from_float(obj.get("area"))
        coordinates = from_list(from_float, obj.get("coordinates"))
        depth = from_float(obj.get("depth"))
        height = from_float(obj.get("height"))
        shape = ColumnShape(obj.get("shape"))
        type = ColumnType(obj.get("type"))
        width = from_float(obj.get("width"))
        minimumDepth = from_union([from_float, from_none], obj.get("minimumDepth"))
        minimumWidth = from_union([from_float, from_none], obj.get("minimumWidth"))
        return ColumnElement(area, coordinates, depth, height, shape, type, width, minimumDepth, minimumWidth)

    def to_dict(self) -> dict:
        result: dict = {}
        result["area"] = to_float(self.area)
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["depth"] = to_float(self.depth)
        result["height"] = to_float(self.height)
        result["shape"] = to_enum(ColumnShape, self.shape)
        result["type"] = to_enum(ColumnType, self.type)
        result["width"] = to_float(self.width)
        if self.minimumDepth is not None:
            result["minimumDepth"] = from_union([to_float, from_none], self.minimumDepth)
        if self.minimumWidth is not None:
            result["minimumWidth"] = from_union([to_float, from_none], self.minimumWidth)
        return result


@dataclass
class EffectiveParameters:
    """Effective data of the magnetic core"""

    effectiveArea: float
    """This is the equivalent section that the magnetic flux traverses, because the shape of the
    core is not uniform and its section changes along the path
    """
    effectiveLength: float
    """This is the equivalent length that the magnetic flux travels through the core."""

    effectiveVolume: float
    """This is the product of the effective length by the effective area, and represents the
    equivalent volume that is magnetized by the field
    """
    minimumArea: float
    """This is the minimum area seen by the magnetic flux along its path"""

    @staticmethod
    def from_dict(obj: Any) -> 'EffectiveParameters':
        assert isinstance(obj, dict)
        effectiveArea = from_float(obj.get("effectiveArea"))
        effectiveLength = from_float(obj.get("effectiveLength"))
        effectiveVolume = from_float(obj.get("effectiveVolume"))
        minimumArea = from_float(obj.get("minimumArea"))
        return EffectiveParameters(effectiveArea, effectiveLength, effectiveVolume, minimumArea)

    def to_dict(self) -> dict:
        result: dict = {}
        result["effectiveArea"] = to_float(self.effectiveArea)
        result["effectiveLength"] = to_float(self.effectiveLength)
        result["effectiveVolume"] = to_float(self.effectiveVolume)
        result["minimumArea"] = to_float(self.minimumArea)
        return result


@dataclass
class CoreProcessedDescription:
    """The data from the core after been processed, and ready to use by the analytical models"""

    columns: List[ColumnElement]
    """List of columns in the core"""

    depth: float
    """Total depth of the core"""

    effectiveParameters: EffectiveParameters
    height: float
    """Total height of the core"""

    width: float
    """Total width of the core"""

    windingWindows: List[WindingWindowElement]
    """List of winding windows, all elements in the list must be of the same type"""

    thermalResistance: Optional[float] = None
    """Parameter describing steady state temperature rise versus dissipated power within a given
    device.
    """

    @staticmethod
    def from_dict(obj: Any) -> 'CoreProcessedDescription':
        assert isinstance(obj, dict)
        columns = from_list(ColumnElement.from_dict, obj.get("columns"))
        depth = from_float(obj.get("depth"))
        effectiveParameters = EffectiveParameters.from_dict(obj.get("effectiveParameters"))
        height = from_float(obj.get("height"))
        width = from_float(obj.get("width"))
        windingWindows = from_list(WindingWindowElement.from_dict, obj.get("windingWindows"))
        thermalResistance = from_union([from_float, from_none], obj.get("thermalResistance"))
        return CoreProcessedDescription(columns, depth, effectiveParameters, height, width, windingWindows, thermalResistance)

    def to_dict(self) -> dict:
        result: dict = {}
        result["columns"] = from_list(lambda x: to_class(ColumnElement, x), self.columns)
        result["depth"] = to_float(self.depth)
        result["effectiveParameters"] = to_class(EffectiveParameters, self.effectiveParameters)
        result["height"] = to_float(self.height)
        result["width"] = to_float(self.width)
        result["windingWindows"] = from_list(lambda x: to_class(WindingWindowElement, x), self.windingWindows)
        if self.thermalResistance is not None:
            result["thermalResistance"] = from_union([to_float, from_none], self.thermalResistance)
        return result


@dataclass
class MagneticCore:
    """Data describing the magnetic core.
    
    The description of a magnetic core
    """
    functionalDescription: CoreFunctionalDescription
    """The data from the core based on its function, in a way that can be used by analytical
    models.
    """
    distributorsInfo: Optional[List[DistributorInfo]] = None
    """The lists of distributors of the magnetic core"""

    geometricalDescription: Optional[List[CoreGeometricalDescriptionElement]] = None
    """List with data from the core based on its geometrical description, in a way that can be
    used by CAD models.
    """
    manufacturerInfo: Optional[ManufacturerInfo] = None
    name: Optional[str] = None
    """The name of core"""

    processedDescription: Optional[CoreProcessedDescription] = None
    """The data from the core after been processed, and ready to use by the analytical models"""

    @staticmethod
    def from_dict(obj: Any) -> 'MagneticCore':
        assert isinstance(obj, dict)
        functionalDescription = CoreFunctionalDescription.from_dict(obj.get("functionalDescription"))
        distributorsInfo = from_union([lambda x: from_list(DistributorInfo.from_dict, x), from_none], obj.get("distributorsInfo"))
        geometricalDescription = from_union([lambda x: from_list(CoreGeometricalDescriptionElement.from_dict, x), from_none], obj.get("geometricalDescription"))
        manufacturerInfo = from_union([ManufacturerInfo.from_dict, from_none], obj.get("manufacturerInfo"))
        name = from_union([from_str, from_none], obj.get("name"))
        processedDescription = from_union([CoreProcessedDescription.from_dict, from_none], obj.get("processedDescription"))
        return MagneticCore(functionalDescription, distributorsInfo, geometricalDescription, manufacturerInfo, name, processedDescription)

    def to_dict(self) -> dict:
        result: dict = {}
        result["functionalDescription"] = to_class(CoreFunctionalDescription, self.functionalDescription)
        if self.distributorsInfo is not None:
            result["distributorsInfo"] = from_union([lambda x: from_list(lambda x: to_class(DistributorInfo, x), x), from_none], self.distributorsInfo)
        if self.geometricalDescription is not None:
            result["geometricalDescription"] = from_union([lambda x: from_list(lambda x: to_class(CoreGeometricalDescriptionElement, x), x), from_none], self.geometricalDescription)
        if self.manufacturerInfo is not None:
            result["manufacturerInfo"] = from_union([lambda x: to_class(ManufacturerInfo, x), from_none], self.manufacturerInfo)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.processedDescription is not None:
            result["processedDescription"] = from_union([lambda x: to_class(CoreProcessedDescription, x), from_none], self.processedDescription)
        return result


@dataclass
class MagneticManufacturerRecommendations:
    ratedCurrent: Optional[float] = None
    """The manufacturer's rated current for this part"""

    ratedCurrentTemperatureRise: Optional[float] = None
    """The temperature rise for which the rated current is calculated"""

    ratedMagneticFlux: Optional[float] = None
    """The manufacturer's rated magnetic flux or volt-seconds for this part"""

    saturationCurrent: Optional[float] = None
    """The manufacturer's saturation current for this part"""

    saturationCurrentInductanceDrop: Optional[float] = None
    """Percentage of inductance drop at saturation current"""

    @staticmethod
    def from_dict(obj: Any) -> 'MagneticManufacturerRecommendations':
        assert isinstance(obj, dict)
        ratedCurrent = from_union([from_float, from_none], obj.get("ratedCurrent"))
        ratedCurrentTemperatureRise = from_union([from_float, from_none], obj.get("ratedCurrentTemperatureRise"))
        ratedMagneticFlux = from_union([from_float, from_none], obj.get("ratedMagneticFlux"))
        saturationCurrent = from_union([from_float, from_none], obj.get("saturationCurrent"))
        saturationCurrentInductanceDrop = from_union([from_float, from_none], obj.get("saturationCurrentInductanceDrop"))
        return MagneticManufacturerRecommendations(ratedCurrent, ratedCurrentTemperatureRise, ratedMagneticFlux, saturationCurrent, saturationCurrentInductanceDrop)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.ratedCurrent is not None:
            result["ratedCurrent"] = from_union([to_float, from_none], self.ratedCurrent)
        if self.ratedCurrentTemperatureRise is not None:
            result["ratedCurrentTemperatureRise"] = from_union([to_float, from_none], self.ratedCurrentTemperatureRise)
        if self.ratedMagneticFlux is not None:
            result["ratedMagneticFlux"] = from_union([to_float, from_none], self.ratedMagneticFlux)
        if self.saturationCurrent is not None:
            result["saturationCurrent"] = from_union([to_float, from_none], self.saturationCurrent)
        if self.saturationCurrentInductanceDrop is not None:
            result["saturationCurrentInductanceDrop"] = from_union([to_float, from_none], self.saturationCurrentInductanceDrop)
        return result


@dataclass
class MagneticManufacturerInfo:
    name: str
    """The name of the manufacturer of the part"""

    cost: Optional[str] = None
    """The manufacturer's price for this part"""

    datasheetUrl: Optional[str] = None
    """The manufacturer's URL to the datasheet of the product"""

    family: Optional[str] = None
    """The family of a magnetic, as defined by the manufacturer"""

    recommendations: Optional[MagneticManufacturerRecommendations] = None
    reference: Optional[str] = None
    """The manufacturer's reference of this part"""

    status: Optional[Status] = None
    """The production status of a part according to its manufacturer"""

    @staticmethod
    def from_dict(obj: Any) -> 'MagneticManufacturerInfo':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        cost = from_union([from_str, from_none], obj.get("cost"))
        datasheetUrl = from_union([from_str, from_none], obj.get("datasheetUrl"))
        family = from_union([from_str, from_none], obj.get("family"))
        recommendations = from_union([MagneticManufacturerRecommendations.from_dict, from_none], obj.get("recommendations"))
        reference = from_union([from_str, from_none], obj.get("reference"))
        status = from_union([Status, from_none], obj.get("status"))
        return MagneticManufacturerInfo(name, cost, datasheetUrl, family, recommendations, reference, status)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        if self.cost is not None:
            result["cost"] = from_union([from_str, from_none], self.cost)
        if self.datasheetUrl is not None:
            result["datasheetUrl"] = from_union([from_str, from_none], self.datasheetUrl)
        if self.family is not None:
            result["family"] = from_union([from_str, from_none], self.family)
        if self.recommendations is not None:
            result["recommendations"] = from_union([lambda x: to_class(MagneticManufacturerRecommendations, x), from_none], self.recommendations)
        if self.reference is not None:
            result["reference"] = from_union([from_str, from_none], self.reference)
        if self.status is not None:
            result["status"] = from_union([lambda x: to_enum(Status, x), from_none], self.status)
        return result


@dataclass
class Magnetic:
    """The description of a magnetic"""

    coil: Coil
    """Data describing the coil"""

    core: MagneticCore
    """Data describing the magnetic core."""

    distributorsInfo: Optional[List[DistributorInfo]] = None
    """The lists of distributors of the magnetic"""

    manufacturerInfo: Optional[MagneticManufacturerInfo] = None
    rotation: Optional[List[float]] = None
    """The rotation of the magnetic, by default the winding column goes vertical"""

    @staticmethod
    def from_dict(obj: Any) -> 'Magnetic':
        assert isinstance(obj, dict)
        coil = Coil.from_dict(obj.get("coil"))
        core = MagneticCore.from_dict(obj.get("core"))
        distributorsInfo = from_union([lambda x: from_list(DistributorInfo.from_dict, x), from_none], obj.get("distributorsInfo"))
        manufacturerInfo = from_union([MagneticManufacturerInfo.from_dict, from_none], obj.get("manufacturerInfo"))
        rotation = from_union([lambda x: from_list(from_float, x), from_none], obj.get("rotation"))
        return Magnetic(coil, core, distributorsInfo, manufacturerInfo, rotation)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coil"] = to_class(Coil, self.coil)
        result["core"] = to_class(MagneticCore, self.core)
        if self.distributorsInfo is not None:
            result["distributorsInfo"] = from_union([lambda x: from_list(lambda x: to_class(DistributorInfo, x), x), from_none], self.distributorsInfo)
        if self.manufacturerInfo is not None:
            result["manufacturerInfo"] = from_union([lambda x: to_class(MagneticManufacturerInfo, x), from_none], self.manufacturerInfo)
        if self.rotation is not None:
            result["rotation"] = from_union([lambda x: from_list(to_float, x), from_none], self.rotation)
        return result


class ResultOrigin(Enum):
    """Origin of the value of the result"""

    manufacturer = "manufacturer"
    measurement = "measurement"
    simulation = "simulation"


@dataclass
class CoreLossesOutput:
    """Data describing the output core losses
    
    Data describing the core losses and the intermediate inputs used to calculate them
    """
    coreLosses: float
    """Value of the core losses"""

    methodUsed: str
    """Model used to calculate the core losses in the case of simulation, or method used to
    measure it
    """
    origin: ResultOrigin
    eddyCurrentCoreLosses: Optional[float] = None
    """Part of the core losses due to eddy currents"""

    hysteresisCoreLosses: Optional[float] = None
    """Part of the core losses due to hysteresis"""

    magneticFluxDensity: Optional[SignalDescriptor] = None
    """Excitation of the B field that produced the core losses"""

    massLosses: Optional[float] = None
    """Mass value of the core losses"""

    temperature: Optional[float] = None
    """temperature in the core that produced the core losses"""

    volumetricLosses: Optional[float] = None
    """Volumetric value of the core losses"""

    @staticmethod
    def from_dict(obj: Any) -> 'CoreLossesOutput':
        assert isinstance(obj, dict)
        coreLosses = from_float(obj.get("coreLosses"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        eddyCurrentCoreLosses = from_union([from_float, from_none], obj.get("eddyCurrentCoreLosses"))
        hysteresisCoreLosses = from_union([from_float, from_none], obj.get("hysteresisCoreLosses"))
        magneticFluxDensity = from_union([SignalDescriptor.from_dict, from_none], obj.get("magneticFluxDensity"))
        massLosses = from_union([from_float, from_none], obj.get("massLosses"))
        temperature = from_union([from_float, from_none], obj.get("temperature"))
        volumetricLosses = from_union([from_float, from_none], obj.get("volumetricLosses"))
        return CoreLossesOutput(coreLosses, methodUsed, origin, eddyCurrentCoreLosses, hysteresisCoreLosses, magneticFluxDensity, massLosses, temperature, volumetricLosses)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coreLosses"] = to_float(self.coreLosses)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        if self.eddyCurrentCoreLosses is not None:
            result["eddyCurrentCoreLosses"] = from_union([to_float, from_none], self.eddyCurrentCoreLosses)
        if self.hysteresisCoreLosses is not None:
            result["hysteresisCoreLosses"] = from_union([to_float, from_none], self.hysteresisCoreLosses)
        if self.magneticFluxDensity is not None:
            result["magneticFluxDensity"] = from_union([lambda x: to_class(SignalDescriptor, x), from_none], self.magneticFluxDensity)
        if self.massLosses is not None:
            result["massLosses"] = from_union([to_float, from_none], self.massLosses)
        if self.temperature is not None:
            result["temperature"] = from_union([to_float, from_none], self.temperature)
        if self.volumetricLosses is not None:
            result["volumetricLosses"] = from_union([to_float, from_none], self.volumetricLosses)
        return result


@dataclass
class ImpedanceMatrixAtFrequency:
    frequency: float
    """Frequency of the inductance matrix"""

    magnitude: List[List[DimensionWithTolerance]]
    phase: List[List[DimensionWithTolerance]]

    @staticmethod
    def from_dict(obj: Any) -> 'ImpedanceMatrixAtFrequency':
        assert isinstance(obj, dict)
        frequency = from_float(obj.get("frequency"))
        magnitude = from_list(lambda x: from_list(DimensionWithTolerance.from_dict, x), obj.get("magnitude"))
        phase = from_list(lambda x: from_list(DimensionWithTolerance.from_dict, x), obj.get("phase"))
        return ImpedanceMatrixAtFrequency(frequency, magnitude, phase)

    def to_dict(self) -> dict:
        result: dict = {}
        result["frequency"] = to_float(self.frequency)
        result["magnitude"] = from_list(lambda x: from_list(lambda x: to_class(DimensionWithTolerance, x), x), self.magnitude)
        result["phase"] = from_list(lambda x: from_list(lambda x: to_class(DimensionWithTolerance, x), x), self.phase)
        return result


@dataclass
class InductanceMatrixAtFrequency:
    frequency: float
    """Frequency of the inductance matrix"""

    magnitude: List[List[DimensionWithTolerance]]

    @staticmethod
    def from_dict(obj: Any) -> 'InductanceMatrixAtFrequency':
        assert isinstance(obj, dict)
        frequency = from_float(obj.get("frequency"))
        magnitude = from_list(lambda x: from_list(DimensionWithTolerance.from_dict, x), obj.get("magnitude"))
        return InductanceMatrixAtFrequency(frequency, magnitude)

    def to_dict(self) -> dict:
        result: dict = {}
        result["frequency"] = to_float(self.frequency)
        result["magnitude"] = from_list(lambda x: from_list(lambda x: to_class(DimensionWithTolerance, x), x), self.magnitude)
        return result


@dataclass
class ResistanceMatrixAtFrequency:
    frequency: float
    """Frequency of the resitance matrix"""

    magnitude: List[List[DimensionWithTolerance]]

    @staticmethod
    def from_dict(obj: Any) -> 'ResistanceMatrixAtFrequency':
        assert isinstance(obj, dict)
        frequency = from_float(obj.get("frequency"))
        magnitude = from_list(lambda x: from_list(DimensionWithTolerance.from_dict, x), obj.get("magnitude"))
        return ResistanceMatrixAtFrequency(frequency, magnitude)

    def to_dict(self) -> dict:
        result: dict = {}
        result["frequency"] = to_float(self.frequency)
        result["magnitude"] = from_list(lambda x: from_list(lambda x: to_class(DimensionWithTolerance, x), x), self.magnitude)
        return result


@dataclass
class ImpedanceOutput:
    """Data describing the output impedance
    
    Data describing the impendance and the intermediate inputs used to calculate them
    """
    inductanceMatrix: List[InductanceMatrixAtFrequency]
    """List of inductance matrix per frequency"""

    methodUsed: str
    """Model used to calculate the impedance in the case of simulation, or method used to
    measure it
    """
    origin: ResultOrigin
    resistanceMatrix: List[ResistanceMatrixAtFrequency]
    """List of resistance matrix per frequency"""

    impedanceMatrix: Optional[List[ImpedanceMatrixAtFrequency]] = None
    """List of impedance matrix per frequency"""

    @staticmethod
    def from_dict(obj: Any) -> 'ImpedanceOutput':
        assert isinstance(obj, dict)
        inductanceMatrix = from_list(InductanceMatrixAtFrequency.from_dict, obj.get("inductanceMatrix"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        resistanceMatrix = from_list(ResistanceMatrixAtFrequency.from_dict, obj.get("resistanceMatrix"))
        impedanceMatrix = from_union([lambda x: from_list(ImpedanceMatrixAtFrequency.from_dict, x), from_none], obj.get("impedanceMatrix"))
        return ImpedanceOutput(inductanceMatrix, methodUsed, origin, resistanceMatrix, impedanceMatrix)

    def to_dict(self) -> dict:
        result: dict = {}
        result["inductanceMatrix"] = from_list(lambda x: to_class(InductanceMatrixAtFrequency, x), self.inductanceMatrix)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        result["resistanceMatrix"] = from_list(lambda x: to_class(ResistanceMatrixAtFrequency, x), self.resistanceMatrix)
        if self.impedanceMatrix is not None:
            result["impedanceMatrix"] = from_union([lambda x: from_list(lambda x: to_class(ImpedanceMatrixAtFrequency, x), x), from_none], self.impedanceMatrix)
        return result


class VoltageType(Enum):
    """Type of the voltage"""

    AC = "AC"
    DC = "DC"


@dataclass
class DielectricVoltage:
    """Data describing the output insulation that the magnetic has
    
    List of voltages that the magnetic can withstand
    """
    origin: ResultOrigin
    """Origin of the value of the result"""

    voltage: float
    """Voltage that the magnetic withstands"""

    voltageType: VoltageType
    """Type of the voltage"""

    duration: Optional[float] = None
    """Duration of the voltate, or undefined if the field is not present"""

    methodUsed: Optional[str] = None
    """Model used to calculate the voltage in the case of simulation, or method used to measure
    it
    """

    @staticmethod
    def from_dict(obj: Any) -> 'DielectricVoltage':
        assert isinstance(obj, dict)
        origin = ResultOrigin(obj.get("origin"))
        voltage = from_float(obj.get("voltage"))
        voltageType = VoltageType(obj.get("voltageType"))
        duration = from_union([from_float, from_none], obj.get("duration"))
        methodUsed = from_union([from_str, from_none], obj.get("methodUsed"))
        return DielectricVoltage(origin, voltage, voltageType, duration, methodUsed)

    def to_dict(self) -> dict:
        result: dict = {}
        result["origin"] = to_enum(ResultOrigin, self.origin)
        result["voltage"] = to_float(self.voltage)
        result["voltageType"] = to_enum(VoltageType, self.voltageType)
        if self.duration is not None:
            result["duration"] = from_union([to_float, from_none], self.duration)
        if self.methodUsed is not None:
            result["methodUsed"] = from_union([from_str, from_none], self.methodUsed)
        return result


@dataclass
class InsulationCoordinationOutput:
    """Data describing the output insulation coordination that the magnetic has
    
    List of voltages that the magnetic can withstand
    """
    clearance: float
    """Clearance required for this magnetic"""

    creepageDistance: float
    """Creepage distance required for this magnetic"""

    distanceThroughInsulation: float
    """Distance through insulation required for this magnetic"""

    withstandVoltage: float
    """Voltage that the magnetic withstands"""

    withstandVoltageDuration: Optional[float] = None
    """Duration of the voltate, or undefined if the field is not present"""

    withstandVoltageType: Optional[VoltageType] = None
    """Type of the voltage"""

    @staticmethod
    def from_dict(obj: Any) -> 'InsulationCoordinationOutput':
        assert isinstance(obj, dict)
        clearance = from_float(obj.get("clearance"))
        creepageDistance = from_float(obj.get("creepageDistance"))
        distanceThroughInsulation = from_float(obj.get("distanceThroughInsulation"))
        withstandVoltage = from_float(obj.get("withstandVoltage"))
        withstandVoltageDuration = from_union([from_float, from_none], obj.get("withstandVoltageDuration"))
        withstandVoltageType = from_union([VoltageType, from_none], obj.get("withstandVoltageType"))
        return InsulationCoordinationOutput(clearance, creepageDistance, distanceThroughInsulation, withstandVoltage, withstandVoltageDuration, withstandVoltageType)

    def to_dict(self) -> dict:
        result: dict = {}
        result["clearance"] = to_float(self.clearance)
        result["creepageDistance"] = to_float(self.creepageDistance)
        result["distanceThroughInsulation"] = to_float(self.distanceThroughInsulation)
        result["withstandVoltage"] = to_float(self.withstandVoltage)
        if self.withstandVoltageDuration is not None:
            result["withstandVoltageDuration"] = from_union([to_float, from_none], self.withstandVoltageDuration)
        if self.withstandVoltageType is not None:
            result["withstandVoltageType"] = from_union([lambda x: to_enum(VoltageType, x), from_none], self.withstandVoltageType)
        return result


@dataclass
class LeakageInductanceOutput:
    """Data describing the output leakage inductance
    
    Data describing the leakage inductance and the intermediate inputs used to calculate them
    """
    leakageInductancePerWinding: List[DimensionWithTolerance]
    methodUsed: str
    """Model used to calculate the leakage inductance in the case of simulation, or method used
    to measure it
    """
    origin: ResultOrigin

    @staticmethod
    def from_dict(obj: Any) -> 'LeakageInductanceOutput':
        assert isinstance(obj, dict)
        leakageInductancePerWinding = from_list(DimensionWithTolerance.from_dict, obj.get("leakageInductancePerWinding"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        return LeakageInductanceOutput(leakageInductancePerWinding, methodUsed, origin)

    def to_dict(self) -> dict:
        result: dict = {}
        result["leakageInductancePerWinding"] = from_list(lambda x: to_class(DimensionWithTolerance, x), self.leakageInductancePerWinding)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        return result


@dataclass
class AirGapReluctanceOutput:
    """Data describing the reluctance of an air gap"""

    fringingFactor: float
    """Value of the Fringing Factor"""

    maximumStorableMagneticEnergy: float
    """Value of the maximum magnetic energy storable in the gap"""

    methodUsed: str
    """Model used to calculate the magnetizing inductance in the case of simulation, or method
    used to measure it
    """
    origin: ResultOrigin
    reluctance: float
    """Value of the reluctance of the gap"""

    @staticmethod
    def from_dict(obj: Any) -> 'AirGapReluctanceOutput':
        assert isinstance(obj, dict)
        fringingFactor = from_float(obj.get("fringingFactor"))
        maximumStorableMagneticEnergy = from_float(obj.get("maximumStorableMagneticEnergy"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        reluctance = from_float(obj.get("reluctance"))
        return AirGapReluctanceOutput(fringingFactor, maximumStorableMagneticEnergy, methodUsed, origin, reluctance)

    def to_dict(self) -> dict:
        result: dict = {}
        result["fringingFactor"] = to_float(self.fringingFactor)
        result["maximumStorableMagneticEnergy"] = to_float(self.maximumStorableMagneticEnergy)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        result["reluctance"] = to_float(self.reluctance)
        return result


@dataclass
class MagnetizingInductanceOutput:
    """Data describing the output magnetizing inductance
    
    Data describing the magnetizing inductance and the intermediate inputs used to calculate
    them
    """
    coreReluctance: float
    """Value of the reluctance of the core"""

    magnetizingInductance: DimensionWithTolerance
    """Value of the magnetizing inductance"""

    methodUsed: str
    """Model used to calculate the magnetizing inductance in the case of simulation, or method
    used to measure it
    """
    origin: ResultOrigin
    gappingReluctance: Optional[float] = None
    """Value of the reluctance of the gaps"""

    maximumFringingFactor: Optional[float] = None
    """Maximum value of the fringing of the gaps"""

    maximumMagneticEnergyCore: Optional[float] = None
    """Value of the maximum magnetic energy storable in the core"""

    maximumStorableMagneticEnergyGapping: Optional[float] = None
    """Value of the maximum magnetic energy storable in the gaps"""

    reluctancePerGap: Optional[List[AirGapReluctanceOutput]] = None
    """Value of the maximum magnetic energy storable in the gaps"""

    ungappedCoreReluctance: Optional[float] = None
    """Value of the reluctance of the core"""

    @staticmethod
    def from_dict(obj: Any) -> 'MagnetizingInductanceOutput':
        assert isinstance(obj, dict)
        coreReluctance = from_float(obj.get("coreReluctance"))
        magnetizingInductance = DimensionWithTolerance.from_dict(obj.get("magnetizingInductance"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        gappingReluctance = from_union([from_float, from_none], obj.get("gappingReluctance"))
        maximumFringingFactor = from_union([from_float, from_none], obj.get("maximumFringingFactor"))
        maximumMagneticEnergyCore = from_union([from_float, from_none], obj.get("maximumMagneticEnergyCore"))
        maximumStorableMagneticEnergyGapping = from_union([from_float, from_none], obj.get("maximumStorableMagneticEnergyGapping"))
        reluctancePerGap = from_union([lambda x: from_list(AirGapReluctanceOutput.from_dict, x), from_none], obj.get("reluctancePerGap"))
        ungappedCoreReluctance = from_union([from_float, from_none], obj.get("ungappedCoreReluctance"))
        return MagnetizingInductanceOutput(coreReluctance, magnetizingInductance, methodUsed, origin, gappingReluctance, maximumFringingFactor, maximumMagneticEnergyCore, maximumStorableMagneticEnergyGapping, reluctancePerGap, ungappedCoreReluctance)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coreReluctance"] = to_float(self.coreReluctance)
        result["magnetizingInductance"] = to_class(DimensionWithTolerance, self.magnetizingInductance)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        if self.gappingReluctance is not None:
            result["gappingReluctance"] = from_union([to_float, from_none], self.gappingReluctance)
        if self.maximumFringingFactor is not None:
            result["maximumFringingFactor"] = from_union([to_float, from_none], self.maximumFringingFactor)
        if self.maximumMagneticEnergyCore is not None:
            result["maximumMagneticEnergyCore"] = from_union([to_float, from_none], self.maximumMagneticEnergyCore)
        if self.maximumStorableMagneticEnergyGapping is not None:
            result["maximumStorableMagneticEnergyGapping"] = from_union([to_float, from_none], self.maximumStorableMagneticEnergyGapping)
        if self.reluctancePerGap is not None:
            result["reluctancePerGap"] = from_union([lambda x: from_list(lambda x: to_class(AirGapReluctanceOutput, x), x), from_none], self.reluctancePerGap)
        if self.ungappedCoreReluctance is not None:
            result["ungappedCoreReluctance"] = from_union([to_float, from_none], self.ungappedCoreReluctance)
        return result


@dataclass
class SixCapacitorNetworkPerWinding:
    """Network of six equivalent capacitors that describe the capacitance between two given
    windings
    """
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    C6: float

    @staticmethod
    def from_dict(obj: Any) -> 'SixCapacitorNetworkPerWinding':
        assert isinstance(obj, dict)
        C1 = from_float(obj.get("C1"))
        C2 = from_float(obj.get("C2"))
        C3 = from_float(obj.get("C3"))
        C4 = from_float(obj.get("C4"))
        C5 = from_float(obj.get("C5"))
        C6 = from_float(obj.get("C6"))
        return SixCapacitorNetworkPerWinding(C1, C2, C3, C4, C5, C6)

    def to_dict(self) -> dict:
        result: dict = {}
        result["C1"] = to_float(self.C1)
        result["C2"] = to_float(self.C2)
        result["C3"] = to_float(self.C3)
        result["C4"] = to_float(self.C4)
        result["C5"] = to_float(self.C5)
        result["C6"] = to_float(self.C6)
        return result


@dataclass
class TripoleCapacitancePerWinding:
    """The three values of a three input electrostatic multipole that describe the capacitance
    between two given windings
    """
    C1: float
    C2: float
    C3: float

    @staticmethod
    def from_dict(obj: Any) -> 'TripoleCapacitancePerWinding':
        assert isinstance(obj, dict)
        C1 = from_float(obj.get("C1"))
        C2 = from_float(obj.get("C2"))
        C3 = from_float(obj.get("C3"))
        return TripoleCapacitancePerWinding(C1, C2, C3)

    def to_dict(self) -> dict:
        result: dict = {}
        result["C1"] = to_float(self.C1)
        result["C2"] = to_float(self.C2)
        result["C3"] = to_float(self.C3)
        return result


@dataclass
class StrayCapacitanceOutput:
    """Data describing the output stray capacitance
    
    Data describing the stray capacitance and the intermediate inputs used to calculate them
    """
    methodUsed: str
    """Model used to calculate the stray capacitance in the case of simulation, or method used
    to measure it
    """
    origin: ResultOrigin
    """Origin of the value of the result"""

    sixCapacitorNetworkPerWinding: Optional[SixCapacitorNetworkPerWinding] = None
    """Network of six equivalent capacitors that describe the capacitance between two given
    windings
    """
    tripoleCapacitancePerWinding: Optional[TripoleCapacitancePerWinding] = None
    """The three values of a three input electrostatic multipole that describe the capacitance
    between two given windings
    """
    voltageDividerEndPerTurn: Optional[List[float]] = None
    """Voltage divider at the end of the physical turn"""

    voltageDividerStartPerTurn: Optional[List[float]] = None
    """Voltage divider at the start of the physical turn"""

    voltagePerTurn: Optional[List[float]] = None
    """Voltage at the beginning of the physical turn"""

    @staticmethod
    def from_dict(obj: Any) -> 'StrayCapacitanceOutput':
        assert isinstance(obj, dict)
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        sixCapacitorNetworkPerWinding = from_union([SixCapacitorNetworkPerWinding.from_dict, from_none], obj.get("sixCapacitorNetworkPerWinding"))
        tripoleCapacitancePerWinding = from_union([TripoleCapacitancePerWinding.from_dict, from_none], obj.get("tripoleCapacitancePerWinding"))
        voltageDividerEndPerTurn = from_union([lambda x: from_list(from_float, x), from_none], obj.get("voltageDividerEndPerTurn"))
        voltageDividerStartPerTurn = from_union([lambda x: from_list(from_float, x), from_none], obj.get("voltageDividerStartPerTurn"))
        voltagePerTurn = from_union([lambda x: from_list(from_float, x), from_none], obj.get("voltagePerTurn"))
        return StrayCapacitanceOutput(methodUsed, origin, sixCapacitorNetworkPerWinding, tripoleCapacitancePerWinding, voltageDividerEndPerTurn, voltageDividerStartPerTurn, voltagePerTurn)

    def to_dict(self) -> dict:
        result: dict = {}
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        if self.sixCapacitorNetworkPerWinding is not None:
            result["sixCapacitorNetworkPerWinding"] = from_union([lambda x: to_class(SixCapacitorNetworkPerWinding, x), from_none], self.sixCapacitorNetworkPerWinding)
        if self.tripoleCapacitancePerWinding is not None:
            result["tripoleCapacitancePerWinding"] = from_union([lambda x: to_class(TripoleCapacitancePerWinding, x), from_none], self.tripoleCapacitancePerWinding)
        if self.voltageDividerEndPerTurn is not None:
            result["voltageDividerEndPerTurn"] = from_union([lambda x: from_list(to_float, x), from_none], self.voltageDividerEndPerTurn)
        if self.voltageDividerStartPerTurn is not None:
            result["voltageDividerStartPerTurn"] = from_union([lambda x: from_list(to_float, x), from_none], self.voltageDividerStartPerTurn)
        if self.voltagePerTurn is not None:
            result["voltagePerTurn"] = from_union([lambda x: from_list(to_float, x), from_none], self.voltagePerTurn)
        return result


@dataclass
class TemperaturePoint:
    coordinates: List[float]
    """The coordinates of the temperature point, referred to the center of the main column"""

    value: float
    """temperature at the point, in Celsius"""

    @staticmethod
    def from_dict(obj: Any) -> 'TemperaturePoint':
        assert isinstance(obj, dict)
        coordinates = from_list(from_float, obj.get("coordinates"))
        value = from_float(obj.get("value"))
        return TemperaturePoint(coordinates, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["coordinates"] = from_list(to_float, self.coordinates)
        result["value"] = to_float(self.value)
        return result


@dataclass
class TemperatureOutput:
    """Data describing the output temperature
    
    Data describing the temperature and the intermediate inputs used to calculate them
    """
    maximumTemperature: float
    """maximum temperature reached"""

    methodUsed: str
    """Model used to calculate the temperature in the case of simulation, or method used to
    measure it
    """
    origin: ResultOrigin
    bulkThermalResistance: Optional[float] = None
    """bulk thermal resistance of the whole magnetic"""

    initialTemperature: Optional[float] = None
    """Temperature of the magnetic before it started working. If missing ambient temperature
    must be assumed
    """
    temperaturePoint: Optional[TemperaturePoint] = None

    @staticmethod
    def from_dict(obj: Any) -> 'TemperatureOutput':
        assert isinstance(obj, dict)
        maximumTemperature = from_float(obj.get("maximumTemperature"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        bulkThermalResistance = from_union([from_float, from_none], obj.get("bulkThermalResistance"))
        initialTemperature = from_union([from_float, from_none], obj.get("initialTemperature"))
        temperaturePoint = from_union([TemperaturePoint.from_dict, from_none], obj.get("temperaturePoint"))
        return TemperatureOutput(maximumTemperature, methodUsed, origin, bulkThermalResistance, initialTemperature, temperaturePoint)

    def to_dict(self) -> dict:
        result: dict = {}
        result["maximumTemperature"] = to_float(self.maximumTemperature)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        if self.bulkThermalResistance is not None:
            result["bulkThermalResistance"] = from_union([to_float, from_none], self.bulkThermalResistance)
        if self.initialTemperature is not None:
            result["initialTemperature"] = from_union([to_float, from_none], self.initialTemperature)
        if self.temperaturePoint is not None:
            result["temperaturePoint"] = from_union([lambda x: to_class(TemperaturePoint, x), from_none], self.temperaturePoint)
        return result


@dataclass
class OhmicLosses:
    """List of value of the winding ohmic losses"""

    losses: float
    """Value of the losses"""

    origin: ResultOrigin
    """Origin of the value of the result"""

    methodUsed: Optional[str] = None
    """Model used to calculate the magnetizing inductance in the case of simulation, or method
    used to measure it
    """

    @staticmethod
    def from_dict(obj: Any) -> 'OhmicLosses':
        assert isinstance(obj, dict)
        losses = from_float(obj.get("losses"))
        origin = ResultOrigin(obj.get("origin"))
        methodUsed = from_union([from_str, from_none], obj.get("methodUsed"))
        return OhmicLosses(losses, origin, methodUsed)

    def to_dict(self) -> dict:
        result: dict = {}
        result["losses"] = to_float(self.losses)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        if self.methodUsed is not None:
            result["methodUsed"] = from_union([from_str, from_none], self.methodUsed)
        return result


@dataclass
class WindingLossElement:
    """List of value of the winding proximity losses per harmonic
    
    Data describing the losses due to either DC, skin effect, or proximity effect; in a given
    element, which can be winding, section, layer or physical turn
    
    List of value of the winding skin losses per harmonic
    """
    harmonicFrequencies: List[float]
    """List of frequencies of the harmonics that are producing losses"""

    lossesPerHarmonic: List[float]
    """Losses produced by each harmonic"""

    methodUsed: str
    """Model used to calculate the magnetizing inductance in the case of simulation, or method
    used to measure it
    """
    origin: ResultOrigin

    @staticmethod
    def from_dict(obj: Any) -> 'WindingLossElement':
        assert isinstance(obj, dict)
        harmonicFrequencies = from_list(from_float, obj.get("harmonicFrequencies"))
        lossesPerHarmonic = from_list(from_float, obj.get("lossesPerHarmonic"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        return WindingLossElement(harmonicFrequencies, lossesPerHarmonic, methodUsed, origin)

    def to_dict(self) -> dict:
        result: dict = {}
        result["harmonicFrequencies"] = from_list(to_float, self.harmonicFrequencies)
        result["lossesPerHarmonic"] = from_list(to_float, self.lossesPerHarmonic)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        return result


@dataclass
class WindingLossesPerElement:
    ohmicLosses: Optional[OhmicLosses] = None
    """List of value of the winding ohmic losses"""

    proximityEffectLosses: Optional[WindingLossElement] = None
    """List of value of the winding proximity losses per harmonic"""

    skinEffectLosses: Optional[WindingLossElement] = None
    """List of value of the winding skin losses per harmonic"""

    @staticmethod
    def from_dict(obj: Any) -> 'WindingLossesPerElement':
        assert isinstance(obj, dict)
        ohmicLosses = from_union([OhmicLosses.from_dict, from_none], obj.get("ohmicLosses"))
        proximityEffectLosses = from_union([WindingLossElement.from_dict, from_none], obj.get("proximityEffectLosses"))
        skinEffectLosses = from_union([WindingLossElement.from_dict, from_none], obj.get("skinEffectLosses"))
        return WindingLossesPerElement(ohmicLosses, proximityEffectLosses, skinEffectLosses)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.ohmicLosses is not None:
            result["ohmicLosses"] = from_union([lambda x: to_class(OhmicLosses, x), from_none], self.ohmicLosses)
        if self.proximityEffectLosses is not None:
            result["proximityEffectLosses"] = from_union([lambda x: to_class(WindingLossElement, x), from_none], self.proximityEffectLosses)
        if self.skinEffectLosses is not None:
            result["skinEffectLosses"] = from_union([lambda x: to_class(WindingLossElement, x), from_none], self.skinEffectLosses)
        return result


@dataclass
class WindingLossesOutput:
    """Data describing the output winding losses
    
    Data describing the winding losses and the intermediate inputs used to calculate them
    """
    methodUsed: str
    """Model used to calculate the winding losses in the case of simulation, or method used to
    measure it
    """
    origin: ResultOrigin
    windingLosses: float
    """Value of the winding losses"""

    currentDividerPerTurn: Optional[List[float]] = None
    """Excitation of the current per physical turn that produced the winding losses"""

    currentPerWinding: Optional[OperatingPoint] = None
    """Excitation of the current per winding that produced the winding losses"""

    dcResistancePerTurn: Optional[List[float]] = None
    """List of DC resistance per turn"""

    dcResistancePerWinding: Optional[List[float]] = None
    """List of DC resistance per winding"""

    resistanceMatrix: Optional[List[ResistanceMatrixAtFrequency]] = None
    """List of resistance matrix per frequency"""

    temperature: Optional[float] = None
    """temperature in the winding that produced the winding losses"""

    windingLossesPerLayer: Optional[List[WindingLossesPerElement]] = None
    windingLossesPerSection: Optional[List[WindingLossesPerElement]] = None
    windingLossesPerTurn: Optional[List[WindingLossesPerElement]] = None
    windingLossesPerWinding: Optional[List[WindingLossesPerElement]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'WindingLossesOutput':
        assert isinstance(obj, dict)
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        windingLosses = from_float(obj.get("windingLosses"))
        currentDividerPerTurn = from_union([lambda x: from_list(from_float, x), from_none], obj.get("currentDividerPerTurn"))
        currentPerWinding = from_union([OperatingPoint.from_dict, from_none], obj.get("currentPerWinding"))
        dcResistancePerTurn = from_union([lambda x: from_list(from_float, x), from_none], obj.get("dcResistancePerTurn"))
        dcResistancePerWinding = from_union([lambda x: from_list(from_float, x), from_none], obj.get("dcResistancePerWinding"))
        resistanceMatrix = from_union([lambda x: from_list(ResistanceMatrixAtFrequency.from_dict, x), from_none], obj.get("resistanceMatrix"))
        temperature = from_union([from_float, from_none], obj.get("temperature"))
        windingLossesPerLayer = from_union([lambda x: from_list(WindingLossesPerElement.from_dict, x), from_none], obj.get("windingLossesPerLayer"))
        windingLossesPerSection = from_union([lambda x: from_list(WindingLossesPerElement.from_dict, x), from_none], obj.get("windingLossesPerSection"))
        windingLossesPerTurn = from_union([lambda x: from_list(WindingLossesPerElement.from_dict, x), from_none], obj.get("windingLossesPerTurn"))
        windingLossesPerWinding = from_union([lambda x: from_list(WindingLossesPerElement.from_dict, x), from_none], obj.get("windingLossesPerWinding"))
        return WindingLossesOutput(methodUsed, origin, windingLosses, currentDividerPerTurn, currentPerWinding, dcResistancePerTurn, dcResistancePerWinding, resistanceMatrix, temperature, windingLossesPerLayer, windingLossesPerSection, windingLossesPerTurn, windingLossesPerWinding)

    def to_dict(self) -> dict:
        result: dict = {}
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        result["windingLosses"] = to_float(self.windingLosses)
        if self.currentDividerPerTurn is not None:
            result["currentDividerPerTurn"] = from_union([lambda x: from_list(to_float, x), from_none], self.currentDividerPerTurn)
        if self.currentPerWinding is not None:
            result["currentPerWinding"] = from_union([lambda x: to_class(OperatingPoint, x), from_none], self.currentPerWinding)
        if self.dcResistancePerTurn is not None:
            result["dcResistancePerTurn"] = from_union([lambda x: from_list(to_float, x), from_none], self.dcResistancePerTurn)
        if self.dcResistancePerWinding is not None:
            result["dcResistancePerWinding"] = from_union([lambda x: from_list(to_float, x), from_none], self.dcResistancePerWinding)
        if self.resistanceMatrix is not None:
            result["resistanceMatrix"] = from_union([lambda x: from_list(lambda x: to_class(ResistanceMatrixAtFrequency, x), x), from_none], self.resistanceMatrix)
        if self.temperature is not None:
            result["temperature"] = from_union([to_float, from_none], self.temperature)
        if self.windingLossesPerLayer is not None:
            result["windingLossesPerLayer"] = from_union([lambda x: from_list(lambda x: to_class(WindingLossesPerElement, x), x), from_none], self.windingLossesPerLayer)
        if self.windingLossesPerSection is not None:
            result["windingLossesPerSection"] = from_union([lambda x: from_list(lambda x: to_class(WindingLossesPerElement, x), x), from_none], self.windingLossesPerSection)
        if self.windingLossesPerTurn is not None:
            result["windingLossesPerTurn"] = from_union([lambda x: from_list(lambda x: to_class(WindingLossesPerElement, x), x), from_none], self.windingLossesPerTurn)
        if self.windingLossesPerWinding is not None:
            result["windingLossesPerWinding"] = from_union([lambda x: from_list(lambda x: to_class(WindingLossesPerElement, x), x), from_none], self.windingLossesPerWinding)
        return result


@dataclass
class FieldPoint:
    """Data describing the value of a field in a 2D or 3D space"""

    point: List[float]
    """The coordinates of the point of the field"""

    value: float
    """Value of the field at this point"""

    label: Optional[str] = None
    """If this point has some special significance, can be identified with this label"""

    rotation: Optional[float] = None
    """Rotation of the rectangle defining the turn, in degrees"""

    turnIndex: Optional[int] = None
    """If this field point is inside of a wire, this is the index of the turn"""

    turnLength: Optional[float] = None
    """If this field point is inside of a wire, this is the length of the turn"""

    @staticmethod
    def from_dict(obj: Any) -> 'FieldPoint':
        assert isinstance(obj, dict)
        point = from_list(from_float, obj.get("point"))
        value = from_float(obj.get("value"))
        label = from_union([from_str, from_none], obj.get("label"))
        rotation = from_union([from_float, from_none], obj.get("rotation"))
        turnIndex = from_union([from_int, from_none], obj.get("turnIndex"))
        turnLength = from_union([from_float, from_none], obj.get("turnLength"))
        return FieldPoint(point, value, label, rotation, turnIndex, turnLength)

    def to_dict(self) -> dict:
        result: dict = {}
        result["point"] = from_list(to_float, self.point)
        result["value"] = to_float(self.value)
        if self.label is not None:
            result["label"] = from_union([from_str, from_none], self.label)
        if self.rotation is not None:
            result["rotation"] = from_union([to_float, from_none], self.rotation)
        if self.turnIndex is not None:
            result["turnIndex"] = from_union([from_int, from_none], self.turnIndex)
        if self.turnLength is not None:
            result["turnLength"] = from_union([to_float, from_none], self.turnLength)
        return result


@dataclass
class Field:
    """Data describing a field in a 2D or 3D space"""

    data: List[FieldPoint]
    """Value of the magnetizing inductance"""

    frequency: float
    """Value of the field at this point"""

    @staticmethod
    def from_dict(obj: Any) -> 'Field':
        assert isinstance(obj, dict)
        data = from_list(FieldPoint.from_dict, obj.get("data"))
        frequency = from_float(obj.get("frequency"))
        return Field(data, frequency)

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = from_list(lambda x: to_class(FieldPoint, x), self.data)
        result["frequency"] = to_float(self.frequency)
        return result


@dataclass
class WindingWindowCurrentFieldOutput:
    """Data describing the output current field
    
    Data describing the curren in the different chunks used in field calculation
    """
    fieldPerFrequency: List[Field]
    methodUsed: str
    """Model used to calculate the current field"""

    origin: ResultOrigin

    @staticmethod
    def from_dict(obj: Any) -> 'WindingWindowCurrentFieldOutput':
        assert isinstance(obj, dict)
        fieldPerFrequency = from_list(Field.from_dict, obj.get("fieldPerFrequency"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        return WindingWindowCurrentFieldOutput(fieldPerFrequency, methodUsed, origin)

    def to_dict(self) -> dict:
        result: dict = {}
        result["fieldPerFrequency"] = from_list(lambda x: to_class(Field, x), self.fieldPerFrequency)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        return result


@dataclass
class ComplexFieldPoint:
    """Data describing the complex value of a field in a 2D or 3D space"""

    imaginary: float
    """Imaginary value of the field at this point"""

    point: List[float]
    """The coordinates of the point of the field"""

    real: float
    """Real value of the field at this point"""

    label: Optional[str] = None
    """If this point has some special significance, can be identified with this label"""

    turnIndex: Optional[int] = None
    """If this field point is inside of a wire, this is the index of the turn"""

    turnLength: Optional[float] = None
    """If this field point is inside of a wire, this is the length of the turn"""

    @staticmethod
    def from_dict(obj: Any) -> 'ComplexFieldPoint':
        assert isinstance(obj, dict)
        imaginary = from_float(obj.get("imaginary"))
        point = from_list(from_float, obj.get("point"))
        real = from_float(obj.get("real"))
        label = from_union([from_str, from_none], obj.get("label"))
        turnIndex = from_union([from_int, from_none], obj.get("turnIndex"))
        turnLength = from_union([from_float, from_none], obj.get("turnLength"))
        return ComplexFieldPoint(imaginary, point, real, label, turnIndex, turnLength)

    def to_dict(self) -> dict:
        result: dict = {}
        result["imaginary"] = to_float(self.imaginary)
        result["point"] = from_list(to_float, self.point)
        result["real"] = to_float(self.real)
        if self.label is not None:
            result["label"] = from_union([from_str, from_none], self.label)
        if self.turnIndex is not None:
            result["turnIndex"] = from_union([from_int, from_none], self.turnIndex)
        if self.turnLength is not None:
            result["turnLength"] = from_union([to_float, from_none], self.turnLength)
        return result


@dataclass
class ComplexField:
    """Data describing a field in a 2D or 3D space"""

    data: List[ComplexFieldPoint]
    """Value of the magnetizing inductance"""

    frequency: float
    """Value of the field at this point"""

    @staticmethod
    def from_dict(obj: Any) -> 'ComplexField':
        assert isinstance(obj, dict)
        data = from_list(ComplexFieldPoint.from_dict, obj.get("data"))
        frequency = from_float(obj.get("frequency"))
        return ComplexField(data, frequency)

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = from_list(lambda x: to_class(ComplexFieldPoint, x), self.data)
        result["frequency"] = to_float(self.frequency)
        return result


@dataclass
class WindingWindowMagneticStrengthFieldOutput:
    """Data describing the output magnetic strength field"""

    fieldPerFrequency: List[ComplexField]
    methodUsed: str
    """Model used to calculate the magnetic strength field"""

    origin: ResultOrigin

    @staticmethod
    def from_dict(obj: Any) -> 'WindingWindowMagneticStrengthFieldOutput':
        assert isinstance(obj, dict)
        fieldPerFrequency = from_list(ComplexField.from_dict, obj.get("fieldPerFrequency"))
        methodUsed = from_str(obj.get("methodUsed"))
        origin = ResultOrigin(obj.get("origin"))
        return WindingWindowMagneticStrengthFieldOutput(fieldPerFrequency, methodUsed, origin)

    def to_dict(self) -> dict:
        result: dict = {}
        result["fieldPerFrequency"] = from_list(lambda x: to_class(ComplexField, x), self.fieldPerFrequency)
        result["methodUsed"] = from_str(self.methodUsed)
        result["origin"] = to_enum(ResultOrigin, self.origin)
        return result


@dataclass
class Outputs:
    """The description of the outputs that result of simulating a Magnetic"""

    coreLosses: Optional[CoreLossesOutput] = None
    """Data describing the output core losses"""

    impedance: Optional[ImpedanceOutput] = None
    """Data describing the output impedance"""

    insulation: Optional[List[DielectricVoltage]] = None
    """Data describing the output insulation that the magnetic has"""

    insulationCoordination: Optional[InsulationCoordinationOutput] = None
    """Data describing the output insulation coordination that the magnetic has"""

    leakageInductance: Optional[LeakageInductanceOutput] = None
    """Data describing the output leakage inductance"""

    magnetizingInductance: Optional[MagnetizingInductanceOutput] = None
    """Data describing the output magnetizing inductance"""

    strayCapacitance: Optional[List[StrayCapacitanceOutput]] = None
    """Data describing the output stray capacitance"""

    temperature: Optional[TemperatureOutput] = None
    """Data describing the output temperature"""

    windingLosses: Optional[WindingLossesOutput] = None
    """Data describing the output winding losses"""

    windingWindowCurrentDensityField: Optional[WindingWindowCurrentFieldOutput] = None
    """Data describing the output current field"""

    windingWindowCurrentField: Optional[WindingWindowCurrentFieldOutput] = None
    """Data describing the output current field"""

    windingWindowMagneticStrengthField: Optional[WindingWindowMagneticStrengthFieldOutput] = None
    """Data describing the output magnetic strength field"""

    @staticmethod
    def from_dict(obj: Any) -> 'Outputs':
        assert isinstance(obj, dict)
        coreLosses = from_union([CoreLossesOutput.from_dict, from_none], obj.get("coreLosses"))
        impedance = from_union([ImpedanceOutput.from_dict, from_none], obj.get("impedance"))
        insulation = from_union([lambda x: from_list(DielectricVoltage.from_dict, x), from_none], obj.get("insulation"))
        insulationCoordination = from_union([InsulationCoordinationOutput.from_dict, from_none], obj.get("insulationCoordination"))
        leakageInductance = from_union([LeakageInductanceOutput.from_dict, from_none], obj.get("leakageInductance"))
        magnetizingInductance = from_union([MagnetizingInductanceOutput.from_dict, from_none], obj.get("magnetizingInductance"))
        strayCapacitance = from_union([lambda x: from_list(StrayCapacitanceOutput.from_dict, x), from_none], obj.get("strayCapacitance"))
        temperature = from_union([TemperatureOutput.from_dict, from_none], obj.get("temperature"))
        windingLosses = from_union([WindingLossesOutput.from_dict, from_none], obj.get("windingLosses"))
        windingWindowCurrentDensityField = from_union([WindingWindowCurrentFieldOutput.from_dict, from_none], obj.get("windingWindowCurrentDensityField"))
        windingWindowCurrentField = from_union([WindingWindowCurrentFieldOutput.from_dict, from_none], obj.get("windingWindowCurrentField"))
        windingWindowMagneticStrengthField = from_union([WindingWindowMagneticStrengthFieldOutput.from_dict, from_none], obj.get("windingWindowMagneticStrengthField"))
        return Outputs(coreLosses, impedance, insulation, insulationCoordination, leakageInductance, magnetizingInductance, strayCapacitance, temperature, windingLosses, windingWindowCurrentDensityField, windingWindowCurrentField, windingWindowMagneticStrengthField)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.coreLosses is not None:
            result["coreLosses"] = from_union([lambda x: to_class(CoreLossesOutput, x), from_none], self.coreLosses)
        if self.impedance is not None:
            result["impedance"] = from_union([lambda x: to_class(ImpedanceOutput, x), from_none], self.impedance)
        if self.insulation is not None:
            result["insulation"] = from_union([lambda x: from_list(lambda x: to_class(DielectricVoltage, x), x), from_none], self.insulation)
        if self.insulationCoordination is not None:
            result["insulationCoordination"] = from_union([lambda x: to_class(InsulationCoordinationOutput, x), from_none], self.insulationCoordination)
        if self.leakageInductance is not None:
            result["leakageInductance"] = from_union([lambda x: to_class(LeakageInductanceOutput, x), from_none], self.leakageInductance)
        if self.magnetizingInductance is not None:
            result["magnetizingInductance"] = from_union([lambda x: to_class(MagnetizingInductanceOutput, x), from_none], self.magnetizingInductance)
        if self.strayCapacitance is not None:
            result["strayCapacitance"] = from_union([lambda x: from_list(lambda x: to_class(StrayCapacitanceOutput, x), x), from_none], self.strayCapacitance)
        if self.temperature is not None:
            result["temperature"] = from_union([lambda x: to_class(TemperatureOutput, x), from_none], self.temperature)
        if self.windingLosses is not None:
            result["windingLosses"] = from_union([lambda x: to_class(WindingLossesOutput, x), from_none], self.windingLosses)
        if self.windingWindowCurrentDensityField is not None:
            result["windingWindowCurrentDensityField"] = from_union([lambda x: to_class(WindingWindowCurrentFieldOutput, x), from_none], self.windingWindowCurrentDensityField)
        if self.windingWindowCurrentField is not None:
            result["windingWindowCurrentField"] = from_union([lambda x: to_class(WindingWindowCurrentFieldOutput, x), from_none], self.windingWindowCurrentField)
        if self.windingWindowMagneticStrengthField is not None:
            result["windingWindowMagneticStrengthField"] = from_union([lambda x: to_class(WindingWindowMagneticStrengthFieldOutput, x), from_none], self.windingWindowMagneticStrengthField)
        return result


@dataclass
class Mas:
    """All the data structure used in the Magnetic Agnostic Structure"""

    inputs: Inputs
    """The description of the inputs that can be used to design a Magnetic"""

    magnetic: Magnetic
    """The description of a magnetic"""

    outputs: List[Outputs]
    """The description of the outputs that are produced after designing a Magnetic"""

    @staticmethod
    def from_dict(obj: Any) -> 'Mas':
        assert isinstance(obj, dict)
        inputs = Inputs.from_dict(obj.get("inputs"))
        magnetic = Magnetic.from_dict(obj.get("magnetic"))
        outputs = from_list(Outputs.from_dict, obj.get("outputs"))
        return Mas(inputs, magnetic, outputs)

    def to_dict(self) -> dict:
        result: dict = {}
        result["inputs"] = to_class(Inputs, self.inputs)
        result["magnetic"] = to_class(Magnetic, self.magnetic)
        result["outputs"] = from_list(lambda x: to_class(Outputs, x), self.outputs)
        return result


def Masfromdict(s: Any) -> Mas:
    return Mas.from_dict(s)


def Mastodict(x: Mas) -> Any:
    return to_class(Mas, x)
