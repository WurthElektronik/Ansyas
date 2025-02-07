import math
from enum import Enum
import MAS_models as MAS


color_ferrite = [89, 94, 107]
color_copper = [242, 140, 102]
color_litz = [0xd3, 0xe1, 0xf5]
color_teflon = [0x53, 0x97, 0x96]
color_bobbin = [52, 52, 52]
color_tape = [0xed, 0xbe, 0x1c]


class DimensionalValues(Enum):
    maximum = "Maximum"
    nominal = "Nominal"
    minimum = "Minimum"


def resolve_dimensional_values(dimensionValue: MAS.DimensionWithTolerance, preferredValue: DimensionalValues = DimensionalValues.nominal):
    if preferredValue is DimensionalValues.maximum:
        if dimensionValue.maximum is not None:
            return dimensionValue.maximum
        elif dimensionValue.nominal is not None:
            return dimensionValue.nominal
        elif dimensionValue.minimum is not None:
            return dimensionValue.minimum
    elif preferredValue is DimensionalValues.nominal:
        if dimensionValue.nominal is not None:
            return dimensionValue.nominal
        elif dimensionValue.maximum is not None and dimensionValue.minimum is not None:
            return (dimensionValue.maximum + dimensionValue.minimum) / 2
        elif dimensionValue.maximum is not None:
            return dimensionValue.maximum
        elif dimensionValue.minimum is not None:
            return dimensionValue.minimum
    elif preferredValue is DimensionalValues.minimum:
        if dimensionValue.minimum is not None:
            return dimensionValue.minimum
        elif dimensionValue.nominal is not None:
            return dimensionValue.nominal
        elif dimensionValue.maximum is not None:
            return dimensionValue.maximum
    else:
        assert 0, "Unknown type of dimension, options are {MAXIMUM, NOMINAL, MINIMUM}"


def convert_units(units, scale):
    if isinstance(units, list):
        new_units = []
        for unit in units:
            new_units.append(unit * scale)
        return new_units

    else:
        return units * scale


def convert_axis(coordinates):
    if len(coordinates) == 2:
        return [0, coordinates[0], coordinates[1]]
    elif len(coordinates) == 3:
        return [coordinates[2], coordinates[0], coordinates[1]]
    else:
        assert False, "Invalid coordinates length"


def convert_axis_toroidal_core(coordinates):
    if len(coordinates) == 2:
        return [-coordinates[1], coordinates[0], 0]
    elif len(coordinates) == 3:
        return [-coordinates[1], coordinates[0], coordinates[2]]
    else:
        assert False, "Invalid coordinates length"


def clean_name(name):
    return name.replace(" ", "_").title()


def get_distance(point_a, point_b):
    distance = 0
    for index in range(0, len(point_a)):
        distance += math.pow(point_a[index] - point_b[index], 2)

    return math.sqrt(distance)


def get_closest_face(object3D, position=[0, 0, 0]):
    faces = object3D.faces
    closest_face = None
    closest_distance = math.inf
    for face in faces:
        distance = get_distance(face.center, position)
        if distance < closest_distance:
            closest_distance = distance
            closest_face = face

    return closest_face


def sweep_around_axis_with_offset_and_angle(project, object3D, axis, offset=[0, 0, 0], offset_rotation=0, offset_rotation_axis="Z", angle=360, number_of_segments=12):
    object3D.rotate(offset_rotation_axis, -offset_rotation)
    object3D.move([-offset[0], -offset[1], -offset[2]])
    
    project.modeler.sweep_around_axis(
        assignment=object3D,
        axis=axis,
        sweep_angle=angle,
        draft_angle=0,
        number_of_segments=number_of_segments
    )
    object3D.move(offset)
    assert 0

    return object3D


def cartesian_to_polar(value, radialHeight):
    angle = math.atan2(value[1], value[0]) * 180 / math.pi
    if angle < 0:
        angle += 360

    radius = math.hypot(value[0], value[1])
    turnRadialHeight = radialHeight - radius
    return [turnRadialHeight, angle]
