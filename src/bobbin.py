import pyaedt
import ansyas_utils
import MAS_models as MAS


class Bobbin:

    def __init__(self, project, number_segments_arcs=12):
        self.number_segments_arcs = number_segments_arcs
        self.project = project

    def calculate_null_bobbin(self, core: MAS.MagneticCore):

        bobbin = MAS.Bobbin()

        bobbin_data = {
            "processedDescription": {
                "columnDepth": core.processedDescription.columns[0].depth / 2,
                "columnShape": str(core.processedDescription.columns[0].shape.name),
                "columnThickness": 0,
                "columnWidth": core.processedDescription.columns[0].width / 2,
                "coordinates": [
                    0,
                    0,
                    0
                ],
                "wallThickness": 0,
                "windingWindows": [
                    {
                        "area": 0.00018173250000000002,
                        "coordinates": core.processedDescription.windingWindows[0].coordinates,
                        "height": core.processedDescription.windingWindows[0].height,
                        "radialHeight": core.processedDescription.windingWindows[0].radialHeight,
                        "angle": core.processedDescription.windingWindows[0].angle,
                        "sectionsOrientation": "overlapping",
                        "shape": "round" if core.functionalDescription.shape.family == MAS.CoreShapeFamily.t else "rectangular",
                        "width": core.processedDescription.windingWindows[0].width
                    }
                ]
            }
        }
        bobbin = MAS.Bobbin.from_dict(bobbin_data)

        return bobbin

    def create_simple_bobbin(self, bobbin: MAS.Bobbin, material="plastic"):
        bobbinData = bobbin.processedDescription
        if round(bobbinData.wallThickness, 12) == 0 or round(bobbinData.columnThickness, 12) == 0:
            return None

        total_height = bobbinData.windingWindows[0].height + bobbinData.wallThickness * 2
        total_width = bobbinData.windingWindows[0].width + bobbinData.columnWidth
        total_depth = bobbinData.windingWindows[0].width + bobbinData.columnDepth
        if bobbinData.columnShape is MAS.ColumnShape.round:
            bobbin = self.project.modeler.create_cylinder(
                orientation=pyaedt.constants.AXIS.Z,
                origin=[0, 0, -total_height / 2],
                radius=total_width,
                height=total_height,
                num_sides=self.number_segments_arcs if material is not None else 0,
                name="bobbin",
                material=material
            )
            negative_winding_window = self.project.modeler.create_cylinder(
                orientation=pyaedt.constants.AXIS.Z,
                origin=[0, 0, -bobbinData.windingWindows[0].height / 2],
                radius=total_width,
                height=bobbinData.windingWindows[0].height,
                num_sides=self.number_segments_arcs if material is not None else 0,
                name="negative_winding_window",
                material=material
            )
            central_column = self.project.modeler.create_cylinder(
                orientation=pyaedt.constants.AXIS.Z,
                origin=[0, 0, -bobbinData.windingWindows[0].height / 2],
                radius=bobbinData.columnWidth,
                height=bobbinData.windingWindows[0].height,
                num_sides=self.number_segments_arcs if material is not None else 0,
                name="central_column",
                material=material
            )
            central_hole = self.project.modeler.create_cylinder(
                orientation=pyaedt.constants.AXIS.Z,
                origin=[0, 0, -total_height / 2],
                radius=bobbinData.columnWidth - bobbinData.columnThickness,
                height=total_height,
                num_sides=self.number_segments_arcs if material is not None else 0,
                name="central_hole",
                material=material
            )
            negative_winding_window.subtract(central_column, False)
            bobbin.subtract(negative_winding_window, False)
            bobbin.subtract(central_hole, False)

        else:
            bobbin = self.project.modeler.create_box(
                origin=[-total_depth, -total_width, -total_height / 2],
                sizes=[total_depth * 2, total_width * 2, total_height],
                name="bobbin",
                material=material
            )
            negative_winding_window = self.project.modeler.create_box(
                origin=[-total_depth, -total_width, -bobbinData.windingWindows[0].height / 2],
                sizes=[total_depth * 2, total_width * 2, bobbinData.windingWindows[0].height],
                name="negative_winding_window",
                material=material
            )
            central_column = self.project.modeler.create_box(
                origin=[-bobbinData.columnDepth, -bobbinData.columnWidth, -bobbinData.windingWindows[0].height / 2],
                sizes=[bobbinData.columnDepth * 2, bobbinData.columnWidth * 2, bobbinData.windingWindows[0].height],
                name="central_column",
                material=material
            )
            central_hole = self.project.modeler.create_box(
                origin=[-(bobbinData.columnDepth - bobbinData.columnThickness), -(bobbinData.columnWidth - bobbinData.columnThickness), -total_height / 2],
                sizes=[(bobbinData.columnDepth - bobbinData.columnThickness) * 2, (bobbinData.columnWidth - bobbinData.columnThickness) * 2, total_height],
                name="central_hole",
                material=material
            )
            negative_winding_window.subtract(central_column, False)
            bobbin.subtract(negative_winding_window, False)
            bobbin.subtract(central_hole, False)

        bobbin.color = ansyas_utils.color_bobbin
        bobbin.model = material is not None
        return bobbin
