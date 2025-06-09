import math
import MAS_models as MAS


class Outputs:

    def __init__(self, project):
        self.project = project

    def create_report(self):
        self.project.create_output_variable("Q", "2*Pi*freq*current_matrix.L(current_0_in,current_0_in)/current_matrix.R(current_0_in,current_0_in)")
        # self.project.create_output_variable("Zreal", f"2 * Pi * freq * current_matrix.L(current_0_in,current_0_in) * pwl($material_{material_name}_complexImaginaryPermeability, Freq)")
        # self.project.create_output_variable("Zimaginary", f"2 * Pi * freq * current_matrix.L(current_0_in,current_0_in) * pwl($material_{material_name}_complexRealPermeability, Freq)")
        report = self.project.post.create_report(plot_name="L Table 1")
        report.add_trace_to_report(["re(solution_matrix.Z(Primary_winding_parallel_0,Primary_winding_parallel_0))"])
        # report.add_trace_to_report(["current_matrix.L(current_0_in,current_0_in)", "current_matrix.R(current_0_in,current_0_in)", "Q"])
        # self.project.export_results(export_folder=f"{os.path.expanduser('~')}\\Downloads\\")

    def get_results(self):

        def get_category_data(category, include_phase=False):
            category_data = []
            available_report_quantities = self.project.post.available_report_quantities(context={"solution_matrix": "windings"}, quantities_category=category)
            data = self.project.post.get_solution_data(expressions=available_report_quantities, context={"solution_matrix": "windings"})

            number_windings = int(math.sqrt(len(available_report_quantities)))
            frequency_multiplier = 1e9
            if data.units_sweeps == "GHz":
                frequency_multiplier = 1e9
            for frequency in [x * frequency_multiplier for x in data.primary_sweep_values]:
                matrix_per_frequency = {"frequency": frequency, "magnitude": []}
                for _ in range(number_windings):
                    matrix_per_frequency["magnitude"].append([None] * number_windings)
                if include_phase:
                    matrix_per_frequency["phase"] = []
                    for _ in range(number_windings):
                        matrix_per_frequency["phase"].append([None] * number_windings)

                category_data.append(matrix_per_frequency) 

            for expression_index, expression in enumerate(available_report_quantities):
                horizontal_winding_index = int(math.floor(expression_index / number_windings))
                vertical_winding_index = expression_index % number_windings
                data_per_frequency = data.data_magnitude(expression=expression, convert_to_SI=True)
                for frequency_index, datum in enumerate(data_per_frequency):
                    category_data[frequency_index]["magnitude"][horizontal_winding_index][vertical_winding_index] = {"nominal": datum}
                if include_phase:
                    data_per_frequency = data.data_phase(expression=expression)
                    for frequency_index, datum in enumerate(data_per_frequency):
                        category_data[frequency_index]["phase"][horizontal_winding_index][vertical_winding_index] = {"nominal": datum}

            return category_data

        impedance_dict = {
            "methodUsed": "Ansys Maxwell",
            "origin": MAS.ResultOrigin.simulation,
            "inductanceMatrix": get_category_data("L"),
            "resistanceMatrix": get_category_data("R"),
            "impedanceMatrix": get_category_data("Z", True),
        }
        import pprint
        pprint.pprint(impedance_dict)
        impedance = MAS.ImpedanceOutput.from_dict(impedance_dict)

        return impedance
