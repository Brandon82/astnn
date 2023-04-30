import rdflib
import json
from config import *
from data_classes import Mutant, ParserStatistic
from diff_parser import UnifiedDiffParser
from utils import list_to_csv, list_to_pkl, split_data
import csv
import splitfolders

class DatasetParser:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.json_dict = None
        self.extracted_data = []
        self.ps = ParserStatistic()

    def load_dataset(self):
        g = rdflib.Graph()
        g.parse(self.dataset_file, format="turtle")
        serialized_json = g.serialize(format='json-ld', indent=4)
        self.json_dict = json.loads(serialized_json)
    
    def parse_mutants(self, num_to_parse):
        if self.json_dict is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first")

        for i, element in enumerate(self.json_dict):
            # Elements to ignore/filter based on their dict key
            if any(key in element for key in keys_to_ignore) or difference_key not in element or operator_key not in element:
                self.ps.num_ignored += 1
                continue
            # Stop if num_parsed has been reached
            if self.ps.num_parsed == num_to_parse and num_to_parse != 0:
                break
            
            # Now create the mutant data object
            mutant = Mutant()
            mutant.id_num = i

            # Cleaning the dataset:
            if id_val := element.get('@id'):
                id_val_cleaned = id_val.replace("mb:mutant#", "")
                mutant.id_str = id_val_cleaned
            else:
                raise ValueError("ID Key not found")

            if program_key in element:
                program = element[program_key][0].get('@id')
                program_cleaned = program.replace("mb:program#", "")
                mutant.program = program_cleaned
            else:
                raise ValueError("Program key not found")
            
            if difference_key in element:
                difference = element[difference_key][0].get('@value')
                mutant.difference = difference
            else:
                raise ValueError("Difference key not found")

            if operator_key in element:
                operators_list = element[operator_key]
                if len(operators_list) == 1:
                    mutant.operator = operators_list[0]['@id'].replace("mb:operator#", "").strip()
                else:
                    for operators in operators_list:
                        operator_ids = operators['@id'].split(",")
                        for operator_id in operator_ids:
                            operator_cleaned = operator_id.replace("mb:operator#", "").strip()
                            mutant.operator.append(operator_cleaned)
            else:
                raise ValueError("Operator key not found")

            if equivalence_key in element:
                equivalence = element[equivalence_key][0].get('@value')
                mutant.equivalence = equivalence
                if mutant.equivalence == 'true':
                    mutant.label = 2
                    self.ps.num_equiv += 1
                    if mutant.type == 'java':
                        self.ps.num_java_equiv +=1
                elif mutant.equivalence == 'false':
                    mutant.label = 1
                    self.ps.num_non_equiv += 1
                    if mutant.type == 'java':
                        self.ps.num_java_nonequiv +=1
            else:
                raise ValueError("Equivalence key not found")

            mutant.calculate_type()
            self.ps.num_parsed += 1

            # Now that information about the mutant has been stored,
            # the data can be further processed and filtered:

            if mutant.type == "java":
                self.ps.num_java_mutants += 1
                dp = UnifiedDiffParser(mutant.program, mutant.difference)
                full_code = dp.parse_file(save_to_file=save_mutants_to_file, extract_methods=save_only_methods)
                print('parsed ' + mutant.program)
                if None not in [mutant.id_str, mutant.type, mutant.equivalence] and full_code is not None and isinstance(mutant.operator, str):
                    self.extracted_data.append({'id': mutant.id_num, 'code': full_code[0], 'operator': mutant.operator, 'label': mutant.label, 'method' : ''})
            elif mutant.type == "c":
                self.ps.num_c_mutants += 1

        # After the entire dataset is iterated and processed, save to c2v file, save split dataset, and return the parser stats
        list_to_csv(self.extracted_data)
        list_to_pkl(self.extracted_data)
        split_data()
        return self.ps


if __name__ == "__main__":
    parser = DatasetParser(dataset_file)
    parser.load_dataset()
    parser_stats = parser.parse_mutants(num_to_parse=num_mutants_to_parse)

    print(f"DICT SIZE: {len(parser.json_dict)}\n")
    print(f"NUM_PARSED = {parser_stats.num_parsed}")
    print(f"NUM_IGNORED = {parser_stats.num_ignored}")
    print(f"NUM_EQUIV = {parser_stats.num_equiv}")
    print(f"NUM_NON_EQUIV = {parser_stats.num_non_equiv}")
    print(f"NUM_JAVA_MUTANTS = {parser_stats.num_java_mutants}")
    print(f"NUM_C_MUTANTS = {parser_stats.num_c_mutants}")
    print(f"NUM_JAVA_EQUIV = {parser_stats.num_java_equiv}")
    print(f"NUM_JAVA_NONEQUIV = {parser_stats.num_java_nonequiv}")