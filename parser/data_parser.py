import rdflib
import json
from config import *
from data_classes import Mutant, ParserStatistic
from diff_parser import UnifiedDiffParser
from utils import *

class DatasetParser:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.json_dict = None
        self.extracted_data = []

    def load_dataset(self):
        g = rdflib.Graph()
        g.parse(self.dataset_file, format="turtle")
        serialized_json = g.serialize(format='json-ld', indent=4)
        self.json_dict = json.loads(serialized_json)
    
    def parse_mutants(self, num_to_parse):
        if self.json_dict is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first")
        
        ps = ParserStatistic()

        for i, element in enumerate(self.json_dict):
            # Elements to ignore/filter based on their dictionary key
            if any(key in element for key in keys_to_ignore) or difference_key not in element or operator_key not in element:
                ps.num_ignored += 1
                continue
            # Stop if num_parsed has been reached
            if ps.num_parsed == num_to_parse and num_to_parse != 0:
                break
            
            if id_val := element.get('@id'):
                id_str = id_val.replace("mb:mutant#", "")
            else:
                raise ValueError("ID Key not found")

            if program_key in element:
                program = element[program_key][0].get('@id')
                program_name = program.replace("mb:program#", "")
                extension = get_file_extension(program_name);
            else:
                raise ValueError("Program key not found")
            
            if difference_key in element:
                unified_diff = element[difference_key][0].get('@value')
            else:
                raise ValueError("Difference key not found")

            if operator_key in element:
                op_list_uncleaned = element[operator_key]
                operator_list = []
                for operators in op_list_uncleaned:
                    operator_ids = operators['@id'].split(",")
                    for operator_id in operator_ids:
                        operator_cleaned = operator_id.replace("mb:operator#", "").strip()
                        operator_list.append(operator_cleaned)
            else:
                raise ValueError("Operator key not found")

            if equivalence_key in element:
                equiv = element[equivalence_key][0].get('@value')
            else:
                raise ValueError("Equivalence key not found")

            mutant = Mutant(id_str, i+1, operator_list, program_name, extension, unified_diff, equiv)
            ps.num_parsed += 1

            if mutant.file_extension == "java":
                ps.num_java_mutants += 1
                dp = UnifiedDiffParser(mutant.program_name, mutant.unified_diff)
                full_code = dp.parse_file(save_to_file=save_mutants_to_file, extract_methods=save_only_methods)
                print('parsed ' + mutant.program_name + ' ' + str(mutant.id_num))
                if None not in [mutant.id_str, mutant.file_extension, mutant.equivalence] and full_code is not None:
                    self.extracted_data.append({'id': mutant.id_num, 'code': full_code[0], 'operator': mutant.operator_list, 'label': mutant.equivalence})
            elif mutant.file_extension == "c":
                ps.num_c_mutants += 1

        # Save processed dataset to c2v/pkl file, and return the parser stats
        list_to_csv(self.extracted_data, pkl_save_path)
        list_to_pkl(self.extracted_data, pkl_save_path)
        return ps


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