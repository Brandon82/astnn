from config import *
from utils import *

class Mutant:
    def __init__(self, id_str='', operator=None, equivalence='', program='', difference=''):
        # Each of these represent the values of the dataset
        self.id_str = id_str
        self.id_num = None
        self.operator = operator if operator is not None else []
        self.equivalence = equivalence
        self.label = None
        self.program = program
        self.name = ''
        self.difference = difference
        self.type = None
    
            
    def calculate_type(self):
        self.type = get_file_extension(self.program)


class ParserStatistic:
    def __init__(self, num_parsed=0, num_ignored=0, num_equiv=0, num_non_equiv=0, num_java_mutants=0, num_c_mutants=0, num_java_equiv=0, num_java_nonequiv=0):
        self.num_parsed = num_parsed
        self.num_ignored = num_ignored
        self.num_equiv = num_equiv
        self.num_non_equiv = num_non_equiv
        self.num_java_mutants = num_java_mutants
        self.num_c_mutants = num_c_mutants
        self.num_java_equiv = num_java_equiv
        self.num_java_nonequiv = num_java_nonequiv