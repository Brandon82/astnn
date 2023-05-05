from config import *
from dataclasses import dataclass
from typing import List

@dataclass
class Mutant:
    id_str: str
    id_num: int
    operator_list: List[str]
    program_name: str
    file_extension: str
    unified_diff: str
    equivalence: bool

@dataclass
class ParserStatistic:
    num_parsed: int = 0
    num_ignored: int = 0
    num_equiv: int = 0
    num_non_equiv: int = 0
    num_java_mutants: int = 0
    num_c_mutants: int = 0
    num_java_equiv: int = 0
    num_java_nonequiv: int = 0