import os
from utils import *
from config import *

class UnifiedDiffParser:
    def __init__(self, program=None, difference=None):
        self.program = program
        self.difference = difference
        self.string_to_del = ""
        self.string_to_add = ""
        self.line_num_to_del = None
        self.line_num_to_del2 = None
        self.line_num_to_add = None
        self.line_num_to_add2 = None
        self.new_file_name = ''
        self.counter = 1
        self.parse_unified_diff()
  
    def parse_unified_diff(self):
        # split the self.difference into lines
        lines = self.difference.strip().split("\n")
        self.line_num_to_del = int(lines[0].split(" ")[1].split(",")[0][1:])
        self.line_num_to_del2 = int(lines[0].split(" ")[1].split(",")[1][0:]) if len(lines[0].split(" ")[1].split(",")) > 1 else None
        self.line_num_to_add = int(lines[0].split(" ")[2].split(",")[0][1:])
        self.line_num_to_add2 = int(lines[0].split(" ")[2].split(",")[1][0:]) if len(lines[0].split(" ")[2].split(",")) > 1 else None
        # remove the first line
        lines = lines[1:]
   
        for i, line in enumerate(lines):
            if line.startswith("-"):
                self.string_to_del += line[1:]
                if i+1 < len(lines):
                    if lines[i+1].startswith("-"):
                        self.string_to_del += "\n"
            elif line.startswith("+"):
                self.string_to_add += line[1:]
                if i+1 < len(lines):
                    if lines[i+1].startswith("+"):
                        self.string_to_add += "\n"
            else:
                pass

    def parse_file(self, save_to_file, extract_methods):
        with open(f"{original_program_path}{self.program}", 'r') as f:
            file = iter(f)
            file_copy = list(file)

        #end_line = self.line_num_to_del + self.line_num_to_del2 if self.line_num_to_del2 is not None else self.line_num_to_add
        #del file_copy[self.line_num_to_add-1:end_line]
        #file_copy.insert(self.line_num_to_add - 1, self.string_to_add + '\n')

        if self.line_num_to_add2 is None and self.line_num_to_del2 is None:
            end_line = self.line_num_to_del2 if self.line_num_to_del2 is not None else self.line_num_to_del
            del file_copy[self.line_num_to_del-1:end_line]
            file_copy.insert(self.line_num_to_add-1, self.string_to_add + '\n')

            mod_str = ''.join(file_copy)

            if extract_methods and self.program != 'Profit.java':
                mod_str = extract_java_method(mod_str, self.line_num_to_del-1)
                if save_to_file:
                    self.save_str_to_file(mod_str, mode=1)
            else:
                if save_to_file or self.program == 'Profit.java':
                    self.save_str_to_file(file_copy, mode=1)

            return [mod_str]
        else:
            print('Ignoring mutant which diff lines > 1')
            return None

    def save_str_to_file(self, file_content, save_path=mutant_save_path, mode=0):
        original_file_name = f"{self.program}.original"

        filename, extension = os.path.splitext(self.program)
        self.new_file_name = f"{filename}_{self.counter}{extension}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        modified_file_path = os.path.join(save_path, self.new_file_name)

        while os.path.exists(modified_file_path):
            self.new_file_name = f"{filename}_{self.counter}{extension}"
            modified_file_path = os.path.join(save_path, self.new_file_name)
            self.counter += 1

        if mode == 0:  # Save no files
            pass
        elif mode == 1:  # Save only the modified file
            if file_content is not None:
                with open(modified_file_path, "w") as mod_file:
                    mod_file.write(''.join(file_content))
            else:
                print('skipped null list')
        else:
            raise ValueError("Invalid mode. Use 0 for saving no files, 1 for saving only the modified file")
        
    def print_all(self):
        print('del ' + str(self.line_num_to_del))
        print('del2 ' + str(self.line_num_to_del2))
        print('add ' + str(self.line_num_to_add))
        print('add2 ' + str(self.line_num_to_add2))
        print('deleting: ' + self.string_to_del)
        print('adding: ' + self.string_to_add)
        print('file: ' + self.program)
        print('diff: ' + self.difference)