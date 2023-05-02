import re
import csv
from config import *
import os
import random
import shutil
import pickle

def get_file_extension(program_name: str) -> str:
    return program_name.split(".")[-1]

def extract_java_method(full_src: str, line_num_hint: int) -> str:    
    lines = full_src.split('\n')
    start_line = find_method_start(line_num_hint, lines)
    end_line = find_method_end(start_line, lines)
    return '\n'.join(lines[start_line:end_line+1])

def find_method_start(line_num_hint, lines):
    for i in range(line_num_hint, -1, -1):
        line = lines[i].strip()
        if line.startswith("public") or line.startswith("private") or line.startswith("protected"):
            return i
    return None

def find_method_end(start_line_num, lines):
    line_count = start_line_num
    for _ in range(line_count, line_count + 1):
        for char in lines:
            if char == '{':
                line_count += 1
                break
    brace_count = 0
    for i in range(line_count, len(lines)):
        line = lines[i].strip()
        for char in line:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        if brace_count == 0:
            return i
    return None

def list_to_csv(extracted_data_list, fp):
    with open(fp +'myprograms.csv', mode='w', newline='') as csv_file:
        fieldnames = ['id', 'code', 'operator', 'label', 'method']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_data_list)

def list_to_pkl(extracted_data_list, fp):
    with open(fp + '/myprograms.pkl', 'wb') as pkl_file:
        pickle.dump(extracted_data_list, pkl_file)