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
    """
    Extracts the Java method that contains the given line number hint from the provided source code.

    Args:
        full_src (str): The source code of the Java program.
        line_num_hint (int): The line number hint, which should be within the method to extract.

    Returns:
        str: The code for the Java method that contains the given line number hint, or None if the method cannot be found.

    """

    # Split the source code into lines
    lines = full_src.split('\n')
    start_line = find_method_start(line_num_hint, lines)
    end_line = find_method_end(start_line, lines)

    method_code = '\n'.join(lines[start_line:end_line+1])
    return method_code


def find_method_start(line_num_hint, lines):
    for i in range(line_num_hint, -1, -1):
        line = lines[i].strip()
        if line.startswith("public") or line.startswith("private") or line.startswith("protected"):
            return i
    return None

def find_method_end(start_line_num, lines):
    line_count = start_line_num
    for i in range(start_line_num, start_line_num+1):
        for char in lines:
            if char == '{':
                line_count += 1
                break
            else:
                pass

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
    return -1


def list_to_csv(extracted_data_list):
    with open('myprograms.csv', mode='w', newline='') as csv_file:
        fieldnames = ['id', 'code', 'operator', 'label', 'method']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_data_list)

def list_to_pkl(extracted_data_list):
    with open('myprograms.pkl', 'wb') as pkl_file:
        pickle.dump(extracted_data_list, pkl_file)


def split_data(ratio=(0.7, 0.15, 0.15)):
    # ratio is a tuple with the proportions of train, val, and test data, respectively
    assert sum(ratio) == 1.0, "The sum of the ratio values must be 1.0"

    # create the output directories
    train_path = os.path.join(mutant_save_split_path, "train")
    val_path = os.path.join(mutant_save_split_path, "val")
    test_path = os.path.join(mutant_save_split_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # get the list of files to split
    files = os.listdir(mutant_save_path)
    random.shuffle(files)

    # split the files based on the ratio
    n = len(files)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    # copy the files to the output directories
    for filename in train_files:
        src_path = os.path.join(mutant_save_path, filename)
        dst_path = os.path.join(train_path, filename)
        shutil.copy(src_path, dst_path)

    for filename in val_files:
        src_path = os.path.join(mutant_save_path, filename)
        dst_path = os.path.join(val_path, filename)
        shutil.copy(src_path, dst_path)

    for filename in test_files:
        src_path = os.path.join(mutant_save_path, filename)
        dst_path = os.path.join(test_path, filename)
        shutil.copy(src_path, dst_path)