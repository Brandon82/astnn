import os
import pickle
import pandas as pd

file_paths = ['./data/ast.pkl',
                './data/programs.pkl',
                './data/split_data/dev/blocks.pkl',
                './data/split_data/test/blocks.pkl',
                './data/split_data/train/blocks.pkl',
                './data/split_data/dev/dev.pkl',
                './data/split_data/test/test.pkl',
                './data/split_data/train/train.pkl'
            ]


def load_pickle_to_dataframe(file_path):
    with open(file_path, 'rb') as infile:
        return pd.read_pickle(infile)

def pkl_to_csv(path):
    df = load_pickle_to_dataframe(path)
    csv_file = path.replace('.pkl', '.csv')
    df.to_csv(csv_file, index=False)
    print(f'Successfully converted {path} to {csv_file}')

def tsv_to_csv(path):
    df = pd.read_csv(path, sep='\t')
    csv_file = path.replace('.tsv', '.csv')
    df.to_csv(csv_file, index=False)
    print(f'Successfully converted {path} to {csv_file}')

def convert_all():
    for path in file_paths:
        if path.endswith('.pkl'):
            pkl_to_csv(path)
            print('Saved csv')
        elif path.endswith('.tsv'):
            tsv_to_csv(path)
        else:
            print(f'Unsupported file format for {path}')


# convert_all()
