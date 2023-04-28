import os
import pickle
import pandas as pd

file3 = './data/ast.pkl'
file4 = './data/programs.csv'
file5 = './data/c/programs.pkl'
file6 = './data/java/bcb_pair_ids.pkl'
file7 = './data/java/bcb_funcs_all.tsv'

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


#df = load_pickle_to_dataframe(file6)
#print(df)

#pkl_to_csv(file5)
#pkl_to_csv(file6)
#tsv_to_csv(file7)