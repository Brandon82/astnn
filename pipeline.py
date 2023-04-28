import pandas as pd
import os
from tqdm.auto import tqdm
tqdm.pandas()

data_path = './data/'
programs_data_path = './data/pre/programs.csv'
parsed_data_path = './data/pre/ast.csv'

split_train_path = './data/split_data/train/'
split_test_path = './data/split_data/test/'
split_dev_path = './data/split_data/dev/'

embedding_save_path = './data/embedding/train'

blocks_train_save_path = './data/split_data/train/'
blocks_test_save_path = './data/split_data/test/'
blocks_dev_save_path = './data/split_data/dev/'


class Pipeline:
    """Pipeline class

    Args:
        ratio ([type]): [description]
        root (str): Path to the folder containing the data
    """

    def __init__(self, ratio, language):
        self.language = language.lower()
        self.ratio = ratio
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def get_parsed_source(self, input_file: str,
                          output_file: str = None) -> pd.DataFrame:
        """Parse C code using pycparser

        If the user doesn't provide `output_file`, the method reads the a
        DataFrame containing the columns id, code (C code parsed by
        pycparser) and label. Otherwise it reads a Dataframe from `input_file`
        containing the columns id, code (input C code) and label, applies the
        c_parser to the code column and stores the resulting dataframe into
        `output_file`

        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file

        Returns:
            pd.DataFrame: DataFrame with the columns id, code (C code parsed by
                pycparser) and label.
        """
        input_file_path = os.path.join(data_path, input_file)
        if output_file is None:
            source = pd.read_csv(input_file_path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            source = pd.read_csv(input_file_path)
            source.columns = ['id', 'code', 'label']
            source['code'] = source['code'].progress_apply(parser.parse)

            source.to_csv(os.path.join(data_path, output_file))
        self.sources = source
        return source

    # split data for training, developing and testing
    def split_data(self):
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        
        check_or_create('./data/split_data')
        check_or_create(split_train_path)
        self.train_file_path = split_train_path+'train.csv'
        train.to_csv(self.train_file_path)

        check_or_create(split_dev_path)
        self.dev_file_path = split_train_path+'dev.csv'
        dev.to_csv(self.dev_file_path)

        check_or_create(split_test_path)
        self.test_file_path = split_test_path+'test.csv'
        test.to_csv(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_csv(input_file)

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        check_or_create('./data/embedding/')   
        check_or_create('./data/embedding/train/')   

        from prepare_data import get_sequences

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(embedding_save_path + '/programs_ast.csv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, vector_size=size, workers=16, sg=1, min_count=3)
        w2v.save(embedding_save_path + '/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self, data_path, part):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load('./data/split_data/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.read_csv(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_csv('./data/split_data/' + part + '/blocks.csv')
       

    # run for processing data to train
    def run(self):
        print('parse source code...')
        if os.path.exists(os.path.join(data_path, 'ast.csv')):
            self.get_parsed_source(input_file='ast.csv')
        else:
            self.get_parsed_source(input_file='programs.csv',
                                   output_file='ast.csv')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None, 128)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline('3:1:1', language='java')
ppl.run()
