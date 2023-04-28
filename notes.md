

Pipeline:

1. Parses source code (get_parsed_source())
- Opens input_file_path (ast.pkl or programs.pkl)
- Reads the columns ['id', 'code', 'label'], into a dataframe
- Applys c_parser on each element in 'code' column
- Saves that file with the updated c_parsed code to a .pkl (output_file)

2. Splits data (split_data())
- Shuffles and splits data into /train/train_.pkl, /dev/dev_.pkl, /test/test_.pkl

3. Constructs dict and train word embeddings (dictionary_and_embedding())
- creates a /train/embedding folder
- Saves ASTs as the corpus (Adds each AST into one str_corpus)
- saves AST code dict in to /train/programs_ns.tsv
- Load corpus to word2vec (train word2vec)
- Saves word2vec embedding to /data/train/embedding/node_w2v_x

4. Generate block sequences
- Load word2vec embeddings
- Load trees dict from data_path file
- Apply trans2seq to all 'code' columns
- Saves new dict as  /blocks.pkl
- This happens 3 times for train, dev, and test

