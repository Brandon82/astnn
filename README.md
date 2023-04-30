# ASTNN - An Equivalent Mutant Identifier based on Abstract Syntax Tree
An Abstract Syntax Tree Neural Network which aims to identify the equivalent mutants in a dataset of mutants.<br>
Please note that the /parser folder is not necessary and was used to do additional preprocessing on our dataset before inputting it into the pipeline
- Our original dataset consisted of only unified difference strings, so we create a unified-diff parser to insert the mutations into their original programs
	
### Requirements
```bash
  python 3.10
  pandas 2.0.0
  gensim 4.3.1
  scikit-learn 1.2.2
  torch 2.0.0
  pycparser 2.21
  javalang 0.13
```

Please note that original ASTNN implementation from which this was forked used python 3.6 and older libraries.<br>
I have updated the project to 3.10 and to use more current library versions.


### How to install
Install all the dependent packages via pip:

	$ pip install pandas==0.20.3 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0
 
Install pytorch according to your environment, see https://pytorch.org/ 


### Source Code Classification
1. `cd astnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train.py` for training and evaluation

### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.

### How to use it on your own dataset

Please refer to the `pkl` files in the corresponding directories of the two tasks. These files can be loaded by `pandas`.
For example, to realize clone detection, you need to replace the two files in /clone/data/java, bcb_pair_ids.pkl and bcb_funcs_all.tsv.
Specifically, the data format of bcb_pair_ids.pkl  is "id1, id2, label", where id1/2 correspond to the id in  bcb_funcs_all.tsv and label indicates whether they are clone or which clone type (i.e., 0 and 1-5 , 0 and 1 in a non-type case).
The data format of bcb_funcs_all.tsv is "id, function".
