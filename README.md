# ASTNN - An Equivalent Mutant Identifier
An **Abstract Syntax Tree Neural Network** aiming to identify equivalent mutants within a dataset of mutants. 
- Please note that the `/parser` folder isn't necessary and was used for additional preprocessing on our dataset before integrating it into the pipeline.
- Our initial dataset consisted solely of unified difference strings, so we created a unified-diff parser to insert the mutations into their original programs.
	
## Requirements
```bash
  python 3.10
  pandas 2.0.0
  gensim 4.3.1
  scikit-learn 1.2.2
  torch 2.0.0
  pycparser 2.21
  javalang 0.13
```
> Please note that the original ASTNN implementation, from which this was forked, utilized Python 3.6 and older library versions. We have updated the project to Python 3.10 and more recent library versions.

## Added Features
Here's a list of changes compared to the original repository:

- Implemented a [mutant dataset](https://b2share.eudat.eu/records/fd8e674385214fe9a327941525c31f53) from Mutantbench
  - Created a parser that inserts each mutation into its original program and returns the mutated method/file.
- Updated to Python 3.10
- Updated to more recent library versions
- Added `open_data.py` to convert `.pkl` files to `.csv` for easier viewing
- Updated the model to support equivalent mutant identification
- Saved the trained model to allow for inference
- Added `test_from_trained.py` to get predictions on additional data from the trained model
- General refactoring of `pipeline.py` and `train.py` to improve code quality