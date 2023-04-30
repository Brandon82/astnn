# Dataset Parser
The data_parser.py is resposible for parsing the dataset.ttl file:
- Coverts dataset into a Python dictionary
- Iterates over each element in the dict and saves their data into mutant objects
- Using diff_parser.py, inserts each mutation into it's original program and returns the mutated program
- Converts the mutated program into the single method in which the mutant is located
- Saves each of the mutated methods into .java files 
	- Can also save the mutated methods into a single csv/pkl file

# Usage
- Alter the variables in config.py to change things like file paths
- Run data_parser.py

# Dataset used 
- https://b2share.eudat.eu/records/fd8e674385214fe9a327941525c31f53
- Saved as dataset.ttl

# Libraries
```
(python 3.10)
pip install rdflib==6.3.2
```