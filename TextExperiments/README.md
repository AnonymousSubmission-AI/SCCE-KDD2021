# Text Experiments

This folder contains the code to reproduce the experiments on textual data for the paper - Explainable Clustering via Simultaneous Construction of Clusters and Exemplars: Complexity and Provably Good Approximation Algorithms

## Data Format

IMPORTANT : we cannot distribute the Harry Potter Data, so we provide the first chapter of the book only, the results will be different than the paper (and it probably requires to tune epsilon again)

Input data must fit the following format:

1. "sentences.txt" contains the sentences, one by line
1. "embeddings.txt" contains the corresponding sentence embedding in csv format, the order correspond to the sentences order in "sentences.txt"
1. A folder with summaries in txt files

They must be stored in the Data folder

## Computing Rouge evaluation

Just run the Rouge.py script