#coding : utf-8
from ExemplarClustering import ExemplarCluster

import random
import string
import glob

from sklearn.preprocessing import normalize
from rouge import Rouge
import sys
sys.setrecursionlimit(10000)

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

#Authors: Antoine Gourru And Michael Livanos, May 2020

def build_summary(sentences,embeddings,epsilon,cluster,limit,bounded = False,verbose = False):
    ec = ExemplarCluster(embeddings,epsilon,cluster, algorithm='kMeans',limit=limit, verbose=False)
    ec.fit(bounded)
    exemplars = ec.get_exemplars()
    clusters = ec.get_clusters()

    if verbose:
        print("%d Clusters" % len(exemplars))
        print("Repartition by clusters and # of examplars:")    
        print([len(sublist) for sublist in clusters])
        print([len(i) for i in exemplars])
    
    summary = [sentences[item] for sublist in exemplars for item in sublist]

    n_sum = len(summary)
    summary = ". ".join(summary)
    
    if verbose:
        print("Produced %d exemplars" % n_sum)
        print(summary)
    c = 0
    y = np.zeros((embeddings.shape[0],))
    for i in ec.clusters:
        for j in i:
            y[j] = int(c)        
        c += 1
        
    return summary,n_sum,list(y)


# Import HP
sentences = [line.rstrip('\n') for line in open(r'Data/sentences.txt', 'r',encoding="utf-8")]
embedding = np.genfromtxt('Data/embeddings.txt', delimiter=',')

n = len(sentences)
d = embedding.shape[1]

print("%d documents in the dataset" % n)
print("Embedding size: %d" % d)

print("IMPORTANT !!!")
print("If you are using the demo data, it is only the first chapter of the books, you should use the full book to reproduce our results")

# We removed the punctuation
punk = string.punctuation+'”“'
sentences = [i.translate(str.maketrans('', '', punk)) for i in sentences]

methods = ["SCCE","SCCRB","PageRank","Random"]

#Path is a folder containing the summaries in text format
path = 'Data/Summaries'
files = [f for f in glob.glob(path + "**/*.txt", recursive=True)]

n_methods = len(methods)
n_files = len(files)


scores_tab = np.zeros((n_files,n_methods*2))

#Parameters
n_run = 20
fi = 0
#Epsilon depends on the embedding scheme, so it should be fine tuned
epsilon = 9

print("Building the sim matrix")
bn = normalize(embedding)
sim_mat = bn @ bn.T
np.fill_diagonal(sim_mat, 0)

print("Creating the network with networkx")
nx_graph = nx.from_numpy_array(sim_mat)

print("Computing Page Rank values and sorting the sentences by pr value")
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

print("Computing Rouge Score for each summaries")
for f in files:    
    summary_GT = open(f, 'r',encoding = "utf-8").read()
    k = len(summary_GT.split("."))
    summary_GT = summary_GT.translate(str.maketrans('', '', punk))
    
    print("Summary %d, length: %d" % (fi+1,k),end ="")
    
    print(".",end ="")
    summary_embedding,n_sum_embedding,y_embedding = build_summary(sentences,embedding,epsilon,6,limit = k,bounded = False)  
    scores = Rouge().get_scores(summary_embedding, summary_GT)[0]
    scores_tab[fi,0] = scores['rouge-1']['f']*100
    scores_tab[fi,(n_methods)] = scores['rouge-2']['f']*100

    print(".",end ="")
    summary_embedding,n_sum_embedding,y_embedding = build_summary(sentences,embedding,epsilon,6,limit = k,bounded = True)
    scores = Rouge().get_scores(summary_embedding, summary_GT)[0]
    scores_tab[fi,1] = scores['rouge-1']['f']*100
    scores_tab[fi,(1 + n_methods)] = scores['rouge-2']['f']*100
    
    print(".",end ="")
    
    summary_auto = ". ".join([ranked_sentences[i][1] for i in range(k)])
    scores = Rouge().get_scores(summary_auto, summary_GT)[0]
    scores_tab[fi,2] = scores['rouge-1']['f']*100
    scores_tab[fi,(2 + n_methods)] = scores['rouge-2']['f']*100

    print(".",end ="")
    r1 = 0
    r2 = 0
    for test in range(n_run):
        negative_examples = random.choices(sentences, k=k)
        negative_examples = ". ".join(negative_examples)
        scores = Rouge().get_scores(negative_examples, summary_GT)[0]
        r1 += scores['rouge-1']['f']*100
        r2 += scores['rouge-2']['f']*100
        
    scores_tab[fi,3] = r1/n_run
    scores_tab[fi,(3 + n_methods)] = r2/n_run

    
    print(".")
    
    
    fi += 1

print(scores_tab)
row_names = ["Summary " + str(i) for i in range(n_files)]
col_names = methods + methods

#Save the results
df = pd.DataFrame(scores_tab, index=row_names, columns=col_names)
df.to_csv('rouge.csv', index=True, header=True, sep='\t', float_format='%.2f')

input("Press Enter to quit")
