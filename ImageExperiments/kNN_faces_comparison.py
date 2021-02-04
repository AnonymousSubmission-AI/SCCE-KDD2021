from ExemplarClustering import ExemplarCluster as EC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from copy import deepcopy
from random import choices, seed
import numpy as np
import pickle, os

seed(0)
print ('----------')
embeddings_path = os.path.join('embeddings1.pickle')
data = pickle.loads(open(embeddings_path, "rb").read())
embeddings = data['embeddings']
labels = data['names']

sampling_size = 20
testing_size = 40 - sampling_size
classes = 3
k = classes
pcs = 16
epsilon = 0.78
algorithm= 'kMeans'

lookup = {}
name_index = {}
current_label = ''
new_embeddings = []
test_embeddings = []
test_labels = []
new_labels = []
i = 0
j = 0
k = 0
for label in labels:
	if current_label != label:
		name_index[label] = j
		new_embeddings += embeddings[i:i+sampling_size]
		new_labels += labels[i:i+sampling_size]
		test_embeddings += embeddings[i+sampling_size:i+sampling_size+testing_size]
		test_labels += labels[i+sampling_size:i+sampling_size+testing_size]
		current_label = label
		j += 1
		k += 1
		if k == classes:
			break
	i += 1

embeddings = np.asarray(new_embeddings)

labels = new_labels

pca = PCA(n_components=pcs)
pc = deepcopy(pca.fit_transform(embeddings))
pc_test = pca.fit_transform(test_embeddings)

clustering = EC(pc, epsilon, k, algorithm=algorithm, verbose=False)
clustering.fit()
clusters = clustering.get_clusters()
exemplars = clustering.get_exemplars()

exemplar_points = []
exemplar_labels = []
i = 1
for c in exemplars:
	for e in c:
		exemplar_points.append(pc[e])
		exemplar_labels.append(labels[(i-1)*sampling_size])
	i += 1

i = 1
j = 0
cluster_centers = []
cluster_labels = []
for c in clusters:
	cluster_centers.append([])
	for p in c:
		cluster_centers[j].append(pc[p])
	cluster_centers[j] = np.asarray(np.mean(cluster_centers[j], axis=0))
	cluster_labels.append(labels[(i-1)*sampling_size])
	i += 1
	j += 1

random_exemplars = []
random_labels = []

i = 0
for c in clusters:
	random_exemplars += choices(c, k=len(exemplars[i]))
	for j in range(len(exemplars[i])):
		random_labels.append(labels[i*sampling_size])
	i += 1
for j in range(len(random_exemplars)):
	random_exemplars[j] = pc[random_exemplars[j]]

big_c = []
big_labels = []
i = 0
for c in clusters:
	for p in c:
		big_c.append(pc[p])
		big_labels.append(labels[i*sampling_size])
	i += 1

knnE = KNeighborsClassifier(n_neighbors=1)
knnE.fit(exemplar_points, exemplar_labels)
knnC = KNeighborsClassifier(n_neighbors=1)
knnC.fit(cluster_centers, cluster_labels)
knnR = KNeighborsClassifier(n_neighbors=1)
knnR.fit(random_exemplars, random_labels)
knnB = KNeighborsClassifier(n_neighbors=1)
knnB.fit(big_c, big_labels)

print ('Exemplar: ' + str(knnE.score(pc_test, test_labels)))
print ('Centorid: ' + str(knnC.score(pc_test, test_labels)))
print ('Random: ' + str(knnR.score(pc_test, test_labels)))
print ('Whole Cluster: ' + str(knnB.score(pc_test, test_labels)))



