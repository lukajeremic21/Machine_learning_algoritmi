from cv2 import kmeans
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

dataset = pd.read_csv('Iris.csv')
dataset1 = dataset[["SepalLengthCm", "SepalWidthCm"]]

KMedoids = KMedoids(n_clusters=5).fit(dataset1)
# print(kmeans.predict(adultData1))
# print(kmeans.cluster_centers_)
dataset1.plot("SepalLengthCm","SepalWidthCm",kind='scatter',cmap='viridis',c=KMedoids.labels_)
plt.show()