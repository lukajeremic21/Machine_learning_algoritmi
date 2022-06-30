from cv2 import kmeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
x_y = df[['Annual Income (k$)','Spending Score (1-100)']]
# X = df['Annual Income (k$)']
# Y = df['Spending Score (1-100)']


kmeans = KMeans(n_clusters=5, random_state=0).fit(x_y)
x_y.plot("Annual Income (k$)","Spending Score (1-100)",kind='scatter',cmap='seismic',c=kmeans.labels_)
plt.show()

