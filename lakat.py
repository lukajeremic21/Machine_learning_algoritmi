import pandas as pd
from sklearn.cluster import KMeans 
from yellowbrick.cluster import KElbowVisualizer
df = pd.read_csv('Mall_Customers.csv')
x_y = df[['Annual Income (k$)','Spending Score (1-100)']]
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(1,30))
visualizer.fit(x_y)
visualizer.show()