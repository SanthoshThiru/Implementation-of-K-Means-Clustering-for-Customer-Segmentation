# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step-1:
Import the necessary packages using import statement.
### Step-2:
Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
### Step-3:
Import KMeans and use for loop to cluster the data.
### Step-4:
Predict the cluster and plot data graphs.
### Step-5:
Print the outputs and end the program.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Santhosh T
RegisterNumber: 212223220100 
*/

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

#Load data from CSV
data = pd.read_csv("Mall_Customers.csv")
data

#Extract features
X = data[['Annual Income (k$)','Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
#Number of clusters
k = 5
#Initialize KMeans
kmeans = KMeans(n_clusters=k)
#Fit the data
kmeans.fit(X)
centroids = kmeans.cluster_centers_
#Get the cluster labels for each data point
labels = kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r','g','b','c','m'] #Define colors for each cluster
for i in range(k):
  cluster_points=X[labels==i] #Get data points belonging to cluster i
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],
              color=colors[i],label=f'Cluster(i+1)')
  #Find minimum enclosing circle
distances=euclidean_distances(cluster_points,[centroids[i]])
radius=np.max(distances)
circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
plt.gca().add_patch(circle)

#Plotting the centroids
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.legend()
plt.grid(True)
plt.axis('equal') #Ensure aspect ratio is equal
plt.show()

```

## Output:
### data.head
![Screenshot 2024-04-29 075205](https://github.com/SanthoshThiru/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/148958618/b6fe03ac-e362-4d2e-aae8-843ee110695a)
### data.info
![Screenshot 2024-04-29 074702](https://github.com/SanthoshThiru/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/148958618/fc1c6246-4fa3-4098-b22a-23b3d35b4464)
### data.isnull().sum
![Screenshot 2024-04-29 075215](https://github.com/SanthoshThiru/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/148958618/a3a9fe26-489b-4028-8628-fb0b7d455c71)

![Screenshot 2024-04-29 074640](https://github.com/SanthoshThiru/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/148958618/f0d14363-4ca8-4ff4-a505-ac7ef958d15a)

![Screenshot 2024-04-29 074620](https://github.com/SanthoshThiru/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/148958618/a2f81c9c-183a-40ee-a5ff-8ec1aaa162a9)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
