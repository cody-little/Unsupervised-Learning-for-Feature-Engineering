# Unsupervised-Learning-for-Feature-Engineering
A machine learning project comparing the inclusion of an unsupervised learning label as a feature for the same task in a comparative analysis of results and methods.

## Summary

This machine learning project intends to discuss the usage of an unsupervised learning prediction as a feature in a supervised learning task. To do this I take a data set with various wine charachteristics and a target label of a quality rating and apply a k-means clustering algorithim on the the feature space minus the quality rating. I then use this unsupervised prediction cluster label as a feature in a supervised learning task with six outcome labels. Below is a short bulleted breakdown of what the project entails.

#### Task 1: Predict a six way classification label for wine quality based off available features
#### Task 2: Create an experiment where I compare unsupervised feature label inclusion in task 1

* Get a cluster label on data from a k-means algorithim
* Discuss methods of evaluating the number of clusters
* Perform a supervised learning task in an experimental fashion showing the effect of an unsupervised feature label
  - Perform the same task, on the same test set, with the same features except for one difference
  - The difference is one application of the task includes the unsupervised cluster label as a feature
  - I perform the supervised learning task with a grid search and 5-k fold cross validated decison tree
  - Discuss results and efficacy on this multi-label classification task
  
  
## Sections

* Exploratory Analysis
* Unsupervised Learning and Evaluation of Clusters
* Supervised Learning Task Experiment
* Results

## Exploratory Analysis

The first step of this exploratory data analysis (EDA) is to just look at the data and see what I am working with. I am not a chemist or a sommelier so I know nothing about wine but all of the columns in this data set are numeric which means I can understand their distributions. 

![](https://github.com/cody-little/Unsupervised-Learning-for-Feature-Engineering/blob/master/img/unsupervised%20learning%20label.PNG)

This table gives me a good indication that later on I will have to normalize my numeric features to perform the k-means algorithim. Looking at the min/max and the standard deviation I can clearly see that some variables used as features will be pretty influential. Another thing I see immmediatly is that this isn't a huge data set. With just under 1,600 observations I wil not be able to use a more data hungry algorithim later on.

The next step I take is to look at the correlations among the features and the target labels.

![](https://github.com/cody-little/Unsupervised-Learning-for-Feature-Engineering/blob/master/img/unsupervised%20cor%20plot.PNG)


By looking at correlations between the features and target lable I can start to picture what quality wines have in common from a conceptual level. When I evaluate clusters through a human judgement approach later this will come in handy. Since I can see that some combinations of features influence the quality of wine I can plot them visually with scatterplots to dig a little deeper in understanding their relationship. 

![](https://github.com/cody-little/Unsupervised-Learning-for-Feature-Engineering/blob/master/img/unsupervised%20scatter%20plot.PNG)

Once I spent some more time exploring combinations visually and different counts across quartiles I wanted to get a better understanding of my target labels. I do a quick value counts on the data set and see something really important

```
print(wine['quality'].value_counts())
5    681
6    638
7    199
4     53
8     18
3     10
Name: quality, dtype: int64
```

This target label is highly unbalanced. In normal circumstances I would fix this in a few different ways:
- The first would be that I wouldn't have wanted a six-way classification in the first place, I would have tried to simplify the outcomes in a binary or perhaps three label classification for wine quality (bad,okay,good)
- The second could be creating synthetic data to even out the proportions if you really wanted to use a six-way classification (though this has its own drawbacks and disadvantages)

For the purposes of this experiment I thought it would be interesting to let this go. I want to test the efficacy of using an unsupervised cluster label as a feature in a supervised learning task and so far this task seems like it will be difficult due to a few disadvantages. The first is the small data set size, and the second is the unbalanced ratios of target labels. A third is that it will be difficult to learn the minute differences between very close labels (the 5 and 6 for example).

##### Outcomes of EDA

My first impressions from this exploration told me three things moving forward:

- I will need to normalize the data
- I don't have a huge amount of data to train on
- This is a highly unbalanced target label on a six way classification task so keep outcome goals reasonable

## Unsupervised Learning and Evaluation of Clusters

##### Step 1: Normalize
This section walks through the steps I took to normalize the data, apply the k-means algorithim, and evaluate the number of clusters I wanted to move forward with.
The first step was normalizing the data so that no one feauture would draw too much attention to itself. In order to do this I just used sklearns built min normalizer to normalize the data set. First I removed the target label because I didn't want the target mean slipping knowledge into the clusters. 

```
scaler = MinMaxScaler()

wine_norm = wine.drop(['quality'], axis = 1)
columnnames = wine_norm.columns
wine_norm = pd.DataFrame(scaler.fit_transform(wine_norm))
wine_norm.columns = columnnames
```
##### Step 2: Evaluate Cluster Count with Human Judgement

An important aspect of k-means clustering is determing the number of clusters to use. One method that you can employ is using your own judgement. Similar to a PCA we can use the clusters to understand the variation in means for each of our features. I create a function to print this information for all features and that is located in the notebook section of this repository. An example below is for a peak at what this syntax looks like and the output you get.

```
def describe_cluster(cluster_instances): 
  fixed_acidity = cluster_instances["fixed acidity"].mean()
  volatile_acidity = cluster_instances['volatile acidity'].mean()
   print(f'Fixed acidity mean {fixed_acidity:.2f}')
  print(f'Volatile Acidity Mean {volatile_acidity:.2f}')
  
for k in range(2,5):
  print(f"Cluster Statistics for k = {k}")

  learner = KMeans(n_clusters=k)
  clusters = learner.fit(wine_norm)



  wine_norm["cluster"] = clusters.labels_
  for cluster_id in wine_norm["cluster"].unique():
        print(f"Cluster {cluster_id}")
        cluster_instances = wine_norm.loc[wine_norm.cluster == cluster_id]
        print(f"  {100*len(cluster_instances)/len(wine_norm):.1f}% of data.")
        describe_cluster(cluster_instances)

  print(f"-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
  
  Output:
  Cluster Statistics for k = 2
Cluster 1
  58.7% of data.
Fixed acidity mean 0.24
Volatile Acidity Mean 0.33
Cluster 0
  41.3% of data.
Fixed acidity mean 0.45
Volatile Acidity Mean 0.20
```
As you can see here or in the notebook this prints out each cluster for any range of k clusters that you may want. It also allows you to see how the means for each feature change across different clusters. Looking at the full output for all k clusters and features you can visually see how these metrics change. 

Another method that is commmon and kind of fun is to plot on a two dimensional space the cluster outcomes. Below is a picture of a scatterplot where the hue is the cluster labels for k = 4. 


