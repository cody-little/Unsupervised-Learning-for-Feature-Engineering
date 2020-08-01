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

![](https://github.com/cody-little/Unsupervised-Learning-for-Feature-Engineering/blob/master/img/unsupervised%20cluster%20plot.PNG)

Using these two methods of manually checking how much data is in each cluster, the feature means, and a visualization of the clusters can be a good way to intuitively understand what the clusteing algorithim is doing behind the scenes. This also sides more on the data exploration side as well because we can gain an understanding of feature importance within the unsupervised task. 

##### Step 3: A More Objective Approach to n Cluster Selection the Silhouette score

My favorite objective measurement of clusters is the silhouette score. The silhouette score explained simply is a metric describing how similar observations are within their own cluster compared to other clusters. To implement the silhouette score analysis and find an optimal number of clusters it takes only a few lines of code.

```
from sklearn.metrics import silhouette_score as silly

sils = []


for i in range(2,11):
  learner = KMeans(n_clusters=i, random_state=500)
  clusters = learner.fit(wine_norm)
  wine_norm["cluster"] = clusters.labels_
  silouhette = silly(wine_norm, wine_norm["cluster"])
  sils.append(silouhette)

  print(f"K = {i}, silouette = {silouhette: .3f} ")
  
  K = 2, silouette =  0.517 
K = 3, silouette =  0.567 
K = 4, silouette =  0.612 
K = 5, silouette =  0.638 
K = 6, silouette =  0.641 
K = 7, silouette =  0.646 
K = 8, silouette =  0.661 
K = 9, silouette =  0.665 
K = 10, silouette =  0.669 
```

## Supervised Learning Task Experiment

Here is the fun stuff. I create an experiment for the six way supervised learning task. I want to see if there is a real difference between outcome metrics for the same task when one training set includes a cluster label as a feature. 

First and foremost I need to actually add the clusters. I went with 9 because I am trusting the silhouette score for this specific task. I felt as if the gain in score for a tenth cluster wasn't worth the extra dimension in the data. Below is the simple syntax to add the clusters as a feature to our original data set.

```
final_learner = KMeans(n_clusters=9)
final_learner_clusters = final_learner.fit(wine_norm)
wine_norm["cluster"] = final_learner_clusters.labels_
wine['cluster'] = wine_norm['cluster']
```

Next I use scikit learns implementation of a train test split so I have a randomly sampled train and test set with an approximate 80/20 split. Important to note in the syntax below is that I create *two* identical training sets with one specific differnece. "train set C" includes a cluster label within its data set. This ensures that the train sets are exaclty the same in every way except for that unsupervised label inclusion. It also ensures that they are tested on the same 20% test set. 

```
train_setC, test_set = train_test_split(wine, train_size = .80, test_size = .20,random_state = 500)
train_set = train_setC.drop(['cluster'],axis=1)
```

Now that I have two almost identical training sets for the experiment I have to pick a supervised learning algorithim to perform the task. I chose a simple single decision tree for the task. I don't often get to work with a single tree because many times I opt for some type of forest algorithim instead so it is always fun to get back to my roots. The decision tree (CART specifically) was the first classifer I learned for machine learning purposes. Below I use a 5-k fold cross validation on a grid search to find optimal hyperperameters. Below I have some sample syntax of what this looks like. I walk through what this process was below

##### Explanation

Here I split an X and a y or the cluster data set. I designate it as a valid with the 'V' and then I create a scorer that uses cohen kappa coeffecient. I really like using this metric for validation purposes because it helps to give a better understanding of how the model will generalize to new data. I use the param_grid to look across various max depths, the splitting criterion (purity or information gain), the maximum number of features used and the splitter method of decided how to split the leaves. I did this same syntax for the train set without the clusters and surpirsinly the optimal hyperperameters for both were the same. The output for the cluster included model is below showing what those optimal hyperperameters are along with their respective kappa score.

```
Vcluster_X = train_setC.drop(['quality'],axis=1)
Vcluster_y = train_setC['quality']

kap_scorer = make_scorer(cohen_kappa_score)
DT_Clusters_cv = GridSearchCV(DecisionTreeClassifier(random_state=500), param_grid={'criterion': ['gini','entropy'], 'max_depth': [100,500,1000,1500,2000,2500,3000],
                                                                    'splitter': ['best','random'], 'max_features':[None,'auto','log2']}, cv = 5, scoring = kap_scorer)
DT_Clusters_cv.fit(Vcluster_X,Vcluster_y)
print('Cluster Decision Tree')
print(DT_Clusters_cv.best_params_)
dt_clusterbest = DT_Clusters_cv.best_params_
print(f'best score is {DT_Clusters_cv.best_score_:.3f}')
print('_______________________________________________')


Cluster Decision Tree
{'criterion': 'gini', 'max_depth': 100, 'max_features': None, 'splitter': 'best'}
best score is 0.394
```

Next I fit the final models to these specifications, I include the full syntax just in case a reader is interested and doesn't feel like flipping to the notebook. I include the metric scores for various ways of evaluating the task as well. This is performed on the held out test set in order to maintain a robustness of results.

```
### Make Final Models and Test Them#
#split the test set#
testX = test_set.drop(['quality'],axis=1)
testX_nocluster=test_set.drop(['quality','cluster'],axis=1)
testy = test_set['quality']


#Fit both final models#
FinalCluster_DT = DecisionTreeClassifier(criterion='gini',max_depth=100,max_features=None,splitter='best',random_state=500).fit(Vcluster_X,Vcluster_y)
Final_DT = DecisionTreeClassifier(criterion='gini',max_depth=100, max_features=None, splitter = 'best',random_state=500).fit(V_X,V_y)

Cluster_predict = FinalCluster_DT.predict(testX)
DT_predict = Final_DT.predict(testX_nocluster)

#Get accuracy scores#
cluster_acc = accuracy_score(testy,Cluster_predict)
DT_acc = accuracy_score(testy,DT_predict)

cluster_kap = cohen_kappa_score(testy,Cluster_predict,weights='quadratic')
DT_kap = cohen_kappa_score(testy, DT_predict)
cluster_conf = confusion_matrix(testy,Cluster_predict)
DT_conf = confusion_matrix(testy,DT_predict)
```


## Results

Below is the output from a little results cell I made.


